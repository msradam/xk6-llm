package llm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"

	"github.com/grafana/sobek"
	"go.k6.io/k6/v2/js/common"
)

// Dataset is a deterministic, replayable corpus of chat requests. Loaded once
// per process from a JSONL file and shared across VUs via dsCache; per-Dataset
// instances each carry their own cursor and shuffle permutation so two VUs
// reading from the same file do not see the same order unless they share a
// (path, seed) pair.
type Dataset struct {
	path   string
	items  []datasetItem
	order  []int
	cursor atomic.Uint64
}

// datasetItem stores the raw JSONL bytes; toJS re-decodes on each call so JS
// mutations to the returned object never leak across VUs.
type datasetItem struct {
	raw []byte
}

func (it *datasetItem) toJS() map[string]any {
	var m map[string]any
	_ = json.Unmarshal(it.raw, &m) // validated at load time; cannot fail here
	return m
}

// dsCache holds shared *[]datasetItem keyed by absolute file path. Loading a
// dataset twice from the same path is free.
var dsCache sync.Map

func loadDatasetItems(path string) ([]datasetItem, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolve path: %w", err)
	}
	if v, ok := dsCache.Load(abs); ok {
		if cached, ok := v.(*[]datasetItem); ok {
			return *cached, nil
		}
	}
	f, err := os.Open(abs) // #nosec G304 -- user-supplied dataset path is the feature
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}
	defer func() { _ = f.Close() }()

	out := make([]datasetItem, 0, 256)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var probe struct {
			Messages []json.RawMessage `json:"messages"`
		}
		if err := json.Unmarshal(line, &probe); err != nil {
			return nil, fmt.Errorf("line %d: invalid json: %w", lineNo, err)
		}
		if len(probe.Messages) == 0 {
			return nil, fmt.Errorf("line %d: missing or empty 'messages'", lineNo)
		}
		buf := make([]byte, len(line))
		copy(buf, line)
		out = append(out, datasetItem{raw: buf})
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read: %w", err)
	}
	if len(out) == 0 {
		return nil, errors.New("dataset is empty")
	}
	dsCache.Store(abs, &out)
	return out, nil
}

type datasetOptions struct {
	Path    string
	Seed    uint64
	Shuffle bool
}

func parseDatasetOptions(raw any) (*datasetOptions, error) {
	o := &datasetOptions{Seed: 42}
	if raw == nil {
		return nil, errors.New("llm.Dataset: options object with 'path' is required")
	}
	m, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("llm.Dataset: options must be an object, got %T", raw)
	}
	path, ok := m["path"].(string)
	if !ok || path == "" {
		return nil, errors.New("llm.Dataset: 'path' (string) is required")
	}
	o.Path = path
	if v, ok := m["shuffle"].(bool); ok {
		o.Shuffle = v
	}
	if v, ok := m["seed"]; ok && v != nil {
		f, ok := asFloat(v)
		if !ok {
			return nil, fmt.Errorf("llm.Dataset: 'seed' must be a number, got %T", v)
		}
		if f < 0 {
			return nil, fmt.Errorf("llm.Dataset: 'seed' must be non-negative, got %v", f)
		}
		o.Seed = uint64(f)
	}
	return o, nil
}

func (m *module) newDataset(call sobek.ConstructorCall) *sobek.Object {
	rt := m.vu.Runtime()
	var raw any
	if len(call.Arguments) > 0 {
		raw = call.Arguments[0].Export()
	}
	opts, err := parseDatasetOptions(raw)
	if err != nil {
		common.Throw(rt, err)
	}
	items, err := loadDatasetItems(opts.Path)
	if err != nil {
		common.Throw(rt, fmt.Errorf("llm.Dataset(%q): %w", opts.Path, err))
	}

	order := make([]int, len(items))
	for i := range order {
		order[i] = i
	}
	if opts.Shuffle {
		// #nosec G404 -- deterministic, seeded shuffle for reproducible workloads; not crypto
		rng := rand.New(rand.NewPCG(opts.Seed, opts.Seed^0x9E3779B97F4A7C15))
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	}

	ds := &Dataset{path: opts.Path, items: items, order: order}
	return rt.ToValue(ds).ToObject(rt)
}

// Size returns the number of items in the dataset.
func (d *Dataset) Size() int { return len(d.items) }

// Next advances the internal cursor and returns the next request, wrapping at
// the end. Concurrency-safe across VUs (within a single process) when the same
// Dataset instance is reused; in k6 each VU constructs its own instance, so
// "wrap" semantics apply per VU.
func (d *Dataset) Next() map[string]any {
	if len(d.items) == 0 {
		return nil
	}
	sz := uint64(len(d.items))
	n := d.cursor.Add(1) - 1
	// #nosec G115 -- n%sz < sz <= len(d.items), so always fits in int
	idx := int(n % sz)
	return d.items[d.order[idx]].toJS()
}

// At returns the i-th request (modulo dataset size, negative-safe) without
// advancing the cursor. Use this when the caller wants to derive the index
// from __VU and __ITER for fully reproducible workloads.
func (d *Dataset) At(i int64) map[string]any {
	if len(d.items) == 0 {
		return nil
	}
	n := len(d.items)
	idx := int(i % int64(n))
	if idx < 0 {
		idx += n
	}
	return d.items[d.order[idx]].toJS()
}

// Reset rewinds the internal cursor. Useful for repeated runs in tests.
func (d *Dataset) Reset() { d.cursor.Store(0) }
