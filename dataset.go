package llm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"sync"
	"sync/atomic"

	"github.com/grafana/sobek"
	"go.k6.io/k6/v2/js/common"
	"go.k6.io/k6/v2/lib/fsext"
)

// Dataset is a deterministic, replayable corpus of chat requests. Loaded once
// per process from a JSONL file and shared across VUs via dsCache; per-Dataset
// instances each carry their own cursor and shuffle permutation so two VUs
// reading from the same file do not see the same order unless they share a
// (path, seed) pair.
type Dataset struct {
	path  string
	items []datasetItem
	order []int
	// cursor is shared by every Dataset built from the same (path, seed,
	// shuffle) in the process. Each VU constructs its own Dataset, and a
	// cursor per instance made every VU replay the same prompts in the same
	// order, which a server with prefix caching answers from cache.
	cursor *atomic.Uint64
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
// dataset twice from the same path is free. dsCursors holds the shared
// cursors, keyed by path, seed and shuffle.
var (
	dsCache   sync.Map
	dsCursors sync.Map
)

// loadDatasetItems reads the corpus through k6's filesystem, never the host's
// directly: that is what puts the file in a `k6 archive` or `k6 cloud` bundle
// and keeps a script on a shared runner inside the paths k6 allows it.
func loadDatasetItems(fs fsext.Fs, abs string) ([]datasetItem, error) {
	if v, ok := dsCache.Load(abs); ok {
		if cached, ok := v.(*[]datasetItem); ok {
			return *cached, nil
		}
	}
	data, err := fsext.ReadFile(fs, abs)
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}

	out := make([]datasetItem, 0, 256)
	for i, line := range bytes.Split(data, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var probe struct {
			Messages []json.RawMessage `json:"messages"`
		}
		if err := json.Unmarshal(line, &probe); err != nil {
			return nil, fmt.Errorf("line %d: invalid json: %w", i+1, err)
		}
		if len(probe.Messages) == 0 {
			return nil, fmt.Errorf("line %d: missing or empty 'messages'", i+1)
		}
		out = append(out, datasetItem{raw: line})
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
	// Files load in the init context only, like k6's own open(): that is when
	// k6 records what a script reads so it can bundle it.
	env := m.vu.InitEnv()
	if env == nil {
		common.Throw(rt, errors.New("llm.Dataset: construct it in the init context, outside the default function"))
	}
	// k6 always sets both. modulestest sets neither, so a unit test falls
	// back to the host filesystem, which is all it has.
	abs, fs := opts.Path, env.FileSystems["file"]
	if env.CWD != nil {
		abs = env.GetAbsFilePath(opts.Path)
	}
	if fs == nil {
		fs = fsext.NewOsFs()
	}
	items, err := loadDatasetItems(fs, abs)
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

	key := fmt.Sprintf("%s|%d|%t", abs, opts.Seed, opts.Shuffle)
	cursor, _ := dsCursors.LoadOrStore(key, new(atomic.Uint64))
	//nolint:forcetypeassert // only *atomic.Uint64 is ever stored
	ds := &Dataset{path: opts.Path, items: items, order: order, cursor: cursor.(*atomic.Uint64)}
	return rt.ToValue(ds).ToObject(rt)
}

// Size returns the number of items in the dataset.
func (d *Dataset) Size() int { return len(d.items) }

// Next advances the shared cursor and returns the next request, wrapping at
// the end. VUs in one process draw from one sequence, so no two of them send
// the same prompt until the corpus wraps. Across k6 instances use At() with an
// index derived from the execution context.
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

// Reset rewinds the shared cursor so the next call to Next() returns the first
// item again, for every VU drawing from it. Has no effect on the shuffle
// permutation or the cached items themselves.
func (d *Dataset) Reset() { d.cursor.Store(0) }
