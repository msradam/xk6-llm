package llm

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func newTestRng(seed uint64) *rand.Rand {
	// Test-only deterministic RNG; not crypto.
	return rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15)) //nolint:gosec
}

// content extracts the first user message text from a dataset request map,
// failing the test on any shape mismatch.
func content(t *testing.T, m map[string]any) string {
	t.Helper()
	msgs, ok := m["messages"].([]any)
	require.True(t, ok, "messages must be []any")
	require.NotEmpty(t, msgs)
	first, ok := msgs[0].(map[string]any)
	require.True(t, ok, "messages[0] must be object")
	s, ok := first["content"].(string)
	require.True(t, ok, "messages[0].content must be string")
	return s
}

func writeJSONL(t *testing.T, lines []string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "ds.jsonl")
	require.NoError(t, os.WriteFile(p, []byte(strings.Join(lines, "\n")+"\n"), 0o600))
	return p
}

func TestParseDatasetOptions(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		raw     any
		wantErr string
	}{
		{"nil", nil, "options object with 'path' is required"},
		{"not-object", "x", "options must be an object"},
		{"missing-path", map[string]any{}, "'path' (string) is required"},
		{"empty-path", map[string]any{"path": ""}, "'path' (string) is required"},
		{"bad-seed", map[string]any{"path": "x", "seed": "abc"}, "'seed' must be a number"},
		{"ok", map[string]any{"path": "x", "seed": int64(7), "shuffle": true}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := parseDatasetOptions(tc.raw)
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			require.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestLoadDatasetItemsValidation(t *testing.T) {
	t.Parallel()
	t.Run("empty file", func(t *testing.T) {
		t.Parallel()
		p := writeJSONL(t, []string{""})
		_, err := loadDatasetItems(p)
		require.ErrorContains(t, err, "dataset is empty")
	})
	t.Run("invalid json", func(t *testing.T) {
		t.Parallel()
		p := writeJSONL(t, []string{`{not json`})
		_, err := loadDatasetItems(p)
		require.ErrorContains(t, err, "line 1: invalid json")
	})
	t.Run("missing messages", func(t *testing.T) {
		t.Parallel()
		p := writeJSONL(t, []string{`{"max_tokens": 10}`})
		_, err := loadDatasetItems(p)
		require.ErrorContains(t, err, "missing or empty 'messages'")
	})
	t.Run("ignores blank lines", func(t *testing.T) {
		t.Parallel()
		p := writeJSONL(t, []string{
			`{"messages":[{"role":"user","content":"a"}]}`,
			"",
			`{"messages":[{"role":"user","content":"b"}]}`,
		})
		items, err := loadDatasetItems(p)
		require.NoError(t, err)
		require.Len(t, items, 2)
	})
	t.Run("nonexistent path", func(t *testing.T) {
		t.Parallel()
		_, err := loadDatasetItems(filepath.Join(t.TempDir(), "missing.jsonl"))
		require.ErrorContains(t, err, "open:")
	})
}

func TestLoadDatasetCacheSharing(t *testing.T) {
	t.Parallel()
	p := writeJSONL(t, []string{
		`{"messages":[{"role":"user","content":"x"}]}`,
	})
	a, err := loadDatasetItems(p)
	require.NoError(t, err)
	b, err := loadDatasetItems(p)
	require.NoError(t, err)
	require.Same(t, &a[0], &b[0], "cache must return the same backing slice element")
}

func TestDatasetItemToJSDeepCopy(t *testing.T) {
	t.Parallel()
	p := writeJSONL(t, []string{
		`{"messages":[{"role":"user","content":"hello"}],"max_tokens":42}`,
	})
	items, err := loadDatasetItems(p)
	require.NoError(t, err)
	got1 := items[0].toJS()
	got2 := items[0].toJS()
	require.Equal(t, "hello", content(t, got1))
	require.InDelta(t, 42, got1["max_tokens"], 0)

	// Mutate one copy; the other must be unaffected.
	msgs1, ok := got1["messages"].([]any)
	require.True(t, ok)
	first1, ok := msgs1[0].(map[string]any)
	require.True(t, ok)
	first1["content"] = "MUTATED"
	require.Equal(t, "hello", content(t, got2))
}

func makeDataset(t *testing.T, n int, shuffle bool, seed uint64) *Dataset {
	t.Helper()
	lines := make([]string, n)
	for i := range lines {
		lines[i] = fmt.Sprintf(`{"messages":[{"role":"user","content":"item-%d"}]}`, i)
	}
	p := writeJSONL(t, lines)
	items, err := loadDatasetItems(p)
	require.NoError(t, err)
	order := make([]int, len(items))
	for i := range order {
		order[i] = i
	}
	if shuffle {
		rng := newTestRng(seed)
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	}
	return &Dataset{path: p, items: items, order: order}
}

func TestDatasetNextWraps(t *testing.T) {
	t.Parallel()
	d := makeDataset(t, 3, false, 0)
	require.Equal(t, 3, d.Size())
	seen := make([]string, 0, 6)
	for range 6 {
		seen = append(seen, content(t, d.Next()))
	}
	require.Equal(t, []string{"item-0", "item-1", "item-2", "item-0", "item-1", "item-2"}, seen)
}

func TestDatasetAtNegativeAndModulo(t *testing.T) {
	t.Parallel()
	d := makeDataset(t, 3, false, 0)
	cases := map[int64]string{0: "item-0", 1: "item-1", 2: "item-2", 3: "item-0", 7: "item-1", -1: "item-2"}
	for i, want := range cases {
		require.Equal(t, want, content(t, d.At(i)), "At(%d)", i)
	}
}

func TestDatasetShuffleDeterministic(t *testing.T) {
	t.Parallel()
	d1 := makeDataset(t, 8, true, 42)
	d2 := makeDataset(t, 8, true, 42)
	d3 := makeDataset(t, 8, true, 7)
	require.Equal(t, d1.order, d2.order, "same seed => same permutation")
	require.NotEqual(t, d1.order, d3.order, "different seed => different permutation")
	// Permutation must be a valid permutation of [0,n).
	seen := make(map[int]bool, 8)
	for _, v := range d1.order {
		seen[v] = true
	}
	require.Len(t, seen, 8)
}

func TestDatasetNextRaceSafe(t *testing.T) {
	t.Parallel()
	d := makeDataset(t, 100, false, 0)
	const n = 1000
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range n / 8 {
				require.NotNil(t, d.Next())
			}
		})
	}
	wg.Wait()
	// Cursor should have advanced exactly n*(8/8)=n times rounded by integer div.
	require.GreaterOrEqual(t, d.cursor.Load(), uint64(n-7))
}

func TestDatasetReset(t *testing.T) {
	t.Parallel()
	d := makeDataset(t, 3, false, 0)
	_ = d.Next()
	_ = d.Next()
	d.Reset()
	require.Equal(t, "item-0", content(t, d.Next()))
}
