package llm

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// sseServer returns an httptest server that emits the supplied chunks as SSE,
// with `gap` between consecutive writes (so ITL is measurable).
func sseServer(t *testing.T, gap time.Duration, chunks []string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		require.True(t, ok)
		for i, c := range chunks {
			if i > 0 && gap > 0 {
				time.Sleep(gap)
			}
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func contentChunk(s string) string {
	return fmt.Sprintf(`{"choices":[{"index":0,"delta":{"content":%q},"finish_reason":null}]}`, s)
}

const roleChunk = `{"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`

func usageChunk(prompt, completion int) string {
	return fmt.Sprintf(`{"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
		prompt, completion, prompt+completion)
}

func TestParseStream_SkipsRoleOnlyForTTFT(t *testing.T) {
	t.Parallel()
	srv := sseServer(t, 10*time.Millisecond, []string{
		roleChunk,
		contentChunk("Hello"),
		contentChunk(" world"),
		contentChunk("!"),
		usageChunk(7, 3),
		"[DONE]",
	})
	resp, err := http.Get(srv.URL)
	require.NoError(t, err)
	defer resp.Body.Close()

	start := time.Now()
	res, err := parseStream(resp.Body, start)
	require.NoError(t, err)

	require.Equal(t, "Hello world!", res.Content)
	require.Equal(t, 7, res.PromptTokens)
	require.Equal(t, 3, res.CompletionTokens)

	// 3 content chunks → 2 ITL samples (per vLLM: first ITL is chunk[2]-chunk[1]).
	require.Len(t, res.ITL, 2, "ITL count = content_chunks - 1")

	// TTFT measured at FIRST content chunk, not at role-only chunk.
	// Role chunk is at t≈0; first content chunk at t≈10ms.
	require.GreaterOrEqual(t, res.TTFT, 8*time.Millisecond)
	// Each ITL should be ~10ms (the inter-chunk gap), NOT include the start→first gap.
	for _, itl := range res.ITL {
		require.GreaterOrEqual(t, itl, 5*time.Millisecond)
		require.Less(t, itl, 50*time.Millisecond, "ITL must not include start→first-content gap")
	}
}

func TestParseStream_UsageOmitted(t *testing.T) {
	t.Parallel()
	srv := sseServer(t, 0, []string{
		roleChunk,
		contentChunk("a"),
		contentChunk("b"),
		"[DONE]",
	})
	resp, err := http.Get(srv.URL)
	require.NoError(t, err)
	defer resp.Body.Close()

	res, err := parseStream(resp.Body, time.Now())
	require.NoError(t, err)
	require.Equal(t, "ab", res.Content)
	require.Equal(t, 0, res.PromptTokens)
	require.Equal(t, 0, res.CompletionTokens)
}

func TestParseStream_MidStreamError(t *testing.T) {
	t.Parallel()
	srv := sseServer(t, 0, []string{
		roleChunk,
		contentChunk("partial"),
		`{"error":{"message":"upstream exploded","type":"server_error"}}`,
		"[DONE]",
	})
	resp, err := http.Get(srv.URL)
	require.NoError(t, err)
	defer resp.Body.Close()

	_, err = parseStream(resp.Body, time.Now())
	require.Error(t, err)
	require.Contains(t, err.Error(), "upstream exploded")
}

func TestParseStream_SingleContentChunk_NoITL(t *testing.T) {
	t.Parallel()
	srv := sseServer(t, 0, []string{
		roleChunk,
		contentChunk("only"),
		"[DONE]",
	})
	resp, err := http.Get(srv.URL)
	require.NoError(t, err)
	defer resp.Body.Close()

	res, err := parseStream(resp.Body, time.Now())
	require.NoError(t, err)
	require.Equal(t, "only", res.Content)
	require.Empty(t, res.ITL, "one content chunk → zero ITL samples")
	require.Greater(t, res.TTFT, time.Duration(0))
}

func TestParseStream_MultiTokenChunk(t *testing.T) {
	t.Parallel()
	// Simulates TGI / batched configurations where one SSE chunk carries multiple tokens.
	// We still count it as one ITL sample boundary — see RESEARCH.md §A.7.
	srv := sseServer(t, 5*time.Millisecond, []string{
		contentChunk("Hello world "),
		contentChunk("how are you?"),
		"[DONE]",
	})
	resp, err := http.Get(srv.URL)
	require.NoError(t, err)
	defer resp.Body.Close()

	res, err := parseStream(resp.Body, time.Now())
	require.NoError(t, err)
	require.Equal(t, "Hello world how are you?", res.Content)
	require.Len(t, res.ITL, 1)
}

// TestITL_ClockHonesty verifies that when chunks arrive on the wire with a known
// inter-chunk gap of G, the reported ITL values are tightly clustered around G —
// not inflated by HTTP setup, not skewed by goroutine scheduling.
//
// This is the "is my clock honest" check before pointing the extension at a real
// LLM server. Loose tolerance (±25%) accounts for goroutine scheduling jitter on
// loaded test runners.
func TestITL_ClockHonesty(t *testing.T) {
	t.Parallel()
	const (
		gap      = 50 * time.Millisecond
		nContent = 6 // → 5 ITL samples
		tol      = 12 * time.Millisecond
	)
	chunks := make([]string, 0, nContent+2)
	chunks = append(chunks, roleChunk)
	for i := 0; i < nContent; i++ {
		chunks = append(chunks, contentChunk(fmt.Sprintf("tok%d ", i)))
	}
	chunks = append(chunks, "[DONE]")

	srv := sseServer(t, gap, chunks)
	resp, err := http.Get(srv.URL)
	require.NoError(t, err)
	defer resp.Body.Close()

	res, err := parseStream(resp.Body, time.Now())
	require.NoError(t, err)
	require.Len(t, res.ITL, nContent-1, "5 inter-chunk gaps for 6 content chunks")

	for i, itl := range res.ITL {
		require.InDelta(t, float64(gap), float64(itl), float64(tol),
			"ITL[%d] = %v; want ≈ %v ± %v", i, itl, gap, tol)
	}
}

func TestParseOptions(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		in      any
		wantErr bool
		check   func(t *testing.T, o *Options)
	}{
		{
			name: "minimal override",
			in:   map[string]any{"model": "x"},
			check: func(t *testing.T, o *Options) {
				require.Equal(t, "x", o.Model)
				require.Equal(t, defaultBaseURL, o.BaseURL)
				require.Equal(t, defaultTimeout, o.Timeout)
			},
		},
		{
			name: "all defaults",
			in:   nil,
			check: func(t *testing.T, o *Options) {
				require.Equal(t, defaultModel, o.Model)
				require.Equal(t, defaultBaseURL, o.BaseURL)
			},
		},
		{
			name: "strips trailing slash",
			in:   map[string]any{"model": "x", "base_url": "http://h/v1/"},
			check: func(t *testing.T, o *Options) {
				require.Equal(t, "http://h/v1", o.BaseURL)
			},
		},
		{
			name: "timeout from float",
			in:   map[string]any{"model": "x", "timeout_ms": float64(2500)},
			check: func(t *testing.T, o *Options) {
				require.Equal(t, 2500*time.Millisecond, o.Timeout)
			},
		},
		{name: "wrong type", in: "string", wantErr: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			o, err := parseOptions(tc.in)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			tc.check(t, o)
		})
	}
}
