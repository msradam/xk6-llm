package llm

import (
	"context"
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
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		require.True(t, ok)
		for i, c := range chunks {
			if i > 0 && gap > 0 {
				time.Sleep(gap)
			}
			_, _ = fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// httpGet does an HTTP GET with an explicit context. Tests use this in place of
// http.Get so noctx is satisfied.
func httpGet(t *testing.T, url string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, nil)
	require.NoError(t, err)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

func contentChunk(s string) string {
	return fmt.Sprintf(`{"choices":[{"index":0,"delta":{"content":%q},"finish_reason":null}]}`, s)
}

const roleChunk = `{"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`

func usageChunk(prompt, completion int) string {
	return fmt.Sprintf(`{"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
		prompt, completion, prompt+completion)
}

// newTestClient builds a Client suitable for testing doChat/parseStream paths.
// The module field is nil, so emit/emitError are no-ops (they guard on vu.State()==nil).
func newTestClient(t *testing.T, baseURL string) *Client {
	t.Helper()
	return &Client{
		mod: nil,
		cfg: &Options{
			BaseURL: baseURL,
			Model:   "test-model",
			Timeout: 10 * time.Second,
		},
		http: &http.Client{Timeout: 10 * time.Second},
	}
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
	resp := httpGet(t, srv.URL)

	start := time.Now()
	res, err := parseStream(resp.Body, start)
	require.NoError(t, err)

	require.Equal(t, "Hello world!", res.Content)
	require.Equal(t, 7, res.PromptTokens)
	require.Equal(t, 3, res.CompletionTokens)
	require.Equal(t, 3, res.Chunks, "3 content-bearing chunks counted")

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
	resp := httpGet(t, srv.URL)

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
	resp := httpGet(t, srv.URL)

	_, err := parseStream(resp.Body, time.Now())
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
	resp := httpGet(t, srv.URL)

	res, err := parseStream(resp.Body, time.Now())
	require.NoError(t, err)
	require.Equal(t, "only", res.Content)
	require.Empty(t, res.ITL, "one content chunk → zero ITL samples")
	require.Greater(t, res.TTFT, time.Duration(0))
}

func TestParseStream_MultiTokenChunk(t *testing.T) {
	t.Parallel()
	// Simulates TGI or batched configurations where one SSE chunk carries multiple tokens.
	// Counted as one ITL sample boundary regardless. See RESEARCH.md §A.7.
	srv := sseServer(t, 5*time.Millisecond, []string{
		contentChunk("Hello world "),
		contentChunk("how are you?"),
		"[DONE]",
	})
	resp := httpGet(t, srv.URL)

	res, err := parseStream(resp.Body, time.Now())
	require.NoError(t, err)
	require.Equal(t, "Hello world how are you?", res.Content)
	require.Len(t, res.ITL, 1)
}

// TestITL_ClockHonesty verifies that when chunks arrive on the wire with a known
// inter-chunk gap of G, the reported ITL values cluster tightly around G:
// not inflated by HTTP setup, not skewed by goroutine scheduling.
//
// Loose tolerance (around 25%) accounts for goroutine scheduling jitter on
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
	for i := range nContent {
		chunks = append(chunks, contentChunk(fmt.Sprintf("tok%d ", i)))
	}
	chunks = append(chunks, "[DONE]")

	srv := sseServer(t, gap, chunks)
	resp := httpGet(t, srv.URL)

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

func TestParseSLO(t *testing.T) {
	t.Parallel()
	t.Run("from options", func(t *testing.T) {
		t.Parallel()
		o, err := parseOptions(map[string]any{
			"model": "x",
			"slo":   map[string]any{"ttft_ms": float64(500), "tpot_ms": float64(50), "e2el_ms": float64(5000)},
		})
		require.NoError(t, err)
		require.NotNil(t, o.DefaultSLO)
		require.Equal(t, 500.0, o.DefaultSLO.TTFTMs)
		require.Equal(t, 50.0, o.DefaultSLO.TPOTMs)
		require.Equal(t, 5000.0, o.DefaultSLO.E2ELMs)
		require.False(t, o.DefaultSLO.Empty())
	})
	t.Run("empty when zero", func(t *testing.T) {
		t.Parallel()
		s := &SLOPredicate{}
		require.True(t, s.Empty())
		var nilSLO *SLOPredicate
		require.True(t, nilSLO.Empty())
	})
	t.Run("partial", func(t *testing.T) {
		t.Parallel()
		s, err := parseSLO(map[string]any{"ttft_ms": float64(500)})
		require.NoError(t, err)
		require.Equal(t, 500.0, s.TTFTMs)
		require.Equal(t, 0.0, s.TPOTMs)
		require.False(t, s.Empty())
	})
	t.Run("wrong type", func(t *testing.T) {
		t.Parallel()
		_, err := parseSLO(map[string]any{"ttft_ms": "500"})
		require.Error(t, err)
	})
}

func TestChatResult_TPOT(t *testing.T) {
	t.Parallel()
	t.Run("happy path", func(t *testing.T) {
		t.Parallel()
		r := &chatResult{
			TTFT:             100 * time.Millisecond,
			Duration:         1100 * time.Millisecond,
			CompletionTokens: 11,
		}
		require.True(t, r.TPOTDerivable())
		// (1100 - 100) / (11 - 1) = 100ms
		require.Equal(t, 100*time.Millisecond, r.TPOT())
	})
	t.Run("single token: not derivable", func(t *testing.T) {
		t.Parallel()
		r := &chatResult{TTFT: 100 * time.Millisecond, Duration: 200 * time.Millisecond, CompletionTokens: 1}
		require.False(t, r.TPOTDerivable())
		require.Zero(t, r.TPOT())
	})
	t.Run("zero tokens: not derivable", func(t *testing.T) {
		t.Parallel()
		r := &chatResult{TTFT: 100 * time.Millisecond, Duration: 200 * time.Millisecond, CompletionTokens: 0}
		require.False(t, r.TPOTDerivable())
	})
	t.Run("ttft equals duration: not derivable", func(t *testing.T) {
		t.Parallel()
		r := &chatResult{TTFT: 100 * time.Millisecond, Duration: 100 * time.Millisecond, CompletionTokens: 10}
		require.False(t, r.TPOTDerivable())
	})
}

func TestParseChatRequest_StripsControlFields(t *testing.T) {
	t.Parallel()
	c := newTestClient(t, "http://example")
	req, err := c.parseChatRequest(map[string]any{
		"messages":    []any{map[string]any{"role": "user", "content": "hi"}},
		"max_tokens":  float64(64),
		"slo":         map[string]any{"ttft_ms": float64(500)},
		"cache_state": "cold",
		"tags":        map[string]any{"region": "us-east"},
	})
	require.NoError(t, err)
	// Control fields must NOT leak into the upstream OpenAI body.
	require.NotContains(t, req.body, "slo")
	require.NotContains(t, req.body, "cache_state")
	require.NotContains(t, req.body, "tags")
	// OpenAI fields preserved.
	require.Contains(t, req.body, "messages")
	require.Contains(t, req.body, "max_tokens")
	// Parsed control fields visible on chatRequest.
	require.Equal(t, 500.0, req.slo.TTFTMs)
	require.Equal(t, "cold", req.cacheState)
	require.Equal(t, "us-east", req.tags["region"])
}

func TestParseChatRequest_InheritsDefaultSLO(t *testing.T) {
	t.Parallel()
	c := newTestClient(t, "http://example")
	c.cfg.DefaultSLO = &SLOPredicate{TTFTMs: 1000}
	req, err := c.parseChatRequest(map[string]any{"messages": []any{}})
	require.NoError(t, err)
	require.Same(t, c.cfg.DefaultSLO, req.slo, "absent per-request SLO inherits client default")
}

func TestParseChatRequest_PerCallSLOOverride(t *testing.T) {
	t.Parallel()
	c := newTestClient(t, "http://example")
	c.cfg.DefaultSLO = &SLOPredicate{TTFTMs: 1000}
	req, err := c.parseChatRequest(map[string]any{
		"messages": []any{},
		"slo":      map[string]any{"ttft_ms": float64(200)},
	})
	require.NoError(t, err)
	require.Equal(t, 200.0, req.slo.TTFTMs, "per-call SLO overrides default")
}

func TestParseChatRequest_InvalidCacheState(t *testing.T) {
	t.Parallel()
	c := newTestClient(t, "http://example")
	_, err := c.parseChatRequest(map[string]any{"cache_state": "lukewarm"})
	require.Error(t, err)
}

func TestErrorClassification(t *testing.T) {
	t.Parallel()
	t.Run("http 4xx", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "bad request", http.StatusBadRequest)
		}))
		defer srv.Close()
		c := newTestClient(t, srv.URL)
		_, err := c.doChat(context.Background(), &chatRequest{body: map[string]any{}})
		require.Error(t, err)
		require.Equal(t, errKindHTTP4xx, errorKind(err))
	})
	t.Run("http 5xx", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "boom", http.StatusInternalServerError)
		}))
		defer srv.Close()
		c := newTestClient(t, srv.URL)
		_, err := c.doChat(context.Background(), &chatRequest{body: map[string]any{}})
		require.Error(t, err)
		require.Equal(t, errKindHTTP5xx, errorKind(err))
	})
	t.Run("network", func(t *testing.T) {
		t.Parallel()
		// Port 1 is reserved; on macOS/Linux this gives ECONNREFUSED.
		c := newTestClient(t, "http://127.0.0.1:1")
		_, err := c.doChat(context.Background(), &chatRequest{body: map[string]any{}})
		require.Error(t, err)
		require.Equal(t, errKindNetwork, errorKind(err))
	})
	t.Run("timeout", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			time.Sleep(500 * time.Millisecond)
		}))
		defer srv.Close()
		c := newTestClient(t, srv.URL)
		c.http = &http.Client{Timeout: 50 * time.Millisecond}
		_, err := c.doChat(context.Background(), &chatRequest{body: map[string]any{}})
		require.Error(t, err)
		require.Equal(t, errKindTimeout, errorKind(err))
	})
	t.Run("stream", func(t *testing.T) {
		t.Parallel()
		srv := sseServer(t, 0, []string{
			roleChunk,
			contentChunk("partial"),
			`{"error":{"message":"upstream exploded","type":"server_error"}}`,
			"[DONE]",
		})
		c := newTestClient(t, "")
		// Point Client at the SSE server; we bypass the URL build by setting BaseURL.
		c.cfg.BaseURL = srv.URL
		// httptest.NewServer responds at "/" to any path.
		_, err := c.doChat(context.Background(), &chatRequest{body: map[string]any{}})
		require.Error(t, err)
		require.Equal(t, errKindStream, errorKind(err))
	})
}

func TestDoChat_ResponseHeadersTiming(t *testing.T) {
	t.Parallel()
	const headerDelay = 30 * time.Millisecond
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(headerDelay) // hold off WriteHeader
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		flusher.Flush()
		time.Sleep(20 * time.Millisecond)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", contentChunk("hi"))
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	c := newTestClient(t, srv.URL)
	res, err := c.doChat(context.Background(), &chatRequest{body: map[string]any{}})
	require.NoError(t, err)
	require.GreaterOrEqual(t, res.ResponseHeaders, 25*time.Millisecond)
	require.Less(t, res.ResponseHeaders, 100*time.Millisecond)
	// TTFT must include the header delay + the post-header gap before first content.
	require.GreaterOrEqual(t, res.TTFT, res.ResponseHeaders, "TTFT >= response_headers")
}
