package llm

import (
	"context"
	"errors"
	"net/url"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// fakeTimeoutError implements net.Error with Timeout() returning true.
type fakeTimeoutError struct{}

func (fakeTimeoutError) Error() string   { return "fake timeout" }
func (fakeTimeoutError) Timeout() bool   { return true }
func (fakeTimeoutError) Temporary() bool { return true }

// Pure-function tests covering helpers that don't need a k6 VU runtime.

func TestChatError_ErrorAndUnwrap(t *testing.T) {
	t.Parallel()
	inner := errors.New("upstream down")
	ce := newChatError(errKindNetwork, inner)
	require.Equal(t, "upstream down", ce.Error())
	require.Same(t, inner, errors.Unwrap(ce))
	require.Equal(t, errKindNetwork, errorKind(ce))
	require.Equal(t, errKindNetwork, errorKind(inner)) // fallback when not a chatError
}

func TestClassifyTransportError(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		err  error
		want string
	}{
		{"nil", nil, ""},
		{"net.Error timeout", fakeTimeoutError{}, errKindTimeout},
		{"context.DeadlineExceeded", context.DeadlineExceeded, errKindTimeout},
		{"url.Error wrapping timeout", &url.Error{Op: "Get", URL: "http://x", Err: fakeTimeoutError{}}, errKindTimeout},
		{"plain error falls back to network", errors.New("connection refused"), errKindNetwork},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tc.want, classifyTransportError(tc.err))
		})
	}
}

func TestChatRequest_TagSet(t *testing.T) {
	t.Parallel()
	t.Run("nil when empty", func(t *testing.T) {
		t.Parallel()
		p := &chatRequest{}
		require.Nil(t, p.tagSet())
	})
	t.Run("cache_state only", func(t *testing.T) {
		t.Parallel()
		p := &chatRequest{cacheState: "warm"}
		require.Equal(t, map[string]string{"cache_state": "warm"}, p.tagSet())
	})
	t.Run("user tags only", func(t *testing.T) {
		t.Parallel()
		p := &chatRequest{tags: map[string]string{"region": "us-east"}}
		require.Equal(t, map[string]string{"region": "us-east"}, p.tagSet())
	})
	t.Run("merge", func(t *testing.T) {
		t.Parallel()
		p := &chatRequest{cacheState: "cold", tags: map[string]string{"shape": "short"}}
		got := p.tagSet()
		require.Equal(t, "cold", got["cache_state"])
		require.Equal(t, "short", got["shape"])
		require.Len(t, got, 2)
	})
}

func TestChatResult_ToJSObject(t *testing.T) {
	t.Parallel()
	r := &chatResult{
		Content:          "hello",
		TTFT:             50 * time.Millisecond,
		ITL:              []time.Duration{10 * time.Millisecond, 12 * time.Millisecond},
		Duration:         200 * time.Millisecond,
		ResponseHeaders:  20 * time.Millisecond,
		Chunks:           3,
		PromptTokens:     8,
		CompletionTokens: 4,
		FinishReason:     "stop",
	}
	m := r.toJSObject()
	require.Equal(t, "hello", m["content"])
	ttft, ok := m["ttft_ms"].(float64)
	require.True(t, ok)
	require.InDelta(t, 50.0, ttft, 1e-6)
	itl, ok := m["itl_ms"].([]float64)
	require.True(t, ok)
	require.Len(t, itl, 2)
	require.InDelta(t, 10.0, itl[0], 1e-6)
	require.InDelta(t, 12.0, itl[1], 1e-6)
	dur, ok := m["duration_ms"].(float64)
	require.True(t, ok)
	require.InDelta(t, 200.0, dur, 1e-6)
	rh, ok := m["response_headers_ms"].(float64)
	require.True(t, ok)
	require.InDelta(t, 20.0, rh, 1e-6)
	require.Equal(t, 3, m["chunks"])
	require.Equal(t, 8, m["prompt_tokens"])
	require.Equal(t, 4, m["completion_tokens"])
	require.Equal(t, "stop", m["finish_reason"])
	// TPOT = (200 - 50) / (4 - 1) = 50ms
	tpot, ok := m["tpot_ms"].(float64)
	require.True(t, ok)
	require.InDelta(t, 50.0, tpot, 1e-6)
}

func TestChatResult_ToJSObject_NotDerivableTPOT(t *testing.T) {
	t.Parallel()
	r := &chatResult{Content: "x", Duration: 10 * time.Millisecond, CompletionTokens: 1}
	m := r.toJSObject()
	// Single token cannot yield TPOT; result should still expose the field as 0.
	tpot, ok := m["tpot_ms"].(float64)
	require.True(t, ok)
	require.InDelta(t, 0.0, tpot, 1e-9)
}
