package llm

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// Anthropic Messages stream events. message_start, ping, and the empty-text
// content_block_start are the analogue of OpenAI's role-only delta: they arrive
// before any content and must not set TTFT.
const (
	anthMessageStart = `{"type":"message_start","message":{"id":"msg_1","role":"assistant","usage":{"input_tokens":7,"output_tokens":0}}}`
	anthPing         = `{"type":"ping"}`
	anthBlockStart   = `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
	anthBlockStop    = `{"type":"content_block_stop","index":0}`
	anthMessageDelta = `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`
	anthMessageStop  = `{"type":"message_stop"}`
)

func anthText(s string) string {
	return `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + s + `"}}`
}

func TestParseAnthropicStream_SkipsPreambleForTTFT(t *testing.T) {
	t.Parallel()
	srv := sseServer(t, 10*time.Millisecond, []string{
		anthMessageStart,
		anthPing,
		anthBlockStart,
		anthText("Hello"),
		anthText(" world"),
		anthText("!"),
		anthBlockStop,
		anthMessageDelta,
		anthMessageStop,
	})
	resp := httpGet(t, srv.URL)

	start := time.Now()
	res, err := parseAnthropicStream(context.Background(), resp.Body, start, abortPolicy{})
	require.NoError(t, err)

	require.Equal(t, "Hello world!", res.Content)
	require.Equal(t, 7, res.PromptTokens, "input_tokens from message_start")
	require.Equal(t, 3, res.CompletionTokens, "output_tokens from message_delta")
	require.Equal(t, 3, res.Chunks)
	require.Equal(t, "end_turn", res.FinishReason)

	// 3 content deltas → 2 ITL samples, same rule as the OpenAI path.
	require.Len(t, res.ITL, 2)

	// TTFT is measured at the first text_delta. Three non-content events precede
	// it at 10ms spacing, so a parser that timed message_start would report
	// ~0ms and one that timed content_block_start would report ~20ms.
	require.GreaterOrEqual(t, res.TTFT, 28*time.Millisecond,
		"TTFT must not be set by message_start, ping, or the empty content_block_start")

	for _, itl := range res.ITL {
		require.GreaterOrEqual(t, itl, 5*time.Millisecond)
		require.Less(t, itl, 50*time.Millisecond, "ITL must exclude the start→first-content gap")
	}
}

func TestParseAnthropicStream_ToolUse(t *testing.T) {
	t.Parallel()
	// tool_use blocks carry id/name on content_block_start; arguments stream as
	// input_json_delta fragments that must be concatenated per index.
	srv := sseServer(t, 5*time.Millisecond, []string{
		anthMessageStart,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_a","name":"get_weather"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":":\"Berlin\"}"}}`,
		anthBlockStop,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":11}}`,
		anthMessageStop,
	})
	resp := httpGet(t, srv.URL)

	res, err := parseAnthropicStream(context.Background(), resp.Body, time.Now(), abortPolicy{})
	require.NoError(t, err)

	require.Len(t, res.ToolCalls, 1)
	require.Equal(t, "toolu_a", res.ToolCalls[0].ID)
	require.Equal(t, "get_weather", res.ToolCalls[0].Name)
	require.JSONEq(t, `{"city":"Berlin"}`, res.ToolCalls[0].Arguments)
	require.Equal(t, "tool_use", res.FinishReason)
	require.Empty(t, res.ITL, "tool-argument fragments must not contribute ITL samples")
	require.Positive(t, res.TTFT, "a tool_use block still establishes TTFT")
}

func TestParseAnthropicStream_MidStreamError(t *testing.T) {
	t.Parallel()
	srv := sseServer(t, time.Millisecond, []string{
		anthMessageStart,
		anthText("partial"),
		`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
	})
	resp := httpGet(t, srv.URL)

	_, err := parseAnthropicStream(context.Background(), resp.Body, time.Now(), abortPolicy{})
	require.ErrorContains(t, err, "Overloaded")
}

func TestParseAnthropicStream_StopsAtMessageStop(t *testing.T) {
	t.Parallel()
	// There is no [DONE] sentinel on this wire. Anything after message_stop must
	// not be consumed — a trailing event would otherwise inflate Chunks.
	srv := sseServer(t, time.Millisecond, []string{
		anthMessageStart,
		anthText("one"),
		anthMessageStop,
		anthText("must-not-be-read"),
	})
	resp := httpGet(t, srv.URL)

	res, err := parseAnthropicStream(context.Background(), resp.Body, time.Now(), abortPolicy{})
	require.NoError(t, err)
	require.Equal(t, "one", res.Content)
	require.Equal(t, 1, res.Chunks)
}

func TestAnthropicBody_HoistsSystemAndDropsOpenAIOnlyFields(t *testing.T) {
	t.Parallel()
	got := anthropicBody(map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "be terse"},
			map[string]any{"role": "user", "content": "hi"},
		},
		"seed":           42,
		"stream_options": map[string]any{"include_usage": true},
		"temperature":    0.0,
	}, "claude-sonnet-4-6", false)

	require.Equal(t, "claude-sonnet-4-6", got["model"])
	require.Equal(t, true, got["stream"])
	require.Equal(t, "be terse", got["system"], "system role hoisted to top-level param")
	require.Equal(t, 512, got["max_tokens"], "Messages requires max_tokens")

	msgs, ok := got["messages"].([]any)
	require.True(t, ok)
	require.Len(t, msgs, 1, "system message removed from the messages array")

	require.NotContains(t, got, "seed")
	require.NotContains(t, got, "stream_options")
	require.Contains(t, got, "temperature", "shared fields pass through")
}

func TestParseOptions_Wire(t *testing.T) {
	t.Parallel()

	o, err := parseOptions(map[string]any{})
	require.NoError(t, err)
	require.Equal(t, WireOpenAI, o.Wire, "default wire is openai")

	o, err = parseOptions(map[string]any{"wire": "anthropic"})
	require.NoError(t, err)
	require.Equal(t, WireAnthropic, o.Wire)

	_, err = parseOptions(map[string]any{"wire": "cohere"})
	require.ErrorContains(t, err, "wire")
}

// TestParseAnthropicStream_ReasoningModel replays the exact event sequence
// captured from claude-opus-5 on 2026-09-10, including the empty
// thinking_delta that Anthropic's default thinking display ("omitted")
// produces and the output_tokens_details reasoning breakout.
//
// Before reasoning deltas were handled, this stream reported a TPOT of a few
// hundred nanoseconds: 199 billed tokens divided into the wall-clock left
// after TTFT, when only ~41 of those tokens ever streamed. Against live
// claude-opus-5 the observed values were 130ns-3.23ms.
//
// The synthetic events below are spaced 10ms apart so the ordering assertions
// have room. On the live API the reasoning and first-text deltas arrive within
// microseconds of each other (see parseAnthropicStream's doc comment), so the
// TTFT/TTFText gap this test asserts is a property of the parser's ordering,
// not a latency saving to expect in production.
func TestParseAnthropicStream_ReasoningModel(t *testing.T) {
	t.Parallel()
	srv := sseServer(t, 10*time.Millisecond, []string{
		`{"type":"message_start","message":{"id":"msg_1","role":"assistant","usage":{"input_tokens":27,"output_tokens":2}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		anthPing,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"CAISlgYKjgEIERgCKkDdSUs6"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"A Prometheus ex"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"emplar is a sample."}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"input_tokens":27,"output_tokens":200,"output_tokens_details":{"thinking_tokens":159}}}`,
		anthMessageStop,
	})
	resp := httpGet(t, srv.URL)

	start := time.Now()
	res, err := parseAnthropicStream(context.Background(), resp.Body, start, abortPolicy{})
	require.NoError(t, err)

	require.Equal(t, "A Prometheus exemplar is a sample.", res.Content)
	require.Equal(t, 200, res.CompletionTokens)
	require.Equal(t, 159, res.ThinkingTokens, "reasoning breakout decoded")
	require.Equal(t, 2, res.ThinkingChunks, "thinking_delta and signature_delta both count")
	require.Equal(t, 2, res.Chunks, "Chunks counts text deltas only")

	// The stream is emitted at 10ms intervals. The first reasoning delta is
	// event 4 (~30ms) and the first text delta is event 8 (~70ms). TTFT must
	// track the reasoning delta, not the text delta.
	require.Greater(t, res.TTFText, res.TTFT,
		"time-to-first-text is later than TTFT on a reasoning model")
	require.Less(t, res.TTFT, 60*time.Millisecond,
		"TTFT must not be pushed out to the first text delta")
	require.Greater(t, res.TTFText, 60*time.Millisecond)

	// TPOT is computed over the 41 streamed text tokens from the text phase,
	// not over all 200 billed tokens from TTFT — the latter is what produced
	// sub-microsecond nonsense.
	require.True(t, res.TPOTDerivable())
	tokens, from := res.streamedOutput()
	require.Equal(t, 41, tokens)
	require.Equal(t, res.TTFText, from)
	require.Greater(t, res.TPOT(), time.Microsecond,
		"a plausible per-token time, not a nanosecond artifact")

	// ITL samples come from text deltas only; one empty reasoning delta says
	// nothing about cadence.
	require.Len(t, res.ITL, 1)
}

func TestChatResult_TPOT_NonReasoningPathUnchanged(t *testing.T) {
	t.Parallel()
	// With no reasoning tokens the definition must be exactly the original:
	// (duration - ttft) / (completion_tokens - 1). This is the OpenAI path.
	res := &chatResult{
		TTFT:             100 * time.Millisecond,
		Duration:         1100 * time.Millisecond,
		CompletionTokens: 11,
	}
	tokens, from := res.streamedOutput()
	require.Equal(t, 11, tokens)
	require.Equal(t, 100*time.Millisecond, from)
	require.Equal(t, 100*time.Millisecond, res.TPOT())
}

func TestParseAnthropicStream_CachedTokens(t *testing.T) {
	t.Parallel()
	// Anthropic reports the cached prefix as cache_read_input_tokens on
	// message_start.
	srv := sseServer(t, time.Millisecond, []string{
		`{"type":"message_start","message":{"id":"m","role":"assistant","usage":{` +
			`"input_tokens":2048,"cache_read_input_tokens":1900,"cache_creation_input_tokens":100,"output_tokens":0}}}`,
		anthText("hi"),
		anthMessageDelta,
		anthMessageStop,
	})
	resp := httpGet(t, srv.URL)

	res, err := parseAnthropicStream(context.Background(), resp.Body, time.Now(), abortPolicy{})
	require.NoError(t, err)
	// input_tokens on this wire excludes the cache buckets; the result
	// reports the inclusive total so it means what OpenAI's prompt_tokens
	// means.
	require.Equal(t, 2048+1900+100, res.PromptTokens)
	require.Equal(t, 1900, res.CachedTokens)
	require.Equal(t, 100, res.CacheWriteTokens, "cache writes are billed at a different rate, so a cost model needs the split")
}
