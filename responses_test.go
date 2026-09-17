package llm

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// Event sequences below are the shapes captured from the live Responses API on
// 2026-09-10 against gpt-4.1-mini and gpt-5-nano.

const (
	respCreated     = `{"type":"response.created","sequence_number":0}`
	respInProgress  = `{"type":"response.in_progress","sequence_number":1}`
	respItemAdded   = `{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress"}}`
	respPartAdded   = `{"type":"response.content_part.added","output_index":0,"content_index":0}`
	respTextDone    = `{"type":"response.output_text.done","output_index":0}`
	respPartDone    = `{"type":"response.content_part.done","output_index":0}`
	respItemDoneMsg = `{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","status":"completed"}}`
)

func respText(s string) string {
	return `{"type":"response.output_text.delta","content_index":0,"output_index":0,"delta":"` + s + `"}`
}

func respCompleted(in, out, reasoning int) string {
	return `{"type":"response.completed","response":{"status":"completed","incomplete_details":null,"usage":{` +
		`"input_tokens":` + itoa(in) + `,"input_tokens_details":{"cached_tokens":0},` +
		`"output_tokens":` + itoa(out) + `,"output_tokens_details":{"reasoning_tokens":` + itoa(reasoning) + `},` +
		`"total_tokens":` + itoa(in+out) + `}}}`
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

func TestParseResponsesStream_TextAndUsage(t *testing.T) {
	t.Parallel()
	srv := sseServer(t, 10*time.Millisecond, []string{
		respCreated,
		respInProgress,
		respItemAdded,
		respPartAdded,
		respText("A Prometheus ex"),
		respText("emplar links a metric"),
		respText(" to a trace."),
		respTextDone,
		respPartDone,
		respItemDoneMsg,
		respCompleted(8, 10, 0),
	})
	resp := httpGet(t, srv.URL)

	start := time.Now()
	res, err := parseResponsesStream(context.Background(), resp.Body, start, abortPolicy{})
	require.NoError(t, err)

	require.Equal(t, "A Prometheus exemplar links a metric to a trace.", res.Content)
	require.Equal(t, 3, res.Chunks)
	require.Equal(t, 8, res.PromptTokens)
	require.Equal(t, 10, res.CompletionTokens)
	require.Equal(t, "completed", res.FinishReason)
	require.Len(t, res.ITL, 2, "3 text deltas → 2 ITL samples")

	// Four structural events precede the first text delta at 10ms spacing.
	// They must not establish TTFT; they are this wire's role-only delta.
	require.GreaterOrEqual(t, res.TTFT, 38*time.Millisecond,
		"created/in_progress/output_item.added/content_part.added must not set TTFT")
	require.Equal(t, res.TTFT, res.TTFText, "no reasoning phase, so these coincide")
}

// TestParseResponsesStream_ReasoningIncomplete reproduces gpt-5-nano at
// max_output_tokens=200: all output tokens are spent on reasoning, no text is
// ever emitted, and the stream terminates with response.incomplete.
//
// This is the case a naive parser reports as a successful empty completion.
func TestParseResponsesStream_ReasoningIncomplete(t *testing.T) {
	t.Parallel()
	srv := sseServer(t, 5*time.Millisecond, []string{
		respCreated,
		respInProgress,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"in_progress"}}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","status":"completed"}}`,
		`{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":17,"output_tokens":200,"output_tokens_details":{"reasoning_tokens":200},"total_tokens":217}}}`,
	})
	resp := httpGet(t, srv.URL)

	res, err := parseResponsesStream(context.Background(), resp.Body, time.Now(), abortPolicy{})
	require.NoError(t, err, "an incomplete response is a result, not a transport error")

	require.Empty(t, res.Content)
	require.Equal(t, 0, res.Chunks)
	require.Equal(t, "max_output_tokens", res.FinishReason,
		"the incomplete reason is more useful than the bare status")
	require.Equal(t, 200, res.CompletionTokens)
	require.Equal(t, 200, res.ThinkingTokens)

	// Every output token was reasoning, so no text streamed and TPOT is not
	// derivable, and reporting one would be inventing a number.
	tokens, _ := res.streamedOutput()
	require.Equal(t, 0, tokens)
	require.False(t, res.TPOTDerivable())
	require.Zero(t, res.TPOT())
}

func TestParseResponsesStream_ReasoningSummaryStreams(t *testing.T) {
	t.Parallel()
	// Unlike Anthropic, reasoning summary text streams incrementally on this
	// wire, so reasoning deltas are real observable events.
	summary := func(s string) string {
		return `{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"` + s + `"}`
	}
	srv := sseServer(t, 10*time.Millisecond, []string{
		respCreated,
		summary("Considering "),
		summary("exemplars. "),
		summary("Now answering."),
		`{"type":"response.reasoning_summary_text.done","output_index":0}`,
		respText("An exemplar"),
		respText(" links a metric to a trace."),
		respCompleted(17, 159, 64),
	})
	resp := httpGet(t, srv.URL)

	start := time.Now()
	res, err := parseResponsesStream(context.Background(), resp.Body, start, abortPolicy{})
	require.NoError(t, err)

	require.Equal(t, "An exemplar links a metric to a trace.", res.Content,
		"reasoning summary must not leak into the completion")
	require.Equal(t, 3, res.ThinkingChunks)
	require.Equal(t, 2, res.Chunks)
	require.Equal(t, 64, res.ThinkingTokens)

	require.Greater(t, res.TTFText, res.TTFT,
		"TTFT lands on the first reasoning delta, TTFText later")
	require.Len(t, res.ITL, 1, "reasoning deltas contribute no ITL samples")

	// TPOT over the 95 streamed text tokens, from the text phase.
	tokens, from := res.streamedOutput()
	require.Equal(t, 95, tokens)
	require.Equal(t, res.TTFText, from)
	require.True(t, res.TPOTDerivable())
}

func TestParseResponsesStream_FunctionCall(t *testing.T) {
	t.Parallel()
	// Captured shape: identity on output_item.added, arguments streamed as
	// fragments, then an authoritative assembled string on
	// function_call_arguments.done.
	srv := sseServer(t, 5*time.Millisecond, []string{
		respCreated,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_7Iwe","name":"get_weather"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\""}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"city\":\"Ber"}`,
		`{"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"city\":\"Berlin\"}"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\"city\":\"Berlin\"}","call_id":"call_7Iwe","name":"get_weather"}}`,
		respCompleted(60, 18, 0),
	})
	resp := httpGet(t, srv.URL)

	res, err := parseResponsesStream(context.Background(), resp.Body, time.Now(), abortPolicy{})
	require.NoError(t, err)

	require.Len(t, res.ToolCalls, 1)
	require.Equal(t, "call_7Iwe", res.ToolCalls[0].ID, "call_id is the id the caller echoes back")
	require.Equal(t, "get_weather", res.ToolCalls[0].Name)
	require.JSONEq(t, `{"city":"Berlin"}`, res.ToolCalls[0].Arguments,
		"the assembled arguments replace the streamed fragments")
	require.Positive(t, res.TTFT, "a function call establishes TTFT")
	require.Empty(t, res.ITL, "argument fragments contribute no ITL samples")
}

func TestParseResponsesStream_Failed(t *testing.T) {
	t.Parallel()
	srv := sseServer(t, time.Millisecond, []string{
		respCreated,
		respText("partial"),
		`{"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"upstream exploded"}}}`,
	})
	resp := httpGet(t, srv.URL)

	_, err := parseResponsesStream(context.Background(), resp.Body, time.Now(), abortPolicy{})
	require.ErrorContains(t, err, "upstream exploded")
}

func TestResponsesBody_TranslatesChatShape(t *testing.T) {
	t.Parallel()
	got := responsesBody(map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "be terse"},
			map[string]any{"role": "user", "content": "hi"},
		},
		"max_tokens":  128,
		"seed":        42,
		"stop":        []any{"x"},
		"temperature": 0.2,
	}, "gpt-4.1-mini")

	require.Equal(t, "gpt-4.1-mini", got["model"])
	require.Equal(t, true, got["stream"])
	require.Equal(t, 128, got["max_output_tokens"], "max_tokens -> max_output_tokens")
	require.NotContains(t, got, "max_tokens")
	require.Equal(t, "be terse", got["instructions"], "system -> instructions")
	require.NotContains(t, got, "messages", "messages -> input")
	require.NotContains(t, got, "seed")
	require.NotContains(t, got, "stop", "Responses has no stop parameter")
	require.Equal(t, 0.2, got["temperature"])

	input, ok := got["input"].([]any)
	require.True(t, ok)
	require.Len(t, input, 1, "the system message left the input array")
	require.Equal(t, "user", asMap(t, input[0])["role"])
}

func TestResponsesTools_AreFlat(t *testing.T) {
	t.Parallel()
	// Responses declares tools flat; chat completions nests them under
	// "function". Sending the nested form is a 400.
	got := responsesTools([]any{
		map[string]any{"type": "function", "function": map[string]any{
			"name": "get_weather", "description": "w",
			"parameters": map[string]any{"type": "object"},
		}},
		map[string]any{"type": "function", "function": map[string]any{"name": "now"}},
		map[string]any{"type": "function", "name": "already_flat", "parameters": map[string]any{}},
	})
	list, ok := got.([]any)
	require.True(t, ok)
	require.Len(t, list, 3)

	first := asMap(t, list[0])
	require.Equal(t, "function", first["type"])
	require.Equal(t, "get_weather", first["name"], "name is top-level, not nested")
	require.Equal(t, "w", first["description"])
	require.Contains(t, first, "parameters")
	require.NotContains(t, first, "function")

	require.Equal(t, map[string]any{"type": "object", "properties": map[string]any{}},
		asMap(t, list[1])["parameters"], "no-arg tool gets an empty schema")
	require.Equal(t, "already_flat", asMap(t, list[2])["name"])
}

func TestResponsesInput_AgentLoop(t *testing.T) {
	t.Parallel()
	input, instructions := responsesInput([]any{
		map[string]any{"role": "system", "content": "sys"},
		map[string]any{"role": "user", "content": "weather?"},
		map[string]any{"role": "assistant", "content": "checking", "tool_calls": []any{
			map[string]any{"id": "call_a", "type": "function", "function": map[string]any{
				"name": "get_weather", "arguments": `{"city":"Berlin"}`,
			}},
		}},
		map[string]any{"role": "tool", "tool_call_id": "call_a", "content": "12C"},
	})

	require.Equal(t, "sys", instructions)
	// user, assistant text, function_call, function_call_output
	require.Len(t, input, 4)

	call := asMap(t, input[2])
	require.Equal(t, "function_call", call["type"])
	require.Equal(t, "call_a", call["call_id"])
	require.Equal(t, "get_weather", call["name"])
	require.Equal(t, `{"city":"Berlin"}`, call["arguments"],
		"Responses keeps arguments as a JSON string, unlike Anthropic")

	out := asMap(t, input[3])
	require.Equal(t, "function_call_output", out["type"])
	require.Equal(t, "call_a", out["call_id"])
	require.Equal(t, "12C", out["output"])
}

func TestResponsesToolChoice(t *testing.T) {
	t.Parallel()
	// Bare strings carry over unchanged on this wire.
	require.Equal(t, "auto", responsesToolChoice("auto"))
	require.Equal(t, "required", responsesToolChoice("required"))
	require.Equal(t, map[string]any{"type": "function", "name": "f"},
		responsesToolChoice(map[string]any{
			"type": "function", "function": map[string]any{"name": "f"},
		}))
}

func TestParseOptions_WireResponses(t *testing.T) {
	t.Parallel()
	o, err := parseOptions(map[string]any{"wire": "responses"})
	require.NoError(t, err)
	require.Equal(t, WireResponses, o.Wire)

	_, err = parseOptions(map[string]any{"wire": "gemini"})
	require.ErrorContains(t, err, "wire")
}

func TestParseResponsesStream_CachedTokens(t *testing.T) {
	t.Parallel()
	// input_tokens_details.cached_tokens is what makes a repeated-prompt probe
	// interpretable. Observed live on gpt-4.1-mini: 1152 of 1337 input tokens
	// cached on some requests and 0 on others, for byte-identical prompts.
	srv := sseServer(t, time.Millisecond, []string{
		respCreated,
		respText("hi"),
		`{"type":"response.completed","response":{"status":"completed","usage":{` +
			`"input_tokens":1337,"input_tokens_details":{"cached_tokens":1152},` +
			`"output_tokens":35,"output_tokens_details":{"reasoning_tokens":0},` +
			`"total_tokens":1372}}}`,
	})
	resp := httpGet(t, srv.URL)

	res, err := parseResponsesStream(context.Background(), resp.Body, time.Now(), abortPolicy{})
	require.NoError(t, err)
	require.Equal(t, 1337, res.PromptTokens)
	require.Equal(t, 1152, res.CachedTokens, "a sub-bucket of PromptTokens, not additive")
	require.Equal(t, 35, res.CompletionTokens)
}
