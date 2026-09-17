package llm

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// The field and shape expectations below were verified against the live
// Anthropic API on 2026-09-10: `stop`, `max_completion_tokens`, `user`, and
// OpenAI-shaped `tools` each return 400 "Extra inputs are not permitted" or a
// tag-mismatch error, while the translated forms are accepted.

func TestAnthropicBody_RenamesAndDropsOpenAIFields(t *testing.T) {
	t.Parallel()
	got := anthropicBody(map[string]any{
		"messages":              []any{map[string]any{"role": "user", "content": "hi"}},
		"stop":                  []any{"STOP"},
		"max_completion_tokens": 99,
		"user":                  "u1",
		"logit_bias":            map[string]any{"1": 1},
		"temperature":           0.5,
	}, "claude-haiku-4-5", false)

	require.Equal(t, []any{"STOP"}, got["stop_sequences"], "stop -> stop_sequences")
	require.NotContains(t, got, "stop")
	require.Equal(t, 99, got["max_tokens"], "max_completion_tokens -> max_tokens")
	require.NotContains(t, got, "max_completion_tokens")
	require.NotContains(t, got, "user")
	require.NotContains(t, got, "logit_bias")
	require.Equal(t, 0.5, got["temperature"], "shared fields survive")
}

func TestAnthropicBody_ExplicitMessagesFieldWins(t *testing.T) {
	t.Parallel()
	// A caller who sets both must not have their explicit Messages field
	// silently overwritten by the translated OpenAI one.
	got := anthropicBody(map[string]any{
		"messages":       []any{map[string]any{"role": "user", "content": "hi"}},
		"stop":           []any{"from_openai"},
		"stop_sequences": []any{"explicit"},
		"max_tokens":     7,
	}, "m", false)
	require.Equal(t, []any{"explicit"}, got["stop_sequences"])
	require.Equal(t, 7, got["max_tokens"])
}

func TestTranslateTools(t *testing.T) {
	t.Parallel()
	got := translateTools([]any{
		map[string]any{"type": "function", "function": map[string]any{
			"name":        "get_weather",
			"description": "w",
			"parameters": map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
			},
		}},
		// No parameters: Messages still requires input_schema.
		map[string]any{"type": "function", "function": map[string]any{"name": "now"}},
		// Already Messages-shaped: pass through untouched.
		map[string]any{"name": "native", "input_schema": map[string]any{"type": "object"}},
	})

	list, ok := got.([]any)
	require.True(t, ok)
	require.Len(t, list, 3)

	first := asMap(t, list[0])
	require.Equal(t, "get_weather", first["name"])
	require.Equal(t, "w", first["description"])
	require.Contains(t, first, "input_schema")
	require.NotContains(t, first, "function", "the OpenAI wrapper is removed")
	require.NotContains(t, first, "parameters")

	second := asMap(t, list[1])
	require.Equal(t, map[string]any{"type": "object", "properties": map[string]any{}},
		second["input_schema"], "a no-arg tool gets an empty object schema")

	require.Equal(t, "native", asMap(t, list[2])["name"])
}

func TestTranslateToolChoice(t *testing.T) {
	t.Parallel()
	require.Equal(t, map[string]any{"type": "auto"}, translateToolChoice("auto"))
	require.Equal(t, map[string]any{"type": "any"}, translateToolChoice("required"))
	require.Equal(t, map[string]any{"type": "none"}, translateToolChoice("none"))
	require.Equal(t, map[string]any{"type": "tool", "name": "f"},
		translateToolChoice(map[string]any{
			"type": "function", "function": map[string]any{"name": "f"},
		}))
	// Already Messages-shaped, and unknown strings, pass through.
	native := map[string]any{"type": "any"}
	require.Equal(t, native, translateToolChoice(native))
	require.Equal(t, "weird", translateToolChoice("weird"))
}

func TestTranslateMessages_AgentLoop(t *testing.T) {
	t.Parallel()
	// A full OpenAI tool-calling turn: system, user, assistant with two
	// parallel tool_calls, then two tool results.
	msgs, system := translateMessages([]any{
		map[string]any{"role": "system", "content": "be terse"},
		map[string]any{"role": "user", "content": "weather and time?"},
		map[string]any{"role": "assistant", "content": "checking", "tool_calls": []any{
			map[string]any{"id": "call_a", "type": "function", "function": map[string]any{
				"name": "get_weather", "arguments": `{"city":"Berlin"}`,
			}},
			map[string]any{"id": "call_b", "type": "function", "function": map[string]any{
				"name": "get_time", "arguments": "",
			}},
		}},
		map[string]any{"role": "tool", "tool_call_id": "call_a", "content": "12C"},
		map[string]any{"role": "tool", "tool_call_id": "call_b", "content": "10:00"},
	})

	require.Equal(t, "be terse", system, "system hoisted out of the array")

	// user, assistant(tool_use), user(tool_result x2). The two tool results
	// merge into ONE user message.
	require.Len(t, msgs, 3)

	assistant := asMap(t, msgs[1])
	require.Equal(t, "assistant", assistant["role"])
	blocks := asList(t, assistant["content"])
	require.Len(t, blocks, 3, "leading text block plus two tool_use blocks")
	require.Equal(t, "text", asMap(t, blocks[0])["type"])

	call := asMap(t, blocks[1])
	require.Equal(t, "tool_use", call["type"])
	require.Equal(t, "call_a", call["id"])
	require.Equal(t, "get_weather", call["name"])
	require.Equal(t, map[string]any{"city": "Berlin"}, call["input"],
		"arguments decoded from JSON string to object")

	noArgs := asMap(t, blocks[2])
	require.Equal(t, map[string]any{}, noArgs["input"],
		"empty arguments become an empty object, not a nil input")

	results := asMap(t, msgs[2])
	require.Equal(t, "user", results["role"], "tool results are carried by a user message")
	resultBlocks := asList(t, results["content"])
	require.Len(t, resultBlocks, 2, "consecutive tool results merge into one message")
	require.Equal(t, "tool_result", asMap(t, resultBlocks[0])["type"])
	require.Equal(t, "call_a", asMap(t, resultBlocks[0])["tool_use_id"])
	require.Equal(t, "12C", asMap(t, resultBlocks[0])["content"])
}

func TestTranslateMessages_PlainChatUnchanged(t *testing.T) {
	t.Parallel()
	in := []any{
		map[string]any{"role": "user", "content": "hi"},
		map[string]any{"role": "assistant", "content": "hello"},
		map[string]any{"role": "user", "content": "bye"},
	}
	msgs, system := translateMessages(in)
	require.Empty(t, system)
	require.Equal(t, in, msgs, "a plain conversation passes through untouched")
}

func TestDecodeToolArguments(t *testing.T) {
	t.Parallel()
	require.Equal(t, map[string]any{"a": "b"}, decodeToolArguments(`{"a":"b"}`))
	require.Equal(t, map[string]any{}, decodeToolArguments(""))
	require.Equal(t, map[string]any{}, decodeToolArguments("   "))
	// A malformed argument echoed from a previous turn must not fail the
	// request; Messages requires input to be an object.
	require.Equal(t, map[string]any{}, decodeToolArguments(`{"a":`))
	require.Equal(t, map[string]any{}, decodeToolArguments(`"a string"`))
	require.Equal(t, map[string]any{"x": 1}, decodeToolArguments(map[string]any{"x": 1}))
	require.Equal(t, map[string]any{}, decodeToolArguments(nil))
}

func TestAbortBudget_CountsReasoningChunks(t *testing.T) {
	t.Parallel()
	// A reasoning-only stream: no text ever arrives. An abort budget that
	// counted text chunks alone would never fire, and that is the exact stream you most
	// want to cut off.
	reasoning := `{"choices":[{"index":0,"delta":{"reasoning_content":"thinking "},"finish_reason":null}]}`
	srv := sseServer(t, 2*time.Millisecond, []string{
		roleChunk, reasoning, reasoning, reasoning, reasoning, reasoning,
		contentChunk("never reached"),
		"[DONE]",
	})
	resp := httpGet(t, srv.URL)

	res, err := parseStream(context.Background(), resp.Body, time.Now(), abortPolicy{MaxTokens: 3})
	require.NoError(t, err)
	require.True(t, res.Aborted, "budget must fire on reasoning chunks alone")
	require.Equal(t, 3, res.ThinkingChunks)
	require.Equal(t, 0, res.Chunks)
	require.Empty(t, res.Content)
}

func TestAbortBudget_CountsReasoningChunks_AnthropicWire(t *testing.T) {
	t.Parallel()
	think := `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"x"}}`
	srv := sseServer(t, 2*time.Millisecond, []string{
		anthMessageStart, think, think, think, think,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"never"}}`,
		anthMessageStop,
	})
	resp := httpGet(t, srv.URL)

	res, err := parseAnthropicStream(context.Background(), resp.Body, time.Now(), abortPolicy{MaxTokens: 2})
	require.NoError(t, err)
	require.True(t, res.Aborted)
	require.Equal(t, 2, res.ThinkingChunks)
	require.Empty(t, res.Content)
}

func TestTranslate_AcceptsGoTypedSlices(t *testing.T) {
	t.Parallel()
	// The Session helper builds history in Go as []map[string]any, not the
	// []any a script produces. Asserting only []any skipped translation
	// entirely for Session traffic; the Messages API then rejected it with
	// "messages.0: use the top-level 'system' parameter".
	msgs, system := translateMessages([]map[string]any{
		{"role": "system", "content": "be terse"},
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": "hello"},
	})
	require.Equal(t, "be terse", system, "system hoisted from a Go-typed slice")
	require.Len(t, msgs, 2)
	require.Equal(t, "user", asMap(t, msgs[0])["role"])

	// Same normalization for tools.
	tools := translateTools([]map[string]any{
		{"type": "function", "function": map[string]any{"name": "f"}},
	})
	list, ok := tools.([]any)
	require.True(t, ok)
	require.Equal(t, "f", asMap(t, list[0])["name"])
	require.NotContains(t, asMap(t, list[0]), "function")

	// And for a Go-typed tool_calls array inside an assistant message.
	msgs, _ = translateMessages([]map[string]any{
		{"role": "assistant", "tool_calls": []map[string]any{
			{"id": "c1", "type": "function", "function": map[string]any{
				"name": "f", "arguments": `{"k":"v"}`,
			}},
		}},
	})
	require.Len(t, msgs, 1)
	blocks := asList(t, asMap(t, msgs[0])["content"])
	require.Len(t, blocks, 1)
	require.Equal(t, "tool_use", asMap(t, blocks[0])["type"])
	require.Equal(t, map[string]any{"k": "v"}, asMap(t, blocks[0])["input"])
}

func TestAnthropicBody_SessionShapedHistory(t *testing.T) {
	t.Parallel()
	// End to end through anthropicBody with the exact shape Session.Send
	// passes: a Go-typed slice whose first entry is the system message.
	got := anthropicBody(map[string]any{
		"messages": []map[string]any{
			{"role": "system", "content": "sys"},
			{"role": "user", "content": "q1"},
			{"role": "assistant", "content": "a1"},
			{"role": "user", "content": "q2"},
		},
		"max_tokens": 120,
	}, "claude-haiku-4-5", false)

	require.Equal(t, "sys", got["system"], "hoisted to the top-level parameter")
	list, ok := got["messages"].([]any)
	require.True(t, ok, "messages normalized to []any for the encoder")
	require.Len(t, list, 3, "system removed from the array")
	for _, m := range list {
		require.NotEqual(t, "system", asMap(t, m)["role"],
			"no system role may remain in the array")
	}
}

// asMap and asList assert the shape of a translated element so a mismatch
// fails the test instead of panicking inside it.
func asMap(t *testing.T, v any) map[string]any {
	t.Helper()
	m, ok := v.(map[string]any)
	require.True(t, ok, "want map[string]any, got %T", v)
	return m
}

func asList(t *testing.T, v any) []any {
	t.Helper()
	l, ok := v.([]any)
	require.True(t, ok, "want []any, got %T", v)
	return l
}

func TestAnthropicBody_ResponseFormatBecomesOutputConfig(t *testing.T) {
	t.Parallel()
	schema := map[string]any{"type": "object", "properties": map[string]any{"ok": map[string]any{"type": "boolean"}}}
	out := anthropicBody(map[string]any{
		"messages":        []any{map[string]any{"role": "user", "content": "hi"}},
		"response_format": map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "r", "schema": schema}},
	}, "m", false)
	_, leaked := out["response_format"]
	require.False(t, leaked, "Anthropic rejects unknown top-level fields")
	cfg := asMap(t, out["output_config"])
	format := asMap(t, cfg["format"])
	require.Equal(t, "json_schema", format["type"])
	require.Equal(t, schema, format["schema"])

	// json_object has no schema to carry and is dropped rather than sent.
	out = anthropicBody(map[string]any{
		"messages":        []any{map[string]any{"role": "user", "content": "hi"}},
		"response_format": map[string]any{"type": "json_object"},
	}, "m", false)
	_, has := out["output_config"]
	require.False(t, has)
}

// Prompt caching needs the system prompt as content blocks with a
// cache_control marker. Verified against the live API on 2026-09-16: passing
// that message through as role system returns 400 "use the top-level
// 'system' parameter".
func TestTranslateMessages_SystemBlocksHoisted(t *testing.T) {
	t.Parallel()
	block := map[string]any{"type": "text", "text": "rules", "cache_control": map[string]any{"type": "ephemeral"}}
	msgs, system := translateMessages([]any{
		map[string]any{"role": "system", "content": "terse"},
		map[string]any{"role": "system", "content": []any{block}},
		map[string]any{"role": "user", "content": "hi"},
	})
	require.Len(t, msgs, 1, "no system role message survives")
	blocks, ok := system.([]any)
	require.True(t, ok, "any block content turns the whole system prompt into blocks")
	require.Len(t, blocks, 2)
	require.Equal(t, map[string]any{"type": "text", "text": "terse"}, blocks[0])
	require.Equal(t, block, blocks[1], "the cache_control marker survives")

	_, none := translateMessages([]any{map[string]any{"role": "user", "content": "hi"}})
	require.Nil(t, none, "no system content means no system field")
}
