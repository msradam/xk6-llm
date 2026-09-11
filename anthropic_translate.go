package llm

import (
	"encoding/json"
	"strings"
)

// This file translates an OpenAI chat-completions request body into Anthropic
// Messages shape.
//
// Anthropic rejects unknown top-level fields outright ("Extra inputs are not
// permitted"), and its tool and tool-result shapes differ from OpenAI's, so a
// pass-through body fails for anything beyond plain text. Verified against the
// live API on 2026-09-10: `stop`, `max_completion_tokens`, `user`, and
// OpenAI-shaped `tools` each return 400.

// asAnySlice normalizes a message or tool list to []any.
//
// A list supplied from a k6 script exports as []any, but the Session helper
// builds its history in Go as []map[string]any. Asserting only []any silently
// skipped translation for Session traffic, which the Messages API then
// rejected with "messages.0: use the top-level 'system' parameter for the
// initial system prompt".
func asAnySlice(raw any) ([]any, bool) {
	switch v := raw.(type) {
	case []any:
		return v, true
	case []map[string]any:
		out := make([]any, len(v))
		for i, m := range v {
			out[i] = m
		}
		return out, true
	}
	return nil, false
}

// openAIOnlyFields have no Messages equivalent and are dropped. Anthropic
// rejects unknown fields, so anything left here becomes a 400.
var openAIOnlyFields = map[string]bool{
	"audio":               true,
	"frequency_penalty":   true,
	"ignore_eos":          true,
	"logit_bias":          true,
	"logprobs":            true,
	"modalities":          true,
	"n":                   true,
	"parallel_tool_calls": true,
	"prediction":          true,
	"presence_penalty":    true,
	"reasoning_effort":    true,
	"response_format":     true,
	"seed":                true,
	"service_tier":        true,
	"store":               true,
	"stream_options":      true,
	"top_logprobs":        true,
	"user":                true,
}

// openAIRenamedFields map an OpenAI field to its Messages equivalent.
var openAIRenamedFields = map[string]string{
	"stop":                  "stop_sequences",
	"max_completion_tokens": "max_tokens",
}

// translateTools converts OpenAI function tools to Messages tools:
//
//	{"type":"function","function":{"name","description","parameters"}}
//	  -> {"name","description","input_schema"}
//
// A tool already in Messages shape (no "function" wrapper) passes through, so
// a script can hand-write Anthropic tools if it prefers.
func translateTools(raw any) any {
	list, ok := asAnySlice(raw)
	if !ok {
		return raw
	}
	out := make([]any, 0, len(list))
	for _, item := range list {
		tool, ok := item.(map[string]any)
		if !ok {
			out = append(out, item)
			continue
		}
		fn, ok := tool["function"].(map[string]any)
		if !ok {
			out = append(out, tool) // already Messages-shaped
			continue
		}
		converted := map[string]any{"name": fn["name"]}
		if d, ok := fn["description"]; ok {
			converted["description"] = d
		}
		if p, ok := fn["parameters"]; ok {
			converted["input_schema"] = p
		} else {
			// Messages requires input_schema; a no-arg tool needs an empty
			// object schema rather than a missing field.
			converted["input_schema"] = map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}
		}
		out = append(out, converted)
	}
	return out
}

// translateToolChoice converts OpenAI tool_choice to Messages shape:
//
//	"auto"     -> {"type":"auto"}
//	"required" -> {"type":"any"}
//	"none"     -> {"type":"none"}
//	{"type":"function","function":{"name":"x"}} -> {"type":"tool","name":"x"}
func translateToolChoice(raw any) any {
	switch v := raw.(type) {
	case string:
		switch v {
		case "auto":
			return map[string]any{"type": "auto"}
		case "required":
			return map[string]any{"type": "any"}
		case "none":
			return map[string]any{"type": "none"}
		}
		return raw
	case map[string]any:
		if fn, ok := v["function"].(map[string]any); ok {
			return map[string]any{"type": "tool", "name": fn["name"]}
		}
		return raw
	}
	return raw
}

// translateMessages converts an OpenAI message array to Messages shape.
//
// Three transformations, all required for a tool-calling loop:
//
//   - A system-role message becomes the top-level `system` parameter, returned
//     separately because Messages has no system role.
//   - An assistant message carrying `tool_calls` becomes content blocks of
//     type `tool_use` (with `input` parsed from the JSON argument string).
//   - A `tool`-role message becomes a `user` message containing a
//     `tool_result` block keyed by `tool_use_id`. Consecutive tool messages
//     merge into one user message, because Anthropic expects all tool results
//     for a turn in a single message — splitting them trains the model to stop
//     making parallel calls.
func translateMessages(raw any) (messages []any, system string) {
	list, ok := asAnySlice(raw)
	if !ok {
		return nil, ""
	}

	out := make([]any, 0, len(list))
	var systemParts []string

	// pendingResults accumulates consecutive tool results so they land in one
	// user message.
	var pendingResults []any
	flush := func() {
		if len(pendingResults) > 0 {
			out = append(out, map[string]any{"role": "user", "content": pendingResults})
			pendingResults = nil
		}
	}

	for _, item := range list {
		msg, ok := item.(map[string]any)
		if !ok {
			flush()
			out = append(out, item)
			continue
		}
		role, _ := msg["role"].(string)

		switch role {
		case "system":
			flush()
			if text, ok := msg["content"].(string); ok {
				systemParts = append(systemParts, text)
				continue
			}
			out = append(out, msg)

		case "tool":
			result := map[string]any{"type": "tool_result"}
			if id, ok := msg["tool_call_id"].(string); ok {
				result["tool_use_id"] = id
			}
			if content, ok := msg["content"]; ok {
				result["content"] = content
			}
			pendingResults = append(pendingResults, result)

		case "assistant":
			flush()
			calls, hasCalls := asAnySlice(msg["tool_calls"])
			if !hasCalls {
				out = append(out, msg)
				continue
			}
			blocks := make([]any, 0, len(calls)+1)
			if text, ok := msg["content"].(string); ok && text != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": text})
			}
			for _, c := range calls {
				call, ok := c.(map[string]any)
				if !ok {
					continue
				}
				fn, _ := call["function"].(map[string]any)
				block := map[string]any{"type": "tool_use"}
				if id, ok := call["id"].(string); ok {
					block["id"] = id
				}
				if fn != nil {
					block["name"] = fn["name"]
					// OpenAI streams arguments as a JSON *string*; Messages
					// expects a decoded object.
					block["input"] = decodeToolArguments(fn["arguments"])
				}
				blocks = append(blocks, block)
			}
			out = append(out, map[string]any{"role": "assistant", "content": blocks})

		default:
			flush()
			out = append(out, msg)
		}
	}
	flush()

	return out, strings.Join(systemParts, "\n\n")
}

// decodeToolArguments turns OpenAI's JSON-string arguments into an object.
// An empty or unparseable string becomes an empty object rather than an error:
// a malformed argument echoed back from a previous turn should not fail the
// whole request, and Messages requires `input` to be an object.
func decodeToolArguments(raw any) map[string]any {
	switch v := raw.(type) {
	case map[string]any:
		return v
	case string:
		if strings.TrimSpace(v) == "" {
			return map[string]any{}
		}
		var out map[string]any
		if err := json.Unmarshal([]byte(v), &out); err != nil || out == nil {
			return map[string]any{}
		}
		return out
	}
	return map[string]any{}
}
