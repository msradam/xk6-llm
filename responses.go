package llm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
)

// WireResponses is the OpenAI Responses encoding (POST /v1/responses), which
// Codex and newer OpenAI clients use.
//
// It is a different API from chat completions, not a variant: messages become
// `input`, `max_tokens` becomes `max_output_tokens`, a system message becomes
// `instructions`, tool definitions are flat rather than wrapped in a
// `function` object, and the stream is a taxonomy of named `response.*` events
// rather than choice deltas.
const WireResponses Wire = "responses"

// responsesEvent is the union of Responses stream event fields this parser
// reads. Event shapes captured from the live API on 2026-09-10 against
// gpt-4.1-mini and gpt-5-nano.
type responsesEvent struct {
	Type string `json:"type"`
	// Delta carries text on response.output_text.delta, reasoning summary text
	// on response.reasoning_summary_text.delta, and argument fragments on
	// response.function_call_arguments.delta.
	Delta string `json:"delta"`
	// Item appears on response.output_item.added/done. A function_call item
	// carries the call identity; arguments stream separately.
	Item *struct {
		ID        string `json:"id"`
		Type      string `json:"type"`
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"item"`
	OutputIndex int `json:"output_index"`
	// Arguments is the assembled argument string on
	// response.function_call_arguments.done.
	Arguments string `json:"arguments"`
	// Response appears on the terminal events: completed, incomplete, failed.
	Response *struct {
		Status            string `json:"status"`
		IncompleteDetails *struct {
			Reason string `json:"reason"`
		} `json:"incomplete_details"`
		Error *struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
		Usage *struct {
			InputTokens        int `json:"input_tokens"`
			InputTokensDetails *struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"input_tokens_details"`
			OutputTokens        int `json:"output_tokens"`
			OutputTokensDetails *struct {
				ReasoningTokens int `json:"reasoning_tokens"`
			} `json:"output_tokens_details"`
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	} `json:"response"`
	// Top-level error event.
	Message string `json:"message"`
	Code    string `json:"code"`
}

// responsesOnlyRenames map a chat-completions field to its Responses
// equivalent.
var responsesRenamedFields = map[string]string{
	"max_tokens":            "max_output_tokens",
	"max_completion_tokens": "max_output_tokens",
}

// chatOnlyFieldsForResponses have no Responses equivalent.
var chatOnlyFieldsForResponses = map[string]bool{
	"audio":             true,
	"frequency_penalty": true,
	"ignore_eos":        true,
	"logit_bias":        true,
	"logprobs":          true,
	"modalities":        true,
	"n":                 true,
	"prediction":        true,
	"presence_penalty":  true,
	"response_format":   true,
	"seed":              true,
	"stop":              true,
	"stream_options":    true,
	"top_logprobs":      true,
}

// responsesBody rewrites a chat-completions request body into Responses shape.
func responsesBody(body map[string]any, model string) map[string]any {
	out := make(map[string]any, len(body)+3)
	for k, v := range body {
		if chatOnlyFieldsForResponses[k] {
			continue
		}
		if renamed, ok := responsesRenamedFields[k]; ok {
			if _, exists := body[renamed]; !exists {
				out[renamed] = v
			}
			continue
		}
		switch k {
		case fieldMessages:
			// Handled below, together with instructions.
		case "tools":
			out[k] = responsesTools(v)
		case "tool_choice":
			out[k] = responsesToolChoice(v)
		default:
			out[k] = v
		}
	}
	out["model"] = model
	out["stream"] = true

	if msgs, ok := body["messages"]; ok {
		input, instructions := responsesInput(msgs)
		if input != nil {
			out["input"] = input
		}
		if instructions != "" {
			if _, exists := out["instructions"]; !exists {
				out["instructions"] = instructions
			}
		}
	}
	return out
}

// responsesTools flattens chat-completions function tools. Responses declares
// tools as {"type":"function","name":…,"description":…,"parameters":…} with no
// nested "function" object.
func responsesTools(raw any) any {
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
			out = append(out, tool) // already flat
			continue
		}
		flat := map[string]any{"type": "function", "name": fn["name"]}
		if d, ok := fn["description"]; ok {
			flat["description"] = d
		}
		if p, ok := fn["parameters"]; ok {
			flat["parameters"] = p
		} else {
			flat["parameters"] = map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}
		}
		out = append(out, flat)
	}
	return out
}

// responsesToolChoice converts a chat-completions tool_choice. The bare
// strings carry over unchanged; the named-function object loses its wrapper.
func responsesToolChoice(raw any) any {
	if m, ok := raw.(map[string]any); ok {
		if fn, ok := m["function"].(map[string]any); ok {
			return map[string]any{"type": "function", "name": fn["name"]}
		}
	}
	return raw
}

// responsesInput converts a chat-completions message array into a Responses
// input array, returning any system content separately as `instructions`.
//
//   - system role      -> instructions (Responses has no system input item)
//   - assistant with tool_calls -> function_call items
//   - tool role        -> function_call_output items
//   - everything else  -> passed through as a role/content item
func responsesInput(raw any) ([]any, string) {
	list, ok := asAnySlice(raw)
	if !ok {
		return nil, ""
	}
	out := make([]any, 0, len(list))
	var systemParts []string

	for _, item := range list {
		msg, ok := item.(map[string]any)
		if !ok {
			out = append(out, item)
			continue
		}
		switch role, _ := msg["role"].(string); role {
		case roleSystem:
			if text, ok := msg["content"].(string); ok {
				systemParts = append(systemParts, text)
				continue
			}
			out = append(out, msg)

		case roleTool:
			entry := map[string]any{"type": "function_call_output"}
			if id, ok := msg["tool_call_id"].(string); ok {
				entry["call_id"] = id
			}
			if content, ok := msg["content"].(string); ok {
				entry["output"] = content
			}
			out = append(out, entry)

		case roleAssistant:
			calls, hasCalls := asAnySlice(msg["tool_calls"])
			if !hasCalls {
				out = append(out, msg)
				continue
			}
			if text, ok := msg["content"].(string); ok && text != "" {
				out = append(out, map[string]any{"role": "assistant", "content": text})
			}
			for _, c := range calls {
				call, ok := c.(map[string]any)
				if !ok {
					continue
				}
				fn, _ := call["function"].(map[string]any)
				entry := map[string]any{"type": "function_call"}
				if id, ok := call["id"].(string); ok {
					entry["call_id"] = id
				}
				if fn != nil {
					entry["name"] = fn["name"]
					// Responses wants the raw JSON string here, unlike
					// Anthropic which wants a decoded object.
					if args, ok := fn["arguments"].(string); ok {
						entry["arguments"] = args
					} else {
						entry["arguments"] = "{}"
					}
				}
				out = append(out, entry)
			}

		default:
			out = append(out, msg)
		}
	}
	return out, strings.Join(systemParts, "\n\n")
}

// parseResponsesStream consumes a Responses SSE stream, applying the same
// timing semantics as the other wires so metrics stay comparable.
//
//   - TTFT is timed at the first content-bearing delta: output text, a
//     reasoning summary delta, or a function-call argument fragment. The
//     structural events (response.created, in_progress, output_item.added,
//     content_part.added) are skipped — they are this wire's analogue of the
//     role-only chat delta.
//   - Unlike Anthropic, reasoning summary text streams incrementally here (70
//     deltas observed on gpt-5-nano), so reasoning is genuinely observable on
//     this wire. It still does not contribute ITL samples, which stay a
//     property of the text phase.
//   - Terminal events are response.completed, response.incomplete, and
//     response.failed. A reasoning model that exhausts max_output_tokens
//     before emitting text terminates with response.incomplete and no text at
//     all — observed on gpt-5-nano at max_output_tokens=200.
//
// Caveat when reading ITL and TPOT on a reasoning model: they measure
// *delivery* cadence, not generation cadence. Measured on gpt-5-nano
// (2026-09-10, reasoning summary enabled, 142 output tokens of which 64 were
// reasoning): reasoning summary streamed from 1351ms, the first text delta
// arrived at 2025ms, and the remaining 58 text deltas then drained in ~28ms —
// itl_p50 of 2.4us, itl_mean 0.48ms, tpot 0.56ms. The generation had already
// happened; the text was flushed from a buffer. Compare gpt-4.1-mini on the
// same prompt: 56 deltas, tpot 15.6ms, chunks tracking token count, which is
// genuine per-token streaming. Comparing ITL across a reasoning and a
// non-reasoning model therefore compares two different things.
//
//nolint:maintidx,nestif // one event loop per stream taxonomy; splitting it would separate the timing rules from the events they apply to
func parseResponsesStream(reqCtx context.Context, r io.Reader, start time.Time, abort abortPolicy) (*chatResult, error) {
	res := &chatResult{}
	var buf strings.Builder

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)

	var (
		gotFirstToken bool
		lastContentT  time.Time
		toolBuf       = map[int]*ToolCall{}
		toolOrder     []int
		stop          bool
	)

	toolAt := func(idx int) *ToolCall {
		tc, ok := toolBuf[idx]
		if !ok {
			tc = &ToolCall{}
			toolBuf[idx] = tc
			toolOrder = append(toolOrder, idx)
		}
		return tc
	}

	for sc.Scan() && !stop {
		line := sc.Text()
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}

		var ev responsesEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return nil, fmt.Errorf("decode event: %w (data=%q)", err, data)
		}

		now := time.Now()
		var hadContent bool

		switch ev.Type {
		case "response.output_text.delta":
			if ev.Delta == "" {
				break
			}
			hadContent = true
			if res.TTFText == 0 {
				res.TTFText = now.Sub(start)
			}
			buf.WriteString(ev.Delta)
			res.Chunks++
			if !lastContentT.IsZero() {
				res.ITL = append(res.ITL, now.Sub(lastContentT))
			}
			lastContentT = now

		case "response.reasoning_summary_text.delta":
			if ev.Delta == "" {
				break
			}
			hadContent = true
			res.ThinkingChunks++

		case "response.output_item.added":
			if ev.Item != nil && ev.Item.Type == itemFunctionCall {
				hadContent = true
				tc := toolAt(ev.OutputIndex)
				tc.ID = ev.Item.CallID
				tc.Name = ev.Item.Name
			}

		case "response.function_call_arguments.delta":
			if ev.Delta == "" {
				break
			}
			hadContent = true
			toolAt(ev.OutputIndex).Arguments += ev.Delta

		case "response.function_call_arguments.done":
			// Authoritative assembled arguments; replace the accumulated
			// fragments so a dropped delta cannot corrupt the result.
			if ev.Arguments != "" {
				toolAt(ev.OutputIndex).Arguments = ev.Arguments
			}

		case "response.output_item.done":
			if ev.Item != nil && ev.Item.Type == "function_call" {
				tc := toolAt(ev.OutputIndex)
				if ev.Item.CallID != "" {
					tc.ID = ev.Item.CallID
				}
				if ev.Item.Name != "" {
					tc.Name = ev.Item.Name
				}
				if ev.Item.Arguments != "" {
					tc.Arguments = ev.Item.Arguments
				}
			}

		case "response.failed":
			msg := "response failed"
			if ev.Response != nil && ev.Response.Error != nil && ev.Response.Error.Message != "" {
				msg = ev.Response.Error.Message
			}
			return nil, fmt.Errorf("stream error: %s", msg)

		case "error":
			msg := ev.Message
			if msg == "" {
				msg = "stream error"
			}
			return nil, fmt.Errorf("stream error: %s", msg)

		case "response.completed", "response.incomplete":
			if ev.Response != nil {
				res.FinishReason = ev.Response.Status
				if d := ev.Response.IncompleteDetails; d != nil && d.Reason != "" {
					res.FinishReason = d.Reason
				}
				if u := ev.Response.Usage; u != nil {
					res.PromptTokens = u.InputTokens
					res.CompletionTokens = u.OutputTokens
					if d := u.InputTokensDetails; d != nil {
						res.CachedTokens = d.CachedTokens
					}
					if d := u.OutputTokensDetails; d != nil {
						res.ThinkingTokens = d.ReasoningTokens
					}
				}
			}
			stop = true
		}

		if !gotFirstToken && hadContent {
			res.TTFT = now.Sub(start)
			gotFirstToken = true
		}

		// Reasoning chunks count toward the abort budget — see parseStream.
		if abort.MaxTokens > 0 && res.Chunks+res.ThinkingChunks >= abort.MaxTokens {
			res.Aborted = true
			stop = true
		}
	}

	if err := sc.Err(); err != nil {
		if abort.MaxDuration > 0 && reqCtx != nil && reqCtx.Err() != nil {
			res.Aborted = true
		} else {
			return nil, fmt.Errorf("read stream: %w", err)
		}
	}

	if len(toolOrder) > 0 {
		sort.Ints(toolOrder)
		res.ToolCalls = make([]ToolCall, 0, len(toolOrder))
		for _, idx := range toolOrder {
			res.ToolCalls = append(res.ToolCalls, *toolBuf[idx])
		}
	}

	res.Duration = time.Since(start)
	res.Content = buf.String()
	return res, nil
}
