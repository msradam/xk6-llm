package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// Wire selects the request/response encoding a Client speaks.
type Wire string

const (
	// WireOpenAI is the OpenAI-compatible chat-completions encoding.
	WireOpenAI Wire = "openai"
	// WireAnthropic is the Anthropic Messages encoding.
	WireAnthropic Wire = "anthropic"
)

// anthropicVersion is sent when the caller has not supplied the header itself.
// Anthropic rejects requests without it.
const anthropicVersion = "2023-06-01"

// anthropicEvent is the union of the streaming event shapes whose fields we
// read. Anthropic tags events by a `type` field rather than positionally, so
// one permissive struct decodes every event we care about.
type anthropicEvent struct {
	Type    string `json:"type"`
	Message *struct {
		StopReason string `json:"stop_reason"`
		Usage      *struct {
			InputTokens           int `json:"input_tokens"`
			OutputTokens          int `json:"output_tokens"`
			CacheReadInputTokens  int `json:"cache_read_input_tokens"`
			CacheCreationInputTok int `json:"cache_creation_input_tokens"`
		} `json:"usage"`
	} `json:"message"`
	Index        int `json:"index"`
	ContentBlock *struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
	} `json:"content_block"`
	Delta *struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		// OutputTokensDetails breaks reasoning tokens out of OutputTokens.
		// They are a sub-bucket, not an additive one.
		OutputTokensDetails *struct {
			ThinkingTokens int `json:"thinking_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// anthropicBody rewrites an OpenAI-shaped request body into Messages shape.
//
// Only the fields that change meaning are translated: a `system` role message
// becomes the top-level `system` parameter, and OpenAI-only knobs that
// Anthropic rejects are dropped. Everything else passes through so callers can
// still set provider-specific fields directly.
func anthropicBody(body map[string]any, model string, ignoreEOS bool) map[string]any {
	out := make(map[string]any, len(body)+3)
	for k, v := range body {
		if openAIOnlyFields[k] {
			continue
		}
		if renamed, ok := openAIRenamedFields[k]; ok {
			// Do not clobber an explicitly supplied Messages field.
			if _, exists := body[renamed]; !exists {
				out[renamed] = v
			}
			continue
		}
		switch k {
		case "tools":
			out[k] = translateTools(v)
		case "tool_choice":
			out[k] = translateToolChoice(v)
		case "response_format":
			if cfg, ok := translateResponseFormat(v); ok {
				out["output_config"] = cfg
			}
		default:
			out[k] = v
		}
	}
	out["model"] = model
	out["stream"] = true

	// Anthropic requires max_tokens. Mirror a sane default rather than 400ing
	// on a caller who omitted it, matching how the OpenAI path behaves.
	if _, ok := out["max_tokens"]; !ok {
		out["max_tokens"] = 512
	}

	// Translate the message array: hoist system, convert assistant tool_calls
	// to tool_use blocks, and convert tool-role messages to tool_result blocks.
	if msgs, ok := out["messages"]; ok {
		translated, system := translateMessages(msgs)
		if translated != nil {
			out["messages"] = translated
		}
		if system != nil {
			if _, exists := out["system"]; !exists {
				out["system"] = system
			}
		}
	}

	_ = ignoreEOS // Anthropic has no ignore_eos equivalent.
	return out
}

// parseAnthropicStream consumes an Anthropic Messages SSE stream and applies
// the same timing semantics as parseStream, so metrics are comparable across
// wires.
//
//   - TTFT is timed at the first content-bearing event: a text_delta with
//     non-empty text, a tool_use content_block_start, or an input_json_delta.
//     message_start, ping, and the empty text content_block_start are skipped
//     — these are the Anthropic analogue of OpenAI's role-only delta, and
//     counting them would report TTFT one event early.
//
//   - ITL samples are gaps between consecutive text deltas; the first sample
//     is t[delta2] - t[delta1]. Tool-argument fragments do not contribute,
//     matching the OpenAI path.
//
//   - Reasoning deltas (thinking_delta, signature_delta) establish TTFT even
//     when their payload is empty, and are excluded from ITL and from Chunks.
//
//     Measured against claude-opus-5 on 2026-09-10, reasoning output does not
//     stream during the reasoning phase under either display mode: with
//     display "omitted" one empty thinking_delta arrives at 3918ms and the
//     first text_delta at 3918ms; with "summarized", 31 thinking_deltas
//     arrive starting at 7109ms and the first text_delta at 7114ms. The
//     reasoning deltas land when reasoning finishes, not while it runs.
//
//     So on a reasoning model TTFT measures time-to-end-of-reasoning, not
//     time-to-first-token — there is no observable event in between. That is a
//     provider property, not something this parser can improve. Counting
//     reasoning deltas keeps TTFT defined as "first observable generated
//     output"; the practical win here is correct token accounting and a TPOT
//     that is computed over the tokens that actually streamed.
//
//   - Prompt tokens arrive on message_start, completion tokens on
//     message_delta. There is no [DONE] sentinel; message_stop ends the stream.
func parseAnthropicStream(reqCtx context.Context, r io.Reader, start time.Time, abort abortPolicy) (*chatResult, error) {
	res := &chatResult{}
	var buf strings.Builder

	sc := sseScanner(r)

	var (
		gotFirstToken bool
		lastContentT  time.Time
		toolBuf       = map[int]*ToolCall{}
		toolOrder     []int
		stop          bool
	)

	for sc.Scan() && !stop {
		line := sc.Text()
		// Anthropic frames carry both `event:` and `data:` lines; the type is
		// also inside the JSON, so only data lines need decoding.
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}

		var ev anthropicEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return nil, fmt.Errorf("decode event: %w (data=%q)", err, data)
		}

		if ev.Error != nil && ev.Error.Message != "" {
			return nil, fmt.Errorf("stream error: %s", ev.Error.Message)
		}

		now := time.Now()
		var hadContent, hadTool bool

		switch ev.Type {
		case "message_start":
			if ev.Message != nil && ev.Message.Usage != nil {
				// Anthropic's input_tokens excludes both cache buckets, where
				// OpenAI's prompt_tokens includes cached_tokens. Measured on
				// 2026-09-16: a 7466-token cached prefix reported input_tokens
				// of 16. Normalised to the inclusive contract so prompt
				// counts, cost and the export mean the same thing on every
				// wire; the cache fields stay sub-buckets of the total.
				u := ev.Message.Usage
				res.PromptTokens = u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTok
				res.CachedTokens = u.CacheReadInputTokens
				res.CacheWriteTokens = u.CacheCreationInputTok
				// Anthropic reports a running output count here too; keep the
				// larger value seen so an early non-zero is not lost.
				if ev.Message.Usage.OutputTokens > res.CompletionTokens {
					res.CompletionTokens = ev.Message.Usage.OutputTokens
				}
			}

		case "content_block_start":
			if ev.ContentBlock != nil && ev.ContentBlock.Type == "tool_use" {
				hadTool = true
				tc, ok := toolBuf[ev.Index]
				if !ok {
					tc = &ToolCall{}
					toolBuf[ev.Index] = tc
					toolOrder = append(toolOrder, ev.Index)
				}
				tc.ID = ev.ContentBlock.ID
				tc.Name = ev.ContentBlock.Name
			}

		case "content_block_delta":
			if ev.Delta == nil {
				break
			}
			switch ev.Delta.Type {
			case "text_delta":
				if ev.Delta.Text == "" {
					break
				}
				hadContent = true
				if res.TTFText == 0 {
					res.TTFText = now.Sub(start)
				}
				buf.WriteString(ev.Delta.Text)
				res.Chunks++
				// Gate on the previous *text* timestamp, not on gotFirstToken:
				// a reasoning or tool delta can establish TTFT without ever
				// setting lastContentT, and subtracting the zero time yields a
				// MaxInt64 duration.
				if !lastContentT.IsZero() {
					res.ITL = append(res.ITL, now.Sub(lastContentT))
				}
				lastContentT = now
			case "thinking_delta", "signature_delta":
				// Generation has started even when the payload is empty.
				// Counted for TTFT, excluded from ITL and from Chunks (which
				// documents text chunks).
				hadContent = true
				res.ThinkingChunks++
			case "input_json_delta":
				if ev.Delta.PartialJSON == "" {
					break
				}
				hadTool = true
				tc, ok := toolBuf[ev.Index]
				if !ok {
					tc = &ToolCall{}
					toolBuf[ev.Index] = tc
					toolOrder = append(toolOrder, ev.Index)
				}
				tc.Arguments += ev.Delta.PartialJSON
			}

		case "message_delta":
			if ev.Delta != nil && ev.Delta.StopReason != "" {
				res.FinishReason = ev.Delta.StopReason
			}
			if ev.Usage != nil && ev.Usage.OutputTokens > 0 {
				res.CompletionTokens = ev.Usage.OutputTokens
				if d := ev.Usage.OutputTokensDetails; d != nil {
					res.ThinkingTokens = d.ThinkingTokens
				}
			}

		case "message_stop":
			// Terminal event. No [DONE] sentinel on this wire.
			stop = true
		}

		if !gotFirstToken && (hadContent || hadTool) {
			res.TTFT = now.Sub(start)
			gotFirstToken = true
		}

		// Reasoning chunks count toward the budget too — see parseStream.
		if abort.MaxTokens > 0 && res.Chunks+res.ThinkingChunks >= abort.MaxTokens {
			res.Aborted = true
			stop = true
		}
	}

	if err := finishScan(reqCtx, sc, abort, res); err != nil {
		return nil, err
	}
	res.ToolCalls = orderedToolCalls(toolBuf, toolOrder)

	res.Duration = time.Since(start)
	res.Content = buf.String()
	return res, nil
}
