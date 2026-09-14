package llm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// WireProviderWireV4 is the encoding AI SDK gateways speak: POST {base_url}/language-model
// with AI SDK LanguageModelV4 call options as the body, the model ID and
// streaming mode in headers, and SSE stream parts back with no [DONE] sentinel.
const WireProviderWireV4 Wire = "providerwire-v4"

// v4CallOptions maps OpenAI chat-completion request fields to V4 call options.
var v4CallOptions = map[string]string{
	"max_tokens":        "maxOutputTokens",
	"temperature":       "temperature",
	"top_p":             "topP",
	"top_k":             "topK",
	"seed":              "seed",
	"stop":              "stopSequences",
	"presence_penalty":  "presencePenalty",
	"frequency_penalty": "frequencyPenalty",
}

// providerWireV4Body converts an OpenAI-shaped chat request body to V4 call options.
// Fields with no V4 equivalent are rejected so a load test never silently drops them.
func providerWireV4Body(body map[string]any) (map[string]any, error) {
	out := make(map[string]any, len(body))
	for key, value := range body {
		if key == "messages" {
			prompt, err := providerWireV4Prompt(value)
			if err != nil {
				return nil, err
			}
			out["prompt"] = prompt
			continue
		}
		name, ok := v4CallOptions[key]
		if !ok {
			return nil, fmt.Errorf("llm: %q is not supported with wire %q", key, WireProviderWireV4)
		}
		if s, isString := value.(string); isString && key == "stop" {
			value = []any{s}
		}
		out[name] = value
	}
	if _, ok := out["prompt"]; !ok {
		return nil, fmt.Errorf("llm: messages are required")
	}
	return out, nil
}

func providerWireV4Prompt(raw any) ([]any, error) {
	messages, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("llm: messages must be an array, got %T", raw)
	}
	prompt := make([]any, 0, len(messages))
	for i, entry := range messages {
		message, ok := entry.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("llm: messages[%d] must be an object", i)
		}
		role, _ := message["role"].(string)
		switch role {
		case "system":
			text, ok := message["content"].(string)
			if !ok {
				return nil, fmt.Errorf("llm: messages[%d]: system content must be a string", i)
			}
			prompt = append(prompt, map[string]any{"role": "system", "content": text})
		case "user", "assistant":
			parts, err := v4TextParts(message["content"])
			if err != nil {
				return nil, fmt.Errorf("llm: messages[%d]: %w", i, err)
			}
			prompt = append(prompt, map[string]any{"role": role, "content": parts})
		default:
			return nil, fmt.Errorf("llm: messages[%d]: role %q is not supported with wire %q", i, role, WireProviderWireV4)
		}
	}
	return prompt, nil
}

func v4TextParts(content any) ([]any, error) {
	switch value := content.(type) {
	case string:
		return []any{map[string]any{"type": "text", "text": value}}, nil
	case []any:
		parts := make([]any, 0, len(value))
		for _, entry := range value {
			part, ok := entry.(map[string]any)
			text, isText := part["text"].(string)
			if !ok || part["type"] != "text" || !isText {
				return nil, fmt.Errorf("only text content parts are supported")
			}
			parts = append(parts, map[string]any{"type": "text", "text": text})
		}
		return parts, nil
	default:
		return nil, fmt.Errorf("content must be a string or an array of text parts, got %T", content)
	}
}

type v4TokenCount struct {
	Total *int `json:"total"`
}

type v4StreamPart struct {
	Type  string `json:"type"`
	Delta string `json:"delta"`
	Usage struct {
		InputTokens  v4TokenCount `json:"inputTokens"`
		OutputTokens v4TokenCount `json:"outputTokens"`
	} `json:"usage"`
	FinishReason struct {
		Unified string `json:"unified"`
	} `json:"finishReason"`
	Error struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

// parseProviderWireV4Stream applies the same timing rules as parseStream to V4
// stream parts: TTFT at the first non-empty text-delta, ITL between text-deltas,
// token counts from the finish part, and an error part fails the request.
func parseProviderWireV4Stream(reqCtx context.Context, r io.Reader, start time.Time, abort abortPolicy) (*chatResult, error) {
	res := &chatResult{}
	var buf strings.Builder
	var lastContentT time.Time

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
parts:
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		var part v4StreamPart
		if err := json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data:"))), &part); err != nil {
			return nil, fmt.Errorf("decode part: %w", err)
		}
		switch part.Type {
		case "text-delta":
			if part.Delta == "" {
				continue
			}
			now := time.Now()
			if res.Chunks == 0 {
				res.TTFT = now.Sub(start)
			} else {
				res.ITL = append(res.ITL, now.Sub(lastContentT))
			}
			lastContentT = now
			buf.WriteString(part.Delta)
			res.Chunks++
			if abort.MaxTokens > 0 && res.Chunks >= abort.MaxTokens {
				res.Aborted = true
				break parts
			}
		case "finish":
			if part.Usage.InputTokens.Total != nil {
				res.PromptTokens = *part.Usage.InputTokens.Total
			}
			if part.Usage.OutputTokens.Total != nil {
				res.CompletionTokens = *part.Usage.OutputTokens.Total
			}
			res.FinishReason = part.FinishReason.Unified
		case "error":
			return nil, fmt.Errorf("stream error: %s (%s)", part.Error.Message, part.Error.Code)
		}
	}
	if err := sc.Err(); err != nil {
		if abort.MaxDuration > 0 && reqCtx != nil && reqCtx.Err() != nil {
			res.Aborted = true
		} else {
			return nil, fmt.Errorf("read stream: %w", err)
		}
	}
	res.Duration = time.Since(start)
	res.Content = buf.String()
	return res, nil
}
