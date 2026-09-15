package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// maxUnaryResponseBytes bounds how much of a non-streamed response is read.
const maxUnaryResponseBytes = 32 << 20

// unaryParser decodes a complete non-streamed response body for one wire.
//
// A unary call has no stream, so it reports no TTFT, ITL, TPOT or chunk
// counts. Everything that is a property of the whole response still is:
// duration, response headers, token usage, finish reason, content and tool
// calls. Parsers set Unary so emit and the SLO logic can tell the difference.
type unaryParser func(raw []byte) (*chatResult, error)

func unaryParserFor(w Wire) unaryParser {
	switch w {
	case WireAnthropic:
		return parseAnthropicMessage
	case WireResponses:
		return parseResponsesObject
	case WireProviderWireV4:
		return parseProviderWireV4Unary
	case WireOpenAI:
		return parseChatCompletion
	default:
		return parseChatCompletion
	}
}

func readUnaryBody(r io.Reader) ([]byte, error) {
	raw, err := io.ReadAll(io.LimitReader(r, maxUnaryResponseBytes+1))
	if err != nil {
		return nil, err
	}
	if len(raw) > maxUnaryResponseBytes {
		return nil, fmt.Errorf("response exceeds %d bytes", maxUnaryResponseBytes)
	}
	return raw, nil
}

// textFromContent accepts a message content that is a string, null, or an
// array of typed parts, which OpenAI-compatible servers use interchangeably.
func textFromContent(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" || p.Type == "output_text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

type chatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Content   json.RawMessage `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *sseUsage `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// parseChatCompletion decodes an OpenAI-compatible chat completion.
func parseChatCompletion(raw []byte) (*chatResult, error) {
	var body chatCompletionResponse
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if body.Error != nil && body.Error.Message != "" {
		return nil, fmt.Errorf("response error: %s", body.Error.Message)
	}
	if len(body.Choices) == 0 {
		return nil, errors.New("decode response: no choices")
	}
	res := &chatResult{Unary: true}
	if u := body.Usage; u != nil {
		res.PromptTokens = u.PromptTokens
		res.CompletionTokens = u.CompletionTokens
		if d := u.CompletionTokensDetails; d != nil {
			res.ThinkingTokens = d.ReasoningTokens
		}
		if d := u.PromptTokensDetails; d != nil {
			res.CachedTokens = d.CachedTokens
		}
	}
	choice := body.Choices[0]
	res.FinishReason = choice.FinishReason
	res.Content = textFromContent(choice.Message.Content)
	for _, tc := range choice.Message.ToolCalls {
		res.ToolCalls = append(res.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
	}
	return res, nil
}

type anthropicMessageResponse struct {
	Type    string `json:"type"`
	Content []struct {
		Type  string          `json:"type"`
		Text  string          `json:"text"`
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      *struct {
		InputTokens          int `json:"input_tokens"`
		OutputTokens         int `json:"output_tokens"`
		CacheReadInputTokens int `json:"cache_read_input_tokens"`
		OutputTokensDetails  *struct {
			ThinkingTokens int `json:"thinking_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// parseAnthropicMessage decodes an Anthropic Messages response.
func parseAnthropicMessage(raw []byte) (*chatResult, error) {
	var body anthropicMessageResponse
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if body.Error != nil && body.Error.Message != "" {
		return nil, fmt.Errorf("response error: %s", body.Error.Message)
	}
	res := &chatResult{Unary: true, FinishReason: body.StopReason}
	if u := body.Usage; u != nil {
		res.PromptTokens = u.InputTokens
		res.CompletionTokens = u.OutputTokens
		res.CachedTokens = u.CacheReadInputTokens
		if d := u.OutputTokensDetails; d != nil {
			res.ThinkingTokens = d.ThinkingTokens
		}
	}
	var text strings.Builder
	for _, block := range body.Content {
		switch block.Type {
		case "text":
			text.WriteString(block.Text)
		case "tool_use":
			args := string(block.Input)
			if args == "" || args == "null" {
				args = "{}"
			}
			res.ToolCalls = append(res.ToolCalls, ToolCall{ID: block.ID, Name: block.Name, Arguments: args})
		}
	}
	res.Content = text.String()
	return res, nil
}

type responsesObjectResponse struct {
	Status            string `json:"status"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
	Output []struct {
		Type    string `json:"type"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"output"`
	Usage *struct {
		InputTokens        int `json:"input_tokens"`
		InputTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"input_tokens_details"`
		OutputTokens        int `json:"output_tokens"`
		OutputTokensDetails *struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
}

// parseResponsesObject decodes an OpenAI Responses object.
func parseResponsesObject(raw []byte) (*chatResult, error) {
	var body responsesObjectResponse
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if body.Status == "failed" || (body.Error != nil && body.Error.Message != "") {
		msg := "response failed"
		if body.Error != nil && body.Error.Message != "" {
			msg = body.Error.Message
		}
		return nil, fmt.Errorf("response error: %s", msg)
	}
	res := &chatResult{Unary: true, FinishReason: body.Status}
	if d := body.IncompleteDetails; d != nil && d.Reason != "" {
		res.FinishReason = d.Reason
	}
	if u := body.Usage; u != nil {
		res.PromptTokens = u.InputTokens
		res.CompletionTokens = u.OutputTokens
		if d := u.InputTokensDetails; d != nil {
			res.CachedTokens = d.CachedTokens
		}
		if d := u.OutputTokensDetails; d != nil {
			res.ThinkingTokens = d.ReasoningTokens
		}
	}
	var text strings.Builder
	for _, item := range body.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" {
					text.WriteString(part.Text)
				}
			}
		case "function_call":
			res.ToolCalls = append(res.ToolCalls, ToolCall{ID: item.CallID, Name: item.Name, Arguments: item.Arguments})
		}
	}
	res.Content = text.String()
	return res, nil
}

type v4UnaryResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	FinishReason struct {
		Unified string `json:"unified"`
	} `json:"finishReason"`
	Usage struct {
		InputTokens  v4TokenCount `json:"inputTokens"`
		OutputTokens v4TokenCount `json:"outputTokens"`
	} `json:"usage"`
}

// parseProviderWireV4Unary decodes a ProviderWire V4 unary document.
func parseProviderWireV4Unary(raw []byte) (*chatResult, error) {
	var body v4UnaryResponse
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	res := &chatResult{Unary: true, FinishReason: body.FinishReason.Unified}
	if body.Usage.InputTokens.Total != nil {
		res.PromptTokens = *body.Usage.InputTokens.Total
	}
	if body.Usage.OutputTokens.Total != nil {
		res.CompletionTokens = *body.Usage.OutputTokens.Total
	}
	var text strings.Builder
	for _, part := range body.Content {
		if part.Type == "text" {
			text.WriteString(part.Text)
		}
	}
	res.Content = text.String()
	return res, nil
}
