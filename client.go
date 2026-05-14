package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/grafana/sobek"
	"go.k6.io/k6/v2/js/common"
	"go.k6.io/k6/v2/js/promises"
)

// Client is the JS-facing OpenAI-compatible chat client.
type Client struct {
	mod  *module
	cfg  *Options
	http *http.Client
}

func (m *module) newClient(call sobek.ConstructorCall) *sobek.Object {
	rt := m.vu.Runtime()
	var raw any
	if len(call.Arguments) > 0 {
		raw = call.Arguments[0].Export()
	}
	opts, err := parseOptions(raw)
	if err != nil {
		common.Throw(rt, err)
	}
	c := &Client{
		mod:  m,
		cfg:  opts,
		http: &http.Client{Timeout: opts.Timeout},
	}
	return rt.ToValue(c).ToObject(rt)
}

// chatResult is the internal carrier for measurements; Chat resolves a JS-friendly map.
type chatResult struct {
	Content          string
	TTFT             time.Duration
	ITL              []time.Duration
	Duration         time.Duration
	PromptTokens     int
	CompletionTokens int
	FinishReason     string
}

func (r *chatResult) toJSObject() map[string]any {
	itlMs := make([]float64, len(r.ITL))
	for i, d := range r.ITL {
		itlMs[i] = float64(d) / float64(time.Millisecond)
	}
	return map[string]any{
		"content":           r.Content,
		"ttft_ms":           float64(r.TTFT) / float64(time.Millisecond),
		"itl_ms":            itlMs,
		"duration_ms":       float64(r.Duration) / float64(time.Millisecond),
		"prompt_tokens":     r.PromptTokens,
		"completion_tokens": r.CompletionTokens,
		"finish_reason":     r.FinishReason,
	}
}

// Chat sends a streaming chat completion request. Returns a Promise resolving to chatResult.
//
// JS: client.chat({ messages: [...], max_tokens: 256, temperature: 0 })
func (c *Client) Chat(req map[string]any) *sobek.Promise {
	promise, resolve, reject := promises.New(c.mod.vu)
	ctx := c.mod.vu.Context()
	model := c.cfg.Model
	go func() {
		res, err := c.doChat(ctx, req)
		if err != nil {
			c.emitError(model)
			reject(err)
			return
		}
		c.emit(model, res)
		resolve(res.toJSObject())
	}()
	return promise
}

// openAI streaming wire types — only the fields we read.
type sseChoiceDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type sseChoice struct {
	Index        int            `json:"index"`
	Delta        sseChoiceDelta `json:"delta"`
	FinishReason *string        `json:"finish_reason"`
}

type sseUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type sseChunk struct {
	Choices []sseChoice `json:"choices"`
	Usage   *sseUsage   `json:"usage,omitempty"`
}

type sseErrEnvelope struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func (c *Client) doChat(ctx context.Context, req map[string]any) (*chatResult, error) {
	body := map[string]any{}
	for k, v := range req {
		body[k] = v
	}
	body["model"] = c.cfg.Model
	body["stream"] = true
	body["stream_options"] = map[string]any{"include_usage": true}
	if c.cfg.IgnoreEOS {
		body["ignore_eos"] = true
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.cfg.BaseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if c.cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}

	start := time.Now()
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("http %d: %s", resp.StatusCode, bytes.TrimSpace(raw))
	}

	return parseStream(resp.Body, start)
}

// parseStream consumes an SSE chat-completion stream and applies vLLM-aligned timing:
//   - TTFT is timed at the first chunk with non-empty choices[0].delta.content.
//     Role-only deltas are skipped (matches vLLM endpoint_request_func.py).
//   - ITL samples are deltas between *consecutive content chunks*. The first ITL
//     sample is t[chunk2] - t[chunk1], NOT t[chunk1] - start. vLLM does not include
//     the start→first-content gap as an ITL sample.
func parseStream(r io.Reader, start time.Time) (*chatResult, error) {
	res := &chatResult{}
	var buf strings.Builder

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)

	var (
		gotFirstContent bool
		lastContentT    time.Time
	)

	for sc.Scan() {
		line := sc.Text()
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}

		// Surface OpenAI-shaped {"error": ...} envelopes regardless of HTTP status.
		if bytes.Contains([]byte(data), []byte(`"error"`)) {
			var env sseErrEnvelope
			if err := json.Unmarshal([]byte(data), &env); err == nil && env.Error.Message != "" {
				return nil, fmt.Errorf("stream error: %s", env.Error.Message)
			}
		}

		var chunk sseChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, fmt.Errorf("decode chunk: %w (data=%q)", err, data)
		}

		if chunk.Usage != nil {
			res.PromptTokens = chunk.Usage.PromptTokens
			res.CompletionTokens = chunk.Usage.CompletionTokens
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		ch := chunk.Choices[0]
		if ch.FinishReason != nil && *ch.FinishReason != "" {
			res.FinishReason = *ch.FinishReason
		}
		if ch.Delta.Content == "" {
			// Role-only or empty delta — does not count toward TTFT/ITL.
			continue
		}

		now := time.Now()
		buf.WriteString(ch.Delta.Content)

		if !gotFirstContent {
			res.TTFT = now.Sub(start)
			gotFirstContent = true
			lastContentT = now
			continue
		}
		res.ITL = append(res.ITL, now.Sub(lastContentT))
		lastContentT = now
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read stream: %w", err)
	}

	res.Duration = time.Since(start)
	res.Content = buf.String()
	return res, nil
}
