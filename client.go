// Package llm registers `k6/x/llm`, a k6 extension for LLM-aware load
// testing. See the project README for the metric set and per-request
// semantics; see RESEARCH.md for the upstream definitions each metric
// matches.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"net/url"
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

// Categorized error kinds. Surfaced as the `error_type` tag on llm_errors.
const (
	errKindNetwork = "network"
	errKindTimeout = "timeout"
	errKindHTTP4xx = "http_4xx"
	errKindHTTP5xx = "http_5xx"
	errKindStream  = "stream"
	errKindDecode  = "decode"
)

type chatError struct {
	Kind string
	Err  error
}

func (e *chatError) Error() string { return e.Err.Error() }
func (e *chatError) Unwrap() error { return e.Err }

func newChatError(kind string, err error) error {
	return &chatError{Kind: kind, Err: err}
}

func errorKind(err error) string {
	var ce *chatError
	if errors.As(err, &ce) {
		return ce.Kind
	}
	return errKindNetwork
}

// classifyTransportError maps a Go HTTP-transport error to a kind.
func classifyTransportError(err error) string {
	if err == nil {
		return ""
	}
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return errKindTimeout
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return errKindTimeout
	}
	var ue *url.Error
	if errors.As(err, &ue) {
		if ue.Timeout() {
			return errKindTimeout
		}
	}
	return errKindNetwork
}

type chatRequest struct {
	body       map[string]any // serialized to the OpenAI request
	slo        *SLOPredicate
	cacheState string            // "cold"|"warm"|""
	tags       map[string]string // user-supplied request-scoped tags
}

// Control-key set: fields recognized by xk6-llm and stripped before the OpenAI POST.
var controlKeys = map[string]bool{
	"slo":         true,
	"cache_state": true,
	"tags":        true,
}

type chatResult struct {
	Content          string
	TTFT             time.Duration
	ITL              []time.Duration
	Duration         time.Duration
	ResponseHeaders  time.Duration
	Chunks           int
	PromptTokens     int
	CompletionTokens int
	FinishReason     string
	SLO              *SLOPredicate // copied from request; used by emit() to decide which Rates to push
}

// TPOTDerivable reports whether TPOT can be computed.
//
// TPOT (matches AIPerf/genai-perf/MLPerf "TPOT", which is what those tools confusingly
// call ITL): (e2el - ttft) / (output_tokens - 1). Requires output_tokens > 1.
// See RESEARCH.md §A.11.
func (r *chatResult) TPOTDerivable() bool {
	return r.CompletionTokens > 1 && r.TTFT > 0 && r.Duration > r.TTFT
}

// TPOT returns the scalar inter-token time. Caller should check TPOTDerivable first.
func (r *chatResult) TPOT() time.Duration {
	if !r.TPOTDerivable() {
		return 0
	}
	return (r.Duration - r.TTFT) / time.Duration(r.CompletionTokens-1)
}

func (r *chatResult) toJSObject() map[string]any {
	itlMs := make([]float64, len(r.ITL))
	for i, d := range r.ITL {
		itlMs[i] = float64(d) / float64(time.Millisecond)
	}
	out := map[string]any{
		"content":             r.Content,
		"ttft_ms":             float64(r.TTFT) / float64(time.Millisecond),
		"itl_ms":              itlMs,
		"duration_ms":         float64(r.Duration) / float64(time.Millisecond),
		"response_headers_ms": float64(r.ResponseHeaders) / float64(time.Millisecond),
		"chunks":              r.Chunks,
		"prompt_tokens":       r.PromptTokens,
		"completion_tokens":   r.CompletionTokens,
		"finish_reason":       r.FinishReason,
	}
	if r.TPOTDerivable() {
		out["tpot_ms"] = float64(r.TPOT()) / float64(time.Millisecond)
	} else {
		out["tpot_ms"] = 0.0
	}
	return out
}

// Chat sends a streaming chat completion request. Returns a Promise resolving to
// a result object (see chatResult.toJSObject for the full shape) or rejecting
// with the categorized error.
//
// JS:
//
//	client.chat({
//	  messages: [...],
//	  max_tokens: 256,
//	  // control fields (peeled off before the upstream POST):
//	  slo:         { ttft_ms: 500, tpot_ms: 50, e2el_ms: 5000 },
//	  cache_state: "cold",
//	  tags:        { region: "us-east", shape: "short" },
//	})
func (c *Client) Chat(req map[string]any) *sobek.Promise {
	promise, resolve, reject := promises.New(c.mod.vu)
	ctx := c.mod.vu.Context()
	model := c.cfg.Model

	parsed, err := c.parseChatRequest(req)
	if err != nil {
		// Emit an error sample so synchronous validation failures show in the run summary.
		c.emitError(ctx, model, errKindDecode, nil)
		reject(err)
		return promise
	}

	go func() {
		res, err := c.doChat(ctx, parsed)
		if err != nil {
			c.emitError(ctx, model, errorKind(err), parsed.tagSet())
			reject(err)
			return
		}
		res.SLO = parsed.slo
		c.emit(ctx, model, res, parsed.tagSet())
		resolve(res.toJSObject())
	}()
	return promise
}

func (c *Client) parseChatRequest(raw map[string]any) (*chatRequest, error) {
	req := &chatRequest{
		body: make(map[string]any, len(raw)),
		slo:  c.cfg.DefaultSLO,
	}
	for k, v := range raw {
		if controlKeys[k] {
			continue
		}
		req.body[k] = v
	}
	if v, ok := raw["slo"]; ok && v != nil {
		slo, err := parseSLO(v)
		if err != nil {
			return nil, err
		}
		req.slo = slo
	}
	if v, ok := raw["cache_state"].(string); ok {
		switch v {
		case "cold", "warm", "":
			req.cacheState = v
		default:
			return nil, fmt.Errorf("llm: cache_state must be 'cold' or 'warm', got %q", v)
		}
	}
	if v, ok := raw["tags"]; ok && v != nil {
		m, ok := v.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("llm: tags must be an object, got %T", v)
		}
		req.tags = make(map[string]string, len(m))
		for tk, tv := range m {
			s, ok := tv.(string)
			if !ok {
				return nil, fmt.Errorf("llm: tags[%q] must be a string, got %T", tk, tv)
			}
			req.tags[tk] = s
		}
	}
	return req, nil
}

// tagSet returns request-scoped tags (cache_state + user tags) merged.
func (p *chatRequest) tagSet() map[string]string {
	if p.cacheState == "" && len(p.tags) == 0 {
		return nil
	}
	out := make(map[string]string, len(p.tags)+1)
	maps.Copy(out, p.tags)
	if p.cacheState != "" {
		out["cache_state"] = p.cacheState
	}
	return out
}

// openAI streaming wire types. Only the fields we read.
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

func (c *Client) doChat(ctx context.Context, req *chatRequest) (*chatResult, error) {
	body := req.body
	body["model"] = c.cfg.Model
	body["stream"] = true
	body["stream_options"] = map[string]any{"include_usage": true}
	if c.cfg.IgnoreEOS {
		body["ignore_eos"] = true
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, newChatError(errKindDecode, fmt.Errorf("marshal request: %w", err))
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.cfg.BaseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return nil, newChatError(errKindNetwork, fmt.Errorf("build request: %w", err))
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if c.cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}
	for k, v := range c.cfg.Headers {
		httpReq.Header.Set(k, v)
	}

	start := time.Now()
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, newChatError(classifyTransportError(err), err)
	}
	headersAt := time.Now()
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		kind := errKindHTTP4xx
		if resp.StatusCode >= 500 {
			kind = errKindHTTP5xx
		}
		return nil, newChatError(kind, fmt.Errorf("http %d: %s", resp.StatusCode, bytes.TrimSpace(raw)))
	}

	res, err := parseStream(resp.Body, start)
	if err != nil {
		return nil, newChatError(errKindStream, err)
	}
	res.ResponseHeaders = headersAt.Sub(start)
	return res, nil
}

// parseStream consumes an SSE chat-completion stream and applies vLLM-aligned timing.
//
//   - TTFT is timed at the first chunk with non-empty choices[0].delta.content.
//     Role-only deltas are skipped (matches vLLM endpoint_request_func.py).
//   - ITL samples are deltas between consecutive content chunks. The first ITL
//     sample is t[chunk2] - t[chunk1], NOT t[chunk1] - start.
//   - res.Chunks counts content-bearing chunks. When res.Chunks < CompletionTokens,
//     the server is emitting multi-token chunks (TGI batched mode, spec-dec).
//     See RESEARCH.md §A.11.
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
		if strings.Contains(data, `"error"`) {
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
			// Role-only or empty delta; does not count toward TTFT, ITL, or Chunks.
			continue
		}

		now := time.Now()
		buf.WriteString(ch.Delta.Content)
		res.Chunks++

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
