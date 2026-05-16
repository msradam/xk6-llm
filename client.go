// Package llm registers `k6/x/llm`, a k6 extension for LLM-aware load
// testing. See the project README for the metric set and per-request
// semantics.
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
	"sort"
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
	abort      abortPolicy
}

// abortPolicy bounds how long / how many tokens we consume before cancelling
// the upstream stream. Either limit can be active; the first to trip wins.
type abortPolicy struct {
	MaxDuration time.Duration // 0 = disabled. Wall clock from request start.
	MaxTokens   int           // 0 = disabled. Counted in content-bearing chunks.
}

// Control-key set: fields recognized by xk6-llm and stripped before the OpenAI POST.
var controlKeys = map[string]bool{
	"slo":                true,
	"cache_state":        true,
	"tags":               true,
	"abort_after_ms":     true,
	"abort_after_tokens": true,
}

// ToolCall is one assembled tool invocation from a streamed assistant turn.
// Arguments is the raw JSON string the model produced; callers parse it.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
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
	ToolCalls        []ToolCall
	Aborted          bool          // true when the stream was cut short by abort_after_ms or abort_after_tokens
	SLO              *SLOPredicate // copied from request; used by emit() to decide which Rates to push
}

// TPOTDerivable reports whether TPOT can be computed.
//
// TPOT (matches AIPerf/genai-perf/MLPerf "TPOT", which is what those tools confusingly
// call ITL): (e2el - ttft) / (output_tokens - 1). Requires output_tokens > 1.
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
	out["aborted"] = r.Aborted
	if len(r.ToolCalls) > 0 {
		tcs := make([]map[string]any, len(r.ToolCalls))
		for i, tc := range r.ToolCalls {
			tcs[i] = map[string]any{
				"id":        tc.ID,
				"name":      tc.Name,
				"arguments": tc.Arguments,
			}
		}
		out["tool_calls"] = tcs
	} else {
		out["tool_calls"] = []map[string]any{}
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
	if v, ok := raw["abort_after_ms"]; ok && v != nil {
		f, ok := asFloat(v)
		if !ok || f < 0 {
			return nil, fmt.Errorf("llm: abort_after_ms must be a non-negative number, got %v", v)
		}
		req.abort.MaxDuration = time.Duration(f * float64(time.Millisecond))
	}
	if v, ok := raw["abort_after_tokens"]; ok && v != nil {
		f, ok := asFloat(v)
		if !ok || f < 0 {
			return nil, fmt.Errorf("llm: abort_after_tokens must be a non-negative integer, got %v", v)
		}
		req.abort.MaxTokens = int(f)
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
type sseToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type sseToolCallDelta struct {
	Index    int                 `json:"index"`
	ID       string              `json:"id,omitempty"`
	Type     string              `json:"type,omitempty"`
	Function sseToolCallFunction `json:"function"`
}

type sseChoiceDelta struct {
	Role      string             `json:"role,omitempty"`
	Content   string             `json:"content,omitempty"`
	ToolCalls []sseToolCallDelta `json:"tool_calls,omitempty"`
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

	// Wire abort_after_ms into the request context so closing the body
	// propagates a TCP close back to the server and stops generation.
	reqCtx := ctx
	if req.abort.MaxDuration > 0 {
		var cancel context.CancelFunc
		reqCtx, cancel = context.WithTimeout(ctx, req.abort.MaxDuration)
		defer cancel()
		// Rebuild the request with the deadline-bound context.
		httpReq = httpReq.WithContext(reqCtx)
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

	res, err := parseStream(reqCtx, resp.Body, start, req.abort)
	if err != nil {
		return nil, newChatError(errKindStream, err)
	}
	res.ResponseHeaders = headersAt.Sub(start)
	return res, nil
}

// parseStream consumes an SSE chat-completion stream and applies vLLM-aligned timing.
//
//   - TTFT is timed at the first chunk with non-empty choices[0].delta.content
//     OR the first tool_call delta carrying a name/id/arguments. Role-only
//     deltas are skipped (matches vLLM endpoint_request_func.py).
//   - ITL samples are deltas between consecutive content chunks. The first ITL
//     sample is t[chunk2] - t[chunk1], NOT t[chunk1] - start. tool_call deltas
//     do not contribute to ITL because their per-chunk granularity is arbitrary.
//   - res.Chunks counts content-bearing chunks. When res.Chunks < CompletionTokens,
//     the server is emitting multi-token chunks (TGI batched mode, spec-dec).
//   - tool_call arguments are JSON strings streamed in fragments; they are
//     concatenated per `index` and surfaced raw (not parsed) in res.ToolCalls.
//   - When abort.MaxTokens is set, the loop breaks (with res.Aborted=true)
//     after that many content chunks. When abort.MaxDuration is set, the
//     outer context cancellation makes the body Read fail; if reqCtx.Err()
//     indicates that, the partial result is returned with res.Aborted=true
//     instead of bubbling the read error.
func parseStream(reqCtx context.Context, r io.Reader, start time.Time, abort abortPolicy) (*chatResult, error) {
	res := &chatResult{}
	var buf strings.Builder

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)

	var (
		gotFirstToken bool
		lastContentT  time.Time
		toolBuf       = map[int]*ToolCall{}
		toolOrder     []int
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

		now := time.Now()
		hadContent := ch.Delta.Content != ""
		hadTool := false

		if hadContent {
			buf.WriteString(ch.Delta.Content)
			res.Chunks++
			if gotFirstToken {
				res.ITL = append(res.ITL, now.Sub(lastContentT))
			}
			lastContentT = now
		}

		if abort.MaxTokens > 0 && res.Chunks >= abort.MaxTokens {
			res.Aborted = true
			break
		}

		for _, tc := range ch.Delta.ToolCalls {
			if tc.ID == "" && tc.Function.Name == "" && tc.Function.Arguments == "" {
				continue
			}
			hadTool = true
			existing, ok := toolBuf[tc.Index]
			if !ok {
				existing = &ToolCall{}
				toolBuf[tc.Index] = existing
				toolOrder = append(toolOrder, tc.Index)
			}
			if tc.ID != "" {
				existing.ID = tc.ID
			}
			if tc.Function.Name != "" {
				existing.Name = tc.Function.Name
			}
			if tc.Function.Arguments != "" {
				existing.Arguments += tc.Function.Arguments
			}
		}

		if !gotFirstToken && (hadContent || hadTool) {
			res.TTFT = now.Sub(start)
			gotFirstToken = true
		}
	}
	if err := sc.Err(); err != nil {
		// If our deadline tripped, treat the read failure as an intentional
		// abort and return the partial result.
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
