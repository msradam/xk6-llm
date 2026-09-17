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
	"strconv"
	"strings"
	"time"

	"github.com/grafana/agento11y/go/agento11y"
	"github.com/grafana/sobek"
	"go.k6.io/k6/v2/js/common"
	"go.k6.io/k6/v2/js/promises"
)

// Client is the JS-facing OpenAI-compatible chat client.
type Client struct {
	mod  *module
	cfg  *Options
	http *http.Client
	a11y *agento11y.Client
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
	if opts.Agento11y != nil && opts.Agento11y.Protocol != "none" {
		c.a11y = newAgento11yClient(opts.Agento11y)
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
	// errKindExport marks a failure to ship a generation record. The chat call
	// itself succeeded; only telemetry was lost.
	errKindExport = "export"
	// errKindUnsupported marks a call the configured wire cannot serve at all.
	errKindUnsupported = "unsupported"
)

// Chat-completions role and item names every wire translation switches on.
const (
	roleSystem       = "system"
	roleAssistant    = "assistant"
	roleTool         = "tool"
	partText         = "text"
	fieldMessages    = "messages"
	itemFunctionCall = "function_call"
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
	// generationID is the id this call is exported under. Supplied by the
	// caller or generated per call, and always returned to the script so a
	// multi-call workload can declare its own call graph.
	generationID string
	// parentGenerationIDs records which call(s) caused this one. A synthetic
	// workload knows its own graph; a proxy observing stateless HTTP requests
	// cannot reconstruct it, which is why this has to come from the client.
	parentGenerationIDs []string
	// unary is true when the caller set stream: false. The request is sent
	// without streaming and the whole response is parsed at once.
	unary bool
}

// abortPolicy bounds how long / how many tokens we consume before cancelling
// the upstream stream. Either limit can be active; the first to trip wins.
type abortPolicy struct {
	MaxDuration time.Duration // 0 = disabled. Wall clock from request start.
	MaxTokens   int           // 0 = disabled. Counted in content-bearing chunks, reasoning included.
}

// Control-key set: fields recognized by xk6-llm and stripped before the OpenAI POST.
var controlKeys = map[string]bool{
	"slo":                   true,
	"cache_state":           true,
	"tags":                  true,
	"abort_after_ms":        true,
	"abort_after_tokens":    true,
	"generation_id":         true,
	"parent_generation_ids": true,
	"stream":                true,
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
	// CachedTokens is the cached sub-bucket of PromptTokens when the provider
	// reports it. Not additive — these tokens are already in PromptTokens.
	//
	// Recording it is what makes a repeated-prompt probe interpretable.
	// Measured against gpt-4.1-mini on 2026-09-10 with a 1337-token prompt:
	// twelve byte-identical requests produced four cache hits of 1152 tokens
	// and eight misses, while twelve requests with a unique nonce prefix
	// produced zero hits. So provider caching is best-effort and
	// non-deterministic — the same probe alternates between two different
	// prompt-processing regimes for reasons unrelated to provider health.
	//
	// No latency effect was demonstrated at that sample size (hit median
	// 760ms vs miss median 563ms across n=4 hits, against a TTFT spread of
	// 424-2761ms). The reason to record this field is therefore to explain
	// variance, not to correct a known directional bias.
	CachedTokens int
	// CacheWriteTokens is the prompt-cache write sub-bucket of PromptTokens
	// (Anthropic cache_creation_input_tokens). Not additive.
	CacheWriteTokens int
	// ThinkingTokens is the reasoning sub-bucket of CompletionTokens when the
	// provider reports it (Anthropic: output_tokens_details.thinking_tokens).
	// It is not additive — these tokens are already counted in
	// CompletionTokens.
	ThinkingTokens int
	// ThinkingChunks counts reasoning stream events. With Anthropic's default
	// thinking display of "omitted" this is ~1 regardless of how many
	// reasoning tokens were generated, because the text is withheld: the
	// tokens are billed but never stream individually.
	ThinkingChunks int
	// GenerationID is the id this call was exported under, echoed back so a
	// caller can pass it as a parent of a later call.
	GenerationID string
	// TTFText is the time to the first text delta, which on a reasoning model
	// is later than TTFT by the whole reasoning phase.
	TTFText      time.Duration
	FinishReason string
	ToolCalls    []ToolCall
	Aborted      bool // true when the stream was cut short by abort_after_ms or abort_after_tokens
	// Unary is true for a stream: false call. TTFT, ITL, TPOT and Chunks are
	// not measured for it.
	Unary bool
	SLO   *SLOPredicate // copied from request; used by emit() to decide which Rates to push
}

// sloOutcome reports the per-dimension SLO results and whether every
// applicable dimension passed.
//
// Both the k6 Rate samples and the Agent Observability goodput metadata read
// this, so the two can never disagree about whether a request met its SLO.
type sloOutcome struct {
	TTFTChecked, TTFTPass bool
	TPOTChecked, TPOTPass bool
	E2ELChecked, E2ELPass bool
	AllPass               bool
}

// TPOTDerivable reports whether TPOT can be computed.
//
// TPOT (matches AIPerf/genai-perf/MLPerf "TPOT", which is what those tools confusingly
// call ITL): (e2el - ttft) / (output_tokens - 1). Requires output_tokens > 1.
func (r *chatResult) TPOTDerivable() bool {
	tokens, from := r.streamedOutput()
	return tokens > 1 && from > 0 && r.Duration > from
}

// TPOT returns the scalar inter-token time. Caller should check TPOTDerivable first.
func (r *chatResult) TPOT() time.Duration {
	if !r.TPOTDerivable() {
		return 0
	}
	tokens, from := r.streamedOutput()
	return (r.Duration - from) / time.Duration(tokens-1)
}

func (r *chatResult) sloOutcome() sloOutcome {
	out := sloOutcome{AllPass: true}
	if r.SLO == nil || r.SLO.Empty() {
		return out
	}
	if r.SLO.TTFTMs > 0 && !r.Unary {
		out.TTFTChecked = true
		out.TTFTPass = float64(r.TTFT)/float64(time.Millisecond) <= r.SLO.TTFTMs
		out.AllPass = out.AllPass && out.TTFTPass
	}
	if r.SLO.TPOTMs > 0 && r.TPOTDerivable() {
		out.TPOTChecked = true
		out.TPOTPass = float64(r.TPOT())/float64(time.Millisecond) <= r.SLO.TPOTMs
		out.AllPass = out.AllPass && out.TPOTPass
	}
	if r.SLO.E2ELMs > 0 {
		out.E2ELChecked = true
		out.E2ELPass = float64(r.Duration)/float64(time.Millisecond) <= r.SLO.E2ELMs
		out.AllPass = out.AllPass && out.E2ELPass
	}
	return out
}

// streamedOutput returns the output tokens that actually arrived as stream
// events, and the instant they began arriving.
//
// On a reasoning model with thinking display omitted, reasoning tokens are
// billed in CompletionTokens but never stream individually — Anthropic sends
// one empty thinking_delta for the whole phase. Dividing the post-TTFT window
// by all completion tokens then yields a per-token time that is physically
// impossible (observed: 130ns). Only the text phase is measurable, so TPOT is
// computed over it.
//
// With no reasoning tokens reported this is the original definition exactly,
// so the OpenAI path is unchanged.
func (r *chatResult) streamedOutput() (int, time.Duration) {
	if r.ThinkingTokens <= 0 {
		return r.CompletionTokens, r.TTFT
	}
	// Text tokens only, timed from the text phase. When reasoning consumed
	// every output token no text streamed at all, so this is (0, 0) and TPOT
	// is correctly not derivable — observed on gpt-5-nano terminating with
	// response.incomplete at max_output_tokens.
	// Floored at zero defensively: reasoning tokens are documented as a
	// sub-bucket of the output total, so a negative count should not happen.
	text := max(r.CompletionTokens-r.ThinkingTokens, 0)
	return text, r.TTFText
}

func (r *chatResult) toJSObject() map[string]any {
	itlMs := make([]float64, len(r.ITL))
	for i, d := range r.ITL {
		itlMs[i] = float64(d) / float64(time.Millisecond)
	}
	out := map[string]any{
		"generation_id":       r.GenerationID,
		"cached_tokens":       r.CachedTokens,
		"cache_write_tokens":  r.CacheWriteTokens,
		"content":             r.Content,
		"ttft_ms":             float64(r.TTFT) / float64(time.Millisecond),
		"ttf_text_ms":         float64(r.TTFText) / float64(time.Millisecond),
		"thinking_tokens":     r.ThinkingTokens,
		"thinking_chunks":     r.ThinkingChunks,
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
	out["stream"] = !r.Unary
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

// Chat sends a chat completion request, streamed unless stream: false is set. Returns a Promise resolving to
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
//	  stream:      false, // unary call; no TTFT, ITL or TPOT
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

	if parsed.generationID == "" {
		parsed.generationID = newGenerationID()
	}

	go func() {
		start := time.Now()
		res, err := c.doChat(ctx, parsed)
		if err != nil {
			c.emitError(ctx, model, errorKind(err), parsed.tagSet())
			// A failed call is still a generation the canary made. Export it
			// with the error so a consumer sees the failure, not a gap.
			c.exportGeneration(ctx, model, parsed, &chatResult{
				Duration:     time.Since(start),
				Unary:        parsed.unary,
				GenerationID: parsed.generationID,
			}, err)
			reject(err)
			return
		}
		res.SLO = parsed.slo
		res.GenerationID = parsed.generationID
		c.emit(ctx, model, res, parsed.tagSet())
		c.exportGeneration(ctx, model, parsed, res, nil)
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
	if v, ok := raw["generation_id"].(string); ok && v != "" {
		req.generationID = v
	}
	if v, ok := raw["parent_generation_ids"]; ok && v != nil {
		ids, err := parseParentIDs(v)
		if err != nil {
			return nil, err
		}
		req.parentGenerationIDs = ids
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
	if v, ok := raw["stream"]; ok && v != nil {
		b, ok := v.(bool)
		if !ok {
			return nil, fmt.Errorf("llm: stream must be a boolean, got %T", v)
		}
		req.unary = !b
	}
	if req.unary && req.abort.MaxTokens > 0 {
		return nil, errors.New("llm: abort_after_tokens requires a streamed call; use abort_after_ms with stream: false")
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
	if p.cacheState == "" && len(p.tags) == 0 && !p.unary {
		return nil
	}
	out := make(map[string]string, len(p.tags)+2)
	maps.Copy(out, p.tags)
	if p.cacheState != "" {
		out["cache_state"] = p.cacheState
	}
	if p.unary {
		out["mode"] = "unary"
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
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
	// ReasoningContent carries reasoning text on OpenAI-compatible servers
	// that expose it: vLLM and SGLang with a reasoning parser, and the
	// DeepSeek API, all use `reasoning_content`. OpenRouter uses `reasoning`.
	// Both are accepted because both are in the wild.
	ReasoningContent string             `json:"reasoning_content,omitempty"`
	Reasoning        string             `json:"reasoning,omitempty"`
	ToolCalls        []sseToolCallDelta `json:"tool_calls,omitempty"`
}

// reasoningText returns the reasoning payload under whichever field name the
// server used.
func (d sseChoiceDelta) reasoningText() string {
	if d.ReasoningContent != "" {
		return d.ReasoningContent
	}
	return d.Reasoning
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
	// CompletionTokensDetails breaks reasoning tokens out of
	// CompletionTokens on reasoning models. A sub-bucket, never additive.
	CompletionTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details,omitempty"`
	// PromptTokensDetails breaks cached tokens out of PromptTokens.
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details,omitempty"`
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
	anthropic := c.cfg.Wire == WireAnthropic
	responses := c.cfg.Wire == WireResponses
	v4 := c.cfg.Wire == WireProviderWireV4
	unary := req.unary

	body := req.body
	switch {
	case anthropic:
		body = anthropicBody(body, c.cfg.Model, c.cfg.IgnoreEOS)
	case responses:
		body = responsesBody(body, c.cfg.Model)
	case v4:
		var err error
		if body, err = providerWireV4Body(body); err != nil {
			return nil, newChatError(errKindUnsupported, err)
		}
	default:
		body["model"] = c.cfg.Model
		if !unary {
			body["stream"] = true
			body["stream_options"] = map[string]any{"include_usage": true}
		}
		if c.cfg.IgnoreEOS {
			body["ignore_eos"] = true
		}
	}
	if unary && (anthropic || responses) {
		body["stream"] = false
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, newChatError(errKindDecode, fmt.Errorf("marshal request: %w", err))
	}

	path := "/chat/completions"
	switch {
	case anthropic:
		path = "/messages"
	case responses:
		path = "/responses"
	case v4:
		path = "/language-model"
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.cfg.BaseURL+path, bytes.NewReader(payload))
	if err != nil {
		return nil, newChatError(errKindNetwork, fmt.Errorf("build request: %w", err))
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if unary {
		httpReq.Header.Set("Accept", "application/json")
	} else {
		httpReq.Header.Set("Accept", "text/event-stream")
	}
	if v4 {
		httpReq.Header.Set("Ai-Language-Model-Id", c.cfg.Model)
		httpReq.Header.Set("Ai-Language-Model-Specification-Version", "4")
		httpReq.Header.Set("Ai-Language-Model-Streaming", strconv.FormatBool(!unary))
	}
	if anthropic {
		httpReq.Header.Set("Anthropic-Version", anthropicVersion)
		if c.cfg.APIKey != "" {
			httpReq.Header.Set("X-Api-Key", c.cfg.APIKey)
		}
	} else if c.cfg.APIKey != "" {
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
	resp, err := c.send(httpReq)
	if err != nil {
		return nil, err
	}
	headersAt := time.Now()
	defer func() { _ = resp.Body.Close() }()

	if unary {
		raw, err := readUnaryBody(resp.Body)
		if err != nil {
			if req.abort.MaxDuration > 0 && reqCtx.Err() != nil {
				return &chatResult{Unary: true, Aborted: true, Duration: time.Since(start), ResponseHeaders: headersAt.Sub(start)}, nil
			}
			return nil, newChatError(classifyTransportError(err), fmt.Errorf("read response: %w", err))
		}
		res, err := unaryParserFor(c.cfg.Wire)(raw)
		if err != nil {
			return nil, newChatError(errKindDecode, err)
		}
		res.Duration = time.Since(start)
		res.ResponseHeaders = headersAt.Sub(start)
		return res, nil
	}

	parse := parseStream
	switch {
	case anthropic:
		parse = parseAnthropicStream
	case responses:
		parse = parseResponsesStream
	case v4:
		parse = parseProviderWireV4Stream
	}
	res, err := parse(reqCtx, resp.Body, start, req.abort)
	if err != nil {
		return nil, newChatError(errKindStream, err)
	}
	res.ResponseHeaders = headersAt.Sub(start)
	return res, nil
}

// sseScanner returns a line scanner sized for SSE payloads: a 64K initial
// buffer with a 1M ceiling, which covers the largest single chunk any of the
// wires has been observed to send.
func sseScanner(r io.Reader) *bufio.Scanner {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	return sc
}

// finishScan decides what a scanner error means at the end of a stream. When
// the abort deadline tripped, the read failure is the abort itself and the
// partial result stands; anything else is a real read error.
func finishScan(reqCtx context.Context, sc *bufio.Scanner, abort abortPolicy, res *chatResult) error {
	err := sc.Err()
	if err != nil && abort.MaxDuration > 0 && reqCtx != nil && reqCtx.Err() != nil {
		res.Aborted = true
		return nil //nolint:nilerr // the read error is the abort the caller asked for
	}
	if err != nil {
		return fmt.Errorf("read stream: %w", err)
	}
	return nil
}

// orderedToolCalls assembles streamed tool-call fragments in index order.
func orderedToolCalls(toolBuf map[int]*ToolCall, toolOrder []int) []ToolCall {
	if len(toolOrder) == 0 {
		return nil
	}
	sort.Ints(toolOrder)
	out := make([]ToolCall, 0, len(toolOrder))
	for _, idx := range toolOrder {
		out = append(out, *toolBuf[idx])
	}
	return out
}

// send performs the request and classifies the failure modes every wire
// shares: transport errors by kind, and a 4xx or 5xx status by class with the
// body as the message.
func (c *Client) send(httpReq *http.Request) (*http.Response, error) {
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, newChatError(classifyTransportError(err), err)
	}
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		kind := errKindHTTP4xx
		if resp.StatusCode >= 500 {
			kind = errKindHTTP5xx
		}
		return nil, newChatError(kind, fmt.Errorf("http %d: %s", resp.StatusCode, bytes.TrimSpace(raw)))
	}
	return resp, nil
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

	sc := sseScanner(r)

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
			if d := chunk.Usage.CompletionTokensDetails; d != nil {
				res.ThinkingTokens = d.ReasoningTokens
			}
			if d := chunk.Usage.PromptTokensDetails; d != nil {
				res.CachedTokens = d.CachedTokens
			}
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

		// Reasoning deltas are generated output: they establish TTFT and are
		// counted separately, but they do not contribute ITL samples or text
		// Chunks. Reasoning tokens are billed inside CompletionTokens while
		// arriving under a different field, so a parser that ignores them
		// times TTFT at the first *text* delta — after the entire reasoning
		// phase — and computes TPOT over tokens that never streamed as text.
		if reasoning := ch.Delta.reasoningText(); reasoning != "" {
			hadContent = true
			res.ThinkingChunks++
		}

		if ch.Delta.Content != "" {
			if res.TTFText == 0 {
				res.TTFText = now.Sub(start)
			}
			buf.WriteString(ch.Delta.Content)
			res.Chunks++
			// Gate on the previous *content* timestamp, not on gotFirstToken.
			// A tool_call delta sets gotFirstToken without ever setting
			// lastContentT, so subtracting the zero time here would emit a
			// MaxInt64 ITL sample on any stream whose first delta is a tool
			// call.
			if !lastContentT.IsZero() {
				res.ITL = append(res.ITL, now.Sub(lastContentT))
			}
			lastContentT = now
		}

		// Count reasoning chunks toward the budget as well. A runaway
		// reasoning model may never emit a text delta, so a budget that only
		// counts text would never fire on exactly the stream you most want to
		// cut off.
		if abort.MaxTokens > 0 && res.Chunks+res.ThinkingChunks >= abort.MaxTokens {
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
	if err := finishScan(reqCtx, sc, abort, res); err != nil {
		return nil, err
	}
	res.ToolCalls = orderedToolCalls(toolBuf, toolOrder)

	res.Duration = time.Since(start)
	res.Content = buf.String()
	return res, nil
}
