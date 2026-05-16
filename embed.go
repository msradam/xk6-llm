package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/grafana/sobek"
	"go.k6.io/k6/v2/js/promises"
)

type embedRequest struct {
	input []string
	model string
	tags  map[string]string
}

type embedResult struct {
	Model        string
	Embeddings   [][]float64
	PromptTokens int
	Duration     time.Duration
	Inputs       int
}

func (r *embedResult) toJSObject() map[string]any {
	embs := make([]any, len(r.Embeddings))
	for i, v := range r.Embeddings {
		row := make([]any, len(v))
		for j, x := range v {
			row[j] = x
		}
		embs[i] = row
	}
	return map[string]any{
		"model":         r.Model,
		"embeddings":    embs,
		"prompt_tokens": r.PromptTokens,
		"duration_ms":   float64(r.Duration) / float64(time.Millisecond),
		"inputs":        r.Inputs,
	}
}

// Embed dispatches a request to /v1/embeddings. The `input` field accepts a
// string or an array of strings; the response always exposes `embeddings` as
// number[][] in input order.
//
// JS:
//
//	client.embed({ input: "hello", model?: "nomic-embed-text", tags?: {...} })
//	client.embed({ input: ["hello", "world"] })
func (c *Client) Embed(req map[string]any) *sobek.Promise {
	promise, resolve, reject := promises.New(c.mod.vu)
	ctx := c.mod.vu.Context()

	parsed, err := c.parseEmbedRequest(req)
	if err != nil {
		c.emitEmbedError(ctx, parsed.model, errKindDecode, parsed.tags)
		reject(err)
		return promise
	}

	go func() {
		res, err := c.doEmbed(ctx, parsed)
		if err != nil {
			c.emitEmbedError(ctx, parsed.model, errorKind(err), parsed.tags)
			reject(err)
			return
		}
		c.emitEmbed(ctx, res, parsed.tags)
		resolve(res.toJSObject())
	}()
	return promise
}

func (c *Client) parseEmbedRequest(raw map[string]any) (*embedRequest, error) {
	req := &embedRequest{model: c.cfg.Model}
	if v, ok := raw["model"].(string); ok && v != "" {
		req.model = v
	}
	switch v := raw["input"].(type) {
	case string:
		if v == "" {
			return req, errors.New("llm.embed: input must not be empty")
		}
		req.input = []string{v}
	case []any:
		if len(v) == 0 {
			return req, errors.New("llm.embed: input array must not be empty")
		}
		req.input = make([]string, len(v))
		for i, x := range v {
			s, ok := x.(string)
			if !ok {
				return req, fmt.Errorf("llm.embed: input[%d] must be a string, got %T", i, x)
			}
			req.input[i] = s
		}
	case nil:
		return req, errors.New("llm.embed: input is required")
	default:
		return req, fmt.Errorf("llm.embed: input must be a string or string array, got %T", v)
	}
	if v, ok := raw["tags"]; ok && v != nil {
		m, ok := v.(map[string]any)
		if !ok {
			return req, fmt.Errorf("llm.embed: tags must be an object, got %T", v)
		}
		req.tags = make(map[string]string, len(m))
		for tk, tv := range m {
			s, ok := tv.(string)
			if !ok {
				return req, fmt.Errorf("llm.embed: tags[%q] must be a string, got %T", tk, tv)
			}
			req.tags[tk] = s
		}
	}
	return req, nil
}

type embedAPIData struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type embedAPIResponse struct {
	Model string         `json:"model"`
	Data  []embedAPIData `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

func (c *Client) doEmbed(ctx context.Context, req *embedRequest) (*embedResult, error) {
	// Accept a bare string when only one input is supplied: matches both OpenAI
	// and Ollama wire conventions and avoids unnecessary array wrapping.
	var inputField any = req.input
	if len(req.input) == 1 {
		inputField = req.input[0]
	}
	body := map[string]any{"model": req.model, "input": inputField}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, newChatError(errKindDecode, fmt.Errorf("marshal request: %w", err))
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.cfg.BaseURL+"/embeddings", bytes.NewReader(payload))
	if err != nil {
		return nil, newChatError(errKindNetwork, fmt.Errorf("build request: %w", err))
	}
	httpReq.Header.Set("Content-Type", "application/json")
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
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		kind := errKindHTTP4xx
		if resp.StatusCode >= 500 {
			kind = errKindHTTP5xx
		}
		return nil, newChatError(kind, fmt.Errorf("http %d: %s", resp.StatusCode, bytes.TrimSpace(raw)))
	}

	var api embedAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&api); err != nil {
		return nil, newChatError(errKindDecode, fmt.Errorf("decode response: %w", err))
	}
	if len(api.Data) == 0 {
		return nil, newChatError(errKindDecode, errors.New("embeddings response had no data"))
	}

	// Order by `index` (spec-compliant) so the response matches input order even
	// if the server returned them out of order.
	out := make([][]float64, len(api.Data))
	for _, d := range api.Data {
		if d.Index < 0 || d.Index >= len(out) {
			return nil, newChatError(errKindDecode, fmt.Errorf("embeddings index out of range: %d", d.Index))
		}
		out[d.Index] = d.Embedding
	}

	return &embedResult{
		Model:        api.Model,
		Embeddings:   out,
		PromptTokens: api.Usage.PromptTokens,
		Duration:     time.Since(start),
		Inputs:       len(req.input),
	}, nil
}
