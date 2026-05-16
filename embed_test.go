package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func embedServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestParseEmbedRequest(t *testing.T) {
	t.Parallel()
	c := &Client{cfg: &Options{Model: "default"}}

	got, err := c.parseEmbedRequest(map[string]any{"input": "hello"})
	require.NoError(t, err)
	require.Equal(t, []string{"hello"}, got.input)
	require.Equal(t, "default", got.model)

	got, err = c.parseEmbedRequest(map[string]any{
		"input": []any{"a", "b"},
		"model": "override",
	})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, got.input)
	require.Equal(t, "override", got.model)

	_, err = c.parseEmbedRequest(map[string]any{})
	require.ErrorContains(t, err, "input is required")

	_, err = c.parseEmbedRequest(map[string]any{"input": ""})
	require.ErrorContains(t, err, "input must not be empty")

	_, err = c.parseEmbedRequest(map[string]any{"input": []any{}})
	require.ErrorContains(t, err, "input array must not be empty")

	_, err = c.parseEmbedRequest(map[string]any{"input": []any{"a", 42}})
	require.ErrorContains(t, err, "input[1] must be a string")

	_, err = c.parseEmbedRequest(map[string]any{"input": 5})
	require.ErrorContains(t, err, "must be a string or string array")
}

func TestDoEmbed_SingleString(t *testing.T) {
	t.Parallel()
	srv := embedServer(t, `{
		"model": "test-emb",
		"data": [{"object":"embedding","embedding":[0.1,0.2,0.3],"index":0}],
		"usage": {"prompt_tokens": 4, "total_tokens": 4}
	}`)

	c := &Client{cfg: &Options{BaseURL: srv.URL, Model: "test-emb"}, http: srv.Client()}
	res, err := c.doEmbed(context.Background(), &embedRequest{
		input: []string{"hello"}, model: "test-emb",
	})
	require.NoError(t, err)
	require.Equal(t, "test-emb", res.Model)
	require.Equal(t, []float64{0.1, 0.2, 0.3}, res.Embeddings[0])
	require.Equal(t, 4, res.PromptTokens)
	require.Equal(t, 1, res.Inputs)
}

func TestDoEmbed_BatchAndReordering(t *testing.T) {
	t.Parallel()
	// Server returns out-of-order data; we must reorder by `index`.
	srv := embedServer(t, `{
		"model": "test-emb",
		"data": [
			{"object":"embedding","embedding":[2.0],"index":1},
			{"object":"embedding","embedding":[0.0],"index":0},
			{"object":"embedding","embedding":[3.0],"index":2}
		],
		"usage": {"prompt_tokens": 9, "total_tokens": 9}
	}`)

	c := &Client{cfg: &Options{BaseURL: srv.URL, Model: "test-emb"}, http: srv.Client()}
	res, err := c.doEmbed(context.Background(), &embedRequest{
		input: []string{"a", "b", "c"}, model: "test-emb",
	})
	require.NoError(t, err)
	require.Equal(t, []float64{0.0}, res.Embeddings[0])
	require.Equal(t, []float64{2.0}, res.Embeddings[1])
	require.Equal(t, []float64{3.0}, res.Embeddings[2])
	require.Equal(t, 3, res.Inputs)
}

func TestDoEmbed_HTTPError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad model"}`))
	}))
	t.Cleanup(srv.Close)

	c := &Client{cfg: &Options{BaseURL: srv.URL, Model: "x"}, http: srv.Client()}
	_, err := c.doEmbed(context.Background(), &embedRequest{input: []string{"x"}, model: "x"})
	require.Error(t, err)
	require.Equal(t, errKindHTTP4xx, errorKind(err))
}

func TestDoEmbed_OutOfRangeIndex(t *testing.T) {
	t.Parallel()
	srv := embedServer(t, `{
		"model":"x",
		"data":[{"object":"embedding","embedding":[0],"index":7}],
		"usage":{"prompt_tokens":1,"total_tokens":1}
	}`)
	c := &Client{cfg: &Options{BaseURL: srv.URL, Model: "x"}, http: srv.Client()}
	_, err := c.doEmbed(context.Background(), &embedRequest{input: []string{"a"}, model: "x"})
	require.ErrorContains(t, err, "index out of range")
}

func TestDoEmbed_SerializesSingleAsBareString(t *testing.T) {
	t.Parallel()
	// Verifies the wire-level choice to send `input: "x"` rather than `["x"]`
	// when there's exactly one input. Both work with OpenAI; some self-hosted
	// servers only support the bare-string form, so prefer it.
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"model":"x","data":[{"embedding":[0],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)
	c := &Client{cfg: &Options{BaseURL: srv.URL, Model: "x"}, http: srv.Client()}
	_, err := c.doEmbed(context.Background(), &embedRequest{input: []string{"hello"}, model: "x"})
	require.NoError(t, err)
	s, ok := got["input"].(string)
	require.True(t, ok, "single input should serialize as a string, got %T", got["input"])
	require.Equal(t, "hello", s)
}

func TestDoEmbed_BatchSerializesAsArray(t *testing.T) {
	t.Parallel()
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"model":"x","data":[
			{"embedding":[0],"index":0},{"embedding":[1],"index":1}
		],"usage":{"prompt_tokens":2,"total_tokens":2}}`))
	}))
	t.Cleanup(srv.Close)
	c := &Client{cfg: &Options{BaseURL: srv.URL, Model: "x"}, http: srv.Client()}
	_, err := c.doEmbed(context.Background(), &embedRequest{input: []string{"a", "b"}, model: "x"})
	require.NoError(t, err)
	arr, ok := got["input"].([]any)
	require.True(t, ok, "batched input should serialize as an array, got %T", got["input"])
	require.Len(t, arr, 2)
}

func TestEmbedResult_ToJSObject(t *testing.T) {
	t.Parallel()
	r := &embedResult{Model: "m", Embeddings: [][]float64{{1, 2}, {3, 4}}, PromptTokens: 8, Inputs: 2}
	out := r.toJSObject()
	require.Equal(t, "m", out["model"])
	require.Equal(t, 8, out["prompt_tokens"])
	require.Equal(t, 2, out["inputs"])
	embs, ok := out["embeddings"].([]any)
	require.True(t, ok)
	require.Len(t, embs, 2)
}

func TestParseEmbedRequest_TagValidation(t *testing.T) {
	t.Parallel()
	c := &Client{cfg: &Options{Model: "default"}}
	_, err := c.parseEmbedRequest(map[string]any{
		"input": "x",
		"tags":  "not a map",
	})
	require.ErrorContains(t, err, "tags must be an object")

	_, err = c.parseEmbedRequest(map[string]any{
		"input": "x",
		"tags":  map[string]any{"k": 5},
	})
	require.ErrorContains(t, err, `tags["k"]`)
}
