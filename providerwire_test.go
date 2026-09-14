package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProviderWireV4Body_MapsMessagesAndOptions(t *testing.T) {
	body, err := providerWireV4Body(map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "be brief"},
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": "hello"}}},
		},
		"max_tokens":  int64(64),
		"temperature": 0.0,
		"stop":        "END",
	})
	require.NoError(t, err)
	got, err := json.Marshal(body)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"prompt": [
			{"role": "system", "content": "be brief"},
			{"role": "user", "content": [{"type": "text", "text": "hi"}]},
			{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}
		],
		"maxOutputTokens": 64,
		"temperature": 0,
		"stopSequences": ["END"]
	}`, string(got))
}

func TestProviderWireV4Body_RejectsWhatV4CannotCarry(t *testing.T) {
	for name, body := range map[string]map[string]any{
		"unmapped field":   {"messages": []any{map[string]any{"role": "user", "content": "hi"}}, "tools": []any{}},
		"tool role":        {"messages": []any{map[string]any{"role": "tool", "content": "x"}}},
		"image part":       {"messages": []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "image_url"}}}}},
		"missing messages": {"max_tokens": 1},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := providerWireV4Body(body)
			require.Error(t, err)
		})
	}
}

func TestParseProviderWireV4Stream(t *testing.T) {
	srv := sseServer(t, 5*time.Millisecond, []string{
		`{"type":"stream-start","warnings":[]}`,
		`{"type":"response-metadata","modelId":"grafana/test"}`,
		`{"type":"text-start","id":"0"}`,
		`{"type":"text-delta","id":"0","delta":"Hel"}`,
		`{"type":"text-delta","id":"0","delta":""}`,
		`{"type":"text-delta","id":"0","delta":"lo"}`,
		`{"type":"text-end","id":"0"}`,
		`{"type":"finish","usage":{"inputTokens":{"total":3},"outputTokens":{"total":2}},"finishReason":{"unified":"stop","raw":"stop"}}`,
	})
	resp := httpGet(t, srv.URL)
	res, err := parseProviderWireV4Stream(context.Background(), resp.Body, time.Now(), abortPolicy{})
	require.NoError(t, err)
	assert.Equal(t, "Hello", res.Content)
	assert.Equal(t, 2, res.Chunks)
	assert.Len(t, res.ITL, 1)
	assert.Positive(t, res.TTFT)
	assert.Equal(t, 3, res.PromptTokens)
	assert.Equal(t, 2, res.CompletionTokens)
	assert.Equal(t, "stop", res.FinishReason)
}

func TestParseProviderWireV4Stream_ErrorPart(t *testing.T) {
	srv := sseServer(t, 0, []string{
		`{"type":"stream-start","warnings":[]}`,
		`{"type":"error","error":{"message":"request timed out","code":"timeout","statusCode":504,"retryable":true}}`,
	})
	resp := httpGet(t, srv.URL)
	_, err := parseProviderWireV4Stream(context.Background(), resp.Body, time.Now(), abortPolicy{})
	require.ErrorContains(t, err, "request timed out")
}

func TestDoChat_ProviderWireV4Request(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/language-model", r.URL.Path)
		assert.Equal(t, "grafana/test", r.Header.Get("ai-language-model-id"))
		assert.Equal(t, "4", r.Header.Get("ai-language-model-specification-version"))
		assert.Equal(t, "true", r.Header.Get("ai-language-model-streaming"))
		assert.Equal(t, "token", r.Header.Get("X-Access-Token"))
		raw, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		assert.JSONEq(t, `{"prompt":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`, string(raw))
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"text-delta\",\"id\":\"0\",\"delta\":\"ok\"}\n\n")
	}))
	t.Cleanup(srv.Close)

	c := newTestClient(t, srv.URL)
	c.cfg.Wire = WireProviderWireV4
	c.cfg.Model = "grafana/test"
	c.cfg.Headers = map[string]string{"X-Access-Token": "token"}
	res, err := c.doChat(context.Background(), &chatRequest{body: map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}})
	require.NoError(t, err)
	assert.Equal(t, "ok", res.Content)
}

func TestParseOptions_WireProviderWireV4(t *testing.T) {
	o, err := parseOptions(map[string]any{"wire": "providerwire-v4"})
	require.NoError(t, err)
	assert.Equal(t, WireProviderWireV4, o.Wire)
}
