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

func TestParseChatRequest_Stream(t *testing.T) {
	t.Parallel()
	c := newTestClient(t, "http://unused")

	req, err := c.parseChatRequest(map[string]any{"messages": []any{}, "stream": false})
	require.NoError(t, err)
	assert.True(t, req.unary)
	assert.NotContains(t, req.body, "stream", "stream is a control field, not forwarded as-is")
	assert.Equal(t, map[string]string{"mode": "unary"}, req.tagSet())

	req, err = c.parseChatRequest(map[string]any{"messages": []any{}, "stream": true})
	require.NoError(t, err)
	assert.False(t, req.unary)
	assert.Nil(t, req.tagSet(), "streamed calls keep their existing tag set")

	_, err = c.parseChatRequest(map[string]any{"messages": []any{}, "stream": "no"})
	require.ErrorContains(t, err, "stream must be a boolean")

	_, err = c.parseChatRequest(map[string]any{"messages": []any{}, "stream": false, "abort_after_tokens": 3})
	require.ErrorContains(t, err, "abort_after_tokens requires a streamed call")

	_, err = c.parseChatRequest(map[string]any{"messages": []any{}, "stream": false, "abort_after_ms": 50})
	require.NoError(t, err, "a wall-clock deadline still applies to a unary call")
}

func TestDoChat_Unary(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name       string
		wire       Wire
		path       string
		response   string
		checkReq   func(t *testing.T, r *http.Request, body map[string]any)
		content    string
		prompt     int
		completion int
		cached     int
		finish     string
		tools      []ToolCall
	}{
		{
			name: "openai chat completion",
			wire: WireOpenAI,
			path: "/chat/completions",
			response: `{"choices":[{"message":{"role":"assistant","content":"hi there","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"id\":42}"}}]},"finish_reason":"tool_calls"}],
				"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18,"prompt_tokens_details":{"cached_tokens":4}}}`,
			checkReq: func(t *testing.T, _ *http.Request, body map[string]any) {
				assert.NotContains(t, body, "stream")
				assert.NotContains(t, body, "stream_options")
			},
			content: "hi there", prompt: 11, completion: 7, cached: 4, finish: "tool_calls",
			tools: []ToolCall{{ID: "call_1", Name: "lookup", Arguments: `{"id":42}`}},
		},
		{
			name: "anthropic message",
			wire: WireAnthropic,
			path: "/messages",
			response: `{"type":"message","content":[{"type":"text","text":"hi "},{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"id":42}},{"type":"text","text":"there"}],
				"stop_reason":"tool_use","usage":{"input_tokens":12,"output_tokens":8,"cache_read_input_tokens":5}}`,
			checkReq: func(t *testing.T, _ *http.Request, body map[string]any) {
				assert.Equal(t, false, body["stream"])
			},
			content: "hi there", prompt: 12, completion: 8, cached: 5, finish: "tool_use",
			tools: []ToolCall{{ID: "toolu_1", Name: "lookup", Arguments: `{"id":42}`}},
		},
		{
			name: "openai responses object",
			wire: WireResponses,
			path: "/responses",
			response: `{"status":"completed","output":[{"type":"reasoning"},{"type":"message","content":[{"type":"output_text","text":"hi there"}]},{"type":"function_call","call_id":"call_9","name":"lookup","arguments":"{\"id\":42}"}],
				"usage":{"input_tokens":13,"input_tokens_details":{"cached_tokens":6},"output_tokens":9,"output_tokens_details":{"reasoning_tokens":2}}}`,
			checkReq: func(t *testing.T, _ *http.Request, body map[string]any) {
				assert.Equal(t, false, body["stream"])
			},
			content: "hi there", prompt: 13, completion: 9, cached: 6, finish: "completed",
			tools: []ToolCall{{ID: "call_9", Name: "lookup", Arguments: `{"id":42}`}},
		},
		{
			name:     "providerwire v4 unary document",
			wire:     WireProviderWireV4,
			path:     "/language-model",
			response: `{"content":[{"type":"text","text":"hi there"}],"finishReason":{"unified":"stop","raw":"stop"},"usage":{"inputTokens":{"total":14},"outputTokens":{"total":10}}}`,
			checkReq: func(t *testing.T, r *http.Request, body map[string]any) {
				assert.Equal(t, "false", r.Header.Get("Ai-Language-Model-Streaming"))
				assert.Contains(t, body, "prompt")
			},
			content: "hi there", prompt: 14, completion: 10, finish: "stop",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, tc.path, r.URL.Path)
				assert.Equal(t, "application/json", r.Header.Get("Accept"))
				raw, err := io.ReadAll(r.Body)
				assert.NoError(t, err)
				var body map[string]any
				assert.NoError(t, json.Unmarshal(raw, &body))
				tc.checkReq(t, r, body)
				time.Sleep(5 * time.Millisecond)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.response)
			}))
			t.Cleanup(srv.Close)

			c := newTestClient(t, srv.URL)
			c.cfg.Wire = tc.wire
			req, err := c.parseChatRequest(map[string]any{
				"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
				"max_tokens": 16,
				"stream":     false,
			})
			require.NoError(t, err)
			res, err := c.doChat(context.Background(), req)
			require.NoError(t, err)

			assert.True(t, res.Unary)
			assert.Equal(t, tc.content, res.Content)
			assert.Equal(t, tc.prompt, res.PromptTokens)
			assert.Equal(t, tc.completion, res.CompletionTokens)
			assert.Equal(t, tc.cached, res.CachedTokens)
			assert.Equal(t, tc.finish, res.FinishReason)
			assert.Equal(t, tc.tools, res.ToolCalls)
			assert.Zero(t, res.TTFT, "a unary call has no first token to time")
			assert.Empty(t, res.ITL)
			assert.Zero(t, res.Chunks)
			assert.False(t, res.TPOTDerivable())
			assert.Greater(t, res.Duration, time.Duration(0))
			assert.Greater(t, res.ResponseHeaders, time.Duration(0))
			assert.Equal(t, false, res.toJSObject()["stream"])
		})
	}
}

func TestDoChat_UnaryErrors(t *testing.T) {
	t.Parallel()
	t.Run("http error is classified before parsing", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = io.WriteString(w, `{"error":{"message":"boom"}}`)
		}))
		t.Cleanup(srv.Close)
		c := newTestClient(t, srv.URL)
		req, err := c.parseChatRequest(map[string]any{"messages": []any{}, "stream": false})
		require.NoError(t, err)
		_, err = c.doChat(context.Background(), req)
		require.Error(t, err)
		assert.Equal(t, errKindHTTP5xx, errorKind(err))
	})

	t.Run("undecodable body is a decode error", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = io.WriteString(w, `data: {"not":"json for unary"}`)
		}))
		t.Cleanup(srv.Close)
		c := newTestClient(t, srv.URL)
		req, err := c.parseChatRequest(map[string]any{"messages": []any{}, "stream": false})
		require.NoError(t, err)
		_, err = c.doChat(context.Background(), req)
		require.Error(t, err)
		assert.Equal(t, errKindDecode, errorKind(err))
	})

	t.Run("in-band error envelope fails the call", func(t *testing.T) {
		t.Parallel()
		_, err := parseChatCompletion([]byte(`{"error":{"message":"overloaded"}}`))
		require.ErrorContains(t, err, "overloaded")
		_, err = parseResponsesObject([]byte(`{"status":"failed","error":{"message":"quota"}}`))
		require.ErrorContains(t, err, "quota")
	})
}

func TestChatResult_UnarySLOSkipsTTFT(t *testing.T) {
	t.Parallel()
	r := &chatResult{Unary: true, Duration: 80 * time.Millisecond, SLO: &SLOPredicate{TTFTMs: 1, E2ELMs: 100}}
	out := r.sloOutcome()
	assert.False(t, out.TTFTChecked, "TTFT is not measured on a unary call, so it cannot pass or fail")
	assert.True(t, out.E2ELChecked)
	assert.True(t, out.AllPass)
}

func TestTextFromContent(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "", textFromContent(nil))
	assert.Equal(t, "", textFromContent(json.RawMessage(`null`)))
	assert.Equal(t, "plain", textFromContent(json.RawMessage(`"plain"`)))
	assert.Equal(t, "ab", textFromContent(json.RawMessage(`[{"type":"text","text":"a"},{"type":"image_url"},{"type":"text","text":"b"}]`)))
}
