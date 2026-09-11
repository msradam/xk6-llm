package llm

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/grafana/agento11y/go/agento11y/model"
	agento11yv1 "github.com/grafana/agento11y/go/proto/agento11y/v1"
	"github.com/grafana/agento11y/go/proto/agento11y/wire"
	"github.com/stretchr/testify/require"
)

func TestParseAgento11y(t *testing.T) {
	t.Parallel()

	cfg, err := parseAgento11y(map[string]any{"endpoint": "http://127.0.0.1:9400"})
	require.NoError(t, err)
	require.Equal(t, "http", cfg.Protocol, "http is the default protocol")
	require.Equal(t, "none", cfg.AuthMode)
	require.True(t, cfg.Synthetic, "load-generated traffic is synthetic by default")
	require.False(t, cfg.CaptureContent, "content capture is opt-in")

	_, err = parseAgento11y(map[string]any{})
	require.ErrorContains(t, err, "endpoint")

	_, err = parseAgento11y(map[string]any{"endpoint": "x", "protocol": "carrier-pigeon"})
	require.ErrorContains(t, err, "protocol")

	_, err = parseAgento11y(map[string]any{"endpoint": "x", "auth_mode": "mtls"})
	require.ErrorContains(t, err, "auth_mode")

	_, err = parseAgento11y(map[string]any{"endpoint": "x", "tags": map[string]any{"k": 1}})
	require.ErrorContains(t, err, "tags.k")

	cfg, err = parseAgento11y(map[string]any{
		"endpoint": "x", "synthetic": false, "capture_content": true,
		"agent_name": "canary", "agent_version": "v2",
		"tags": map[string]any{"team": "ai"},
	})
	require.NoError(t, err)
	require.False(t, cfg.Synthetic)
	require.True(t, cfg.CaptureContent)
	require.Equal(t, "canary", cfg.AgentName)
	require.Equal(t, map[string]string{"team": "ai"}, cfg.Tags)
}

func TestLoopbackEndpoint(t *testing.T) {
	t.Parallel()
	// Cleartext defaults on for loopback only; a remote endpoint must not be
	// silently downgraded off TLS.
	for _, e := range []string{"127.0.0.1:9400", "http://localhost:1234", "https://[::1]:9"} {
		require.True(t, loopbackEndpoint(e), e)
	}
	for _, e := range []string{"agento11y.grafana.net", "https://example.com", "10.0.0.5:9400"} {
		require.False(t, loopbackEndpoint(e), e)
	}
}

func TestITLStats(t *testing.T) {
	t.Parallel()
	mean, p50, max := itlStats([]time.Duration{
		10 * time.Millisecond,
		30 * time.Millisecond,
		20 * time.Millisecond,
	})
	require.InDelta(t, 20.0, mean, 0.001)
	require.InDelta(t, 20.0, p50, 0.001)
	require.InDelta(t, 30.0, max, 0.001)
}

func TestSLOOutcome_SharedByMetricsAndExport(t *testing.T) {
	t.Parallel()

	// TTFT passes, E2EL fails → not all pass. The k6 Rate samples and the
	// exported goodput metadata both read this, so they cannot disagree.
	res := &chatResult{
		TTFT:             50 * time.Millisecond,
		Duration:         900 * time.Millisecond,
		CompletionTokens: 10,
		SLO:              &SLOPredicate{TTFTMs: 100, E2ELMs: 500},
	}
	got := res.sloOutcome()
	require.True(t, got.TTFTChecked)
	require.True(t, got.TTFTPass)
	require.True(t, got.E2ELChecked)
	require.False(t, got.E2ELPass)
	require.False(t, got.TPOTChecked, "no TPOT threshold supplied")
	require.False(t, got.AllPass)

	// No predicate → vacuously passing, and nothing marked checked.
	none := (&chatResult{}).sloOutcome()
	require.True(t, none.AllPass)
	require.False(t, none.TTFTChecked)
}

func TestGenerationMetadata_CarriesMetricsWithNoSchemaField(t *testing.T) {
	t.Parallel()
	c := &Client{cfg: &Options{
		Agento11y: &Agento11yConfig{},
		Cost:      &CostModel{USDPerMInputTokens: 1, USDPerMOutputTokens: 2},
	}}
	res := &chatResult{
		TTFT:             120 * time.Millisecond,
		ITL:              []time.Duration{10 * time.Millisecond, 12 * time.Millisecond},
		Duration:         500 * time.Millisecond,
		ResponseHeaders:  100 * time.Millisecond,
		Chunks:           2,
		PromptTokens:     1_000_000,
		CompletionTokens: 1_000_000,
	}

	meta := c.generationMetadata(res)

	// TTFT has a real home in the schema; these do not, which is the gap this
	// sink is meant to make visible.
	require.InDelta(t, 120.0, meta[MetaTTFTMs], 0.001)
	require.InDelta(t, 11.0, meta[MetaITLMeanMs], 0.001)
	require.InDelta(t, 12.0, meta[MetaITLMaxMs], 0.001)
	require.Equal(t, 2, meta[MetaITLSamples])
	require.Contains(t, meta, MetaTPOTMs)
	require.InDelta(t, 3.0, meta[MetaCostUSD], 0.001)
	require.NotContains(t, meta, MetaEnergyJ, "omitted when no energy model is configured")
}

func TestGenerationTags_SyntheticMarker(t *testing.T) {
	t.Parallel()
	c := &Client{cfg: &Options{Agento11y: &Agento11yConfig{
		Synthetic: true,
		Tags:      map[string]string{"team": "ai", "path": "client-scope"},
	}}}
	req := &chatRequest{tags: map[string]string{"path": "request-scope"}}

	tags := c.generationTags(req)
	require.Equal(t, "true", tags[SyntheticTagKey])
	require.Equal(t, syntheticProducer, tags[SyntheticProducerTagKey])
	require.Equal(t, "ai", tags["team"])
	require.Equal(t, "request-scope", tags["path"], "request tags win over client tags")

	c.cfg.Agento11y.Synthetic = false
	require.NotContains(t, c.generationTags(req), SyntheticTagKey)
}

// TestExportGeneration_RoundTrip drives a real SDK client against an HTTP
// receiver and decodes the payload with the SDK's own wire codec, so this
// asserts the actual on-the-wire contract rather than our view of it.
func TestExportGeneration_RoundTrip(t *testing.T) {
	t.Parallel()

	var (
		mu       sync.Mutex
		gotPath  string
		gotCount int
		gotTags  map[string]string
		gotModel string
		gotUsage int64
		gotMeta  map[string]any
	)
	done := make(chan struct{}, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		req, err := wire.UnmarshalExportGenerationsJSON(body)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		mu.Lock()
		gotPath = r.URL.Path
		gotCount += len(req.GetGenerations())
		if g := req.GetGenerations(); len(g) > 0 {
			gotTags = g[0].GetTags()
			gotModel = g[0].GetModel().GetName()
			gotUsage = g[0].GetUsage().GetOutputTokens()
			gotMeta = g[0].GetMetadata().AsMap()
		}
		mu.Unlock()
		// The SDK requires one result per submitted generation; an empty
		// response is a protocol error, not an accepted batch.
		results := make([]*agento11yv1.ExportGenerationResult, 0, len(req.GetGenerations()))
		for _, g := range req.GetGenerations() {
			results = append(results, &agento11yv1.ExportGenerationResult{
				GenerationId: g.GetId(),
				Accepted:     true,
			})
		}
		payload, err := wire.MarshalExportGenerationsResponseJSON(
			&agento11yv1.ExportGenerationsResponse{Results: results})
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
		select {
		case done <- struct{}{}:
		default:
		}
	}))
	t.Cleanup(srv.Close)

	cfg := &Agento11yConfig{
		Endpoint:  srv.URL,
		Protocol:  "http",
		AuthMode:  "none",
		Synthetic: true,
		AgentName: "k6-canary",
		Tags:      map[string]string{"scenario": "roundtrip"},
	}
	c := &Client{
		cfg:  &Options{Model: "stub-model", Agento11y: cfg, Wire: WireAnthropic},
		a11y: newAgento11yClient(cfg),
	}

	res := &chatResult{
		TTFT:             90 * time.Millisecond,
		ITL:              []time.Duration{11 * time.Millisecond, 9 * time.Millisecond},
		Duration:         400 * time.Millisecond,
		ResponseHeaders:  80 * time.Millisecond,
		Chunks:           2,
		PromptTokens:     42,
		CompletionTokens: 7,
		FinishReason:     "end_turn",
		Content:          "hello",
	}
	req := &chatRequest{body: map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}}

	c.exportGeneration(t.Context(), "stub-model", req, res)
	require.NoError(t, c.Flush())

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("no export received")
	}

	mu.Lock()
	defer mu.Unlock()
	require.Equal(t, wire.GenerationExportHTTPPath, gotPath)
	require.Equal(t, 1, gotCount)
	require.Equal(t, "stub-model", gotModel)
	require.Equal(t, int64(7), gotUsage)
	require.Equal(t, "true", gotTags[SyntheticTagKey],
		"a consumer must be able to exclude load-generated traffic")
	require.Equal(t, "roundtrip", gotTags["scenario"])
	require.InDelta(t, 10.0, gotMeta[MetaITLMeanMs], 0.001,
		"ITL survives the wire as metadata because the schema has no field for it")
	require.InDelta(t, 90.0, gotMeta[MetaTTFTMs], 0.001)
}

func TestParseParentIDs(t *testing.T) {
	t.Parallel()

	ids, err := parseParentIDs("a")
	require.NoError(t, err)
	require.Equal(t, []string{"a"}, ids)

	ids, err = parseParentIDs([]any{"a", "b"})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, ids, "a call may have several causes")

	ids, err = parseParentIDs([]any{"a", ""})
	require.NoError(t, err)
	require.Equal(t, []string{"a"}, ids, "empty entries dropped")

	ids, err = parseParentIDs("")
	require.NoError(t, err)
	require.Nil(t, ids)

	_, err = parseParentIDs([]any{"a", 7})
	require.ErrorContains(t, err, "parent_generation_ids[1]")

	_, err = parseParentIDs(42)
	require.ErrorContains(t, err, "must be a string or string array")
}

func TestParseChatRequest_GenerationLineageIsControlOnly(t *testing.T) {
	t.Parallel()
	c := &Client{cfg: &Options{Model: "m"}}

	req, err := c.parseChatRequest(map[string]any{
		"messages":              []any{map[string]any{"role": "user", "content": "hi"}},
		"generation_id":         "gen-child",
		"parent_generation_ids": []any{"gen-a", "gen-b"},
	})
	require.NoError(t, err)
	require.Equal(t, "gen-child", req.generationID)
	require.Equal(t, []string{"gen-a", "gen-b"}, req.parentGenerationIDs)

	// Lineage is telemetry, not a model input. Leaking it into the provider
	// request would be rejected by Anthropic outright and silently billed as
	// prompt tokens by others.
	require.NotContains(t, req.body, "generation_id")
	require.NotContains(t, req.body, "parent_generation_ids")
	require.Contains(t, req.body, "messages")
}

func TestNewGenerationID_Unique(t *testing.T) {
	t.Parallel()
	seen := make(map[string]bool, 1000)
	for i := 0; i < 1000; i++ {
		id := newGenerationID()
		require.False(t, seen[id], "duplicate generation id: %s", id)
		seen[id] = true
	}
}

// TestExportGeneration_CarriesParentLineage proves the parent edges survive
// the wire into the generation record.
//
// This is the field a gateway cannot fill. Nothing in a stateless HTTP request
// says which earlier request caused it, so an observing proxy has to guess;
// a synthetic workload knows its own call graph and can state it.
func TestExportGeneration_CarriesParentLineage(t *testing.T) {
	t.Parallel()

	var (
		mu      sync.Mutex
		gotID   string
		gotPar  []string
		gotConv string
	)
	done := make(chan struct{}, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		req, err := wire.UnmarshalExportGenerationsJSON(body)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		results := make([]*agento11yv1.ExportGenerationResult, 0, len(req.GetGenerations()))
		for _, g := range req.GetGenerations() {
			mu.Lock()
			gotID = g.GetId()
			gotPar = g.GetParentGenerationIds()
			gotConv = g.GetConversationId()
			mu.Unlock()
			results = append(results, &agento11yv1.ExportGenerationResult{
				GenerationId: g.GetId(), Accepted: true,
			})
		}
		payload, _ := wire.MarshalExportGenerationsResponseJSON(
			&agento11yv1.ExportGenerationsResponse{Results: results})
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(payload)
		select {
		case done <- struct{}{}:
		default:
		}
	}))
	t.Cleanup(srv.Close)

	cfg := &Agento11yConfig{Endpoint: srv.URL, Protocol: "http", AuthMode: "none", Synthetic: true}
	c := &Client{cfg: &Options{Model: "m", Agento11y: cfg}, a11y: newAgento11yClient(cfg)}

	res := &chatResult{
		GenerationID: "gen-child", TTFT: 10 * time.Millisecond,
		Duration: 100 * time.Millisecond, PromptTokens: 5, CompletionTokens: 3,
	}
	req := &chatRequest{
		body:                map[string]any{"messages": []any{}},
		parentGenerationIDs: []string{"gen-root", "gen-sibling"},
		tags:                map[string]string{"session_id": "conv-1"},
	}

	c.exportGeneration(t.Context(), "m", req, res)
	require.NoError(t, c.Flush())

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("no export received")
	}

	mu.Lock()
	defer mu.Unlock()
	require.Equal(t, "gen-child", gotID, "the client-minted id is the record id")
	require.Equal(t, []string{"gen-root", "gen-sibling"}, gotPar,
		"multi-parent lineage survives the wire")
	require.Equal(t, "conv-1", gotConv)
}

// TestSystemPrompt_BothWireShapes covers the two places a system instruction
// can live. The Anthropic and Responses translation layers hoist it out of the
// messages array into a top-level field, so reading only `messages` returns
// empty for two of the three wires.
func TestSystemPrompt_BothWireShapes(t *testing.T) {
	t.Parallel()

	openAI := &chatRequest{body: map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "be terse"},
			map[string]any{"role": "user", "content": "hi"},
		},
	}}
	require.Equal(t, "be terse", systemPrompt(openAI))

	// Anthropic / Responses shape after translation.
	hoisted := &chatRequest{body: map[string]any{
		"system":   "be terse",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}}
	require.Equal(t, "be terse", systemPrompt(hoisted))

	// Session builds history as []map[string]any, not []any.
	session := &chatRequest{body: map[string]any{
		"messages": []map[string]any{
			{"role": "system", "content": "be terse"},
			{"role": "user", "content": "hi"},
		},
	}}
	require.Equal(t, "be terse", systemPrompt(session))

	require.Empty(t, systemPrompt(&chatRequest{body: map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}}), "no system message means no system prompt")
}

// TestSystemPrompt_MultipleMessagesJoin mirrors anthropicBody, which joins
// several system messages with a blank line rather than keeping the first.
func TestSystemPrompt_MultipleMessagesJoin(t *testing.T) {
	t.Parallel()
	req := &chatRequest{body: map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "first"},
			map[string]any{"role": "system", "content": "second"},
			map[string]any{"role": "user", "content": "hi"},
		},
	}}
	require.Equal(t, "first\n\nsecond", systemPrompt(req))
}

// TestPromptMessages_ExcludesSystem guards against double-reporting. The system
// instruction travels in Generation.SystemPrompt; emitting it in Input too
// would duplicate it and, because the role switch has no "system" case, would
// mislabel it as a user turn.
func TestPromptMessages_ExcludesSystem(t *testing.T) {
	t.Parallel()
	msgs := promptMessages(&chatRequest{body: map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "be terse"},
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "content": "hello"},
		},
	}})
	require.Len(t, msgs, 2)
	require.Equal(t, model.RoleUser, msgs[0].Role)
	require.Equal(t, model.RoleAssistant, msgs[1].Role)
	for _, m := range msgs {
		require.NotEqual(t, "be terse", m.Parts[0].Text, "system prompt must not appear in Input")
	}
}
