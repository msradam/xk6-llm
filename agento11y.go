package llm

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/grafana/agento11y/go/agento11y"
	"github.com/grafana/agento11y/go/agento11y/model"
	metricnoop "go.opentelemetry.io/otel/metric/noop"
	tracenoop "go.opentelemetry.io/otel/trace/noop"
)

// Agento11yConfig configures export of one Grafana Agent Observability
// generation record per chat call.
//
// The extension owns its own k6 metrics and OTel-free timing, so the SDK client
// is built with no-op tracer and meter: only the generation-export path runs.
type Agento11yConfig struct {
	// Endpoint is the Agent Observability generation-export base URL. The SDK
	// appends the export path. Empty disables export.
	Endpoint string
	// Protocol is "http" (default), "grpc", or "none".
	Protocol string
	// AuthMode is "none" (default), "tenant", "bearer", or "basic".
	AuthMode      string
	TenantID      string
	BearerToken   string
	BasicUser     string
	BasicPassword string
	// Insecure permits cleartext export. Defaults to true for a loopback
	// endpoint and false otherwise, so a local demo works without ceremony
	// while a remote endpoint is not silently downgraded.
	Insecure *bool

	AgentName    string
	AgentVersion string
	// Synthetic marks every exported generation as load-generated. Default
	// true: traffic from a load generator is synthetic by construction, and a
	// consumer that cannot distinguish it will corrupt its own cost and usage
	// reporting.
	Synthetic bool
	// CaptureContent sends prompt and completion text. Default false —
	// benchmark corpora are usually uninteresting and sometimes proprietary.
	CaptureContent bool
	// Tags are merged into every exported generation.
	Tags map[string]string
	// FlushInterval bounds how long a record waits before export.
	FlushInterval time.Duration
}

// SyntheticTagKey marks a generation as produced by a load generator rather
// than by real traffic.
//
// A consumer that treats synthetic records as real traffic will overstate
// usage and cost. Until Agent Observability reserves a tag for this, the
// convention has to live somewhere public — hence a constant rather than a
// string literal in an example script.
const SyntheticTagKey = "agento11y.synthetic"

// SyntheticProducerTagKey identifies which generator produced the record, so a
// consumer can tell a k6 canary apart from other synthetic sources.
const SyntheticProducerTagKey = "agento11y.synthetic.producer"

const syntheticProducer = "xk6-llm"

// Metadata keys for measurements the Generation schema has no field for.
//
// Agent Observability records gen_ai.client.time_to_first_token and the total
// operation duration, but nothing about the shape of the stream in between: a
// response that streams smoothly at 40 tok/s and one that stalls for two
// seconds mid-generation are indistinguishable in the schema today. Until ITL
// and TPOT have real fields, they travel as metadata.
const (
	MetaTTFTMs            = "llm.ttft_ms"
	MetaITLMeanMs         = "llm.itl_mean_ms"
	MetaITLP50Ms          = "llm.itl_p50_ms"
	MetaITLMaxMs          = "llm.itl_max_ms"
	MetaITLSamples        = "llm.itl_samples"
	MetaTPOTMs            = "llm.tpot_ms"
	MetaChunks            = "llm.chunks"
	MetaThinkingTokens    = "llm.thinking_tokens"
	MetaCachedTokens      = "llm.cached_tokens"
	MetaThinkingChunks    = "llm.thinking_chunks"
	MetaTTFTextMs         = "llm.ttf_text_ms"
	MetaResponseHeadersMs = "llm.response_headers_ms"
	MetaCostUSD           = "llm.cost_usd"
	MetaEnergyJ           = "llm.energy_j"
	MetaGoodput           = "llm.goodput"
	MetaAborted           = "llm.aborted"
)

// parseParentIDs accepts a single id or an array of ids.
func parseParentIDs(raw any) ([]string, error) {
	switch v := raw.(type) {
	case string:
		if v == "" {
			return nil, nil
		}
		return []string{v}, nil
	case []any:
		out := make([]string, 0, len(v))
		for i, item := range v {
			s, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("llm: parent_generation_ids[%d] must be a string, got %T", i, item)
			}
			if s != "" {
				out = append(out, s)
			}
		}
		return out, nil
	case []string:
		return v, nil
	}
	return nil, fmt.Errorf("llm: parent_generation_ids must be a string or string array, got %T", raw)
}

// newGenerationID returns a process-unique generation id.
//
// The id is minted client-side rather than left to the SDK because the caller
// needs it before the call completes: a workload declares its own call graph by
// passing an earlier id as a parent, and it cannot do that with an id it never
// sees.
func newGenerationID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		// A counter alone is still unique within the process; the random
		// prefix only guards against collisions across VUs and runs.
		return "xk6-" + strconv.FormatUint(generationSeq.Add(1), 36)
	}
	return "xk6-" + hex.EncodeToString(b[:]) + "-" + strconv.FormatUint(generationSeq.Add(1), 36)
}

var generationSeq atomic.Uint64

func parseAgento11y(raw any) (*Agento11yConfig, error) {
	m, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("llm.Client: 'agento11y' must be an object, got %T", raw)
	}
	cfg := &Agento11yConfig{
		Protocol:  "http",
		AuthMode:  string(agento11y.ExportAuthModeNone),
		Synthetic: true,
	}
	str := func(key string, dst *string) {
		if v, ok := m[key].(string); ok && v != "" {
			*dst = v
		}
	}
	str("endpoint", &cfg.Endpoint)
	str("protocol", &cfg.Protocol)
	str("auth_mode", &cfg.AuthMode)
	str("tenant_id", &cfg.TenantID)
	str("bearer_token", &cfg.BearerToken)
	str("basic_user", &cfg.BasicUser)
	str("basic_password", &cfg.BasicPassword)
	str("agent_name", &cfg.AgentName)
	str("agent_version", &cfg.AgentVersion)

	if v, ok := m["synthetic"].(bool); ok {
		cfg.Synthetic = v
	}
	if v, ok := m["capture_content"].(bool); ok {
		cfg.CaptureContent = v
	}
	if v, ok := m["insecure"].(bool); ok {
		cfg.Insecure = &v
	}
	if v, ok := m["flush_interval_ms"]; ok {
		if d, ok := asMillis(v); ok {
			cfg.FlushInterval = d
		}
	}
	if v, ok := m["tags"]; ok && v != nil {
		tm, ok := v.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("llm.Client: 'agento11y.tags' must be an object, got %T", v)
		}
		cfg.Tags = make(map[string]string, len(tm))
		for k, tv := range tm {
			s, ok := tv.(string)
			if !ok {
				return nil, fmt.Errorf("llm.Client: 'agento11y.tags.%s' must be a string, got %T", k, tv)
			}
			cfg.Tags[k] = s
		}
	}

	if cfg.Endpoint == "" {
		return nil, fmt.Errorf("llm.Client: 'agento11y.endpoint' is required")
	}
	switch cfg.Protocol {
	case "http", "grpc", "none":
	default:
		return nil, fmt.Errorf("llm.Client: 'agento11y.protocol' must be http, grpc, or none, got %q", cfg.Protocol)
	}
	switch agento11y.ExportAuthMode(cfg.AuthMode) {
	case agento11y.ExportAuthModeNone, agento11y.ExportAuthModeTenant,
		agento11y.ExportAuthModeBearer, agento11y.ExportAuthModeBasic:
	default:
		return nil, fmt.Errorf("llm.Client: 'agento11y.auth_mode' must be none, tenant, bearer, or basic, got %q", cfg.AuthMode)
	}
	return cfg, nil
}

// loopbackEndpoint reports whether an endpoint targets the local host, which is
// the only case where defaulting to cleartext is safe.
func loopbackEndpoint(endpoint string) bool {
	e := strings.ToLower(endpoint)
	e = strings.TrimPrefix(strings.TrimPrefix(e, "http://"), "https://")
	return strings.HasPrefix(e, "localhost") ||
		strings.HasPrefix(e, "127.0.0.1") ||
		strings.HasPrefix(e, "[::1]")
}

func newAgento11yClient(cfg *Agento11yConfig) *agento11y.Client {
	insecure := loopbackEndpoint(cfg.Endpoint)
	if cfg.Insecure != nil {
		insecure = *cfg.Insecure
	}
	capture := agento11y.ContentCaptureModeMetadataOnly
	if cfg.CaptureContent {
		capture = agento11y.ContentCaptureModeFull
	}
	return agento11y.NewClient(agento11y.Config{
		GenerationExport: agento11y.GenerationExportConfig{
			Protocol: agento11y.GenerationExportProtocol(cfg.Protocol),
			Endpoint: cfg.Endpoint,
			Insecure: &insecure,
			Auth: agento11y.AuthConfig{
				Mode:          agento11y.ExportAuthMode(cfg.AuthMode),
				TenantID:      cfg.TenantID,
				BearerToken:   cfg.BearerToken,
				BasicUser:     cfg.BasicUser,
				BasicPassword: cfg.BasicPassword,
			},
			FlushInterval: cfg.FlushInterval,
		},
		ContentCapture: capture,
		// The extension owns its own metrics and timing; the SDK's tracer and
		// meter would double-count and drag OTel into the VU hot path.
		Tracer: tracenoop.NewTracerProvider().Tracer("xk6-llm"),
		Meter:  metricnoop.NewMeterProvider().Meter("xk6-llm"),
	})
}

// exportGeneration ships one generation record for a completed chat call.
//
// Failures are reported through the existing k6 metric set rather than failing
// the iteration: a telemetry sink that can break a load test is worse than one
// that drops records.
func (c *Client) exportGeneration(ctx context.Context, modelName string, req *chatRequest, res *chatResult) {
	if c.a11y == nil || c.cfg.Agento11y == nil {
		return
	}
	acfg := c.cfg.Agento11y

	startedAt := time.Now().Add(-res.Duration)
	tags := c.generationTags(req)

	start := agento11y.GenerationStart{
		ID:             res.GenerationID,
		ConversationID: tags["session_id"],
		AgentName:      acfg.AgentName,
		AgentVersion:   acfg.AgentVersion,
		Model:          agento11y.ModelRef{Provider: c.providerName(), Name: modelName},
		// The call graph a synthetic workload declares about itself. This is
		// the field a stateless proxy cannot populate: nothing in an HTTP
		// request says which earlier request caused it.
		ParentGenerationIDs: req.parentGenerationIDs,
		Tags:                tags,
		StartedAt:           startedAt,
	}

	_, rec := c.a11y.StartStreamingGeneration(ctx, start)
	if res.TTFT > 0 {
		rec.SetFirstTokenAt(startedAt.Add(res.TTFT))
	}

	gen := agento11y.Generation{
		ID:                  res.GenerationID,
		ParentGenerationIDs: req.parentGenerationIDs,
		Model:               agento11y.ModelRef{Provider: c.providerName(), Name: modelName},
		ResponseModel:       modelName,
		StopReason:          res.FinishReason,
		Usage: model.TokenUsage{
			InputTokens:  int64(res.PromptTokens),
			OutputTokens: int64(res.CompletionTokens),
			TotalTokens:  int64(res.PromptTokens + res.CompletionTokens),
			// Sub-buckets of the totals above, never additive.
			ReasoningTokens:      int64(res.ThinkingTokens),
			CacheReadInputTokens: int64(res.CachedTokens),
		},
		StartedAt:   startedAt,
		CompletedAt: startedAt.Add(res.Duration),
		Tags:        tags,
		Metadata:    c.generationMetadata(res),
	}
	sys := systemPrompt(req)

	// Agent identity must not depend on content capture.
	//
	// With no declared agent_version the collector derives a version by
	// hashing the system prompt, but it only ever sees that prompt under
	// content capture, which is off by default. Leaving identity to the
	// collector therefore collapsed every agent to one hash of the empty
	// string in the default configuration, and toggling capture moved an
	// unchanged prompt to a different version. Deriving the digest here fixes
	// both: EffectiveVersion is metadata, so it survives metadata-only mode.
	//
	// A declared version is the caller's to own, so this only fills the gap.
	if acfg.AgentVersion == "" && sys != "" {
		gen.EffectiveVersion = effectiveVersion(sys)
	}

	// SystemPrompt is content, so it ships only under capture. The SDK strips
	// it in metadata-only mode, so sending it otherwise would be dropped
	// downstream and imply a guarantee that does not hold.
	if acfg.CaptureContent {
		gen.SystemPrompt = sys
		gen.Input = promptMessages(req)
		// Only attach parts that carry something. A text part holding an empty
		// string sets no payload field, which fails SDK validation with
		// "generation.output[0].parts[0] must set exactly one payload field"
		// and drops the entire record. That would silently lose exactly the
		// case these checks exist to catch: a request that billed for output
		// and returned no text. The token counts carry it instead.
		var outParts []model.Part
		if res.Content != "" {
			outParts = append(outParts, model.Part{Kind: model.PartKindText, Text: res.Content})
		}
		// A tool call is the only record that the model asked for evidence. It
		// belongs on the output side: Sigil's tool_calls projection reads calls
		// from output and results from the *next* generation's input, and an
		// agentic turn often has no text at all, so without this the turn
		// exports as an empty generation.
		for _, tc := range res.ToolCalls {
			outParts = append(outParts, model.Part{
				Kind:     model.PartKindToolCall,
				ToolCall: &model.ToolCall{ID: tc.ID, Name: tc.Name, InputJSON: rawJSON(tc.Arguments)},
			})
		}
		if len(outParts) > 0 {
			gen.Output = []model.Message{{Role: model.RoleAssistant, Parts: outParts}}
		}
	}

	rec.SetResult(gen, nil)
	rec.End()
	if err := rec.Err(); err != nil {
		c.emitError(ctx, modelName, errKindExport, req.tagSet())
	}
}

// providerName labels the generation with the wire it was measured over, which
// is the only provider identity the extension can know for certain — a
// base_url may point at a gateway, a proxy, or a self-hosted server.
func (c *Client) providerName() string {
	if c.cfg.Wire == WireAnthropic {
		return "anthropic"
	}
	// Both the chat-completions and Responses wires are OpenAI-shaped.
	return "openai"
}

func (c *Client) generationTags(req *chatRequest) map[string]string {
	acfg := c.cfg.Agento11y
	tags := make(map[string]string, len(acfg.Tags)+len(req.tags)+3)
	for k, v := range acfg.Tags {
		tags[k] = v
	}
	// Request-scoped tags win over client-scoped ones, matching how the k6
	// metric tags resolve.
	for k, v := range req.tagSet() {
		tags[k] = v
	}
	if acfg.Synthetic {
		tags[SyntheticTagKey] = "true"
		tags[SyntheticProducerTagKey] = syntheticProducer
	}
	return tags
}

func (c *Client) generationMetadata(res *chatResult) map[string]any {
	meta := map[string]any{
		MetaChunks:            res.Chunks,
		MetaResponseHeadersMs: msOf(res.ResponseHeaders),
	}
	if res.TTFT > 0 {
		meta[MetaTTFTMs] = msOf(res.TTFT)
	}
	if len(res.ITL) > 0 {
		mean, p50, max := itlStats(res.ITL)
		meta[MetaITLMeanMs] = mean
		meta[MetaITLP50Ms] = p50
		meta[MetaITLMaxMs] = max
		meta[MetaITLSamples] = len(res.ITL)
	}
	if res.TPOTDerivable() {
		meta[MetaTPOTMs] = msOf(res.TPOT())
	}
	if res.ThinkingTokens > 0 || res.ThinkingChunks > 0 {
		// Reasoning tokens are billed but, with thinking display omitted, do
		// not stream individually — so TPOT above covers the text phase only.
		// Recording the breakout keeps that interpretable downstream.
		meta[MetaThinkingTokens] = res.ThinkingTokens
		meta[MetaThinkingChunks] = res.ThinkingChunks
	}
	if res.TTFText > 0 {
		meta[MetaTTFTextMs] = msOf(res.TTFText)
	}
	if res.CachedTokens > 0 {
		meta[MetaCachedTokens] = res.CachedTokens
	}
	if res.Aborted {
		meta[MetaAborted] = true
	}
	if !c.cfg.Cost.Empty() {
		meta[MetaCostUSD] = c.cfg.Cost.USD(res.PromptTokens, res.CompletionTokens)
	}
	if !c.cfg.Energy.Empty() {
		meta[MetaEnergyJ] = c.cfg.Energy.Joules(res.PromptTokens, res.CompletionTokens, res.Duration)
	}
	if res.SLO != nil && !res.SLO.Empty() {
		meta[MetaGoodput] = res.sloOutcome().AllPass
	}
	return meta
}

// itlStats returns mean, median, and max in milliseconds. The caller has
// already checked for a non-empty slice.
func itlStats(itl []time.Duration) (mean, p50, max float64) {
	sorted := make([]time.Duration, len(itl))
	copy(sorted, itl)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	var total time.Duration
	for _, d := range sorted {
		total += d
	}
	return msOf(total / time.Duration(len(sorted))),
		msOf(sorted[len(sorted)/2]),
		msOf(sorted[len(sorted)-1])
}

func msOf(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

// systemPrompt returns the request's system instruction.
//
// This field carries agent *identity*, not just content. When a caller
// declares no agent_version, the collector derives the agent's effective
// version by hashing the system prompt, so leaving it empty collapses every
// agent to one constant version and a version-delta view has nothing to
// compare. Two wire shapes produce it: an OpenAI `role: system` message, or
// the top-level `system` field the Anthropic and Responses translation layers
// hoist it into.
func systemPrompt(req *chatRequest) string {
	if s, ok := req.body["system"].(string); ok && s != "" {
		return s
	}
	raw, ok := asAnySlice(req.body["messages"])
	if !ok {
		return ""
	}
	var parts []string
	for _, item := range raw {
		m, ok := item.(map[string]any)
		if !ok || m["role"] != "system" {
			continue
		}
		if text, ok := m["content"].(string); ok && text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, "\n\n")
}

// effectiveVersion returns a stable version identity for a system prompt.
//
// This is deliberately a digest rather than the prompt itself, even though the
// SDK hashes the field again on the way out (codec.EffectiveVersionDigest has
// no already-canonical guard, so the value that reaches the collector is a
// digest of this digest). Two reasons not to "fix" that by passing the raw
// prompt:
//
//   - One export path ships the field unhashed. otel_export.go assigns
//     EffectiveVersion straight through, and while xk6-llm never enables otel
//     mode today, a prompt in that field would leave the process in clear the
//     day someone does. Content capture is off by default for a reason, and
//     version identity must not smuggle content past it.
//   - Nothing downstream needs the preimage. The collector stores the digest
//     as an opaque catalog key, so double hashing costs nothing and the
//     property that matters survives: one prompt is one version, every run,
//     whatever the capture mode.
//
// It follows that this digest does not match what a differently instrumented
// agent reports for the same prompt, which is correct. They are different
// agents.
func effectiveVersion(sys string) string {
	sum := sha256.Sum256([]byte(sys))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// promptMessages converts the request's OpenAI-shaped messages into SDK
// messages. Non-string content (multimodal parts) is skipped rather than
// guessed at: a wrong shape in a telemetry record is worse than a missing one.
func promptMessages(req *chatRequest) []model.Message {
	raw, ok := asAnySlice(req.body["messages"])
	if !ok {
		return nil
	}
	out := make([]model.Message, 0, len(raw))
	for _, item := range raw {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		// The system instruction travels in Generation.SystemPrompt. Emitting
		// it here too would duplicate it, and the role mapping below would
		// mislabel it as a user turn.
		if m["role"] == "system" {
			continue
		}
		role := model.RoleUser
		switch m["role"] {
		case "assistant":
			role = model.RoleAssistant
		case "tool":
			role = model.RoleTool
		}
		parts := messageParts(m, role)
		if len(parts) == 0 {
			continue
		}
		out = append(out, model.Message{Role: role, Parts: parts})
	}
	return out
}

// messageParts converts one OpenAI-shaped message into SDK parts.
//
// A tool result and a tool call are typed parts rather than text, because
// Sigil's tool_calls projection reads the typed fields and ignores text. A
// result flattened to text is a result that never arrived as far as any
// consumer of that projection is concerned.
func messageParts(m map[string]any, role model.Role) []model.Part {
	if role == model.RoleTool {
		// The call id is the correlation key. Without it the result cannot be
		// matched to its call, so a text part is the honest shape.
		id, _ := m["tool_call_id"].(string)
		content, _ := m["content"].(string)
		if id == "" {
			if content == "" {
				return nil
			}
			return []model.Part{{Kind: model.PartKindText, Text: content}}
		}
		name, _ := m["name"].(string)
		// is_error has no place in OpenAI's wire format, so the script says
		// so. Inferring it from the content would be guesswork, and a result
		// wrongly marked sound is the failure this whole path exists to catch.
		isError, _ := m["is_error"].(bool)
		return []model.Part{{
			Kind: model.PartKindToolResult,
			ToolResult: &model.ToolResult{
				ToolCallID: id, Name: name, IsError: isError, Content: content,
			},
		}}
	}

	var parts []model.Part
	if text, ok := m["content"].(string); ok && text != "" {
		parts = append(parts, model.Part{Kind: model.PartKindText, Text: text})
	}
	// An assistant turn that only called tools has null content, so it used to
	// be dropped whole along with its calls.
	calls, _ := asAnySlice(m["tool_calls"])
	for _, raw := range calls {
		tc, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		fn, _ := tc["function"].(map[string]any)
		if fn == nil {
			continue
		}
		name, _ := fn["name"].(string)
		if name == "" {
			continue
		}
		id, _ := tc["id"].(string)
		args, _ := fn["arguments"].(string)
		parts = append(parts, model.Part{
			Kind:     model.PartKindToolCall,
			ToolCall: &model.ToolCall{ID: id, Name: name, InputJSON: rawJSON(args)},
		})
	}
	return parts
}

// rawJSON passes through valid JSON and drops anything else. A provider can
// return arguments that do not parse, and an invalid RawMessage fails the
// whole export rather than just that field.
func rawJSON(s string) json.RawMessage {
	if s == "" || !json.Valid([]byte(s)) {
		return nil
	}
	return json.RawMessage(s)
}

// Flush exports any queued generation records.
//
// A script that needs every record must call this itself, inside the
// iteration. See the note below on why the extension cannot do it.
func (c *Client) Flush() error {
	if c.a11y == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), flushTimeout)
	defer cancel()
	return c.a11y.Flush(ctx)
}

// flushTimeout bounds the end-of-iteration flush. Long enough for a batch to
// land, short enough that a dead collector cannot stall a run.
const flushTimeout = 10 * time.Second

// Records queued late in an iteration are lost if the test ends before the
// flush interval elapses, and the records most likely to be dropped are the
// last ones — for a canary, the observation nearest whatever you were trying
// to catch. Observed dropping the final node of a 4-call agent tree, which was
// also the only multi-parent node in it.
//
// There is no reliable hook to fix this inside the extension. Flushing from a
// goroutine that waits on the VU context does not work: nothing waits for that
// goroutine, so k6 exits before its request completes (verified — the same
// record was still dropped). k6 exposes no per-VU teardown, and teardown() runs
// module init again, so a client constructed there has an empty queue.
//
// A script that needs every record therefore calls Flush() at the end of its
// exported function, where it is synchronous and inside the iteration.
