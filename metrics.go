package llm

import (
	"context"
	"time"

	"go.k6.io/k6/v2/js/modules"
	"go.k6.io/k6/v2/lib"
	"go.k6.io/k6/v2/metrics"
)

// llmMetrics is the registered metric set. Names are stable; consumers may
// build alerts/dashboards on them.
//
//	llm_requests           Counter           successful chat completions
//	llm_errors             Counter           failures (tag error_type)
//	llm_request_duration   Trend, Time       end-to-end wall time
//	llm_response_headers   Trend, Time       time to HTTP response headers
//	llm_ttft               Trend, Time       time to first content chunk
//	llm_itl                Trend, Time       per-chunk inter-arrival (alias: chunk_latency)
//	llm_tpot               Trend, Time       scalar (e2el-ttft)/(n-1)
//	llm_chunks_per_request Trend, Default    content chunks per request
//	llm_prompt_tokens      Counter           server-reported usage.prompt_tokens
//	llm_completion_tokens  Counter           server-reported usage.completion_tokens
//	llm_goodput            Rate              all SLOs met (only emitted when SLO supplied)
//	llm_slo_ttft           Rate              ttft <= slo.ttft (only when slo.ttft > 0)
//	llm_slo_tpot           Rate              tpot <= slo.tpot (only when slo.tpot > 0 AND tpot derivable)
//	llm_slo_e2el           Rate              e2el <= slo.e2el (only when slo.e2el > 0)
//	llm_tool_calls         Counter           tool invocations the model emitted (omitted when zero)
//	llm_embed_requests     Counter           successful /v1/embeddings calls
//	llm_embed_duration     Trend, Time       end-to-end wall time of an embed call
//	llm_embed_tokens       Counter           server-reported usage.prompt_tokens for embed calls
//	llm_embed_inputs       Counter           number of input strings per embed call
//	llm_embed_errors       Counter           embed-call failures (tag error_type)
//	llm_aborted            Counter           chat completions cut short by abort_after_ms or abort_after_tokens
//	llm_ratelimit_remaining_requests  Gauge  provider-reported remaining requests (only when the header is present)
//	llm_ratelimit_remaining_tokens    Gauge  provider-reported remaining tokens (only when the header is present)
type llmMetrics struct {
	Requests         *metrics.Metric
	Errors           *metrics.Metric
	Duration         *metrics.Metric
	ResponseHeaders  *metrics.Metric
	TTFT             *metrics.Metric
	ITL              *metrics.Metric
	TPOT             *metrics.Metric
	Chunks           *metrics.Metric
	PromptTokens     *metrics.Metric
	CompletionTokens *metrics.Metric
	Goodput          *metrics.Metric
	SLOTTFT          *metrics.Metric
	SLOTPOT          *metrics.Metric
	SLOE2EL          *metrics.Metric
	EnergyJ          *metrics.Metric
	EnergyJPerToken  *metrics.Metric
	CostUSD          *metrics.Metric
	ToolCalls        *metrics.Metric
	EmbedRequests    *metrics.Metric
	EmbedDuration    *metrics.Metric
	EmbedTokens      *metrics.Metric
	EmbedInputs      *metrics.Metric
	EmbedErrors      *metrics.Metric
	Aborted          *metrics.Metric
	// Rate-limit headroom the provider reported after the call. Gauges, so a
	// threshold can hold the latest value above a floor.
	RateLimitRemainingRequests *metrics.Metric
	RateLimitRemainingTokens   *metrics.Metric
}

func registerMetrics(vu modules.VU) (llmMetrics, error) {
	r := vu.InitEnv().Registry
	type spec struct {
		name string
		typ  metrics.MetricType
		val  metrics.ValueType
		into **metrics.Metric
	}
	var m llmMetrics
	specs := []spec{
		{"llm_requests", metrics.Counter, metrics.Default, &m.Requests},
		{"llm_errors", metrics.Counter, metrics.Default, &m.Errors},
		{"llm_request_duration", metrics.Trend, metrics.Time, &m.Duration},
		{"llm_response_headers", metrics.Trend, metrics.Time, &m.ResponseHeaders},
		{"llm_ttft", metrics.Trend, metrics.Time, &m.TTFT},
		{"llm_itl", metrics.Trend, metrics.Time, &m.ITL},
		{"llm_tpot", metrics.Trend, metrics.Time, &m.TPOT},
		{"llm_chunks_per_request", metrics.Trend, metrics.Default, &m.Chunks},
		{"llm_prompt_tokens", metrics.Counter, metrics.Default, &m.PromptTokens},
		{"llm_completion_tokens", metrics.Counter, metrics.Default, &m.CompletionTokens},
		{"llm_goodput", metrics.Rate, metrics.Default, &m.Goodput},
		{"llm_slo_ttft", metrics.Rate, metrics.Default, &m.SLOTTFT},
		{"llm_slo_tpot", metrics.Rate, metrics.Default, &m.SLOTPOT},
		{"llm_slo_e2el", metrics.Rate, metrics.Default, &m.SLOE2EL},
		{"llm_energy_j", metrics.Trend, metrics.Default, &m.EnergyJ},
		{"llm_energy_j_per_token", metrics.Trend, metrics.Default, &m.EnergyJPerToken},
		{"llm_cost_usd", metrics.Trend, metrics.Default, &m.CostUSD},
		{"llm_tool_calls", metrics.Counter, metrics.Default, &m.ToolCalls},
		{"llm_embed_requests", metrics.Counter, metrics.Default, &m.EmbedRequests},
		{"llm_embed_duration", metrics.Trend, metrics.Time, &m.EmbedDuration},
		{"llm_embed_tokens", metrics.Counter, metrics.Default, &m.EmbedTokens},
		{"llm_embed_inputs", metrics.Counter, metrics.Default, &m.EmbedInputs},
		{"llm_embed_errors", metrics.Counter, metrics.Default, &m.EmbedErrors},
		{"llm_aborted", metrics.Counter, metrics.Default, &m.Aborted},
		{"llm_ratelimit_remaining_requests", metrics.Gauge, metrics.Default, &m.RateLimitRemainingRequests},
		{"llm_ratelimit_remaining_tokens", metrics.Gauge, metrics.Default, &m.RateLimitRemainingTokens},
	}
	for _, s := range specs {
		metric, err := r.NewMetric(s.name, s.typ, s.val)
		if err != nil {
			return m, err
		}
		*s.into = metric
	}
	return m, nil
}

// sampler builds samples that share one timestamp, tag set and metadata.
type sampler struct {
	now     time.Time
	tags    *metrics.TagSet
	meta    map[string]string
	samples []metrics.Sample
}

func (s *sampler) add(m *metrics.Metric, v float64) {
	s.samples = append(s.samples, metrics.Sample{
		Time:       s.now,
		TimeSeries: metrics.TimeSeries{Metric: m, Tags: s.tags},
		Value:      v,
		Metadata:   s.meta,
	})
}

func (s *sampler) addBool(m *metrics.Metric, pass bool) {
	v := 0.0
	if pass {
		v = 1.0
	}
	s.add(m, v)
}

// newSampler tags every sample with the VU's current tags plus model and the
// request-scoped extras. It returns nil outside a running VU.
func (c *Client) newSampler(model string, extraTags map[string]string) (*sampler, *lib.State) {
	state := c.mod.vu.State()
	if state == nil {
		return nil, nil
	}
	ctm := state.Tags.GetCurrentValues()
	tags := ctm.Tags.With("model", model)
	for k, v := range extraTags {
		tags = tags.With(k, v)
	}
	return &sampler{now: time.Now(), tags: tags, meta: ctm.Metadata}, state
}

// emit pushes the per-request metric samples for a successful chat completion.
func (c *Client) emit(ctx context.Context, model string, r *chatResult, extraTags map[string]string) {
	s, state := c.newSampler(model, extraTags)
	if s == nil {
		return
	}
	mx := c.mod.metrics

	s.add(mx.Requests, 1)
	s.add(mx.Duration, metrics.D(r.Duration))
	if !r.Unary {
		s.add(mx.Chunks, float64(r.Chunks))
	}
	if r.ResponseHeaders > 0 {
		s.add(mx.ResponseHeaders, metrics.D(r.ResponseHeaders))
	}
	if r.TTFT > 0 {
		s.add(mx.TTFT, metrics.D(r.TTFT))
	}
	for _, itl := range r.ITL {
		s.add(mx.ITL, metrics.D(itl))
	}
	if r.TPOTDerivable() {
		s.add(mx.TPOT, metrics.D(r.TPOT()))
	}
	if r.PromptTokens > 0 {
		s.add(mx.PromptTokens, float64(r.PromptTokens))
	}
	if r.CompletionTokens > 0 {
		s.add(mx.CompletionTokens, float64(r.CompletionTokens))
	}
	if r.Aborted {
		s.add(mx.Aborted, 1)
	}
	if n := len(r.ToolCalls); n > 0 {
		s.add(mx.ToolCalls, float64(n))
	}
	if r.RateLimitRemainingRequests >= 0 {
		s.add(mx.RateLimitRemainingRequests, float64(r.RateLimitRemainingRequests))
	}
	if r.RateLimitRemainingTokens >= 0 {
		s.add(mx.RateLimitRemainingTokens, float64(r.RateLimitRemainingTokens))
	}

	// Per-SLO and goodput samples, only when an SLO predicate was supplied.
	if r.SLO != nil && !r.SLO.Empty() {
		slo := r.sloOutcome()
		if slo.TTFTChecked {
			s.addBool(mx.SLOTTFT, slo.TTFTPass)
		}
		if slo.TPOTChecked {
			s.addBool(mx.SLOTPOT, slo.TPOTPass)
		}
		if slo.E2ELChecked {
			s.addBool(mx.SLOE2EL, slo.E2ELPass)
		}
		s.addBool(mx.Goodput, slo.AllPass)
	}

	if usd, ok := c.costOf(r); ok {
		s.add(mx.CostUSD, usd)
	}

	if em := c.cfg.Energy; !em.Empty() {
		total := em.Joules(r.PromptTokens, r.CompletionTokens, r.Duration)
		s.add(mx.EnergyJ, total)
		if r.CompletionTokens > 0 {
			s.add(mx.EnergyJPerToken, total/float64(r.CompletionTokens))
		}
	}

	metrics.PushIfNotDone(ctx, state.Samples, metrics.ConnectedSamples{Samples: s.samples})
}

// emitError pushes a chat error sample tagged with the categorized error_type.
func (c *Client) emitError(ctx context.Context, model, errorType string, extraTags map[string]string) {
	c.emitErrorOn(ctx, c.mod.metrics.Errors, model, errorType, extraTags)
}

// emitEmbedError is emitError for the embed metric; the error kinds are shared.
func (c *Client) emitEmbedError(ctx context.Context, model, errorType string, extraTags map[string]string) {
	c.emitErrorOn(ctx, c.mod.metrics.EmbedErrors, model, errorType, extraTags)
}

func (c *Client) emitErrorOn(ctx context.Context, metric *metrics.Metric, model, errorType string, extraTags map[string]string) {
	s, state := c.newSampler(model, extraTags)
	if s == nil {
		return
	}
	s.tags = s.tags.With("error_type", errorType)
	s.add(metric, 1)
	metrics.PushIfNotDone(ctx, state.Samples, metrics.ConnectedSamples{Samples: s.samples})
}

// emitGoodputMiss counts a failed call against goodput when an SLO applies.
//
// Goodput is the share of requests that were useful. A request that errored or
// timed out was not, and leaving it out made goodput rise as a server fell
// over: the slowest requests turned into errors and left the denominator.
func (c *Client) emitGoodputMiss(ctx context.Context, model string, req *chatRequest) {
	if req.slo.Empty() {
		return
	}
	s, state := c.newSampler(model, req.tagSet())
	if s == nil {
		return
	}
	s.addBool(c.mod.metrics.Goodput, false)
	metrics.PushIfNotDone(ctx, state.Samples, metrics.ConnectedSamples{Samples: s.samples})
}

// emitEmbed pushes per-request samples for a successful /v1/embeddings call.
func (c *Client) emitEmbed(ctx context.Context, r *embedResult, extraTags map[string]string) {
	s, state := c.newSampler(r.Model, extraTags)
	if s == nil {
		return
	}
	mx := c.mod.metrics
	s.add(mx.EmbedRequests, 1)
	s.add(mx.EmbedDuration, metrics.D(r.Duration))
	s.add(mx.EmbedInputs, float64(r.Inputs))
	if r.PromptTokens > 0 {
		s.add(mx.EmbedTokens, float64(r.PromptTokens))
	}
	metrics.PushIfNotDone(ctx, state.Samples, metrics.ConnectedSamples{Samples: s.samples})
}

// costOf returns the USD cost of a call: the provider's own figure when it
// reported one, otherwise the client-side model, otherwise nothing.
func (c *Client) costOf(r *chatResult) (float64, bool) {
	if r.ServerCost > 0 {
		return r.ServerCost, true
	}
	if cm := c.cfg.Cost; !cm.Empty() {
		return cm.USD(r.PromptTokens, r.CachedTokens, r.CompletionTokens), true
	}
	return 0, false
}
