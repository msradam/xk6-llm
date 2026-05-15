package llm

import (
	"context"
	"time"

	"go.k6.io/k6/v2/js/modules"
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

// emit pushes the per-request metric samples for a successful chat completion.
func (c *Client) emit(ctx context.Context, model string, r *chatResult, extraTags map[string]string) {
	state := c.mod.vu.State()
	if state == nil {
		return
	}
	ctm := state.Tags.GetCurrentValues()
	tags := ctm.Tags.With("model", model)
	for k, v := range extraTags {
		tags = tags.With(k, v)
	}
	now := time.Now()
	mx := c.mod.metrics

	samples := []metrics.Sample{
		{
			Time:       now,
			TimeSeries: metrics.TimeSeries{Metric: mx.Requests, Tags: tags},
			Value:      1,
			Metadata:   ctm.Metadata,
		},
		{
			Time:       now,
			TimeSeries: metrics.TimeSeries{Metric: mx.Duration, Tags: tags},
			Value:      metrics.D(r.Duration),
			Metadata:   ctm.Metadata,
		},
		{
			Time:       now,
			TimeSeries: metrics.TimeSeries{Metric: mx.Chunks, Tags: tags},
			Value:      float64(r.Chunks),
			Metadata:   ctm.Metadata,
		},
	}
	if r.ResponseHeaders > 0 {
		samples = append(samples, metrics.Sample{
			Time:       now,
			TimeSeries: metrics.TimeSeries{Metric: mx.ResponseHeaders, Tags: tags},
			Value:      metrics.D(r.ResponseHeaders),
			Metadata:   ctm.Metadata,
		})
	}
	if r.TTFT > 0 {
		samples = append(samples, metrics.Sample{
			Time:       now,
			TimeSeries: metrics.TimeSeries{Metric: mx.TTFT, Tags: tags},
			Value:      metrics.D(r.TTFT),
			Metadata:   ctm.Metadata,
		})
	}
	for _, itl := range r.ITL {
		samples = append(samples, metrics.Sample{
			Time:       now,
			TimeSeries: metrics.TimeSeries{Metric: mx.ITL, Tags: tags},
			Value:      metrics.D(itl),
			Metadata:   ctm.Metadata,
		})
	}
	if r.TPOTDerivable() {
		samples = append(samples, metrics.Sample{
			Time:       now,
			TimeSeries: metrics.TimeSeries{Metric: mx.TPOT, Tags: tags},
			Value:      metrics.D(r.TPOT()),
			Metadata:   ctm.Metadata,
		})
	}
	if r.PromptTokens > 0 {
		samples = append(samples, metrics.Sample{
			Time:       now,
			TimeSeries: metrics.TimeSeries{Metric: mx.PromptTokens, Tags: tags},
			Value:      float64(r.PromptTokens),
			Metadata:   ctm.Metadata,
		})
	}
	if r.CompletionTokens > 0 {
		samples = append(samples, metrics.Sample{
			Time:       now,
			TimeSeries: metrics.TimeSeries{Metric: mx.CompletionTokens, Tags: tags},
			Value:      float64(r.CompletionTokens),
			Metadata:   ctm.Metadata,
		})
	}

	// Per-SLO and goodput samples, only when an SLO predicate was supplied.
	if r.SLO != nil && !r.SLO.Empty() {
		ok := true
		if r.SLO.TTFTMs > 0 {
			pass := float64(r.TTFT)/float64(time.Millisecond) <= r.SLO.TTFTMs
			samples = append(samples, boolSample(now, mx.SLOTTFT, tags, ctm.Metadata, pass))
			ok = ok && pass
		}
		if r.SLO.TPOTMs > 0 && r.TPOTDerivable() {
			pass := float64(r.TPOT())/float64(time.Millisecond) <= r.SLO.TPOTMs
			samples = append(samples, boolSample(now, mx.SLOTPOT, tags, ctm.Metadata, pass))
			ok = ok && pass
		}
		if r.SLO.E2ELMs > 0 {
			pass := float64(r.Duration)/float64(time.Millisecond) <= r.SLO.E2ELMs
			samples = append(samples, boolSample(now, mx.SLOE2EL, tags, ctm.Metadata, pass))
			ok = ok && pass
		}
		samples = append(samples, boolSample(now, mx.Goodput, tags, ctm.Metadata, ok))
	}

	if cm := c.cfg.Cost; !cm.Empty() {
		samples = append(samples, metrics.Sample{
			Time:       now,
			TimeSeries: metrics.TimeSeries{Metric: mx.CostUSD, Tags: tags},
			Value:      cm.USD(r.PromptTokens, r.CompletionTokens),
			Metadata:   ctm.Metadata,
		})
	}

	if em := c.cfg.Energy; !em.Empty() {
		total := em.Joules(r.PromptTokens, r.CompletionTokens, r.Duration)
		samples = append(samples, metrics.Sample{
			Time:       now,
			TimeSeries: metrics.TimeSeries{Metric: mx.EnergyJ, Tags: tags},
			Value:      total,
			Metadata:   ctm.Metadata,
		})
		if r.CompletionTokens > 0 {
			samples = append(samples, metrics.Sample{
				Time:       now,
				TimeSeries: metrics.TimeSeries{Metric: mx.EnergyJPerToken, Tags: tags},
				Value:      total / float64(r.CompletionTokens),
				Metadata:   ctm.Metadata,
			})
		}
	}

	metrics.PushIfNotDone(ctx, state.Samples, metrics.ConnectedSamples{Samples: samples})
}

// emitError pushes an error sample tagged with the categorized error_type.
func (c *Client) emitError(ctx context.Context, model, errorType string, extraTags map[string]string) {
	state := c.mod.vu.State()
	if state == nil {
		return
	}
	ctm := state.Tags.GetCurrentValues()
	tags := ctm.Tags.With("model", model).With("error_type", errorType)
	for k, v := range extraTags {
		tags = tags.With(k, v)
	}
	metrics.PushIfNotDone(ctx, state.Samples, metrics.ConnectedSamples{
		Samples: []metrics.Sample{{
			Time:       time.Now(),
			TimeSeries: metrics.TimeSeries{Metric: c.mod.metrics.Errors, Tags: tags},
			Value:      1,
			Metadata:   ctm.Metadata,
		}},
	})
}

func boolSample(t time.Time, metric *metrics.Metric, tags *metrics.TagSet, meta map[string]string, pass bool) metrics.Sample {
	v := 0.0
	if pass {
		v = 1.0
	}
	return metrics.Sample{
		Time:       t,
		TimeSeries: metrics.TimeSeries{Metric: metric, Tags: tags},
		Value:      v,
		Metadata:   meta,
	}
}
