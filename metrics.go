package llm

import (
	"time"

	"go.k6.io/k6/v2/js/modules"
	"go.k6.io/k6/v2/metrics"
)

type llmMetrics struct {
	Requests         *metrics.Metric
	Errors           *metrics.Metric
	Duration         *metrics.Metric
	TTFT             *metrics.Metric
	ITL              *metrics.Metric
	PromptTokens     *metrics.Metric
	CompletionTokens *metrics.Metric
}

func registerMetrics(vu modules.VU) (llmMetrics, error) {
	r := vu.InitEnv().Registry
	var (
		m   llmMetrics
		err error
	)
	if m.Requests, err = r.NewMetric("llm_requests", metrics.Counter); err != nil {
		return m, err
	}
	if m.Errors, err = r.NewMetric("llm_errors", metrics.Counter); err != nil {
		return m, err
	}
	if m.Duration, err = r.NewMetric("llm_request_duration", metrics.Trend, metrics.Time); err != nil {
		return m, err
	}
	if m.TTFT, err = r.NewMetric("llm_ttft", metrics.Trend, metrics.Time); err != nil {
		return m, err
	}
	if m.ITL, err = r.NewMetric("llm_itl", metrics.Trend, metrics.Time); err != nil {
		return m, err
	}
	if m.PromptTokens, err = r.NewMetric("llm_prompt_tokens", metrics.Counter); err != nil {
		return m, err
	}
	if m.CompletionTokens, err = r.NewMetric("llm_completion_tokens", metrics.Counter); err != nil {
		return m, err
	}
	return m, nil
}

func (c *Client) emit(model string, r *chatResult) {
	state := c.mod.vu.State()
	if state == nil {
		return
	}
	ctm := state.Tags.GetCurrentValues()
	tags := ctm.Tags.With("model", model)
	now := time.Now()
	ctx := c.mod.vu.Context()
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
	metrics.PushIfNotDone(ctx, state.Samples, metrics.ConnectedSamples{Samples: samples})
}

func (c *Client) emitError(model string) {
	state := c.mod.vu.State()
	if state == nil {
		return
	}
	ctm := state.Tags.GetCurrentValues()
	tags := ctm.Tags.With("model", model)
	metrics.PushIfNotDone(c.mod.vu.Context(), state.Samples, metrics.ConnectedSamples{
		Samples: []metrics.Sample{{
			Time:       time.Now(),
			TimeSeries: metrics.TimeSeries{Metric: c.mod.metrics.Errors, Tags: tags},
			Value:      1,
			Metadata:   ctm.Metadata,
		}},
	})
}
