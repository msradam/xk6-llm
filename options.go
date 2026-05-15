package llm

import (
	"fmt"
	"strings"
	"time"
)

// Options configures an llm.Client.
type Options struct {
	BaseURL   string
	APIKey    string
	Model     string
	Timeout   time.Duration
	IgnoreEOS bool
	// Headers are sent on every request. Use for custom auth schemes, gateway
	// routing keys (e.g. OpenRouter "HTTP-Referer"), or observability headers.
	Headers map[string]string
	// DefaultSLO applies to every chat() call that doesn't supply its own.
	DefaultSLO *SLOPredicate
	// Energy, when set, enables per-request energy estimation. See EnergyModel.
	Energy *EnergyModel
	// Cost, when set, enables per-request USD estimation. See CostModel.
	Cost *CostModel
}

// CostModel parameterises a per-request USD cost estimate, computed from
// server-reported token counts:
//
//	usd = prompt_tokens     * usd_per_million_input_tokens  / 1e6
//	    + completion_tokens * usd_per_million_output_tokens / 1e6
//
// Use hosted-API published rates directly; for self-hosted inference, compute
// your effective $/M-token rate offline (idle GPU $/hr divided by sustained
// throughput, plus marginal electricity) and plug it in here.
type CostModel struct {
	USDPerMInputTokens  float64
	USDPerMOutputTokens float64
}

// Empty reports whether the model would produce zero for any request.
func (c *CostModel) Empty() bool {
	return c == nil || (c.USDPerMInputTokens <= 0 && c.USDPerMOutputTokens <= 0)
}

// USD returns the dollar cost for a request with the given token counts.
func (c *CostModel) USD(promptTokens, completionTokens int) float64 {
	if c.Empty() {
		return 0
	}
	return float64(promptTokens)*c.USDPerMInputTokens/1e6 +
		float64(completionTokens)*c.USDPerMOutputTokens/1e6
}

// EnergyModel parameterises a per-request energy estimate. The math, per request:
//
//	dynamic_j = prompt_tokens * j_per_input_token + completion_tokens * j_per_output_token
//	static_j  = idle_w * (duration_s)
//	total_j   = dynamic_j + static_j
//
// Coefficients must be measured for your (GPU, model, batch regime) tuple.
// Under concurrent load the static term over-attributes idle power; divide
// idle_w by your expected per-VU concurrency for wall-plug accuracy. This is
// a budgeting metric, not a measurement; see RESEARCH.md.
type EnergyModel struct {
	JPerInputToken  float64
	JPerOutputToken float64
	IdleW           float64
}

// Empty reports whether the model would produce zero for any request.
func (e *EnergyModel) Empty() bool {
	return e == nil || (e.JPerInputToken <= 0 && e.JPerOutputToken <= 0 && e.IdleW <= 0)
}

// Joules returns the total energy budget for a request with the given token
// counts and wall-clock duration. Zero when Empty.
func (e *EnergyModel) Joules(promptTokens, completionTokens int, duration time.Duration) float64 {
	if e.Empty() {
		return 0
	}
	dyn := float64(promptTokens)*e.JPerInputToken + float64(completionTokens)*e.JPerOutputToken
	stat := e.IdleW * (float64(duration) / float64(time.Second))
	return dyn + stat
}

// SLOPredicate is the per-request SLO used to compute goodput and per-SLO
// attainment Rates. A zero field disables that SLO (it always passes).
//
// Semantics match vLLM's `--goodput ttft:X tpot:Y e2el:Z` flag (PR #9338,
// shipped v0.6.4). See RESEARCH.md §A.10.
type SLOPredicate struct {
	TTFTMs float64
	TPOTMs float64
	E2ELMs float64
}

// Empty reports whether the predicate is functionally a no-op.
func (s *SLOPredicate) Empty() bool {
	return s == nil || (s.TTFTMs <= 0 && s.TPOTMs <= 0 && s.E2ELMs <= 0)
}

const (
	defaultBaseURL = "http://localhost:11434/v1"
	defaultModel   = "granite4.1:3b"
	defaultTimeout = 60 * time.Second
)

func parseOptions(raw any) (*Options, error) {
	o := &Options{
		BaseURL: defaultBaseURL,
		Model:   defaultModel,
		Timeout: defaultTimeout,
	}
	if raw == nil {
		return o, nil
	}
	m, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("llm.Client: options must be an object, got %T", raw)
	}
	if v, ok := m["base_url"].(string); ok && v != "" {
		o.BaseURL = strings.TrimRight(v, "/")
	}
	if v, ok := m["api_key"].(string); ok {
		o.APIKey = v
	}
	if v, ok := m["model"].(string); ok {
		o.Model = v
	}
	if v, ok := m["timeout_ms"]; ok {
		if d, ok := asMillis(v); ok {
			o.Timeout = d
		}
	}
	if v, ok := m["ignore_eos"].(bool); ok {
		o.IgnoreEOS = v
	}
	if v, ok := m["headers"]; ok && v != nil {
		hm, ok := v.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("llm.Client: 'headers' must be an object, got %T", v)
		}
		o.Headers = make(map[string]string, len(hm))
		for hk, hv := range hm {
			s, ok := hv.(string)
			if !ok {
				return nil, fmt.Errorf("llm.Client: headers[%q] must be a string, got %T", hk, hv)
			}
			o.Headers[hk] = s
		}
	}
	if v, ok := m["slo"]; ok && v != nil {
		slo, err := parseSLO(v)
		if err != nil {
			return nil, err
		}
		o.DefaultSLO = slo
	}
	if v, ok := m["energy"]; ok && v != nil {
		em, err := parseEnergy(v)
		if err != nil {
			return nil, err
		}
		o.Energy = em
	}
	if v, ok := m["cost"]; ok && v != nil {
		cm, err := parseCost(v)
		if err != nil {
			return nil, err
		}
		o.Cost = cm
	}
	return o, nil
}

func parseCost(raw any) (*CostModel, error) {
	m, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("llm: 'cost' must be an object, got %T", raw)
	}
	c := &CostModel{}
	for _, key := range []string{"usd_per_million_input_tokens", "usd_per_million_output_tokens"} {
		v, present := m[key]
		if !present {
			continue
		}
		f, ok := asFloat(v)
		if !ok {
			return nil, fmt.Errorf("llm: cost.%s must be a number, got %T", key, v)
		}
		if f < 0 {
			return nil, fmt.Errorf("llm: cost.%s must be non-negative, got %v", key, f)
		}
		switch key {
		case "usd_per_million_input_tokens":
			c.USDPerMInputTokens = f
		case "usd_per_million_output_tokens":
			c.USDPerMOutputTokens = f
		}
	}
	return c, nil
}

func parseEnergy(raw any) (*EnergyModel, error) {
	m, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("llm: 'energy' must be an object, got %T", raw)
	}
	e := &EnergyModel{}
	for _, key := range []string{"j_per_input_token", "j_per_output_token", "idle_w"} {
		v, present := m[key]
		if !present {
			continue
		}
		f, ok := asFloat(v)
		if !ok {
			return nil, fmt.Errorf("llm: energy.%s must be a number, got %T", key, v)
		}
		if f < 0 {
			return nil, fmt.Errorf("llm: energy.%s must be non-negative, got %v", key, f)
		}
		switch key {
		case "j_per_input_token":
			e.JPerInputToken = f
		case "j_per_output_token":
			e.JPerOutputToken = f
		case "idle_w":
			e.IdleW = f
		}
	}
	return e, nil
}

func parseSLO(raw any) (*SLOPredicate, error) {
	m, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("llm: 'slo' must be an object, got %T", raw)
	}
	s := &SLOPredicate{}
	for _, key := range []string{"ttft_ms", "tpot_ms", "e2el_ms"} {
		v, present := m[key]
		if !present {
			continue
		}
		f, ok := asFloat(v)
		if !ok {
			return nil, fmt.Errorf("llm: slo.%s must be a number, got %T", key, v)
		}
		switch key {
		case "ttft_ms":
			s.TTFTMs = f
		case "tpot_ms":
			s.TPOTMs = f
		case "e2el_ms":
			s.E2ELMs = f
		}
	}
	return s, nil
}

func asFloat(v any) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case int64:
		return float64(t), true
	case int:
		return float64(t), true
	}
	return 0, false
}

func asMillis(v any) (time.Duration, bool) {
	f, ok := asFloat(v)
	if !ok {
		return 0, false
	}
	return time.Duration(f) * time.Millisecond, true
}
