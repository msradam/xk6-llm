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
	// DefaultSLO applies to every chat() call that doesn't supply its own.
	DefaultSLO *SLOPredicate
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
	if v, ok := m["slo"]; ok && v != nil {
		slo, err := parseSLO(v)
		if err != nil {
			return nil, err
		}
		o.DefaultSLO = slo
	}
	return o, nil
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
