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
		switch t := v.(type) {
		case int64:
			o.Timeout = time.Duration(t) * time.Millisecond
		case float64:
			o.Timeout = time.Duration(t) * time.Millisecond
		}
	}
	if v, ok := m["ignore_eos"].(bool); ok {
		o.IgnoreEOS = v
	}
	return o, nil
}
