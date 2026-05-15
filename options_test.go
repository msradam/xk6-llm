package llm

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestParseEnergy(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		raw     any
		want    *EnergyModel
		wantErr string
	}{
		{
			name: "all fields",
			raw: map[string]any{
				"j_per_input_token":  0.5,
				"j_per_output_token": 1.2,
				"idle_w":             50.0,
			},
			want: &EnergyModel{JPerInputToken: 0.5, JPerOutputToken: 1.2, IdleW: 50},
		},
		{
			name: "partial: only output",
			raw:  map[string]any{"j_per_output_token": 1.0},
			want: &EnergyModel{JPerOutputToken: 1.0},
		},
		{
			name:    "non-object",
			raw:     "x",
			wantErr: "must be an object",
		},
		{
			name:    "wrong type",
			raw:     map[string]any{"idle_w": "hot"},
			wantErr: "energy.idle_w must be a number",
		},
		{
			name:    "negative rejected",
			raw:     map[string]any{"j_per_input_token": -0.1},
			wantErr: "must be non-negative",
		},
		{
			name: "int converts to float",
			raw:  map[string]any{"idle_w": int64(75)},
			want: &EnergyModel{IdleW: 75},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := parseEnergy(tc.raw)
			if tc.wantErr != "" {
				require.Error(t, err)
				require.Contains(t, err.Error(), tc.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.want, got)
		})
	}
}

func TestEnergyModelEmpty(t *testing.T) {
	t.Parallel()
	require.True(t, (*EnergyModel)(nil).Empty())
	require.True(t, (&EnergyModel{}).Empty())
	require.False(t, (&EnergyModel{IdleW: 1}).Empty())
	require.False(t, (&EnergyModel{JPerOutputToken: 0.001}).Empty())
}

func TestEnergyModelJoules(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name      string
		em        *EnergyModel
		prompt    int
		complete  int
		duration  time.Duration
		want      float64
		tolerance float64
	}{
		{
			name:   "empty model returns zero",
			em:     &EnergyModel{},
			prompt: 100, complete: 200, duration: 5 * time.Second,
			want: 0, tolerance: 0,
		},
		{
			name:   "nil model returns zero",
			em:     nil,
			prompt: 100, complete: 200, duration: 5 * time.Second,
			want: 0, tolerance: 0,
		},
		{
			name:   "dynamic only",
			em:     &EnergyModel{JPerInputToken: 0.5, JPerOutputToken: 1.2},
			prompt: 100, complete: 200, duration: 5 * time.Second,
			want: 100*0.5 + 200*1.2, tolerance: 1e-9,
		},
		{
			name:   "static only",
			em:     &EnergyModel{IdleW: 50},
			prompt: 100, complete: 200, duration: 5 * time.Second,
			want: 50 * 5.0, tolerance: 1e-9,
		},
		{
			name:   "combined",
			em:     &EnergyModel{JPerInputToken: 0.5, JPerOutputToken: 1.2, IdleW: 50},
			prompt: 100, complete: 200, duration: 5 * time.Second,
			want: 50 + 240 + 250, tolerance: 1e-9,
		},
		{
			name:   "fractional duration",
			em:     &EnergyModel{IdleW: 100},
			prompt: 0, complete: 0, duration: 1500 * time.Millisecond,
			want: 150, tolerance: 1e-9,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := tc.em.Joules(tc.prompt, tc.complete, tc.duration)
			require.InDelta(t, tc.want, got, tc.tolerance)
		})
	}
}

func TestParseCost(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		raw     any
		want    *CostModel
		wantErr string
	}{
		{
			name: "both rates",
			raw:  map[string]any{"usd_per_million_input_tokens": 0.5, "usd_per_million_output_tokens": 1.5},
			want: &CostModel{USDPerMInputTokens: 0.5, USDPerMOutputTokens: 1.5},
		},
		{
			name: "output only",
			raw:  map[string]any{"usd_per_million_output_tokens": 2.0},
			want: &CostModel{USDPerMOutputTokens: 2.0},
		},
		{name: "non-object", raw: "x", wantErr: "must be an object"},
		{name: "wrong type", raw: map[string]any{"usd_per_million_input_tokens": "free"}, wantErr: "must be a number"},
		{name: "negative rejected", raw: map[string]any{"usd_per_million_output_tokens": -1.0}, wantErr: "must be non-negative"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := parseCost(tc.raw)
			if tc.wantErr != "" {
				require.Error(t, err)
				require.Contains(t, err.Error(), tc.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.want, got)
		})
	}
}

func TestCostModelUSD(t *testing.T) {
	t.Parallel()
	require.InDelta(t, 0, (*CostModel)(nil).USD(1000, 1000), 1e-12)
	require.InDelta(t, 0, (&CostModel{}).USD(1000, 1000), 1e-12)
	c := &CostModel{USDPerMInputTokens: 0.5, USDPerMOutputTokens: 1.5}
	// 1M input → $0.50, 2M output → $3.00, total $3.50
	require.InDelta(t, 3.5, c.USD(1_000_000, 2_000_000), 1e-9)
	// Small request: 100 in, 200 out → 100 * 0.5e-6 + 200 * 1.5e-6 = 5e-5 + 3e-4 = 3.5e-4
	require.InDelta(t, 0.00035, c.USD(100, 200), 1e-12)
}

func TestParseOptionsEnergyIntegration(t *testing.T) {
	t.Parallel()
	o, err := parseOptions(map[string]any{
		"base_url": "http://x",
		"model":    "m",
		"energy":   map[string]any{"j_per_output_token": 1.5, "idle_w": 60.0},
	})
	require.NoError(t, err)
	require.NotNil(t, o.Energy)
	require.InDelta(t, 1.5, o.Energy.JPerOutputToken, 1e-9)
	require.InDelta(t, 60.0, o.Energy.IdleW, 1e-9)
}
