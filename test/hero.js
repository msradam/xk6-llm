// Hero workload for screenshots: realistic mix of short and long completions
// against a production-class server, with cost + energy + SLO predicates wired
// up so every dashboard panel lights up.
import llm from 'k6/x/llm';

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-arrival-rate',
      startRate: 1,
      timeUnit: '1s',
      preAllocatedVUs: 30,
      maxVUs: 80,
      stages: [
        { target: 4,  duration: '20s' },
        { target: 8,  duration: '30s' },
        { target: 12, duration: '40s' },
        { target: 4,  duration: '20s' },
      ],
    },
  },
  thresholds: {
    llm_errors: ['count==0'],
    'llm_ttft': ['p(95)<2000'],
    'llm_goodput': ['rate>0.95'],
  },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL,
  model:    __ENV.LLM_MODEL,
  timeout_ms: 60000,
  slo: { ttft_ms: 1500, tpot_ms: 80, e2el_ms: 10000 },
  // 70B AWQ on A100 — calibrated rough estimates
  energy: { j_per_input_token: 0.4, j_per_output_token: 1.0, idle_w: 50 / 8 },
  // Pretend hosted-API price for cost dashboard demo
  cost:   { usd_per_million_input_tokens: 0.3, usd_per_million_output_tokens: 0.9 },
});

const dataset = new llm.Dataset({
  path: 'data/sample-prompts.jsonl',
  seed: 42,
  shuffle: true,
});

export default async function () {
  const req = dataset.next();
  await client.chat({ ...req, temperature: 0 });
}
