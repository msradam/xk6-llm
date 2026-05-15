// A/B compare two providers on identical traffic. Two scenarios run in
// parallel against two different Clients; every metric is tagged with
// `provider`, so the Grafana dashboard shows them side by side and the cost
// panel becomes a literal procurement decision.
//
// This example points at two Ollama models. Swap base_url / api_key to compare
// OpenAI vs Anthropic vs your self-hosted vLLM, etc.
//
// Run:
//   ./build/k6 run examples/ab-providers.js
import llm from 'k6/x/llm';

export const options = {
  scenarios: {
    a: {
      executor: 'constant-arrival-rate', rate: 2, timeUnit: '1s', duration: '30s',
      preAllocatedVUs: 8, maxVUs: 30, exec: 'runA', tags: { provider: 'a' },
    },
    b: {
      executor: 'constant-arrival-rate', rate: 2, timeUnit: '1s', duration: '30s',
      preAllocatedVUs: 8, maxVUs: 30, exec: 'runB', tags: { provider: 'b' },
    },
  },
  thresholds: {
    llm_errors: ['count==0'],
  },
};

const clientA = new llm.Client({
  base_url: __ENV.A_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.A_MODEL    ?? 'qwen2.5:0.5b',
  cost: { usd_per_million_input_tokens: 0.10, usd_per_million_output_tokens: 0.30 },
});

const clientB = new llm.Client({
  base_url: __ENV.B_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.B_MODEL    ?? 'granite4.1:3b',
  cost: { usd_per_million_input_tokens: 0.20, usd_per_million_output_tokens: 0.60 },
});

const PROMPT = [{ role: 'user', content: 'In one sentence: what is a hash join?' }];

async function callWith(client) {
  const res = await client.chat({ messages: PROMPT, max_tokens: 80, temperature: 0 });
  if (!res.content) throw new Error('empty');
}

export async function runA() { await callWith(clientA); }
export async function runB() { await callWith(clientB); }
