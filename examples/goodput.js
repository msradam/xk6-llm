// Demonstrates the goodput + per-SLO attainment metric set against an
// OpenAI-compatible server. Defaults target local Ollama with granite4.1:3b.
//
// Run:
//   ./build/k6 run examples/goodput.js
import llm from 'k6/x/llm';

export const options = {
  vus: 1,
  iterations: 5,
  thresholds: {
    llm_errors: ['count==0'],
  },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.LLM_MODEL    ?? 'granite4.1:3b',
  // Generous SLOs so the local CPU/MPS server actually passes them.
  slo: {
    ttft_ms: 500,
    tpot_ms: 100,
    e2el_ms: 10000,
  },
});

export default async function () {
  const res = await client.chat({
    messages:   [{ role: 'user', content: 'In one short sentence: what is Poisson arrival?' }],
    max_tokens: 48,
    temperature: 0,
    cache_state: __ITER === 0 ? 'cold' : 'warm',
    tags: { prompt: 'poisson-1s' },
  });
  console.log(
    `iter=${__ITER} cache=${__ITER === 0 ? 'cold' : 'warm'} ` +
    `ttft=${res.ttft_ms.toFixed(0)}ms tpot=${res.tpot_ms.toFixed(1)}ms ` +
    `dur=${res.duration_ms.toFixed(0)}ms toks=${res.completion_tokens} chunks=${res.chunks}`,
  );
}
