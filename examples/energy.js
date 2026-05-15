// Per-request energy estimation. Coefficients shown are illustrative for a
// 70B-class AWQ model on a single A100 80GB; calibrate with `nvidia-smi
// --query-gpu=power.draw --format=csv -lms 100` for your own setup.
//
// Run:
//   ./build/k6 run examples/energy.js \
//     -e LLM_BASE_URL=http://localhost:8000/v1 \
//     -e LLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ
import llm from 'k6/x/llm';

export const options = {
  vus: 4,
  duration: '20s',
  thresholds: {
    llm_errors: ['count==0'],
    'llm_energy_j_per_token': ['p(95)<3.0'],
  },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.LLM_MODEL    ?? 'granite4.1:3b',
  energy: {
    j_per_input_token:  0.5,
    j_per_output_token: 1.2,
    // 50 W baseline divided by VU count = per-request idle attribution under
    // steady-state concurrency.
    idle_w: 50 / 4,
  },
});

export default async function () {
  const res = await client.chat({
    messages:   [{ role: 'user', content: 'Summarize the CAP theorem in two sentences.' }],
    max_tokens: 96,
    temperature: 0,
  });
  if (!res.content) throw new Error('empty response');
}
