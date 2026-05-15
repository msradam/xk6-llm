// Replay a JSONL prompt corpus through an OpenAI-compatible server.
//
// The dataset is loaded once per process (cached by absolute path), so even
// at high VU counts there is one parse pass per file. Per-VU iteration order
// is determined by the seeded shuffle, making runs reproducible.
//
// Run:
//   ./build/k6 run examples/dataset.js \
//     -e LLM_BASE_URL=http://localhost:8000/v1 \
//     -e LLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ \
//     -e LLM_DATASET=examples/data/sample-prompts.jsonl
import llm from 'k6/x/llm';

const RATE = parseInt(__ENV.LLM_RATE ?? '4', 10);
const DURATION = __ENV.LLM_DURATION ?? '30s';

export const options = {
  scenarios: {
    poisson_replay: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: DURATION,
      preAllocatedVUs: Math.max(10, RATE * 4),
      maxVUs: Math.max(50, RATE * 20),
    },
  },
  thresholds: {
    llm_errors: ['count==0'],
    llm_ttft:   ['p(95)<5000'],
  },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  api_key:  __ENV.LLM_API_KEY  ?? '',
  model:    __ENV.LLM_MODEL    ?? 'granite4.1:3b',
  timeout_ms: 120000,
  slo: { ttft_ms: 1500, tpot_ms: 80, e2el_ms: 30000 },
});

const dataset = new llm.Dataset({
  path: __ENV.LLM_DATASET ?? 'examples/data/sample-prompts.jsonl',
  seed: parseInt(__ENV.LLM_SEED ?? '42', 10),
  shuffle: true,
});

console.log(`loaded ${dataset.size()} prompts from ${__ENV.LLM_DATASET ?? 'examples/data/sample-prompts.jsonl'}`);

export default async function () {
  const req = dataset.next();
  const res = await client.chat({
    ...req,
    temperature: 0,
  });
  if (!res.content) throw new Error('empty content');
}
