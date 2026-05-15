// Parity-run driver for cross-validation against `vllm bench serve`.
//
// Both this script and `vllm bench serve` are pointed at the same idle vLLM
// server, run sequentially with the same Poisson rate, num_prompts, and seed.
// The xk6 side replays a ShareGPT-derived JSONL via llm.Dataset; the vllm
// side uses --dataset-name sharegpt with the same seed. Compare aggregate
// distributions per docs/validation-parity.md.
//
// Run:
//   ./build/k6 run test/parity.js \
//     -e LLM_BASE_URL=https://<pod>-8000.proxy.runpod.net/v1 \
//     -e LLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ \
//     -e LLM_DATASET=data/sample-prompts.jsonl \
//     --summary-export=/tmp/xk6_parity.json
import llm from 'k6/x/llm';

const RATE = parseInt(__ENV.LLM_RATE ?? '4', 10);
const NUM_PROMPTS = parseInt(__ENV.LLM_NUM_PROMPTS ?? '200', 10);
const MAX_TOKENS = parseInt(__ENV.LLM_MAX_TOKENS ?? '128', 10);

export const options = {
  discardResponseBodies: true,
  scenarios: {
    parity: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: `${Math.ceil(NUM_PROMPTS / RATE)}s`,
      preAllocatedVUs: Math.max(20, RATE * 4),
      maxVUs: Math.max(80, RATE * 20),
    },
  },
  thresholds: {
    llm_errors: ['count==0'],
  },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL,
  model: __ENV.LLM_MODEL,
  timeout_ms: 120000,
});

const dataset = new llm.Dataset({
  path: __ENV.LLM_DATASET ?? 'data/sample-prompts.jsonl',
  seed: parseInt(__ENV.LLM_SEED ?? '42', 10),
  shuffle: true,
});

console.log(`parity: ${NUM_PROMPTS} prompts @ ${RATE} rps, dataset size=${dataset.size()}, max_tokens=${MAX_TOKENS}`);

export default async function () {
  const req = dataset.next();
  await client.chat({
    ...req,
    max_tokens: MAX_TOKENS,
    temperature: 0,
    ignore_eos: true,
  });
}
