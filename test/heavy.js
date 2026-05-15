// Heavy-load validation: proves xk6-llm produces sensible TTFT / ITL / TPOT /
// goodput numbers under realistic concurrency against a real vLLM server.
//
// Scenarios run sequentially via startTime so results don't interfere:
//   1. warmup       — 3 iters, 1 VU, primes prefix cache
//   2. concurrency  — 1 -> 4 -> 16 VU constant-arrival sweeps
//   3. poisson      — constant-arrival-rate 6 rps, 30s
//   4. long_gen     — 4 VUs, 512 max_tokens, exercises long ITL streams
//
// Run:
//   ./build/k6 run test/heavy.js \
//     -e LLM_BASE_URL=https://<pod>-8000.proxy.runpod.net/v1 \
//     -e LLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ
import llm from 'k6/x/llm';

const BASE = __ENV.LLM_BASE_URL;
const MODEL = __ENV.LLM_MODEL;

export const options = {
  discardResponseBodies: true,
  scenarios: {
    warmup: {
      executor: 'per-vu-iterations',
      vus: 1, iterations: 3, maxDuration: '30s',
      exec: 'short', startTime: '0s',
      tags: { phase: 'warmup' },
    },
    conc_1: {
      executor: 'constant-vus',
      vus: 1, duration: '20s',
      exec: 'short', startTime: '35s',
      tags: { phase: 'conc_1' },
    },
    conc_4: {
      executor: 'constant-vus',
      vus: 4, duration: '20s',
      exec: 'short', startTime: '60s',
      tags: { phase: 'conc_4' },
    },
    conc_16: {
      executor: 'constant-vus',
      vus: 16, duration: '20s',
      exec: 'short', startTime: '85s',
      tags: { phase: 'conc_16' },
    },
    poisson: {
      executor: 'constant-arrival-rate',
      rate: 6, timeUnit: '1s', duration: '30s',
      preAllocatedVUs: 24, maxVUs: 64,
      exec: 'short', startTime: '110s',
      tags: { phase: 'poisson' },
    },
    long_gen: {
      executor: 'constant-vus',
      vus: 4, duration: '30s',
      exec: 'long', startTime: '145s',
      tags: { phase: 'long_gen' },
    },
  },
  thresholds: {
    llm_errors: ['count==0'],
    'llm_ttft{phase:conc_1}':    ['p(95)<2000'],
    'llm_ttft{phase:conc_16}':   ['p(95)<8000'],
    'llm_request_duration{phase:long_gen}': ['p(95)<30000'],
  },
};

const client = new llm.Client({
  base_url: BASE,
  model: MODEL,
  timeout_ms: 60000,
  slo: { ttft_ms: 1500, tpot_ms: 80, e2el_ms: 15000 },
});

export async function short() {
  const res = await client.chat({
    messages: [
      { role: 'system', content: 'You are a concise assistant.' },
      { role: 'user', content: 'In one sentence, what is a Poisson process?' },
    ],
    max_tokens: 64,
    temperature: 0,
    cache_state: __ITER === 0 ? 'cold' : 'warm',
  });
  if (!res.content) throw new Error('empty content');
  if (res.completion_tokens <= 0) throw new Error('no completion_tokens from server');
  if (res.chunks < 2) throw new Error(`suspiciously few chunks: ${res.chunks}`);
}

export async function long() {
  const res = await client.chat({
    messages: [
      { role: 'user', content: 'Write a detailed explanation of how token streaming works in LLM inference servers, covering KV cache, scheduler, and SSE chunk emission.' },
    ],
    max_tokens: 512,
    temperature: 0,
  });
  if (!res.content) throw new Error('empty content');
  if (res.itl_ms.length < 10) throw new Error(`expected long ITL stream, got ${res.itl_ms.length}`);
}
