// Full-spectrum showcase. Each iteration is a 3-turn conversation, so a
// single ramping-arrival-rate scenario exercises:
//   - prefix-cache speedup (turn 1 cold, turns 2-3 warm)
//   - throughput under increasing concurrency
//   - SLO predicates, goodput, per-SLO Rates
//   - cost in USD per request
//   - energy in joules per request and per token
//
// Run:
//   ./build/k6 run examples/showcase.js \
//     -e LLM_BASE_URL=http://localhost:8000/v1 \
//     -e LLM_MODEL=ibm-granite/granite-4.1-30b
import llm from 'k6/x/llm';

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-arrival-rate',
      startRate: 1, timeUnit: '1s',
      preAllocatedVUs: 20, maxVUs: 60,
      stages: [
        { target: 2, duration: '20s' },
        { target: 6, duration: '40s' },
        { target: 8, duration: '30s' },
        { target: 3, duration: '20s' },
      ],
    },
  },
  thresholds: {
    llm_errors:    ['count==0'],
    'llm_ttft':    ['p(95)<3000'],
    'llm_goodput': ['rate>0.85'],
  },
};

// Realistic hosted-API pricing (Together-style mid-tier 70B-class, 2026-Q1).
const COST = {
  usd_per_million_input_tokens:  0.88,
  usd_per_million_output_tokens: 0.88,
};

// Per-token energy estimates for a Blackwell-class accelerator. Calibrate with
// nvidia-smi --query-gpu=power.draw --format=csv -lms 100 for your own setup.
const ENERGY = {
  j_per_input_token:  0.4,
  j_per_output_token: 1.0,
  idle_w: 90 / 8,
};

const client = new llm.Client({
  base_url:   __ENV.LLM_BASE_URL,
  model:      __ENV.LLM_MODEL,
  timeout_ms: 60000,
  slo:    { ttft_ms: 1500, tpot_ms: 80, e2el_ms: 12000 },
  energy: ENERGY,
  cost:   COST,
});

const CONVERSATIONS = [
  ['Explain how a hash join works in one paragraph.',
   'Walk through a small example with 3 rows.',
   'When would a nested loop join win instead?'],
  ['What is a B-tree index?',
   'Why is it the default for OLTP workloads?',
   'When does a hash index outperform it?'],
  ['Describe the role of write-ahead logging.',
   'How does group commit improve throughput?',
   'What is the tradeoff with durability guarantees?'],
  ['What is the CAP theorem in two sentences?',
   'How does PACELC refine it?',
   'Give one real system that picks AP and one that picks CP.'],
];

const SYSTEM = 'You are a senior database engineer. Answer concisely.';

// llm.Session keeps the history and stamps session_id, turn and cache_state
// (cold on turn 1, warm after) on every call, which is what the dashboard's
// per-turn panels group by.
export default async function () {
  const conv = CONVERSATIONS[__VU % CONVERSATIONS.length];
  const s = new llm.Session(client, { system: SYSTEM, id: `s-${__VU}-${__ITER}` });
  for (const q of conv) {
    const r = await s.send({ content: q, max_tokens: 160, temperature: 0 });
    if (!r.content) throw new Error(`empty turn ${s.turn()}`);
    const snippet = r.content.replace(/\s+/g, ' ').slice(0, 60);
    console.log(
      `[vu=${__VU} t=${s.turn()}] ttft=${r.ttft_ms.toFixed(0)}ms ` +
      `tpot=${r.tpot_ms.toFixed(1)}ms toks=${r.completion_tokens} ` +
      `${s.turn() === 1 ? 'COLD' : 'warm'} | "${snippet}…"`,
    );
  }
}
