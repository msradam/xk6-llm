// Multi-turn conversation simulation.
//
// Real chatbot sessions are 3 to 12 turns long, each carrying the prior turn's
// state. Turn 1 hits cold KV cache; turn N typically hits warm prefix cache for
// most of its input. Single-shot benchmarks miss this entirely.
//
// This example models a 5-turn debugging session and tags every chat() call
// with session_id, turn, and cache_state. The dashboard can then aggregate by
// session_id (total tokens per session, p95 session duration) and slice by
// turn (TTFT degradation across turns).
//
// Run:
//   ./build/k6 run examples/multi-turn.js \
//     -e LLM_BASE_URL=http://localhost:11434/v1 -e LLM_MODEL=granite4.1:3b
import llm from 'k6/x/llm';

export const options = {
  scenarios: {
    sessions: {
      executor: 'constant-arrival-rate',
      rate: 1, timeUnit: '1s', duration: '60s',
      preAllocatedVUs: 8, maxVUs: 30,
    },
  },
  thresholds: {
    llm_errors: ['count==0'],
    'llm_ttft{turn:1}':  ['p(95)<2000'],
    'llm_ttft{turn:5}':  ['p(95)<1000'],   // warm prefix cache should be faster
  },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.LLM_MODEL    ?? 'granite4.1:3b',
  timeout_ms: 60000,
  slo: { ttft_ms: 1500, tpot_ms: 80, e2el_ms: 15000 },
});

// Session keeps conversation state per VU. The Go-side Client knows nothing
// about it; this is pure JS that any user can copy and modify.
class Session {
  constructor(systemPrompt) {
    this.id = `s-${__VU}-${__ITER}`;
    this.messages = [{ role: 'system', content: systemPrompt }];
    this.turn = 0;
    this.tokens = 0;
  }

  async send(userText) {
    this.turn++;
    this.messages.push({ role: 'user', content: userText });
    const res = await client.chat({
      messages:    this.messages,
      max_tokens:  256,
      temperature: 0,
      cache_state: this.turn === 1 ? 'cold' : 'warm',
      tags: {
        session_id: this.id,
        turn:       String(this.turn),
      },
    });
    this.messages.push({ role: 'assistant', content: res.content });
    this.tokens += res.completion_tokens;
    return res;
  }
}

const PROMPTS = [
  'My query joining 3 tables (10M rows each) takes 45 seconds. Where do I start investigating?',
  'EXPLAIN ANALYZE shows a hash join on the largest table is the slowest step.',
  "It's joining on a non-indexed timestamp column. Should I add an index?",
  'The column is updated frequently. What is the write-amplification tradeoff?',
  'Last question: BRIN or BTREE for time-series data with 100M rows?',
];

export default async function () {
  const s = new Session('You are a senior SRE helping debug Postgres performance.');
  for (const prompt of PROMPTS) {
    const r = await s.send(prompt);
    if (!r.content) throw new Error(`empty turn ${s.turn}`);
  }
}
