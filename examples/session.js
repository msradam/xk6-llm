// Session API smoke test. The Go-side Session auto-accumulates conversation
// history and stamps session_id, turn, and cache_state tags on every chat()
// call so dashboards can roll up per-session totals and slice TTFT by turn.
//
// Run:
//   ./build/k6 run examples/session.js \
//     -e LLM_BASE_URL=http://localhost:11434/v1 -e LLM_MODEL=granite4.1:3b
import llm from 'k6/x/llm';

export const options = {
  vus: 1,
  iterations: 1,
  thresholds: { llm_errors: ['count==0'] },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.LLM_MODEL    ?? 'granite4.1:3b',
  timeout_ms: 60000,
});

const PROMPTS = [
  'What is a B-tree index, in one sentence?',
  'And how does it differ from a hash index?',
  'Which one would you choose for range queries?',
];

export default async function () {
  const s = new llm.Session(client, { system: 'You are terse.' });

  for (const p of PROMPTS) {
    const r = await s.send({ content: p, max_tokens: 80, temperature: 0 });
    const snippet = r.content.slice(0, 60).replace(/\n/g, ' ');
    console.log(
      `[${s.id()} turn=${s.turn()}] ttft=${r.ttft_ms.toFixed(0)}ms ` +
      `tok=${r.completion_tokens} | "${snippet}…"`,
    );
  }

  const tok = s.tokens();
  console.log(`session total: prompt=${tok.prompt} completion=${tok.completion}`);

  if (s.messages().length !== 1 + PROMPTS.length * 2) {
    throw new Error(`history len: got ${s.messages().length}`);
  }
}
