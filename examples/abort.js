// Mid-stream cancellation. Real chat UIs abort generation when the user
// clicks stop or navigates away, but the server keeps producing tokens until
// it sees the TCP close. This example stresses both abort paths so you can
// load-test the cancellation handling on your inference server.
//
// abort_after_ms     — wall-clock cap (UI gives up while waiting).
// abort_after_tokens — content-cap (UI stopped reading after N tokens).
//
// Run:
//   ./build/k6 run examples/abort.js \
//     -e LLM_BASE_URL=http://localhost:11434/v1 -e LLM_MODEL=granite4.1:3b
import llm from 'k6/x/llm';

export const options = {
  vus: 1, iterations: 4,
  thresholds: {
    llm_errors:  ['count==0'],
    llm_aborted: ['count>=3'],
  },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.LLM_MODEL    ?? 'granite4.1:3b',
  timeout_ms: 30000,
});

const TASK = { role: 'user', content: 'Write a 300-word essay about CAP theorem.' };

export default async function () {
  if (__ITER === 0) {
    const r = await client.chat({
      messages: [TASK], max_tokens: 500, temperature: 0,
      tags: { variant: 'unbounded' },
    });
    console.log(`unbounded: aborted=${r.aborted} chunks=${r.chunks} tok=${r.completion_tokens} dur=${r.duration_ms.toFixed(0)}ms`);
    return;
  }

  if (__ITER === 1) {
    const r = await client.chat({
      messages: [TASK], max_tokens: 500, temperature: 0,
      abort_after_tokens: 20,
      tags: { variant: 'token-cap' },
    });
    console.log(`token-cap: aborted=${r.aborted} chunks=${r.chunks} dur=${r.duration_ms.toFixed(0)}ms | "${r.content.slice(0, 60)}…"`);
    if (!r.aborted) throw new Error('expected token-cap abort');
    if (r.chunks > 20) throw new Error(`expected <=20 chunks, got ${r.chunks}`);
    return;
  }

  if (__ITER === 2) {
    const r = await client.chat({
      messages: [TASK], max_tokens: 500, temperature: 0,
      abort_after_ms: 200,
      tags: { variant: 'time-cap' },
    });
    console.log(`time-cap: aborted=${r.aborted} chunks=${r.chunks} dur=${r.duration_ms.toFixed(0)}ms`);
    if (!r.aborted) throw new Error('expected time-cap abort');
    return;
  }

  // Both limits set; whichever trips first wins.
  const r = await client.chat({
    messages: [TASK], max_tokens: 500, temperature: 0,
    abort_after_tokens: 5,
    abort_after_ms: 5000,
    tags: { variant: 'both' },
  });
  console.log(`both: aborted=${r.aborted} chunks=${r.chunks} dur=${r.duration_ms.toFixed(0)}ms`);
  if (!r.aborted) throw new Error('expected abort');
}
