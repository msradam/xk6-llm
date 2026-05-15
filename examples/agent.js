// Agent loop simulation: model decides to call a tool, we run the tool, feed
// the result back, repeat until the model produces a final answer or hits
// max_iterations. The metric of interest is the FULL envelope: prompt-to-final
// answer latency, total tokens consumed, total cost.
//
// To stay model-agnostic, the example uses a prompted convention: the model
// emits `TOOL: name(arg)` to call a tool, or `DONE: <answer>` to stop. Real
// production uses provider-native function calling (OpenAI tools, Anthropic
// tool_use) with the same instrumentation pattern.
//
// Run:
//   ./build/k6 run examples/agent.js \
//     -e LLM_BASE_URL=http://localhost:11434/v1 -e LLM_MODEL=granite4.1:3b
import llm from 'k6/x/llm';

const MAX_ITERATIONS = 4;

export const options = {
  vus: 4, duration: '30s',
  thresholds: {
    llm_errors: ['count==0'],
    'llm_request_duration{agent:true}': ['p(95)<8000'],
  },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.LLM_MODEL    ?? 'granite4.1:3b',
  timeout_ms: 30000,
});

// Fake tool catalogue. In production these would call your real services and
// have their own latency budgets.
const TOOLS = {
  search: (q) => `Top results for "${q}": [1] CAP theorem (Wikipedia), [2] PACELC extension, [3] Brewer's 12-year retrospective.`,
  calc:   (expr) => {
    try { return String(Function(`"use strict"; return (${expr})`)()); }
    catch (e) { return `error: ${e.message}`; }
  },
};

const SYSTEM = `You are a research agent. You may call tools by emitting:
TOOL: search(query)   -> a web search
TOOL: calc(expr)      -> arithmetic, e.g. calc(40*1.5)
When you have the final answer, emit:
DONE: <one-sentence answer>
Emit exactly one TOOL or DONE per turn. No prose around it.`;

function parseAction(text) {
  const t = text.trim();
  let m = t.match(/^TOOL:\s*(\w+)\s*\((.*)\)\s*$/s);
  if (m) return { kind: 'tool', name: m[1], arg: m[2] };
  m = t.match(/^DONE:\s*(.+)$/s);
  if (m) return { kind: 'done', answer: m[1].trim() };
  return { kind: 'done', answer: t.slice(0, 200) };  // fallback: treat as final
}

async function runAgent(userQuery) {
  const agentId = `a-${__VU}-${__ITER}`;
  const messages = [
    { role: 'system', content: SYSTEM },
    { role: 'user',   content: userQuery },
  ];
  let toolCalls = 0;

  for (let i = 1; i <= MAX_ITERATIONS; i++) {
    const res = await client.chat({
      messages,
      max_tokens:  120,
      temperature: 0,
      tags: { agent: 'true', agent_id: agentId, iteration: String(i) },
    });
    messages.push({ role: 'assistant', content: res.content });
    const action = parseAction(res.content);
    if (action.kind === 'done') return { iterations: i, toolCalls, answer: action.answer };
    const tool = TOOLS[action.name];
    const result = tool ? tool(action.arg) : `error: unknown tool ${action.name}`;
    messages.push({ role: 'user', content: `TOOL_RESULT: ${result}` });
    toolCalls++;
  }
  return { iterations: MAX_ITERATIONS, toolCalls, answer: '(timed out)' };
}

const QUERIES = [
  'What does the CAP theorem say in one sentence?',
  'Compute 40 * 1.5 + 12.',
  'Search for the PACELC extension and summarize what it adds.',
];

export default async function () {
  const q = QUERIES[__ITER % QUERIES.length];
  const out = await runAgent(q);
  if (!out.answer) throw new Error('no answer');
}
