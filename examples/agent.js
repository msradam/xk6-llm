// Agent loop with native OpenAI tool-calling. The model decides which tool to
// invoke, we execute it in JS, append the result as a `role: "tool"` message,
// and loop until the model produces final content or we hit max_iterations.
// The metric of interest is the FULL envelope: prompt-to-final answer latency,
// total tokens consumed, total cost, total tool calls.
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

// Tool catalogue. In production these would call real services with their own
// latency budgets; here they return canned values so the test stays hermetic.
const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'search',
      description: 'Web search. Returns top results.',
      parameters: {
        type: 'object',
        properties: { query: { type: 'string' } },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'calc',
      description: 'Evaluate a JavaScript arithmetic expression.',
      parameters: {
        type: 'object',
        properties: { expr: { type: 'string' } },
        required: ['expr'],
      },
    },
  },
];

const FNS = {
  search: ({ query }) =>
    `Top results for "${query}": [1] CAP theorem (Wikipedia), [2] PACELC extension, [3] Brewer 12-year retrospective.`,
  calc: ({ expr }) => {
    try { return String(Function(`"use strict"; return (${expr})`)()); }
    catch (e) { return `error: ${e.message}`; }
  },
};

async function runAgent(userQuery) {
  const agentId = `a-${__VU}-${__ITER}`;
  const messages = [
    { role: 'system', content: 'Use the provided tools to answer the user. Stop when you have a final answer.' },
    { role: 'user',   content: userQuery },
  ];
  let toolCalls = 0;

  for (let i = 1; i <= MAX_ITERATIONS; i++) {
    const res = await client.chat({
      messages, tools: TOOLS,
      max_tokens: 200, temperature: 0,
      tags: { agent: 'true', agent_id: agentId, iteration: String(i) },
    });

    if (!res.tool_calls.length) {
      return { iterations: i, toolCalls, answer: res.content };
    }

    messages.push({
      role: 'assistant',
      content: res.content || null,
      tool_calls: res.tool_calls.map((tc) => ({
        id: tc.id, type: 'function',
        function: { name: tc.name, arguments: tc.arguments },
      })),
    });

    for (const tc of res.tool_calls) {
      const fn = FNS[tc.name];
      const args = JSON.parse(tc.arguments || '{}');
      const out = fn ? fn(args) : `error: unknown tool ${tc.name}`;
      messages.push({ role: 'tool', tool_call_id: tc.id, content: String(out) });
      toolCalls++;
    }
  }
  return { iterations: MAX_ITERATIONS, toolCalls, answer: '(timed out)' };
}

const QUERIES = [
  'What does the CAP theorem say in one sentence? Use search if you need it.',
  'Compute 40 * 1.5 + 12 using the calc tool.',
  'Search for the PACELC extension and summarize what it adds.',
];

export default async function () {
  const q = QUERIES[__ITER % QUERIES.length];
  const out = await runAgent(q);
  if (!out.answer) throw new Error('no answer');
}
