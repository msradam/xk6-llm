// Native OpenAI tool-calling. Pass `tools` to chat(); the model responds with
// `tool_calls`, the script executes them in JS, appends `role: "tool"`
// messages, and calls chat() again to get the final answer.
//
// The Session helper isn't used here because tool-calling needs to interleave
// custom messages (tool results); using Client directly keeps that clear.
//
// Run:
//   ./build/k6 run examples/tools.js \
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

const tools = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get current weather in a city.',
      parameters: {
        type: 'object',
        properties: { city: { type: 'string' } },
        required: ['city'],
      },
    },
  },
];

const fns = {
  get_weather: ({ city }) => ({ city, temp_f: 68, conditions: 'cloudy' }),
};

export default async function () {
  const messages = [
    { role: 'system', content: 'Use tools when asked about the weather.' },
    { role: 'user',   content: 'Whats the weather in Boston?' },
  ];

  const r1 = await client.chat({
    messages, tools, max_tokens: 256, temperature: 0,
    tags: { phase: 'decide' },
  });
  console.log(
    `decide: ttft=${r1.ttft_ms.toFixed(0)}ms tool_calls=${r1.tool_calls.length} ` +
    `finish=${r1.finish_reason}`,
  );

  if (!r1.tool_calls.length) {
    throw new Error(`model did not call a tool; content=${r1.content}`);
  }

  messages.push({
    role: 'assistant',
    content: r1.content || null,
    tool_calls: r1.tool_calls.map((tc) => ({
      id: tc.id, type: 'function',
      function: { name: tc.name, arguments: tc.arguments },
    })),
  });

  for (const tc of r1.tool_calls) {
    const fn = fns[tc.name];
    if (!fn) throw new Error(`no JS impl for tool ${tc.name}`);
    const args = JSON.parse(tc.arguments || '{}');
    const out = fn(args);
    console.log(`  exec ${tc.name}(${tc.arguments}) -> ${JSON.stringify(out)}`);
    messages.push({
      role: 'tool', tool_call_id: tc.id, content: JSON.stringify(out),
    });
  }

  const r2 = await client.chat({
    messages, max_tokens: 128, temperature: 0,
    tags: { phase: 'answer' },
  });
  console.log(`answer: ttft=${r2.ttft_ms.toFixed(0)}ms tok=${r2.completion_tokens} | "${r2.content}"`);
}
