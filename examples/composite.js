// One iteration exercises an AI feature end to end: a REST login, a model
// call that is offered a tool, the tool executed against the application,
// the answer stored through the API, and the page that renders it opened in
// a browser. k6's http, k6/browser and k6/x/llm share the iteration, the
// tags and the clock, so `feature_e2e_ms` covers every hop and two checks
// hold that no single-protocol tool can make: the application's own count of
// lookups matches the tool calls the model made, and the answer painted.
//
// Start the mock application first, then run:
//   python3 examples/app/server.py
//   K6_BROWSER_HEADLESS=true ./build/k6 run examples/composite.js \
//     -e LLM_BASE_URL=http://localhost:11434/v1 -e LLM_MODEL=granite4.1:3b
import http from 'k6/http';
import { browser } from 'k6/browser';
import { check } from 'k6';
import { Trend } from 'k6/metrics';
import llm from 'k6/x/llm';

const APP = __ENV.APP_URL ?? 'http://127.0.0.1:8089';
const ITERATIONS = 3;

export const options = {
  scenarios: {
    feature: {
      executor: 'shared-iterations',
      vus: 1,
      iterations: ITERATIONS,
      options: { browser: { type: 'chromium' } },
    },
  },
  thresholds: {
    llm_errors:       ['count==0'],
    checks:           ['rate==1'],
    feature_e2e_ms:   ['p(95)<30000'],
    'llm_ttft':       ['p(95)<5000'],
  },
};

const featureE2E = new Trend('feature_e2e_ms', true);

const client = new llm.Client({
  base_url:   __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model:      __ENV.LLM_MODEL    ?? 'granite4.1:3b',
  timeout_ms: 60000,
});

const tools = [{
  type: 'function',
  function: {
    name: 'lookup_doc',
    description: 'Search the internal documentation. Always use this before answering a database question.',
    parameters: {
      type: 'object',
      properties: { query: { type: 'string', description: 'search terms' } },
      required: ['query'],
    },
  },
}];

const QUESTIONS = ['What is a B-tree index?', 'When is a hash index the right choice?', 'What is a write-ahead log for?'];

export function setup() {
  http.post(`${APP}/reset`);
}

export default async function () {
  const t0 = Date.now();

  // The application's own count of lookups, before the model gets to act.
  const before = http.get(`${APP}/stats`, { tags: { hop: 'stats' } }).json().lookups;

  // 1. REST: the session the feature needs.
  const login = http.post(`${APP}/login`, null, { tags: { hop: 'login' } });
  check(login, { 'login: session issued': (r) => r.status === 204 && r.cookies.session !== undefined });

  // 2. Model: offered the tool, asked a question it needs the tool for.
  const messages = [
    { role: 'system', content: 'You answer database questions. Call lookup_doc first, then answer in one sentence from what it returns.' },
    { role: 'user',   content: QUESTIONS[__ITER % QUESTIONS.length] },
  ];
  const decide = await client.chat({ messages, tools, max_tokens: 200, temperature: 0, tags: { hop: 'decide' } });
  check(decide, { 'model: asked for the tool': (r) => r.tool_calls.length > 0 });

  // 3. Tool: executed against the application, which records the call.
  messages.push({
    role: 'assistant', content: decide.content || null,
    tool_calls: decide.tool_calls.map((tc) => ({ id: tc.id, type: 'function', function: { name: tc.name, arguments: tc.arguments } })),
  });
  for (const tc of decide.tool_calls) {
    const args = JSON.parse(tc.arguments || '{}');
    const res = http.get(`${APP}/docs/search?q=${encodeURIComponent(args.query ?? '')}`, { tags: { hop: 'lookup' } });
    check(res, { 'tool: lookup answered': (r) => r.status === 200 });
    messages.push({ role: 'tool', tool_call_id: tc.id, content: res.body });
  }

  // 4. Model again: the answer, now grounded in the lookup.
  const answer = await client.chat({ messages, max_tokens: 160, temperature: 0, tags: { hop: 'answer' } });
  check(answer, { 'model: answered': (r) => r.content.trim().length > 0 });

  // 5. REST: the answer stored where the page reads it.
  const stored = http.post(`${APP}/answers`, JSON.stringify({ answer: answer.content }),
    { headers: { 'Content-Type': 'application/json' }, tags: { hop: 'store' } });
  check(stored, { 'api: answer stored': (r) => r.status === 200 });

  // 6. Browser: the page paints the answer the model gave.
  const page = await browser.newPage();
  try {
    await page.goto(`${APP}/`);
    const painted = await page.locator('#answer').textContent();
    check(painted, { 'page: answer painted': (t) => t.trim() === answer.content.trim() });
  } finally {
    await page.close();
  }

  // 7. The application's own record, not the model's account of itself.
  const stats = http.get(`${APP}/stats`, { tags: { hop: 'stats' } }).json();
  check(stats, { 'app: lookups match tool calls': (s) => s.lookups - before === decide.tool_calls.length });

  featureE2E.add(Date.now() - t0);
}
