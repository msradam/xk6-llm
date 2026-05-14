// Quick-start script. Build with: xk6 build --with github.com/msradam/xk6-llm=.
// Run with: ./k6 run script.js
import llm from 'k6/x/llm';

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model: __ENV.LLM_MODEL ?? 'granite4.1:3b',
});

export default async function () {
  const res = await client.chat({
    messages: [{ role: 'user', content: 'Say hi.' }],
    max_tokens: 32,
  });
  console.log(`ttft=${res.ttft_ms}ms duration=${res.duration_ms}ms tokens=${res.completion_tokens}`);
}
