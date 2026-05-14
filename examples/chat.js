import llm from 'k6/x/llm';

// Rate is requests per LLM_TIME_UNIT (default "1s"). Use e.g. LLM_RATE=1 LLM_TIME_UNIT=2s for 0.5 rps.
const RATE = parseInt(__ENV.LLM_RATE ?? '4', 10);
const TIME_UNIT = __ENV.LLM_TIME_UNIT ?? '1s';
const DURATION = __ENV.LLM_DURATION ?? '20s';

export const options = {
  scenarios: {
    poisson: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: TIME_UNIT,
      duration: DURATION,
      preAllocatedVUs: Math.max(10, RATE * 4),
      maxVUs: Math.max(50, RATE * 20),
    },
  },
  thresholds: {
    llm_errors: ['count==0'],
  },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  api_key: __ENV.LLM_API_KEY ?? '',
  model: __ENV.LLM_MODEL ?? 'granite4.1:3b',
  timeout_ms: 60000,
});

export default async function () {
  const res = await client.chat({
    messages: [
      { role: 'system', content: 'You are a concise assistant.' },
      { role: 'user', content: 'Explain Poisson arrivals in two sentences.' },
    ],
    max_tokens: Number(__ENV.LLM_MAX_TOKENS ?? 64),
    temperature: 0,
  });
  if (!res.content) throw new Error('empty response');
}
