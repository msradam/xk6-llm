// RAG simulation: embed query -> vector retrieve -> LLM generate, with each
// phase measured against its own SLO. Emits k6-native Trends for embed and
// retrieve so the dashboard can show the full envelope.
//
// Set EMBED_URL and RETRIEVE_URL to real endpoints, or leave unset to use the
// stub URLs (which return immediately). The point is the instrumentation
// pattern, which works the same against a real embeddings server (OpenAI,
// vLLM, TEI) and a real vector DB (Qdrant, Pinecone, pgvector).
//
// Run:
//   ./build/k6 run examples/rag.js \
//     -e LLM_BASE_URL=http://localhost:11434/v1 -e LLM_MODEL=granite4.1:3b
import http from 'k6/http';
import { Trend } from 'k6/metrics';
import llm from 'k6/x/llm';

const ragEmbed    = new Trend('rag_embed_ms', true);
const ragRetrieve = new Trend('rag_retrieve_ms', true);

export const options = {
  vus: 4, duration: '30s',
  thresholds: {
    llm_errors: ['count==0'],
    rag_embed_ms:    ['p(95)<200'],
    rag_retrieve_ms: ['p(95)<150'],
    llm_ttft:        ['p(95)<2000'],
  },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.LLM_MODEL    ?? 'granite4.1:3b',
});

const EMBED_URL    = __ENV.EMBED_URL    ?? 'https://httpbin.org/anything/embed';
const RETRIEVE_URL = __ENV.RETRIEVE_URL ?? 'https://httpbin.org/anything/retrieve';

function embed(text) {
  const t0 = Date.now();
  http.post(EMBED_URL, JSON.stringify({ input: text }), { headers: { 'Content-Type': 'application/json' } });
  ragEmbed.add(Date.now() - t0);
  return [/* fake embedding */];
}

function retrieve(_vector, k) {
  const t0 = Date.now();
  http.post(RETRIEVE_URL, JSON.stringify({ k }), { headers: { 'Content-Type': 'application/json' } });
  ragRetrieve.add(Date.now() - t0);
  return [
    'CAP theorem: a distributed system can satisfy at most two of consistency, availability, partition tolerance.',
    'PACELC extension: when there is no Partition, choose between Latency and Consistency.',
    'Brewer revisited the theorem in 2012 to clarify CAP is a coarse model.',
  ].slice(0, k);
}

const QUERIES = [
  'Summarize the CAP theorem in two sentences.',
  'What does PACELC add over CAP?',
  'Why did Brewer revisit CAP in 2012?',
];

export default async function () {
  const q = QUERIES[__ITER % QUERIES.length];
  const vec    = embed(q);
  const chunks = retrieve(vec, 3);
  const prompt = `Context:\n- ${chunks.join('\n- ')}\n\nQuestion: ${q}`;
  const res = await client.chat({
    messages:    [{ role: 'user', content: prompt }],
    max_tokens:  120,
    temperature: 0,
    tags:        { workflow: 'rag' },
  });
  if (!res.content) throw new Error('empty');
}
