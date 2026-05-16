// RAG simulation: embed query -> vector retrieve -> LLM generate, with each
// phase measured against its own SLO. The embed step uses client.embed()
// against the same OpenAI-compatible server, so token counts and latency
// flow into llm_embed_* metrics and show up on the dashboard.
//
// Set EMBED_URL/EMBED_MODEL to point at a separate embedding server when the
// generation model server doesn't host an /embeddings endpoint. Set
// RETRIEVE_URL to a real vector DB (Qdrant, Pinecone, pgvector) for retrieval
// numbers that mean something.
//
// Run:
//   ./build/k6 run examples/rag.js \
//     -e LLM_BASE_URL=http://localhost:11434/v1 -e LLM_MODEL=granite4.1:3b \
//     -e EMBED_MODEL=nomic-embed-text
import http from 'k6/http';
import { Trend } from 'k6/metrics';
import llm from 'k6/x/llm';

const ragRetrieve = new Trend('rag_retrieve_ms', true);

export const options = {
  vus: 4, duration: '30s',
  thresholds: {
    llm_errors:        ['count==0'],
    llm_embed_errors:  ['count==0'],
    llm_embed_duration:['p(95)<500'],
    rag_retrieve_ms:   ['p(95)<150'],
    llm_ttft:          ['p(95)<3000'],
  },
};

const generator = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.LLM_MODEL    ?? 'granite4.1:3b',
});

// Same base_url by default since Ollama hosts both endpoints. Point at a
// dedicated embedding server (TEI, vLLM-emb, Infinity) for production.
const embedder = new llm.Client({
  base_url: __ENV.EMBED_URL    ?? __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.EMBED_MODEL  ?? 'nomic-embed-text',
});

const RETRIEVE_URL = __ENV.RETRIEVE_URL ?? 'https://httpbin.org/anything/retrieve';

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

  const emb = await embedder.embed({ input: q, tags: { workflow: 'rag' } });
  const chunks = retrieve(emb.embeddings[0], 3);

  const prompt = `Context:\n- ${chunks.join('\n- ')}\n\nQuestion: ${q}`;
  const res = await generator.chat({
    messages:    [{ role: 'user', content: prompt }],
    max_tokens:  120,
    temperature: 0,
    tags:        { workflow: 'rag' },
  });
  if (!res.content) throw new Error('empty');
}
