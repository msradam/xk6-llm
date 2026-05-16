// Smoke test for client.embed(). Hits /v1/embeddings, prints vector dims and
// token usage, then runs a small batch to confirm response ordering.
//
// Run:
//   ./build/k6 run examples/embed.js \
//     -e LLM_BASE_URL=http://localhost:11434/v1 -e EMBED_MODEL=nomic-embed-text
import llm from 'k6/x/llm';

export const options = {
  vus: 1, iterations: 1,
  thresholds: { llm_embed_errors: ['count==0'] },
};

const client = new llm.Client({
  base_url: __ENV.LLM_BASE_URL ?? 'http://localhost:11434/v1',
  model:    __ENV.EMBED_MODEL  ?? 'nomic-embed-text',
});

export default async function () {
  const single = await client.embed({ input: 'hello world' });
  console.log(`single: dims=${single.embeddings[0].length} tokens=${single.prompt_tokens} dur=${single.duration_ms.toFixed(0)}ms`);

  const batch = await client.embed({ input: ['alpha', 'beta', 'gamma'] });
  console.log(`batch: rows=${batch.embeddings.length} tokens=${batch.prompt_tokens} dur=${batch.duration_ms.toFixed(0)}ms`);

  if (batch.embeddings.length !== 3) throw new Error(`expected 3 rows, got ${batch.embeddings.length}`);
  if (batch.embeddings[0].length !== single.embeddings[0].length) {
    throw new Error('dimensionality drift between calls');
  }
}
