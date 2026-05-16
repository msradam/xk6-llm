// Smoke test for client.embed() registration. Exercises the synchronous
// surface only (no network) so it runs in CI without a live server.
import llm from 'k6/x/llm';

export default function () {
  const c = new llm.Client({ base_url: 'http://127.0.0.1:1/v1', model: 'x' });
  if (typeof c.embed !== 'function') throw new Error('client.embed missing');
}
