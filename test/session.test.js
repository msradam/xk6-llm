// Smoke test for the registered Session constructor. Runs under `xk6 test`
// via the validate workflow and exercises the synchronous surface only (no
// network call), so it works in CI without a live LLM server.
import llm from 'k6/x/llm';

export default function () {
  if (typeof llm.Session !== 'function') {
    throw new Error('llm.Session constructor missing');
  }
  const client = new llm.Client({ base_url: 'http://127.0.0.1:1/v1', model: 'x' });
  const s = new llm.Session(client, { system: 'be terse', id: 's-fixed' });

  if (s.id() !== 's-fixed') throw new Error('id: ' + s.id());
  if (s.turn() !== 0) throw new Error('turn: ' + s.turn());

  const msgs = s.messages();
  if (msgs.length !== 1 || msgs[0].role !== 'system') {
    throw new Error('expected system message only, got ' + JSON.stringify(msgs));
  }

  const tok = s.tokens();
  if (tok.prompt !== 0 || tok.completion !== 0 || tok.total !== 0) {
    throw new Error('expected zero tokens, got ' + JSON.stringify(tok));
  }

  s.reset();
  if (s.turn() !== 0 || s.messages().length !== 1) {
    throw new Error('reset failed');
  }

  let threw = false;
  try { new llm.Session(); } catch (_) { threw = true; }
  if (!threw) throw new Error('expected throw when client missing');
}
