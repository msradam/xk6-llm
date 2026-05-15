// Smoke test for the registered Dataset constructor. Runs under `xk6 test`
// via the validate workflow.
import llm from 'k6/x/llm';

export default function () {
  if (typeof llm.Dataset !== 'function') {
    throw new Error('llm.Dataset constructor missing');
  }
  const ds = new llm.Dataset({ path: 'examples/data/sample-prompts.jsonl' });
  if (typeof ds.size !== 'function') throw new Error('dataset.size missing');
  if (typeof ds.next !== 'function') throw new Error('dataset.next missing');
  if (typeof ds.at   !== 'function') throw new Error('dataset.at missing');
  if (typeof ds.reset !== 'function') throw new Error('dataset.reset missing');
  if (ds.size() < 1) throw new Error('expected non-empty dataset');
  const r = ds.next();
  if (!r.messages || !Array.isArray(r.messages)) throw new Error('messages missing from dataset.next()');
}
