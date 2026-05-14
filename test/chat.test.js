// Smoke test for the registered module. Runs under `xk6 test` via the validate workflow.
import llm from 'k6/x/llm';

export default function () {
  if (typeof llm.Client !== 'function') {
    throw new Error('llm.Client constructor missing');
  }
  // Construction with defaults succeeds and exposes chat().
  const c = new llm.Client();
  if (typeof c.chat !== 'function') throw new Error('client.chat missing');
}
