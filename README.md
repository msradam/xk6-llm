# xk6-llm

A k6 extension for load-testing LLM inference servers and AI gateways. Streaming TTFT, ITL, TPOT, goodput, cost and energy metrics over the OpenAI, Anthropic, Responses and AI SDK wires, with optional export of every call to Grafana Agent Observability.

![xk6-llm Grafana dashboard](./quickstart/img/dashboard.png)

*1,470 requests, 0 errors, 100% goodput, $0.382 total cost. Three-turn conversations against `ibm-granite/granite-4.1-30b` on a single NVIDIA B300 SXM6. The "TTFT by turn" panel surfaces prefix-cache speedup across turns; the cost and energy panels are derived from server-reported token counts.*

![xk6-llm demo](./quickstart/img/demo.gif)

*Recorded against Ollama on an M3 MacBook Air. The same script and dashboard work against a hosted API or a vLLM cluster; only the numbers change.*

## Build

```bash
go install go.k6.io/xk6/cmd/xk6@latest
mkdir -p build
xk6 build --with github.com/msradam/xk6-llm@latest --output build/k6
./build/k6 version
```

If you're on a platform `xk6 build` does not recognize (linux/s390x, linux/ppc64le, freebsd, etc.), see [`CONTRIBUTING.md`](./CONTRIBUTING.md#building-on-unsupported-platforms) for a `go build` recipe that bypasses xk6's allow-list.

## Example

```js
import llm from 'k6/x/llm';

const client = new llm.Client({
  base_url: 'http://localhost:11434/v1',
  model:    'granite4.1:3b',
});

export default async function () {
  const res = await client.chat({
    messages:   [{ role: 'user', content: 'Write a haiku about arrival rates.' }],
    max_tokens: 128,
    temperature: 0,
  });
  console.log(`ttft=${res.ttft_ms.toFixed(1)}ms tpot=${res.tpot_ms.toFixed(1)}ms tokens=${res.completion_tokens}`);
}
```

```bash
./build/k6 run examples/chat.js
```

## Wires

Scripts always write OpenAI-shaped requests. The `wire` option picks the encoding the client speaks, and the extension translates the request and parses the stream.

| `wire` | Endpoint | Speaks to |
|---|---|---|
| `openai` (default) | `POST {base_url}/chat/completions` | vLLM, SGLang, TGI, llama.cpp, Ollama, NIM, OpenAI, OpenRouter, any OpenAI-compatible server |
| `anthropic` | `POST {base_url}/messages` | Anthropic Messages API and gateways that expose it |
| `responses` | `POST {base_url}/responses` | OpenAI Responses API |
| `providerwire-v4` | `POST {base_url}/language-model` | AI SDK gateways such as [Grafana AI Gateway](https://github.com/grafana/ai-sdk/tree/main/ai-gateway) |

Every wire measures the same things the same way: TTFT at the first content-bearing event (reasoning included), ITL between text deltas, token counts from the server's usage report. Reasoning models report `thinking_tokens` and `ttf_text_ms` so TPOT is computed over the text phase only.

## Metrics

Every metric is tagged `model`. Errors are additionally tagged `error_type`. Unary calls (`stream: false`) are tagged `mode=unary`. Per-request `cache_state` and arbitrary `tags` are propagated when supplied.

| Name | Type | Description |
|---|---|---|
| `llm_requests` | Counter | Successful chat completions. |
| `llm_errors` | Counter | Failures. Tag `error_type` in `{network, timeout, http_4xx, http_5xx, stream, decode, unsupported, export}`. |
| `llm_request_duration` | Trend (Time) | End-to-end wall time. |
| `llm_response_headers` | Trend (Time) | Submit to HTTP response headers. |
| `llm_ttft` | Trend (Time) | Time to first token. First content-bearing SSE chunk; role-only deltas skipped. |
| `llm_itl` | Trend (Time) | Per-chunk inter-arrival vector. First sample is `t[chunk₂] - t[chunk₁]`. |
| `llm_tpot` | Trend (Time) | Scalar `(e2e - ttft) / (n - 1)`. Emitted only when `completion_tokens > 1`. |
| `llm_chunks_per_request` | Trend | Content chunks per request. |
| `llm_prompt_tokens` | Counter | Server-reported `usage.prompt_tokens`. |
| `llm_completion_tokens` | Counter | Server-reported `usage.completion_tokens`. |
| `llm_goodput` | Rate | All SLOs met. Emitted only when an `slo` predicate is supplied. |
| `llm_slo_ttft` | Rate | `ttft_ms <= slo.ttft_ms`. |
| `llm_slo_tpot` | Rate | `tpot_ms <= slo.tpot_ms`. |
| `llm_slo_e2el` | Rate | `duration_ms <= slo.e2el_ms`. |
| `llm_cost_usd` | Trend | USD per request. Emitted only when `cost` is supplied. |
| `llm_energy_j` | Trend | Estimated joules per request. Emitted only when `energy` is supplied. |
| `llm_energy_j_per_token` | Trend | `llm_energy_j / completion_tokens`. |
| `llm_tool_calls` | Counter | Tool invocations the model emitted. Omitted when zero. |
| `llm_aborted` | Counter | Chat completions cut short by `abort_after_ms` or `abort_after_tokens`. |
| `llm_embed_requests` | Counter | Successful `/v1/embeddings` calls. |
| `llm_embed_errors` | Counter | Embed failures (tag `error_type`). |
| `llm_embed_duration` | Trend (Time) | End-to-end wall time of an embed call. |
| `llm_embed_tokens` | Counter | Server-reported `usage.prompt_tokens` for embed calls. |
| `llm_embed_inputs` | Counter | Number of input strings per embed call. |

The names map onto the OpenTelemetry GenAI client metrics, which are still in development: `llm_ttft` is `gen_ai.client.operation.time_to_first_chunk`, `llm_itl` and `llm_tpot` are `gen_ai.client.operation.time_per_output_chunk`, `llm_request_duration` is `gen_ai.client.operation.duration`, and the token counters are `gen_ai.client.token.usage` split by token type. The k6 names stay stable while the convention settles.

## API

### `new llm.Client(opts)`

```ts
{
  base_url?:   string,                          // default: http://localhost:11434/v1
  api_key?:    string,
  model?:      string,
  timeout_ms?: number,                          // default: 60000
  ignore_eos?: boolean,
  headers?:    Record<string, string>,
  wire?:       'openai' | 'anthropic' | 'responses' | 'providerwire-v4', // default: openai
  slo?:        { ttft_ms?, tpot_ms?, e2el_ms? },
  cost?:       { usd_per_million_input_tokens?, usd_per_million_output_tokens? },
  energy?:     { j_per_input_token?, j_per_output_token?, idle_w? },
  agento11y?:  { endpoint, agent_name?, ... },  // see docs/agent-observability.md
}
```

### `client.chat(req)`

`req` accepts the OpenAI chat-completion fields (`messages`, `max_tokens`, `temperature`, `top_p`, `seed`, `tools`, `tool_choice`, etc.) plus these control fields, which are stripped before the request is sent:

| Field | Purpose |
|---|---|
| `slo` | Per-call SLO override. |
| `cache_state` | `'cold'` or `'warm'`, emitted as a metric tag. |
| `tags` | Request-scoped tags applied to every sample and exported record. |
| `stream` | Default `true`. `false` sends a unary request. |
| `abort_after_ms`, `abort_after_tokens` | Cut the stream short; the result resolves with `aborted: true`. |
| `generation_id`, `parent_generation_ids` | Declare the call graph for Agent Observability. |

Returns a Promise resolving to:

```ts
{
  content:             string,
  generation_id:       string,
  stream:              boolean,   // false for a unary call
  ttft_ms:             number,
  ttf_text_ms:         number,    // first text delta; later than ttft_ms on a reasoning model
  itl_ms:              number[],
  tpot_ms:             number,
  duration_ms:         number,
  response_headers_ms: number,
  chunks:              number,
  prompt_tokens:       number,
  completion_tokens:   number,
  cached_tokens:       number,    // prompt-cache read sub-bucket, when reported
  cache_write_tokens:  number,    // prompt-cache write sub-bucket (Anthropic)
  thinking_tokens:     number,    // reasoning sub-bucket of completion_tokens
  thinking_chunks:     number,
  finish_reason:       string,
  aborted:             boolean,
  tool_calls:          { id, name, arguments }[],
}
```

Calls stream by default. A `stream: false` call is sent without streaming on every wire and the whole response is parsed at once. It reports `llm_requests`, `llm_errors`, `llm_request_duration`, `llm_response_headers`, token counts, cost, energy and the `e2el_ms` SLO, all tagged `mode=unary`. It does not report `llm_ttft`, `llm_itl`, `llm_tpot` or `llm_chunks_per_request`, and a `ttft_ms` or `tpot_ms` SLO is not evaluated for it. `abort_after_tokens` requires a streamed call; `abort_after_ms` works for both.

### `client.embed(req)`

`POST {base_url}/embeddings` with `input` as one string or an array. Resolves to `{ model, embeddings, prompt_tokens, duration_ms, inputs }` and reports the `llm_embed_*` metrics. See [`examples/embed.js`](./examples/embed.js) and [`examples/rag.js`](./examples/rag.js).

### `client.flush()`

Exports queued Agent Observability records now. Call it at the end of an iteration when every record matters; see [`docs/agent-observability.md`](./docs/agent-observability.md#flushing).

### `new llm.Session(client, opts)`

A multi-turn conversation. Each `send()` appends the user message, calls `chat()`, and appends the assistant reply to the history. Every call is tagged `session_id` and `turn`, with `cache_state` set to `cold` on turn 1 and `warm` after, so the dashboard can show prefix-cache speedup across turns.

```js
const s = new llm.Session(client, { system: 'You are terse.' });
const r = await s.send({ content: 'What is a B-tree index?', max_tokens: 80 });
s.turn(); s.id(); s.messages(); s.tokens(); s.reset();
```

`send()` takes a string or an object with `content` plus any `chat()` option. `tokens()` sums prompt and completion tokens across every call, which is what a hosted API bills.

### `new llm.Dataset(opts)`

Replays a JSONL prompt corpus. One line per request: `{"messages": [...], "max_tokens"?: N, ...}`. Loaded once per process and cached by absolute path.

```ts
{
  path:     string,
  seed?:    number,    // default: 42
  shuffle?: boolean,
}
```

Methods: `dataset.size()`, `dataset.next()`, `dataset.at(i)`, `dataset.reset()`.

A converter for ShareGPT V3 to this format lives at `scripts/sharegpt_to_jsonl.py`.

## Agent Observability

With `agento11y` configured on the client, every `chat()` call exports one generation record to [Grafana Agent Observability](https://grafana.com/docs/grafana-cloud/machine-learning/agent-observability/), including failed calls. Records carry the model, token buckets, stop reason, sampling parameters, the tools the model was offered, and the call graph the script declares through `parent_generation_ids`. TTFT, ITL, TPOT, cost, energy and goodput travel as metadata. Every record is tagged `agento11y.synthetic=true` so a consumer can keep load-generated traffic out of usage and cost reporting. Prompt and completion text stay home unless `capture_content: true`.

```js
const client = new llm.Client({
  base_url: 'http://localhost:8000/v1',
  model: 'ibm-granite/granite-4.1-30b',
  agento11y: {
    endpoint:   __ENV.AGENTO11Y_ENDPOINT,
    auth_mode:  'bearer',
    bearer_token: __ENV.AGENTO11Y_TOKEN,
    agent_name: 'k6-canary',
  },
});

export default async function () {
  const r = await client.chat({ messages, max_tokens: 64 });
  client.flush();
}
```

[`docs/agent-observability.md`](./docs/agent-observability.md) has the field mapping, the flush rule and the content-capture behaviour.

## What you can simulate

k6 is a Go binary that executes test scripts written in JavaScript (TypeScript supported), so a single VU can carry conversation state, branch on responses, and drive multi-call workflows in user code.

| Example | What it shows |
|---|---|
| [`session.js`](./examples/session.js) | `llm.Session`: a 3-turn conversation with per-turn tags and token accounting. |
| [`multi-turn.js`](./examples/multi-turn.js) | The same pattern with `Client` directly, at a constant arrival rate, so the dashboard can show TTFT degradation across turns and prefix-cache speedup (typically 5 to 15x by turn 5). |
| [`tools.js`](./examples/tools.js) | Native tool calling. Pass `tools`, run the `tool_calls` the model returns, append `role: "tool"` results, call again. |
| [`agent.js`](./examples/agent.js) | The tool loop run to completion: call, execute, feed back, repeat until the model answers or `MAX_ITERATIONS`. Tags `agent_id` and `iteration` so the dashboard rolls up per-session totals and full-envelope p95. |
| [`rag.js`](./examples/rag.js) | Embed, retrieve, generate. Each phase has its own k6 Trend with independent SLO thresholds. |
| [`abort.js`](./examples/abort.js) | Abandoned requests with `abort_after_ms` and `abort_after_tokens`, which is how a gateway's cancellation path gets tested. |
| [`ab-providers.js`](./examples/ab-providers.js) | Two `Client` instances under two scenarios with different `cost` configs. Identical traffic, side-by-side latency and dollar-cost panels in one run. |
| [`showcase.js`](./examples/showcase.js) | Ramping arrival rate, SLOs, goodput, cost and energy in one run. |
| [`composite.js`](./examples/composite.js) | An AI feature end to end in one iteration: REST login, a model call offered a tool, the tool run against the application, the answer stored, and the page that renders it opened with `k6/browser`. |

### Combined testing

`k6/x/llm` is an ordinary k6 module, so `http`, `k6/browser` and the model client run in one iteration under one set of tags and one clock. `composite.js` uses that for two checks a single-protocol tool cannot make: the application's own count of tool lookups has to match the tool calls the model made, and the answer has to paint on the page. It ships with a stdlib-only mock application in [`examples/app/`](./examples/app/):

```bash
python3 examples/app/server.py
K6_BROWSER_HEADLESS=true ./build/k6 run examples/composite.js
```

The examples default to Ollama on a laptop. `multi-turn.js` and `showcase.js` ramp to rates a laptop cannot serve; point them at a real inference server or lower `LLM_RATE`.

## Quickstart with Grafana

A Docker Compose stack with a pre-provisioned dashboard is in [`quickstart/`](./quickstart/). See [`QUICKSTART.md`](./QUICKSTART.md).

## Validation

Cross-validated against `vllm bench serve` on a real vLLM 0.21.0 server (Qwen2.5-72B-Instruct-AWQ, A100 80GB). TPOT/ITL/E2EL agree within 5% on identical workloads; token counts are bit-exact with vLLM `/metrics`. Numbers and methodology in [`docs/validation-parity.md`](./docs/validation-parity.md) and [`docs/validation-results.md`](./docs/validation-results.md).

The `providerwire-v4` wire and the Agent Observability export are exercised against a Grafana AI Gateway build and an `agento11y local serve` receiver by a separate synthetic-agent harness, which uses this extension to catch buffered streams, dropped usage, leaked cancellations and lost records.

## Compatibility

| xk6-llm | k6 | xk6 |
|---|---|---|
| v0.x | v2.0.0 to v2.2.0 | 1.4.3 |

Grafana Cloud k6 runs a fixed set of extensions and does not build custom binaries, so this extension runs with k6 OSS, the k6 Operator, or a self-hosted runner. Results still reach Grafana through the Prometheus remote write or OpenTelemetry outputs.

## Attribution

This codebase was developed with assistance from [Claude Code](https://claude.com/claude-code). PRs are reviewed and merged by humans.

## License

Apache-2.0
