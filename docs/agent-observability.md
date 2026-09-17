# Agent Observability export

With `agento11y` set on an `llm.Client`, every `chat()` call exports one generation record to Grafana Agent Observability through the [agento11y Go SDK](https://github.com/grafana/agento11y/tree/main/go). The extension keeps its own k6 metrics and its own timing. The SDK runs with a no-op tracer and meter, so only its generation-export path is active and nothing OTel-shaped enters the VU hot path.

## Configuration

```js
const client = new llm.Client({
  base_url: 'http://localhost:8000/v1',
  model: 'ibm-granite/granite-4.1-30b',
  agento11y: {
    endpoint:      'https://agent-observability-ingest.grafana.net',
    auth_mode:     'bearer',           // none | tenant | bearer | basic
    bearer_token:  __ENV.AGENTO11Y_TOKEN,
    agent_name:    'k6-canary',
    agent_version: 'v3',               // optional, see Version identity
    tags:          { team: 'inference', env: 'staging' },
    capture_content: false,            // default
    synthetic:       true,             // default
    flush_interval_ms: 1000,
  },
});
```

| Field | Default | Meaning |
|---|---|---|
| `endpoint` | required | Generation-export base URL. The SDK appends the export path. |
| `protocol` | `http` | `http`, `grpc`, or `none` to keep the config but export nothing. |
| `auth_mode` | `none` | With `tenant_id`, `bearer_token`, or `basic_user` and `basic_password`. |
| `insecure` | loopback only | Cleartext export is on for `localhost`, `127.0.0.1` and `[::1]`, off elsewhere. |
| `agent_name` | empty | The agent the records belong to. |
| `agent_version` | derived | See below. |
| `synthetic` | `true` | Tag every record as load-generated. |
| `capture_content` | `false` | Send prompt and completion text. |
| `tags` | none | Merged into every record. Request `tags` win on conflict. |
| `flush_interval_ms` | SDK default | How long a record waits in the queue before export. |

For a local receiver, run `agento11y local serve` from the [agento11y plugin](https://github.com/grafana/agento11y/tree/main/plugins/agento11y) and point `endpoint` at the URL it prints.

## What a record carries

| Generation field | Source |
|---|---|
| `id` | `generation_id` from the request, or one minted per call and returned on the result. |
| `parent_generation_ids` | `parent_generation_ids` from the request. |
| `conversation_id` | The `session_id` tag, which `llm.Session` sets. |
| `mode` | `STREAM`, or `SYNC` for a `stream: false` call. |
| `model.provider` | `anthropic` for the Anthropic wire, `openai` for the rest. |
| `model.name`, `response_model` | The client's `model`. |
| `usage` | Server-reported prompt and completion tokens, with cache read, cache write and reasoning sub-buckets when the provider reports them. |
| `stop_reason` | The wire's finish reason, untranslated. |
| `response_id` | The provider's request id from `x-request-id`, `request-id` or `x-generation-id`, when sent. |
| `max_tokens`, `temperature`, `top_p`, `tool_choice` | Read from the request before wire translation. An object `tool_choice` collapses to the function it names. |
| `thinking_enabled` | True when the response carried reasoning tokens or reasoning events. |
| `tools` | The definitions the model was offered, on every wire shape. The SDK strips descriptions and schemas in metadata-only mode and keeps the names. |
| `started_at`, `completed_at`, first token time | From the extension's own clock. |
| `call_error` | The error a failed call raised. In metadata-only mode the SDK replaces the text with a category such as `client_error` or `timeout`. |
| `tags` | Client `agento11y.tags`, then request `tags`, then the synthetic markers. |
| `metadata` | The measurements below. |

Failed calls export too. A canary exists to catch failures, and a consumer that sees only the successful calls of a canary that is hitting 5xx has been told nothing. Before this was the case, a 404 from the upstream produced no conversation at all on a local receiver.

### Metadata keys

The Generation schema records time to first token and total duration but nothing about the shape of the stream in between. A response that streams at 40 tokens per second and one that stalls for two seconds mid-generation look the same. Until the schema has fields for them, the stream measurements travel as metadata:

| Key | Value |
|---|---|
| `llm.ttft_ms` | Time to first content-bearing event. |
| `llm.ttf_text_ms` | Time to first text delta, on reasoning models. |
| `llm.itl_mean_ms`, `llm.itl_p50_ms`, `llm.itl_max_ms`, `llm.itl_samples` | Inter-token latency summary. |
| `llm.tpot_ms` | Time per output token over the text phase. |
| `llm.chunks`, `llm.thinking_chunks` | Stream event counts. |
| `llm.thinking_tokens`, `llm.cached_tokens` | Token sub-buckets, duplicated here for consumers that only read metadata. |
| `llm.response_headers_ms` | Submit to response headers. |
| `llm.server_processing_ms` | The provider's `openai-processing-ms`, when sent. Subtracted from the duration it isolates network and gateway time. |
| `llm.cost_usd`, `llm.energy_j` | Present when the client has a `cost` or `energy` model. |
| `llm.goodput` | Whether every configured SLO passed. Computed by the same function that emits the k6 Rate samples, so the two cannot disagree. |
| `llm.aborted` | Present when `abort_after_ms` or `abort_after_tokens` cut the stream. |
| `llm.error_type` | The k6 `error_type` of a failed call: `network`, `timeout`, `http_4xx`, `http_5xx`, `stream`, `decode` or `unsupported`. Metadata is not filtered by capture mode, so this survives when the error text does not. |

## Synthetic traffic

Every record carries `agento11y.synthetic=true` and `agento11y.synthetic.producer=xk6-llm` unless `synthetic: false` is set. Traffic from a load generator is synthetic by construction, and a consumer that cannot tell it apart from real traffic overstates usage and cost. Agent Observability has no reserved field for this yet, so the convention lives in two exported constants, `SyntheticTagKey` and `SyntheticProducerTagKey`, in [`agento11y.go`](../agento11y.go).

## Version identity

When `agent_version` is empty, the record's `effective_version` is a SHA-256 digest of the system prompt. The collector would otherwise derive a version by hashing the prompt itself, but it only sees the prompt under content capture, which is off by default. Left to the collector, every agent collapsed to one hash of the empty string and toggling capture moved an unchanged prompt to a new version. The digest is metadata, so it survives metadata-only mode, and one prompt is one version whatever the capture setting.

A declared `agent_version` is the caller's to own and is passed through unchanged.

## Content capture

`capture_content: false` (the default) exports structure, usage, timing, tool names and ids, and strips text, tool arguments, tool results, system prompts, tool descriptions and schemas, and error text. `capture_content: true` sends the prompt messages, the system prompt, the completion text, and tool calls and results as typed parts, so the tool-call projection in Agent Observability sees them.

Metadata and tags are never filtered by capture mode. Do not put prompt text in `tags`.

## Flushing

Records queue in the SDK and export on a timer. Records queued late in an iteration are lost when the test ends before the interval elapses, and the ones most likely to be dropped are the last ones, which for a canary are the observations nearest whatever it was trying to catch.

The extension cannot flush for you. k6 has no per-VU teardown hook ([grafana/k6#5382](https://github.com/grafana/k6/issues/5382)), `teardown()` runs module init again so a client constructed there has an empty queue, and a goroutine waiting on the VU context is not waited for, so the process exits before its request completes. A script that needs every record calls `client.flush()` at the end of its default function, where it runs synchronously inside the iteration. `flush()` waits at most ten seconds and throws on failure. An export failure never fails the chat call itself; it is counted on `llm_errors` with `error_type=export`.

## Not exported

Embedding calls and tool executions are span-only in the SDK, and the extension runs without a tracer, so `embed()` reports k6 metrics only. Workflow steps are not exported. A script that runs tools in JavaScript can still make the call graph visible by passing the tool-calling generation's id as a parent of the next call.
