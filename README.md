# xk6-llm

**LLM-aware load testing for k6. TTFT, ITL, TPOT, goodput, and token-throughput metrics for any OpenAI-compatible chat-completions server.**

A k6 extension for benchmarking LLM inference servers (vLLM, SGLang, TGI, llama.cpp server, Ollama, NIM, OpenAI). Streaming-first. Per-chunk timing matches `vllm bench serve` semantics so results cross-validate.

## Why

Existing benchmark tools (`vllm bench serve`, GuideLLM, SGLang `bench_serving`, LLMPerf, AIPerf) are Python-centric and run as standalone CLIs. They are excellent for one-off measurements; they are not built to live in a continuous-load-test pipeline alongside the rest of your performance suite. `xk6-llm` plugs LLM-aware metrics into k6, so a single k6 run can mix LLM inference, HTTP, gRPC, and database tests with consistent reporting, thresholds, and time-series export to Prometheus / InfluxDB / Cloud.

The metric definitions match upstream conventions verbatim. See [RESEARCH.md Part A](./RESEARCH.md) for citations.

## Install

```bash
go install go.k6.io/xk6/cmd/xk6@latest
xk6 build --with github.com/msradam/xk6-llm=. --output build/k6
./build/k6 version  # confirms k6/x/llm appears under Extensions
```

## Quick start

```js
import llm from 'k6/x/llm';

const client = new llm.Client({
  base_url: 'http://localhost:11434/v1',  // Ollama default
  model:    'granite4.1:3b',
});

export default async function () {
  const res = await client.chat({
    messages:   [{ role: 'user', content: 'Write a haiku about Poisson arrivals.' }],
    max_tokens: 128,
    temperature: 0,
  });
  console.log(`ttft=${res.ttft_ms.toFixed(1)}ms tpot=${res.tpot_ms.toFixed(1)}ms tokens=${res.completion_tokens}`);
}
```

Run with `./build/k6 run script.js`. Defaults point at Ollama; override `base_url`, `model`, and `api_key` for any OpenAI-compatible endpoint.

## Metrics

Every metric is tagged `model`. Error samples are additionally tagged `error_type`. Per-request `cache_state` and arbitrary `tags` are propagated when supplied.

| Name | Type | Description |
|---|---|---|
| `llm_requests` | Counter | Successful chat completions. |
| `llm_errors` | Counter | Failed requests. Tag `error_type` in `{network, timeout, http_4xx, http_5xx, stream, decode}`. |
| `llm_request_duration` | Trend (Time) | End-to-end wall time. |
| `llm_response_headers` | Trend (Time) | Time from request submit to HTTP response headers. Separates queue+RTT from prefill+generation when TTFT spikes. |
| `llm_ttft` | Trend (Time) | Time to first token. Measured at the first SSE chunk with non-empty `choices[0].delta.content`. Role-only deltas are skipped. |
| `llm_itl` | Trend (Time) | Inter-chunk latency, vector form. First sample is `t[chunk₂] - t[chunk₁]`, not `t[chunk₁] - start`. Matches vLLM and GuideLLM. |
| `llm_tpot` | Trend (Time) | Scalar inter-token time, `(e2e - ttft) / (n - 1)`. Matches AIPerf, genai-perf, and MLPerf. Emitted only when `completion_tokens > 1`. |
| `llm_chunks_per_request` | Trend (Default) | Count of content chunks. When `chunks < completion_tokens`, the server is batching tokens per chunk (TGI, speculative decoding). |
| `llm_prompt_tokens` | Counter | Server-reported `usage.prompt_tokens` (when emitted). |
| `llm_completion_tokens` | Counter | Server-reported `usage.completion_tokens` (when emitted). |
| `llm_goodput` | Rate | Per-request: 1 if all configured SLOs met, else 0. Emitted only when an `slo` predicate is supplied. |
| `llm_slo_ttft` | Rate | `ttft_ms <= slo.ttft_ms`. Emitted only when `slo.ttft_ms > 0`. |
| `llm_slo_tpot` | Rate | `tpot_ms <= slo.tpot_ms`. Emitted only when `slo.tpot_ms > 0` and TPOT is derivable. |
| `llm_slo_e2el` | Rate | `duration_ms <= slo.e2el_ms`. Emitted only when `slo.e2el_ms > 0`. |

### Resolved value of `chat()`

```ts
{
  content:             string,
  ttft_ms:             number,
  itl_ms:              number[],   // per-chunk inter-arrival, ms
  tpot_ms:             number,     // 0 if not derivable
  duration_ms:         number,
  response_headers_ms: number,
  chunks:              number,     // content-bearing chunks
  prompt_tokens:       number,
  completion_tokens:   number,
  finish_reason:       string,
}
```

## Goodput and SLO predicates

Goodput (requests/second satisfying all per-request SLOs) is the primary capacity metric in current LLM-serving literature. See [RESEARCH.md §A.10](./RESEARCH.md).

```js
const client = new llm.Client({
  base_url: 'https://api.example.com/v1',
  model:    'llama-3.1-8b',
  slo: {                  // default applied to every request
    ttft_ms: 500,
    tpot_ms: 50,
    e2el_ms: 5000,
  },
});

export const options = {
  thresholds: {
    llm_goodput:  ['rate>0.99'],   // 99% of requests meet all three SLOs
    llm_slo_ttft: ['rate>0.99'],
    llm_slo_tpot: ['rate>0.99'],
    llm_errors:   ['count==0'],
  },
};

export default async function () {
  await client.chat({ messages: [...], max_tokens: 256 });
}
```

When the run finishes, `llm_goodput` reports the fraction of requests that satisfied every SLO simultaneously. When goodput drops, the per-SLO Rates identify which constraint failed: if `llm_slo_ttft` collapsed but `llm_slo_tpot` held, the queue is backed up. If TPOT collapsed but TTFT held, generation is slow.

SLO syntax (`ttft:X tpot:Y e2el:Z`, milliseconds) matches `vllm bench serve --goodput` and `sglang bench_serving --goodput`.

## Prefix-cache and speculative-decoding caveats

Two engine features will silently invalidate naive comparisons. See [RESEARCH.md §A.11](./RESEARCH.md).

**Prefix caching** (vLLM `--enable-prefix-caching` default since v0.5, SGLang RadixAttention, TRT-LLM KV reuse) makes TTFT 5–10x lower on cache-hit requests. A run with shared prefixes mixes cold and warm samples into a meaningless mean. Tag every request explicitly:

```js
await client.chat({
  messages: [...],
  cache_state: requestIndex === 0 ? 'cold' : 'warm',
});
```

Then aggregate metrics filtered by `cache_state` tag in your dashboard.

**Speculative decoding** (vLLM `--speculative-config`, SGLang `--speculative-algorithm`, TRT-LLM EAGLE/Medusa) emits multiple accepted tokens per SSE chunk. `llm_itl` becomes multimodal: short gaps between batched-together tokens, long gaps between verification batches. Reach for `llm_tpot` and `llm_chunks_per_request` when this matters. `llm_chunks_per_request < llm_completion_tokens` means the server is batching.

## Validation

`xk6-llm` is designed to cross-validate against `vllm bench serve` and (as a second opinion) GuideLLM and SGLang `bench_serving`. Run the same dataset slice and Poisson rate against both tools and compare per [RESEARCH.md §A.8](./RESEARCH.md) (tolerances: mean TTFT within 5ms or 2%, mean ITL within 0.5ms or 5%, goodput within 1 percentage point).

## k6 compatibility

| xk6-llm | k6 | xk6 |
|---|---|---|
| v0.x | v2.0.0 | 1.4.1 |

## Attribution

This codebase was developed with assistance from [Claude Code](https://claude.com/claude-code). Metric definitions, build conventions, and design decisions are documented in [`CLAUDE.md`](./CLAUDE.md) and [`RESEARCH.md`](./RESEARCH.md); both are checked-in working notes, not generated boilerplate. Issues and PRs are reviewed and merged by humans.

## License

Apache-2.0
