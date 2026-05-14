# xk6-llm

**LLM-aware load testing for k6 — TTFT, ITL, and token-throughput metrics for any OpenAI-compatible chat-completions server.**

A k6 extension that turns k6 into a production-grade benchmark client for LLM inference servers (vLLM, TGI, llama.cpp server, NIM, etc.). Streaming-first, with per-chunk timing that matches `vllm bench serve` semantics so results are directly comparable.

## Metrics

| Name | Type | Description |
|---|---|---|
| `llm_requests` | Counter | Successful chat completions |
| `llm_errors` | Counter | Failed requests (HTTP error, stream error, timeout) |
| `llm_request_duration` | Trend (Time) | End-to-end wall time |
| `llm_ttft` | Trend (Time) | **T**ime **T**o **F**irst **T**oken — measured at the first SSE chunk with non-empty `choices[0].delta.content`. Role-only deltas are skipped. |
| `llm_itl` | Trend (Time) | **I**nter-token (per-chunk) latency. First sample is `t[chunk₂] - t[chunk₁]`, NOT `t[chunk₁] - start`. Matches vLLM. |
| `llm_prompt_tokens` | Counter | From the server's `usage.prompt_tokens`, if emitted |
| `llm_completion_tokens` | Counter | From the server's `usage.completion_tokens`, if emitted |

All samples are tagged with `model`.

## Usage

```js
import llm from 'k6/x/llm';

const client = new llm.Client({
  base_url: 'http://localhost:11434/v1',  // Ollama default
  model: 'granite4.1:3b',
  timeout_ms: 60000,
});

export default async function () {
  const res = await client.chat({
    messages: [{ role: 'user', content: 'Write a haiku about Poisson arrivals.' }],
    max_tokens: 128,
    temperature: 0,
  });
  console.log(`TTFT=${res.ttft_ms.toFixed(1)}ms tokens=${res.completion_tokens}`);
}
```

## Build

```bash
go install go.k6.io/xk6/cmd/xk6@latest
xk6 build --with github.com/msradam/xk6-llm=. --output build/k6
./build/k6 run examples/chat.js
```

## Validation

`xk6-llm`'s metrics are cross-validated against `vllm bench serve` on the same model, dataset slice, and Poisson rate. See [RESEARCH.md §A.8](./RESEARCH.md) for the exact recipe and tolerances.

## k6 compatibility

| xk6-llm | k6        | xk6   |
|---------|-----------|-------|
| v0.x    | v2.0.0-rc1 | 1.4.1 |

## License

Apache-2.0
