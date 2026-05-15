# Validation results

Run on 2026-05-14. Target: `vllm/vllm-openai:latest` (`vllm-0.21.0-f3716e14`) serving `Qwen/Qwen2.5-72B-Instruct-AWQ` on a single A100 80GB PCIe (RunPod). Pod recipe: [`runpod-vllm.md`](./runpod-vllm.md). Driver: [`test/heavy.js`](../test/heavy.js).

581 requests, 0 errors, 98.62% goodput, all thresholds passed.

## Scenarios

| phase | executor | shape |
|---|---|---|
| `warmup`  | per-vu-iterations    | 1 VU × 3 iters |
| `conc_1`  | constant-vus         | 1 VU × 20s |
| `conc_4`  | constant-vus         | 4 VUs × 20s |
| `conc_16` | constant-vus         | 16 VUs × 20s |
| `poisson` | constant-arrival-rate | 6 rps × 30s |
| `long_gen`| constant-vus         | 4 VUs × 30s, 512 tokens |

SLOs: `ttft_ms: 1500, tpot_ms: 80, e2el_ms: 15000`.

## Aggregates

```
llm_requests............ 581
llm_errors..............   0
llm_goodput............. 98.62%  (573/581)
llm_slo_ttft............ 100.00%
llm_slo_tpot............ 100.00%
llm_slo_e2el............  98.62%  (8 failures, all in long_gen)
llm_prompt_tokens....... 17,630   (98.2 tok/s)
llm_completion_tokens... 16,129   (89.8 tok/s)
```

| metric | p50 | p95 | max |
|---|---|---|---|
| `llm_ttft` | 253 ms | 576 ms | 761 ms |
| `llm_tpot` | 36 ms | 43 ms | 45 ms |
| `llm_itl`  | 34 ms | 48 ms | 207 ms |
| `llm_response_headers` | 142 ms | 498 ms | 656 ms |
| `llm_request_duration` | 1.07 s | 1.36 s | 17.34 s |

## Cross-check vs vLLM `/metrics`

`vllm:*` counters from `https://<pod>/metrics` immediately after the run. The server saw one extra request from a prior `curl` smoke test (36 prompt / 6 completion tokens), accounted for in the rightmost column.

| metric | xk6-llm | vLLM `/metrics` | xk6 + smoke |
|---|---:|---:|---:|
| requests | 581 | 582 | 582 |
| prompt_tokens | 17,630 | 17,666 | 17,666 |
| completion_tokens | 16,129 | 16,135 | 16,135 |
| `finish_reason=stop`   | 573 | 574 | 574 |
| `finish_reason=length` |   8 |   8 |   8 |

## TTFT decomposition

| layer | mean |
|---|---:|
| vLLM internal (`vllm:time_to_first_token_seconds_sum / _count`) | 117 ms |
| `llm_response_headers` (network + proxy) | 203 ms |
| sum | 320 ms |
| `llm_ttft` (client-perceived) | 316 ms |

The 4 ms gap is sampling noise. To compare to vLLM's internal TTFT directly, subtract `llm_response_headers` from `llm_ttft`.

## Reproducing

```bash
./build/k6 run test/heavy.js \
  -e LLM_BASE_URL=https://<podId>-8000.proxy.runpod.net/v1 \
  -e LLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ \
  --summary-export=/tmp/heavy_summary.json
```
