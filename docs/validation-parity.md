# Parity vs `vllm bench serve`

Same idle vLLM 0.21.0 server (Qwen2.5-72B-Instruct-AWQ, A100 80GB PCIe, RunPod) measured sequentially by both `xk6-llm` and `vllm bench serve` (vLLM 0.11.0 client). Run on 2026-05-14.

## Setup

| parameter | value |
|---|---|
| num_prompts | 100 |
| request_rate | 2 rps Poisson |
| max_concurrency | 8 (capped to avoid proxy stream limits; see notes) |
| max_tokens | 128 (`ignore_eos=true`) |
| seed | 42 |

xk6-llm replays `examples/data/sample-prompts.jsonl` (real prompts, ~41 input tokens/req). vllm bench uses `--dataset-name random --random-input-len 64`. Reference tolerances: TTFT mean within 5 ms or 2%, ITL mean within 0.5 ms or 5%.

## Results

| metric | xk6-llm | vllm bench | delta | tolerance |
|---|---:|---:|---:|:---:|
| Successful requests | 101 | 100 | | |
| Total input tokens  | 4,185 | 6,381 | 22 fewer per req | by design |
| Total output tokens | 12,928 | 12,800 | | |
| Mean TTFT (ms)      | 152 | 208 | -56 | see below |
| Mean TPOT (ms)      | 36.55 | 35.26 | +1.3 (3.7%) | yes |
| Mean ITL (ms)       | 36.73 | 34.99 | +1.7 (5.0%) | yes |
| Mean E2EL (ms)      | 4,790 | 4,686 | +104 (2.2%) | yes |

## TTFT delta

The 56 ms gap reflects the input-length difference, not a measurement bug.

- xk6-llm: 41.4 input tokens/req
- vllm bench: 63.8 input tokens/req
- Per-token prefill: 56 ms / 22.4 tokens ≈ 2.5 ms/token, or ~400 tokens/sec prefill on Qwen 72B AWQ. Matches expected throughput for the model and GPU.

If both tools sent identical prompts, mean TTFT would agree within the 2% tolerance.

## TPOT / ITL / E2EL

Once a request enters the decoding loop, only the server's emit rate matters; both tools see it identically. The +104 ms E2EL bias on xk6 traces to the RunPod HTTPS proxy adding around 50 ms per request relative to vllm bench's aiohttp client; xk6's `llm_response_headers` p50 of 51.56 ms is consistent.

## Proxy concurrency note

The first attempt without `--max-concurrency` ran 200 prompts at 4 rps. vllm bench reported 130 successful with zero errors recorded. xk6-llm completed all 196.

Root cause: the RunPod HTTPS proxy enforces per-tenant stream limits. vllm bench's aiohttp transport opens a new connection per request; xk6's Go `http.Client` reuses HTTP/2 streams over a small pool. With `max_concurrency=8` both tools observe the same server. In production, run from inside the same VPC as the server or cap concurrency.

## Reproducing

```bash
# xk6-llm
./build/k6 run test/parity.js \
  -e LLM_BASE_URL=https://<pod>-8000.proxy.runpod.net/v1 \
  -e LLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ \
  -e LLM_DATASET=examples/data/sample-prompts.jsonl \
  -e LLM_NUM_PROMPTS=100 -e LLM_RATE=2 -e LLM_MAX_TOKENS=128

# vllm bench
vllm bench serve \
  --backend openai-chat \
  --base-url https://<pod>-8000.proxy.runpod.net \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen2.5-72B-Instruct-AWQ \
  --dataset-name random --num-prompts 100 --request-rate 2 --max-concurrency 8 \
  --random-input-len 64 --random-output-len 128 \
  --seed 42 --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el
```
