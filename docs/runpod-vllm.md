# Running vLLM on RunPod

A working recipe for the vLLM stack used in [`validation-results.md`](./validation-results.md) and [`validation-parity.md`](./validation-parity.md).

## Prerequisites

- `runpodctl` (`brew install runpod/runpodctl/runpodctl`), authenticated via `runpodctl doctor` or `RUNPOD_API_KEY`.
- A k6 binary built with this extension.

## Pod config

| field | value | notes |
|---|---|---|
| Image | `vllm/vllm-openai:latest` | Pin a tag for reproducibility. |
| GPU | `NVIDIA A100 80GB PCIe` | Holds the AWQ model with KV-cache headroom. |
| Cloud | `SECURE` | More predictable network than community. |
| Container disk | 100 GB | Image + AWQ weights staging + headroom. |
| Volume | 60 GB at `/root/.cache/huggingface` | Persists the HF cache across restarts. |
| Ports | `8000/http, 22/tcp` | RunPod proxies HTTP at `https://<podId>-8000.proxy.runpod.net`. |
| Model | `Qwen/Qwen2.5-72B-Instruct-AWQ` | Ungated; fits on a single A100 80GB. |
| `--max-model-len` | 8192 | |
| `--gpu-memory-utilization` | 0.92 | |
| `--dtype` | `float16` | AWQ kernels expect fp16 activations. |

## Create

```bash
runpodctl pod create \
  --name xk6-llm-vllm \
  --image vllm/vllm-openai:latest \
  --gpu-id "NVIDIA A100 80GB PCIe" \
  --gpu-count 1 \
  --container-disk-in-gb 100 \
  --volume-in-gb 60 \
  --volume-mount-path /root/.cache/huggingface \
  --ports "8000/http,22/tcp" \
  --cloud-type SECURE \
  --ssh \
  --docker-args "--model Qwen/Qwen2.5-72B-Instruct-AWQ --max-model-len 8192 --gpu-memory-utilization 0.92 --dtype float16"

POD_ID=$(runpodctl pod list -o json | jq -r '.[] | select(.name=="xk6-llm-vllm") | .id')
ENDPOINT="https://${POD_ID}-8000.proxy.runpod.net"
```

## Cold-start timings

| phase | duration |
|---|---|
| Container scheduling | 30–60 s |
| Image pull | 1–2 min |
| HF model download (40GB AWQ) | 4–6 min |
| vLLM init + CUDA graph capture | 60–90 s |
| **Total** | **~7–10 min** |

A restart with the cached volume skips the download (~2 min).

## Wait for ready

```bash
until curl -fsS -m 5 "$ENDPOINT/v1/models" >/dev/null; do sleep 15; done
```

## Smoke test

```bash
curl -sS "$ENDPOINT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "messages": [{"role":"user","content":"Say hi in 5 words."}],
    "max_tokens": 32,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

`stream_options.include_usage` is required for `xk6-llm` to read server-truth token counts; without it the extension cannot populate `prompt_tokens` / `completion_tokens`.

## Run

```bash
./build/k6 run examples/chat.js \
  -e LLM_BASE_URL="$ENDPOINT/v1" \
  -e LLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ
```

## Logs

`runpodctl` has no `logs` subcommand. Either use the web console (`https://www.runpod.io/console/pods` → Logs) or SSH:

```bash
runpodctl ssh info $POD_ID   # prints the ssh command
ssh ... 'cat /proc/1/fd/1'   # vLLM runs as PID 1
```

Common boot failures:
- `OOM during CUDA graph capture`: lower `--gpu-memory-utilization` to 0.88.
- `Model not found`: typo in model id, or the model is gated and `HF_TOKEN` is not set.

## Lifecycle

```bash
runpodctl pod stop   $POD_ID   # halts compute billing; volume still billed
runpodctl pod start  $POD_ID   # resume; HF cache survives
runpodctl pod delete $POD_ID   # terminate; deletes volume
```

A100 80GB PCIe SECURE is $1.39/hr at time of writing. Use `--stop-after '<RFC3339 timestamp>'` on `pod create` for a hard stop.

## vllm bench serve from the same pod

```bash
pip install vllm[bench]
vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-72B-Instruct-AWQ \
  --dataset-name sharegpt \
  --num-prompts 200 --request-rate 4 --seed 42
```
