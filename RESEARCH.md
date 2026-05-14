# xk6-llm Research

Reference material for building and validating `xk6-llm`. Two parts:

1. **Part A — LLM inference benchmark landscape.** What existing tools measure, exactly how they compute TTFT / ITL / throughput, and which to cross-validate `xk6-llm` against.
2. **Part B — xk6 extension conventions.** Project layout, module registration, custom metrics, CI, and a canonical skeleton drawn from real Grafana extensions.

For *how to keep the code stable* (linting, testing, build verification, `make check`), see [CLAUDE.md](./CLAUDE.md).

---

# Part A — LLM Inference Benchmark Tools

A reference for validating `xk6-llm`'s TTFT / ITL / token-throughput metrics against community-standard tools. All file/line citations verified against upstream `main` in May 2026.

## A.1 vLLM `benchmark_serving` (now `vllm bench serve`)

**Repo:** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) — Apache-2.0 — ~80,020 stars — last push 2026-05-14 (very active).

> The legacy script `benchmarks/benchmark_serving.py` is a deprecation shim; the live implementation lives at `vllm/benchmarks/serve.py` and `vllm/benchmarks/lib/endpoint_request_func.py` and is invoked via `vllm bench serve`.

### Metric computation

All timestamps use `time.perf_counter()`. Per-request fields live on a `RequestFuncOutput` object that the driver fills inside the streaming loop.

- **TTFT** — measured at first SSE chunk that carries content. From `endpoint_request_func.py`:
  - Completions API (lines ~243–247): `if not first_chunk_received: ttft = time.perf_counter() - st; output.ttft = ttft`
  - Chat Completions (lines ~360–362): `if ttft == 0.0: ttft = timestamp - st; output.ttft = ttft`
  - Pooling/non-streaming backends (lines ~539–540): `output.ttft = output.latency = time.perf_counter() - st` (TTFT collapses to total latency).
- **ITL** — appended **per SSE chunk**, not per token (lines ~250–251 and ~365–366):
  ```python
  output.itl.append(timestamp - most_recent_timestamp)
  most_recent_timestamp = timestamp
  ```
  Caveat: if the server emits multi-token chunks (some TGI builds, batched configs), one "ITL sample" covers >1 tokens. vLLM streams one token per chunk by default, so this is usually accurate for vLLM, but is a real footgun when pointed at TGI / TRT-LLM / Ollama.
- **TPOT** — `serve.py` line ~897: `latency_minus_ttft / (output_len - 1)`. Excludes first token; guards against `output_len == 1`.
- **Output token throughput** — `serve.py` line ~1044: `sum(actual_output_lens) / dur_s` where `dur_s` is wall-clock duration of the whole benchmark.
- **Request throughput** — `num_completed / dur_s`.

### Arrival pattern (`get_request` in serve.py lines ~434–574)

- `request_rate == inf` → all prompts fired immediately (closed-loop concurrency only).
- `burstiness == 1.0` (default) → **Poisson** process; inter-arrival is exponential with mean `1/request_rate`.
- `burstiness != 1.0` → **Gamma**-distributed inter-arrivals; `<1` burstier, `>1` smoother.
- Supports linear/exponential **ramp-up** strategies (lines ~308–328).

### Datasets

Recognised `--dataset-name` values (`serve.py` ~line 2048): `sharegpt`, `sonnet`, `random`, `random-mm`, `custom`, `hf`, `prefix_repetition`, `hf_output_len`, `spec_bench`. ShareGPT and Sonnet are the de-facto community standards.

### Backends

`vllm`, `openai`, `openai-chat`, `openai-audio`, `openai-embeddings`, `openai-embeddings-chat`, `openai-embeddings-clip`, `openai-embeddings-vlm2vec`, `infinity-embeddings`, `infinity-embeddings-clip`, `vllm-pooling`, `vllm-rerank`. Legacy `tgi`, `tensorrt-llm`, `deepspeed-mii` backends still exist in older `backend_request_func.py`.

### Output (when `--save-result`)

JSON with top-level keys: `date`, `backend`, `model_id`, `label`, `request_throughput`, `total_token_throughput`, `output_throughput`, `max_output_tokens_per_s`, `mean_ttft_ms`, `median_ttft_ms`, `std_ttft_ms`, `p99_ttft_ms` (same for `tpot`, `itl`, `e2el`), plus per-request arrays `input_lens`, `output_lens`, `ttfts`, `itls`, `generated_texts`, `errors`, `start_times` (serve.py ~lines 1055–1058, 2170–2182).

### Footguns

- TTFT is **first-chunk** not first-token. For backends that buffer, TTFT is inflated.
- ITL granularity follows chunk granularity. Multi-token chunks → undercounted ITL count, overstated per-sample ITL.
- Tokenizer mismatch: `output_len` is recomputed client-side using the HF tokenizer named by `--tokenizer`; per-token metrics drift if the server's tokenizer differs.
- TPOT uses `output_len - 1`; if your provider returns `output_tokens` from `usage` but the streamed text re-tokenizes to a different count, TPOT can disagree with `mean(itl)`.

---

## A.2 GuideLLM (Red Hat / Neural Magic)

**Repo:** [github.com/neuralmagic/guidellm](https://github.com/neuralmagic/guidellm) — Apache-2.0 — ~1,135 stars — last push 2026-05-14 (active).

Source layout: `src/guidellm/{backends, benchmark, scheduler, schemas, data, mock_server, cli, extras, utils}`. Key modules: `benchmark/benchmarker.py`, `benchmark/profiles.py`, `backends/` (OpenAI HTTP + vLLM-native).

### Metric computation

Event-driven aggregator; per-token timestamps via `RequestStats`; summary stats in `benchmark/benchmarker.py`.

- **TTFT** — wall time from submit to first streamed content chunk (same semantic as vLLM).
- **ITL** — per-chunk delta between consecutive content chunks.
- **TPOT** — `(e2e_latency - TTFT) / (output_tokens - 1)` (matches genai-perf; differs from vLLM in that vLLM derives `output_len` from the *client-side* re-tokenization).
- **Output tokens/sec per request** — `output_tokens / e2e_latency`.
- **System output token throughput** — `sum(output_tokens) / wall_clock_window`.

### Arrival pattern (Profiles)

`benchmark/profiles.py`: `synchronous` (one in-flight at a time), `concurrent` (closed-loop fixed N), `throughput` (open-loop max-rate), `constant` (constant-rate open-loop), `poisson` (Poisson open-loop), and **`sweep`** — auto-sweeps from min-sustainable to max-throughput. The sweep is the unique value prop.

### Datasets

HuggingFace datasets directly, local `.json`/`.jsonl`/`.csv`/`.txt`, synthetic generator with configurable prompt/output token distributions. Multimodal (text + image/audio/video) in 0.6.x.

### Backends

OpenAI-compatible HTTP and vLLM-native. AsyncIO/`aiohttp`-based client.

### Output

`benchmarks.json` (full per-request timings), `benchmarks.csv` (flattened), `benchmarks.html` (interactive charts), console tables.

### Footguns

- The sweep profile estimates max throughput first; cold-start (compile, prefix-cache warm-up) can skew first sweep point. Pre-warm with `--warmup-percent`.
- Token counting either from server `usage` or client tokenizer (default depends on backend). Discrepancies show as TPOT vs `mean(ITL)` mismatch.

---

## A.3 LLMPerf (Ray)

**Repo:** [github.com/ray-project/llmperf](https://github.com/ray-project/llmperf) — Apache-2.0 — ~1,116 stars — **archived 2025-12-17** (read-only).

Primary driver `token_benchmark_ray.py`; per-API client at `src/llmperf/ray_clients/openai_chat_completions_client.py`.

### Metric computation

- **TTFT**: `if not ttft: ttft = time.monotonic() - start_time`, set when the first chunk with `delta.content` (non-None, non-empty) arrives — explicitly skips empty `[DONE]` and role-only deltas.
- **Per-token timing**: `time_to_next_token.append(time.monotonic() - most_recent_received_token_time)` on every content-bearing chunk; this list is the ITL stream.
- **Mean ITL** (`token_benchmark_ray.py` lines ~117–118): returned as a **summed** inter-token latency divided by `num_output_tokens` at aggregation — across versions, this is "mean across the whole request including TTFT-adjacent tokens" in some forms. Read the code carefully if cross-checking.
- **Output throughput**: per-request `num_output_tokens / e2e_latency`; system-level across the whole window.

### Arrival pattern

Closed-loop only. `num_concurrent_requests` Ray worker threads each loop "submit → wait → submit" until `max_num_completed_requests` or `test_timeout_s` elapses. **No Poisson, no rate control.** Biggest limitation versus vLLM/GuideLLM.

### Datasets

Random Shakespeare-sonnet lines, length-controlled via token count. Correctness suite uses synthesized "number-to-digits" prompts.

### Backends

OpenAI, Anthropic, Together, HuggingFace TGI, LiteLLM, Vertex AI, SageMaker. Tokenization is **hard-coded to `LlamaTokenizer`** for cross-provider consistency — a deliberate choice that introduces drift vs the actual server tokenizer.

### Output

Two JSON files per run: a summary (model id, concurrency, p25/p50/p75/p90/p95/p99, mean/min/max/stddev) and per-request raw.

### Footguns

- **Archived**: no fixes incoming.
- **LlamaTokenizer everywhere**: throughput not directly comparable to vLLM benchmarks.
- No open-loop / Poisson — cannot measure tail latencies under a fixed offered load.

---

## A.4 MLPerf Inference — LLM tracks

**Repo:** [github.com/mlcommons/inference](https://github.com/mlcommons/inference) — Apache-2.0 — ~1,566 stars — last push 2026-05-14. **Rules:** [inference_policies/inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc).

### Models (v5.1)

GPT-J (edge/datacenter), Llama2-70B (datacenter, Conversational and Interactive sub-categories), Llama3.1-8B, Llama3.1-405B, Mixtral-8x7B, DeepSeek-R1.

### Scenarios

- **Offline** — all queries available at t=0; reports throughput (tokens/s).
- **Server** — **LoadGen issues queries as a Poisson process** for 600 s; must meet per-benchmark TTFT and TPOT 99th-percentile SLOs.

### Latency constraints (from `inference_rules.adoc`)

| Benchmark | TTFT (ms) | TPOT (ms) |
|---|---|---|
| Llama2-70B Conversational | 2000 | 200 |
| Llama2-70B Interactive | 450 | 40 |
| Llama3.1-405B Server | 6000 | 175 |
| Llama3.1-405B Interactive | 4500 | 80 |

SLO is on the **99th percentile**, measured by LoadGen across the run.

### How metrics are measured

- LoadGen records `query_start_time` per query.
- SUT calls back `FirstTokenComplete` for streaming benchmarks → TTFT.
- TPOT = `(last_token_time - first_token_time) / (n_tokens - 1)`.
- Per-token throughput = aggregated tokens/s over valid run duration.

### Datasets

- Llama2-70B: filtered **OpenOrca**, 24,576 samples, pre-tokenized pickle.
- Llama3.1-405B: LongBench, LongDataCollections, Ruler, GovReport.
- Mixtral-8x7B: OpenOrca + MBXP + GSM8K.
- GPT-J: CNN/DailyMail summarization.

### Footguns

- "Tokens" are **dataset reference tokens**, tokenized once by the official tokenizer; SUTs must emit the reference token count distribution. Comparing arbitrary `usage.completion_tokens` against MLPerf without aligning tokenizers is invalid.
- 600 s minimum run, valid statistical sample requirements — too heavy for quick checks.
- Requires building LoadGen as a C++ library; not a drop-in HTTP client.

---

## A.5 genai-perf (NVIDIA Triton) → being replaced by AIPerf

**Repo:** [github.com/triton-inference-server/perf_analyzer](https://github.com/triton-inference-server/perf_analyzer) — BSD-3-Clause — ~143 stars — last push 2026-05-08. **Deprecated** in favor of AIPerf (repo not yet public under searched names).

### Metric definitions ([NVIDIA NIM benchmarking docs](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html))

- **TTFT** — query submit → first token receipt.
- **ITL** (and TPOT, used synonymously here) — `(e2e_latency - TTFT) / (output_tokens - 1)`. **Mean across the request**, *not* per-chunk deltas — definitional difference from vLLM/GuideLLM.
- **Time to second token** — separately reported (useful for catching prefill spillover).
- **Output Throughput Per User** = `output_seq_len / e2e_latency` (asymptotes to `1 / ITL`).
- **System output token throughput** — system-wide tokens/s over the **window between first request and last response** (excludes warmup). NIM doc explicitly notes: "GenAI-Perf and LLM-Perf calculate this differently."
- **Request throughput** — successful requests / second.

`genai-perf/genai_perf/metrics/llm_metrics.py` defines the `LLMMetrics` container.

### Arrival pattern

Three modes (from Perf Analyzer): **Concurrency** (closed-loop), **Request Rate** (open-loop constant or Poisson via `--request-distribution poisson`), **Custom Interval** (replay a timing file).

### Datasets

`--input-dataset` accepts a path; built-in synthetic generators. Multimodal supported. OpenAI completions/chat/embeddings, Triton TRT-LLM, Triton vLLM are first-class.

### Output (in `artifacts/data/`)

- `inputs.json`
- `profile_export.json` — raw Perf Analyzer per-request events.
- `profile_export_genai_perf.json` — summary + CLI args.
- `profile_export_genai_perf.csv` — printed tables.
- `images/` — HTML + JPEG plots.

### Footguns

- ITL is a **scalar mean per request**, not a vector. To compare with a vector tool, summarize before comparing.
- "Output Token Throughput" window differs from LLMPerf — don't compare directly.
- Invest in AIPerf for forward compatibility.

---

## A.6 Alternatives

### A.6a. Locust-based scripts
Plenty of community gists exist. **No standardization** — TTFT/ITL semantics are whatever the script author wrote. Not useful as validation reference. Useful as competitive landscape: `xk6-llm`'s value-add over Locust is k6's compiled-Go HTTP stack (higher per-VU throughput) and built-in time-series export.

### A.6b. OpenAI Evals
[github.com/openai/evals](https://github.com/openai/evals) — focused on **quality/accuracy**, not latency/throughput. Skip for validation.

### A.6c. NVIDIA TensorRT-LLM benchmarks
[github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NOASSERTION — ~13,642 stars — last push 2026-05-14.
- `benchmarks/cpp/gptManagerBenchmark` — C++ in-process; not HTTP. Not comparable.
- `benchmarks/Suite/` (Python) — wraps genai-perf and `vllm bench serve`. Validation is transitive.

### A.6d. AIPerf (the genai-perf successor)
NVIDIA's stated replacement. Repo not at a stable public path discoverable via `gh api` yet. Definitions documented at the NIM benchmarking site; intended to match genai-perf semantics.

---

## A.7 Cross-cutting comparison

| Tool | TTFT def | ITL def | Arrival | Tokenizer | Open-loop? |
|---|---|---|---|---|---|
| vLLM bench serve | first content chunk | per-chunk vector | Poisson + Gamma + const | client (HF, configurable) | yes |
| GuideLLM | first content chunk | per-chunk vector | sync / concurrent / Poisson / const / sweep | configurable | yes |
| LLMPerf | first content chunk | sum-of-deltas / N (scalar) | closed-loop only | hard-coded LlamaTokenizer | **no** |
| MLPerf Server | LoadGen callback at first token | scalar from totals | Poisson @ target QPS | dataset-fixed reference tokenizer | yes |
| genai-perf | first chunk (via Perf Analyzer events) | scalar `(e2e-TTFT)/(N-1)` | concurrency / req-rate (Poisson opt) / custom | configurable | yes |

The biggest definitional fork: **vLLM and GuideLLM keep ITL as a vector of per-chunk deltas; genai-perf and MLPerf use a single scalar derived from totals.** If `xk6-llm` emits per-chunk deltas as a k6 trend metric, you can produce both views.

Second fork: **chunk vs token.** "ITL" as computed by every tool above is really "inter-chunk latency". On TGI multi-token chunks are common; on vLLM one token per chunk is default. `xk6-llm` should document this explicitly.

---

## A.8 Validation strategy for `xk6-llm`

**Recommendation: cross-validate primarily against `vllm bench serve`, secondarily against GuideLLM.**

Reasons:
1. vLLM bench serve is the de-facto industry reference; actively maintained; ships in the same repo as the most popular OSS serving engine.
2. Per-chunk-vector ITL semantics are what an HTTP load tester like k6 can naturally produce.
3. GuideLLM gives a second opinion with the same semantics plus richer profile sweep.

**Avoid as primary references:** LLMPerf (archived, hard-coded tokenizer, closed-loop only), MLPerf (heavy harness, needs LoadGen C++, tokenizer-locked), genai-perf (scalar ITL — definitional mismatch).

### Reproducible cross-check recipe

**Server under test:**
- Model: `meta-llama/Meta-Llama-3-8B-Instruct` (or any 7-8B you can run).
- Serving engine: vLLM, `--port 8000 --max-model-len 4096 --disable-log-requests --seed 0`.
- Pre-warm with 50 dummy requests before measurement.

**Dataset:** ShareGPT, filtered to `200 <= prompt_tokens <= 1024` and `output_len = 256` (forced via `max_tokens=256, ignore_eos=true`). Forcing fixed output length removes EOS variance, the biggest source of cross-tool disagreement.

**Workload:**
- 500 total requests
- Open-loop, Poisson, `--request-rate 4.0` req/s
- Same server, same dataset slice, same RNG seed for both tools.

**Commands:**

```bash
# vLLM reference
vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 --request-rate 4.0 --burstiness 1.0 \
  --sharegpt-output-len 256 --ignore-eos \
  --seed 0 \
  --save-result --result-filename vllm.json

# xk6-llm under test
k6 run --out json=xk6.json xk6-llm-script.js
```

**Comparison checks (decreasing tolerance):**

1. **Token counts per request** must match exactly (±1 for BOS/EOS). If not, your tokenizer or `max_tokens` plumbing is wrong — fix first.
2. **Mean TTFT** within ±5 ms or ±2% (whichever larger). Bigger gap = measurement-point bug.
3. **p50/p99 TTFT** within ±5%.
4. **Mean ITL** within ±0.5 ms or ±5%.
5. **Output token throughput** (system) within ±2%.
6. **Request throughput** within ±1% (basically `n / wall_clock`).

**Typical disagreement diagnoses:**

- TTFT off by tens of ms → timing `response.headers_received` instead of "first SSE data line with non-empty `choices[0].delta.content`". Skip role-only first chunk OpenAI emits.
- ITL inflated on first sample → including the gap from TTFT to first content chunk. vLLM does NOT; first ITL sample is `chunk[2].t - chunk[1].t`.
- Throughput low → client-tokenizer divergence; use server `usage.completion_tokens` when available.
- Disagreement only at high concurrency → VUs saturated, accidentally running closed-loop instead of Poisson open-loop.

### Stretch: second cross-check with GuideLLM

After vLLM agreement, run GuideLLM `poisson` profile at the same rate/dataset. Agreement on TTFT/ITL means `xk6-llm`'s chunk-level timing is robust across two independent client implementations.

### What NOT to do

- Don't cross-check against LLMPerf — tokenizer divergence and closed-loop will systematically differ.
- Don't cross-check against MLPerf for day-to-day work — only if/when you want submission-grade comparability.
- Don't compare vector-ITL mean from `xk6-llm` to scalar ITL from genai-perf without reconciling: compute `(e2e - ttft) / (n - 1)` from raw and compare *that*.

---

## A.9 Sources

- [vLLM](https://github.com/vllm-project/vllm)
- [GuideLLM](https://github.com/neuralmagic/guidellm)
- [LLMPerf (archived)](https://github.com/ray-project/llmperf)
- [MLPerf inference](https://github.com/mlcommons/inference) · [rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- [Perf Analyzer / genai-perf](https://github.com/triton-inference-server/perf_analyzer)
- [NVIDIA NIM LLM benchmarking metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)
- [GenAI-Perf NVIDIA docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)
- [MLCommons Llama2-70B blog](https://mlcommons.org/2024/03/mlperf-llama2-70b/)
- [MLPerf Inference v5.0 LLM update](https://mlcommons.org/2025/04/llm-inference-v5/)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

---

# Part B — xk6 Extension Conventions

A survey of conventions across major `xk6` extensions, with citations, plus a canonical skeleton.

## B.1 Quick answers

- **Official template: YES.** [`grafana/xk6-example`](https://github.com/grafana/xk6-example). Both a GitHub template and the basis for `xk6 new`. Start here.
- **Go version (current `xk6-sql` and `xk6-example`):** `go 1.25.0`, `toolchain go1.25.10`.
- **k6 version:** `go.k6.io/k6/v2 v2.0.0-rc1`. CI pins `k6-versions: '["v2.0.0-rc1"]'` and `xk6-version: "1.4.1"`. **For a NEW extension, use the v2 import path:** `go.k6.io/k6/v2/js/modules`. Older extensions (`xk6-redis`, `xk6-browser`, `xk6-disruptor`, `xk6-kafka`) still import `go.k6.io/k6` (v1) — don't copy that.
- **Import path convention:** `k6/x/<short-name>`, where `<short-name>` is the repo name minus `xk6-`. So `xk6-llm` → **`k6/x/llm`**. Confirmed across every extension. Sub-paths are legal: `xk6-browser` uses `k6/x/browser/async`.

---

## B.2 Per-extension survey

### B.2.1 `grafana/xk6-example` (canonical template)

Root files:
```
.devcontainer/  .editorconfig  .github/  .gitignore  .vscode/
CODEOWNERS  CODE_OF_CONDUCT.md  CONTRIBUTING.md  LICENSE  Makefile  README.md
register.go  module.go  module_test.go
greeting.go   greeting_test.go
base32.go     base32_test.go
random.go     random_test.go
index.d.ts    tsconfig.json
script.js     examples/   test/
go.mod  go.sum  renovate.json
```

**Layout:** flat — all Go files at repo root in `package example`. No `pkg/` or sub-package. Recommended default for small/medium extensions.

**`register.go`:**
```go
package example

import "go.k6.io/k6/v2/js/modules"

const importPath = "k6/x/example"

func init() {
    modules.Register(importPath, new(rootModule))
}
```

**`module.go`** — `rootModule`, `module`, `Exports()`:
```go
type rootModule struct{}

func (*rootModule) NewModuleInstance(vu modules.VU) modules.Instance {
    return &module{vu}
}

type module struct {
    vu modules.VU
}

func (m *module) Exports() modules.Exports {
    return modules.Exports{
        Named: map[string]any{
            "greeting":  m.greeting,
            "b32encode": m.b32encode,
            "b32decode": m.b32decode,
            "Random":    m.random,   // constructor (uppercase)
        },
    }
}

var _ modules.Module = (*rootModule)(nil)
```

**`module_test.go`** — uses `modulestest`:
```go
import (
    "go.k6.io/k6/v2/js/modulestest"
    "github.com/stretchr/testify/require"
)

func Test_module(t *testing.T) {
    runtime := modulestest.NewRuntime(t)
    require.NoError(t, runtime.SetupModuleSystem(
        map[string]any{importPath: new(rootModule)}, nil, nil))
    _, err := runtime.RunOnEventLoop(`let mod = require("` + importPath + `")`)
    require.NoError(t, err)
}
```

**Constructor pattern (`Random`):** a Go method returns a struct whose getters/setters become JS properties via sobek reflection.

---

### B.2.2 `grafana/xk6-sql`

Layout: "root register + sub-package implementation". `register.go` in `package sql` at root, real code in `./sql/`.

```go
// register.go
package sql
import (
    "github.com/grafana/xk6-sql/sql"
    "go.k6.io/k6/v2/js/modules"
)
func init() {
    modules.Register(sql.ImportPath, sql.New())
}
```

```go
// sql/module.go
const ImportPath = "k6/x/sql"

func New() modules.Module { return new(rootModule) }

type rootModule struct{}

func (*rootModule) NewModuleInstance(vu modules.VU) modules.Instance {
    instance := &module{}
    instance.exports.Default = instance                                // Default export
    instance.exports.Named   = map[string]interface{}{"open": instance.Open}
    instance.vu = vu
    return instance
}
```

Exposes BOTH `Default` and `Named` — JS can `import sql from "k6/x/sql"` or `import { open } from "k6/x/sql"`.

README structure: badges (API Reference / Release / Go Report Card / GH Actions), one-line tagline, link to TypeDoc API docs, TypeScript usage, **Usage** with `file=examples/example.js` annotated block (mdcode), **Build** explaining `xk6 build --with github.com/grafana/xk6-sql=.`.

---

### B.2.3 `grafana/xk6-faker`

Two-package split: `register.go` at root + `module/` (k6 glue) + `faker/` (pure logic).

```go
// register.go
func register() { modules.Register(module.ImportPath, module.New()) }
func init()     { register() } //nolint:gochecknoinits
```

```go
// module/module.go
func (root *rootModule) NewModuleInstance(vu modules.VU) modules.Instance {
    mod := &module{exports: modules.Exports{
        Named:   make(map[string]interface{}),
        Default: faker.New(getseed(vu), vu.Runtime()),
    }}
    mod.exports.Named["Faker"] = faker.Constructor
    return mod
}
```

Shows **VU-aware init**: seed read from `vu.InitEnv().LookupEnv("XK6_FAKER_SEED")` inside `NewModuleInstance`.

`module/module_test.go` is the cleanest `modulestest` example — injects `InitEnvField.LookupEnv` and `InitEnvField.RuntimeOptions.Env`. Worth copying.

---

### B.2.4 `grafana/xk6-redis` (canonical async-promise pattern)

Default branch `master`. Layout: `register.go` + `redis/`. Imports old v1 (`go.k6.io/k6/js/modules`) but the patterns translate to v2 unchanged.

```go
type ModuleInstance struct {
    vu modules.VU
    *Client
}

func (mi *ModuleInstance) Exports() modules.Exports {
    return modules.Exports{Named: map[string]any{"Client": mi.NewClient}}
}

func (mi *ModuleInstance) NewClient(call sobek.ConstructorCall) *sobek.Object {
    // validate args; build *Client; return rt.ToValue(client).ToObject(rt)
}
```

**The canonical async pattern** — every method returns `*sobek.Promise`:
```go
func (c *Client) Set(key string, value any, expiration int) *sobek.Promise {
    promise, resolve, reject := promises.New(c.vu)
    go func() {
        result, err := c.redisClient.Set(c.vu.Context(), key, value, ...).Result()
        if err != nil { reject(err); return }
        resolve(result)
    }()
    return promise
}
```

`promises.New(vu)` from `go.k6.io/k6/js/promises`. `common.Throw(rt, err)` from `go.k6.io/k6/js/common` for JS exceptions.

**For `xk6-llm` (fundamentally async HTTP), follow this exactly.**

---

### B.2.5 `grafana/xk6-disruptor`

Large project: `disruptor.go` at root + `pkg/api/`, `pkg/disruptors/`, `pkg/kubernetes/`, `cmd/`, `e2e/`, `docs/`. Ships a CLI sub-command, rolls own `build.sh`/`release.sh`/`package.sh`. The "complex extension" archetype. **Not needed for `xk6-llm`.**

---

### B.2.6 `grafana/xk6-browser`

Multi-namespace extension. Registers `k6/x/browser/async` and `k6/x/browser`. Model for exposing rich object graphs (many `*_mapping.go` files). Overkill for `xk6-llm`.

---

### B.2.7 `mostafa/xk6-kafka` (best **custom metrics** reference)

Layout: `pkg/kafka/` (35+ files). Key files: `pkg/kafka/stats.go` (declarations), `pkg/kafka/compatibility_metrics.go` (emission).

**Metric declaration** (`stats.go`):
```go
import (
    "go.k6.io/k6/js/modules"
    "go.k6.io/k6/metrics"
)

type kafkaMetrics struct {
    ReaderDials *metrics.Metric
    ReaderBytes *metrics.Metric
    // ...
}

var kafkaMetricDefinitions = []kafkaMetricDefinition{
    metricDef("kafka_reader_dial_count", metrics.Counter, func(km *kafkaMetrics, m *metrics.Metric) {
        km.ReaderDials = m
    }),
    typedMetricDef("kafka_reader_message_bytes", metrics.Counter, metrics.Data, func(km *kafkaMetrics, m *metrics.Metric) {
        km.ReaderBytes = m
    }),
}

func registerKafkaMetric(reg *metrics.Registry, def kafkaMetricDefinition) (*metrics.Metric, error) {
    if def.hasValueType {
        return reg.NewMetric(def.name, def.metricType, def.valueType)
    }
    return reg.NewMetric(def.name, def.metricType)
}
```

`registerMetrics(vu)` is called from `NewModuleInstance`:
```go
func (*RootModule) NewModuleInstance(virtualUser modules.VU) modules.Instance {
    runtime := virtualUser.Runtime()
    metrics, err := registerMetrics(virtualUser)
    if err != nil { common.Throw(runtime, err) }
    // ...
}
```

**Emission** (`compatibility_metrics.go`):
```go
state := k.vu.State()
ctx   := k.vu.Context()
ctm   := state.Tags.GetCurrentValues()
sampleTags := ctm.Tags.With("topic", topic)

metrics.PushIfNotDone(ctx, state.Samples, metrics.ConnectedSamples{
    Samples: []metrics.Sample{{
        Time: time.Now(),
        TimeSeries: metrics.TimeSeries{
            Metric: k.metrics.WriterErrors,
            Tags:   sampleTags,
        },
        Value:    errorValue,
        Metadata: ctm.Metadata,
    }},
})
```

**Key APIs** (v2 paths: insert `/v2/`):
- `go.k6.io/k6/v2/metrics` — `Metric`, `Sample`, `TimeSeries`, `ConnectedSamples`, `Counter`/`Gauge`/`Trend`/`Rate`, `Data`/`Time`/`Default` value types, `PushIfNotDone`, `D(time.Duration) float64`.
- `vu.InitEnv().Registry.NewMetric(name, type[, valueType])` — registry lookup.
- `vu.State().Samples` — per-VU sample sink (outside init context only).
- `vu.State().Tags.GetCurrentValues()` — current tag set.

For `xk6-llm`: declare `llm_request_count` (Counter), `llm_request_duration` (Trend/Time), `llm_ttft` (Trend/Time), `llm_itl` (Trend/Time), `llm_prompt_tokens` (Counter), `llm_completion_tokens` (Counter), `llm_errors` (Counter).

---

## B.3 CI / workflows

### B.3.1 The shared xk6 workflows (use these)

`grafana/xk6` provides two reusable workflows nearly every modern Grafana xk6 extension consumes:

- `extension-validate.yml` — lint (golangci-lint), tests (Go matrix), build verification (`xk6 build`), runs JS test scripts, multi-platform.
- `extension-release.yml` — on tag push, matrix-builds binaries `(os × arch)`, publishes a GitHub Release with prebuilt k6+extension binaries.

`xk6-example/.github/workflows/validate.yml`:
```yaml
jobs:
  validate:
    uses: grafana/xk6/.github/workflows/extension-validate.yml@1556f094f3883e37ebff04b967fddac30681c466 # v1.4.1
    permissions:
      pages: write
      id-token: write
      contents: read
    with:
      go-version: "1.25.x"
      go-versions: '["1.25.x"]'
      golangci-lint-version: "v2.7.1"
      platforms: '["ubuntu-latest", "windows-latest", "macos-latest"]'
      k6-versions: '["v2.0.0-rc1"]'
      xk6-version: "1.4.1"
      xk6-test-pattern: "test/*.test.{j,t}s"
```

`xk6-example/.github/workflows/release.yml`:
```yaml
jobs:
  release:
    uses: grafana/xk6/.github/workflows/extension-release.yml@1556f094f3883e37ebff04b967fddac30681c466 # v1.4.1
    permissions: { contents: write }
    with:
      go-version: "1.25.x"
      os:   '["linux", "windows", "darwin"]'
      arch: '["amd64", "arm64"]'
      k6-version: "v2.0.0-rc1"
      xk6-version: "1.4.1"
```

**Use these for `xk6-llm`.** Binaries ARE published (matrix `xk6 build` per (os, arch), uploaded as release assets).

---

## B.4 Linting (`.golangci.yml`)

From [`xk6-sql/.golangci.yml`](https://raw.githubusercontent.com/grafana/xk6-sql/main/.golangci.yml):
```yaml
version: "2"
linters:
  default: all
  disable:
    - gochecknoinits   # k6 extensions register from init()
    - ireturn          # k6 module constructor returns an interface
    - exhaustruct      # options structs commonly partial
    - varnamelen       # short stdlib-style names are fine
    - wrapcheck        # wrapping is noisy in extensions
  settings:
    depguard:
      rules:
        prevent_accidental_imports:
          allow:
            - $gostd
            - github.com/stretchr/testify/require
            - go.k6.io/k6
            - github.com/grafana/sobek
            - github.com/grafana/xk6-sql
            - github.com/proullon/ramsql/driver
issues:
  max-issues-per-linter: 0
  max-same-issues: 0
formatters:
  enable:
    - gci
    - gofmt
    - gofumpt
    - goimports
```

Copy verbatim, swap depguard `allow` to include `github.com/msradam/xk6-llm` + any LLM-SDK deps you add. Shared workflow pins `golangci-lint-version: "v2.7.1"` (v2 config).

---

## B.5 Release process

Tag `vX.Y.Z` on `main`. Push tag → `release.yml` reusable workflow runs → `xk6 build` matrix-builds → uploaded as GH Release assets. **No goreleaser, no release-please.** Source also via Go modules. After publishing first version, add registry entry.

`xk6-sql` has `releases/` directory with per-version hand-edited notes (e.g. `releases/v1.0.5.md`). Optional but neat.

---

## B.6 README conventions

Every reviewed extension's README has roughly:
1. Badges (Release, Go Report Card, GH Actions, optionally API Reference).
2. `# xk6-<name>` heading, bold one-line tagline.
3. One-paragraph description.
4. (Optional) TypeScript hint with `jsconfig.json`/`tsconfig.json`.
5. **Usage** with short JS example (xk6-sql uses `mdcode` `file=examples/example.js` annotations).
6. **Build**: `xk6 build --with github.com/<org>/<repo>=.`
7. **Download** link to Releases.
8. **k6 compatibility** (registry requirement).
9. **Contribute** linking to `CONTRIBUTING.md`.

---

## B.7 Canonical skeleton for `xk6-llm`

Simplest working layout — derived from `xk6-example`, metrics from `xk6-kafka`, async-promise from `xk6-redis`:

```
xk6-llm/
├── .github/
│   └── workflows/
│       ├── validate.yml            # uses grafana/xk6/.../extension-validate.yml@v1.4.1
│       └── release.yml             # uses grafana/xk6/.../extension-release.yml@v1.4.1
├── examples/
│   └── chat.js                     # at least one runnable .js (registry requirement)
├── test/
│   └── chat.test.js                # picked up by xk6-test-pattern
├── .editorconfig
├── .gitignore
├── .golangci.yml                   # copy from xk6-sql; depguard allow xk6-llm + your SDKs
├── CODEOWNERS
├── LICENSE                         # Apache-2.0
├── README.md
├── go.mod                          # go 1.25; require go.k6.io/k6/v2 v2.0.0-rc1
├── go.sum
├── register.go                     # init() -> modules.Register("k6/x/llm", new(rootModule))
├── module.go                       # rootModule, module, Exports()
├── module_test.go                  # uses modulestest.NewRuntime + RunOnEventLoop
├── client.go                       # Client struct, JS-facing methods returning *sobek.Promise
├── client_test.go
├── options.go                      # config struct + parser
├── metrics.go                      # llmMetrics struct + register + emit helpers
├── index.d.ts                      # TypeScript types (optional but standard)
└── script.js                       # quick-start script
```

### Key file contents

**`register.go`** — side-effect separated so tests don't double-register:
```go
package llm

import "go.k6.io/k6/v2/js/modules"

const importPath = "k6/x/llm"

func init() {
    modules.Register(importPath, new(rootModule))
}
```

**`module.go`**:
```go
package llm

import "go.k6.io/k6/v2/js/modules"

type rootModule struct{}

func (*rootModule) NewModuleInstance(vu modules.VU) modules.Instance {
    m, err := registerMetrics(vu)
    if err != nil { panic(err) }
    return &module{vu: vu, metrics: m}
}

type module struct {
    vu      modules.VU
    metrics llmMetrics
}

func (m *module) Exports() modules.Exports {
    return modules.Exports{
        Named: map[string]any{
            "Client": m.newClient, // JS: new llm.Client({...})
        },
    }
}

var _ modules.Module = (*rootModule)(nil)
```

**`client.go`** — async methods returning JS promises:
```go
package llm

import (
    "github.com/grafana/sobek"
    "go.k6.io/k6/v2/js/common"
    "go.k6.io/k6/v2/js/modules"
    "go.k6.io/k6/v2/js/promises"
)

type Client struct {
    mod *module
    cfg *Options
}

func (m *module) newClient(call sobek.ConstructorCall) *sobek.Object {
    rt := m.vu.Runtime()
    if len(call.Arguments) != 1 {
        common.Throw(rt, errors.New("llm.Client requires one options argument"))
    }
    opts, err := parseOptions(call.Arguments[0].Export())
    if err != nil { common.Throw(rt, err) }
    return rt.ToValue(&Client{mod: m, cfg: opts}).ToObject(rt)
}

func (c *Client) Chat(req map[string]any) *sobek.Promise {
    promise, resolve, reject := promises.New(c.mod.vu)
    go func() {
        resp, err := c.doChat(c.mod.vu.Context(), req)
        if err != nil { reject(err); return }
        c.emit(resp)
        resolve(resp)
    }()
    return promise
}
```

**`metrics.go`** — table-driven (xk6-kafka pattern):
```go
package llm

import (
    "time"

    "go.k6.io/k6/v2/js/modules"
    "go.k6.io/k6/v2/metrics"
)

type llmMetrics struct {
    Requests         *metrics.Metric // Counter
    Errors           *metrics.Metric // Counter
    Duration         *metrics.Metric // Trend, Time
    TTFT             *metrics.Metric // Trend, Time
    ITL              *metrics.Metric // Trend, Time
    PromptTokens     *metrics.Metric // Counter
    CompletionTokens *metrics.Metric // Counter
}

func registerMetrics(vu modules.VU) (llmMetrics, error) {
    r := vu.InitEnv().Registry
    var m llmMetrics
    var err error
    if m.Requests, err = r.NewMetric("llm_requests", metrics.Counter); err != nil { return m, err }
    if m.Errors, err = r.NewMetric("llm_errors", metrics.Counter); err != nil { return m, err }
    if m.Duration, err = r.NewMetric("llm_request_duration", metrics.Trend, metrics.Time); err != nil { return m, err }
    if m.TTFT, err = r.NewMetric("llm_ttft", metrics.Trend, metrics.Time); err != nil { return m, err }
    if m.ITL, err = r.NewMetric("llm_itl", metrics.Trend, metrics.Time); err != nil { return m, err }
    if m.PromptTokens, err = r.NewMetric("llm_prompt_tokens", metrics.Counter); err != nil { return m, err }
    if m.CompletionTokens, err = r.NewMetric("llm_completion_tokens", metrics.Counter); err != nil { return m, err }
    return m, nil
}

// emit from inside the request goroutine
func (c *Client) emitTTFT(model string, ttft time.Duration) {
    state := c.mod.vu.State()
    if state == nil { return }
    ctm := state.Tags.GetCurrentValues()
    tags := ctm.Tags.With("model", model)
    metrics.PushIfNotDone(c.mod.vu.Context(), state.Samples, metrics.ConnectedSamples{
        Samples: []metrics.Sample{{
            Time:       time.Now(),
            TimeSeries: metrics.TimeSeries{Metric: c.mod.metrics.TTFT, Tags: tags},
            Value:      metrics.D(ttft),
            Metadata:   ctm.Metadata,
        }},
    })
}
```

**`module_test.go`**:
```go
package llm

import (
    "testing"

    "github.com/stretchr/testify/require"
    "go.k6.io/k6/v2/js/modulestest"
)

func Test_module(t *testing.T) {
    t.Parallel()
    rt := modulestest.NewRuntime(t)
    require.NoError(t, rt.SetupModuleSystem(
        map[string]any{importPath: new(rootModule)}, nil, nil))
    _, err := rt.RunOnEventLoop(`let llm = require("` + importPath + `")`)
    require.NoError(t, err)
}
```

For methods that hit real APIs, stub the HTTP layer (xk6-redis uses `redis/stub_test.go` — a Miniredis equivalent). Use `httptest.NewServer` for OpenAI-compatible responses.

**Workflows:** copy verbatim from `xk6-example`. No changes needed for `xk6-llm`.

**Registry PR** after first tag — append to [`grafana/k6-extension-registry/registry.yaml`](https://github.com/grafana/k6-extension-registry/blob/main/registry.yaml):
```yaml
- module: github.com/msradam/xk6-llm
  description: LLM-aware load testing — TTFT, ITL, token throughput for OpenAI-compatible servers
  imports:
    - k6/x/llm
  versions:
    - "v0.1.0"
```
Then run `k6registry -q --lint registry.yaml`.

---

## B.8 Gotchas

- **k6 v1 vs v2 import paths.** New code: `go.k6.io/k6/v2/js/modules`, `.../v2/js/promises`, `.../v2/js/common`, `.../v2/js/modulestest`, `.../v2/metrics`. Older reference extensions still use unversioned paths — don't mix.
- **`sobek`, not `goja`.** k6 forked goja → `github.com/grafana/sobek`. Old tutorials reference goja; don't use.
- **Init context restrictions.** `vu.State()` returns `nil` in init context. Don't try to emit metrics from constructor — only inside methods invoked at runtime.
- **Always guard `if state := c.vu.State(); state == nil { return }`** before sampling.
- **`metrics.D(time.Duration) float64`** converts a duration to float seconds k6 expects for `Time`-valued metrics.
- **Tag immutability.** `state.Tags.GetCurrentValues()` is a snapshot; `.Tags.With("k","v")` returns a new tag set.
- **Don't capture `vu.Context()` into a long-lived goroutine outside the request scope** — it's scoped to the iteration.

---

## B.9 Source URLs

- **Template:** [github.com/grafana/xk6-example](https://github.com/grafana/xk6-example)
- **xk6-sql:** `register.go`, `sql/module.go`, `sql/module_test.go`, `sql/module_internal_test.go`, `go.mod`, `.golangci.yml`, `.github/workflows/{validate,release}.yml`, `Makefile`, `README.md`
- **xk6-faker:** `register.go`, `module/module.go`, `module/module_test.go`
- **xk6-redis:** `register.go`, `redis/module.go`, `redis/client.go` (default branch `master`)
- **xk6-disruptor:** `disruptor.go`
- **xk6-browser:** `register.go`, `browser/registry.go`
- **xk6-kafka:** `pkg/kafka/{module.go,stats.go,compatibility_metrics.go}` (at `github.com/mostafa/xk6-kafka`)
- **Registry:** [github.com/grafana/k6-extension-registry/blob/main/registry.yaml](https://github.com/grafana/k6-extension-registry/blob/main/registry.yaml)
- **Shared workflows:** [github.com/grafana/xk6/tree/v1.4.1/.github/workflows](https://github.com/grafana/xk6/tree/v1.4.1/.github/workflows)
