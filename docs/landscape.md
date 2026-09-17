---
type: reference
---

# Where xk6-llm sits

Research notes as of 2026-09-16, gathered while preparing the extension for review by the k6 team. Each claim carries its source; anything that could not be verified says so.

## The k6 extension registry

The registry has two tiers, `official` and `community` ([schema](https://registry.k6.io/schema/)). Official extensions are "owned and maintained by Grafana Labs, with support for a wide range of versions"; community extensions are "developed by the community, with support for specific versions" ([docs](https://grafana.com/docs/k6/latest/extensions/explore/)). Living under the `grafana` GitHub organisation does not by itself confer the official tier ([registry README](https://github.com/grafana/k6-extension-registry)). No documented process moves a community extension to official; that conversation happens with the k6 team directly.

Listing requires an allowed license, a README, a valid `go.mod`, at least one runnable example under `examples/`, at least one versioned release, and a golangci-lint workflow on every PR and merge to main ([submit an extension](https://grafana.com/docs/k6/latest/extensions/submit-extension/)). `xk6 lint --preset community` runs the same checks locally: security (gosec), vulnerability (govulncheck), module, readme, license, git, versions, build, smoke, examples, types and codeowners. This repository passes the `community` preset as of this commit.

Grafana Cloud k6 provisions binaries only for a fixed list of extensions and does not build custom xk6 binaries ([use k6 extensions in Grafana Cloud](https://grafana.com/docs/grafana-cloud/testing/k6/author-run/use-k6-extensions/)). No LLM or inference extension is on that list. This extension runs with k6 OSS, the k6 Operator or a self-hosted runner, and its metrics still reach Grafana through the Prometheus remote write or OpenTelemetry outputs.

Grafana's earlier `grafana/xk6-ai`, a proof of concept for agent evaluation over A2A with validator models, was archived to `grafana-cold-storage` on 2026-06-05 ([archive](https://github.com/grafana-cold-storage/xk6-ai)). It answered a different question (is the agent's output acceptable) from this extension (does the inference path hold under load), and it is the nearest precedent a reviewer will remember.

## Grafana products this extension talks to

Grafana Agent Observability reached general availability on 2026-07-27 ([press release](https://grafana.com/press/2026/07/27/grafana-labs-ships-six-tools-that-power-agentic-operations-from-planning-to-production/)), after a public preview from 2026-04-21 under the name AI Observability. Its SDKs (Go, Python, TypeScript, Java, .NET) emit two channels: OTLP `gen_ai.*` spans and histograms through the normal OTel pipeline, and generation, tool execution, embedding, workflow step and score records to a dedicated ingest endpoint ([Go SDK README](https://github.com/grafana/agento11y/blob/main/go/README.md)). This extension uses the second channel only, through the Go SDK at v0.18.0, which is the newest `go/` tag ([changelog](https://github.com/grafana/agento11y/blob/main/go/CHANGELOG.md)). The SDK has no field for synthetic traffic; the extension tags records itself, and a reserved field is a reasonable request to make of the Agent Observability team.

`grafana/ai-sdk` is a Go port of Vercel's AI SDK ([pkg.go.dev](https://pkg.go.dev/github.com/grafana/ai-sdk)), and its `ai-gateway/` module is a service that fronts model providers with authentication, policy, routing and fallback. The gateway accepts ProviderWire V4 (the SDK's native wire, which this extension speaks as `providerwire-v4`), OpenAI chat completions and Responses, and Anthropic Messages, and it exports Prometheus, OTLP and agento11y ([gateway README](https://github.com/grafana/ai-sdk/tree/main/ai-gateway)). Its own Prometheus metrics (`aisdk_model_time_to_first_output_seconds`, `aisdk_model_inter_chunk_delay_seconds`, and so on) measure the gateway from the inside; this extension measures it from the outside, which is how a flush-buffering regression that passed a curl smoke test was caught. No product page for a hosted "Grafana AI Gateway" was found; as far as could be verified it is an open-source component.

Grafana Assistant, the Grafana Cloud OpenLIT integration, k6 Studio 2.0 and Grafana Cloud Traces are adjacent but not integration targets today. OpenLIT dashboards ingest OTLP `gen_ai.*` data, which a future OpenTelemetry output mapping could feed.

## k6 itself

k6 v2.2.0 (2026-08-10) is current ([releases](https://github.com/grafana/k6/releases)). v2.0.0 moved the module path to `go.k6.io/k6/v2` and changed the OTel output flags; v2.1.0 added the `native-histograms` feature flag for trend metrics; v2.2.0 added `handleSummaryTimeout` and cloud log streaming. Nothing added a per-VU teardown hook: [grafana/k6#5382](https://github.com/grafana/k6/issues/5382) is open, which is why `client.flush()` exists. No `k6/x/ai` or LLM module exists in k6 or the registry.

## What the other benchmarking tools measure

Compared against `vllm bench serve` ([docs](https://docs.vllm.ai/en/latest/contributing/benchmarks.html)), NVIDIA AIPerf ([repo](https://github.com/ai-dynamo/aiperf)), inference-perf ([repo](https://github.com/kubernetes-sigs/inference-perf)), GuideLLM ([repo](https://github.com/vllm-project/guidellm)), llm-load-test ([repo](https://github.com/openshift-psap/llm-load-test)) and LLMPerf ([repo](https://github.com/ray-project/llmperf)).

| Capability | Elsewhere | Here |
|---|---|---|
| Streaming TTFT, ITL, TPOT, end-to-end latency | all | yes, cross-validated against vLLM within 5% |
| Goodput from per-metric SLOs | vllm, AIPerf, inference-perf | yes, same semantics as `vllm bench serve --goodput` |
| Constant and ramping arrival rates, open model | all | yes, k6 executors |
| Poisson or gamma arrival processes | vllm, AIPerf, inference-perf, GuideLLM | no. k6 arrival-rate executors are deterministic. A parity run at the same mean rate agreed with vLLM's Poisson run, so the gap is in shape, not mean. |
| Rate sweeps to find saturation | AIPerf, inference-perf, GuideLLM | by hand, with one scenario per rate |
| Named datasets (ShareGPT, sonnet, HF) | vllm, AIPerf, inference-perf, GuideLLM | ShareGPT through a converter; any JSONL through `llm.Dataset` |
| Synthetic prompts with token-length distributions | AIPerf, GuideLLM, LLMPerf | no |
| Output length forcing | vllm, AIPerf | yes, `ignore_eos` and `max_tokens` |
| Warmup excluded from statistics | AIPerf, inference-perf, GuideLLM | by scenario tags and threshold selectors, not built in |
| Percentiles, mean, min, max | all | yes, k6 trend statistics |
| Server-side Prometheus scraped into the report | AIPerf, inference-perf | no; the Grafana dashboard joins them instead |
| Multi-turn conversations | AIPerf, GuideLLM, inference-perf | yes, `llm.Session` |
| Tool calling and agent loops | none | yes |
| Abandoned requests | none | yes, `abort_after_ms` and `abort_after_tokens` |
| Cost and energy per request | none | yes |
| Anthropic, Responses and AI SDK wires | vllm (OpenAI only), AIPerf (OpenAI only) | yes |
| Export of every call to an observability product | none | yes, Agent Observability |

The two things worth adding next are a Poisson arrival helper and a synthetic prompt generator with token-length distributions. Both belong in script-side helpers and neither blocks the current use cases.

## OpenTelemetry GenAI semantic conventions

The conventions are in development, not stable. On 2026-06-12 (semconv v1.42.0) they moved out of the main repository into [open-telemetry/semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai), which has no tagged release. The client metrics are `gen_ai.client.token.usage`, `gen_ai.client.operation.duration`, `gen_ai.client.operation.time_to_first_chunk` and `gen_ai.client.operation.time_per_output_chunk`; the `gen_ai.server.*` names are reserved for the server's own measurement ([metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md)). The required attributes are `gen_ai.operation.name` and `gen_ai.provider.name`; `gen_ai.system` is gone. The README records the mapping and keeps the `llm_*` names stable while the convention settles.
