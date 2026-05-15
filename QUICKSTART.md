# xk6-llm quickstart

A Docker Compose stack with Prometheus, Grafana, and a pre-provisioned xk6-llm dashboard.

## Prerequisites

- Docker
- An OpenAI-compatible inference server (vLLM, SGLang, TGI, llama.cpp server, [Ollama](https://ollama.com), NIM, OpenAI, etc.)
- A k6 binary built with this extension (`xk6 build --with github.com/msradam/xk6-llm@latest --output build/k6`)

## Run

```bash
cd quickstart
docker compose up -d
```

Prometheus listens on `:9090` (with `remote_write_receiver` and native histograms enabled). Grafana listens on `:3000` with anonymous read access. The dashboard is auto-provisioned.

```bash
K6_PROMETHEUS_RW_TREND_STATS="p(50),p(95),p(99),min,max,avg,count,sum" \
./build/k6 run \
  -o experimental-prometheus-rw=http://localhost:9090/api/v1/write \
  -e LLM_BASE_URL=http://host.docker.internal:11434/v1 \
  -e LLM_MODEL=granite4.1:3b \
  examples/chat.js
```

`K6_PROMETHEUS_RW_TREND_STATS` is required, otherwise only `p(99)` is emitted by default and most panels are empty. On Linux, replace `host.docker.internal` with `172.17.0.1`.

Open [http://localhost:3000](http://localhost:3000) → Dashboards → **xk6-llm**.

## Use with an existing Grafana

Enable `--web.enable-remote-write-receiver` and `--enable-feature=native-histograms` on Prometheus, then import `quickstart/grafana/dashboards/xk6-llm.json`.

## Tear down

```bash
docker compose down
```
