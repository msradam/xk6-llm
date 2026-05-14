# CLAUDE.md — xk6-llm

Working instructions for `xk6-llm`, a k6 extension for LLM inference load testing.

**See [RESEARCH.md](./RESEARCH.md) for:**
- Part A — LLM benchmark landscape and validation strategy (TTFT/ITL definitions, cross-check recipe against `vllm bench serve`).
- Part B — xk6 extension conventions, canonical skeleton, citations.

This file is the **runbook**: stack, layout choices, build, deterministic checks, gotchas. Keep it short. Push depth to RESEARCH.md.

---

## Stack

- **Language:** Go 1.25.x (toolchain `go1.25.10`).
- **k6 API:** `go.k6.io/k6/v2 v2.0.0-rc1` — **always the `/v2/` import paths** (`go.k6.io/k6/v2/js/modules`, `.../js/promises`, `.../js/common`, `.../js/modulestest`, `.../metrics`).
- **JS runtime binding:** `github.com/grafana/sobek` (NOT `dop251/goja`).
- **xk6:** `1.4.1`.
- **golangci-lint:** v2.7.1 (config `version: "2"`).
- **License:** Apache-2.0.
- **Import path (JS-side):** `k6/x/llm`. The Go module path is `github.com/msradam/xk6-llm`.

---

## Project layout

Flat layout (xk6-example style), with metrics split out:

```
xk6-llm/
├── .github/workflows/{validate,release}.yml   # use grafana/xk6 reusable workflows @v1.4.1
├── examples/chat.js                            # at least one runnable .js (registry req)
├── test/chat.test.js
├── .editorconfig  .gitignore  .golangci.yml
├── CODEOWNERS  LICENSE  README.md  Makefile
├── go.mod  go.sum
├── register.go                                 # init() — only side effect
├── module.go                                   # rootModule, module, Exports()
├── module_test.go                              # modulestest.NewRuntime + RunOnEventLoop
├── client.go                                   # Client struct, JS-facing async methods
├── client_test.go                              # httptest.NewServer for OpenAI mock
├── options.go                                  # config parser
├── metrics.go                                  # llmMetrics struct + registerMetrics + emit
├── index.d.ts                                  # TS types for editor autocomplete
└── script.js                                   # quick-start
```

Skeleton code (canonical patterns from `xk6-example`, `xk6-redis`, `xk6-kafka`) is in [RESEARCH.md §B.7](./RESEARCH.md). Copy from there.

---

## Build

```bash
# Local dev: build a custom k6 with this extension
go install go.k6.io/xk6/cmd/xk6@latest
xk6 build --with github.com/msradam/xk6-llm=. --output build/k6
./build/k6 version

# Run a script
./build/k6 run examples/chat.js
```

`go build ./...` alone catches compile errors but doesn't verify the xk6 integration. Always also run `xk6 build` before pushing.

---

## Deterministic checks

Run before every commit. Ordered fastest → slowest so failures surface early.

### Quick (~5s)

```bash
make fmt vet
```

- `gofumpt -l -w .`
- `goimports -local github.com/msradam/xk6-llm -w .`
- `go vet ./...`

### Pre-commit (~30s)

```bash
make fmt vet modtidy-verify lint test
```

Adds:
- `go mod tidy && git diff --exit-code go.mod go.sum` (must be no-op)
- `go mod verify`
- `golangci-lint run` (config below)
- `go test -count 1 -race -coverprofile=coverage.txt -timeout 2m ./...`

### Full (~1–3min, before push / PR)

```bash
make check
```

Adds:
- `gosec -quiet ./...`
- `govulncheck ./...`
- `xk6 build --with github.com/msradam/xk6-llm=. --output build/k6 && ./build/k6 version`

### `Makefile`

```makefile
SHELL := bash
.SHELLFLAGS := -e -o pipefail -c
MODULE := github.com/msradam/xk6-llm

.PHONY: check fmt vet lint test build-verify modtidy-verify security
check: fmt vet modtidy-verify lint security test build-verify

fmt:
	gofumpt -l -w .
	goimports -local $(MODULE) -w .
	@test -z "$$(git status --porcelain '*.go')" || { \
	  echo "fmt produced changes; commit them"; git diff --stat; exit 1; }

vet:
	go vet ./...

modtidy-verify:
	go mod tidy
	@test -z "$$(git status --porcelain go.mod go.sum)" || { \
	  echo "go.mod/go.sum not tidy; run 'go mod tidy' and commit"; exit 1; }
	go mod verify

lint:
	golangci-lint run

security:
	gosec -quiet ./...
	govulncheck ./...

test:
	go test -count 1 -race -coverprofile=coverage.txt -timeout 2m ./...

build-verify:
	@command -v xk6 >/dev/null || go install go.k6.io/xk6/cmd/xk6@latest
	xk6 build --with $(MODULE)=. --output build/k6
	./build/k6 version
```

### One-time tool install

```bash
go install mvdan.cc/gofumpt@latest
go install golang.org/x/tools/cmd/goimports@latest
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.7.1
go install github.com/securego/gosec/v2/cmd/gosec@latest
go install golang.org/x/vuln/cmd/govulncheck@latest
go install go.k6.io/xk6/cmd/xk6@latest
```

### `.golangci.yml`

```yaml
version: "2"
linters:
  default: all
  disable:
    - gochecknoinits   # k6 extensions register from init()
    - ireturn          # extension constructors return an interface
    - exhaustruct      # options structs are partial
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
            - github.com/msradam/xk6-llm
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

---

## Testing conventions

- **`-race` is mandatory.** Every public method runs concurrently from N VUs.
- **Table-driven + `t.Parallel()`** per case. Use `github.com/stretchr/testify/require`.
- **Module-level smoke test** via `modulestest.NewRuntime(t)` + `RunOnEventLoop`. See `xk6-faker/module/module_test.go` for the cleanest example (also injects `InitEnvField.LookupEnv` and `RuntimeOptions.Env`).
- **HTTP under test:** `httptest.NewServer` returning canned OpenAI SSE responses. Don't hit real APIs in unit tests. Include cases for: role-only first chunk (must be skipped for TTFT), multi-token chunks, `[DONE]` sentinel, mid-stream errors, server emitting `usage` block, server omitting `usage` block.
- **Coverage target:** 70–80% of non-glue code. The `init()` / `NewModuleInstance` glue is best covered by the `.js` smoke test under `xk6 build`.
- **Use `_internal_test.go`** for tests needing unexported access; `_test.go` with `package llm_test` for black-box tests (xk6-sql convention).

---

## CI

Use the Grafana reusable workflows. Pin to the `v1.4.1` SHA.

`.github/workflows/validate.yml`:
```yaml
name: Validate
permissions: {}
on:
  workflow_dispatch:
  push: { branches: [main] }
  pull_request: { branches: [main] }
jobs:
  validate:
    uses: grafana/xk6/.github/workflows/extension-validate.yml@1556f094f3883e37ebff04b967fddac30681c466 # v1.4.1
    permissions: { contents: read, pages: write, id-token: write }
    with:
      go-version: "1.25.x"
      go-versions: '["1.25.x"]'
      golangci-lint-version: "v2.7.1"
      platforms: '["ubuntu-latest","windows-latest","macos-latest"]'
      k6-versions: '["v2.0.0-rc1"]'
      xk6-version: "1.4.1"
      xk6-test-pattern: "test/*.test.{j,t}s"
```

`.github/workflows/release.yml`:
```yaml
name: Release
on: { push: { tags: ['v*'] } }
jobs:
  release:
    uses: grafana/xk6/.github/workflows/extension-release.yml@1556f094f3883e37ebff04b967fddac30681c466 # v1.4.1
    permissions: { contents: write }
    with:
      go-version: "1.25.x"
      os:   '["linux","windows","darwin"]'
      arch: '["amd64","arm64"]'
      k6-version: "v2.0.0-rc1"
      xk6-version: "1.4.1"
```

---

## Releasing

1. Bump version, update changelog (or `releases/vX.Y.Z.md` per xk6-sql convention).
2. `make check` must pass.
3. `git tag vX.Y.Z && git push --tags`.
4. Release workflow matrix-builds k6+extension binaries and publishes a GitHub Release.
5. First release: open registry PR against `grafana/k6-extension-registry` adding:
   ```yaml
   - module: github.com/msradam/xk6-llm
     description: LLM-aware load testing — TTFT, ITL, token throughput for OpenAI-compatible servers
     imports:
       - k6/x/llm
     versions:
       - "v0.1.0"
   ```
   Then `k6registry -q --lint registry.yaml`.

---

## Gotchas (frequent mistakes)

- **Don't mix v1 and v2 k6 import paths.** Always `go.k6.io/k6/v2/...` in new code. Reference extensions (`xk6-redis`, `xk6-kafka`, `xk6-disruptor`, `xk6-browser`) still use v1 — the patterns translate, the paths don't.
- **`vu.State()` returns `nil` in init context.** Guard before emitting metrics. Only safe inside methods invoked at runtime.
- **`vu.Context()` is iteration-scoped.** Pass it into goroutines explicitly; don't capture a long-lived reference.
- **TTFT must be timed at the first SSE chunk with non-empty `choices[0].delta.content`** — skip the role-only first chunk OpenAI emits. See [RESEARCH.md §A.1](./RESEARCH.md) for the exact vLLM reference behavior.
- **First ITL sample is `chunk[2].t - chunk[1].t`**, NOT `chunk[1].t - ttft`. vLLM does not include the TTFT-to-first-content gap as an ITL sample.
- **Tokens: prefer server `usage.completion_tokens`** when present; client-side tokenization drift is the #1 cause of throughput-comparison mismatches with `vllm bench serve`.
- **Apache-2.0, not MIT.** Registry strongly prefers Apache-2.0.
- **The `xk6` topic on the GitHub repo is required** for the registry to find you.
- **`sobek`, not `goja`.** k6 forked. Older tutorials are wrong.

---

## Validation against community standard

Primary cross-check: `vllm bench serve` with a vLLM server, same dataset slice, same Poisson rate, same seed, fixed `max_tokens` and `ignore_eos`. Tolerances and the exact recipe are in [RESEARCH.md §A.8](./RESEARCH.md). If `xk6-llm` mean TTFT differs from vLLM's by more than ~5ms / 2% on the same setup, there's a measurement-point bug — fix before shipping.
