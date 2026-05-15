# Contributing to xk6-llm

## Development

```bash
git clone https://github.com/msradam/xk6-llm
cd xk6-llm
make check   # fmt, vet, mod-tidy-verify, lint, security, test, xk6 build
```

The full check runs `gofumpt`, `goimports`, `go vet`, `go mod tidy -diff`, `golangci-lint` (v2.7.1), `gosec`, `govulncheck`, `go test -race`, and `xk6 build`. Individual targets exist (`make test`, `make lint`, `make build-verify`).

### One-time tool install

```bash
go install mvdan.cc/gofumpt@latest
go install golang.org/x/tools/cmd/goimports@latest
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.7.1
go install github.com/securego/gosec/v2/cmd/gosec@latest
go install golang.org/x/vuln/cmd/govulncheck@latest
go install go.k6.io/xk6/cmd/xk6@v1.4.1
```

### Iterating

```bash
go test -race -run TestParseStream ./...                 # focused test
make build-verify && ./build/k6 run examples/chat.js     # end-to-end against Ollama
```

`examples/chat.js` accepts env overrides: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_RATE`, `LLM_TIME_UNIT`, `LLM_DURATION`, `LLM_MAX_TOKENS`.

## Pull requests

- Run `make check` before opening a PR.
- Match commit subject style: a verb-first sentence in lowercase, body lines wrapped at 78.
- New metrics or option fields require a corresponding entry in [`index.d.ts`](./index.d.ts) and the [README](./README.md) metric table.

## Reporting bugs

Include the full output of `./build/k6 version`, the smallest k6 script that reproduces, and the server you pointed at. SSE captures are helpful when the bug is timing-related: `curl -s -N -X POST $URL/chat/completions ...` saved to a file shows exactly what `xk6-llm` saw.

## Writing style

Public-facing documentation and code comments avoid em dashes and marketing prose. Comments explain non-obvious *why*, not what the code visibly does. See the existing files for tone.
