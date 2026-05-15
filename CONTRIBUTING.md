# Contributing to xk6-llm

## Development

```bash
git clone https://github.com/msradam/xk6-llm
cd xk6-llm
make check   # fmt, vet, mod-tidy-verify, lint, security, test, xk6 build
```

The full check runs `gofumpt`, `goimports`, `go vet`, `go mod tidy`, `golangci-lint` (v2.7.1), `gosec`, `go test -race`, and `xk6 build`. Individual targets exist (`make test`, `make lint`, `make build-verify`). `make vuln` runs `govulncheck` separately because it can flag advisories tied to the local Go toolchain version that the CI matrix does not see.

### One-time tool install

```bash
make tools
```

This installs `gofumpt`, `goimports`, `golangci-lint` (v2.7.1), `gosec`, `govulncheck`, and `xk6` (v1.4.1) into `$GOPATH/bin`.

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

## Building on unsupported platforms

`xk6 build` only supports the platforms in its built-in allow-list (linux/{amd64,arm64,386}, darwin/{amd64,arm64}, windows/{amd64,386,arm64}). On other Linux architectures (s390x, ppc64le, riscv64) or BSDs, it errors with `invalid platform`. `go build` itself handles those targets, so a small wrapper bypasses xk6:

```bash
mkdir -p /tmp/k6build && cd /tmp/k6build

cat > go.mod <<'EOF'
module k6build

go 1.25
EOF

cat > main.go <<'EOF'
package main

import (
	k6cmd "go.k6.io/k6/v2/cmd"
)

func main() {
	k6cmd.Execute()
}
EOF

cat > ext.go <<'EOF'
package main

import _ "github.com/msradam/xk6-llm"
EOF

go get go.k6.io/k6/v2@v2.0.0
go get github.com/msradam/xk6-llm@v0.1.0
go mod tidy
go build -o ./k6 .
./k6 version
```

The wrapper is byte-equivalent to what `xk6 build` would generate. Add more `import _ "..."` lines for additional extensions.

## Reporting bugs

Include the full output of `./build/k6 version`, the smallest k6 script that reproduces, and the server you pointed at. SSE captures are helpful when the bug is timing-related: `curl -s -N -X POST $URL/chat/completions ...` saved to a file shows exactly what `xk6-llm` saw.

## Writing style

Public-facing documentation and code comments avoid em dashes and marketing prose. Comments explain non-obvious *why*, not what the code visibly does. See the existing files for tone.
