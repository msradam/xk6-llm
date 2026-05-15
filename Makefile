SHELL := bash
.SHELLFLAGS := -e -o pipefail -c
MODULE := github.com/msradam/xk6-llm
XK6_VERSION := v1.4.1

GOBIN := $(shell go env GOPATH)/bin
export PATH := $(GOBIN):$(PATH)

.PHONY: check fmt vet lint test build-verify modtidy-verify security vuln tools clean
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

# vuln is intentionally NOT part of `check`. govulncheck reports advisories
# scoped to the local Go toolchain version, which can fail locally while CI
# (pinned to 1.25.x) is clean. Run on demand or in scheduled CI.
vuln:
	govulncheck ./...

test:
	go test -count 1 -race -coverprofile=coverage.txt -timeout 2m ./...

build-verify:
	@command -v xk6 >/dev/null || go install go.k6.io/xk6/cmd/xk6@$(XK6_VERSION)
	xk6 build --with $(MODULE)=. --output build/k6
	./build/k6 version

tools:
	go install mvdan.cc/gofumpt@latest
	go install golang.org/x/tools/cmd/goimports@latest
	go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.7.1
	go install github.com/securego/gosec/v2/cmd/gosec@latest
	go install golang.org/x/vuln/cmd/govulncheck@latest
	go install go.k6.io/xk6/cmd/xk6@$(XK6_VERSION)

clean:
	rm -rf build/ coverage.txt
