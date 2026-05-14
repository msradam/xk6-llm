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
