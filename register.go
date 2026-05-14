package llm

import "go.k6.io/k6/v2/js/modules"

const importPath = "k6/x/llm"

func init() {
	modules.Register(importPath, new(rootModule))
}
