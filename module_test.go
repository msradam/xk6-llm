package llm

import (
	"testing"

	"github.com/stretchr/testify/require"
	"go.k6.io/k6/v2/js/modulestest"
)

func Test_module_registers(t *testing.T) {
	t.Parallel()
	rt := modulestest.NewRuntime(t)
	require.NoError(t, rt.SetupModuleSystem(
		map[string]any{importPath: new(rootModule)}, nil, nil))
	_, err := rt.RunOnEventLoop(`let llm = require("` + importPath + `"); if (typeof llm.Client !== "function") throw "no Client";`)
	require.NoError(t, err)
}
