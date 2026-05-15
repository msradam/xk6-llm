package llm

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"go.k6.io/k6/v2/js/modulestest"
)

func TestModule_Registers(t *testing.T) {
	t.Parallel()
	rt := modulestest.NewRuntime(t)
	require.NoError(t, rt.SetupModuleSystem(
		map[string]any{importPath: new(rootModule)}, nil, nil,
	))
	_, err := rt.RunOnEventLoop(`let llm = require("` + importPath + `");
		if (typeof llm.Client !== "function") throw "no Client";
		if (typeof llm.Dataset !== "function") throw "no Dataset";
	`)
	require.NoError(t, err)
}

func TestModule_DatasetRoundtrip(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	p := filepath.Join(dir, "ds.jsonl")
	require.NoError(t, os.WriteFile(p,
		[]byte(`{"messages":[{"role":"user","content":"a"}],"max_tokens":10}`+"\n"+
			`{"messages":[{"role":"user","content":"b"}],"max_tokens":20}`+"\n"),
		0o600))

	// JSON-encode the path so Windows backslashes are properly escaped when
	// embedded in the JS string literal below.
	jsPath, err := json.Marshal(p)
	require.NoError(t, err)

	rt := modulestest.NewRuntime(t)
	require.NoError(t, rt.SetupModuleSystem(
		map[string]any{importPath: new(rootModule)}, nil, nil,
	))
	_, err = rt.RunOnEventLoop(`
		let llm = require("` + importPath + `");
		let ds = new llm.Dataset({ path: ` + string(jsPath) + ` });
		if (ds.size() !== 2) throw "size: " + ds.size();
		let r = ds.next();
		if (r.messages[0].content !== "a") throw "first: " + r.messages[0].content;
		if (r.max_tokens !== 10) throw "max_tokens: " + r.max_tokens;
		let r2 = ds.next();
		if (r2.messages[0].content !== "b") throw "second: " + r2.messages[0].content;
		// at() does not advance the cursor
		let r3 = ds.at(0);
		if (r3.messages[0].content !== "a") throw "at(0): " + r3.messages[0].content;
		ds.reset();
		let r4 = ds.next();
		if (r4.messages[0].content !== "a") throw "reset: " + r4.messages[0].content;
	`)
	require.NoError(t, err)
}
