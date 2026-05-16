package llm

import (
	"testing"

	"github.com/stretchr/testify/require"
	"go.k6.io/k6/v2/js/modulestest"
)

func TestModule_SessionRegisters(t *testing.T) {
	t.Parallel()
	rt := modulestest.NewRuntime(t)
	require.NoError(t, rt.SetupModuleSystem(
		map[string]any{importPath: new(rootModule)}, nil, nil,
	))
	_, err := rt.RunOnEventLoop(`let llm = require("` + importPath + `");
		if (typeof llm.Session !== "function") throw "no Session";
		let c = new llm.Client({ base_url: "http://127.0.0.1:1/v1", model: "x" });
		let s = new llm.Session(c, { system: "you are terse", id: "s-test" });
		if (s.id() !== "s-test") throw "id: " + s.id();
		if (s.turn() !== 0) throw "turn: " + s.turn();
		if (s.messages().length !== 1) throw "msgs: " + s.messages().length;
		if (s.messages()[0].role !== "system") throw "role: " + s.messages()[0].role;
		s.reset();
		if (s.turn() !== 0) throw "post-reset turn";
		let tok = s.tokens();
		if (tok.total !== 0) throw "tok.total: " + tok.total;
	`)
	require.NoError(t, err)
}

func TestSession_InjectSessionTags(t *testing.T) {
	t.Parallel()
	s := &Session{id: "s-x"}

	// Turn 1, no user tags: stamps cold + session_id + turn.
	req := map[string]any{}
	s.injectSessionTags(req, 1)
	tags, _ := req["tags"].(map[string]any)
	require.Equal(t, "s-x", tags["session_id"])
	require.Equal(t, "1", tags["turn"])
	require.Equal(t, "cold", req["cache_state"])

	// Turn 3: warm cache stamped by default.
	req = map[string]any{}
	s.injectSessionTags(req, 3)
	require.Equal(t, "warm", req["cache_state"])

	// User-supplied tags + cache_state are preserved (not clobbered).
	req = map[string]any{
		"cache_state": "cold",
		"tags":        map[string]any{"region": "us-east", "session_id": "user-override"},
	}
	s.injectSessionTags(req, 2)
	tags, _ = req["tags"].(map[string]any)
	require.Equal(t, "us-east", tags["region"])
	require.Equal(t, "user-override", tags["session_id"], "user tag must win over auto session_id")
	require.Equal(t, "2", tags["turn"], "turn still auto-stamped")
	require.Equal(t, "cold", req["cache_state"], "user cache_state preserved on turn 2")
}

func TestModule_SessionRequiresClient(t *testing.T) {
	t.Parallel()
	rt := modulestest.NewRuntime(t)
	require.NoError(t, rt.SetupModuleSystem(
		map[string]any{importPath: new(rootModule)}, nil, nil,
	))
	_, err := rt.RunOnEventLoop(`let llm = require("` + importPath + `");
		let threw = false;
		try { new llm.Session(); } catch (e) { threw = true; }
		if (!threw) throw "expected throw";
	`)
	require.NoError(t, err)
}
