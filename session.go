package llm

import (
	"errors"
	"fmt"
	"maps"
	"strconv"
	"sync/atomic"

	"github.com/grafana/sobek"
	"go.k6.io/k6/v2/js/common"
	"go.k6.io/k6/v2/js/promises"
)

// Session wraps a Client and carries conversation state for a single
// multi-turn dialogue. Each Send appends the user message, calls chat,
// appends the assistant reply, and auto-tags `session_id`, `turn`, and
// `cache_state` so dashboards can roll up per-session.
type Session struct {
	mod      *module
	client   *Client
	id       string
	system   string
	messages []map[string]any
	turn     int
	tokens   sessionTokens
}

type sessionTokens struct {
	Prompt     int
	Completion int
}

var sessionSeq atomic.Uint64

func (m *module) newSession(call sobek.ConstructorCall) *sobek.Object {
	rt := m.vu.Runtime()
	if len(call.Arguments) == 0 {
		common.Throw(rt, errors.New("llm.Session: client is required"))
	}
	client, ok := call.Arguments[0].Export().(*Client)
	if !ok || client == nil {
		common.Throw(rt, errors.New("llm.Session: first argument must be an llm.Client"))
	}

	s := &Session{mod: m, client: client}

	if len(call.Arguments) > 1 {
		raw, ok := call.Arguments[1].Export().(map[string]any)
		if !ok {
			common.Throw(rt, errors.New("llm.Session: second argument must be an options object"))
		}
		if v, ok := raw["system"].(string); ok {
			s.system = v
		}
		if v, ok := raw["id"].(string); ok {
			s.id = v
		}
	}
	if s.id == "" {
		s.id = "s-" + strconv.FormatUint(sessionSeq.Add(1), 10)
	}
	s.resetMessages()
	return rt.ToValue(s).ToObject(rt)
}

// Send appends a user message and dispatches a chat call. The argument is
// either a string (treated as the user content) or an object with a `content`
// field plus any chat() option (max_tokens, temperature, slo, tags, etc.).
// On resolution the assistant reply is appended to history.
func (s *Session) Send(arg sobek.Value) *sobek.Promise {
	rt := s.mod.vu.Runtime()
	promise, resolve, reject := promises.New(s.mod.vu)

	content, extras, err := s.parseSendArg(arg)
	if err != nil {
		reject(err)
		return promise
	}

	s.turn++
	turn := s.turn
	s.messages = append(s.messages, map[string]any{
		"role":    "user",
		"content": content,
	})

	req := map[string]any{
		"messages": s.messagesCopy(),
	}
	maps.Copy(req, extras)

	s.injectSessionTags(req, turn)

	resultPromise := s.client.Chat(req)
	resultObj := rt.ToValue(resultPromise).ToObject(rt)
	thenFn, _ := sobek.AssertFunction(resultObj.Get("then"))

	onFulfilled := rt.ToValue(func(call sobek.FunctionCall) sobek.Value {
		v := call.Argument(0)
		res, _ := v.Export().(map[string]any)
		if res != nil {
			if c, ok := res["content"].(string); ok {
				s.messages = append(s.messages, map[string]any{
					"role":    "assistant",
					"content": c,
				})
			}
			if pt, ok := res["prompt_tokens"].(int); ok {
				s.tokens.Prompt += pt
			}
			if ct, ok := res["completion_tokens"].(int); ok {
				s.tokens.Completion += ct
			}
		}
		resolve(v.Export())
		return sobek.Undefined()
	})
	onRejected := rt.ToValue(func(call sobek.FunctionCall) sobek.Value {
		reject(call.Argument(0).Export())
		return sobek.Undefined()
	})
	_, _ = thenFn(resultObj, onFulfilled, onRejected)
	return promise
}

// Reset clears conversation history (keeping the system prompt) and rewinds
// the turn counter. Token totals are preserved across resets; use a new
// Session for fresh accounting.
func (s *Session) Reset() {
	s.resetMessages()
	s.turn = 0
}

// Id is named `Id` not `ID` because Sobek's default field-name mapper
// lowercases only the first letter; `ID` would surface in JS as `iD()`.
//
//nolint:revive // see comment above
func (s *Session) Id() string { return s.id }

// Turn returns the 1-based count of completed sends since the last reset.
func (s *Session) Turn() int { return s.turn }

// Messages returns a deep copy of the history; callers may mutate freely.
func (s *Session) Messages() []map[string]any { return s.messagesCopy() }

// Tokens returns cumulative prompt + completion counts. Prompt totals are
// summed across calls (each turn resends the full history), matching billing.
func (s *Session) Tokens() map[string]any {
	return map[string]any{
		"prompt":     s.tokens.Prompt,
		"completion": s.tokens.Completion,
		"total":      s.tokens.Prompt + s.tokens.Completion,
	}
}

func (s *Session) resetMessages() {
	s.messages = s.messages[:0]
	if s.system != "" {
		s.messages = append(s.messages, map[string]any{
			"role":    "system",
			"content": s.system,
		})
	}
}

func (s *Session) parseSendArg(arg sobek.Value) (string, map[string]any, error) {
	if arg == nil || sobek.IsUndefined(arg) || sobek.IsNull(arg) {
		return "", nil, errors.New("llm.Session.send: argument is required")
	}
	switch v := arg.Export().(type) {
	case string:
		return v, nil, nil
	case map[string]any:
		c, ok := v["content"].(string)
		if !ok {
			return "", nil, errors.New("llm.Session.send: object form requires a string `content` field")
		}
		extras := make(map[string]any, len(v))
		for k, val := range v {
			if k == "content" || k == "messages" || k == "role" {
				continue
			}
			extras[k] = val
		}
		return c, extras, nil
	default:
		return "", nil, fmt.Errorf("llm.Session.send: argument must be a string or object, got %T", v)
	}
}

func (s *Session) messagesCopy() []map[string]any {
	out := make([]map[string]any, len(s.messages))
	for i, m := range s.messages {
		copyM := make(map[string]any, len(m))
		maps.Copy(copyM, m)
		out[i] = copyM
	}
	return out
}

// injectSessionTags merges session_id + turn into req.tags without clobbering
// user-supplied tags, and stamps cache_state for turn 1 (cold) vs later (warm)
// when the caller has not set it explicitly.
func (s *Session) injectSessionTags(req map[string]any, turn int) {
	var tags map[string]any
	if existing, ok := req["tags"].(map[string]any); ok {
		tags = existing
	} else {
		tags = make(map[string]any, 2)
		req["tags"] = tags
	}
	if _, ok := tags["session_id"]; !ok {
		tags["session_id"] = s.id
	}
	if _, ok := tags["turn"]; !ok {
		tags["turn"] = strconv.Itoa(turn)
	}
	if _, ok := req["cache_state"]; !ok {
		if turn == 1 {
			req["cache_state"] = "cold"
		} else {
			req["cache_state"] = "warm"
		}
	}
}
