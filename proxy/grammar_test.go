// Tests for buildResponseFormat — the schema-constrained sampling path
// (#33). These tests pin the response_format payload shape that goes
// over the wire to llama-server, so a regression that silently flips
// the default back to loose JSON gets caught.

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestBuildResponseFormat_DefaultIsStrictSchema(t *testing.T) {
	// Default mode (no env var set) must produce the schema-constrained
	// payload. This is the #33 perf optimization — losing the schema
	// means we silently regress to the wasted-token retry pattern.
	t.Setenv("ATLAS_GRAMMAR_MODE", "")
	rf := buildResponseFormat()

	m, ok := rf.(map[string]interface{})
	if !ok {
		t.Fatalf("strict mode should return map[string]interface{}, got %T", rf)
	}
	if m["type"] != "json_object" {
		t.Errorf("expected type=json_object, got %v", m["type"])
	}
	if _, has := m["schema"]; !has {
		t.Error("strict mode must include a 'schema' key — without it " +
			"llama-server falls back to plain valid-JSON-only enforcement " +
			"and the #33 optimization no-ops")
	}
}

func TestBuildResponseFormat_LooseDropsSchema(t *testing.T) {
	// Escape hatch: ATLAS_GRAMMAR_MODE=loose reverts to the pre-#33
	// "any valid JSON" behavior. Used when a model handles the schema
	// poorly or for debugging. The payload must NOT include 'schema'.
	t.Setenv("ATLAS_GRAMMAR_MODE", "loose")
	rf := buildResponseFormat()

	m, ok := rf.(map[string]string)
	if !ok {
		t.Fatalf("loose mode should return map[string]string, got %T", rf)
	}
	if m["type"] != "json_object" {
		t.Errorf("expected type=json_object, got %v", m["type"])
	}
}

func TestBuildResponseFormat_UnknownModeDefaultsToStrict(t *testing.T) {
	// Anything other than "loose" should fall through to strict.
	// Future modes (e.g. "json_schema" for the OpenAI-style payload)
	// would need explicit branches; until then unknown = strict, never
	// silently regress to loose.
	t.Setenv("ATLAS_GRAMMAR_MODE", "experimental-future-thing")
	rf := buildResponseFormat()
	m, ok := rf.(map[string]interface{})
	if !ok {
		t.Fatalf("unknown mode should still produce strict payload, "+
			"got %T (would silently lose the schema)", rf)
	}
	if _, has := m["schema"]; !has {
		t.Error("unknown mode must default to strict (schema included)")
	}
}

func TestDemoBaselineExcludesOrchestrationTool(t *testing.T) {
	ctx := &AgentContext{
		BypassV3: true,
		Messages: []AgentMessage{{Role: "user", Content: "build the project"}},
	}
	prompt := buildSystemPrompt(ctx)
	if strings.Contains(prompt, "plan_tasks") {
		t.Fatal("prompt advertises the removed plan_tasks tool")
	}
	// With no orchestration exclusions, the baseline needs no override
	// grammar on a plain first step.
	if _, grammar := buildStepRequest(ctx); grammar != "" {
		t.Fatalf("baseline unexpectedly received override grammar: %q", grammar)
	}
}

func TestNormalAgentPromptOmitsRemovedTools(t *testing.T) {
	ctx := &AgentContext{Messages: []AgentMessage{{Role: "user", Content: "build the project"}}}
	if prompt := buildSystemPrompt(ctx); strings.Contains(prompt, "plan_tasks") {
		t.Fatal("prompt still mentions the removed plan_tasks tool")
	}
	_, grammar := buildStepRequest(ctx)
	if grammar != "" {
		t.Fatalf("normal agent unexpectedly received override grammar: %q", grammar)
	}
}

// TestSchemaConstrained_ReachesLlamaServerOverTheWire is the
// integration-shaped end of #33: it spins up a fake llama-server with
// httptest, captures the actual JSON the proxy POSTs, and verifies the
// schema field made it through. The unit tests above pin the helper's
// output; this one proves the helper's output actually flows into the
// callLLMOnceWithGrammar request body without getting dropped, renamed,
// or shadowed by a later assignment.
//
// Without this test a future refactor could accidentally route
// callLLMOnceWithGrammar through a different request-construction
// path that ignores buildResponseFormat() and silently regress to
// loose JSON. The user-visible symptom would be "Lens / ASA stay
// happy but token throughput slowly tanks" — exactly the class of
// regression that's hardest to spot without an explicit guard.
func TestSchemaConstrained_ReachesLlamaServerOverTheWire(t *testing.T) {
	t.Setenv("ATLAS_GRAMMAR_MODE", "strict")

	var (
		mu          sync.Mutex
		capturedReq map[string]interface{}
	)

	// Fake llama-server: capture the inbound request body, then return
	// the minimal SSE stream the proxy's streaming reader needs to
	// complete without error. We only need to reach the request-write
	// step; the response can be a no-op DONE.
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			var parsed map[string]interface{}
			_ = json.Unmarshal(body, &parsed)
			mu.Lock()
			capturedReq = parsed
			mu.Unlock()
			// Stream a minimal valid response so callLLMOnceWithGrammar
			// returns without an error path we'd need to handle.
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			fl, _ := w.(http.Flusher)
			io.WriteString(w, `data: {"choices":[{"delta":{"content":"{\"type\":\"done\",\"summary\":\"ok\"}"},"finish_reason":"stop"}],"usage":{"total_tokens":1}}`+"\n\n")
			if fl != nil {
				fl.Flush()
			}
			io.WriteString(w, "data: [DONE]\n\n")
			if fl != nil {
				fl.Flush()
			}
		}))
	defer srv.Close()

	ctx := &AgentContext{
		InferenceURL: srv.URL,
		Ctx:          context.Background(),
		Messages: []AgentMessage{
			{Role: "user", Content: "hi"},
		},
	}

	// Fire the LLM call. We ignore the returned content — we only care
	// that the request body the proxy POSTed to our fake llama-server
	// includes the schema field.
	_, _, err := callLLMOnceWithGrammar(ctx, ctx.Messages, 0.3, "")
	if err != nil {
		// Streaming reader might still error on the minimal payload;
		// that's fine as long as the request was actually sent.
		t.Logf("callLLMOnceWithGrammar returned err (expected on minimal fake): %v", err)
	}

	mu.Lock()
	got := capturedReq
	mu.Unlock()

	if got == nil {
		t.Fatal("fake llama-server never received a request — proxy did " +
			"not POST anything (test infrastructure broken)")
	}

	rf, ok := got["response_format"].(map[string]interface{})
	if !ok {
		t.Fatalf("response_format missing or wrong type in request body, "+
			"got: %v", got["response_format"])
	}
	if rf["type"] != "json_object" {
		t.Errorf("response_format.type = %v, want json_object", rf["type"])
	}
	if _, hasSchema := rf["schema"]; !hasSchema {
		t.Errorf("response_format on the wire MISSING schema field — "+
			"the #33 optimization regressed to loose JSON. "+
			"request body: %v", got)
	}
	kwargs, ok := got["chat_template_kwargs"].(map[string]interface{})
	if !ok || kwargs["enable_thinking"] != false {
		t.Fatalf("chat_template_kwargs.enable_thinking = %v, want false",
			got["chat_template_kwargs"])
	}
	if _, legacy := got["enable_thinking"]; legacy {
		t.Fatal("enable_thinking must be nested under chat_template_kwargs")
	}

	// Strict mode should NOT also send a `grammar` field (mixing the
	// two confuses llama-server).
	if g, hasGrammar := got["grammar"]; hasGrammar {
		t.Errorf("strict mode should not send 'grammar' field alongside "+
			"schema-constrained response_format — llama-server rejects "+
			"requests with both. got grammar=%s", asString(g))
	}
}

// asString is a tiny stringification helper for test error messages.
// Defined inside _test so it doesn't leak into production binaries.
func asString(v interface{}) string {
	b, _ := json.Marshal(v)
	return strings.TrimSpace(string(b))
}

func TestBuildResponseFormat_SchemaMatchesToolRegistry(t *testing.T) {
	// The schema embedded in the response_format must match what
	// buildToolCallSchema() produces. If the two diverge, llama-server's
	// token sampler would constrain output to a stale set of tools and
	// the agent loop would reject responses from the model.
	t.Setenv("ATLAS_GRAMMAR_MODE", "strict")
	rf := buildResponseFormat()
	m := rf.(map[string]interface{})
	embedded, ok := m["schema"].(map[string]interface{})
	if !ok {
		t.Fatalf("schema field should be map[string]interface{}, got %T",
			m["schema"])
	}
	canonical := buildToolCallSchema()
	if len(embedded) != len(canonical) {
		t.Errorf("schema field has %d top-level keys, canonical has %d "+
			"— drift between buildResponseFormat and buildToolCallSchema",
			len(embedded), len(canonical))
	}
}
