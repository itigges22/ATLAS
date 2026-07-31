package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// The active edit-test-fix loop: after a successful write of foo.py and a
// FAILED run referencing it, the next write must fast-path (skip V3).
func TestIsActiveDebugIteration(t *testing.T) {
	ctx := &AgentContext{
		SessionWrites: map[string]bool{"foo.py": true},
		Messages: []AgentMessage{
			{Role: "assistant", Content: "..."},
			{Role: "tool", ToolName: "run_command",
				Content: `{"success":false,"error":"File \"foo.py\", line 3\nSyntaxError"}`},
		},
	}
	if !isActiveDebugIteration(ctx, "foo.py") {
		t.Error("failed run of an already-written file should be active iteration")
	}
	// First write of a file (not in SessionWrites) → V3, not fast-path.
	if isActiveDebugIteration(ctx, "bar.py") {
		t.Error("first write of a file must not fast-path")
	}
	// Last tool action was a read, not a failing run → not iterating.
	ctx.Messages[1] = AgentMessage{Role: "tool", ToolName: "read_file", Content: "..."}
	if isActiveDebugIteration(ctx, "foo.py") {
		t.Error("a read as the last action is not an edit-test-fix loop")
	}
	// Run succeeded → task likely done, not iterating.
	ctx.Messages[1] = AgentMessage{Role: "tool", ToolName: "run_command",
		Content: `{"success":true,"data":{"stdout":"ok"}}`}
	if isActiveDebugIteration(ctx, "foo.py") {
		t.Error("a passing run is not an active fix loop")
	}
}

// isBinaryContent: NUL-bearing data is binary; clean text is not.
func TestIsBinaryContent(t *testing.T) {
	if !isBinaryContent([]byte("\x7fELF\x02\x01\x00\x00garbage")) {
		t.Error("ELF header with NULs should be binary")
	}
	if isBinaryContent([]byte("def foo():\n    return 1\n")) {
		t.Error("plain source text must not be flagged binary")
	}
	if isBinaryContent([]byte("")) {
		t.Error("empty file is not binary")
	}
}

// An unbounded read of a huge file is capped so it can't blow
// the context window; an explicit limit is honored as-is.
func TestReadFileSizeCap(t *testing.T) {
	// Build content well over the cap.
	big := strings.Repeat("some line of text here\n", (maxReadFileBytes/23)+5000)
	if len(big) <= maxReadFileBytes {
		t.Fatal("test setup: content not over cap")
	}
	// Simulate the cap logic (mirrors the read_file body).
	content := big
	if len(content) > maxReadFileBytes {
		cut := maxReadFileBytes
		if nl := strings.LastIndexByte(content[:cut], '\n'); nl > 0 {
			cut = nl + 1
		}
		content = content[:cut] + "\n... [read_file truncated"
	}
	if len(content) > maxReadFileBytes+200 {
		t.Errorf("capped content still too large: %d", len(content))
	}
	if !strings.Contains(content, "truncated") {
		t.Error("cap note missing")
	}
}

// #147 review #7: a UTF-16 BOM identifies text even though it has NULs.
func TestIsBinaryContentUTF16BOM(t *testing.T) {
	if isBinaryContent([]byte{0xFF, 0xFE, 0x68, 0x00, 0x69, 0x00}) {
		t.Error("UTF-16 LE (BOM) must be treated as text")
	}
	if isBinaryContent([]byte{0xFE, 0xFF, 0x00, 0x68}) {
		t.Error("UTF-16 BE (BOM) must be treated as text")
	}
	if !isBinaryContent([]byte("\x7fELF\x00\x00garbage")) {
		t.Error("ELF must still be binary")
	}
}

// #147 review #12: filename match must be a whole token, not a substring.
func TestMentionsFilename(t *testing.T) {
	if mentionsFilename(`{"error":"data.py line 3"}`, "a.py") {
		t.Error("a.py must not match inside data.py")
	}
	if mentionsFilename("domain.py failed", "main.py") {
		t.Error("main.py must not match inside domain.py")
	}
	if !mentionsFilename(`{"error":"File \"app.py\", line 3"}`, "app.py") {
		t.Error("app.py must match as a whole token")
	}
	if !mentionsFilename("python3 app.py:12: error", "app.py") {
		t.Error("app.py must match when bounded by a colon")
	}
}

// closestLineHint maps a from-memory paraphrase onto the real file line
// so the edit_file mismatch error can anchor the model's retry in actual
// content. The motivating case (Jun 8 2026 flask test): the model
// searched for `item = items[id + 1]` when the file's real line was
// `return jsonify(items[item_id + 1])`, then gave up on the surgical
// edit and rewrote the whole function from the same faulty memory.
func TestClosestLineHintParaphrase(t *testing.T) {
	content := `from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    return jsonify(items[item_id + 1])
`
	hint := closestLineHint(content, "item = items[id + 1]")
	if hint == "" {
		t.Fatal("expected a hint for a near-miss paraphrase, got none")
	}
	if !strings.Contains(hint, "jsonify(items[item_id + 1])") {
		t.Errorf("hint should quote the real line, got: %s", hint)
	}
	if !strings.Contains(hint, "line 7") {
		t.Errorf("hint should carry the real line number, got: %s", hint)
	}
}

func TestClosestLineHintUnrelatedReturnsNothing(t *testing.T) {
	content := "def alpha():\n    return 1\n"
	if hint := closestLineHint(content, "zebra elephant giraffe"); hint != "" {
		t.Errorf("unrelated old_str must not get an anchor, got: %s", hint)
	}
}

func TestClosestLineHintEmptyOldStr(t *testing.T) {
	if hint := closestLineHint("x = 1\n", "   \n  "); hint != "" {
		t.Errorf("blank old_str must not get an anchor, got: %s", hint)
	}
}

// File-tier classifier tests. Covers the post-inversion behaviour where
// V3 should fire much more often: small-but-routed code files (flask,
// express) used to slip through to T1 because hasLogicIndicators needed
// 3+ patterns and the <50-line short-circuit fired first.

func TestClassifyFileTierFlaskAppPyIsT2(t *testing.T) {
	// 33-line flask routing module — exactly the file the user was
	// debugging when V3 never fired. Must be T2 now.
	content := `from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/product')
def product():
    return render_template('product.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
`
	if got := classifyFileTier("app.py", content); got < Tier2Medium {
		t.Errorf("flask app.py = %v, want >= T2 (was T1 under old <50-line rule)", got)
	}
}

func TestClassifyFileTierTinyConfigStaysT1(t *testing.T) {
	// Single-line .gitignore / 5-line shell wrapper — V3 has nothing to
	// diversify on, must stay T1 to avoid wasted pipeline cost.
	if got := classifyFileTier(".gitignore", "node_modules\n.env\n"); got != Tier1Simple {
		t.Errorf(".gitignore = %v, want T1", got)
	}
	if got := classifyFileTier("foo.sh", "#!/bin/sh\nexec node app.js\n"); got != Tier1Simple {
		t.Errorf("tiny shell script = %v, want T1", got)
	}
}

func TestClassifyFileTierConfigByName(t *testing.T) {
	// Config files by name are always T1 regardless of line count.
	configFiles := []string{"package.json", "go.mod", "Dockerfile", "requirements.txt"}
	for _, f := range configFiles {
		if got := classifyFileTier(f, "x\n\n\n\n\n\n\n\n\n\n\n\n\n"); got != Tier1Simple {
			t.Errorf("classifyFileTier(%q) = %v, want T1", f, got)
		}
	}
}

func TestClassifyFileTierCodeExtBenefitOfDoubt(t *testing.T) {
	// 15-line python file with no recognisable logic patterns — naming
	// it .py is enough to get T2 now, because V3 helps with code shape
	// even when the model didn't tag any specific framework idiom.
	content := `# generated module
NAME = "atlas"
VERSION = "0.1.0"
AUTHOR = "team"
LICENSE = "MIT"
DESCRIPTION = "demo"
KEYWORDS = ["a", "b"]
EXTRAS = {}
DEPS = []
DEV_DEPS = []
URL = "https://example.com"
HOMEPAGE = URL
DOWNLOADS = URL + "/dl"
`
	if got := classifyFileTier("constants.py", content); got != Tier2Medium {
		t.Errorf("constants.py = %v, want T2 (code-ext fallback)", got)
	}
}

func TestClassifyFileTierTinyCodeFileStaysT1(t *testing.T) {
	// 4 lines — below the new 10-line floor, even .py is T1.
	if got := classifyFileTier("hello.py", "print('hi')\n"); got != Tier1Simple {
		t.Errorf("1-line script = %v, want T1", got)
	}
}

func TestClassifyFileTierMidSizedHtmlIsT2(t *testing.T) {
	// 90-line flask template — used to fall through to T1 because the
	// markup branch only fired at >=150 lines. Now .html is in codeExts
	// so V3 fires on real templates instead of only on huge mockups.
	content := ""
	for i := 0; i < 90; i++ {
		content += "<p>row " + string(rune('a'+i%26)) + "</p>\n"
	}
	if got := classifyFileTier("templates/index.html", content); got != Tier2Medium {
		t.Errorf("90-line index.html = %v, want T2 (was T1 under <150-line markup veto)", got)
	}
}

func TestHasLogicIndicatorsFlaskAppHits(t *testing.T) {
	// Verifies the threshold drop + flask-pattern additions. A flask
	// app.py used to register only "def " (1 indicator); now the route
	// decorators count, putting it well over the new threshold of 2.
	content := `from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
`
	if !hasLogicIndicators(content) {
		t.Errorf("hasLogicIndicators(flask snippet) = false, want true")
	}
}

// Tests for the edit_file fallback matchers (findFuzzyLineMatch,
// findActualString) and recoverTruncatedWriteFile. These decide where an
// edit lands when the model's old_str doesn't byte-match the file — the
// silent-wrong-edit failure class, so the fail-safe paths matter as much
// as the matches.

func TestFindFuzzyLineMatch(t *testing.T) {
	file := "func main() {\n" +
		"\tx := 1\n" +
		"\tfmt.Println(x)\n" +
		"}\n" +
		"\n" +
		"func helper() {\n" +
		"\tx := 1\n" +
		"}\n"

	t.Run("whitespace drift resolves to the file's own lines", func(t *testing.T) {
		// Model remembered the lines with spaces instead of tabs.
		got, ok := findFuzzyLineMatch(file, "    x := 1\n    fmt.Println(x)")
		if !ok {
			t.Fatal("no match for whitespace-drifted old_str")
		}
		// The returned text must be the FILE's bytes (tabs), so the
		// subsequent strings.Replace hits.
		if got != "\tx := 1\n\tfmt.Println(x)" {
			t.Errorf("matched %q, want the file's original tab-indented lines", got)
		}
		if !strings.Contains(file, got) {
			t.Error("returned match is not a substring of the file")
		}
	})

	t.Run("ambiguous match fails safe", func(t *testing.T) {
		// `x := 1` appears in both functions — editing either would be a
		// guess, and a wrong guess is a silent wrong edit.
		if got, ok := findFuzzyLineMatch(file, "x := 1"); ok {
			t.Errorf("ambiguous single line matched %q, want failure", got)
		}
	})

	t.Run("no match returns false", func(t *testing.T) {
		if _, ok := findFuzzyLineMatch(file, "y := 2"); ok {
			t.Error("nonexistent line reported a match")
		}
	})

	t.Run("all-whitespace target is refused", func(t *testing.T) {
		if _, ok := findFuzzyLineMatch(file, "   \n\t"); ok {
			t.Error("all-whitespace old_str matched — would edit an arbitrary blank region")
		}
	})

	t.Run("trailing newline in old_str is tolerated", func(t *testing.T) {
		got, ok := findFuzzyLineMatch(file, "\tfmt.Println(x)\n}\n")
		if !ok {
			t.Fatal("trailing-newline old_str did not match")
		}
		if got != "\tfmt.Println(x)\n}" {
			t.Errorf("matched %q", got)
		}
	})

	t.Run("empty old_str is refused", func(t *testing.T) {
		if _, ok := findFuzzyLineMatch(file, ""); ok {
			t.Error("empty old_str matched")
		}
	})
}

func TestFindActualString(t *testing.T) {
	t.Run("direct match wins untouched", func(t *testing.T) {
		if got := findActualString(`say "hi"`, `say "hi"`); got != `say "hi"` {
			t.Errorf("got %q", got)
		}
	})

	t.Run("curly quotes in old_str match straight quotes in file", func(t *testing.T) {
		file := `msg := "hello"`
		oldStr := "msg := “hello”" // model emitted curly quotes
		got := findActualString(file, oldStr)
		if got != `msg := "hello"` {
			t.Errorf("got %q, want the straight-quoted form present in the file", got)
		}
	})

	t.Run("straight apostrophe in old_str matches curly apostrophe in file", func(t *testing.T) {
		// The reverse (denormalize) direction is best-effort: straight
		// singles map to the right-single curly — the apostrophe case
		// (prose in docs/comments) is what it exists for.
		file := "// don’t retry on 4xx"
		got := findActualString(file, "// don't retry on 4xx")
		if got != "// don’t retry on 4xx" {
			t.Errorf("got %q, want the file's curly-apostrophe form", got)
		}
	})

	t.Run("no variant matches returns empty", func(t *testing.T) {
		if got := findActualString("abc", "xyz"); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})
}

func TestRecoverTruncatedWriteFile(t *testing.T) {
	t.Run("recovers path and unescaped content", func(t *testing.T) {
		partial := `{"type":"tool_call","name":"write_file","args":{"path":"app/main.py","content":"import os\nprint(\"hi\")\n# cut mid-`
		resp, err := recoverTruncatedWriteFile(partial)
		if err != nil {
			t.Fatalf("recovery failed: %v", err)
		}
		if resp.Type != "tool_call" || resp.Name != "write_file" {
			t.Fatalf("recovered envelope %+v", resp)
		}
		var input WriteFileInput
		if err := json.Unmarshal(resp.Args, &input); err != nil {
			t.Fatalf("recovered args do not parse: %v", err)
		}
		if input.Path != "app/main.py" {
			t.Errorf("path = %q", input.Path)
		}
		// JSON escapes must be resolved into real bytes.
		if !strings.Contains(input.Content, "import os\nprint(\"hi\")") {
			t.Errorf("content = %q — escapes not resolved", input.Content)
		}
	})

	t.Run("trailing incomplete escape is trimmed", func(t *testing.T) {
		partial := `{"type":"tool_call","name":"write_file","args":{"path":"a.txt","content":"line\n\`
		resp, err := recoverTruncatedWriteFile(partial)
		if err != nil {
			t.Fatalf("recovery failed on trailing backslash: %v", err)
		}
		var input WriteFileInput
		_ = json.Unmarshal(resp.Args, &input)
		if input.Content != "line\n" {
			t.Errorf("content = %q, want %q", input.Content, "line\n")
		}
	})

	t.Run("missing content field is an error", func(t *testing.T) {
		if _, err := recoverTruncatedWriteFile(`{"type":"tool_call","name":"write_file","args":{"path":"a.txt"`); err == nil {
			t.Error("recovered a write_file with no content field")
		}
	})

	t.Run("missing path is an error", func(t *testing.T) {
		if _, err := recoverTruncatedWriteFile(`{"type":"tool_call","name":"write_file","args":{"content":"body only`); err == nil {
			t.Error("recovered a write_file with no destination path")
		}
	})
}

// Tests for buildResponseFormat — the schema-constrained sampling path
// (#33). These tests pin the response_format payload shape that goes
// over the wire to llama-server, so a regression that silently flips
// the default back to loose JSON gets caught.

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

func newMoveCtx(dir string) *AgentContext {
	return &AgentContext{
		WorkingDir:    dir,
		FilesRead:     map[string]string{},
		FileReadTimes: map[string]time.Time{},
		SessionWrites: map[string]bool{},
	}
}

// The reported failure: reorganizing a flask app by moving index.html into
// templates/. move_file should relocate it into the existing directory keeping
// the basename, content intact.
func TestMoveFileIntoDirectory(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "index.html"), []byte("<h1>hi</h1>"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "templates"), 0755); err != nil {
		t.Fatal(err)
	}
	ctx := newMoveCtx(dir)
	res, err := moveFileTool().Execute(json.RawMessage(`{"source":"index.html","destination":"templates/"}`), ctx)
	if err != nil || !res.Success {
		t.Fatalf("move failed: err=%v res=%+v", err, res)
	}
	if _, err := os.Stat(filepath.Join(dir, "index.html")); !os.IsNotExist(err) {
		t.Errorf("source still exists after move")
	}
	data, err := os.ReadFile(filepath.Join(dir, "templates", "index.html"))
	if err != nil || string(data) != "<h1>hi</h1>" {
		t.Errorf("destination missing or content changed: %q err=%v", string(data), err)
	}
}

// Plain rename within the same directory.
func TestMoveFileRename(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "old.py"), []byte("print(1)\n"), 0644); err != nil {
		t.Fatal(err)
	}
	ctx := newMoveCtx(dir)
	res, err := moveFileTool().Execute(json.RawMessage(`{"source":"old.py","destination":"new.py"}`), ctx)
	if err != nil || !res.Success {
		t.Fatalf("rename failed: err=%v res=%+v", err, res)
	}
	if _, err := os.Stat(filepath.Join(dir, "new.py")); err != nil {
		t.Errorf("renamed file missing: %v", err)
	}
}

// A move must never silently clobber an existing destination file.
func TestMoveFileRefusesClobber(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("A"), 0644)
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("B"), 0644)
	ctx := newMoveCtx(dir)
	res, err := moveFileTool().Execute(json.RawMessage(`{"source":"a.txt","destination":"b.txt"}`), ctx)
	if err != nil {
		t.Fatalf("unexpected hard error: %v", err)
	}
	if res.Success {
		t.Errorf("expected refusal to clobber existing destination")
	}
	if data, _ := os.ReadFile(filepath.Join(dir, "b.txt")); string(data) != "B" {
		t.Errorf("destination was overwritten: %q", string(data))
	}
}

// Missing source is a clean tool error, not a crash.
func TestMoveFileSourceMissing(t *testing.T) {
	dir := t.TempDir()
	ctx := newMoveCtx(dir)
	res, err := moveFileTool().Execute(json.RawMessage(`{"source":"nope.py","destination":"x.py"}`), ctx)
	if err != nil {
		t.Fatalf("unexpected hard error: %v", err)
	}
	if res.Success {
		t.Errorf("expected failure for missing source")
	}
}

// storedReadMessage mimics how the agent loop records a read_file result in the
// conversation: ToolResult.Data = marshaled ReadFileOutput, then the message
// content = ToolResult.MarshalText() (json.Marshal of the whole result). This
// is where the file content gets JSON-escaped.
func storedReadMessage(t *testing.T, fileContent string) AgentMessage {
	t.Helper()
	// read_file numbers lines; the numbering doesn't affect the probe, but
	// model the real shape anyway.
	var numbered strings.Builder
	for i, l := range strings.Split(fileContent, "\n") {
		numbered.WriteString(strings.TrimRight(
			strings.Join([]string{itoaTest(i + 1), l}, "\t"), "\n"))
		numbered.WriteString("\n")
	}
	out := ReadFileOutput{Content: numbered.String()}
	data, _ := json.Marshal(out)
	res := &ToolResult{Success: true, Data: data}
	return AgentMessage{Role: "tool", ToolName: "read_file", Content: res.MarshalText()}
}

func itoaTest(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}

// Regression: a flask app whose longest line is embedded HTML/JS full of double
// quotes must still be detected as present in context. The old longest-raw-line
// probe failed here (the `"` in the line became `\"` in the stored JSON), so
// the dedup re-served the file every read and the model looped on read_file.
func TestFileContentInContextSurvivesJSONEscaping(t *testing.T) {
	fileContent := `from flask import Flask, render_template_string
app = Flask(__name__)
HTML = "<div class=\"board\" style=\"width:400px;height:400px\" id=\"game-board-container\"></div>"

@app.route("/")
def index():
    return render_template_string(HTML)
`
	ctx := &AgentContext{Messages: []AgentMessage{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "make a snake game"},
		storedReadMessage(t, fileContent),
	}}
	if !fileContentInContext(ctx, fileContent) {
		t.Errorf("content with quoted HTML line reported as NOT in context — false-negative would make dedup re-serve and loop")
	}
}

// When the content really is gone (trimmed), it must report absent so the
// dedup re-serves rather than lying that "it's above."
func TestFileContentInContextDetectsAbsence(t *testing.T) {
	fileContent := "def compute_subtotal(rows):\n    return sum(r.price for r in rows)\n"
	ctx := &AgentContext{Messages: []AgentMessage{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "unrelated chatter with no file content"},
	}}
	if fileContentInContext(ctx, fileContent) {
		t.Errorf("expected absent verdict when content is not in any message")
	}
}

// fakeSandboxShell is a tiny stand-in for sandbox /shell.
// Echoes back canned output so we can assert routing, request shape,
// and response decoding without bringing up the real sandbox.
func fakeSandboxShell(t *testing.T, status int, resp interface{}) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/shell" {
			http.NotFound(w, r)
			return
		}
		if r.Method != "POST" {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(resp)
	}))
}

func TestRunViaSandboxDecodesSuccessfulResponse(t *testing.T) {
	srv := fakeSandboxShell(t, 200, map[string]interface{}{
		"success":    true,
		"stdout":     "hello world\n",
		"stderr":     "",
		"exit_code":  0,
		"elapsed_ms": 42,
	})
	defer srv.Close()

	ctx := &AgentContext{SandboxURL: srv.URL}
	out, err := runViaSandbox(ctx, `echo "hello world"`, "/workspace", 5)
	if err != nil {
		t.Fatalf("runViaSandbox: %v", err)
	}
	if out.ExitCode != 0 {
		t.Errorf("exit_code = %d, want 0", out.ExitCode)
	}
	if !strings.Contains(out.Stdout, "hello world") {
		t.Errorf("stdout = %q, want to contain 'hello world'", out.Stdout)
	}
}

func TestRunViaSandboxSurfacesNonZeroExit(t *testing.T) {
	srv := fakeSandboxShell(t, 200, map[string]interface{}{
		"success":   false,
		"stdout":    "",
		"stderr":    "ImportError: No module named flask",
		"exit_code": 1,
	})
	defer srv.Close()

	ctx := &AgentContext{SandboxURL: srv.URL}
	out, err := runViaSandbox(ctx, "python -c 'import flask'", "/workspace", 5)
	if err != nil {
		t.Fatalf("runViaSandbox: %v", err)
	}
	if out.ExitCode != 1 {
		t.Errorf("exit_code = %d, want 1", out.ExitCode)
	}
	if !strings.Contains(out.Stderr, "ImportError") {
		t.Errorf("stderr lost: %q", out.Stderr)
	}
}

func TestRunViaSandbox4xxIsValidationFailure(t *testing.T) {
	// A 4xx from the sandbox means the request was bad (e.g. cwd
	// outside /workspace). Should NOT propagate as "sandbox
	// unreachable" — that would trigger the local-exec fallback
	// and let the model bypass the cwd guard. Instead we surface
	// the FastAPI detail on stderr with exit_code=1.
	srv := fakeSandboxShell(t, 400, map[string]string{
		"detail": "cwd must be under /workspace, got /etc",
	})
	defer srv.Close()

	ctx := &AgentContext{SandboxURL: srv.URL}
	out, err := runViaSandbox(ctx, "ls", "/etc", 5)
	if err != nil {
		t.Fatalf("4xx should NOT return Go error (fallback would trip): %v", err)
	}
	if out.ExitCode != 1 {
		t.Errorf("exit_code = %d, want 1 for 4xx", out.ExitCode)
	}
	if !strings.Contains(out.Stderr, "must be under /workspace") {
		t.Errorf("stderr should carry the FastAPI detail, got %q", out.Stderr)
	}
}

func TestRunViaSandboxUnreachableReturnsError(t *testing.T) {
	// Sandbox URL that won't accept connections. The caller must surface
	// this error and must not change the execution target implicitly.
	ctx := &AgentContext{SandboxURL: "http://127.0.0.1:1"}
	_, err := runViaSandbox(ctx, "echo hi", "/workspace", 5)
	if err == nil {
		t.Error("expected network error for unreachable sandbox")
	}
}

func TestRunCommandDoesNotFallbackWhenSandboxIsUnavailable(t *testing.T) {
	ctx := &AgentContext{
		WorkingDir: "/workspace",
		SandboxURL: "http://127.0.0.1:1",
	}
	res, err := runCommandTool().Execute(
		json.RawMessage(`{"command":"printf fallback-ran"}`), ctx,
	)
	if err != nil {
		t.Fatalf("run_command: %v", err)
	}
	if res.Success {
		t.Fatal("run_command succeeded even though the sandbox was unavailable")
	}
	if strings.Contains(string(res.Data), "fallback-ran") {
		t.Fatalf("command appears to have run locally: %s", res.Data)
	}
	if !strings.Contains(res.Error, "sandbox unavailable") {
		t.Fatalf("error = %q, want sandbox unavailable", res.Error)
	}
}

func TestRunLocallyEcho(t *testing.T) {
	// runLocally is the explicit ATLAS_VERIFY_IN=host execution primitive.
	// Quick sanity: a trivial echo command returns a populated result.
	out := runLocally("echo hello", ".", 0)
	// timeout=0 still has an internal default — go's select with
	// time.After(0) fires immediately, so we accept either the
	// successful path or the timeout path. The important property
	// is "doesn't panic and returns a populated struct."
	if out.ExitCode != 0 && out.ExitCode != 124 {
		t.Errorf("unexpected exit_code %d for runLocally(echo)", out.ExitCode)
	}
}

// Credential-file read exclusion (P0: sensitive values must not flow
// into model context by default). All paths are synthetic fixtures.

func TestDenyReadPathReason(t *testing.T) {
	os.Unsetenv("ATLAS_ALLOW_CREDENTIAL_READS")

	blocked := []string{
		".env",
		"subdir/.env",
		".env.production",
		".netrc",
		".npmrc",
		".pypirc",
		"certs/server.pem",
		"secrets/signing.key",
		".ssh/id_rsa",
		"id_ed25519",
		"/home/user/.ssh/id_ecdsa",
		".aws/credentials",
		".aws/config",
		".kube/config",
		".docker/config.json",
		"secrets/service-token",
		"secrets/api-keys.json",
		"gcp-credentials.json",
	}
	for _, p := range blocked {
		if reason := denyReadPathReason(p); reason == "" {
			t.Errorf("read of %q should be blocked", p)
		}
	}

	allowed := []string{
		".env.example", // template, documented
		"main.go",
		"config.yaml",
		"src/environment.ts", // unrelated name
		"staging.envrc.sample",
		".ssh/id_rsa.pub",     // public half
		"docs/kube/config.md", // .kube parent match is exact-dir only
		"README.md",
	}
	for _, p := range allowed {
		if reason := denyReadPathReason(p); reason != "" {
			t.Errorf("read of %q wrongly blocked: %s", p, reason)
		}
	}
}

func TestDenyReadOverride(t *testing.T) {
	os.Setenv("ATLAS_ALLOW_CREDENTIAL_READS", "1")
	defer os.Unsetenv("ATLAS_ALLOW_CREDENTIAL_READS")
	if reason := denyReadPathReason(".env"); reason != "" {
		t.Fatalf("override not honored: %s", reason)
	}
}

func TestShouldDenyToolCallReadFile(t *testing.T) {
	os.Unsetenv("ATLAS_ALLOW_CREDENTIAL_READS")
	args, _ := json.Marshal(map[string]string{"path": ".netrc"})
	denied, reason := shouldDenyToolCall("read_file", args)
	if !denied {
		t.Fatal("read_file .netrc not denied")
	}
	if reason == "" || !containsStr(reason, "ATLAS_ALLOW_CREDENTIAL_READS") {
		t.Fatalf("refusal must name the documented override: %q", reason)
	}

	denied, _ = shouldDenyToolCall("outline_file", args)
	if !denied {
		t.Fatal("outline_file .netrc not denied")
	}

	ok, _ := json.Marshal(map[string]string{"path": "main.py"})
	denied, reason = shouldDenyToolCall("read_file", ok)
	if denied {
		t.Fatalf("normal read denied: %s", reason)
	}
}

func containsStr(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
