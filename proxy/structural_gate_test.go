package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fakeV3Structural returns the given unresolved names for source containing
// `trigger`, else clean. Lets a test model "original clean, edited broken".
func fakeV3Structural(t *testing.T, trigger string, unresolved []string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/structural_check" {
			http.Error(w, "not found", 404)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		var body struct {
			Source string `json:"source"`
		}
		_ = json.Unmarshal(raw, &body)
		out := map[string]interface{}{"ok": true, "unresolved": []string{}}
		if strings.Contains(body.Source, trigger) {
			out["unresolved"] = unresolved
		}
		b, _ := json.Marshal(out)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(b)
	}))
}

func structCtx(url string) *AgentContext {
	return &AgentContext{V3URL: url, Ctx: context.Background(), WorkingDir: "/workspace"}
}

// An edit that introduces render_template (not in the original) is blocked.
func TestEditIntroducesUnresolvedBlocks(t *testing.T) {
	srv := fakeV3Structural(t, "render_template(", []string{"render_template"})
	defer srv.Close()
	ctx := structCtx(srv.URL)
	orig := "from flask import render_template_string\n@app.route('/')\ndef i(): return 'x'\n"
	edited := "from flask import render_template_string\n@app.route('/')\ndef i(): return render_template('i.html')\n"
	introduced := editIntroducesUnresolved(ctx, "app.py", orig, edited)
	if len(introduced) != 1 || introduced[0] != "render_template" {
		t.Fatalf("expected [render_template] introduced, got %v", introduced)
	}
	msg := structuralRejection("app.py", introduced)
	if !strings.Contains(msg, "`render_template`") || !strings.Contains(msg, "NameError") {
		t.Errorf("rejection should name the call + NameError: %q", msg)
	}
}

// A pre-existing unresolved name (present in BOTH original and edited) is a
// repair-in-progress and must NOT be blocked.
func TestPreexistingUnresolvedAllowed(t *testing.T) {
	srv := fakeV3Structural(t, "render_template(", []string{"render_template"})
	defer srv.Close()
	ctx := structCtx(srv.URL)
	// Both call render_template -> it's not newly introduced.
	orig := "def a(): return render_template('a')\n"
	edited := "def a(): return render_template('a')\ndef b(): return render_template('b')\n"
	if introduced := editIntroducesUnresolved(ctx, "app.py", orig, edited); len(introduced) != 0 {
		t.Errorf("pre-existing unresolved must be allowed, got %v", introduced)
	}
}

// Non-.py files and an unreachable V3 fail open (no block).
func TestStructuralGateFailsOpen(t *testing.T) {
	ctx := structCtx("http://127.0.0.1:0") // unreachable
	if introduced := editIntroducesUnresolved(ctx, "app.py", "x", "y render_template("); introduced != nil {
		t.Errorf("unreachable V3 must fail open, got %v", introduced)
	}
	ctx2 := structCtx("http://example.invalid")
	if _, ok := checkStructuralUnresolved(ctx2, "notes.txt", "render_template()"); ok {
		t.Error("non-.py must not be checked")
	}
}

// fakeV3StructuralResolving actually resolves `name` against the posted
// source: unresolved iff the source calls it AND lacks importLine. Faithful
// enough to express both directions of the #147 regression pair (blocked
// without the import, credited by an import anywhere in the composed file).
func fakeV3StructuralResolving(t *testing.T, name, importLine string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body struct {
			Source string `json:"source"`
		}
		_ = json.Unmarshal(raw, &body)
		out := map[string]interface{}{"ok": true, "unresolved": []string{}}
		if strings.Contains(body.Source, name+"(") && !strings.Contains(body.Source, importLine) {
			out["unresolved"] = []string{name}
		}
		b, _ := json.Marshal(out)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(b)
	}))
}

// #147 scope item 3(b) at gate level: an edit whose new call is satisfied by
// an import living ELSEWHERE in the composed file (outside the edited
// fragment) must pass — the gate checks the whole post-edit file.
func TestImportElsewhereInFilePasses(t *testing.T) {
	srv := fakeV3StructuralResolving(t, "helper_util", "from utils import helper_util")
	defer srv.Close()
	ctx := structCtx(srv.URL)
	orig := "from utils import helper_util\n\ndef old():\n    return 1\n"
	edited := "from utils import helper_util\n\ndef old():\n    return helper_util()\n"
	if introduced := editIntroducesUnresolved(ctx, "app.py", orig, edited); len(introduced) != 0 {
		t.Errorf("import elsewhere in the file must credit the call, got %v", introduced)
	}
}

// Deleting an import a remaining direct call needs is a newly-introduced
// unresolved name and must be blocked.
func TestDeleteImportBlocked(t *testing.T) {
	srv := fakeV3StructuralResolving(t, "helper_util", "from utils import helper_util")
	defer srv.Close()
	ctx := structCtx(srv.URL)
	orig := "from utils import helper_util\n\ndef index():\n    return helper_util()\n"
	edited := "def index():\n    return helper_util()\n"
	introduced := editIntroducesUnresolved(ctx, "app.py", orig, edited)
	if len(introduced) != 1 || introduced[0] != "helper_util" {
		t.Fatalf("deleting the import must block, got %v", introduced)
	}
}

// When the ORIGINAL-side check can't run — a transient service failure on
// the second back-to-back call; note malformed Python is NOT this trigger,
// tree-sitter parses it tolerantly and returns ok:true — the healthy->
// broken comparison has no baseline: the gate must fail open (after one
// retry), not count every unresolved name as newly introduced.
func TestOriginalCheckFailureFailsOpen(t *testing.T) {
	origCalls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body struct {
			Source string `json:"source"`
		}
		_ = json.Unmarshal(raw, &body)
		// The fake models the service failing on the ORIGINAL-side
		// requests (marker comment), succeeding on the edited side.
		if strings.Contains(body.Source, "# original") {
			origCalls++
			_, _ = w.Write([]byte(`{"ok":false,"error":"transient failure"}`))
			return
		}
		_, _ = w.Write([]byte(`{"ok":true,"unresolved":["helper_x"]}`))
	}))
	defer srv.Close()
	ctx := structCtx(srv.URL)
	orig := "# original\ndef a():\n    return helper_x()\n"
	edited := "def fixed():\n    return helper_x()\n"
	if introduced := editIntroducesUnresolved(ctx, "app.py", orig, edited); introduced != nil {
		t.Errorf("original-side check failure must fail open, got %v", introduced)
	}
	if origCalls != 2 {
		t.Errorf("expected one retry of the original-side check (2 calls), got %d", origCalls)
	}
}

// The write_file variant of the rejection must name the operation the
// model actually issued — an "edit" steer on a blocked NEW-file write
// sends the model to edit_file against a file that doesn't exist.
func TestWriteRejectionNamesWrite(t *testing.T) {
	msg := structuralWriteRejection("app.py", []string{"render_template"})
	if !strings.Contains(msg, "write_file for app.py") || !strings.Contains(msg, "`render_template`") ||
		!strings.Contains(msg, "NameError") || !strings.Contains(msg, "re-issue the write_file") {
		t.Errorf("write rejection must be write-flavored and name the call: %q", msg)
	}
	if strings.Contains(msg, "re-issue the edit") {
		t.Errorf("write rejection must not steer toward edit tools: %q", msg)
	}
}

// readOriginalForGate: missing file = first write (empty original, gate
// runs); any other read failure = unknowable original (gate must skip).
func TestReadOriginalForGate(t *testing.T) {
	dir := t.TempDir()
	if content, ok := readOriginalForGate(dir + "/nope.py"); !ok || content != "" {
		t.Errorf("missing file must be (\"\", true), got (%q, %v)", content, ok)
	}
	if _, ok := readOriginalForGate(dir); ok {
		t.Error("unreadable original (a directory) must report not-ok so callers skip the gate")
	}
}

// A nil ctx.Ctx (paths constructed without a request context) must not
// panic — and the gate keeps working via a background context.
func TestNilRequestContextStillGates(t *testing.T) {
	srv := fakeV3Structural(t, "render_template(", []string{"render_template"})
	defer srv.Close()
	ctx := &AgentContext{V3URL: srv.URL, WorkingDir: "/workspace"} // Ctx nil
	orig := "def i(): return 'x'\n"
	edited := "def i(): return render_template('i.html')\n"
	introduced := editIntroducesUnresolved(ctx, "app.py", orig, edited)
	if len(introduced) != 1 || introduced[0] != "render_template" {
		t.Fatalf("nil ctx.Ctx must still gate via background context, got %v", introduced)
	}
}

// #147 review #2: the edited file's own (pre-edit) content must be excluded
// from the project_context sent, so a deleted top-level def isn't credited
// from stale state.
func TestStructuralCheckExcludesEditedSelf(t *testing.T) {
	var gotCtx map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]interface{}
		_ = json.Unmarshal(raw, &body)
		if pc, ok := body["project_context"].(map[string]interface{}); ok {
			gotCtx = pc
		} else {
			gotCtx = map[string]interface{}{}
		}
		_, _ = w.Write([]byte(`{"ok":true,"unresolved":[]}`))
	}))
	defer srv.Close()
	ctx := &AgentContext{
		V3URL: srv.URL, Ctx: context.Background(), WorkingDir: "/workspace",
		FilesRead:     map[string]string{"/workspace/app.py": "def gone(): pass", "/workspace/util.py": "def helper(): pass"},
		FileReadTimes: map[string]time.Time{"/workspace/app.py": time.Now(), "/workspace/util.py": time.Now()},
	}
	_, _ = checkStructuralUnresolved(ctx, "/workspace/app.py", "x = gone()")
	if _, present := gotCtx["app.py"]; present {
		t.Error("edited file app.py must be excluded from project_context")
	}
	if _, present := gotCtx["util.py"]; !present {
		t.Error("other read files should still be included")
	}
}
