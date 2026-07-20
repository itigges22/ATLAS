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
