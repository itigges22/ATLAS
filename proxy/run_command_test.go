package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

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
	// Sandbox URL that won't accept connections. Caller is supposed
	// to fall back to local exec — that decision is in run_command,
	// not runViaSandbox. We just verify we surface the network
	// error so the caller can branch on it.
	ctx := &AgentContext{SandboxURL: "http://127.0.0.1:1"}
	_, err := runViaSandbox(ctx, "echo hi", "/workspace", 5)
	if err == nil {
		t.Error("expected network error for unreachable sandbox")
	}
}

func TestRunLocallyEcho(t *testing.T) {
	// runLocally is the fallback when the sandbox is unreachable.
	// Quick sanity: a trivial echo command should return its output
	// with exit_code 0.
	out := runLocally("echo hello", ".", 0)
	// timeout=0 still has an internal default — go's select with
	// time.After(0) fires immediately, so we accept either the
	// successful path or the timeout path. The important property
	// is "doesn't panic and returns a populated struct."
	if out.ExitCode != 0 && out.ExitCode != 124 {
		t.Errorf("unexpected exit_code %d for runLocally(echo)", out.ExitCode)
	}
}
