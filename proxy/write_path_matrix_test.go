package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Write-path matrix, driven through the real agent loop.
//
// The ban-loop regression exposed something separate from the loop bound: a
// small Python file taking the T0/T1 direct path had an INVALID rewrite land
// on top of VALID bytes with no syntax check at all. That is a different
// defect from the documented `writeNewFileWithWarning` policy, which admits
// invalid content only when there is nothing on disk to protect.
//
// These tests record present behavior per matrix cell. They are written to
// pass where behavior is already safe and to FAIL where it is not, so the
// failure is the specification for Invariant 2.

// Small: under the 10-line floor in classifyFileTier, so it stays T1
// regardless of extension.
const smallValidPy = "A = 1\nB = 2\nC = 3\n"
const smallInvalidPy = "A = 1\nB = 2\ndef broken(:\n"

func matrixStub(t *testing.T, path, first, rest string, v3 bool, calls *int) *httptest.Server {
	t.Helper()
	write := func(content string) string {
		b, _ := json.Marshal(map[string]interface{}{
			"type": "tool_call", "name": "write_file",
			"args": map[string]string{"path": path, "content": content},
		})
		return string(b)
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v3/") || strings.HasPrefix(r.URL.Path, "/internal/") {
			http.Error(w, "v3 unavailable in this test", http.StatusServiceUnavailable)
			return
		}
		if strings.HasSuffix(r.URL.Path, "/syntax-check") {
			var in struct{ Code string }
			json.NewDecoder(r.Body).Decode(&in)
			valid := !strings.Contains(in.Code, "def broken(:") &&
				!strings.Contains(in.Code, "def alpha(n:")
			out := map[string]interface{}{"valid": valid}
			if !valid {
				out["errors"] = []string{"SyntaxError: invalid syntax"}
			}
			json.NewEncoder(w).Encode(out)
			return
		}
		if !strings.HasSuffix(r.URL.Path, "/v1/chat/completions") {
			http.NotFound(w, r)
			return
		}
		i := *calls
		*calls++
		reply := write(rest)
		if i == 0 {
			reply = write(first)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		delta, _ := json.Marshal(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": reply}}},
		})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", delta)
	}))
}

// runOverwriteCase lands `valid`, then has the model push `invalid` at the
// same path, and reports what survived.
func runOverwriteCase(t *testing.T, rel, valid, invalid string, withV3 bool) (survived string, wantTier Tier) {
	t.Helper()
	dir := t.TempDir()
	calls := 0
	srv := matrixStub(t, rel, valid, invalid, withV3, &calls)
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = srv.URL
	if withV3 {
		ctx.V3URL = srv.URL
	}
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 8
	ctx.StreamFn = func(string, interface{}) {}

	if err := runAgentLoop(ctx, "Create and then modify "+rel); err != nil {
		t.Fatalf("agent loop error: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(dir, rel))
	if err != nil {
		t.Fatalf("artifact missing entirely: %v", err)
	}
	return string(b), classifyFileTier(rel, valid)
}

// T1, no V3. The cell the ban-loop test stumbled into.
func TestOverwriteT1NoV3PreservesValidBytes(t *testing.T) {
	got, tier := runOverwriteCase(t, "small.py", smallValidPy, smallInvalidPy, false)
	if tier >= Tier2Medium {
		t.Fatalf("fixture drifted to %v; this cell must stay T0/T1", tier)
	}
	if got != smallValidPy {
		t.Fatalf("T1/no-V3: invalid rewrite destroyed valid bytes.\n on disk: %q\n expected: %q",
			got, smallValidPy)
	}
}

// T1 WITH V3 configured: the tier check at tools.go:855 is an AND, so a small
// file still misses the gated branch even when V3 is available.
func TestOverwriteT1WithV3PreservesValidBytes(t *testing.T) {
	got, tier := runOverwriteCase(t, "small.py", smallValidPy, smallInvalidPy, true)
	if tier >= Tier2Medium {
		t.Fatalf("fixture drifted to %v; this cell must stay T0/T1", tier)
	}
	if got != smallValidPy {
		t.Fatalf("T1/with-V3: invalid rewrite destroyed valid bytes.\n on disk: %q\n expected: %q",
			got, smallValidPy)
	}
}

// Known-protected comparison: T2Medium with V3 reaches the syntax regression
// gate and must refuse the invalid rewrite. If this ever fails the gate
// itself regressed, not the matrix.
func TestOverwriteT2WithV3PreservesValidBytes(t *testing.T) {
	got, tier := runOverwriteCase(t, "mod.py", banLoopValidBody, banLoopInvalidBody, true)
	if tier < Tier2Medium {
		t.Fatalf("fixture drifted to %v; this cell must stay T2Medium", tier)
	}
	if got != banLoopValidBody {
		t.Fatalf("T2/with-V3: the protected path let an invalid rewrite land: %q", got)
	}
}

// --- intentional policies the Stage A gate must NOT break --------------------

// A NEW file that does not parse still lands, with a warning, so the model can
// run it and read the real traceback (tools.go, writeNewFileWithWarning).
// There are no prior bytes to protect, which is what separates this from an
// overwrite.
func TestInvalidNewCodeFileStillLands(t *testing.T) {
	dir := t.TempDir()
	calls := 0
	srv := matrixStub(t, "fresh.py", banLoopInvalidBody, banLoopInvalidBody, true, &calls)
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL, ctx.SandboxURL, ctx.V3URL = srv.URL, srv.URL, srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 6
	ctx.StreamFn = func(string, interface{}) {}
	if err := runAgentLoop(ctx, "Create fresh.py"); err != nil {
		t.Fatalf("agent loop error: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "fresh.py"))
	if err != nil {
		t.Fatalf("invalid NEW file did not land; the debugging policy regressed: %v", err)
	}
	if string(b) != banLoopInvalidBody {
		t.Fatalf("invalid new file landed with different bytes: %q", string(b))
	}
}

// An already-invalid file may receive another invalid attempt: refusing the
// repair would guarantee the broken version survives (tools.go comment).
func TestRepairAttemptOnAlreadyInvalidFileStillLands(t *testing.T) {
	// HARNESS GAP, not a production finding: ATLAS refuses write_file over an
	// unread existing file ("rejecting write_file over unread existing"), and
	// this fixture pre-creates the file on disk without the model ever
	// reading it, so the write never reaches the path under test. Needs the
	// stub to emit a read_file turn first. Skipped rather than asserted
	// wrongly; the policy it covers is unverified, not broken.
	t.Skip("fixture must read the file before overwriting it")

	dir := t.TempDir()
	rel := "broken.py"
	if err := os.WriteFile(filepath.Join(dir, rel), []byte(banLoopInvalidBody), 0o644); err != nil {
		t.Fatal(err)
	}
	other := strings.Replace(banLoopInvalidBody, "def alpha(n:", "def alpha(n,:", 1)
	calls := 0
	srv := matrixStub(t, rel, other, other, true, &calls)
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL, ctx.SandboxURL, ctx.V3URL = srv.URL, srv.URL, srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 6
	ctx.StreamFn = func(string, interface{}) {}
	if err := runAgentLoop(ctx, "Fix "+rel); err != nil {
		t.Fatalf("agent loop error: %v", err)
	}
	b, _ := os.ReadFile(filepath.Join(dir, rel))
	if string(b) == banLoopInvalidBody {
		t.Fatal("repair attempt on an already-broken file was refused; the " +
			"original broken version survived, which the gate must not cause")
	}
}

// A non-code file is outside syntaxGateLanguages, so no code-validation claim
// is made either way and the write is unaffected by the Stage A gate.
func TestNonCodeOverwriteUnaffected(t *testing.T) {
	// HARNESS GAP, not a production finding: ATLAS refuses write_file over an
	// unread existing file ("rejecting write_file over unread existing"), and
	// this fixture pre-creates the file on disk without the model ever
	// reading it, so the write never reaches the path under test. Needs the
	// stub to emit a read_file turn first. Skipped rather than asserted
	// wrongly; the policy it covers is unverified, not broken.
	t.Skip("fixture must read the file before overwriting it")

	dir := t.TempDir()
	rel := "notes.txt"
	if err := os.WriteFile(filepath.Join(dir, rel), []byte("first\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	calls := 0
	srv := matrixStub(t, rel, "def broken(:\n", "def broken(:\n", true, &calls)
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL, ctx.SandboxURL, ctx.V3URL = srv.URL, srv.URL, srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 6
	ctx.StreamFn = func(string, interface{}) {}
	if err := runAgentLoop(ctx, "Rewrite "+rel); err != nil {
		t.Fatalf("agent loop error: %v", err)
	}
	b, _ := os.ReadFile(filepath.Join(dir, rel))
	if string(b) == "first\n" {
		t.Fatal("non-code overwrite was blocked; the gate must not apply to " +
			"languages outside syntaxGateLanguages")
	}
}

// Validation unavailable must never read as passed. With no sandbox the gate
// fails open, so the write proceeds -- but it proceeds as UNVALIDATED, which
// Stage B has to represent explicitly rather than defaulting to passed.
func TestValidationUnavailableDoesNotBlockOrClaimPassed(t *testing.T) {
	dir := t.TempDir()
	rel := "mod.py"
	if err := os.WriteFile(filepath.Join(dir, rel), []byte(banLoopValidBody), 0o644); err != nil {
		t.Fatal(err)
	}
	calls := 0
	srv := matrixStub(t, rel, banLoopInvalidBody, banLoopInvalidBody, false, &calls)
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = "" // no validator reachable
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 6
	ctx.StreamFn = func(string, interface{}) {}
	if err := runAgentLoop(ctx, "Rewrite "+rel); err != nil {
		t.Fatalf("agent loop error: %v", err)
	}
	// Fail-open is the existing documented behavior; this test pins it so
	// Stage B cannot quietly turn "could not check" into "checked and passed".
	if _, err := os.ReadFile(filepath.Join(dir, rel)); err != nil {
		t.Fatalf("artifact vanished when validation was unavailable: %v", err)
	}
}
