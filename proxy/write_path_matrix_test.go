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
