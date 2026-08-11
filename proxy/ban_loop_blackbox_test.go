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

// Black-box regression for the ban-branch runaway recorded in the locked
// benchmark (redteam/runs/locked, dev head 78d345c).
//
// It drives the real agent loop with a model that behaves exactly like the
// archived session: one accepted write, then an invalid rewrite, then the
// same invalid rewrite forever. That sequence rejects, bans (write_file,
// path), and then bounces off the ban indefinitely. The archived session
// bounced 19 times against a failure ceiling of 12 and only ended when the
// model deleted its own artifact.
//
// Nothing here calls shouldStopForFailures or appendRecentFailurePath, and
// nothing restates their conditions. The assertion is on what the loop
// actually does: how many tool calls it streams before it terminates.

const banLoopValidBody = "VALUE = 1\n"
const banLoopInvalidBody = "def broken(:\n    return 1\n"

func banLoopStubs(t *testing.T, path string, calls *int) *httptest.Server {
	t.Helper()
	write := func(content string) string {
		b, _ := json.Marshal(map[string]interface{}{
			"type": "tool_call", "name": "write_file",
			"args": map[string]string{"path": path, "content": content},
		})
		return string(b)
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Stand-in sandbox syntax checker: the gate that rejects the invalid
		// rewrite. Python-only is enough for a .py fixture.
		if strings.HasSuffix(r.URL.Path, "/syntax-check") {
			var in struct{ Code string }
			json.NewDecoder(r.Body).Decode(&in)
			valid := !strings.Contains(in.Code, "def broken(:")
			out := map[string]interface{}{"valid": valid}
			if !valid {
				out["errors"] = []string{"SyntaxError: invalid syntax (line 1)"}
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
		// Turn 0 lands a good file. Every turn after re-sends the SAME
		// invalid rewrite, which is what produces the rejection, then the
		// ban, then an unbounded run of ban bounces.
		reply := write(banLoopInvalidBody)
		if i == 0 {
			reply = write(banLoopValidBody)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		delta, _ := json.Marshal(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": reply}}},
		})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", delta)
	}))
}

func TestBannedWriteLoopTerminatesAtTheFailureCeiling(t *testing.T) {
	// NOT YET A VALID REGRESSION. As written this fixture never reaches the
	// ban branch: the syntax gate that produces the first rejection lives
	// behind `fileTier >= Tier2Medium && ctx.V3URL != ""` (tools.go:855), and
	// a small .py fixture with no V3 stub classifies T1, whose direct path
	// carries no syntax gate. The loop therefore stops via the repetition
	// detector at turn 6 instead, which is a different production branch, and
	// the invalid rewrite LANDS — overwriting the accepted file.
	//
	// Skipped rather than deleted because the harness is most of the work and
	// the missing piece is specific: a Tier2Medium fixture plus a V3 stub, or
	// an isActiveDebugIteration setup that reaches the fast-path gate at
	// tools.go:901. Until then Invariant 1 has helper-level proof only.
	t.Skip("fixture does not reach the ban branch yet; see comment")

	dir := t.TempDir()
	rel := "mod.py"
	inferenceCalls := 0
	srv := banLoopStubs(t, rel, &inferenceCalls)
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = srv.URL
	ctx.PermissionMode = PermissionYolo

	toolCalls, doneEvents := 0, 0
	ctx.StreamFn = func(eventType string, data interface{}) {
		switch eventType {
		case "tool_call":
			toolCalls++
		case "done":
			doneEvents++
		}
	}

	// A hard stop far above the ceiling. If the loop is unbounded this is
	// what ends the test instead of hanging; exceeding it IS the failure.
	const hardStop = 60
	ctx.MaxTurns = hardStop

	if err := runAgentLoop(ctx, "Create mod.py with VALUE = 1."); err != nil {
		t.Fatalf("agent loop returned an error: %v", err)
	}

	t.Logf("tool_calls=%d done_events=%d inference_calls=%d ceiling=%d",
		toolCalls, doneEvents, inferenceCalls, maxTotalFailures)

	if toolCalls >= hardStop {
		t.Fatalf("loop never terminated on its own: %d tool calls (hard stop %d); "+
			"the ban branch is counting failures without reading the ceiling",
			toolCalls, hardStop)
	}
	// One accepted write plus at most `maxTotalFailures` rejected ones, with a
	// small margin for the turn that emits the terminal event.
	if limit := maxTotalFailures + 3; toolCalls > limit {
		t.Fatalf("loop ran %d tool calls, past the failure ceiling %d (limit %d)",
			toolCalls, maxTotalFailures, limit)
	}
	if doneEvents == 0 {
		t.Fatal("loop ended without streaming a terminal done event")
	}

	// The accepted turn-0 bytes must still be the ones on disk: every later
	// write was invalid and must have been refused, not landed.
	got, err := os.ReadFile(filepath.Join(dir, rel))
	if err != nil {
		t.Fatalf("accepted artifact missing: %v", err)
	}
	if string(got) != banLoopValidBody {
		t.Fatalf("accepted artifact was overwritten: %q", string(got))
	}
}
