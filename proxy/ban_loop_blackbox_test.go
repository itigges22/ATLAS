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

// Tier2Medium requires a recognized code extension at >= 10 lines
// (classifyFileTier, tools.go). Both bodies clear that so the write takes the
// V3-gated branch at tools.go:855 rather than the ungated T0/T1 direct path.
const banLoopValidBody = `import sys


def alpha(n):
    if n > 0:
        return n * 2
    return 0


def beta(items):
    total = 0
    for item in items:
        total += alpha(item)
    return total


def main():
    print(beta([1, 2, 3]))


if __name__ == "__main__":
    main()
`

// Same shape, one deliberate syntax error, so the regression gate sees a
// healthy file on disk being replaced by broken content.
const banLoopInvalidBody = `import sys


def alpha(n:
    if n > 0:
        return n * 2
    return 0


def beta(items):
    total = 0
    for item in items:
        total += alpha(item)
    return total


def main():
    print(beta([1, 2, 3]))


if __name__ == "__main__":
    main()
`

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
		// V3 is required only so the write takes the gated branch; making
		// generation unavailable sends writeFileWithV3 down its fallback,
		// which still lands the baseline bytes. Invalid content never gets
		// this far -- the syntax gate runs before V3.
		if strings.HasPrefix(r.URL.Path, "/v3/") || strings.HasPrefix(r.URL.Path, "/internal/") {
			http.Error(w, "v3 unavailable in this test", http.StatusServiceUnavailable)
			return
		}
		if strings.HasSuffix(r.URL.Path, "/syntax-check") {
			var in struct{ Code string }
			json.NewDecoder(r.Body).Decode(&in)
			valid := !strings.Contains(in.Code, "def alpha(n:")
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
	dir := t.TempDir()
	rel := "mod.py"
	inferenceCalls := 0
	srv := banLoopStubs(t, rel, &inferenceCalls)
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = srv.URL
	ctx.V3URL = srv.URL
	ctx.PermissionMode = PermissionYolo

	toolCalls, doneEvents, banEntries := 0, 0, 0
	var doneSummary string
	ctx.StreamFn = func(eventType string, data interface{}) {
		switch eventType {
		case "tool_call":
			toolCalls++
		case "gate":
			// Proves the ban branch specifically, not merely that some
			// breaker stopped the session.
			if b, err := json.Marshal(data); err == nil &&
				strings.Contains(string(b), "no longer available") {
				banEntries++
			}
		case "done":
			doneEvents++
			if b, err := json.Marshal(data); err == nil {
				doneSummary = string(b)
			}
		}
	}

	// A hard stop far above the ceiling. If the loop is unbounded this is
	// what ends the test instead of hanging; exceeding it IS the failure.
	const hardStop = 60
	ctx.MaxTurns = hardStop

	if err := runAgentLoop(ctx, "Create mod.py with VALUE = 1."); err != nil {
		t.Fatalf("agent loop returned an error: %v", err)
	}

	t.Logf("tool_calls=%d ban_entries=%d done_events=%d inference_calls=%d ceiling=%d summary=%s",
		toolCalls, banEntries, doneEvents, inferenceCalls, maxTotalFailures, doneSummary)

	// Routing precondition. If a future tier threshold or V3 setup routes
	// around the intended branch this fails loudly instead of silently
	// measuring a different breaker.
	if tier := classifyFileTier(rel, banLoopValidBody); tier < Tier2Medium {
		t.Fatalf("fixture no longer classifies Tier2Medium (got %v); the write "+
			"would take the ungated direct path and this test would not "+
			"exercise the ban branch", tier)
	}
	if banEntries == 0 {
		t.Fatal("never entered the ban branch: no gate event said the tool was " +
			"no longer available, so this run measured some other breaker")
	}

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
