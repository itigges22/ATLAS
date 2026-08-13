package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// AUDIT of the SUCCESSFUL writeFileWithV3 state machine: V3 answered, and the
// route now decides which bytes land, what evidence covers those exact bytes,
// and whether V3 provenance is authorized.
//
// These fixtures RECORD; they add no classification. Every one runs the real
// write_file dispatch, so what they show is routing, not a reading of the
// source. The stub hashes the body of every check request, which is what makes
// "the evidence covers the bytes that landed" a measured claim rather than an
// assumption: a check whose hash is not the final on-disk hash did not examine
// what the model got.
//
// Terminology, kept apart on purpose:
//
//	authorization  may the candidate's BYTES replace the caller's content
//	evidence       what a checker demonstrated, and about WHICH bytes
//	local gates    the syntax/structural/embedded checks this route runs
//	provenance     the V3 metadata attached to the delivered artifact

func shortHash(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])[:8]
}

type auditCheck struct {
	kind string // "syntax" | "structural" | "embedded"
	hash string
}

type auditStub struct {
	mu     sync.Mutex
	checks []auditCheck
	v3Hits int
}

func (s *auditStub) add(kind, body string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.checks = append(s.checks, auditCheck{kind, shortHash(body)})
}

func (s *auditStub) snapshot() []auditCheck {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]auditCheck(nil), s.checks...)
}

type auditCase struct {
	name      string
	rel       string
	prior     string // on-disk content before the write ("" = absent)
	baseline  string // what the MODEL sends to write_file
	candidate string
	passed    bool // V3's top-level passed flag

	// Verdicts take the request's ordinal as well as its body: several
	// transitions ask the SAME bytes twice (the preflight before generation,
	// a gate after it), and the second answer is the one under audit.
	unresolvedFor      func(src string) []string       // structural_check verdict
	embeddedBadFor     func(src string, call int) bool // embedded_script_check verdict
	syntaxBadFor       func(src string, call int) bool // /syntax-check verdict
	cancelOnStructural bool
	readOnlyDir        bool
}

type auditResult struct {
	res    *ToolResult
	disk   string
	checks []auditCheck
	events []string
}

func runAudit(t *testing.T, c auditCase) auditResult {
	t.Helper()
	dir := t.TempDir()
	st := &auditStub{}
	reqCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var syntaxCalls, embeddedCalls int

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v3/generate":
			st.mu.Lock()
			st.v3Hits++
			st.mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			fl, _ := w.(http.Flusher)
			payload, _ := json.Marshal(map[string]interface{}{
				"code": c.candidate, "passed": c.passed,
				"phase_solved": "phase1", "candidates_tested": 3,
				"winning_score": 0.87,
				"verification_evidence": []map[string]interface{}{
					{"verifier": "sandbox", "status": "passed"},
				},
			})
			for _, line := range []string{"event: result", "data: " + string(payload), "", "data: [DONE]", ""} {
				fmt.Fprint(w, line+"\n")
				if fl != nil {
					fl.Flush()
				}
			}
		case r.URL.Path == "/internal/cyclomatic_complexity":
			json.NewEncoder(w).Encode(map[string]interface{}{"functions": []interface{}{}})
		case r.URL.Path == "/internal/structural_check":
			var body map[string]interface{}
			json.NewDecoder(r.Body).Decode(&body)
			src, _ := body["source"].(string)
			st.add("structural", src)
			if c.cancelOnStructural {
				cancel()
			}
			out := map[string]interface{}{"ok": true, "unresolved": []string{}}
			if c.unresolvedFor != nil {
				if names := c.unresolvedFor(src); len(names) > 0 {
					out["unresolved"] = names
				}
			}
			json.NewEncoder(w).Encode(out)
		case r.URL.Path == "/internal/embedded_script_check":
			var body map[string]interface{}
			json.NewDecoder(r.Body).Decode(&body)
			src, _ := body["source"].(string)
			st.add("embedded", src)
			embeddedCalls++
			out := map[string]interface{}{"ok": true, "findings": []interface{}{}}
			if c.embeddedBadFor != nil && c.embeddedBadFor(src, embeddedCalls) {
				out["findings"] = []map[string]interface{}{{
					"line": 4, "column": 1, "kind": "javascript",
					"where": "the <script> block", "message": "unexpected `)`",
					"text": "bad();)", "hint": "remove it",
				}}
			}
			json.NewEncoder(w).Encode(out)
		case strings.HasSuffix(r.URL.Path, "/syntax-check"):
			var in struct{ Code string }
			json.NewDecoder(r.Body).Decode(&in)
			st.add("syntax", in.Code)
			syntaxCalls++
			bad := c.syntaxBadFor != nil && c.syntaxBadFor(in.Code, syntaxCalls)
			out := map[string]interface{}{"valid": !bad}
			if bad {
				out["errors"] = []string{"SyntaxError: invalid syntax (line 2)"}
			}
			json.NewEncoder(w).Encode(out)
		default:
			http.Error(w, "unexpected endpoint "+r.URL.Path, http.StatusTeapot)
		}
	}))
	defer srv.Close()

	target := filepath.Join(dir, c.rel)
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if c.prior != "" {
		if err := os.WriteFile(target, []byte(c.prior), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	var events []string
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.StreamFn = func(event string, _ interface{}) { events = append(events, event) }
	ctx.Ctx = reqCtx
	ctx.BypassV3 = false
	ctx.V3URL = srv.URL
	ctx.SandboxURL = srv.URL
	ctx.SessionWrites[c.rel] = true
	ctx.Messages = []AgentMessage{
		{Role: "user", Content: "build it"},
		{Role: "tool", ToolName: "read_file", Content: c.prior},
	}

	// Routing premise: only a Tier2+ file with V3 configured, unbypassed and
	// outside an edit-test-fix loop reaches the pipeline at all.
	if tier := classifyFileTier(c.rel, c.baseline); tier < Tier2Medium {
		t.Fatalf("fixture must be Tier2+ to enter the pipeline, got %v", tier)
	}
	if isActiveDebugIteration(ctx, c.rel) {
		t.Fatal("the active-debug predicate must be false, or the fast path takes this write")
	}

	if c.readOnlyDir {
		sub := filepath.Dir(target)
		if err := os.Chmod(sub, 0o555); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.Chmod(sub, 0o755) })
	}

	args, _ := json.Marshal(map[string]string{"path": c.rel, "content": c.baseline})
	res := executeToolCall("write_file", args, ctx)

	if st.v3Hits == 0 {
		t.Fatal("V3 generation was never entered; this is not the pipeline route")
	}
	after, _ := os.ReadFile(target)
	ents, _ := os.ReadDir(filepath.Dir(target))
	for _, e := range ents {
		if strings.Contains(e.Name(), ".atlas.tmp") {
			t.Errorf("temporary artifact survived: %s", e.Name())
		}
	}
	return auditResult{res: res, disk: string(after), checks: st.snapshot(), events: events}
}

// ---------------------------------------------------------------------------
// Fixture content
// ---------------------------------------------------------------------------

const auBaselinePy = `import math


def area(radius):
    if radius < 0:
        raise ValueError("negative radius")
    return math.pi * radius * radius


def perimeter(radius):
    for _ in range(1):
        pass
    return 2 * math.pi * radius
`

const auCandidatePy = `import math


def area(radius):
    if radius < 0:
        raise ValueError("bad radius")
    return math.pi * radius ** 2


def perimeter(radius):
    while False:
        pass
    return math.tau * radius
`

// A BASELINE that itself calls a name the file never binds: the model's own
// content is not automatically the safe option.
const auBaselineUnresolved = `import math


def area(radius):
    if radius < 0:
        raise ValueError("negative radius")
    return base_helper(radius)


def perimeter(radius):
    for _ in range(1):
        pass
    return 2 * math.pi * radius
`

// A candidate that calls a name the file never binds.
const auCandidateUnresolved = `import math


def area(radius):
    if radius < 0:
        raise ValueError("bad radius")
    return missing_helper(radius)


def perimeter(radius):
    while False:
        pass
    return math.tau * radius
`

// A candidate carrying a second module entrypoint.
const auCandidateDoubleMain = auCandidatePy + `

if __name__ == "__main__":
    print(area(1))


if __name__ == "__main__":
    print(perimeter(1))
`

const auBaselineHTML = `<!doctype html>
<html>
<head><title>Board</title></head>
<body>
  <div id="board"></div>
  <script>
    const cells = [];
    for (let i = 0; i < 9; i++) { cells.push(i); }
    document.getElementById('board').textContent = cells.join(',');
  </script>
</body>
</html>
`

const auCandidateHTML = `<!doctype html>
<html>
<head><title>Board</title></head>
<body>
  <div id="board"></div>
  <script>
    const cells = [];
    for (let i = 0; i < 16; i++) { cells.push(i * 2); }
    document.getElementById('board').textContent = cells.join(' ');
  </script>
</body>
</html>
`

// No tags at all: the language-swap shape that replaced an HTML document with
// a pile of JavaScript.
const auCandidateNotHTML = `const cells = [];
for (let i = 0; i < 16; i++) {
  cells.push(i * 2);
}
console.log(cells.join(' '));
const board = cells.map(function (c) { return c + 1; });
console.log(board.length);
if (board.length > 0) { console.log('ok'); }
for (const c of board) { console.log(c); }
`

// ---------------------------------------------------------------------------
// The trace
// ---------------------------------------------------------------------------

func TestV3SuccessStateMachineAudit(t *testing.T) {
	for _, c := range auditCases() {
		c := c
		t.Run(c.name, func(t *testing.T) {
			got := runAudit(t, c)
			sanitized, wasSanitized := sanitizeFileContent(c.rel, c.candidate)

			label := func(h string) string {
				switch h {
				case shortHash(c.baseline):
					return "BASE(" + h + ")"
				case shortHash(c.candidate):
					return "CAND(" + h + ")"
				case shortHash(sanitized):
					return "SANI(" + h + ")"
				case shortHash(c.prior):
					return "PRIOR(" + h + ")"
				case shortHash(""):
					return "EMPTY"
				}
				return "OTHER(" + h + ")"
			}
			var checkLines []string
			for _, ch := range got.checks {
				checkLines = append(checkLines, ch.kind+":"+label(ch.hash))
			}
			finalOnDisk := "NOTHING"
			if got.disk != "" {
				finalOnDisk = label(shortHash(got.disk))
			}
			// authorizedV3 and fellBack are internal; provenance is their
			// observable projection, which is the property under audit.
			t.Logf("\n"+
				"  baseline        %s\n"+
				"  candidate       %s (passed=%v)\n"+
				"  sanitized       %s (changed=%v)\n"+
				"  final on disk   %s\n"+
				"  checks          %v\n"+
				"  success         %v\n"+
				"  mutation/valid  %q / %q %q\n"+
				"  provenance      v3_used=%v candidates=%d score=%.2f phase=%q evidence=%d\n"+
				"  sse             %v\n"+
				"  error           %s",
				label(shortHash(c.baseline)),
				label(shortHash(c.candidate)), c.passed,
				label(shortHash(sanitized)), wasSanitized,
				finalOnDisk, checkLines,
				got.res.Success,
				got.res.MutationStatus, got.res.ValidationKind, got.res.ValidationStatus,
				got.res.V3Used, got.res.CandidatesTested, got.res.WinningScore,
				got.res.PhaseSolved, len(got.res.VerificationEvidence),
				got.events, truncateStr(got.res.Error, 90))

			// The one invariant this audit exists to protect: provenance may
			// only describe bytes V3 authored and this route delivered.
			if got.res.V3Used {
				if got.disk != sanitized {
					t.Errorf("V3 provenance attached to bytes that are not the "+
						"delivered candidate: disk=%s sanitized=%s",
						label(shortHash(got.disk)), label(shortHash(sanitized)))
				}
			}
			if got.disk == c.baseline && got.res.V3Used {
				t.Error("baseline bytes were delivered with V3 provenance")
			}
			if !got.res.Success && got.res.V3Used {
				t.Error("a non-success result carries V3 provenance")
			}
		})
	}
}

func auditCases() []auditCase {
	unresolvedIn := func(marker string) func(string) []string {
		return func(src string) []string {
			if strings.Contains(src, marker+"(") {
				return []string{marker}
			}
			return nil
		}
	}
	return []auditCase{
		// 1. Non-passing response that still carries code. Authorization is
		// Passed, not the presence of Code.
		{name: "01_unauthorized_code_retained_baseline",
			rel: "solve.py", baseline: auBaselinePy, candidate: auCandidatePy, passed: false},

		// 2. Authorized candidate survives every gate.
		{name: "02_authorized_candidate_delivered",
			rel: "solve.py", baseline: auBaselinePy, candidate: auCandidatePy, passed: true},

		// 3. Language-swap gate revokes to the baseline.
		{name: "03_language_swap_revoked",
			rel: "index.html", baseline: auBaselineHTML, candidate: auCandidateNotHTML, passed: true},

		// 4. Sanitization rewrites an authorized candidate after V3 verified it.
		{name: "04_sanitization_rewrites_candidate",
			rel: "solve.py", baseline: auBaselinePy,
			candidate: "```python\n" + auCandidatePy + "```\n", passed: true},

		// 5. Structural gate rejects the candidate; the baseline is clean.
		{name: "05_structural_revokes_to_clean_baseline",
			rel: "solve.py", baseline: auBaselinePy, candidate: auCandidateUnresolved,
			passed: true, unresolvedFor: unresolvedIn("missing_helper")},

		// 5b. Same, but the baseline does not parse: the remaining legacy
		// checkFallbackSyntax call decides, and nothing lands.
		{name: "05b_structural_revocation_blocked_by_broken_baseline",
			rel: "solve.py", baseline: auBaselinePy, candidate: auCandidateUnresolved,
			passed: true, unresolvedFor: unresolvedIn("missing_helper"),
			// The preflight (call 1) passes the baseline; the revocation's own
			// re-check (call 2) finds it broken. Only a sequenced stub can
			// separate those two, which is the point of the ordinal.
			syntaxBadFor: func(src string, call int) bool {
				return call > 1 && src == auBaselinePy
			}},

		// 6. Structural gate finds BOTH unacceptable.
		{name: "06_structural_refuses_both",
			rel: "solve.py", baseline: auBaselineUnresolved, candidate: auCandidateUnresolved,
			passed: true, unresolvedFor: func(src string) []string {
				var names []string
				for _, n := range []string{"missing_helper", "base_helper"} {
					if strings.Contains(src, n+"(") {
						names = append(names, n)
					}
				}
				return names
			}},

		// 7. Embedded-script gate revokes the candidate to the baseline.
		{name: "07_embedded_revokes_to_baseline",
			rel: "index.html", baseline: auBaselineHTML, candidate: auCandidateHTML, passed: true,
			embeddedBadFor: func(src string, call int) bool {
				return call > 1 && strings.Contains(src, "i < 16")
			}},

		// 8. Embedded-script gate refuses both.
		{name: "08_embedded_refuses_both",
			rel: "index.html", baseline: auBaselineHTML, candidate: auCandidateHTML, passed: true,
			// Preflight (call 1) passes the baseline; afterwards the gate
			// condemns the candidate AND the baseline it would fall back to.
			embeddedBadFor: func(_ string, call int) bool { return call > 1 }},

		// 9. Duplicate-main guard refuses whatever is about to land.
		{name: "09_duplicate_main_refused",
			rel: "solve.py", baseline: auBaselinePy, candidate: auCandidateDoubleMain, passed: true},

		// 10. Cancellation during the post-generation gates.
		{name: "10_cancelled_during_gates",
			rel: "solve.py", baseline: auBaselinePy, candidate: auCandidatePy, passed: true,
			cancelOnStructural: true},

		// 11. The candidate write itself fails.
		{name: "11_candidate_write_fails",
			rel: "ro/solve.py", baseline: auBaselinePy, candidate: auCandidatePy, passed: true,
			readOnlyDir: true},

		// 12. The revoked-to-baseline write fails.
		{name: "12_baseline_write_fails",
			rel: "ro/solve.py", baseline: auBaselinePy, candidate: auCandidatePy, passed: false,
			readOnlyDir: true},
	}
}

// Q9, isolated: an unauthorized but non-empty Code must not reach disk, must
// not be checked by any gate, and must not colour any decision. The hash
// recording is what proves it -- no check examined the candidate's bytes.
func TestUnauthorizedCandidateNeverInfluencesAnything(t *testing.T) {
	got := runAudit(t, auditCase{
		rel: "solve.py", baseline: auBaselinePy, candidate: auCandidateUnresolved,
		passed: false,
		// If the unauthorized candidate were ever consulted, this verdict
		// would fire on it and change the outcome.
		unresolvedFor: func(src string) []string {
			if strings.Contains(src, "missing_helper(") {
				return []string{"missing_helper"}
			}
			return nil
		}})

	if got.disk != auBaselinePy {
		t.Fatalf("unauthorized candidate influenced the delivered bytes: %q", got.disk)
	}
	candHash := shortHash(auCandidateUnresolved)
	for _, ch := range got.checks {
		if ch.hash == candHash {
			t.Errorf("a %s check examined the unauthorized candidate", ch.kind)
		}
	}
	if got.res.V3Used {
		t.Error("unauthorized delivery carries V3 provenance")
	}
}

// Q4/Q5, isolated: sanitization rewrites the bytes AFTER V3 earned its
// evidence on them, and nothing re-runs the syntax checker on the result. The
// gates that do run afterwards are structural and embedded, never syntax --
// so the delivered artifact carries provenance for bytes that no longer exist
// in that form, and no local syntax evidence of its own.
func TestSanitizedCandidateIsNotRevalidated(t *testing.T) {
	fenced := "```python\n" + auCandidatePy + "```\n"
	got := runAudit(t, auditCase{
		rel: "solve.py", baseline: auBaselinePy, candidate: fenced, passed: true})

	sanitized, changed := sanitizeFileContent("solve.py", fenced)
	if !changed {
		t.Fatal("fixture did not exercise sanitization")
	}
	if got.disk != sanitized {
		t.Fatalf("delivered bytes are not the sanitized candidate: %q", got.disk)
	}
	if got.disk == fenced {
		t.Fatal("the fenced candidate reached disk unchanged")
	}
	for _, ch := range got.checks {
		if ch.kind == "syntax" && ch.hash == shortHash(sanitized) {
			t.Fatal("a syntax check DID examine the sanitized bytes; the audit " +
				"note about unvalidated delivery would be wrong")
		}
	}
	if !got.res.V3Used {
		t.Fatal("fixture did not reach the authorized-delivery path")
	}
}

// Q6, isolated: when a gate revokes the candidate, is the restored baseline
// freshly checked? Only the structural route re-checks it (that is the last
// remaining legacy checkFallbackSyntax call). The language-swap and
// embedded-script revocations deliver the baseline with no fresh syntax check
// of their own.
func TestRevocationRoutesDifferInBaselineRechecking(t *testing.T) {
	structural := runAudit(t, auditCase{
		rel: "solve.py", baseline: auBaselinePy, candidate: auCandidateUnresolved, passed: true,
		unresolvedFor: func(src string) []string {
			if strings.Contains(src, "missing_helper(") {
				return []string{"missing_helper"}
			}
			return nil
		}})
	swap := runAudit(t, auditCase{
		rel: "index.html", baseline: auBaselineHTML, candidate: auCandidateNotHTML, passed: true})

	countSyntaxOn := func(r auditResult, content string) int {
		n := 0
		for _, ch := range r.checks {
			if ch.kind == "syntax" && ch.hash == shortHash(content) {
				n++
			}
		}
		return n
	}
	// Preflight examined the baseline once before generation. The structural
	// revocation examines it AGAIN on the way to delivering it.
	if got := countSyntaxOn(structural, auBaselinePy); got != 2 {
		t.Errorf("structural revocation: syntax checks on the baseline = %d, want 2 "+
			"(preflight, then the revocation's own)", got)
	}
	if got := countSyntaxOn(swap, auBaselineHTML); got != 1 {
		t.Errorf("language-swap revocation: syntax checks on the baseline = %d, want 1 "+
			"(the preflight's only -- the revocation re-checks nothing)", got)
	}
	if structural.disk != auBaselinePy || swap.disk != auBaselineHTML {
		t.Fatal("both revocations must deliver the baseline")
	}
	if structural.res.V3Used || swap.res.V3Used {
		t.Error("a revoked delivery carries V3 provenance")
	}
}
