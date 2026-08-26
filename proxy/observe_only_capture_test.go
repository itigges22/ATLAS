package main

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// What the observe-only machinery actually writes on a real delivery.
//
// The inertness comparison proves the production path did not change. That
// proof is only worth something if the observers RAN, so this captures what
// they emitted and checks the two properties they promise: every record is
// attributable, and no record carries content.

// captureShadow runs fn with a live shadow sink and returns the records it
// wrote.
func captureShadow(t *testing.T, fn func()) []map[string]interface{} {
	t.Helper()
	path := t.TempDir() + "/capture.jsonl"
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	sink := newShadowSink(f)
	go sink.run()
	prev := activeShadowSink.Swap(sink)
	defer activeShadowSink.Store(prev)

	fn()

	sink.finalize()
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out []map[string]interface{}
	for _, line := range strings.Split(strings.TrimSpace(string(body)), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var rec map[string]interface{}
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("undecodable record %q: %v", line, err)
		}
		out = append(out, rec)
	}
	return out
}

func recordsOfKind(recs []map[string]interface{}, kind string) []map[string]interface{} {
	var out []map[string]interface{}
	for _, r := range recs {
		if r["record_kind"] == kind {
			out = append(out, r)
		}
	}
	return out
}

func TestTheObserversRunAndRecordAttributably(t *testing.T) {
	const secret = "TOKEN = 'hunter2'\nprint(7)\n"
	w := newAuthWorld(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
			`"verification_knowledge":"declared","verification":["pytest --token=hunter2"]}`,
		"solve.py", secret, true)

	recs := captureShadow(t, func() {
		observeInvocationFeasibility(w.ctx)
		outcome := fallbackSyntaxOutcomeFor(w.ctx, w.path, w.code).aggregate()
		if ev, id, ok := observeDeliveredCandidateSyntax(w.ctx, w.path, w.code, outcome); ok {
			observeCandidateAuthorization(w.ctx, w.path, w.code, id, nil, []proxyEvidence{ev})
		} else {
			t.Error("the producer did not run on a declared code output")
		}
	})

	for _, kind := range []string{
		"shadow_invocation_feasibility",
		"candidate_evidence_observation",
		"shadow_authorization_decision",
	} {
		got := recordsOfKind(recs, kind)
		if len(got) != 1 {
			t.Fatalf("%s: %d records, want exactly one", kind, len(got))
		}
		rec := got[0]
		if rec["influences_live_decision"] != false {
			t.Errorf("%s does not declare itself inert", kind)
		}
		if rec["request_id"] != "req-matrix" {
			t.Errorf("%s is not attributable to the request: %v", kind, rec["request_id"])
		}
		if rec["schema_version"] == nil || rec["build_version"] == nil {
			t.Errorf("%s carries no version", kind)
		}
	}

	// The evidence and authorization records join to the same candidate.
	ev := recordsOfKind(recs, "candidate_evidence_observation")[0]
	auth := recordsOfKind(recs, "shadow_authorization_decision")[0]
	for _, field := range []string{"invocation_id", "candidate_instance_id", "candidate_hash"} {
		if ev[field] == "" || ev[field] == nil {
			t.Errorf("the evidence record has no %s", field)
		}
		if ev[field] != auth[field] {
			t.Errorf("%s differs between the evidence (%v) and the decision (%v)",
				field, ev[field], auth[field])
		}
	}

	// The feasibility answer is about this invocation and says what it read.
	feas := recordsOfKind(recs, "shadow_invocation_feasibility")[0]
	if feas["generation_proceeded"] != true {
		t.Error("the feasibility record does not state that generation went ahead")
	}
	if feas["reason"] == nil || !feasibilityReasons[FeasibilityReason(feas["reason"].(string))] {
		t.Errorf("feasibility reason %v is outside the closed vocabulary", feas["reason"])
	}

	// And the decision is refused, because the declared command has no
	// producer: an observe-only run that always said yes would prove nothing.
	if auth["authorized"] != false {
		t.Error("the shadow decision authorized a task whose command has no source")
	}
	if auth["reason"] != string(ReasonEvidenceMissing) {
		t.Errorf("reason %v, want evidence_missing", auth["reason"])
	}

	// No record carries content, anywhere.
	blob, err := json.Marshal(recs)
	if err != nil {
		t.Fatal(err)
	}
	for _, needle := range []string{
		secret, "hunter2", "TOKEN", "print(7)", "pytest",
	} {
		if strings.Contains(string(blob), needle) {
			t.Errorf("the capture carries %q", needle)
		}
	}
}

// TestNoObserverEmitsAnSSEEventOrToolResult pins that none of the three
// observers can reach the caller: the private sink is the only destination.
//
// It reads the CODE, not the prose. A file may say in a comment that it emits
// no ToolResult; what it may not do is build or return one.
func TestNoObserverEmitsAnSSEEventOrToolResult(t *testing.T) {
	fset := token.NewFileSet()
	bannedTypes := map[string]bool{"ToolResult": true, "AgentMessage": true}
	bannedCalls := map[string]bool{
		"StreamFn": true, "endStream": true, "WriteFile": true,
		"Remove": true, "RemoveAll": true, "Rename": true, "Create": true,
	}
	for _, file := range []string{
		"evidence_wiring.go", "authorization_decision.go", "feasibility_decision.go",
	} {
		tree, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(tree, func(n ast.Node) bool {
			switch node := n.(type) {
			case *ast.CompositeLit:
				if id, ok := node.Type.(*ast.Ident); ok && bannedTypes[id.Name] {
					t.Errorf("%s builds a %s", file, id.Name)
				}
			case *ast.CallExpr:
				var name string
				switch fn := node.Fun.(type) {
				case *ast.Ident:
					name = fn.Name
				case *ast.SelectorExpr:
					name = fn.Sel.Name
				}
				if bannedCalls[name] {
					t.Errorf("%s calls %s", file, name)
				}
			case *ast.IndexExpr:
				// ctx.Ledger[...] as an assignment target would be a mutation.
				if sel, ok := node.X.(*ast.SelectorExpr); ok && sel.Sel.Name == "Ledger" {
					t.Errorf("%s indexes the ledger", file)
				}
			}
			return true
		})
	}
}

// TestTheObserversAreSilentWithNoSink pins that the whole apparatus costs
// nothing when capture is off: no record, no panic, same answers.
func TestTheObserversAreSilentWithNoSink(t *testing.T) {
	prev := activeShadowSink.Swap(nil)
	defer activeShadowSink.Store(prev)

	w := newAuthWorld(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`,
		"solve.py", authPy, true)
	f := observeInvocationFeasibility(w.ctx)
	if !f.Feasible {
		t.Errorf("feasibility changed with the sink off: %q", f.Reason)
	}
	outcome := fallbackSyntaxOutcomeFor(w.ctx, w.path, w.code).aggregate()
	ev, id, ok := observeDeliveredCandidateSyntax(w.ctx, w.path, w.code, outcome)
	if !ok {
		t.Fatal("the producer went silent with the sink off")
	}
	d := observeCandidateAuthorization(w.ctx, w.path, w.code, id, nil, []proxyEvidence{ev})
	if !d.Authorized {
		t.Errorf("the decision changed with the sink off: %q", d.Reason)
	}
}
