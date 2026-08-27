package main

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// What enforce actually does at the write path.
//
// Observe is the shipped default and changes nothing. Enforce lets an
// invocation with no closure path skip candidate generation and continue
// through the same direct-write path a V3 outage takes -- which has a product
// consequence worth stating plainly rather than discovering: an ordinary
// unspecified request, owned by nobody's contract, has no structured closure
// path, so under enforce it stops generating candidates and behaves like
// planner-only. That is why enforce is not the default.

func TestObserveModeGeneratesForEveryShape(t *testing.T) {
	for _, contract := range []string{
		"", `{"task_mode":"work"}`,
		`{"task_mode":"work","output_knowledge":"unspecified"}`,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":[]}`,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`,
	} {
		w := newRouteWorld(t, contract, nil)
		// Default: nothing set.
		if _, err := w.write(t); err != nil {
			t.Fatalf("%q: %v", contract, err)
		}
		if w.generateCalls() != 1 {
			t.Errorf("%q: %d generation calls under observe, want 1",
				contract, w.generateCalls())
		}
	}
}

func TestEnforceSkipsOnlyInvocationsWithNoClosurePath(t *testing.T) {
	for _, c := range []struct {
		name, contract string
		wantGenerate   int
		wantWinner     bool
	}{
		// A declared code output has a producer and a closure path.
		{"declared code output",
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`,
			1, true},
		// Nothing structured was stated, so there is no closure path to
		// reach. This is the product consequence, and it is intended.
		{"no contract", "", 0, false},
		{"task mode only", `{"task_mode":"work"}`, 0, false},
		{"unspecified outputs",
			`{"task_mode":"work","output_knowledge":"unspecified"}`, 0, false},
		// Declared-and-empty authorizes no target, so it has nowhere to close.
		{"declared empty",
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":[]}`,
			0, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			w := newRouteWorld(t, c.contract, nil)
			w.ctx.FeasibilityMode = FeasibilityEnforce
			res, err := w.write(t)
			if err != nil {
				t.Fatalf("write failed: %v", err)
			}
			if w.generateCalls() != c.wantGenerate {
				t.Errorf("%d generation calls, want %d", w.generateCalls(), c.wantGenerate)
			}
			onDisk, err := os.ReadFile(w.path)
			if err != nil {
				t.Fatal(err)
			}
			if landed := string(onDisk) == routeWinner; landed != c.wantWinner {
				t.Errorf("winner landed=%v, want %v", landed, c.wantWinner)
			}
			// A skipped invocation writes the caller's own content and claims
			// nothing about a pipeline that never ran.
			if c.wantGenerate == 0 {
				if string(onDisk) != routeBaseline {
					t.Errorf("disk holds %q, want the caller's own content", string(onDisk))
				}
				if res.V3Used || res.CandidatesTested != 0 || res.WinningScore != 0 ||
					res.PhaseSolved != "" || len(res.VerificationEvidence) != 0 {
					t.Errorf("a skipped invocation carries candidate metadata: %+v", res)
				}
				if res.AuthorizedDeliveryHash != "" {
					t.Error("a skipped invocation named an authorization")
				}
				if !res.Success {
					t.Errorf("the direct write did not succeed: %q", res.Error)
				}
			}
		})
	}
}

// TestEnforceMatchesTheDirectPathItFallsBackTo pins that a skip is the SAME
// path a V3 outage takes, not a third behaviour.
func TestEnforceMatchesTheDirectPathItFallsBackTo(t *testing.T) {
	// Enforce, no closure path: skipped.
	skipped := newRouteWorld(t, `{"task_mode":"work"}`, nil)
	skipped.ctx.FeasibilityMode = FeasibilityEnforce
	skippedRes, err := skipped.write(t)
	if err != nil {
		t.Fatal(err)
	}
	// Observe, and the service is unreachable: the existing fallback.
	outage := newRouteWorld(t, `{"task_mode":"work"}`, nil)
	outage.ctx.V3URL = "http://127.0.0.1:1"
	outageRes, err := outage.write(t)
	if err != nil {
		t.Fatal(err)
	}

	for _, c := range []struct {
		name      string
		got, want interface{}
	}{
		{"success", skippedRes.Success, outageRes.Success},
		{"mutation", skippedRes.MutationStatus, outageRes.MutationStatus},
		{"validation kind", skippedRes.ValidationKind, outageRes.ValidationKind},
		{"validation status", skippedRes.ValidationStatus, outageRes.ValidationStatus},
		{"v3 used", skippedRes.V3Used, outageRes.V3Used},
		{"authorization", skippedRes.AuthorizedDeliveryHash, outageRes.AuthorizedDeliveryHash},
	} {
		if c.got != c.want {
			t.Errorf("%s differs from the outage path: %v vs %v", c.name, c.got, c.want)
		}
	}
	a, err := os.ReadFile(skipped.path)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(outage.path)
	if err != nil {
		t.Fatal(err)
	}
	if string(a) != string(b) {
		t.Errorf("disk differs: skipped %q, outage %q", string(a), string(b))
	}
}

func TestASkippedInvocationInventsNoPipelineRecord(t *testing.T) {
	w := newRouteWorld(t, `{"task_mode":"work"}`, nil)
	w.ctx.FeasibilityMode = FeasibilityEnforce
	recs := captureShadow(t, func() {
		if _, err := w.write(t); err != nil {
			t.Fatal(err)
		}
	})
	for _, kind := range []string{
		"candidate_evidence_observation", "candidate_authorization_decision",
		"authorization_grant_event",
	} {
		if got := recordsOfKind(recs, kind); len(got) != 0 {
			t.Errorf("a skipped invocation wrote %d %s records", len(got), kind)
		}
	}
	feas := recordsOfKind(recs, "shadow_invocation_feasibility")
	if len(feas) != 1 {
		t.Fatalf("%d feasibility records, want one", len(feas))
	}
	if feas[0]["generation_proceeded"] != false {
		t.Error("the record says generation proceeded when it did not")
	}
	if feas[0]["influences_live_decision"] != true {
		t.Error("under enforce the record must say the answer decided something")
	}
	if feas[0]["mode"] != string(FeasibilityEnforce) {
		t.Errorf("mode %v, want enforce", feas[0]["mode"])
	}
	if !feasibilityReasons[FeasibilityReason(feas[0]["reason"].(string))] {
		t.Errorf("reason %v is outside the closed vocabulary", feas[0]["reason"])
	}
	blob, err := json.Marshal(recs)
	if err != nil {
		t.Fatal(err)
	}
	for _, needle := range []string{routeBaseline, routeWinner, w.dir, "solve.py"} {
		if strings.Contains(string(blob), needle) {
			t.Errorf("a skipped invocation's records carry %q", needle)
		}
	}
}

func TestObserveRecordsThatItDecidedNothing(t *testing.T) {
	w := newRouteWorld(t, `{"task_mode":"work"}`, nil)
	recs := captureShadow(t, func() {
		if _, err := w.write(t); err != nil {
			t.Fatal(err)
		}
	})
	feas := recordsOfKind(recs, "shadow_invocation_feasibility")
	if len(feas) != 1 {
		t.Fatalf("%d feasibility records, want one", len(feas))
	}
	if feas[0]["influences_live_decision"] != false {
		t.Error("under observe the record must say the answer decided nothing")
	}
	if feas[0]["generation_proceeded"] != true {
		t.Error("observe did not proceed to generation")
	}
	if feas[0]["mode"] != string(FeasibilityObserve) {
		t.Errorf("mode %v, want observe", feas[0]["mode"])
	}
}

// TestTheSkipChangesNothingModelVisible pins the boundary the mode may not
// cross: the plan, the prompt, recovery, terminal rules, permissions and the
// completion gates are not this decision's business.
func TestTheSkipChangesNothingModelVisible(t *testing.T) {
	files := proxyFiles(t)
	banned := map[string]bool{
		"buildSystemPrompt": true, "systemPrompt": true, "callLLMOnce": true,
		"callLLMOnceWithGrammar": true, "fetchFencedContent": true,
		"finalizeCompletion": true, "terminalCompletionAllowed": true,
		"emitTerminal": true, "awaitPermission": true, "requestPermission": true,
		"decideActionDemand": true, "decideVerificationDemand": true,
		"callV3Plan": true,
	}
	for _, fn := range []string{"generationSkipped", "feasibilityModeOf",
		"ParseFeasibilityMode", "decideInvocationFeasibility"} {
		for site := range callSites(files, fn) {
			caller := site[strings.Index(site, ":")+1:]
			if banned[caller] {
				t.Errorf("%s consults the feasibility mode", caller)
			}
		}
	}
	f := files["feasibility_decision.go"]
	if f == nil {
		t.Fatal("the feasibility owner is gone")
	}
	src, err := os.ReadFile("feasibility_decision.go")
	if err != nil {
		t.Fatal(err)
	}
	for name := range banned {
		if strings.Contains(string(src), name+"(") {
			t.Errorf("the feasibility owner calls %s", name)
		}
	}
}

// TestTheDecisionIsFrozenForOneInvocation pins the scope: one answer per
// invocation, a later invocation recomputes, and nothing is retroactive.
func TestTheDecisionIsFrozenForOneInvocation(t *testing.T) {
	w := newRouteWorld(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`, nil)
	w.ctx.FeasibilityMode = FeasibilityEnforce
	if _, err := w.write(t); err != nil {
		t.Fatal(err)
	}
	if w.generateCalls() != 1 {
		t.Fatalf("%d generation calls, want 1", w.generateCalls())
	}
	// Availability disappears. The NEXT invocation recomputes and skips; the
	// one that already ran is untouched.
	w.ctx.SandboxURL = ""
	if _, err := w.write(t); err != nil {
		t.Fatal(err)
	}
	if w.generateCalls() != 1 {
		t.Errorf("%d generation calls after availability went away, want still 1",
			w.generateCalls())
	}
}
