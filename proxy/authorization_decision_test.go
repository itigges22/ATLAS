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

// The observe-only authorization matrix.
//
// Fifteen shapes a candidate can arrive in, each answered by the typed
// decision and none of them delivering anything. A row that comes back
// authorized is describing what WOULD land if a consumer existed; the guards
// in evidence_inertness_test.go prove none does.

type authWorld struct {
	*matrixWorld
	obs []taskObligation
}

func newAuthWorld(t *testing.T, contract, filename, code string, valid bool) *authWorld {
	t.Helper()
	w := newMatrixWorld(t, contract, filename, code, valid)
	w.ctx.HumanTask = "Create it."
	return &authWorld{matrixWorld: w, obs: requestObligations(w.ctx)}
}

// observe runs the wired producer and then the decision over what it saw,
// exactly as production does.
func (w *authWorld) observe(t *testing.T) (proxyEvidence, candidateEvidenceIdentity, bool) {
	t.Helper()
	outcome := fallbackSyntaxOutcomeFor(w.ctx, w.path, w.code).aggregate()
	return observeDeliveredCandidateSyntax(w.ctx, w.path, w.code, outcome)
}

// stage runs the client-declared producer through the real staging wiring
// against the world's executor, exactly as production does.
func (w *authWorld) stage(id candidateEvidenceIdentity) []proxyEvidence {
	behavioral, _ := observeCandidateVerification(w.ctx, w.path, w.code, id)
	return behavioral
}

// decide asks the decision about the candidate the producer just observed,
// exactly as production does: the asked-for identity is the one the producer
// minted, and the evidence has to match it.
func (w *authWorld) decide(id candidateEvidenceIdentity, evidence ...proxyEvidence) AuthorizationDecision {
	return observeCandidateAuthorization(w.ctx, w.path, w.code, id, nil, evidence)
}

func expectReason(t *testing.T, row string, d AuthorizationDecision,
	authorized bool, want AuthorizationReason) {
	t.Helper()
	if d.Authorized != authorized {
		t.Errorf("%s: authorized=%v, want %v (reason %q)", row, d.Authorized, authorized, d.Reason)
	}
	if d.Reason != want {
		t.Errorf("%s: reason %q, want %q", row, d.Reason, want)
	}
	if !authorizationReasons[d.Reason] {
		t.Errorf("%s: %q is outside the closed vocabulary", row, d.Reason)
	}
	if d.InfluencesLiveDecision {
		t.Errorf("%s: the decision claims to influence a live one", row)
	}
}

const authPy = "print(7)\n"

func TestAuthorizationMatrix(t *testing.T) {
	declaredPy := `{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`

	t.Run("declared python output with exact syntax evidence", func(t *testing.T) {
		w := newAuthWorld(t, declaredPy, "solve.py", authPy, true)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		d := w.decide(evID, ev)
		expectReason(t, "exact syntax", d, true, ReasonAuthorized)
		if len(d.Satisfied) != 1 || len(d.Missing) != 0 {
			t.Errorf("satisfied %v missing %v", d.Satisfied, d.Missing)
		}
		if len(d.EvidenceConsumed) != 1 {
			t.Errorf("consumed %v, want the one record", d.EvidenceConsumed)
		}
		if !d.SettlementRequired {
			t.Error("existence is not yet settled and the decision does not say so")
		}
	})

	t.Run("declared document with no fabricated syntax requirement", func(t *testing.T) {
		w := newAuthWorld(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["notes.md"]}`,
			"notes.md", "# notes\n", true)
		if _, _, ok := w.observe(t); ok {
			t.Error("a structural record was fabricated for a document")
		}
		d := w.decide(candidateEvidenceIdentity{})
		// No prerequisite exists to satisfy, so nothing authorizes it. Being
		// the declared target is not evidence about the bytes.
		expectReason(t, "document", d, false, ReasonEvidenceMissing)
		if len(d.Missing) != 0 {
			t.Errorf("missing %v, want no fabricated obligation", d.Missing)
		}
	})

	t.Run("declared command left unstaged", func(t *testing.T) {
		w := newAuthWorld(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
				`"verification_knowledge":"declared","verification":["pytest -q"]}`,
			"solve.py", authPy, true)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		d := w.decide(evID, ev)
		// The syntax obligation is met. Staging was never run for this
		// candidate, so the command has no observation and is simply owed --
		// which is what a missing behavioral record looks like now that one
		// is producible.
		expectReason(t, "declared command", d, false, ReasonEvidenceMissing)
		if len(d.Satisfied) != 1 || len(d.Missing) != 1 {
			t.Errorf("satisfied %v missing %v, want one of each", d.Satisfied, d.Missing)
		}
		if !strings.HasPrefix(d.Missing[0], ObligationDeclaredCommand+":") {
			t.Errorf("missing %v, want the declared command", d.Missing)
		}
	})

	t.Run("exact declared command in a genuine candidate workspace", func(t *testing.T) {
		w := newAuthWorld(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
				`"verification_knowledge":"declared","verification":["pytest -q"]}`,
			"solve.py", authPy, true)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		behavioral := w.stage(evID)
		if len(behavioral) != 1 {
			t.Fatalf("staging produced %d records, want the one declared command",
				len(behavioral))
		}
		if w.shellRuns != 1 {
			t.Errorf("the executor ran %d times, want once per declared command",
				w.shellRuns)
		}
		if got := behavioral[0].Provenance.ObservedStrength; got != "behavioral" {
			t.Errorf("strength %q, want behavioral", got)
		}
		d := w.decide(evID, append([]proxyEvidence{ev}, behavioral...)...)
		expectReason(t, "staged declared command", d, true, ReasonAuthorized)
		if len(d.Missing) != 0 {
			t.Errorf("missing %v, want nothing owed", d.Missing)
		}
		if len(d.Satisfied) != 2 {
			t.Errorf("satisfied %v, want the syntax and the command", d.Satisfied)
		}
	})

	t.Run("declared command that failed in staging", func(t *testing.T) {
		w := newAuthWorld(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
				`"verification_knowledge":"declared","verification":["pytest -q"]}`,
			"solve.py", authPy, true)
		w.shellExit = 1
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		d := w.decide(evID, append([]proxyEvidence{ev}, w.stage(evID)...)...)
		expectReason(t, "failed declared command", d, false, ReasonEvidenceMissing)
	})

	t.Run("declared command that rewrote the candidate", func(t *testing.T) {
		w := newAuthWorld(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
				`"verification_knowledge":"declared","verification":["pytest -q"]}`,
			"solve.py", authPy, true)
		w.shellMutate = true
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		d := w.decide(evID, append([]proxyEvidence{ev}, w.stage(evID)...)...)
		if d.Authorized {
			t.Error("a command that rewrote its own subject authorized the candidate")
		}
	})

	t.Run("model-generated self-test passing", func(t *testing.T) {
		w := newAuthWorld(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`,
			"solve.py", authPy, true)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		// The model ran its own test and it passed. Relabelled as what it is.
		forged := ev
		forged.Provenance.Source = ProvenanceModelGenerated
		d := w.decide(evID, forged)
		expectReason(t, "model self-test", d, false, ReasonProvenanceUntrusted)
	})

	t.Run("existing syntax baseline with equal evidence", func(t *testing.T) {
		w := newAuthWorld(t, declaredPy, "solve.py", authPy, true)
		if w.ctx.Ledger == nil {
			w.ctx.Ledger = map[string]*DeliverableState{}
		}
		w.ctx.Ledger[w.path] = &DeliverableState{
			Path: w.path, CurrentHash: w.hash, Generation: 1,
			ValidationKind: ValidationKindSyntax, ValidationStatus: ValidationPassed,
			ValidatedHash: w.hash,
		}
		w.obs = requestObligations(w.ctx)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		d := w.decide(evID, ev)
		// A syntax baseline is preserved by current syntax evidence over the
		// exact candidate bytes. That is the same claim the baseline holds, so
		// nothing is owed: the alternative was refusing every replacement of a
		// validated file forever.
		expectReason(t, "syntax baseline", d, true, ReasonAuthorized)
		for _, id := range d.Missing {
			if strings.HasPrefix(id, ObligationBaselinePreserved+":") {
				t.Errorf("missing %v: equal evidence did not preserve an equal baseline",
					d.Missing)
			}
		}
	})

	t.Run("existing behavioral baseline with syntax-only evidence", func(t *testing.T) {
		w := newAuthWorld(t, declaredPy, "solve.py", authPy, true)
		w.ctx.VerificationEvidence = append(w.ctx.VerificationEvidence, VerificationRecord{
			Command: "python3 solve.py",
			Covered: map[string]string{w.path: w.hash}, Turn: 1,
		})
		w.obs = requestObligations(w.ctx)
		if got := authorizationFloor(w.obs); got != "behavioral" {
			t.Fatalf("floor %q, want behavioral", got)
		}
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		d := w.decide(evID, ev)
		// Syntax is a weaker claim than the baseline holds. Promoting it here
		// is how a working artifact gets replaced by one that merely parses.
		expectReason(t, "behavioral baseline", d, false, ReasonBaselineNotPreserved)
	})

	t.Run("existing behavioral baseline re-established by its own command", func(t *testing.T) {
		w := newAuthWorld(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
				`"verification_knowledge":"declared","verification":["python3 solve.py"]}`,
			"solve.py", authPy, true)
		w.ctx.VerificationEvidence = append(w.ctx.VerificationEvidence, VerificationRecord{
			Command: "python3 solve.py",
			Covered: map[string]string{w.path: w.hash}, Turn: 1,
		})
		w.obs = requestObligations(w.ctx)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		// The client declared the same command that established the baseline,
		// and staging ran it against the candidate.
		d := w.decide(evID, append([]proxyEvidence{ev}, w.stage(evID)...)...)
		expectReason(t, "behavioral baseline re-run", d, true, ReasonAuthorized)
	})

	t.Run("existing behavioral baseline with a different command passing", func(t *testing.T) {
		w := newAuthWorld(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
				`"verification_knowledge":"declared","verification":["ruff check ."]}`,
			"solve.py", authPy, true)
		w.ctx.VerificationEvidence = append(w.ctx.VerificationEvidence, VerificationRecord{
			Command: "python3 solve.py",
			Covered: map[string]string{w.path: w.hash}, Turn: 1,
		})
		w.obs = requestObligations(w.ctx)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		// A different command exiting zero is behavioural evidence about
		// something, and it is not evidence that what the baseline showed
		// still holds.
		d := w.decide(evID, append([]proxyEvidence{ev}, w.stage(evID)...)...)
		expectReason(t, "behavioral baseline, other command", d, false,
			ReasonBaselineNotPreserved)
	})

	t.Run("a new file owes no preservation", func(t *testing.T) {
		w := newAuthWorld(t, declaredPy, "solve.py", authPy, true)
		// Nothing was ever validated at this path, so there is no baseline to
		// preserve and no obligation to fabricate.
		for _, o := range w.obs {
			if o.Kind == ObligationBaselinePreserved {
				t.Fatal("a path with no validated baseline owed preservation")
			}
		}
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		expectReason(t, "new file", w.decide(evID, ev), true, ReasonAuthorized)
	})

	t.Run("preservation never promotes the evidence it compares", func(t *testing.T) {
		syntax, _ := newTaskObligation(ObligationBaselinePreserved, "/w/solve.py", "syntax", true)
		behavioral, _ := newTaskObligation(ObligationBaselinePreserved, "/w/solve.py", "behavioral", true)
		oracle, _ := newTaskObligation(ObligationBaselinePreserved, "/w/solve.py", "oracle", true)
		syntaxEv := proxyEvidence{Outcome: ValidationPassed,
			Provenance: V3EvidenceProvenance{ObservedStrength: "syntax"}}
		behavioralEv := proxyEvidence{Outcome: ValidationPassed,
			Provenance: V3EvidenceProvenance{ObservedStrength: "behavioral",
				CommandIdentity: contentSHA256("python3 solve.py")}}
		witness := contentSHA256("python3 solve.py")

		for _, c := range []struct {
			name    string
			o       taskObligation
			ev      []proxyEvidence
			witness string
			want    bool
		}{
			{"syntax by syntax", syntax, []proxyEvidence{syntaxEv}, "", true},
			{"syntax by behavioral", syntax, []proxyEvidence{behavioralEv}, witness, true},
			{"behavioral by syntax", behavioral, []proxyEvidence{syntaxEv}, witness, false},
			{"behavioral by its own command", behavioral, []proxyEvidence{behavioralEv}, witness, true},
			{"behavioral by another command", behavioral, []proxyEvidence{behavioralEv},
				contentSHA256("ruff check ."), false},
			{"behavioral with no witness", behavioral, []proxyEvidence{behavioralEv}, "", false},
			{"oracle by behavioral", oracle, []proxyEvidence{behavioralEv}, witness, false},
			{"nothing by nothing", behavioral, nil, witness, false},
		} {
			got, _ := baselinePreservedBy(c.o, c.witness, c.ev)
			if got != c.want {
				t.Errorf("%s: preserved=%v, want %v", c.name, got, c.want)
			}
		}
	})

	t.Run("stale workspace", func(t *testing.T) {
		w := newAuthWorld(t, declaredPy, "solve.py", authPy, true)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		// Something mutated the workspace after the observation.
		stale := ev
		stale.Provenance.WorkspaceGeneration += 3
		d := w.decide(evID, stale)
		expectReason(t, "stale workspace", d, false, ReasonWorkspaceStale)
	})

	t.Run("candidate hash mismatch", func(t *testing.T) {
		w := newAuthWorld(t, declaredPy, "solve.py", authPy, true)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		other := ev
		other.Provenance.CandidateHash = contentSHA256("print(8)\n")
		d := w.decide(evID, other)
		expectReason(t, "hash mismatch", d, false, ReasonCandidateMismatch)
	})

	t.Run("cross-request and cross-invocation borrowing", func(t *testing.T) {
		w := newAuthWorld(t, declaredPy, "solve.py", authPy, true)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		for _, c := range []struct {
			name string
			mut  func(*proxyEvidence)
		}{
			{"another request", func(e *proxyEvidence) { e.Provenance.RequestID = "req-other" }},
			{"another invocation", func(e *proxyEvidence) { e.Provenance.InvocationID = "req-matrix:inv:99" }},
		} {
			borrowed := ev
			c.mut(&borrowed)
			d := w.decide(evID, borrowed)
			expectReason(t, c.name, d, false, ReasonRequestOrInvocationMismatch)
		}
	})

	t.Run("legacy envelope", func(t *testing.T) {
		w := newAuthWorld(t, declaredPy, "solve.py", authPy, true)
		ev, evID, _ := w.observe(t)
		d := observeCandidateAuthorization(w.ctx, w.path, w.code, evID,
			&V3EvidenceEnvelope{}, []proxyEvidence{ev})
		expectReason(t, "legacy envelope", d, false, ReasonLegacyRecord)
	})

	t.Run("undeclared target", func(t *testing.T) {
		w := newAuthWorld(t, declaredPy, "solve.py", authPy, true)
		ev, evID, _ := w.observe(t)
		d := observeCandidateAuthorization(w.ctx,
			resolveAgentPath(w.ctx, "other.py"), w.code, evID, nil, []proxyEvidence{ev})
		expectReason(t, "undeclared target", d, false, ReasonTargetNotDeclared)
	})

	t.Run("multiple obligations with one missing", func(t *testing.T) {
		w := newAuthWorld(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
				`"verification_knowledge":"declared","verification":["pytest -q","ruff check ."]}`,
			"solve.py", authPy, true)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		d := w.decide(evID, ev)
		expectReason(t, "one of three", d, false, ReasonEvidenceMissing)
		if len(d.Satisfied) != 1 || len(d.Missing) != 2 {
			t.Errorf("satisfied %v missing %v, want 1 and 2", d.Satisfied, d.Missing)
		}
	})

	t.Run("repair or refinement candidate", func(t *testing.T) {
		// A later candidate for the same target in the same request. It gets
		// its own identity, and the earlier candidate's evidence does not
		// carry over to it.
		w := newAuthWorld(t, declaredPy, "solve.py", authPy, true)
		first, _, ok := w.observe(t)
		if !ok {
			t.Fatal("the first candidate produced nothing")
		}
		second, secondID, ok := w.observe(t)
		if !ok {
			t.Fatal("the repair candidate produced nothing")
		}
		if first.Provenance.CandidateInstanceID == second.Provenance.CandidateInstanceID {
			t.Fatal("the repair candidate reused the first candidate's identity")
		}
		// The repair candidate carries the same obligation record as a
		// generation candidate does, and authorizes on its own evidence.
		d := w.decide(secondID, second)
		expectReason(t, "repair candidate", d, true, ReasonAuthorized)
		// The first candidate's evidence cannot stand in for it.
		stale := observeCandidateAuthorization(w.ctx, w.path, w.code, secondID, nil,
			[]proxyEvidence{first, second})
		if !stale.Authorized {
			t.Errorf("the current candidate's own evidence stopped working: %q", stale.Reason)
		}
		if len(stale.EvidenceConsumed) != 1 {
			t.Errorf("consumed %v, want only the current candidate's record",
				stale.EvidenceConsumed)
		}
	})

	t.Run("unknown kind, source, strength or version", func(t *testing.T) {
		w := newAuthWorld(t, declaredPy, "solve.py", authPy, true)
		ev, evID, ok := w.observe(t)
		if !ok {
			t.Fatal("the wired producer saw nothing")
		}
		t.Run("unknown source", func(t *testing.T) {
			forged := ev
			forged.Provenance.Source = "something_new"
			expectReason(t, "unknown source", w.decide(evID, forged), false, ReasonProvenanceUntrusted)
		})
		t.Run("unknown obligation kind", func(t *testing.T) {
			in := authorizationInput{
				Obligations: []taskObligation{
					{ID: "artifact_exists:x", Kind: ObligationArtifactExists,
						Subject: w.path, RequiredStrength: "syntax", Required: true},
					{ID: "forged:x", Kind: "forged", Subject: w.path,
						RequiredStrength: "syntax", Required: true},
				},
				TargetPath: w.path, CandidateHash: w.hash,
			}
			// A kind with no role is not a prerequisite and not settlement, so
			// nothing can be shown about it: the task states no satisfiable
			// prerequisite at all.
			d := decideAuthorization(w.ctx, in)
			if d.Authorized {
				t.Errorf("an unclassified obligation authorized: %q", d.Reason)
			}
		})
		t.Run("unsupported obligation", func(t *testing.T) {
			unsup, _ := newTaskObligation(ObligationUnsupported, "a thing", "", true)
			exists, _ := newTaskObligation(ObligationArtifactExists, w.path, "", true)
			d := decideAuthorization(w.ctx, authorizationInput{
				Obligations:   []taskObligation{exists, unsup},
				TargetPath:    w.path,
				CandidateHash: w.hash,
			})
			expectReason(t, "unsupported obligation", d, false, ReasonObligationUnknown)
		})
		t.Run("unknown strength", func(t *testing.T) {
			forged := ev
			forged.Provenance.RequiredStrength = "very_strong"
			d := w.decide(evID, forged)
			if d.Authorized {
				t.Errorf("an unknown strength authorized: %q", d.Reason)
			}
		})
		t.Run("unknown wire version", func(t *testing.T) {
			d := observeCandidateAuthorization(w.ctx, w.path, w.code, evID,
				&V3EvidenceEnvelope{WireVersion: "99.0.0"}, []proxyEvidence{ev})
			expectReason(t, "unknown wire version", d, false, ReasonLegacyRecord)
		})
	})
}

// --- the decision is inert ---------------------------------------------------

func TestTheAuthorizationDecisionCarriesNoContent(t *testing.T) {
	const secret = "TOKEN = 'hunter2'\nprint(7)\n"
	w := newAuthWorld(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`,
		"solve.py", secret, true)
	ev, evID, ok := w.observe(t)
	if !ok {
		t.Fatal("no evidence")
	}
	d := w.decide(evID, ev)
	blob, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	for _, needle := range []string{secret, "hunter2", "TOKEN", "print(7)"} {
		if strings.Contains(string(blob), needle) {
			t.Errorf("the decision carries %q", needle)
		}
	}
}

func TestEveryReasonIsInTheClosedVocabulary(t *testing.T) {
	required := []AuthorizationReason{
		ReasonTargetNotDeclared, ReasonObligationUnknown, ReasonAdapterUnsupported,
		ReasonEvidenceMissing, ReasonEvidenceTooWeak, ReasonProvenanceUntrusted,
		ReasonCandidateMismatch, ReasonRequestOrInvocationMismatch,
		ReasonWorkspaceStale, ReasonBaselineNotPreserved, ReasonCommandMismatch,
		ReasonLegacyRecord, ReasonPostDeliverySettlementPending,
	}
	for _, r := range required {
		if !authorizationReasons[r] {
			t.Errorf("%q is required by the contract and not in the set", r)
		}
	}
	if authorizationReasons["something_new"] {
		t.Error("an unclassified reason is accepted")
	}
}

func TestAContradictoryStateNeverAuthorizes(t *testing.T) {
	w := newAuthWorld(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`,
		"solve.py", authPy, true)
	// No obligations at all: nothing to authorize on.
	d := decideAuthorization(w.ctx, authorizationInput{})
	if d.Authorized {
		t.Error("an empty input authorized")
	}
	if d.Reason != ReasonLegacyRecord {
		t.Errorf("reason %q, want legacy_record", d.Reason)
	}
	// A nil context with obligations still cannot settle.
	exists, _ := newTaskObligation(ObligationArtifactExists, w.path, "", true)
	d = decideAuthorization(nil, authorizationInput{
		Obligations: []taskObligation{exists}, TargetPath: w.path, CandidateHash: w.hash})
	if d.Authorized {
		t.Errorf("a nil context authorized: %q", d.Reason)
	}
}

// TestTheDecisionReachesNoLiveWrite inspects the owner's CALLS, not its prose.
// A docstring may name the live decision it sits beside; the code may not call
// it, nor anything that delivers, mutates, completes or generates.
func TestTheDecisionReachesNoLiveWrite(t *testing.T) {
	const file = "authorization_decision.go"
	fset := token.NewFileSet()
	tree, err := parser.ParseFile(fset, file, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	banned := map[string]bool{
		"authorizedV3Replacement": true, "v3DeliveryAuthorized": true,
		"writeFileRecorded": true, "finalizeCompletion": true,
		"terminalCompletionAllowed": true, "callV3Generate": true,
		"callV3GenerateStreaming": true, "revokeV3": true,
		"WriteFile": true, "Remove": true, "RemoveAll": true,
		"StreamFn": true, "awaitPermission": true,
	}
	ast.Inspect(tree, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		var name string
		switch fn := call.Fun.(type) {
		case *ast.Ident:
			name = fn.Name
		case *ast.SelectorExpr:
			name = fn.Sel.Name
		}
		if banned[name] {
			t.Errorf("the authorization owner calls %s", name)
		}
		return true
	})
	src, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(src), `"influences_live_decision": false`) {
		t.Error("the record does not declare itself inert")
	}
}
