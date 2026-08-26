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

// The observe-only feasibility matrix, and the Stage-A replay it exists for.

func feasibilityFor(t *testing.T, contract string, files map[string]string) FeasibilityDecision {
	t.Helper()
	ctx, obs := rolesFixture(t, contract, files)
	_ = ctx
	return decideInvocationFeasibility(feasibilityInput{
		Obligations:       obs,
		AuthorizedTargets: authorizedTargets(obs),
		Producible:        producibleStrengths(),
	})
}

func expectFeasibility(t *testing.T, row string, d FeasibilityDecision,
	feasible bool, want FeasibilityReason) {
	t.Helper()
	if d.Feasible != feasible {
		t.Errorf("%s: feasible=%v, want %v (reason %q, unreachable %v)",
			row, d.Feasible, feasible, d.Reason, d.Unreachable)
	}
	if d.Reason != want {
		t.Errorf("%s: reason %q, want %q", row, d.Reason, want)
	}
	if !feasibilityReasons[d.Reason] {
		t.Errorf("%s: %q is outside the closed vocabulary", row, d.Reason)
	}
	if d.InfluencesLiveDecision {
		t.Errorf("%s: the decision claims to influence a live one", row)
	}
}

func TestFeasibilityMatrix(t *testing.T) {
	t.Run("declared code output", func(t *testing.T) {
		// The proxy's own gate is wired and reaches the syntax floor.
		d := feasibilityFor(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`, nil)
		expectFeasibility(t, "code output", d, true, FeasibilityClosurePathAvailable)
		if d.Floor != "syntax" {
			t.Errorf("floor %q", d.Floor)
		}
		if len(d.Unreachable) != 0 {
			t.Errorf("unreachable %v", d.Unreachable)
		}
	})

	t.Run("declared document with no verification", func(t *testing.T) {
		// A target and nothing to demonstrate about it. Naming a path is not
		// a closure path.
		d := feasibilityFor(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["notes.md"]}`, nil)
		expectFeasibility(t, "document", d, false, FeasibilityNoTrustedSource)
		if d.Floor != "" {
			t.Errorf("floor %q, want none stated", d.Floor)
		}
	})

	t.Run("declared command with no producer", func(t *testing.T) {
		d := feasibilityFor(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
				`"verification_knowledge":"declared","verification":["pytest -q"]}`, nil)
		expectFeasibility(t, "declared command", d, false, FeasibilityAdapterCannotMeasure)
		if len(d.Unreachable) != 1 ||
			!strings.HasPrefix(d.Unreachable[0], ObligationDeclaredCommand+":") {
			t.Errorf("unreachable %v, want the declared command", d.Unreachable)
		}
	})

	t.Run("existing validated baseline", func(t *testing.T) {
		ctx, _ := rolesFixture(t, "", map[string]string{"solve.py": "print(7)\n"})
		resolved := resolveAgentPath(ctx, "solve.py")
		disk := fileSHA256(ctx, resolved)
		ctx.VerificationEvidence = append(ctx.VerificationEvidence, VerificationRecord{
			Command: "python3 solve.py",
			Covered: map[string]string{resolved: disk}, Turn: 1,
		})
		ctx.TaskContract = mustContract(t, ctx.WorkingDir,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`)
		obs := requestObligations(ctx)
		d := decideInvocationFeasibility(feasibilityInput{
			Obligations:       obs,
			AuthorizedTargets: authorizedTargets(obs),
			Producible:        producibleStrengths(),
		})
		expectFeasibility(t, "behavioral baseline", d, false,
			FeasibilityBaselineFloorUnreachable)
		if d.Floor != "behavioral" {
			t.Errorf("floor %q, want the baseline's behavioral", d.Floor)
		}
	})

	t.Run("unsupported obligation", func(t *testing.T) {
		exists, _ := newTaskObligation(ObligationArtifactExists, "a.py", "", true)
		unsup, _ := newTaskObligation(ObligationUnsupported, "a thing", "", true)
		d := decideInvocationFeasibility(feasibilityInput{
			Obligations:       []taskObligation{exists, unsup},
			AuthorizedTargets: []string{"a.py"},
			Producible:        producibleStrengths(),
		})
		expectFeasibility(t, "unsupported", d, false, FeasibilityUnsupportedObligation)
	})

	t.Run("unspecified contract", func(t *testing.T) {
		for _, body := range []string{
			"", `{"task_mode":"work"}`,
			`{"task_mode":"work","output_knowledge":"unspecified","verification_knowledge":"unspecified"}`,
		} {
			d := feasibilityFor(t, body, nil)
			expectFeasibility(t, body, d, false, FeasibilityUnspecifiedContract)
			if len(d.Unreachable) != 0 {
				t.Errorf("%q named %v unreachable", body, d.Unreachable)
			}
		}
	})

	t.Run("a prerequisite with no target", func(t *testing.T) {
		// Authoritative none for outputs, a declared command for verification.
		// Even if the command had a producer there would be nothing to deliver.
		cmd, _ := newTaskObligation(ObligationSyntacticValidity, "a.py", "", true)
		d := decideInvocationFeasibility(feasibilityInput{
			Obligations:       []taskObligation{cmd},
			AuthorizedTargets: nil,
			Producible:        producibleStrengths(),
		})
		expectFeasibility(t, "no target", d, false, FeasibilityNoTrustedSource)
	})

	t.Run("unknown obligation kind", func(t *testing.T) {
		forged := taskObligation{ID: "forged:0", Kind: "forged", Subject: "a.py",
			RequiredStrength: "syntax", Required: true}
		exists, _ := newTaskObligation(ObligationArtifactExists, "a.py", "", true)
		d := decideInvocationFeasibility(feasibilityInput{
			Obligations:       []taskObligation{exists, forged},
			AuthorizedTargets: []string{"a.py"},
			Producible:        producibleStrengths(),
		})
		// A kind with no role is not a prerequisite, so there is nothing left
		// to demonstrate and no closure path.
		if d.Feasible {
			t.Errorf("an unclassified obligation was called feasible: %q", d.Reason)
		}
	})
}

// --- a later invocation recomputes; this one does not gain authority ---------

func TestALaterInvocationRecomputesFromWhatIsAvailableThen(t *testing.T) {
	_, obs := rolesFixture(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
			`"verification_knowledge":"declared","verification":["pytest -q"]}`, nil)
	in := feasibilityInput{
		Obligations:       obs,
		AuthorizedTargets: authorizedTargets(obs),
		Producible:        producibleStrengths(),
	}
	now := decideInvocationFeasibility(in)
	if now.Feasible {
		t.Fatal("the declared command has no producer and was called feasible")
	}
	// A future build wires the verification producer. The SAME obligations
	// then have a path -- and this is a new computation, not a retroactive
	// upgrade of the one above.
	later := in
	later.Producible = map[string]string{
		ObligationSyntacticValidity: "syntax",
		ObligationDeclaredCommand:   "behavioral",
	}
	future := decideInvocationFeasibility(later)
	expectFeasibility(t, "with a wired producer", future, true,
		FeasibilityClosurePathAvailable)
	if now.Feasible {
		t.Error("the earlier invocation's answer changed")
	}
}

// --- the Stage-A replay ------------------------------------------------------

// TestTheStageARequestShapeHasNoClosurePath replays the shape the sealed
// acquisition actually sent and reaches the same conclusion it took 103
// candidate evaluations to reach, without reading a single outcome.
//
// The historical evidence is untouched: this constructs the REQUEST shape, not
// the run, and asserts nothing about what the run produced.
func TestTheStageARequestShapeHasNoClosurePath(t *testing.T) {
	// The acquisition's own senders declared task_mode and nothing else --
	// pinned by the owned-sender tests and by tests/e2e/conftest.py.
	d := feasibilityFor(t, `{"task_mode":"work"}`, nil)
	expectFeasibility(t, "stage-a shape", d, false, FeasibilityUnspecifiedContract)

	// And the counterfactual the run would have needed: a structured Python
	// task whose closure floor its route could never reach. The behavioural
	// floor comes from a declared command, which has no producer here.
	structured := feasibilityFor(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
			`"verification_knowledge":"declared","verification":["python3 solve.py"]}`, nil)
	if structured.Feasible {
		t.Error("a behavioural floor with no producer was called feasible")
	}
	if structured.Floor != "behavioral" {
		t.Errorf("floor %q, want behavioral", structured.Floor)
	}

	// Quantify what would have been avoided, without changing anything: the
	// acquisition ran 21 tasks x 3 arms, and the sealed record's own count of
	// candidate evaluations under the route that could not close is the number
	// this answer would have skipped had it been consulted.
	const stageACandidateEvaluations = 103
	const underTheUnclosableRoute = 100
	if underTheUnclosableRoute >= stageACandidateEvaluations {
		t.Fatal("the pinned counts are inconsistent")
	}
	t.Logf("observe-only: %d of %d Stage-A candidate evaluations ran under a "+
		"route with no closure path; consulted, this decision would have "+
		"avoided them. Nothing was skipped.",
		underTheUnclosableRoute, stageACandidateEvaluations)
}

// --- inputs are closed -------------------------------------------------------

// TestFeasibilityReadsNoOutcome enumerates what the decision may not read.
func TestFeasibilityReadsNoOutcome(t *testing.T) {
	const file = "feasibility_decision.go"
	src, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	// The forbidden list itself lives in the file; strip its declaration so
	// naming a symbol in order to ban it does not count as reading it.
	if i := strings.Index(body, "var feasibilityForbiddenInputs"); i >= 0 {
		end := strings.Index(body[i:], "\n}")
		body = body[:i] + body[i+end:]
	}
	for _, banned := range feasibilityForbiddenInputs {
		if strings.Contains(body, banned) {
			t.Errorf("the feasibility owner reads %q", banned)
		}
	}
	if len(feasibilityForbiddenInputs) == 0 {
		t.Fatal("the forbidden-input list is empty, so this guard proves nothing")
	}
}

func TestFeasibilityCallsNothingThatGenerates(t *testing.T) {
	const file = "feasibility_decision.go"
	fset := token.NewFileSet()
	tree, err := parser.ParseFile(fset, file, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	banned := map[string]bool{
		"callV3Generate": true, "callV3GenerateStreaming": true,
		"writeFileWithV3": true, "improveContentWithV3": true,
		"authorizedV3Replacement": true, "v3DeliveryAuthorized": true,
		"writeFileRecorded": true, "finalizeCompletion": true,
		"WriteFile": true, "StreamFn": true,
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
			t.Errorf("the feasibility owner calls %s", name)
		}
		return true
	})
}

// TestGenerationProceedsWhateverFeasibilityConcludes pins the call shape at the
// production site: the answer is computed before generation and discarded, and
// no branch sits between it and the V3 call.
func TestGenerationProceedsWhateverFeasibilityConcludes(t *testing.T) {
	src, err := os.ReadFile("tools.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	call := strings.Index(body, "observeInvocationFeasibility(ctx)")
	if call < 0 {
		t.Fatal("feasibility is no longer computed before generation")
	}
	lineStart := strings.LastIndex(body[:call], "\n") + 1
	if prefix := strings.TrimSpace(body[lineStart:call]); prefix != "" {
		t.Errorf("the feasibility answer is captured: %q", prefix)
	}
	gen := strings.Index(body[call:], "callV3GenerateStreaming(")
	if gen < 0 {
		t.Fatal("generation no longer follows the feasibility question")
	}
	between := body[call : call+gen]
	for _, branch := range []string{"if ", "return ", "switch "} {
		if strings.Contains(between, branch) {
			t.Errorf("a %q sits between the feasibility answer and generation", branch)
		}
	}
}

func TestTheFeasibilityRecordCarriesNoContent(t *testing.T) {
	d := feasibilityFor(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
			`"verification_knowledge":"declared","verification":["pytest --token=hunter2"]}`, nil)
	blob, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	for _, needle := range []string{"hunter2", "pytest", "solve.py"} {
		if strings.Contains(string(blob), needle) {
			t.Errorf("the feasibility answer carries %q", needle)
		}
	}
}
