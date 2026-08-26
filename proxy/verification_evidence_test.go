package main

import (
	"context"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A command the client required, run exactly, against these exact bytes.
//
// Everything here is a way of getting less than that and being told so. No
// candidate is delivered: authority is computed and never consulted.

const declaredCmd = "python3 solve.py"

// verificationFixture is a workspace with one candidate on disk, a client that
// declared one command, and a green record of that command naming the file.
type verificationFixture struct {
	ctx  *AgentContext
	obl  taskObligation
	path string
	hash string
}

func newVerificationFixture(t *testing.T, commands ...string) *verificationFixture {
	t.Helper()
	if len(commands) == 0 {
		commands = []string{declaredCmd}
	}
	dir := t.TempDir()
	const code = "print(7)\n"
	if err := os.WriteFile(filepath.Join(dir, "solve.py"), []byte(code), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.Ctx = context.WithValue(context.Background(), requestIDKey, "req-fixture")

	quoted := make([]string, 0, len(commands))
	for _, c := range commands {
		b, _ := json.Marshal(c)
		quoted = append(quoted, string(b))
	}
	ctx.TaskContract = mustContract(t, dir,
		`{"task_mode":"work","verification_knowledge":"declared","verification":[`+
			strings.Join(quoted, ",")+`]}`)

	resolved := resolveAgentPath(ctx, "solve.py")
	hash := fileSHA256(ctx, resolved)
	if hash == "" {
		t.Fatal("no hash for the candidate on disk")
	}
	obl, ok := newTaskObligation(ObligationDeclaredCommand, commands[0], "", true)
	if !ok {
		t.Fatal("declared command obligation refused")
	}
	return &verificationFixture{ctx: ctx, obl: obl, path: resolved, hash: hash}
}

func (f *verificationFixture) request() verificationEvidenceRequest {
	return verificationEvidenceRequest{
		Obligation: f.obl,
		Record: VerificationRecord{
			Command: f.obl.Subject,
			Covered: map[string]string{f.path: f.hash},
			Turn:    1,
		},
		Outcome:             commandExitedZero,
		CandidatePath:       f.path,
		CandidateHash:       f.hash,
		InvocationID:        "inv-1",
		CandidateInstanceID: "cand-1",
	}
}

// --- the positive case -------------------------------------------------------

func TestAnExactDeclaredCommandAgainstExactCandidateBytesIsBehavioral(t *testing.T) {
	f := newVerificationFixture(t)
	ev, ok := produceDeclaredVerificationEvidence(f.ctx, f.request())
	if !ok {
		t.Fatal("an exact declared command against exact bytes produced nothing")
	}
	if ev.Outcome != ValidationPassed {
		t.Errorf("outcome %q, want passed", ev.Outcome)
	}
	p := ev.Provenance
	if p.Source != ProvenanceClientDeclaredVerification {
		t.Errorf("source %q, want client_declared_verification", p.Source)
	}
	if p.ObservedStrength != "behavioral" || p.RequiredStrength != "behavioral" {
		t.Errorf("strengths %q/%q, want behavioral/behavioral",
			p.RequiredStrength, p.ObservedStrength)
	}
	if p.CandidateHash != f.hash {
		t.Error("evidence does not name the exact candidate bytes")
	}
	if p.CommandIdentity != contentSHA256(declaredCmd) {
		t.Error("evidence does not name its command")
	}
	if p.WorkspaceStateHash == "" || p.RequestID != "req-fixture" {
		t.Errorf("identity incomplete: %+v", p)
	}
	if authorized, why := ev.Authorizes(); !authorized {
		t.Errorf("a green declared command did not authorize its obligation: %s", why)
	}
}

// TestADeclaredCommandIsNeverLabelledAnOracle pins the strength ceiling: exit
// zero says the command ran and succeeded, not that an answer was checked
// against a reference.
func TestADeclaredCommandIsNeverLabelledAnOracle(t *testing.T) {
	f := newVerificationFixture(t)
	ev, ok := produceDeclaredVerificationEvidence(f.ctx, f.request())
	if !ok {
		t.Fatal("no evidence to test")
	}
	if ev.Provenance.ObservedStrength == "oracle" {
		t.Error("an arbitrary exit-zero command was labelled oracle evidence")
	}
	oracleObligation, _ := newTaskObligation(ObligationDeclaredExample, "case-1", "", true)
	forged := ev
	forged.Provenance.ObligationID = oracleObligation.ID
	forged.Provenance.RequiredStrength = "oracle"
	if authorized, _ := forged.Authorizes(); authorized {
		t.Error("behavioral evidence closed an oracle obligation")
	}
}

// --- provenance comes from the declaration, not from the text ----------------

func TestTheSameTextFromTheModelRemainsUntrusted(t *testing.T) {
	// The client declared nothing. The model ran a command that looks exactly
	// like verification and it passed against the exact bytes.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "solve.py"), []byte("print(7)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.Ctx = context.WithValue(context.Background(), requestIDKey, "req-fixture")
	ctx.TaskContract = mustContract(t, dir, `{"task_mode":"work"}`)

	resolved := resolveAgentPath(ctx, "solve.py")
	hash := fileSHA256(ctx, resolved)
	obl, _ := newTaskObligation(ObligationDeclaredCommand, declaredCmd, "", true)
	req := verificationEvidenceRequest{
		Obligation: obl,
		Record: VerificationRecord{Command: declaredCmd,
			Covered: map[string]string{resolved: hash}, Turn: 1},
		Outcome: commandExitedZero, CandidatePath: resolved, CandidateHash: hash,
		InvocationID: "inv-1", CandidateInstanceID: "cand-1",
	}
	if _, ok := produceDeclaredVerificationEvidence(ctx, req); ok {
		t.Error("a command the client never declared produced trusted evidence")
	}
}

func TestUnspecifiedVerificationKnowledgeDeclaresNothing(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "solve.py"), []byte("print(7)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.Ctx = context.WithValue(context.Background(), requestIDKey, "req-fixture")
	ctx.TaskContract = mustContract(t, dir,
		`{"task_mode":"work","verification_knowledge":"unspecified"}`)
	if requestDeclaredCommand(ctx, declaredCmd) {
		t.Error("an unspecified caller was read as declaring a command")
	}
}

func TestASimilarButNotIdenticalCommandFailsBinding(t *testing.T) {
	for _, ran := range []string{
		"python3  solve.py",   // an extra space
		"python3 solve.py ",   // a trailing space
		"python3 ./solve.py",  // a different spelling of the path
		"python3 solve.py -v", // an extra flag
		"python solve.py",     // a different interpreter
	} {
		f := newVerificationFixture(t)
		req := f.request()
		req.Record.Command = ran
		if _, ok := produceDeclaredVerificationEvidence(f.ctx, req); ok {
			t.Errorf("%q was accepted for the declared %q", ran, declaredCmd)
		}
	}
}

// --- what the run was actually about -----------------------------------------

func TestASuccessfulUnrelatedCommandProvesNothing(t *testing.T) {
	f := newVerificationFixture(t)
	t.Run("named nothing", func(t *testing.T) {
		req := f.request()
		req.Record.Covered = map[string]string{}
		if _, ok := produceDeclaredVerificationEvidence(f.ctx, req); ok {
			t.Error("a run that covered nothing authorized a candidate")
		}
	})
	t.Run("named another file", func(t *testing.T) {
		other := filepath.Join(f.ctx.WorkingDir, "other.py")
		if err := os.WriteFile(other, []byte("print(1)\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		resolved := resolveAgentPath(f.ctx, other)
		req := f.request()
		req.Record.Covered = map[string]string{resolved: fileSHA256(f.ctx, resolved)}
		if _, ok := produceDeclaredVerificationEvidence(f.ctx, req); ok {
			t.Error("a run about another artifact authorized this candidate")
		}
	})
}

func TestACommandThatPassedBeforeInsertionCannotAuthorizeTheCandidate(t *testing.T) {
	f := newVerificationFixture(t)
	// The command passed against the file as it was, then the candidate
	// replaced it. The record still names the older bytes.
	oldHash := f.hash
	if err := os.WriteFile(f.path, []byte("print(8)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	newHash := fileSHA256(f.ctx, f.path)
	if newHash == oldHash {
		t.Fatal("the candidate did not change")
	}
	req := f.request()
	req.Record.Covered = map[string]string{f.path: oldHash}
	req.CandidateHash = newHash
	if _, ok := produceDeclaredVerificationEvidence(f.ctx, req); ok {
		t.Error("a run from before the candidate landed authorized it")
	}
}

func TestAWorkspaceMutationAfterExecutionInvalidatesEvidence(t *testing.T) {
	f := newVerificationFixture(t)
	req := f.request()
	if _, ok := produceDeclaredVerificationEvidence(f.ctx, req); !ok {
		t.Fatal("the fixture does not produce evidence to invalidate")
	}
	// Someone edits the file after the command passed.
	if err := os.WriteFile(f.path, []byte("print(9)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, ok := produceDeclaredVerificationEvidence(f.ctx, req); ok {
		t.Error("evidence survived a mutation of the bytes it vouched for")
	}
}

func TestACandidateHashMismatchInvalidatesEvidence(t *testing.T) {
	f := newVerificationFixture(t)
	req := f.request()
	req.CandidateHash = contentSHA256("some other candidate")
	if _, ok := produceDeclaredVerificationEvidence(f.ctx, req); ok {
		t.Error("evidence was produced about a candidate the run never covered")
	}
}

func TestABaselineMismatchInvalidatesEvidence(t *testing.T) {
	f := newVerificationFixture(t)
	req := f.request()
	req.BaselineIdentity = "behavioral:baseline-a"
	ev, ok := produceDeclaredVerificationEvidence(f.ctx, req)
	if !ok {
		t.Fatal("no evidence to test")
	}
	asked := ev.Provenance
	asked.BaselineIdentity = "behavioral:baseline-b"
	if bound, _ := ev.Provenance.BindsTo(asked); bound {
		t.Error("evidence earned against one baseline bound to another")
	}
}

// --- every way a command can end without authorizing -------------------------

func TestOnlyACleanForegroundSuccessAuthorizes(t *testing.T) {
	for _, outcome := range []declaredCommandOutcome{
		commandExitedNonZero, commandRefused, commandAltered,
		commandTimedOut, commandCancelled, commandBackgrounded,
		commandOutcomeUnknown,
	} {
		f := newVerificationFixture(t)
		req := f.request()
		req.Outcome = outcome
		ev, ok := produceDeclaredVerificationEvidence(f.ctx, req)
		if !ok {
			continue // declining outright is also a correct answer
		}
		if authorized, _ := ev.Authorizes(); authorized {
			t.Errorf("%s authorized its obligation", outcome)
		}
		if ev.Outcome == ValidationPassed {
			t.Errorf("%s was recorded as a pass", outcome)
		}
	}
}

func TestAnUnknownOutcomeValueFailsClosed(t *testing.T) {
	f := newVerificationFixture(t)
	req := f.request()
	req.Outcome = declaredCommandOutcome("something_new")
	if _, ok := produceDeclaredVerificationEvidence(f.ctx, req); ok {
		t.Error("an unrecognised outcome produced evidence")
	}
}

func TestACancelledRunProducesNothing(t *testing.T) {
	f := newVerificationFixture(t)
	cancelled, cancel := context.WithCancel(
		context.WithValue(context.Background(), requestIDKey, "req-fixture"))
	cancel()
	f.ctx.Ctx = cancelled
	if _, ok := produceDeclaredVerificationEvidence(f.ctx, f.request()); ok {
		t.Error("a cancelled run produced evidence")
	}
}

func TestAMissingIdentityProducesNothing(t *testing.T) {
	for _, mut := range []func(*verificationEvidenceRequest){
		func(r *verificationEvidenceRequest) { r.InvocationID = "" },
		func(r *verificationEvidenceRequest) { r.CandidateInstanceID = "" },
		func(r *verificationEvidenceRequest) { r.CandidatePath = "" },
		func(r *verificationEvidenceRequest) { r.CandidateHash = "" },
	} {
		f := newVerificationFixture(t)
		req := f.request()
		mut(&req)
		if _, ok := produceDeclaredVerificationEvidence(f.ctx, req); ok {
			t.Error("evidence was produced with an incomplete identity")
		}
	}
}

func TestTheProducerRefusesAnyOtherObligationKind(t *testing.T) {
	f := newVerificationFixture(t)
	for _, kind := range []string{
		ObligationSyntacticValidity, ObligationArtifactExists,
		ObligationDeclaredExample, ObligationBaselinePreserved,
		ObligationUnsupported,
	} {
		baseline := ""
		if kind == ObligationBaselinePreserved {
			baseline = "behavioral"
		}
		o, ok := newTaskObligation(kind, declaredCmd, baseline, true)
		if !ok {
			continue
		}
		req := f.request()
		req.Obligation = o
		if _, ok := produceDeclaredVerificationEvidence(f.ctx, req); ok {
			t.Errorf("a declared command described a %s obligation", kind)
		}
	}
}

// --- obligations stay independent --------------------------------------------

func TestOneOfTwoCommandsPassingDoesNotCompleteBoth(t *testing.T) {
	f := newVerificationFixture(t, declaredCmd, "ruff check .")
	second, ok := newTaskObligation(ObligationDeclaredCommand, "ruff check .", "", true)
	if !ok {
		t.Fatal("second obligation refused")
	}
	ev, ok := produceDeclaredVerificationEvidence(f.ctx, f.request())
	if !ok {
		t.Fatal("the first command produced no evidence")
	}
	met, missing := declaredVerificationCoverage(
		[]taskObligation{f.obl, second}, []proxyEvidence{ev})
	if len(met) != 1 || met[0] != f.obl.ID {
		t.Errorf("met %v, want only the command that ran", met)
	}
	if len(missing) != 1 || missing[0] != second.ID {
		t.Errorf("missing %v, want the command that did not run", missing)
	}
}

func TestAllRequiredCommandsMustBeRepresented(t *testing.T) {
	f := newVerificationFixture(t, declaredCmd, "ruff check .")
	second, _ := newTaskObligation(ObligationDeclaredCommand, "ruff check .", "", true)

	secondReq := f.request()
	secondReq.Obligation = second
	secondReq.Record.Command = "ruff check ."
	secondEv, ok := produceDeclaredVerificationEvidence(f.ctx, secondReq)
	if !ok {
		t.Fatal("the second command produced no evidence")
	}
	firstEv, _ := produceDeclaredVerificationEvidence(f.ctx, f.request())

	met, missing := declaredVerificationCoverage(
		[]taskObligation{f.obl, second}, []proxyEvidence{firstEv, secondEv})
	if len(met) != 2 || len(missing) != 0 {
		t.Errorf("met %v missing %v, want both met", met, missing)
	}
}

func TestAFailedObservationDoesNotCoverItsObligation(t *testing.T) {
	f := newVerificationFixture(t)
	req := f.request()
	req.Outcome = commandExitedNonZero
	ev, ok := produceDeclaredVerificationEvidence(f.ctx, req)
	if !ok {
		t.Skip("the producer declined outright, which is also correct")
	}
	_, missing := declaredVerificationCoverage([]taskObligation{f.obl}, []proxyEvidence{ev})
	if len(missing) != 1 {
		t.Error("a failed run covered its obligation")
	}
}

// --- evidence cannot cross a request or an invocation ------------------------

func TestConcurrentRequestsAndInvocationsCannotExchangeEvidence(t *testing.T) {
	a := newVerificationFixture(t)
	b := newVerificationFixture(t)
	b.ctx.Ctx = context.WithValue(context.Background(), requestIDKey, "req-other")

	evA, ok := produceDeclaredVerificationEvidence(a.ctx, a.request())
	if !ok {
		t.Fatal("fixture A produced no evidence")
	}
	reqB := b.request()
	reqB.InvocationID = "inv-2"
	reqB.CandidateInstanceID = "cand-2"
	evB, ok := produceDeclaredVerificationEvidence(b.ctx, reqB)
	if !ok {
		t.Fatal("fixture B produced no evidence")
	}
	if bound, _ := evA.Provenance.BindsTo(evB.Provenance); bound {
		t.Error("one request's evidence bound to another's")
	}
	if bound, _ := evB.Provenance.BindsTo(evA.Provenance); bound {
		t.Error("one invocation's evidence bound to another's")
	}
}

// --- no leaks, no execution, no hidden evaluator -----------------------------

func TestNeitherCommandStringsNorSourceBytesTravel(t *testing.T) {
	const secret = "pytest --token=hunter2 -q"
	f := newVerificationFixture(t, secret)
	req := f.request()
	req.Record.Command = secret
	ev, ok := produceDeclaredVerificationEvidence(f.ctx, req)
	if !ok {
		t.Fatal("no evidence to test")
	}
	blob, err := json.Marshal(ev.Provenance)
	if err != nil {
		t.Fatal(err)
	}
	for _, needle := range []string{secret, "hunter2", "print(7)"} {
		if strings.Contains(string(blob), needle) {
			t.Errorf("the serialised binding carries %q", needle)
		}
	}
}

// TestTheProducerExecutesNothing pins that running a command stays the tool
// path's job, with its own safety checks, permission endpoint and sandbox. A
// producer that executed would be a second route with a second set of rules.
func TestTheProducerExecutesNothing(t *testing.T) {
	src, err := os.ReadFile("verification_evidence.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "verification_evidence.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	banned := map[string]bool{
		"executeCommand": true, "runCommand": true, "runInSandbox": true,
		"exec": true, "Command": true, "CommandContext": true, "Start": true,
		"awaitPermission": true, "requestPermission": true,
		"deleteFile": true, "removeAll": true, "RemoveAll": true, "Remove": true,
	}
	ast.Inspect(file, func(n ast.Node) bool {
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
			t.Errorf("the producer calls %s: execution, permission and deletion "+
				"belong to the tool path", name)
		}
		return true
	})
}

// TestNoHiddenEvaluatorIsReachableFromTheProducers pins that the benchmark's
// own grader has no representation any producer could name.
func TestNoHiddenEvaluatorIsReachableFromTheProducers(t *testing.T) {
	for _, path := range []string{
		"verification_evidence.go", "syntax_evidence.go", "obligation_kinds.go",
	} {
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		low := strings.ToLower(string(body))
		for _, banned := range []string{
			"holdout", "hidden_evaluator", "benchmark_grader", "reference_answer",
			"expected_output", "goldenanswer",
		} {
			if strings.Contains(low, banned) {
				t.Errorf("%s names %q", path, banned)
			}
		}
	}
}

// TestTheOnlyProductionWriterOfVerificationEvidenceIsTheGreenBranch pins where
// the consumed records come from: one append, in the branch that already
// requires a successful foreground run.
func TestTheOnlyProductionWriterOfVerificationEvidenceIsTheGreenBranch(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	writers := map[string]int{}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		body, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		writers[name] = strings.Count(string(body), "ctx.VerificationEvidence = append")
	}
	total := 0
	for name, n := range writers {
		total += n
		if n > 0 && name != "agent.go" {
			t.Errorf("%s writes verification evidence; only the agent loop's green "+
				"branch may", name)
		}
	}
	if total != 1 {
		t.Errorf("%d production writers of verification evidence, want exactly 1", total)
	}
}
