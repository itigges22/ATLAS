package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// A grant is the difference between "these bytes were authorized" and "these
// bytes may land now".
//
// Everything here is a way of holding something that looks like permission and
// being told it is not: the same grant twice, a grant for another candidate, a
// grant whose workspace moved underneath it, a grant the request already
// cancelled. Nothing is delivered in this file -- minting and spending are the
// whole subject, and the delivery owner is pinned separately.

// grantWorld is an authorized candidate and everything the decision was
// reached over, kept so a test can perturb exactly one thing.
type grantWorld struct {
	*authWorld
	in       authorizationInput
	decision AuthorizationDecision
}

func newGrantWorld(t *testing.T, commands ...string) *grantWorld {
	t.Helper()
	var w *authWorld
	if len(commands) == 0 {
		w = newAuthWorld(t,
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`,
			"solve.py", authPy, true)
	} else {
		w = stagedWorld(t, commands...)
	}
	ev, evID := w.mustObserve(t)
	evidence := append([]proxyEvidence{ev}, w.stage(evID)...)

	generation, stateHash := workspaceIdentity(w.ctx)
	in := authorizationInput{
		Obligations:   requestObligations(w.ctx),
		TargetPath:    w.path,
		CandidateHash: contentSHA256(w.code),
		Identity: V3EvidenceProvenance{
			RequestID:           requestIDOf(w.ctx),
			InvocationID:        evID.InvocationID,
			CandidateInstanceID: evID.CandidateInstanceID,
			CandidateHash:       contentSHA256(w.code),
			WorkspaceGeneration: generation,
			WorkspaceStateHash:  stateHash,
			BaselineIdentity:    baselineIdentityFor(w.ctx, w.path),
		},
		Evidence:               evidence,
		BaselineWitnessCommand: grantWitness(w.ctx, w.path),
	}
	d := decideAuthorization(w.ctx, in)
	return &grantWorld{authWorld: w, in: in, decision: d}
}

func grantWitness(ctx *AgentContext, path string) string {
	_, witness := baselineWitness(ctx, path)
	return witness
}

// claim builds what a consumer presents, from the live state -- the same way
// the delivery owner has to.
func (w *grantWorld) claim() grantClaim {
	generation, stateHash := workspaceIdentity(w.ctx)
	return grantClaim{
		RequestID:           requestIDOf(w.ctx),
		InvocationID:        w.in.Identity.InvocationID,
		CandidateInstanceID: w.in.Identity.CandidateInstanceID,
		CandidateHash:       w.in.CandidateHash,
		TargetPath:          w.path,
		WorkspaceGeneration: generation,
		WorkspaceStateHash:  stateHash,
		BaselineIdentity:    baselineIdentityFor(w.ctx, w.path),
		BaselineHash:        fileSHA256(w.ctx, w.path),
		ObligationSetID:     obligationSetIdentity(w.in.Obligations),
		EvidenceSetID:       evidenceSetIdentity(w.in.Evidence),
	}
}

func (w *grantWorld) mint(t *testing.T) *authorizationGrant {
	t.Helper()
	g, ok, why := mintAuthorizationGrant(w.ctx, w.in, w.decision, "selected-1")
	if !ok {
		t.Fatalf("an authorized decision minted no grant: %s", why)
	}
	return g
}

// --- minting ---------------------------------------------------------------------

func TestAnAuthorizedDecisionMintsABoundGrant(t *testing.T) {
	w := newGrantWorld(t)
	if !w.decision.Authorized {
		t.Fatalf("the fixture is not authorized: %s", w.decision.Reason)
	}
	g := w.mint(t)

	for _, c := range []struct{ name, got, want string }{
		{"request_id", g.RequestID, requestIDOf(w.ctx)},
		{"invocation_id", g.InvocationID, w.in.Identity.InvocationID},
		{"candidate_instance_id", g.CandidateInstanceID, w.in.Identity.CandidateInstanceID},
		{"candidate_hash", g.CandidateHash, w.in.CandidateHash},
		{"target", g.TargetPath, w.path},
		{"workspace_state_hash", g.WorkspaceStateHash, w.in.Identity.WorkspaceStateHash},
		{"baseline_identity", g.BaselineIdentity, w.in.Identity.BaselineIdentity},
		{"obligation_set", g.ObligationSetID, obligationSetIdentity(w.in.Obligations)},
		{"evidence_set", g.EvidenceSetID, evidenceSetIdentity(w.in.Evidence)},
		{"selected_candidate", g.SelectedCandidateID, "selected-1"},
	} {
		if c.got != c.want {
			t.Errorf("%s is %q, want %q", c.name, c.got, c.want)
		}
	}
	if g.WorkspaceGeneration != w.in.Identity.WorkspaceGeneration {
		t.Error("the grant does not bind the workspace generation")
	}
	if g.DecisionGeneration == 0 {
		t.Error("the grant has no decision generation")
	}
	if liveGrantCount(w.ctx) != 1 {
		t.Errorf("%d live grants, want the one just minted", liveGrantCount(w.ctx))
	}
}

func TestAnUnauthorizedDecisionMintsNothing(t *testing.T) {
	w := newGrantWorld(t)
	for name, mut := range map[string]func(*grantWorld){
		"refused outright": func(w *grantWorld) {
			w.decision.Authorized, w.decision.Reason = false, ReasonEvidenceMissing
		},
		"authorized with a reason that is not": func(w *grantWorld) {
			w.decision.Reason = ReasonWorkspaceStale
		},
		"something still owed": func(w *grantWorld) {
			w.decision.Missing = []string{"declared_command:owed"}
		},
		"nothing demonstrated": func(w *grantWorld) { w.decision.Satisfied = nil },
		"no contract":          func(w *grantWorld) { w.ctx.TaskContract = nil },
		"no obligations":       func(w *grantWorld) { w.in.Obligations = nil },
		"undeclared target": func(w *grantWorld) {
			w.in.TargetPath = filepath.Join(w.ctx.WorkingDir, "not-declared.py")
		},
		"identity and candidate disagree": func(w *grantWorld) {
			w.in.Identity.CandidateHash = contentSHA256("other")
		},
		"another request's identity": func(w *grantWorld) {
			w.in.Identity.RequestID = "req-somebody-else"
		},
		"no invocation":         func(w *grantWorld) { w.in.Identity.InvocationID = "" },
		"no candidate instance": func(w *grantWorld) { w.in.Identity.CandidateInstanceID = "" },
		"no workspace state":    func(w *grantWorld) { w.in.Identity.WorkspaceStateHash = "" },
	} {
		fresh := newGrantWorld(t)
		mut(fresh)
		if _, ok, _ := mintAuthorizationGrant(fresh.ctx, fresh.in, fresh.decision, "s"); ok {
			t.Errorf("%s minted a grant", name)
		}
	}
	_ = w
}

func TestADeclaredCommandWithoutEvidenceMintsNothing(t *testing.T) {
	w := newGrantWorld(t, "pytest -q")
	if !w.decision.Authorized {
		t.Fatalf("the staged fixture is not authorized: %s", w.decision.Reason)
	}
	// Drop the behavioral record but leave the decision saying yes. Minting
	// re-checks coverage rather than taking the decision's word.
	stripped := w.in
	var kept []proxyEvidence
	for _, e := range w.in.Evidence {
		if e.Provenance.Source != ProvenanceClientDeclaredVerification {
			kept = append(kept, e)
		}
	}
	stripped.Evidence = kept
	if _, ok, _ := mintAuthorizationGrant(w.ctx, stripped, w.decision, "s"); ok {
		t.Error("a declared command with no trusted evidence minted a grant")
	}
}

func TestAliasSpellingsShareOneGrant(t *testing.T) {
	w := newGrantWorld(t)
	g := w.mint(t)
	rel, err := filepath.Rel(w.ctx.WorkingDir, w.path)
	if err != nil {
		t.Fatal(err)
	}
	for _, alias := range []string{rel, "./" + rel, w.path, filepath.Join(w.ctx.WorkingDir, ".", rel)} {
		c := w.claim()
		c.TargetPath = alias
		key := grantKey(c.RequestID, c.InvocationID, c.CandidateInstanceID,
			resolveAgentPath(w.ctx, alias))
		if key != g.ID {
			t.Errorf("%q names a different grant", alias)
		}
	}
	// And minting again under an alias supersedes rather than adding.
	aliased := w.in
	aliased.TargetPath = rel
	if _, ok, why := mintAuthorizationGrant(w.ctx, aliased, w.decision, "s"); !ok {
		t.Fatalf("an alias could not mint: %s", why)
	}
	if liveGrantCount(w.ctx) != 1 {
		t.Errorf("%d live grants for one delivery", liveGrantCount(w.ctx))
	}
}

func TestALaterDecisionSupersedesTheEarlierGrant(t *testing.T) {
	w := newGrantWorld(t)
	first := w.mint(t)
	// A second candidate for the same target.
	later := w.in
	later.Identity.CandidateInstanceID = w.in.Identity.CandidateInstanceID + ":again"
	second, ok, why := mintAuthorizationGrant(w.ctx, later, w.decision, "selected-2")
	if !ok {
		t.Fatalf("the later decision minted nothing: %s", why)
	}
	if second.DecisionGeneration <= first.DecisionGeneration {
		t.Error("the later grant does not order after the earlier one")
	}
	if liveGrantCount(w.ctx) != 1 {
		t.Errorf("%d live grants for one target", liveGrantCount(w.ctx))
	}
	// The superseded one cannot be spent.
	if _, why := consumeAuthorizationGrant(w.ctx, w.claim()); why == "" {
		t.Error("a superseded grant was spent")
	}
}

func TestGrantCapacityRefusesBeforeMutating(t *testing.T) {
	w := newGrantWorld(t)
	for i := 0; i < grantCapacity; i++ {
		in := w.in
		in.Identity.CandidateInstanceID = w.in.Identity.CandidateInstanceID + ":" + string(rune('a'+i%26)) + string(rune('a'+i/26))
		in.Identity.InvocationID = w.in.Identity.InvocationID + ":" + string(rune('a'+i%26)) + string(rune('a'+i/26))
		if _, ok, why := mintAuthorizationGrant(w.ctx, in, w.decision, "s"); !ok {
			t.Fatalf("grant %d refused early: %s", i, why)
		}
	}
	before := liveGrantCount(w.ctx)
	over := w.in
	over.Identity.CandidateInstanceID = "one-too-many"
	over.Identity.InvocationID = "inv-one-too-many"
	_, ok, why := mintAuthorizationGrant(w.ctx, over, w.decision, "s")
	if ok {
		t.Error("capacity did not refuse")
	}
	if !strings.Contains(why, "too many") {
		t.Errorf("refusal %q does not name capacity", why)
	}
	if liveGrantCount(w.ctx) != before {
		t.Error("the refused mint changed the live set")
	}
}

// --- spending --------------------------------------------------------------------

func TestAGrantIsSpentExactlyOnce(t *testing.T) {
	w := newGrantWorld(t)
	g := w.mint(t)
	spent, why := consumeAuthorizationGrant(w.ctx, w.claim())
	if why != "" {
		t.Fatalf("a live grant would not spend: %s", why)
	}
	if spent.ID != g.ID || spent.CandidateHash != g.CandidateHash {
		t.Error("consumption returned a different grant")
	}
	again, why := consumeAuthorizationGrant(w.ctx, w.claim())
	if why == "" || again != nil {
		t.Fatal("a grant was spent twice")
	}
	if why != string(grantConsumed) {
		t.Errorf("replay refusal %q, want %q", why, grantConsumed)
	}
	if liveGrantCount(w.ctx) != 0 {
		t.Error("a spent grant is still live")
	}
}

func TestConcurrentConsumersGetExactlyOneSuccess(t *testing.T) {
	w := newGrantWorld(t)
	w.mint(t)
	const racers = 16
	var wg sync.WaitGroup
	results := make([]string, racers)
	start := make(chan struct{})
	// Built once, before the racers start. Building it inside each goroutine
	// takes the ledger lock and serialises them, which would leave the
	// consumer's own atomicity untested.
	claim := w.claim()
	for i := 0; i < racers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			_, why := consumeAuthorizationGrant(w.ctx, claim)
			results[i] = why
		}(i)
	}
	close(start)
	wg.Wait()

	wins := 0
	for _, why := range results {
		if why == "" {
			wins++
		}
	}
	if wins != 1 {
		t.Errorf("%d of %d consumers succeeded, want exactly one", wins, racers)
	}
}

func TestNoOtherIdentityCanBorrowAGrant(t *testing.T) {
	for name, mut := range map[string]func(*grantClaim){
		"another request":      func(c *grantClaim) { c.RequestID = "req-other" },
		"another invocation":   func(c *grantClaim) { c.InvocationID = "inv-other" },
		"another candidate":    func(c *grantClaim) { c.CandidateInstanceID = "cand-other" },
		"other bytes":          func(c *grantClaim) { c.CandidateHash = contentSHA256("other") },
		"another workspace":    func(c *grantClaim) { c.WorkspaceStateHash = contentSHA256("moved") },
		"a later generation":   func(c *grantClaim) { c.WorkspaceGeneration += 1 },
		"another baseline":     func(c *grantClaim) { c.BaselineIdentity = "syntax:elsewhere" },
		"other baseline bytes": func(c *grantClaim) { c.BaselineHash = contentSHA256("was something else") },
		"another obligation set": func(c *grantClaim) {
			c.ObligationSetID = contentSHA256("a different task")
		},
		"another evidence set": func(c *grantClaim) {
			c.EvidenceSetID = contentSHA256("different records")
		},
	} {
		w := newGrantWorld(t)
		w.mint(t)
		claim := w.claim()
		mut(&claim)
		if _, why := consumeAuthorizationGrant(w.ctx, claim); why == "" {
			t.Errorf("%s spent a grant that was not about it", name)
		}
		// And the grant is untouched: a refused claim must not spend it.
		if liveGrantCount(w.ctx) != 1 {
			t.Errorf("%s consumed the grant it was refused", name)
		}
	}
}

func TestAnotherPathCannotBorrowAGrant(t *testing.T) {
	w := newGrantWorld(t)
	w.mint(t)
	other := filepath.Join(w.ctx.WorkingDir, "other.py")
	if err := os.WriteFile(other, []byte("X = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	claim := w.claim()
	claim.TargetPath = other
	if _, why := consumeAuthorizationGrant(w.ctx, claim); why == "" {
		t.Error("a grant for one path delivered another")
	}
	if liveGrantCount(w.ctx) != 1 {
		t.Error("the refused claim spent the grant")
	}
}

func TestAStaleWorkspaceCannotSpendAGrant(t *testing.T) {
	w := newGrantWorld(t)
	w.mint(t)
	// The workspace moves after the decision.
	bumpWorkspace(w.ctx, w.path, contentSHA256("something else happened\n"))
	if _, why := consumeAuthorizationGrant(w.ctx, w.claim()); why == "" {
		t.Error("a grant survived the workspace moving underneath it")
	}
	if liveGrantCount(w.ctx) != 1 {
		t.Error("a refused stale claim spent the grant")
	}
}

func TestAChangedBaselineCannotSpendAGrant(t *testing.T) {
	w := newGrantWorld(t)
	w.mint(t)
	// Somebody rewrote the file the candidate would replace.
	if err := os.WriteFile(w.path, []byte("REPLACED = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, why := consumeAuthorizationGrant(w.ctx, w.claim()); why == "" {
		t.Error("a grant survived its baseline being rewritten")
	}
}

// --- retirement ------------------------------------------------------------------

func TestCancellationRetiresEveryGrant(t *testing.T) {
	w := newGrantWorld(t)
	w.mint(t)
	cancelled, cancel := context.WithCancel(
		context.WithValue(context.Background(), requestIDKey, requestIDOf(w.ctx)))
	w.ctx.Ctx = cancelled
	cancel()
	if _, why := consumeAuthorizationGrant(w.ctx, w.claim()); why != string(grantCancelled) {
		t.Errorf("refusal %q, want %q", why, grantCancelled)
	}
	if liveGrantCount(w.ctx) != 0 {
		t.Error("a cancelled request still holds a live grant")
	}
}

func TestRetirementIsFinalAndBlocksFurtherMinting(t *testing.T) {
	for _, reason := range []grantRetirement{
		grantTerminal, grantSessionEnd, grantCancelled,
	} {
		w := newGrantWorld(t)
		w.mint(t)
		if n := retireAuthorizationGrants(w.ctx, reason); n != 1 {
			t.Errorf("%s retired %d grants, want 1", reason, n)
		}
		if _, why := consumeAuthorizationGrant(w.ctx, w.claim()); why != string(reason) {
			t.Errorf("refusal %q, want %q", why, reason)
		}
		// And nothing new may be minted afterwards.
		if _, ok, _ := mintAuthorizationGrant(w.ctx, w.in, w.decision, "s"); ok {
			t.Errorf("%s: a grant was minted after retirement", reason)
		}
	}
}

func TestUnrelatedSuccessNeitherRetiresNorRefreshesAGrant(t *testing.T) {
	w := newGrantWorld(t)
	g := w.mint(t)
	// Another file is written successfully. Nothing about this delivery.
	other := filepath.Join(w.ctx.WorkingDir, "notes.md")
	if err := os.WriteFile(other, []byte("# notes\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	observeDeliverable(w.ctx, other, []byte("# notes\n"),
		ValidationKindNone, ValidationNotApplicable, "")

	w.ctx.grantMu.Lock()
	still := w.ctx.grants[g.ID]
	w.ctx.grantMu.Unlock()
	if still == nil {
		t.Fatal("an unrelated success deleted the grant")
	}
	// Not retired: an unrelated success is not a cancellation, and a
	// consumer must not be told the request ended because something else
	// went well.
	if still.retired != grantLive {
		t.Errorf("an unrelated success retired the grant as %q", still.retired)
	}
	// Not refreshed either. Nothing about it was re-derived, so it still
	// describes the moment it was minted -- which is what makes the
	// workspace check below a real one rather than a tautology.
	if still.DecisionGeneration != g.DecisionGeneration ||
		still.WorkspaceStateHash != g.WorkspaceStateHash ||
		still.CandidateHash != g.CandidateHash {
		t.Error("an unrelated success refreshed the grant")
	}
	// And the workspace it was about has genuinely moved on, so it can no
	// longer be spent. That is the conservative answer and the honest one:
	// the grant froze a moment, and this is no longer that moment.
	if _, why := consumeAuthorizationGrant(w.ctx, w.claim()); why == "" {
		t.Error("a grant was spent against a workspace it never saw")
	}
}

// TestAGrantIsOneTimeAcrossAnUnrelatedSuccess pins the part that matters most:
// an unrelated success cannot buy a second consumption of a grant that was
// already spent.
func TestAGrantIsOneTimeAcrossAnUnrelatedSuccess(t *testing.T) {
	w := newGrantWorld(t)
	w.mint(t)
	if _, why := consumeAuthorizationGrant(w.ctx, w.claim()); why != "" {
		t.Fatalf("the grant would not spend: %s", why)
	}
	other := filepath.Join(w.ctx.WorkingDir, "notes.md")
	if err := os.WriteFile(other, []byte("# notes\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	observeDeliverable(w.ctx, other, []byte("# notes\n"),
		ValidationKindNone, ValidationNotApplicable, "")
	if _, why := consumeAuthorizationGrant(w.ctx, w.claim()); why == "" {
		t.Error("an unrelated success bought a second consumption")
	}
}

// --- what a grant may not carry --------------------------------------------------

func TestAGrantCarriesNoContent(t *testing.T) {
	const secret = "TOKEN = 'hunter2'\nprint(7)\n"
	w := newAuthWorld(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
			`"verification_knowledge":"declared","verification":["pytest --token=hunter2"]}`,
		"solve.py", secret, true)
	ev, evID := w.mustObserve(t)
	generation, stateHash := workspaceIdentity(w.ctx)
	in := authorizationInput{
		Obligations: requestObligations(w.ctx), TargetPath: w.path,
		CandidateHash: contentSHA256(secret),
		Identity: V3EvidenceProvenance{
			RequestID: requestIDOf(w.ctx), InvocationID: evID.InvocationID,
			CandidateInstanceID: evID.CandidateInstanceID,
			CandidateHash:       contentSHA256(secret),
			WorkspaceGeneration: generation, WorkspaceStateHash: stateHash,
			BaselineIdentity: baselineIdentityFor(w.ctx, w.path),
		},
		Evidence: append([]proxyEvidence{ev}, w.stage(evID)...),
	}
	d := decideAuthorization(w.ctx, in)
	g, ok, why := mintAuthorizationGrant(w.ctx, in, d, "selected")
	if !ok {
		t.Fatalf("no grant to inspect: %s", why)
	}
	// The canonical target is an identity the grant MUST bind -- a licence
	// that could not say what it was for would be useless. What it may not
	// hold is content: bytes, command text, source, prose.
	blob, err := json.Marshal(g)
	if err != nil {
		t.Fatal(err)
	}
	for _, needle := range []string{
		secret, "hunter2", "TOKEN", "print(7)", "pytest", "--token",
	} {
		if strings.Contains(string(blob), needle) {
			t.Errorf("the grant carries %q", needle)
		}
	}
	if g.TargetPath != w.path {
		t.Error("the grant does not name the target it is for")
	}
}

// TestGrantTelemetryCarriesNoPathOrContent pins the record that leaves the
// process. The grant may hold the canonical path because it has to compare
// against it; an operator log may not, because a workspace path is content.
func TestGrantTelemetryCarriesNoPathOrContent(t *testing.T) {
	const secret = "TOKEN = 'hunter2'\nprint(7)\n"
	w := newAuthWorld(t,
		`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],`+
			`"verification_knowledge":"declared","verification":["pytest --token=hunter2"]}`,
		"solve.py", secret, true)
	ev, evID := w.mustObserve(t)
	generation, stateHash := workspaceIdentity(w.ctx)
	in := authorizationInput{
		Obligations: requestObligations(w.ctx), TargetPath: w.path,
		CandidateHash: contentSHA256(secret),
		Identity: V3EvidenceProvenance{
			RequestID: requestIDOf(w.ctx), InvocationID: evID.InvocationID,
			CandidateInstanceID: evID.CandidateInstanceID,
			CandidateHash:       contentSHA256(secret),
			WorkspaceGeneration: generation, WorkspaceStateHash: stateHash,
			BaselineIdentity: baselineIdentityFor(w.ctx, w.path),
		},
		Evidence: append([]proxyEvidence{ev}, w.stage(evID)...),
	}
	d := decideAuthorization(w.ctx, in)

	captured := captureShadow(t, func() {
		g, ok, why := mintAuthorizationGrant(w.ctx, in, d, "selected")
		if !ok {
			t.Fatalf("no grant: %s", why)
		}
		claim := grantClaim{
			RequestID: requestIDOf(w.ctx), InvocationID: g.InvocationID,
			CandidateInstanceID: g.CandidateInstanceID, CandidateHash: g.CandidateHash,
			TargetPath: w.path, WorkspaceGeneration: g.WorkspaceGeneration,
			WorkspaceStateHash: g.WorkspaceStateHash,
			BaselineIdentity:   g.BaselineIdentity, BaselineHash: g.BaselineHash,
			ObligationSetID: g.ObligationSetID, EvidenceSetID: g.EvidenceSetID,
		}
		if _, why := consumeAuthorizationGrant(w.ctx, claim); why != "" {
			t.Fatalf("the grant would not spend: %s", why)
		}
	})
	if len(captured) == 0 {
		t.Skip("no shadow sink is active in this build")
	}
	for _, rec := range captured {
		blob, err := json.Marshal(rec)
		if err != nil {
			t.Fatal(err)
		}
		for _, needle := range []string{
			secret, "hunter2", "TOKEN", "print(7)", "pytest", "solve.py",
			w.ctx.WorkingDir,
		} {
			if strings.Contains(string(blob), needle) {
				t.Errorf("a grant telemetry record carries %q", needle)
			}
		}
	}
}

// TestConsumptionNeverRecomputesTheDecision reads the consumer's own source.
// A decision recomputed under new state is a different decision wearing the
// old one's identity, and the whole point of a grant is that it froze one.
func TestConsumptionNeverRecomputesTheDecision(t *testing.T) {
	src, err := os.ReadFile("authorization_grant.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	start := strings.Index(body, "func consumeAuthorizationGrant(")
	if start < 0 {
		t.Fatal("the consumer is gone")
	}
	fn := body[start:]
	if next := strings.Index(fn[1:], "\nfunc "); next >= 0 {
		fn = fn[:next+1]
	}
	for _, banned := range []string{
		"decideAuthorization", "requestObligations", "deriveTaskObligations",
		"produceSyntaxEvidence", "produceDeclaredVerificationEvidence",
		"stageCandidate", "observeCandidateAuthorization",
	} {
		if strings.Contains(fn, banned) {
			t.Errorf("consumption calls %s: it must validate, never re-decide", banned)
		}
	}
}

// TestSelectionIsNotAuthorization pins the separation the grant records but
// never acts on: what the pipeline picked has no bearing on whether the
// client's obligations were met.
func TestSelectionIsNotAuthorization(t *testing.T) {
	w := newGrantWorld(t)
	g := w.mint(t)
	if g.SelectedCandidateID == "" {
		t.Fatal("the grant does not record what was selected")
	}
	// A grant whose selected id says something else spends exactly the same:
	// the claim never mentions selection, so selection cannot admit or refuse.
	claim := w.claim()
	w.ctx.grantMu.Lock()
	w.ctx.grants[g.ID].SelectedCandidateID = "a completely different winner"
	w.ctx.grantMu.Unlock()
	if _, why := consumeAuthorizationGrant(w.ctx, claim); why != "" {
		t.Errorf("selection changed the consumption outcome: %s", why)
	}
}
