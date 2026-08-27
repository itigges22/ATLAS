package main

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

// Turning an authorized decision into something that can be spent exactly once.
//
// A decision is a statement about a moment: these bytes, this candidate, this
// workspace, and every obligation the client stated satisfied by evidence that
// binds. It is not a licence. Between deciding and writing, the workspace can
// move, the same candidate can be offered again, the request can be cancelled,
// and two goroutines can arrive at the same conclusion at once -- and a
// decision that is merely TRUE is true every one of those times.
//
// A grant is the decision made spendable. It carries the identities the
// decision was about, it is consumed atomically at one owner, and a second
// attempt gets nothing. That is the whole difference between "these bytes were
// authorized" and "these bytes may land now".
//
// Three separations the vocabulary keeps: selection does not imply
// authorization -- V3 picking a winner says nothing about whether the client's
// obligations are met; authorization does not imply delivery -- a grant can be
// minted and never spent; and consumption never re-decides -- it validates what
// the grant was bound to and refuses, because a decision recomputed under new
// state is a different decision wearing the old one's identity.

// grantCapacity bounds how many live grants one request may hold.
//
// A request delivers a handful of artifacts. A number far above that is a
// runaway, and the honest response to a runaway is to refuse before mutating
// rather than to grow.
const grantCapacity = 64

// grantRetirement says why a grant can no longer be spent. A retired grant is
// kept, not deleted: "spent" and "cancelled" are different answers to a second
// attempt, and collapsing them would make a replay look like a timeout.
type grantRetirement string

const (
	grantLive       grantRetirement = ""
	grantConsumed   grantRetirement = "already_consumed"
	grantCancelled  grantRetirement = "request_cancelled"
	grantTerminal   grantRetirement = "terminal_emitted"
	grantSuperseded grantRetirement = "superseded_by_a_later_decision"
	grantSessionEnd grantRetirement = "session_ended"
)

// authorizationGrant is one spendable authorization.
//
// Every field is an identity or a hash. No command text, candidate byte,
// source fragment, prompt or evidence detail is held: a grant that carried
// content would put it somewhere the redaction rules do not reach.
type authorizationGrant struct {
	// ID is canonical, so two spellings of the same delivery are one grant.
	ID string

	RequestID           string
	InvocationID        string
	CandidateInstanceID string
	CandidateHash       string
	// TargetPath is canonical: resolved, cleaned, absolute.
	TargetPath string

	WorkspaceGeneration int
	WorkspaceStateHash  string

	BaselineIdentity string
	BaselineHash     string
	// BaselineGeneration is the target's own ledger generation when the
	// grant was minted. The workspace digest catches a change anywhere; this
	// catches a move, a recreation or a tombstone at exactly this path, which
	// is the one that decides whether the thing being replaced is still the
	// thing that was authorized.
	BaselineGeneration int
	BaselineTombstoned bool

	// ObligationSetID and EvidenceSetID name the exact sets the decision was
	// reached over. A grant minted against one set may not be spent after
	// either changed, and naming them by hash is what makes that checkable
	// without holding either.
	ObligationSetID string
	EvidenceSetID   string

	// SelectedCandidateID is what the pipeline picked, recorded so the two
	// facts stay distinguishable. Selection is not authorization.
	SelectedCandidateID string

	// DecisionGeneration orders decisions within one request. A later
	// decision for the same target supersedes an earlier grant rather than
	// coexisting with it.
	DecisionGeneration int

	retired grantRetirement
}

// grantKey is the canonical identity of a grant.
//
// Built from the canonical target and the candidate instance, so `solve.py`,
// `./solve.py` and the absolute spelling all name the same grant: an alias
// that minted its own would be a second licence for one delivery.
func grantKey(requestID, invocationID, candidateInstanceID, canonicalTarget string) string {
	return contentSHA256(strings.Join([]string{
		requestID, invocationID, candidateInstanceID, canonicalTarget,
	}, "\x00"))
}

// obligationSetIdentity names a set of obligations without holding them.
func obligationSetIdentity(obs []taskObligation) string {
	ids := make([]string, 0, len(obs))
	for _, o := range obs {
		ids = append(ids, fmt.Sprintf("%s|%v|%s", o.ID, o.Required, o.RequiredStrength))
	}
	sort.Strings(ids)
	return contentSHA256(strings.Join(ids, "\n"))
}

// evidenceSetIdentity names the evidence a decision was reached over.
//
// Outcome and strength are part of the identity: the same records with one
// verdict flipped are a different set, and a grant bound to the old one must
// not survive into the new.
func evidenceSetIdentity(evidence []proxyEvidence) string {
	ids := make([]string, 0, len(evidence))
	for _, e := range evidence {
		p := e.Provenance
		ids = append(ids, strings.Join([]string{
			p.Source, p.ObligationID, p.CandidateInstanceID, p.CandidateHash,
			p.CommandIdentity, p.ObservedStrength, string(e.Outcome),
		}, "|"))
	}
	sort.Strings(ids)
	return contentSHA256(strings.Join(ids, "\n"))
}

// mintAuthorizationGrant turns an authorized decision into a spendable one.
//
// It mints nothing it is not sure of. Every condition below is a fact the
// decision already established or an identity the grant must bind, and a
// missing one is a refusal rather than a grant with a hole in it.
func mintAuthorizationGrant(ctx *AgentContext, in authorizationInput,
	d AuthorizationDecision, selectedCandidateID string) (*authorizationGrant, bool, string) {
	if ctx == nil {
		return nil, false, "no request"
	}
	if !d.Authorized || d.Reason != ReasonAuthorized {
		return nil, false, "the decision did not authorize"
	}
	// Target knowledge must be contract-declared. A grant over a target the
	// client never named is the one thing no amount of evidence can fix.
	if ctx.TaskContract == nil || len(in.Obligations) == 0 {
		return nil, false, "no declared obligations"
	}
	// Canonical before the check: obligations hold canonical paths, and an
	// alias spelling of a declared target is still that target. Resolving
	// afterwards would let `./solve.py` be refused and `solve.py` accepted.
	target := resolveAgentPath(ctx, in.TargetPath)
	if !targetIsAuthorized(in.Obligations, target) {
		return nil, false, "the target is not declared"
	}
	// Every prerequisite satisfied and nothing owed. The decision says so;
	// this refuses to take its word without the accounting agreeing.
	if len(d.Missing) != 0 {
		return nil, false, "an authorization prerequisite is still owed"
	}
	if len(d.Satisfied) == 0 {
		return nil, false, "nothing was demonstrated"
	}
	// Every required declared command must have current trusted evidence.
	// Read from the obligation set rather than from the decision, so a
	// decision that somehow satisfied a command without a record cannot mint.
	if _, missing := declaredVerificationCoverage(in.Obligations, in.Evidence); len(missing) != 0 {
		return nil, false, "a declared command has no current trusted evidence"
	}
	// Modern, non-legacy record only.
	if in.Envelope != nil {
		if availability, _ := in.Envelope.Validate(); availability != EvidenceAvailable {
			return nil, false, "the service record is not usable"
		}
		if !in.Envelope.Evaluation.Supported {
			return nil, false, "the adapter does not support this artifact"
		}
	}

	id := in.Identity
	for _, f := range []struct{ name, value string }{
		{"request_id", id.RequestID},
		{"invocation_id", id.InvocationID},
		{"candidate_instance_id", id.CandidateInstanceID},
		{"candidate_hash", id.CandidateHash},
		{"workspace_state_hash", id.WorkspaceStateHash},
		{"target", target},
	} {
		if strings.TrimSpace(f.value) == "" {
			return nil, false, f.name + " is missing"
		}
	}
	if id.CandidateHash != in.CandidateHash {
		return nil, false, "the identity and the candidate disagree"
	}
	if requestIDOf(ctx) != id.RequestID {
		return nil, false, "the identity is not this request's"
	}

	g := &authorizationGrant{
		ID:                  grantKey(id.RequestID, id.InvocationID, id.CandidateInstanceID, target),
		RequestID:           id.RequestID,
		InvocationID:        id.InvocationID,
		CandidateInstanceID: id.CandidateInstanceID,
		CandidateHash:       id.CandidateHash,
		TargetPath:          target,
		WorkspaceGeneration: id.WorkspaceGeneration,
		WorkspaceStateHash:  id.WorkspaceStateHash,
		BaselineIdentity:    id.BaselineIdentity,
		BaselineHash:        fileSHA256(ctx, target),
		BaselineGeneration:  targetGeneration(ctx, target),
		BaselineTombstoned:  targetTombstoned(ctx, target),
		ObligationSetID:     obligationSetIdentity(in.Obligations),
		EvidenceSetID:       evidenceSetIdentity(in.Evidence),
		SelectedCandidateID: selectedCandidateID,
	}

	ctx.grantMu.Lock()
	defer ctx.grantMu.Unlock()
	if ctx.grantsOff != "" {
		return nil, false, ctx.grantsOff
	}
	if ctx.grants == nil {
		ctx.grants = map[string]*authorizationGrant{}
	}
	// Overflow refuses BEFORE anything is stored. A capacity check that
	// evicted would silently retire a grant somebody else is about to spend.
	if _, exists := ctx.grants[g.ID]; !exists && len(ctx.grants) >= grantCapacity {
		return nil, false, "too many live authorizations for one request"
	}
	// A later decision for the same target supersedes the earlier grant. Two
	// live licences for one delivery is the state this whole type exists to
	// make impossible.
	for _, other := range ctx.grants {
		if other.retired == grantLive && other.TargetPath == g.TargetPath && other.ID != g.ID {
			other.retired = grantSuperseded
		}
	}
	ctx.grantSeq++
	g.DecisionGeneration = ctx.grantSeq
	ctx.grants[g.ID] = g
	recordGrantEvent(ctx, g, "minted", "")
	return g, true, ""
}

// grantClaim is what a consumer must present. It is built from the LIVE state
// at the moment of consumption, never copied off the grant -- a claim taken
// from the grant would be asking every grant whether it matched itself.
type grantClaim struct {
	RequestID           string
	InvocationID        string
	CandidateInstanceID string
	CandidateHash       string
	TargetPath          string
	WorkspaceGeneration int
	WorkspaceStateHash  string
	BaselineIdentity    string
	BaselineHash        string
	BaselineGeneration  int
	BaselineTombstoned  bool
	ObligationSetID     string
	EvidenceSetID       string
}

// consumeAuthorizationGrant spends a grant, exactly once.
//
// Atomic under one lock: the check that it is live and the mark that it is
// spent cannot be separated, or two concurrent deliveries both pass the check.
// It re-validates every bound identity against the claim and refuses on the
// first that differs -- and it never recomputes the decision, because a
// decision recomputed under new state is a different decision wearing the old
// one's identity.
func consumeAuthorizationGrant(ctx *AgentContext, claim grantClaim) (*authorizationGrant, string) {
	if ctx == nil {
		return nil, "no request"
	}
	target := resolveAgentPath(ctx, claim.TargetPath)
	key := grantKey(claim.RequestID, claim.InvocationID, claim.CandidateInstanceID, target)

	ctx.grantMu.Lock()
	defer ctx.grantMu.Unlock()
	g := ctx.grants[key]
	if g == nil {
		return nil, "no authorization for this delivery"
	}
	if g.retired != grantLive {
		return nil, string(g.retired)
	}
	for _, f := range []struct {
		name       string
		held, want string
	}{
		{"request_id", g.RequestID, claim.RequestID},
		{"invocation_id", g.InvocationID, claim.InvocationID},
		{"candidate_instance_id", g.CandidateInstanceID, claim.CandidateInstanceID},
		{"candidate_hash", g.CandidateHash, claim.CandidateHash},
		{"target", g.TargetPath, target},
		{"workspace_state_hash", g.WorkspaceStateHash, claim.WorkspaceStateHash},
		{"baseline_identity", g.BaselineIdentity, claim.BaselineIdentity},
		{"baseline_hash", g.BaselineHash, claim.BaselineHash},
		{"obligation_set", g.ObligationSetID, claim.ObligationSetID},
		{"evidence_set", g.EvidenceSetID, claim.EvidenceSetID},
	} {
		if f.held != f.want {
			recordGrantEvent(ctx, g, "refused", f.name+"_differs")
			return nil, f.name + " is not what the authorization was about"
		}
	}
	if g.BaselineGeneration != claim.BaselineGeneration {
		recordGrantEvent(ctx, g, "refused", "baseline_generation_differs")
		return nil, "the target changed since the authorization"
	}
	if g.BaselineTombstoned != claim.BaselineTombstoned || claim.BaselineTombstoned {
		recordGrantEvent(ctx, g, "refused", "target_tombstoned")
		return nil, "the target was deliberately removed"
	}
	if g.WorkspaceGeneration != claim.WorkspaceGeneration {
		recordGrantEvent(ctx, g, "refused", "workspace_generation_differs")
		return nil, "the workspace moved since the authorization"
	}
	// Cancellation is checked last and inside the lock, so a cancel that
	// lands mid-consumption cannot be raced past.
	if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
		g.retired = grantCancelled
		recordGrantEvent(ctx, g, "retired", string(grantCancelled))
		return nil, string(grantCancelled)
	}

	g.retired = grantConsumed
	recordGrantEvent(ctx, g, "consumed", "")
	spent := *g
	return &spent, ""
}

// retireAuthorizationGrants ends every live grant and refuses future minting.
//
// Called when the request can no longer deliver anything: cancelled, terminal
// emitted, session over. Unrelated success never calls it, and nothing ever
// refreshes a grant -- a licence that could be renewed is not one-time.
func retireAuthorizationGrants(ctx *AgentContext, reason grantRetirement) int {
	if ctx == nil || reason == grantLive {
		return 0
	}
	ctx.grantMu.Lock()
	defer ctx.grantMu.Unlock()
	n := 0
	for _, g := range ctx.grants {
		if g.retired == grantLive {
			g.retired = reason
			recordGrantEvent(ctx, g, "retired", string(reason))
			n++
		}
	}
	ctx.grantsOff = string(reason)
	return n
}

// liveGrantCount is for tests and telemetry. It counts, and changes nothing.
func liveGrantCount(ctx *AgentContext) int {
	if ctx == nil {
		return 0
	}
	ctx.grantMu.Lock()
	defer ctx.grantMu.Unlock()
	n := 0
	for _, g := range ctx.grants {
		if g.retired == grantLive {
			n++
		}
	}
	return n
}

// recordGrantEvent writes one grant transition to the private shadow sink.
//
// Identities only, like every other record on this path. Caller holds grantMu.
func recordGrantEvent(ctx *AgentContext, g *authorizationGrant, event, detail string) {
	sink := activeShadowSink.Load()
	if !sink.enabled() {
		return
	}
	sink.submit(map[string]interface{}{
		"schema_version":        shadowSchemaVersionEvidence,
		"record_kind":           "authorization_grant_event",
		"event":                 event,
		"detail":                detail,
		"grant_id":              g.ID,
		"request_id":            g.RequestID,
		"invocation_id":         g.InvocationID,
		"candidate_instance_id": g.CandidateInstanceID,
		"candidate_hash":        g.CandidateHash,
		"workspace_generation":  g.WorkspaceGeneration,
		"workspace_state_hash":  g.WorkspaceStateHash,
		"baseline_identity":     g.BaselineIdentity,
		"obligation_set_id":     g.ObligationSetID,
		"evidence_set_id":       g.EvidenceSetID,
		"selected_candidate_id": g.SelectedCandidateID,
		"decision_generation":   g.DecisionGeneration,
		"build_version":         APIVersion,
	})
}

// targetGeneration and targetTombstoned read one path's ledger facts.
//
// Separate from workspaceIdentity on purpose: the digest says the workspace
// moved, and these say whether THIS artifact did. A delivery cares about the
// second, and reporting the first for it would blame an unrelated write.
func targetGeneration(ctx *AgentContext, resolved string) int {
	if ctx == nil {
		return 0
	}
	ctx.LedgerMu.Lock()
	defer ctx.LedgerMu.Unlock()
	if d := ctx.Ledger[filepath.Clean(resolved)]; d != nil {
		return d.Generation
	}
	return 0
}

func targetTombstoned(ctx *AgentContext, resolved string) bool {
	if ctx == nil {
		return false
	}
	ctx.LedgerMu.Lock()
	defer ctx.LedgerMu.Unlock()
	if d := ctx.Ledger[filepath.Clean(resolved)]; d != nil {
		return d.Tombstoned
	}
	return false
}
