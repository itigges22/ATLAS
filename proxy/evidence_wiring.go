package main

import (
	"fmt"
	"strings"
)

// Where the evidence producers actually run, and where they cannot.
//
// The producers landed with no production caller: correct for a slice that had
// to prove inertness, and useless past it. This file is the one place either
// producer is invoked on a live request, and the one place the missing
// producer is recorded as missing rather than quietly absent.
//
// Nothing here changes what lands. Records go to the private shadow sink and
// to nowhere else: no SSE event, no ToolResult, no model-visible text, and no
// input to any delivery, completion or generation decision.

// evidenceProducerAvailability says, for each trusted source, whether a live
// production path can actually produce it on this build.
//
// The distinction matters because "no caller" has two very different causes. A
// producer with a wiring bug and a producer with nothing to observe look the
// same to a structural guard, and only one of them is acceptable to ship.
type evidenceProducerAvailability string

const (
	// evidenceProducerWired: exactly one live production call path exists.
	evidenceProducerWired evidenceProducerAvailability = "wired"
	// evidenceProducerUnavailable: the mechanism this producer would observe
	// does not exist in this build. Named here so the absence is a registered
	// fact rather than an oversight.
	evidenceProducerUnavailable evidenceProducerAvailability = "unavailable"
)

// evidenceProducerStatus is the enumerated inventory. A structural guard reads
// it and fails when a producer's real wiring disagrees with its entry.
var evidenceProducerStatus = map[string]evidenceProducerAvailability{
	ProvenanceProxyOwnedValidation: evidenceProducerWired,

	// No live path executes a client-declared command against a staging
	// workspace holding the candidate bytes BEFORE delivery.
	//
	// What exists: the V3 service runs its own smoke checks and self-tests
	// inside the sandbox, and the proxy records a VerificationRecord when the
	// MODEL runs a command through run_command. Neither is the thing this
	// producer needs. The service never receives the task contract, so it
	// cannot run what the client declared; the model's run happens against the
	// production workspace AFTER bytes have landed, so it speaks for what is
	// already there rather than for a candidate.
	//
	// Manufacturing evidence from either would be exactly the fabrication the
	// producer exists to prevent, so behavioral authorization has no source on
	// this build and says so.
	ProvenanceClientDeclaredVerification: evidenceProducerUnavailable,
}

// candidateEvidenceIdentity is the identity a candidate carries through one
// invocation. Minted once, at the point the candidate's bytes are fixed.
type candidateEvidenceIdentity struct {
	InvocationID        string
	CandidateInstanceID string
}

// nextInvocationIdentity mints the identity for one candidate.
//
// The invocation is the generation call; the candidate instance is the exact
// bytes within it. Two candidates in one invocation differ by content hash;
// two invocations for the same bytes differ by sequence. Neither can be
// mistaken for the other, which is what stops one candidate's evidence binding
// to another's.
func nextInvocationIdentity(ctx *AgentContext, candidateHash string) candidateEvidenceIdentity {
	if ctx == nil || strings.TrimSpace(candidateHash) == "" {
		return candidateEvidenceIdentity{}
	}
	ctx.v3InvocationMu.Lock()
	ctx.v3InvocationSeq++
	seq := ctx.v3InvocationSeq
	ctx.v3InvocationMu.Unlock()

	request := requestIDOf(ctx)
	if strings.TrimSpace(request) == "" {
		// Without a request there is nothing to bind to, and an identity that
		// binds to nothing is worse than none.
		return candidateEvidenceIdentity{}
	}
	invocation := fmt.Sprintf("%s:inv:%d", request, seq)
	return candidateEvidenceIdentity{
		InvocationID:        invocation,
		CandidateInstanceID: invocation + ":" + candidateHash[:16],
	}
}

// observeDeliveredCandidateSyntax is THE production call path for the
// proxy-owned syntax producer.
//
// It runs at the final-byte observation the write path already makes: the
// point where the bytes about to land are fixed and the structural gate has
// just reported on exactly those bytes. Nothing is re-checked and no second
// sandbox call is made -- the verdict is handed over, not recomputed.
//
// Returns the evidence for the caller's own inspection in tests. Production
// ignores the return value: the record goes to private telemetry and the
// delivery decision above is unchanged.
func observeDeliveredCandidateSyntax(ctx *AgentContext, path, code string,
	outcome checkOutcome) (proxyEvidence, bool) {
	if ctx == nil || ctx.TaskContract == nil {
		return proxyEvidence{}, false
	}
	obs := requestObligations(ctx)
	if len(obs) == 0 {
		return proxyEvidence{}, false
	}
	resolved := resolveAgentPath(ctx, path)
	// Only a target the client declared. A delivery elsewhere is not a
	// delivery this evidence has anything to say about.
	if !targetIsAuthorized(obs, resolved) {
		return proxyEvidence{}, false
	}
	var syntax taskObligation
	for _, o := range authorizationPrerequisites(obs) {
		if o.Kind == ObligationSyntacticValidity && o.Subject == resolved {
			syntax = o
			break
		}
	}
	if syntax.ID == "" {
		// A class the gate does not govern owes no structural obligation, and
		// inventing one to have something to evidence is the fabrication this
		// whole split exists to prevent.
		return proxyEvidence{}, false
	}

	hash := contentSHA256(code)
	id := nextInvocationIdentity(ctx, hash)
	ev, ok := produceSyntaxEvidence(ctx, syntaxEvidenceRequest{
		Obligation:          syntax,
		Path:                path,
		CandidateBytes:      code,
		CandidateHash:       hash,
		Outcome:             outcome,
		InvocationID:        id.InvocationID,
		CandidateInstanceID: id.CandidateInstanceID,
		BaselineIdentity:    baselineIdentityFor(ctx, resolved),
	})
	if !ok {
		return proxyEvidence{}, false
	}
	recordEvidenceObservation(ctx, ev)
	return ev, true
}

// recordEvidenceObservation writes one observation to the private shadow sink.
//
// Identities and hashes only. No candidate byte, no command string, no path
// content, no diagnostic text -- a structural gate's failure detail can quote
// the source line that failed, so the detail is deliberately not carried.
// influences_live_decision is false and is a fact about this build, not a
// hope: no production consumer of provenance exists.
func recordEvidenceObservation(ctx *AgentContext, ev proxyEvidence) {
	sink := activeShadowSink.Load()
	if !sink.enabled() {
		return
	}
	p := ev.Provenance
	sink.submit(map[string]interface{}{
		"schema_version":           shadowSchemaVersionEvidence,
		"record_kind":              "candidate_evidence_observation",
		"request_id":               p.RequestID,
		"invocation_id":            p.InvocationID,
		"candidate_instance_id":    p.CandidateInstanceID,
		"candidate_hash":           p.CandidateHash,
		"workspace_generation":     p.WorkspaceGeneration,
		"workspace_state_hash":     p.WorkspaceStateHash,
		"baseline_identity":        p.BaselineIdentity,
		"obligation_id":            p.ObligationID,
		"source":                   p.Source,
		"required_strength":        p.RequiredStrength,
		"observed_strength":        p.ObservedStrength,
		"outcome":                  string(ev.Outcome),
		"influences_live_decision": false,
		"build_version":            APIVersion,
	})
}
