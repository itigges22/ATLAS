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

	// Staging supplies what this producer needs and nothing else could: the
	// exact declared command, run against an isolated workspace holding the
	// exact candidate bytes, observed either side.
	//
	// The two mechanisms it is NOT remain what they were. The V3 service runs
	// its own smoke checks and never receives the task contract, so it cannot
	// run what the client declared and cannot manufacture this authority by
	// calling its own endpoint. The MODEL running the same command through
	// run_command is a different, untrusted event against the production
	// workspace after bytes have landed. Neither reaches this producer.
	ProvenanceClientDeclaredVerification: evidenceProducerWired,
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
	outcome checkOutcome) (proxyEvidence, candidateEvidenceIdentity, bool) {
	if ctx == nil || ctx.TaskContract == nil {
		return proxyEvidence{}, candidateEvidenceIdentity{}, false
	}
	obs := requestObligations(ctx)
	if len(obs) == 0 {
		return proxyEvidence{}, candidateEvidenceIdentity{}, false
	}
	resolved := resolveAgentPath(ctx, path)
	// Only a target the client declared. A delivery elsewhere is not a
	// delivery this evidence has anything to say about.
	if !targetIsAuthorized(obs, resolved) {
		return proxyEvidence{}, candidateEvidenceIdentity{}, false
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
		return proxyEvidence{}, candidateEvidenceIdentity{}, false
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
		return proxyEvidence{}, candidateEvidenceIdentity{}, false
	}
	recordEvidenceObservation(ctx, ev)
	return ev, id, true
}

// observeCandidateVerification is THE production call path for the
// client-declared verification producer.
//
// The trust boundary is the whole point of the shape below. The proxy -- not
// the service, not the executor, not the model -- reads the client's declared
// commands out of the validated request, builds the staging request itself,
// and afterwards matches every returned result back against the obligations it
// derived. A result naming a command the request never declared, or an
// obligation the proxy does not own, or bytes other than the candidate,
// matches nothing and produces nothing. Staging reports observations; it does
// not declare its own provenance trusted.
//
// Runs only when a client actually declared commands. A request that declared
// none stages nothing, so no command executes on its behalf.
//
// Returns the evidence for the caller's own inspection in tests. Production
// ignores the return value: records go to private telemetry and no delivery,
// completion or generation decision reads them.
func observeCandidateVerification(ctx *AgentContext, path, code string,
	id candidateEvidenceIdentity) ([]proxyEvidence, bool) {
	if ctx == nil || ctx.TaskContract == nil {
		return nil, false
	}
	if strings.TrimSpace(id.InvocationID) == "" ||
		strings.TrimSpace(id.CandidateInstanceID) == "" {
		return nil, false
	}
	obs := requestObligations(ctx)
	if len(obs) == 0 {
		return nil, false
	}
	resolved := resolveAgentPath(ctx, path)
	if !targetIsAuthorized(obs, resolved) {
		return nil, false
	}

	// The declared commands, from the proxy's own derivation of the validated
	// request. This list is the authority; nothing downstream may add to it.
	var declared []taskObligation
	for _, o := range authorizationPrerequisites(obs) {
		if o.Kind == ObligationDeclaredCommand && o.Required {
			declared = append(declared, o)
		}
	}
	if len(declared) == 0 {
		return nil, false
	}

	hash := contentSHA256(code)
	generation, stateHash := workspaceIdentity(ctx)
	req := stagingRequest{
		WireVersion:    stagingWireVersion,
		CandidateBytes: code,
		Budget:         defaultStagingBudget(),
		Identity: stagingIdentity{
			RequestID:           requestIDOf(ctx),
			InvocationID:        id.InvocationID,
			CandidateInstanceID: id.CandidateInstanceID,
			CandidateHash:       hash,
			TargetPath:          resolved,
			BaselineIdentity:    baselineIdentityFor(ctx, resolved),
			WorkspaceGeneration: generation,
			WorkspaceStateHash:  stateHash,
		},
	}
	for i, o := range declared {
		req.Commands = append(req.Commands, stagingCommand{
			Text:         o.Subject,
			Identity:     contentSHA256(o.Subject),
			ObligationID: o.ID,
			Index:        i,
			Count:        len(declared),
		})
	}
	if ok, _ := req.validate(); !ok {
		// A declared set the staging budget cannot hold whole. Nothing runs:
		// a partial set is not a smaller obligation.
		return nil, false
	}

	result, ok := stageCandidate(ctx, req)
	if !ok {
		return nil, false
	}
	if valid, _ := result.validateAgainst(req); !valid {
		// The result contradicted the request it answers. Refusing here is
		// what makes a malformed or forged result worth nothing.
		return nil, false
	}

	byID := map[string]taskObligation{}
	for _, o := range declared {
		byID[o.ID] = o
	}
	var out []proxyEvidence
	for _, r := range result.Commands {
		o, known := byID[r.ObligationID]
		if !known {
			continue
		}
		ev, ok := produceDeclaredVerificationEvidence(ctx, verificationEvidenceRequest{
			Obligation: o, Result: r, Identity: result.Identity,
		})
		if !ok {
			continue
		}
		recordEvidenceObservation(ctx, ev)
		out = append(out, ev)
	}
	return out, len(out) > 0
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
