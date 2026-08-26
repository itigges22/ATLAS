package main

import (
	"sort"
	"strings"
)

// Whether a candidate WOULD be authorized to land, computed and not consulted.
//
// Every piece of the answer already exists: the request says which targets it
// owns, the task says what it obliges, the adapter says what it can observe,
// the evidence says where it came from and what it is about. Nothing had ever
// put them together, so "may these bytes land" was still answered by
// v3DeliveryAuthorized reading a selection status.
//
// This is that answer, assembled from the typed parts, and it is observe-only.
// It reaches no write, no ledger, no completion, no prompt and no generation
// decision; the existing delivery decisions remain authoritative and unchanged.
// What it produces is a private record saying what would have happened, which
// is the only way to find out whether the typed answer agrees with the live one
// before letting it decide anything.

// AuthorizationReason is the closed vocabulary of why a decision came out the
// way it did. A reason outside this set is a state nobody classified, which is
// a refusal rather than a default.
type AuthorizationReason string

const (
	// Granted.
	ReasonAuthorized AuthorizationReason = "authorized"

	// Refused.
	ReasonTargetNotDeclared             AuthorizationReason = "target_not_declared"
	ReasonObligationUnknown             AuthorizationReason = "obligation_unknown"
	ReasonAdapterUnsupported            AuthorizationReason = "adapter_unsupported"
	ReasonEvidenceMissing               AuthorizationReason = "evidence_missing"
	ReasonEvidenceTooWeak               AuthorizationReason = "evidence_too_weak"
	ReasonProvenanceUntrusted           AuthorizationReason = "provenance_untrusted"
	ReasonCandidateMismatch             AuthorizationReason = "candidate_mismatch"
	ReasonRequestOrInvocationMismatch   AuthorizationReason = "request_or_invocation_mismatch"
	ReasonWorkspaceStale                AuthorizationReason = "workspace_stale"
	ReasonBaselineNotPreserved          AuthorizationReason = "baseline_not_preserved"
	ReasonCommandMismatch               AuthorizationReason = "command_mismatch"
	ReasonLegacyRecord                  AuthorizationReason = "legacy_record"
	ReasonPostDeliverySettlementPending AuthorizationReason = "post_delivery_settlement_pending"
	// ReasonUnknown is never chosen deliberately. It is what a contradictory
	// or unclassified state becomes, and it never authorizes.
	ReasonUnknown AuthorizationReason = "unknown"
)

var authorizationReasons = map[AuthorizationReason]bool{
	ReasonAuthorized: true, ReasonTargetNotDeclared: true,
	ReasonObligationUnknown: true, ReasonAdapterUnsupported: true,
	ReasonEvidenceMissing: true, ReasonEvidenceTooWeak: true,
	ReasonProvenanceUntrusted: true, ReasonCandidateMismatch: true,
	ReasonRequestOrInvocationMismatch: true, ReasonWorkspaceStale: true,
	ReasonBaselineNotPreserved: true, ReasonCommandMismatch: true,
	ReasonLegacyRecord: true, ReasonPostDeliverySettlementPending: true,
	ReasonUnknown: true,
}

// AuthorizationDecision is one complete answer about one candidate.
//
// It names what it consumed as well as what it concluded. A decision that
// cannot say which evidence it would have relied on is one nobody can audit
// against what actually landed.
type AuthorizationDecision struct {
	Authorized bool
	Reason     AuthorizationReason

	// Satisfied and Missing are obligation ids, sorted. Together they account
	// for every authorization prerequisite the task stated.
	Satisfied []string
	Missing   []string

	// EvidenceConsumed names the candidate-instance/obligation pairs this
	// decision would have relied on, sorted. Identities only.
	EvidenceConsumed []string

	// SettlementRequired is true whenever the task states a post-delivery
	// obligation that is not yet answerable. Authorization does not wait for
	// it -- that is the circle this design removed -- but a caller that
	// delivers still owes it afterwards.
	SettlementRequired bool

	// InfluencesLiveDecision is false on this build and is written into the
	// record. It is a fact about the wiring, not an intention.
	InfluencesLiveDecision bool
}

// authorizationInput is everything the decision may look at. Assembling it
// explicitly is what stops the decision reaching for something else.
type authorizationInput struct {
	// Obligations are the task's own, derived from the validated request.
	Obligations []taskObligation
	// TargetPath is the canonical artifact the delivery would replace.
	TargetPath string
	// CandidateHash names the exact bytes proposed.
	CandidateHash string
	// Identity is what the evidence must bind to.
	Identity V3EvidenceProvenance
	// Evidence is what was actually observed about this candidate.
	Evidence []proxyEvidence
	// Envelope is the service's own record, consulted only for adapter
	// support and legacy status.
	Envelope *V3EvidenceEnvelope
}

// decideAuthorization is THE observe-only authorization owner.
//
// The order of the checks is the order in which an answer stops being possible:
// identity before obligations, obligations before evidence, evidence before
// strength. A refusal names the first thing that was wrong, so a reader learns
// what to fix rather than that something was.
func decideAuthorization(ctx *AgentContext, in authorizationInput) AuthorizationDecision {
	d := AuthorizationDecision{Reason: ReasonUnknown, InfluencesLiveDecision: false}

	// Nothing structured was stated, so there is nothing to authorize on. This
	// is not a refusal of the request -- legacy delivery is unchanged and
	// authoritative -- it is this decision declining to have an opinion.
	if len(in.Obligations) == 0 {
		d.Reason = ReasonLegacyRecord
		return d
	}
	// A record produced before provenance existed carries no authority, and a
	// service record whose envelope is unusable is not evidence about anything.
	if in.Envelope != nil {
		if availability, _ := in.Envelope.Validate(); availability != EvidenceAvailable {
			d.Reason = ReasonLegacyRecord
			return d
		}
		if !in.Envelope.Evaluation.Supported {
			d.Reason = ReasonAdapterUnsupported
			return d
		}
	}
	// The client must have asked for this artifact. Identity is necessary and
	// never sufficient: everything below still has to hold.
	if !targetIsAuthorized(in.Obligations, in.TargetPath) {
		d.Reason = ReasonTargetNotDeclared
		return d
	}
	if strings.TrimSpace(in.CandidateHash) == "" {
		d.Reason = ReasonCandidateMismatch
		return d
	}

	prerequisites := authorizationPrerequisites(in.Obligations)
	// Settlement is owed whenever the task states one and the bytes are not
	// yet confirmed. It never blocks authorization. Without a context there is
	// nothing to confirm against, so it is owed by definition.
	settled := false
	if ctx != nil {
		settled, _ = settlementIsComplete(ctx, in.Obligations, in.CandidateHash)
	}
	if !settled {
		d.SettlementRequired = len(postDeliverySettlement(in.Obligations)) > 0
	}

	// An obligation nobody classified, or one nothing can satisfy, ends the
	// question before any evidence is looked at.
	for _, o := range prerequisites {
		if _, ok := obligationRole(o.Kind); !ok {
			d.Reason = ReasonObligationUnknown
			d.Missing = append(d.Missing, o.ID)
			sort.Strings(d.Missing)
			return d
		}
		if obligationUnsatisfiableKinds[o.Kind] {
			d.Reason = ReasonObligationUnknown
			d.Missing = append(d.Missing, o.ID)
			sort.Strings(d.Missing)
			return d
		}
		if strengthRank(o.RequiredStrength) < 0 {
			d.Reason = ReasonObligationUnknown
			d.Missing = append(d.Missing, o.ID)
			sort.Strings(d.Missing)
			return d
		}
	}

	// Match evidence to obligations. The first refusal encountered is the one
	// reported, and it is specific: a caller learns that the workspace moved
	// rather than that "evidence was missing".
	satisfied := map[string]bool{}
	firstRefusal := ReasonUnknown
	for _, ev := range in.Evidence {
		reason := evidenceRefusalFor(ev, in.Identity)
		if reason == ReasonAuthorized {
			satisfied[ev.Provenance.ObligationID] = true
			d.EvidenceConsumed = append(d.EvidenceConsumed,
				ev.Provenance.CandidateInstanceID+"/"+ev.Provenance.ObligationID)
			continue
		}
		if firstRefusal == ReasonUnknown {
			firstRefusal = reason
		}
	}
	sort.Strings(d.EvidenceConsumed)

	for _, o := range prerequisites {
		if !o.Required {
			continue
		}
		if satisfied[o.ID] {
			d.Satisfied = append(d.Satisfied, o.ID)
		} else {
			d.Missing = append(d.Missing, o.ID)
		}
	}
	sort.Strings(d.Satisfied)
	sort.Strings(d.Missing)

	if len(d.Missing) > 0 {
		// A prerequisite nothing spoke for is missing evidence; one something
		// spoke for badly gets that something's own reason.
		d.Reason = ReasonEvidenceMissing
		if firstRefusal != ReasonUnknown {
			d.Reason = firstRefusal
		}
		// A baseline obligation left unmet is its own fact, and the more
		// useful one to report.
		for _, o := range prerequisites {
			if o.Kind == ObligationBaselinePreserved && !satisfied[o.ID] {
				d.Reason = ReasonBaselineNotPreserved
				break
			}
		}
		return d
	}
	if len(d.Satisfied) == 0 {
		// Every prerequisite accounted for and none of them required anything:
		// the task stated no quality obligation at all. Target identity alone
		// does not authorize arbitrary bytes.
		d.Reason = ReasonEvidenceMissing
		return d
	}

	d.Authorized = true
	d.Reason = ReasonAuthorized
	return d
}

// evidenceRefusalFor is why one piece of evidence cannot be relied on, or
// ReasonAuthorized when it can.
//
// Ordered so the most specific mismatch wins: a record about a different
// candidate is a candidate mismatch even though its workspace also differs.
func evidenceRefusalFor(ev proxyEvidence, asked V3EvidenceProvenance) AuthorizationReason {
	p := ev.Provenance
	if _, ok := provenanceCeiling[p.Source]; !ok {
		return ReasonProvenanceUntrusted
	}
	if p.RequestID != asked.RequestID || p.InvocationID != asked.InvocationID {
		return ReasonRequestOrInvocationMismatch
	}
	if p.CandidateHash != asked.CandidateHash ||
		p.CandidateInstanceID != asked.CandidateInstanceID {
		return ReasonCandidateMismatch
	}
	if p.CommandIdentity != asked.CommandIdentity {
		return ReasonCommandMismatch
	}
	if p.BaselineIdentity != asked.BaselineIdentity {
		return ReasonBaselineNotPreserved
	}
	if p.WorkspaceGeneration != asked.WorkspaceGeneration ||
		p.WorkspaceStateHash != asked.WorkspaceStateHash {
		return ReasonWorkspaceStale
	}
	if ev.Outcome != ValidationPassed {
		return ReasonEvidenceMissing
	}
	if ok, _ := p.MayAuthorize(); !ok {
		// Source and strength alone. Distinguish "this source may never close
		// this" from "it could, but did not reach the floor".
		ceiling := provenanceCeiling[p.Source]
		if strengthRank(ceiling) < strengthRank(p.RequiredStrength) {
			return ReasonProvenanceUntrusted
		}
		return ReasonEvidenceTooWeak
	}
	return ReasonAuthorized
}

// recordAuthorizationDecision writes the shadow decision to private telemetry.
//
// Identities, obligation ids and a closed reason. No candidate byte, no
// command string, no path content. influences_live_decision is false and is a
// fact about this build: nothing consumes what this returns.
func recordAuthorizationDecision(ctx *AgentContext, in authorizationInput,
	d AuthorizationDecision) {
	sink := activeShadowSink.Load()
	if !sink.enabled() {
		return
	}
	reason := d.Reason
	if !authorizationReasons[reason] {
		// An unclassified reason is written as the fail-closed member rather
		// than as arbitrary prose.
		reason = ReasonUnknown
	}
	sink.submit(map[string]interface{}{
		"schema_version":           shadowSchemaVersionAuthorization,
		"record_kind":              "shadow_authorization_decision",
		"request_id":               in.Identity.RequestID,
		"invocation_id":            in.Identity.InvocationID,
		"candidate_instance_id":    in.Identity.CandidateInstanceID,
		"candidate_hash":           in.CandidateHash,
		"authorized":               d.Authorized,
		"reason":                   string(reason),
		"obligations_satisfied":    d.Satisfied,
		"obligations_missing":      d.Missing,
		"evidence_consumed":        d.EvidenceConsumed,
		"settlement_required":      d.SettlementRequired,
		"influences_live_decision": false,
		"build_version":            APIVersion,
	})
}

// observeCandidateAuthorization is the one production call path: it assembles
// the input from what the delivery site already has and records what the typed
// answer would have been.
//
// The live authorization ran above it and is unaffected. This is a second,
// silent opinion.
func observeCandidateAuthorization(ctx *AgentContext, path, code string,
	id candidateEvidenceIdentity, envelope *V3EvidenceEnvelope,
	evidence []proxyEvidence) AuthorizationDecision {
	resolved := resolveAgentPath(ctx, path)
	hash := contentSHA256(code)

	// The identity evidence must bind to is built HERE, from the live request
	// and the workspace as it stands right now -- never copied off the record
	// being checked. A decision that read the asked-for identity out of the
	// evidence would be asking every record whether it matched itself, and
	// every mismatch reason would be unreachable.
	generation, stateHash := workspaceIdentity(ctx)
	asked := V3EvidenceProvenance{
		RequestID:           requestIDOf(ctx),
		InvocationID:        id.InvocationID,
		CandidateInstanceID: id.CandidateInstanceID,
		CandidateHash:       hash,
		WorkspaceGeneration: generation,
		WorkspaceStateHash:  stateHash,
		BaselineIdentity:    baselineIdentityFor(ctx, resolved),
	}
	in := authorizationInput{
		Obligations:   requestObligations(ctx),
		TargetPath:    resolved,
		CandidateHash: hash,
		Identity:      asked,
		Evidence:      evidence,
		Envelope:      envelope,
	}
	d := decideAuthorization(ctx, in)
	recordAuthorizationDecision(ctx, in, d)
	return d
}
