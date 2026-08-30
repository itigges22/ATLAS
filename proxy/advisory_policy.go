package main

import (
	"sort"
)

// What the advisory policy may look at, and what disqualifies a candidate
// outright.
//
// The split matters more than either half. A veto is a FACT about this
// candidate -- it does not parse, its oracle failed, it touched something it
// was measuring, the workspace moved underneath it. A signal is an opinion
// about quality, and every signal available here comes from the same model, or
// from a service ranking that model's own output, or from a scorer whose
// normalisation may not even be calibrated on this deployment. One veto is
// enough to refuse. No number of signals is enough to prove anything.
//
// So the vetoes decide, and the signals are recorded. Recording them is not
// hedging: a threshold over these signals cannot be chosen honestly until
// there is a measurement relating them to held-out outcomes, and the records
// this policy writes are what that measurement will be computed from.

// The closed veto vocabulary. Each is a disqualifying observation with an
// owner that is not the model: the proxy's own gates, the staging executor, the
// workspace ledger, or the permission owner.
const (
	// VetoSyntaxOrStructural: the candidate does not parse, or introduces a
	// call nothing binds.
	VetoSyntaxOrStructural = "syntax_or_structural_failure"
	// VetoExecutionUnavailable: execution was required to say anything and
	// could not happen. Distinct from a failure: nothing was observed.
	VetoExecutionUnavailable = "execution_evidence_unavailable"
	// VetoMutatedProtectedAssets: the run changed the candidate it was
	// measuring, or the workspace around it.
	VetoMutatedProtectedAssets = "candidate_mutated_protected_assets"
	// VetoLanguageOrTargetMismatch: the candidate is not the artifact class or
	// the target the request is about.
	VetoLanguageOrTargetMismatch = "language_or_target_mismatch"
	// VetoStaleIdentity: the candidate or the workspace it was observed
	// against is not the one about to be written.
	VetoStaleIdentity = "stale_candidate_or_workspace_identity"
	// VetoDeclaredVerificationFailed: a command the client required ran and
	// did not pass.
	VetoDeclaredVerificationFailed = "declared_verification_failed"
	// VetoUnauthorizedPathExpansion: the delivery target is not one the client
	// declared.
	VetoUnauthorizedPathExpansion = "unauthorized_path_expansion"
	// VetoWeakerThanBaseline: the artifact on disk already carries a stronger
	// current verdict than the candidate earned.
	VetoWeakerThanBaseline = "weaker_than_baseline_on_a_trusted_check"
	// VetoCancelledOrTimedOut: the request or the evidence run ended before it
	// could answer.
	VetoCancelledOrTimedOut = "cancelled_or_timed_out"
	// VetoIncompleteEvidence: some declared obligation has no observation at
	// all, so the evidence set does not cover what was asked.
	VetoIncompleteEvidence = "incomplete_evidence"
	// VetoDestructiveWithoutPermission: a destructive operation was implied
	// and the permission owner did not grant it. Advisory confidence is not a
	// permission.
	VetoDestructiveWithoutPermission = "destructive_operation_without_permission"
	// VetoOutsideMutationScope: the candidate is outside the boundary the
	// model's own tool call defined, or that call defined no boundary at all.
	// A scope is not evidence and never authorizes; this is the one direction
	// it acts in.
	VetoOutsideMutationScope = "outside_structured_mutation_scope"
)

var advisoryVetoNames = map[string]bool{
	VetoSyntaxOrStructural:           true,
	VetoExecutionUnavailable:         true,
	VetoMutatedProtectedAssets:       true,
	VetoLanguageOrTargetMismatch:     true,
	VetoStaleIdentity:                true,
	VetoDeclaredVerificationFailed:   true,
	VetoUnauthorizedPathExpansion:    true,
	VetoWeakerThanBaseline:           true,
	VetoCancelledOrTimedOut:          true,
	VetoIncompleteEvidence:           true,
	VetoDestructiveWithoutPermission: true,
	VetoOutsideMutationScope:         true,
}

// advisoryInput is the closed set of typed facts the policy may read.
//
// Every field is something an owner outside the model already decided. There
// is no field for a model claim, a service verdict treated as authority, or a
// hidden evaluator, and there is no free-form map a future caller could smuggle
// one through.
type advisoryInput struct {
	// Observed is the proxy's own gate verdict on the exact candidate bytes.
	Observed checkOutcome
	// TargetDeclared is whether the client stated what this request produces
	// at all. A request that declared nothing names no target to expand
	// beyond, so the path veto has nothing to be about; a request that
	// declared its outputs owns the answer.
	TargetDeclared bool
	// TargetAuthorized is whether this delivery target is one of them.
	TargetAuthorized bool
	// LanguageOrBoundaryViolation is set when a gate found the candidate is
	// not the artifact this route is about: a language swap, or a rewrite
	// past the edit.
	LanguageOrBoundaryViolation bool
	// Unmet is why each declared obligation went unsatisfied, from the
	// staging owner. The reasons are what separate "it failed" from "it never
	// ran".
	Unmet map[string]AuthorizationReason
	// Decision is the typed authorization answer over the same candidate.
	Decision AuthorizationDecision
	// Evidence is what the trusted producers observed about these bytes.
	Evidence []proxyEvidence
	// Envelope is the service's own record. Advisory only: it is read for
	// ranking signals and never for authority.
	Envelope *V3EvidenceEnvelope
	// Cancelled is whether the request itself is over.
	Cancelled bool
	// DestructivePermitted is the permission owner's answer when the route
	// implies a destructive operation, and true when it implies none.
	DestructivePermitted bool
	// DestructiveImplied says whether it does.
	DestructiveImplied bool
	// MutatedProtectedAssets is the staging owner's report that a run changed
	// the candidate it was measuring or the workspace around it. It is a
	// separate fact from a failing command and is named separately.
	MutatedProtectedAssets bool
	// ScopeAdmits is whether the model's own tool call bounds a mutation that
	// contains these bytes. False covers both "it does not" and "there is no
	// scope", and ScopeRefusal says which.
	ScopeAdmits bool
	// ScopeRefusal is the closed reason, for the record.
	ScopeRefusal string
}

// advisoryVetoes are the disqualifying facts observed about this candidate, in
// canonical order.
//
// Read the reasons, not just the outcomes. A command that could not run
// because the executor was unreachable and one that ran and failed are
// different facts, and only the second is about the candidate -- but neither
// permits a preference, so both veto.
func advisoryVetoes(in advisoryInput) []string {
	fired := map[string]bool{}

	if in.Observed.Status == ValidationFailed {
		fired[VetoSyntaxOrStructural] = true
	}
	if in.LanguageOrBoundaryViolation {
		fired[VetoLanguageOrTargetMismatch] = true
	}
	if in.TargetDeclared && !in.TargetAuthorized {
		fired[VetoUnauthorizedPathExpansion] = true
	}
	if in.Cancelled {
		fired[VetoCancelledOrTimedOut] = true
	}
	if in.DestructiveImplied && !in.DestructivePermitted {
		fired[VetoDestructiveWithoutPermission] = true
	}
	if in.MutatedProtectedAssets {
		fired[VetoMutatedProtectedAssets] = true
	}
	if !in.ScopeAdmits {
		fired[VetoOutsideMutationScope] = true
	}

	for _, why := range in.Unmet {
		switch why {
		case ReasonEvidenceExecutionFailed:
			fired[VetoDeclaredVerificationFailed] = true
		case ReasonProducerUnavailable, ReasonProducerNotRun:
			fired[VetoExecutionUnavailable] = true
		case ReasonEvidenceTimedOut, ReasonEvidenceCancelled:
			fired[VetoCancelledOrTimedOut] = true
		case ReasonEvidenceRefused:
			fired[VetoExecutionUnavailable] = true
		case ReasonEvidenceMissing:
			fired[VetoIncompleteEvidence] = true
		}
	}

	switch in.Decision.Reason {
	case ReasonBaselineNotPreserved:
		fired[VetoWeakerThanBaseline] = true
	case ReasonWorkspaceStale, ReasonCandidateMismatch, ReasonRequestOrInvocationMismatch,
		ReasonCommandMismatch:
		fired[VetoStaleIdentity] = true
	case ReasonTargetNotDeclared:
		fired[VetoUnauthorizedPathExpansion] = true
	}
	// An obligation with no observation at all is not covered, whatever else
	// was seen. The decision reports exactly which ones.
	if len(in.Decision.Missing) > 0 && in.Decision.Reason != ReasonAuthorized {
		fired[VetoIncompleteEvidence] = true
	}

	out := make([]string, 0, len(fired))
	for name := range fired {
		if advisoryVetoNames[name] {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}

// advisorySignals are the quality observations available, recorded verbatim.
//
// None of them is consulted as a threshold, and the reason is written into the
// record rather than left to memory: every one is either the same model
// grading its own output, a service ranking that output, or a scorer whose
// normalisation carries its own calibration flag. Values here describe what
// was seen. They do not describe how likely the candidate is to be correct,
// and nothing may present them as if they did.
func advisorySignals(in advisoryInput) map[string]interface{} {
	out := map[string]interface{}{
		"proxy_gate_status":     string(in.Observed.Status),
		"trusted_observations":  len(in.Evidence),
		"authorization_reason":  string(in.Decision.Reason),
		"scope_admits":          in.ScopeAdmits,
		"scope_refusal":         in.ScopeRefusal,
		"obligations_satisfied": len(in.Decision.Satisfied),
		"obligations_missing":   len(in.Decision.Missing),
	}
	strongest := ""
	for _, ev := range in.Evidence {
		if strengthRank(ev.Provenance.ObservedStrength) > strengthRank(strongest) {
			strongest = ev.Provenance.ObservedStrength
		}
	}
	out["strongest_trusted_strength"] = strongest
	if in.Envelope != nil {
		// The service's own record, labelled as the service's. A reader of
		// this map must be able to tell at a glance which side produced each
		// number, because only one side of it is trusted for anything.
		out["service_closure_eligible"] = in.Envelope.Evaluation.ClosureEligible
		out["service_evidence_strength"] = in.Envelope.Evaluation.EvidenceStrength
		out["service_execution_status"] = in.Envelope.Evaluation.ExecutionStatus
		out["service_requirements_complete"] = in.Envelope.Evaluation.RequirementsComplete
		out["service_selection_status"] = in.Envelope.Selection.Status
		out["service_tied_count"] = in.Envelope.Selection.TiedCount
		out["service_incomparable_count"] = in.Envelope.Selection.IncomparableCount
		out["service_ineligible_count"] = in.Envelope.Selection.IneligibleCount
		out["service_coverage_missing"] = len(in.Envelope.Coverage.Missing)
	}
	return out
}

// decideCandidatePolicy is THE policy owner.
//
// Order is the whole design. Vetoes first, because a disqualifying fact is not
// something a strong signal elsewhere can outweigh. Then the strict answer,
// because trusted evidence meeting a declared floor is the only thing in this
// build that authorizes a delivery. Then the two modes that do not deliver
// here: confirmation, which is a decision the user has not made yet, and
// advisory preference, which is a quality opinion this build records and does
// not act on.
//
// insufficient_confidence is a real answer, not a fallback. A candidate that
// nothing disqualified and nothing supported is exactly that, and saying so is
// how a later calibration can tell "we had no evidence" apart from "we had
// evidence against".
func decideCandidatePolicy(ctx *AgentContext, in advisoryInput,
	strictAuthorized bool) candidatePolicyOutcome {
	mode, source := candidatePolicyOf(ctx)
	out := candidatePolicyOutcome{
		Mode: mode, Source: source,
		Vetoes:  advisoryVetoes(in),
		Signals: advisorySignals(in),
	}
	if len(out.Vetoes) > 0 {
		out.Decision = PolicyCandidateRejectedHardVeto
		return out
	}
	if strictAuthorized {
		// Trusted evidence, bound to these exact bytes, meeting the floor the
		// client declared. The only decision in this build that delivers.
		out.Decision = PolicyCandidateAuthorizedStrict
		out.Delivers = true
		return out
	}
	switch mode {
	case CandidatePolicyConfirm:
		out.Decision = PolicyHumanConfirmationRequired
		return out
	case CandidatePolicyAdvisory:
		if advisoryHasPositiveEvidence(in) {
			out.Decision = PolicyCandidatePreferredAdvisory
			return out
		}
		out.Decision = PolicyInsufficientConfidence
		return out
	}
	// Strict, with nothing that authorizes. The model's own proposal stands.
	out.Decision = PolicyBaselineRetained
	return out
}

// advisoryHasPositiveEvidence reports whether anything trusted was actually
// observed in the candidate's favour.
//
// Deliberately a presence test rather than a score. The trusted producers are
// the proxy's own gate and the client's declared commands; a candidate that
// passed at least one of them, on these exact bytes, has something in its
// favour. How much that is worth is the question a calibration answers, and
// until one exists this build refuses to turn it into a number.
func advisoryHasPositiveEvidence(in advisoryInput) bool {
	if in.Observed.Status != ValidationPassed {
		return false
	}
	for _, ev := range in.Evidence {
		if ev.Outcome != ValidationPassed {
			continue
		}
		if _, trusted := provenanceCeiling[ev.Provenance.Source]; trusted {
			return true
		}
	}
	return false
}
