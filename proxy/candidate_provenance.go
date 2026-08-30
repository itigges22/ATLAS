package main

// Where the bytes on disk came from, said plainly enough to show a user.
//
// The terminal is the last place this is still knowable. Once a file is
// written, the model's own work, a candidate a declared check passed and a
// candidate someone approved by hand are all just bytes -- and they are not
// the same thing to the person reviewing the diff. A vocabulary the UI can
// render is what keeps that difference visible.
//
// None of these values is a confidence, a score or a guarantee. Each names an
// origin and the rule that let it land, and nothing more.
const (
	// DeliveryFromModelProposal: the model wrote it. No candidate replaced it,
	// either because none was proposed or because the policy kept the
	// baseline. This is the default and the overwhelming majority.
	DeliveryFromModelProposal = "model_proposal"
	// DeliveryFromStrictCandidate: a V3 candidate replaced it, and a
	// client-declared verification passed at the declared strength against
	// exactly these bytes.
	DeliveryFromStrictCandidate = "strict_trusted_candidate"
	// DeliveryFromAdvisoryCandidate: a V3 candidate replaced it under the
	// advisory policy. Bounded evidence preferred it. Nothing proved it, and
	// the UI must not present it as proven.
	DeliveryFromAdvisoryCandidate = "advisory_candidate"
	// DeliveryFromHumanApproval: a V3 candidate replaced it because a person
	// approved these exact bytes.
	DeliveryFromHumanApproval = "human_approved_candidate"
)

var deliveryProvenanceValues = map[string]bool{
	DeliveryFromModelProposal:     true,
	DeliveryFromStrictCandidate:   true,
	DeliveryFromAdvisoryCandidate: true,
	DeliveryFromHumanApproval:     true,
}

// deliveryProvenanceFor maps a policy answer to what the user is looking at.
//
// Only the decisions that actually deliver map to a candidate origin. Everything
// else -- a veto, insufficient confidence, a retained baseline, a confirmation
// nobody has given yet -- means the bytes on disk are the model's own, and
// saying anything else about them would be a false claim in the one place a
// person is relying on it.
func deliveryProvenanceFor(out candidatePolicyOutcome) string {
	if !out.Delivers {
		return DeliveryFromModelProposal
	}
	switch out.Decision {
	case PolicyCandidateAuthorizedStrict:
		return DeliveryFromStrictCandidate
	case PolicyCandidatePreferredAdvisory:
		return DeliveryFromAdvisoryCandidate
	case PolicyHumanConfirmationRequired:
		return DeliveryFromHumanApproval
	}
	return DeliveryFromModelProposal
}

// emitDeliveryProvenance tells the terminal what it is about to show, and under
// which rule.
//
// Identities and closed values only: a path the user already knows, the
// provenance, the policy decision and the mode that produced it. No candidate
// bytes and no evidence detail -- the diff the user reviews is the artifact
// itself, and this event is the label on it.
func emitDeliveryProvenance(ctx *AgentContext, path, provenance string,
	out candidatePolicyOutcome) {
	if !deliveryProvenanceValues[provenance] {
		return
	}
	Emit(NewEnvelope(EvtMetric, "candidate_policy", map[string]interface{}{
		"path":          logPath(path),
		"provenance":    provenance,
		"decision":      string(out.Decision),
		"policy_mode":   string(out.Mode),
		"policy_source": string(out.Source),
		// The vetoes are what a user would want to see when a candidate did
		// NOT land, and they are facts rather than scores.
		"vetoes": out.Vetoes,
	}))
}
