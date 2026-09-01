package main

import (
	"os"
	"strings"
)

// Which question a candidate has to answer before it may replace what the
// model wrote.
//
// ATLAS is an interactive coding agent, not a formal verifier. For an ordinary
// coding task there is no oracle to consult and no proof to be had, and a
// system that waited for one would never deliver a candidate at all. What can
// be decided honestly is narrower and still useful: whether a candidate is
// preferable to the model's own proposal on evidence that is bounded, typed,
// and bound to the exact bytes -- with the user in the loop through the
// terminal and the diff.
//
// Three modes, because there are three genuinely different situations and
// running them under one rule is what conflated "the service liked it" with
// "the client proved it":
//
//	strict     a trusted client-declared verification exists and passed at the
//	           strength the client declared. Automatic delivery.
//	advisory   no oracle exists. Bounded evidence may prefer a candidate, and
//	           the preference is a measured quality policy, never a proof.
//	automatic_v3
//	           the exact candidate the V3 selection path named, delivered when
//	           every hard safety requirement holds. No floor is invented for a
//	           request that declared none, and none the client declared is
//	           lowered.
//
// The mode is not a model output and not a service output. It comes from the
// validated client request or from trusted operator configuration, and a value
// neither of those sources produced does not exist.

type candidatePolicyMode string

const (
	// CandidatePolicyStrict is the shipped default and current behaviour: a
	// candidate lands only on trusted evidence meeting the declared floor.
	CandidatePolicyStrict candidatePolicyMode = "strict"
	// CandidatePolicyAdvisory permits preferring a candidate on bounded
	// evidence that proves nothing. It is not a default and does not deliver
	// in this build; the policy computes and records its answer, and delivery
	// still requires strict authorization.
	CandidatePolicyAdvisory candidatePolicyMode = "advisory"
	// CandidatePolicyAutomaticV3 delivers the exact candidate the V3 selection
	// path chose, whenever every hard safety requirement holds.
	//
	// It is not a lower evidence bar for the same question -- it is a
	// different question. Strict asks whether trusted evidence meets a floor
	// the client declared, which is unanswerable for a request that declared
	// nothing, and the honest answer there was to keep the baseline. But V3
	// generating K candidates and picking one IS the product: an interactive
	// coding agent whose internal competition never reaches the artifact is
	// paying for a pipeline it does not use. So the competition stays
	// internal, the user reviews the resulting diff like any other, and the
	// safety requirements that were never about evidence -- path containment,
	// mutation scope, identity freshness, syntax, permissions, exact-byte
	// delivery -- all still hold.
	CandidatePolicyAutomaticV3 candidatePolicyMode = "automatic_v3"
)

var candidatePolicyModes = map[candidatePolicyMode]bool{
	CandidatePolicyStrict:      true,
	CandidatePolicyAdvisory:    true,
	CandidatePolicyAutomaticV3: true,
}

// defaultCandidatePolicy is what a request gets when neither the client nor the
// operator said anything. Strict, because it is the behaviour every existing
// client already has, and because a default that lowered the evidence bar
// would change what lands for callers who never asked for it.
func defaultCandidatePolicy() candidatePolicyMode { return CandidatePolicyStrict }

// Where a mode came from. Recorded with every decision, because "the client
// asked for advisory" and "an operator turned it on for the whole deployment"
// are different facts about the same delivery.
type candidatePolicySource string

const (
	CandidatePolicySourceDefault  candidatePolicySource = "default"
	CandidatePolicySourceOperator candidatePolicySource = "operator_config"
	CandidatePolicySourceClient   candidatePolicySource = "client_request"
)

// ParseCandidatePolicy resolves a declared mode and fails closed.
//
// Empty is the default rather than an error: a caller that says nothing is
// asking for current behaviour. Anything else unrecognised is refused, because
// a mode nobody registered is a state nobody has reasoned about.
func ParseCandidatePolicy(raw string) (candidatePolicyMode, bool) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return defaultCandidatePolicy(), true
	}
	mode := candidatePolicyMode(trimmed)
	if !candidatePolicyModes[mode] {
		return defaultCandidatePolicy(), false
	}
	return mode, true
}

// operatorCandidatePolicy is the deployment-wide setting, or the default when
// the operator set nothing.
//
// Read from the environment the process was started with, which is operator
// territory: nothing the model emits and nothing that arrives on a request can
// reach it. An unreadable value is the default rather than an error, because
// refusing every request in the deployment over a typo in a variable is worse
// than running the behaviour every client already has.
func operatorCandidatePolicy() candidatePolicyMode {
	mode, ok := ParseCandidatePolicy(os.Getenv("ATLAS_CANDIDATE_POLICY"))
	if !ok {
		return defaultCandidatePolicy()
	}
	return mode
}

// candidatePolicyOf is THE place a request's mode is read.
//
// Precedence is client over operator over default: the caller of a single
// request knows what that request is for, and an operator setting is the
// deployment's answer for callers that did not say. Neither source is the
// model or the V3 service, and there is no path from either into this
// function.
func candidatePolicyOf(ctx *AgentContext) (candidatePolicyMode, candidatePolicySource) {
	if ctx != nil && ctx.TaskContract != nil {
		if mode, ok := ParseCandidatePolicy(ctx.TaskContract.CandidatePolicy); ok &&
			strings.TrimSpace(ctx.TaskContract.CandidatePolicy) != "" {
			return mode, CandidatePolicySourceClient
		}
	}
	if mode := operatorCandidatePolicy(); mode != defaultCandidatePolicy() {
		return mode, CandidatePolicySourceOperator
	}
	return defaultCandidatePolicy(), CandidatePolicySourceDefault
}

// --- what the policy answers -------------------------------------------------

// candidatePolicyDecision is the closed vocabulary of honest answers.
//
// Every one of them is a statement about what happened, not about how likely
// the candidate is to be correct. Nothing here is a probability, and nothing
// here may be presented as one.
type candidatePolicyDecision string

const (
	// PolicyBaselineRetained: the model's own proposal is what lands. Either
	// nothing materially different was proposed, or the candidate did not earn
	// its way past the policy.
	PolicyBaselineRetained candidatePolicyDecision = "baseline_retained"
	// PolicyCandidatePreferredAdvisory: bounded evidence prefers the candidate
	// and no hard veto fired. It is a quality preference. It is not proof, and
	// it does not by itself deliver in this build.
	PolicyCandidatePreferredAdvisory candidatePolicyDecision = "candidate_preferred_advisory"
	// PolicyCandidateAuthorizedStrict: trusted client-declared verification
	// passed at the declared strength against these exact bytes.
	PolicyCandidateAuthorizedStrict candidatePolicyDecision = "candidate_authorized_strict"
	// PolicyCandidateAutomaticV3: the V3 selection path chose this candidate
	// and every hard safety requirement holds. The competition that produced
	// it is internal; what the user reviews is the diff.
	PolicyCandidateAutomaticV3 candidatePolicyDecision = "candidate_automatic_v3"
	// PolicyCandidateRejectedHardVeto: something disqualifying was observed.
	// Vetoes are facts, never scores, and one is enough.
	PolicyCandidateRejectedHardVeto candidatePolicyDecision = "candidate_rejected_hard_veto"
	// PolicyInsufficientConfidence: no veto fired and nothing positive was
	// established either. Distinct from a veto on purpose: "nothing to go on"
	// and "something is wrong" are different, and only one of them is a fact
	// about the candidate.
	PolicyInsufficientConfidence candidatePolicyDecision = "insufficient_confidence"
)

var candidatePolicyDecisions = map[candidatePolicyDecision]bool{
	PolicyBaselineRetained:           true,
	PolicyCandidatePreferredAdvisory: true,
	PolicyCandidateAuthorizedStrict:  true,
	PolicyCandidateAutomaticV3:       true,
	PolicyCandidateRejectedHardVeto:  true,
	PolicyInsufficientConfidence:     true,
}

// candidatePolicyOutcome is one policy answer, with everything needed to say
// why it holds.
type candidatePolicyOutcome struct {
	Mode     candidatePolicyMode
	Source   candidatePolicySource
	Decision candidatePolicyDecision
	// Vetoes are the disqualifying facts observed, in canonical order. A
	// non-empty list forces the rejection decision whatever else was seen.
	Vetoes []string
	// Signals are the advisory observations that were available. They are
	// recorded so a later calibration can be computed from what actually
	// happened; none of them is consulted as a threshold here.
	Signals map[string]interface{}
	// Delivers reports whether this outcome is one the delivery path may act
	// on. Only the strict authorization is, in this build.
	Delivers bool
}

// mayDeliverUnderPolicy is the single predicate the delivery path asks.
//
// Advisory preference is computed and recorded and does not deliver: it needs
// a calibrated threshold and a measured regression rate before it may change
// what lands. Recording the answer without acting on it is what makes the
// calibration experiment possible without shipping an unmeasured policy.
// Strict and automatic_v3 are the two that deliver, and each sets Delivers in
// exactly one place.
func (o candidatePolicyOutcome) mayDeliverUnderPolicy() bool { return o.Delivers }

// recordCandidatePolicyDecision writes one policy answer to the private shadow
// sink.
//
// Identities, a closed decision, a closed veto list and the advisory signals as
// they were observed. No candidate byte, no command string, no path content and
// no prose. influences_live_decision is a fact about this build rather than an
// intention: only the strict authorization delivers here, and every other
// decision is recorded so a calibration can be computed from what actually
// happened rather than from what a threshold would have predicted.
func recordCandidatePolicyDecision(ctx *AgentContext, entry routeEntry,
	candidateHash string, out candidatePolicyOutcome) {
	sink := activeShadowSink.Load()
	if !sink.enabled() {
		return
	}
	decision := out.Decision
	if !candidatePolicyDecisions[decision] {
		// An unclassified decision is written as the fail-closed member rather
		// than as arbitrary prose.
		decision = PolicyBaselineRetained
	}
	vetoes := make([]string, 0, len(out.Vetoes))
	for _, v := range out.Vetoes {
		if advisoryVetoNames[v] {
			vetoes = append(vetoes, v)
		}
	}
	sink.submit(map[string]interface{}{
		"schema_version":           shadowSchemaVersionCandidatePolicy,
		"record_kind":              "candidate_policy_decision",
		"request_id":               requestIDOf(ctx),
		"route_entry_id":           entry.ID,
		"candidate_hash":           candidateHash,
		"policy_mode":              string(out.Mode),
		"policy_source":            string(out.Source),
		"decision":                 string(decision),
		"vetoes":                   vetoes,
		"signals":                  out.Signals,
		"delivers":                 out.Delivers,
		"influences_live_decision": out.Delivers,
		"build_version":            APIVersion,
	})
}
