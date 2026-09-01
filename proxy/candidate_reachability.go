package main

// Why the candidate producer was not consulted for a mutation the model
// actually made.
//
// Every route that CAN carry a candidate says how it ended: routeLifecycle
// mints an identity before anything is decided and records one disposition
// whatever happens. But a route entry is minted INSIDE the branch that
// decided to generate, so a mutation the activation predicate turned away
// never becomes a route at all -- and therefore never says anything.
//
// The corrected eligibility pilot ran into exactly that wall. Ten of its
// twenty-four families produced no candidate proposal, and for nine of them
// production had written not one record: no entry, no feasibility answer, no
// disposition. Reading those nine apart took re-implementing classifyFileTier
// and editWarrantsV3 in an audit script and evaluating them against files
// that happened to survive the run, which answers what the code MUST have
// done rather than what it did -- and two families could not be resolved even
// that way. An activation predicate that has to be reconstructed to be seen
// is not instrumented.
//
// So the skips speak for themselves. This changes nothing about which
// mutations reach the producer: each owner below returns the same answer the
// inline condition it replaces returned, in the same order, and the record is
// written after the decision it describes.

type candidateBypassReason string

const (
	// bypassNone is not a skip: the producer is consulted.
	bypassNone candidateBypassReason = ""
	// The file is too small or too simple for the pipeline to be worth its
	// minutes. A cost rule, and the dominant one in the pilot.
	bypassTierBelowThreshold candidateBypassReason = "file_tier_below_threshold"
	// The edit route's second cost rule: a surgical edit on a file under the
	// complexity and size floor is already what was asked for.
	bypassEditBelowComplexityFloor candidateBypassReason = "edit_below_complexity_floor"
	// No producer is configured for this session.
	bypassProducerNotConfigured candidateBypassReason = "producer_not_configured"
	// A producer exists but this session's mode does not permit generation.
	bypassGenerationDisabled candidateBypassReason = "generation_disabled"
	// The edit-test-fix fast path: the session is iterating on a file it just
	// watched fail, and execution is the feedback.
	bypassActiveDebugIteration candidateBypassReason = "active_debug_iteration"
	// A syntax or structural guard answered before the producer could be
	// asked, so no candidate was generated for these bytes.
	bypassProposalFailedSyntaxGuard candidateBypassReason = "proposal_failed_syntax_guard"
	// bypassUnclassified is the fail-closed member. A skip nobody taught this
	// vocabulary about ends here rather than ending silently, and it may never
	// be read as an expected one.
	bypassUnclassified candidateBypassReason = "internal_unclassified"
)

func knownCandidateBypassReason(r candidateBypassReason) bool {
	switch r {
	case bypassTierBelowThreshold, bypassEditBelowComplexityFloor,
		bypassProducerNotConfigured, bypassGenerationDisabled,
		bypassActiveDebugIteration, bypassProposalFailedSyntaxGuard,
		bypassUnclassified:
		return true
	}
	return false
}

// writeGenerationBypass is THE answer to whether the new-file route consults
// the producer, and why not when it does not.
//
// Exactly the condition it replaces:
//
//	fileTier >= Tier2Medium && ctx.V3URL != "" && ctx.V3GenerationEnabled() && !iterating
//
// negated one clause at a time, in the order the `&&` chain evaluated them, so
// the first reason a caller would have failed on is the reason it reports.
// generationPermitted is passed in rather than read here, and the dispatch
// site names it. The owner still owns the ORDER and the reasons; what the site
// owns is the visible dependency, so a reader looking at the call that reaches
// the producer can see that disabling generation reaches it -- without having
// to follow a call to find out.
func writeGenerationBypass(ctx *AgentContext, fileTier Tier, iterating bool,
	generationPermitted bool) candidateBypassReason {
	switch {
	case fileTier < Tier2Medium:
		return bypassTierBelowThreshold
	case ctx == nil || ctx.V3URL == "":
		return bypassProducerNotConfigured
	case !generationPermitted:
		return bypassGenerationDisabled
	case iterating:
		return bypassActiveDebugIteration
	}
	return bypassNone
}

// editGenerationBypass is the same answer for the four edit tools.
//
// Exactly the conditions it replaces, in their original order:
//
//	fileTier < Tier2Medium || !editWarrantsV3(...) || ctx.V3URL == "" || !ctx.V3GenerationEnabled()
//	isActiveDebugIteration(ctx, relPath)
func editGenerationBypass(ctx *AgentContext, fileTier Tier, warrants bool,
	iterating bool, generationPermitted bool) candidateBypassReason {
	switch {
	case fileTier < Tier2Medium:
		return bypassTierBelowThreshold
	case !warrants:
		return bypassEditBelowComplexityFloor
	case ctx == nil || ctx.V3URL == "":
		return bypassProducerNotConfigured
	case !generationPermitted:
		return bypassGenerationDisabled
	case iterating:
		return bypassActiveDebugIteration
	}
	return bypassNone
}

// recordCandidateGenerationBypass writes one skip.
//
// Path-free, like every other capture record: the route entry a consulted
// mutation would have minted is what a reader joins on, and a skipped one has
// none, so this carries the request, the tool and the predicate inputs that
// decided it. Those inputs are the point -- a count of skips that does not say
// which threshold produced them cannot tell a cost rule from an outage.
func recordCandidateGenerationBypass(ctx *AgentContext, tool string,
	reason candidateBypassReason, fileTier Tier, contentLines int) {
	if reason == bypassNone {
		return
	}
	if !knownCandidateBypassReason(reason) {
		reason = bypassUnclassified
	}
	sink := activeShadowSink.Load()
	if !sink.enabled() {
		return
	}
	sink.submit(map[string]interface{}{
		"schema_version": shadowSchemaVersionGenerationBypass,
		"record_kind":    "candidate_generation_bypass",
		"request_id":     requestIDOf(ctx),
		"tool":           tool,
		"reason":         string(reason),
		"file_tier":      int(fileTier),
		"content_lines":  contentLines,
		// A fact about telemetry, not about policy: this record is written
		// after the decision it describes and changes nothing.
		"influences_live_decision": false,
		"build_version":            APIVersion,
	})
}
