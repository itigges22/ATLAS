package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"path/filepath"
)

// Trusted candidate delivery for the edit routes.
//
// Four tools could land a service-authored replacement on an existing file
// with no route identity, no candidate evidence, no proxy authorization, no
// one-time grant, no exact-byte consumption and no settlement. The only gate
// was the service's own envelope -- which is a proposal, not authority.
//
// This route answers the same questions the new-file route answers, in the
// same order, through the same owner. What it does NOT do is change when
// candidates are generated: feasibility stays observe-only here, so adding
// protection cannot quietly start skipping work.

// editRouteOutcome is what the originating tool needs to know.
//
// Delivered means the candidate landed and Result is that tool's own result:
// the caller must not write again. Otherwise Content is the bytes the caller
// should write itself, exactly as it did before.
type editRouteOutcome struct {
	Delivered bool
	Result    *ToolResult
	Content   string
	Meta      V3EditMetadata
	Cancelled *ToolResult
}

// deliverEditCandidate runs one edit attempt through the protected chain.
//
// `edited` is the post-edit full file the tool computed. It is the caller's
// own proposal for this attempt, and it is what stays on the floor whenever
// the candidate does not earn its way past authorization.
func deliverEditCandidate(ctx *AgentContext, tool, path, relPath,
	original, edited string) editRouteOutcome {
	keep := editRouteOutcome{Content: edited}
	if ctx == nil {
		return keep
	}

	// One identity for this attempt, minted before anything is proposed, and
	// one ending recorded for it whatever happens below.
	entry := mintRouteEntry(ctx)
	lifecycle := newRouteLifecycle(entry)
	defer lifecycle.finalizeDefault(ctx)

	// The structured intent of THIS call: the tool the model actually invoked,
	// the canonical target it named, the pre-edit artifact and the caller's own
	// post-edit result. The difference between the last two is the boundary a
	// candidate may not leave.
	scope, scopeOK := deriveMutationScope(ctx, entry, tool, path, original, edited)

	// Observe-only. The answer is recorded for this route and never consulted:
	// generation happens exactly as often as it did before, and any future
	// compute saving here needs its own measurement and its own authorization.
	observeInvocationFeasibility(ctx, entry)

	// The same additive stage envelope the new-file route emits, carrying this
	// entry. The edit route genuinely starts a V3 stage, and an observer
	// watching the event stream could otherwise see a decision that said
	// generation proceeded with nothing on the stream to match it.
	Emit(NewEnvelope(EvtStageStart, "v3", map[string]interface{}{
		"detail":      "file=" + filepath.Base(path),
		"route_entry": entry.ID,
	}))

	improved, meta, err := improveContentWithV3(path, edited, ctx)
	if err != nil {
		if errors.Is(err, context.Canceled) || (ctx.Ctx != nil && ctx.Ctx.Err() != nil) {
			lifecycle.finish(ctx, routingCancelled, "", "")
			keep.Cancelled = &ToolResult{Success: false,
				Error: tool + " cancelled — no content was written"}
			return keep
		}
		if errors.Is(err, context.DeadlineExceeded) {
			lifecycle.finish(ctx, routingProducerTimedOut, "", "")
		} else {
			lifecycle.finish(ctx, routingProducerUnavailable, "", "")
		}
		log.Printf("[%s] V3 unavailable: %v — keeping the caller's content", tool, err)
		return keep
	}

	// The existing boundary guards, unchanged and still ahead of staging: a
	// winner that rewrote past the edit or swapped the language is not a
	// candidate for this attempt and never becomes one.
	switch {
	case improved == "":
		lifecycle.finish(ctx, routingNoCandidate, "", "")
		return keep
	case improved == edited:
		// Nothing materially different was proposed. The caller's own bytes
		// are what land, through the caller's own path, with no licence.
		lifecycle.finish(ctx, routingBaselineRetained, "", "")
		return keep
	case v3RewroteBeyondTheEdit(original, edited, improved) != "":
		log.Printf("[%s] V3 rewrote beyond the edit — keeping the caller's content", tool)
		lifecycle.finish(ctx, routingRevokedByGate, "", "")
		return keep
	case v3SwappedTheLanguage(relPath, edited, improved) != "":
		log.Printf("[%s] V3 swapped the language — keeping the caller's content", tool)
		lifecycle.finish(ctx, routingRevokedByGate, "", "")
		return keep
	}

	// A materially different proposal. From here it is a candidate, and it
	// answers every question a new-file candidate answers.
	observed := fallbackSyntaxOutcomeFor(ctx, path, improved).aggregate()
	if observed.Status == ValidationFailed {
		log.Printf("[%s] the candidate does not parse — keeping the caller's content", tool)
		lifecycle.finish(ctx, routingRevokedByGate, contentSHA256(improved), "")
		return keep
	}

	evidence, evID, seen := observeDeliveredCandidateSyntax(ctx, entry, path, improved, observed)
	var pool []proxyEvidence
	var unmet map[string]AuthorizationReason
	mutatedAssets := false
	if seen {
		pool = []proxyEvidence{evidence}
		behavioral, why, mutated := observeCandidateVerification(ctx, path, improved, evID)
		pool, unmet, mutatedAssets = append(pool, behavioral...), why, mutated
	}

	scopeAdmits, scopeRefusal := false, scopeRefusedNoScope
	if scopeOK {
		scopeAdmits, scopeRefusal = scopeAdmitsCandidate(ctx, scope, improved)
	}
	recordMutationScope(ctx, scope, scopeAdmits, scopeRefusal)

	delivery := authorizeCandidateDelivery(ctx, entry, path, improved, evID,
		meta.Envelope, pool, "", unmet, observed, scope)
	// The same policy owner the new-file route asks, over the same kinds of
	// fact. An edit route that decided this for itself is how the two paths
	// disagreed about what a candidate had to show before it could land.
	policy := decideCandidatePolicy(ctx, advisoryInput{
		Observed:               observed,
		TargetDeclared:         outputKnowledgeDeclared(ctx),
		TargetAuthorized:       targetIsAuthorized(requestObligations(ctx), resolveAgentPath(ctx, path)),
		Unmet:                  unmet,
		Decision:               delivery.Decision,
		Evidence:               pool,
		Envelope:               meta.Envelope,
		Cancelled:              ctx.Ctx != nil && ctx.Ctx.Err() != nil,
		MutatedProtectedAssets: mutatedAssets,
		ScopeAdmits:            scopeAdmits,
		ScopeRefusal:           scopeRefusal,
	}, delivery.Typed && delivery.mayDeliver())
	recordCandidatePolicyDecision(ctx, entry, contentSHA256(improved), policy)
	if !policy.mayDeliverUnderPolicy() {
		// An honest refusal. The model's own edit is the alternative, and
		// nothing has been written yet, so the caller simply proceeds.
		lifecycle.finish(ctx, routingAuthorizationRefused, contentSHA256(improved),
			AuthorizationReason(delivery.Refusal))
		log.Printf("[%s] the policy refused the candidate (%s) — keeping the caller's content",
			tool, policy.Decision)
		emitDeliveryProvenance(ctx, path, DeliveryFromModelProposal, policy)
		return keep
	}

	lifecycle.finish(ctx, routingCandidateAuthorized, contentSHA256(improved), "")
	result, out, derr := deliverCandidateBytes(ctx, candidateDeliveryRequest{
		Tool: tool, Path: path, Code: improved, Grant: delivery.Grant,
		Observed: observed, MetCommands: delivery.MetCommands,
		BaselinePreserved: delivery.BaselinePreserved,
		Write:             editDeliveryWriter(tool),
	})
	if derr != nil && result == nil {
		// Refused before any byte moved: nothing mutated, and the caller's own
		// edit is still the alternative.
		log.Printf("[%s] the candidate was not spendable (%s) — keeping the caller's content",
			tool, out.Reason)
		return keep
	}
	// From here disk has been touched by the delivery. The caller must never
	// write its own edit afterwards: restoration and settlement own what
	// happens next, and a second write would undo a decision they made.
	keep.Delivered = true
	provenance := deliveryProvenanceFor(policy)
	emitDeliveryProvenance(ctx, path, provenance, policy)
	keep.Result, keep.Meta = withDeliveryProvenance(result, provenance), meta
	if derr != nil {
		keep.Result = result
	}
	return keep
}

// editResultPayload is the originating tool's own result body.
//
// A delivery through an edit tool must not answer with a write_file payload:
// the agent loop pairs a tool call with its result and accounts mutation debt
// from it, and a mismatched body describes a call that never happened. A tool
// with no distinct body keeps the write path's.
func editResultPayload(tool, code string) []byte {
	switch tool {
	case "edit_file", "insert_after", "replace_lines":
		body, err := json.Marshal(EditFileOutput{OK: true})
		if err != nil {
			return nil
		}
		return body
	case "structural_edit":
		body, err := json.Marshal(StructuralEditOutput{OK: true, BytesNew: len(code)})
		if err != nil {
			return nil
		}
		return body
	}
	return nil
}

// editDeliveryWriter is the originating tool's own write.
//
// The exact-byte write, the post-write hash check, settlement and restoration
// all stay with the one owner; only the result shape is the tool's, so an edit
// answers with an edit result and the loop's tool-call accounting and mutation
// debt describe the call that actually happened.
func editDeliveryWriter(tool string) func(string, string, *AgentContext) (*ToolResult, error) {
	return func(path, code string, ctx *AgentContext) (*ToolResult, error) {
		res, err := writeFileRecorded(path, code, ctx)
		if err != nil || res == nil {
			return res, err
		}
		if payload := editResultPayload(tool, code); payload != nil {
			res.Data = payload
		}
		return res, nil
	}
}
