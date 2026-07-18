package main

import (
	"fmt"
	"path/filepath"
	"strings"
)

// Repository Planning Graph (RPG) integration — proxy side (V3.2, issue #120).
//
// The RPG planner (v3-service/rpg.py) returns a Plan whose write_file steps each
// carry the projected RPG node's planned interface in PlanStep.Constraints. This
// file threads those constraints into the per-node /v3/generate call so node
// generation stays on the planned signatures / data-flow edges — realizing the
// "RPG plans the repo, PlanSearch fills each node" composition. When the flat
// planner ran (ATLAS_RPG_PLANNING off) steps carry no constraints and these
// helpers return nil, leaving generation behavior unchanged.

// fileProducingActions are the plan-step actions that create or rewrite a file
// and therefore route through V3 generation.
var fileProducingActions = []string{"write_file", "ast_edit", "edit_file"}

// planConstraintsForTarget returns the RPG node constraints attached to the
// plan step that targets `path`, or nil when there is no plan, no matching
// step, or the step carries no constraints (the flat-planner case).
func planConstraintsForTarget(ctx *AgentContext, path string) []string {
	if ctx == nil || ctx.Plan == nil {
		return nil
	}
	// Trust gate: constraints are honored only when the plan carries the RPG
	// graph artifact the RPG planner always attaches. The flat planner returns
	// the LLM's plan dict as-is, so a model that hallucinates a "constraints"
	// array on a step must not steer generation while ATLAS_RPG_PLANNING is
	// off — without this check that leak was the only flag gating proxy-side.
	if ctx.Plan.RPG == nil {
		return nil
	}
	// Pick the most specific overlapping step rather than the first. With
	// suffix-based matching a bare-basename step (e.g. target "user.py") would
	// otherwise shadow the deeper "models/user.py" step it isn't really for, so
	// prefer the longest matching Target. Ranked over ALL file-producing steps
	// (not only constraint-bearing ones) so a deeper unconstrained step wins
	// over a shallower constrained one instead of being shadowed by it.
	best := -1
	bestLen := -1
	for i := range ctx.Plan.Steps {
		step := &ctx.Plan.Steps[i]
		if !isFileProducingAction(step.Action) {
			continue
		}
		if targetsOverlap(step.Target, path) && len(step.Target) > bestLen {
			best = i
			bestLen = len(step.Target)
		}
	}
	if best < 0 || len(ctx.Plan.Steps[best].Constraints) == 0 {
		return nil
	}
	// Copy so callers can append without mutating the plan.
	out := make([]string, len(ctx.Plan.Steps[best].Constraints))
	copy(out, ctx.Plan.Steps[best].Constraints)
	return out
}

func isFileProducingAction(action string) bool {
	for _, a := range fileProducingActions {
		if actionMatchesTool(a, action) || actionMatchesTool(action, a) {
			return true
		}
	}
	return false
}

// ── Drift / re-plan loop (V3.2 Phase 3, issue #120) ──────────

// rpgFileIDForPath returns the RPG node id whose file path matches `path`
// (path-suffix aware), or "" when there is no graph or no match.
func rpgFileIDForPath(plan *Plan, path string) string {
	if plan == nil || plan.RPG == nil {
		return ""
	}
	for _, f := range plan.RPG.Files {
		if targetsOverlap(f.Path, path) {
			return f.ID
		}
	}
	return ""
}

// affectedDownstream returns the file paths of the RPG nodes reachable
// downstream from `fileID` via data-flow edges (consumers, transitively).
// These are the nodes whose generation depended on the drifted one and may
// need re-planning / regeneration. Deterministic order, no duplicates.
func affectedDownstream(plan *Plan, fileID string) []string {
	if plan == nil || plan.RPG == nil || fileID == "" {
		return nil
	}
	adj := map[string][]string{}
	for _, e := range plan.RPG.Edges {
		// Data-flow edges only, as documented: "order" edges express build
		// sequencing, not dependence on the drifted interface. Empty kind
		// defaults to data_flow, matching rpg.py's serialization default.
		if e.Kind != "" && e.Kind != "data_flow" {
			continue
		}
		adj[e.From] = append(adj[e.From], e.To)
	}
	pathByID := map[string]string{}
	for _, f := range plan.RPG.Files {
		pathByID[f.ID] = f.Path
	}
	seen := map[string]bool{fileID: true}
	var out []string
	queue := append([]string{}, adj[fileID]...)
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		if seen[id] {
			continue
		}
		seen[id] = true
		if p, ok := pathByID[id]; ok {
			out = append(out, p)
		}
		queue = append(queue, adj[id]...)
	}
	return out
}

// regenerateOnDrift performs ONE corrective regeneration when a V3 winning
// candidate failed to realize its planned RPG signatures. It re-runs generation
// with the missing signatures injected as a hard constraint and keeps the retry
// only if it realizes strictly more of the plan; otherwise the original result
// stands. Bounded to a single retry — the V3 pipeline is expensive, so this is
// best-effort node-local self-repair, not a loop. No-op unless the request
// carried RPG constraints and the previous result actually drifted.
func regenerateOnDrift(ctx *AgentContext, req V3GenerateRequest, prev *V3GenerateResponse) *V3GenerateResponse {
	if ctx == nil || prev == nil || len(prev.RPGSignatureMissing) == 0 || len(req.Constraints) == 0 {
		return prev
	}

	retryReq := req
	corrective := make([]string, 0, len(req.Constraints)+1)
	corrective = append(corrective, req.Constraints...)
	corrective = append(corrective,
		"REQUIRED — the previous attempt omitted these. Define EXACTLY these "+
			"signatures: "+strings.Join(prev.RPGSignatureMissing, "; "))
	retryReq.Constraints = corrective

	if ctx.StreamFn != nil {
		ctx.StreamFn("rpg_regen", map[string]interface{}{
			"message": fmt.Sprintf("RPG drift on %s — regenerating once to realize: %v",
				filepath.Base(req.FilePath), prev.RPGSignatureMissing),
			"file":    req.FilePath,
			"missing": prev.RPGSignatureMissing,
		})
	}
	Emit(NewEnvelope(EvtStageStart, "rpg_regen", map[string]interface{}{
		"file":    req.FilePath,
		"missing": prev.RPGSignatureMissing,
	}))

	retried, err := callV3GenerateStreaming(ctx.Ctx, ctx.V3URL, retryReq,
		func(stage, detail string, data map[string]interface{}) {
			if ctx.StreamFn != nil && stage != "token" {
				ctx.StreamFn("v3_progress", map[string]string{
					"message": fmt.Sprintf("  │ [regen:%s] %s", stage, detail),
				})
			}
		})
	accepted := false
	defer func() {
		Emit(NewEnvelope(EvtStageEnd, "rpg_regen", map[string]interface{}{
			"file":     req.FilePath,
			"accepted": accepted,
		}))
	}()
	if err != nil || retried == nil || retried.Code == "" {
		return prev
	}
	// Accept the retry when it resolves at least one of the signatures we were
	// correcting for — even a trade for a different miss, as long as the TOTAL
	// drift did not grow. (Resolving one targeted signature while introducing
	// two new misses is a net loss; without the total ceiling the corrective's
	// tunnel-vision prompt could adopt a strictly worse result.) Falling back
	// to total count covers the case where the targeted set is unchanged but
	// the retry still realized more of the plan overall.
	stillMissingTargeted := countOverlap(retried.RPGSignatureMissing, prev.RPGSignatureMissing)
	if stillMissingTargeted < len(prev.RPGSignatureMissing) &&
		len(retried.RPGSignatureMissing) <= len(prev.RPGSignatureMissing) {
		accepted = true
		return retried
	}
	if len(retried.RPGSignatureMissing) < len(prev.RPGSignatureMissing) {
		accepted = true
		return retried
	}
	return prev
}

// countOverlap returns how many DISTINCT elements of `subset` appear in `of`.
// Distinct, because the acceptance rule compares it against len(of the
// targeted set): duplicate entries in a service response must not count a
// resolved signature as still-missing twice.
func countOverlap(subset, of []string) int {
	set := make(map[string]struct{}, len(of))
	for _, s := range of {
		set[s] = struct{}{}
	}
	seen := make(map[string]struct{}, len(subset))
	n := 0
	for _, s := range subset {
		if _, dup := seen[s]; dup {
			continue
		}
		seen[s] = struct{}{}
		if _, ok := set[s]; ok {
			n++
		}
	}
	return n
}

// reportRPGDrift surfaces structural drift after a V3 write: the winning code
// for `path` did not realize the planned signatures `missing`. It emits a drift
// event naming the node and the downstream subgraph that depended on it, so the
// agent loop / user can react. (Automatic regeneration of the subgraph rides the
// continuing agent loop; this is the detection + impact-surfacing half.)
func reportRPGDrift(ctx *AgentContext, path string, missing []string) {
	if ctx == nil || len(missing) == 0 {
		return
	}
	fileID := rpgFileIDForPath(ctx.Plan, path)
	downstream := affectedDownstream(ctx.Plan, fileID)

	// Point event, not a stage — EvtMetric doesn't leave an unclosed
	// stage open in the pipeline pane.
	Emit(NewEnvelope(EvtMetric, "rpg_drift", map[string]interface{}{
		"file":       path,
		"missing":    missing,
		"downstream": downstream,
	}))
	if ctx.StreamFn != nil {
		msg := fmt.Sprintf("RPG drift: %s did not realize planned signature(s) %v", filepath.Base(path), missing)
		if len(downstream) > 0 {
			msg += fmt.Sprintf("; downstream nodes may need regeneration: %v", downstream)
		}
		ctx.StreamFn("rpg_drift", map[string]interface{}{
			"message":    msg,
			"file":       path,
			"missing":    missing,
			"downstream": downstream,
		})
	}
}
