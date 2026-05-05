package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
)

// ---------------------------------------------------------------------------
// Plan adherence — track tool calls against the pre-flight plan
// ---------------------------------------------------------------------------
//
// Adherence is advisory by default: we record which planned steps a tool
// call satisfies and emit metric events, but we don't block the model.
// Hard-blocking off-plan calls would be brittle when the plan was
// suboptimal — the model often discovers correct work the planner
// missed.
//
// What we DO actively do: count the off-plan streak, and once it
// crosses planAutoReviseThreshold we regenerate the plan with whatever
// context the agent has discovered so far. That's the "plan_revise
// escape" — the agent doesn't have to know about a plan_revise tool;
// the loop notices the divergence and re-plans for it.
//
// Adherence rules:
//   - A tool call satisfies the FIRST unsatisfied plan step whose
//     action verb matches the tool name (read_file ↔ "read_file" or
//     "read", run_command ↔ "run_command" or "run").
//   - If the planned step has a target, we additionally require the
//     tool's path/command to mention that target. Loose substring
//     match — paths normalize to basename so /workspace/app.py and
//     ./app.py both match a step targeting "app.py".
//   - Steps are matched in order (first unsatisfied wins) so the
//     model can revisit a planned action without re-satisfying earlier
//     steps. Out-of-order is fine; off-plan is what we count.

const (
	// planAutoReviseThreshold is the number of consecutive off-plan
	// tool calls before we auto-revise the plan. Tuned to be generous
	// enough that one or two exploratory off-plan calls don't trigger
	// thrashing, but tight enough that a fundamentally wrong plan
	// gets re-derived before the agent burns its turn budget.
	planAutoReviseThreshold = 3

	// planMaxRevisions caps how many times we'll regenerate per loop.
	// After this we give up and run plan-free for the remainder.
	planMaxRevisions = 2
)

// matchPlanStep returns the index of the first unsatisfied plan step
// that the tool call (toolName, args) satisfies, or -1 if no match.
// satisfied must be the same length as plan.Steps.
func matchPlanStep(plan *Plan, satisfied []bool, toolName string, args json.RawMessage) int {
	if plan == nil || len(plan.Steps) == 0 {
		return -1
	}
	if len(satisfied) != len(plan.Steps) {
		return -1
	}
	target := extractToolTarget(toolName, args)
	for i, step := range plan.Steps {
		if satisfied[i] {
			continue
		}
		if !actionMatchesTool(step.Action, toolName) {
			continue
		}
		// Target match is advisory — if the step has no target field
		// or the tool args don't carry an obvious target, the action
		// match alone is enough.
		if step.Target != "" && target != "" {
			if !targetsOverlap(step.Target, target) {
				continue
			}
		}
		return i
	}
	return -1
}

// actionMatchesTool reports whether step.Action describes the same
// operation as a tool call named toolName. We check both directions
// (action→tool and tool→action) and normalize underscores so plans
// written as "read file" or "read_file" both match read_file.
func actionMatchesTool(action, toolName string) bool {
	if action == "" || toolName == "" {
		return false
	}
	a := strings.ToLower(strings.ReplaceAll(action, "_", " "))
	t := strings.ToLower(strings.ReplaceAll(toolName, "_", " "))
	if a == t || strings.Contains(a, t) {
		return true
	}
	// Also allow the verb stem ("read" matches "read_file" tool).
	verb := strings.SplitN(t, " ", 2)[0]
	if verb != "" && strings.HasPrefix(a, verb) {
		return true
	}
	return false
}

// targetsOverlap reports whether two paths/targets refer to the same
// thing. For paths: equality or path-suffix match (so
// "templates/index.html" matches "/workspace/templates/index.html").
// For commands (anything with a space or non-path char): loose
// substring match so "curl http://localhost:5000/" matches a plan
// target of "curl http://localhost:5000/hello".
//
// Old version did unconditional substring match, which made "app.py"
// erroneously match "tests/test_app.py" — so reads of the test file
// would tick off the source-file plan step. Tightened to require a
// path-component boundary for path-shaped strings.
func targetsOverlap(planTarget, toolTarget string) bool {
	a := strings.ToLower(strings.TrimSpace(planTarget))
	b := strings.ToLower(strings.TrimSpace(toolTarget))
	if a == "" || b == "" {
		return false
	}
	if a == b {
		return true
	}
	a = strings.TrimPrefix(a, "./")
	b = strings.TrimPrefix(b, "./")
	if a == b {
		return true
	}
	// Path-suffix match: basename or last-N-components alignment.
	if strings.HasSuffix(b, "/"+a) || strings.HasSuffix(a, "/"+b) {
		return true
	}
	// Heuristic: anything with a space looks like a command rather
	// than a filename. Allow substring there so partial command
	// matches still count.
	if strings.ContainsAny(a, " \t") || strings.ContainsAny(b, " \t") {
		return strings.Contains(a, b) || strings.Contains(b, a)
	}
	return false
}

// extractToolTarget returns the most useful "target" string for a
// tool call: file path for file tools, command string for run_command,
// path for list_directory. Empty when the tool has no clear target
// (e.g. plan_revise itself).
func extractToolTarget(toolName string, args json.RawMessage) string {
	switch toolName {
	case "read_file", "delete_file":
		var x struct {
			Path string `json:"path"`
		}
		if json.Unmarshal(args, &x) == nil {
			return x.Path
		}
	case "write_file":
		var x WriteFileInput
		if json.Unmarshal(args, &x) == nil {
			return x.Path
		}
	case "edit_file":
		var x struct {
			Path string `json:"path"`
		}
		if json.Unmarshal(args, &x) == nil {
			return x.Path
		}
	case "run_command":
		var x RunCommandInput
		if json.Unmarshal(args, &x) == nil {
			return x.Command
		}
	case "list_directory":
		var x struct {
			Path string `json:"path"`
		}
		if json.Unmarshal(args, &x) == nil {
			return x.Path
		}
	}
	return ""
}

// recordPlanAdherence is called from the agent loop after each
// tool-call dispatch. It updates ctx.PlanStepsSatisfied and
// ctx.PlanOffStreak, emits a "plan_adherence" metric, and returns
// true if the off-streak crossed the auto-revise threshold (caller
// should regenerate the plan).
func recordPlanAdherence(ctx *AgentContext, toolName string, args json.RawMessage, success bool) bool {
	if ctx.Plan == nil {
		return false
	}
	if ctx.PlanStepsSatisfied == nil {
		ctx.PlanStepsSatisfied = make([]bool, len(ctx.Plan.Steps))
	}

	idx := matchPlanStep(ctx.Plan, ctx.PlanStepsSatisfied, toolName, args)

	// Only successful tool calls count toward step satisfaction.
	// A failed run_command shouldn't tick off the verify_step.
	if idx >= 0 && success {
		ctx.PlanStepsSatisfied[idx] = true
		ctx.PlanOffStreak = 0
		ctx.Stream("plan_adherence", map[string]interface{}{
			"matched":     true,
			"step_index":  idx,
			"step_id":     ctx.Plan.Steps[idx].ID,
			"step_action": ctx.Plan.Steps[idx].Action,
			"satisfied":   countTrue(ctx.PlanStepsSatisfied),
			"total":       len(ctx.PlanStepsSatisfied),
		})
		return false
	}

	// No matching step (or the call failed) — extend the off-streak.
	ctx.PlanOffStreak++
	ctx.Stream("plan_adherence", map[string]interface{}{
		"matched":    false,
		"tool":       toolName,
		"off_streak": ctx.PlanOffStreak,
		"satisfied":  countTrue(ctx.PlanStepsSatisfied),
		"total":      len(ctx.PlanStepsSatisfied),
	})

	// Threshold check — caller should auto-revise when this returns
	// true. We also cap at planMaxRevisions so a chronically
	// off-plan run doesn't loop forever calling /v3/plan.
	if ctx.PlanOffStreak >= planAutoReviseThreshold && ctx.PlanRevisions < planMaxRevisions {
		return true
	}
	return false
}

// revisePlan regenerates the plan with whatever the agent has
// discovered since the original plan was made. The user message
// passed in is the ORIGINAL one (the goal hasn't changed); we
// suffix a short note explaining why we're re-planning so the
// planner can adjust shape.
func revisePlan(ctx *AgentContext, originalUserMessage string, reason string) {
	if ctx.Plan == nil || ctx.PlanRevisions >= planMaxRevisions {
		return
	}
	// Compose a revision-aware user message. The planner prompt is
	// goal-oriented, so we keep the user's original goal verbatim
	// and append a "what we learned" note. This lets the planner
	// re-shape the plan around the new info rather than starting
	// from zero.
	noted := originalUserMessage
	if reason != "" {
		noted = fmt.Sprintf("%s\n\n[Re-planning context: %s]", originalUserMessage, reason)
	}
	log.Printf("[agent] revising plan (revision %d/%d): %s",
		ctx.PlanRevisions+1, planMaxRevisions, reason)
	ctx.Stream("plan_revise", map[string]interface{}{
		"reason":   reason,
		"revision": ctx.PlanRevisions + 1,
	})

	// Carry forward what the agent has read so far — it's the most
	// concrete signal of "what the agent knows now" beyond the
	// original priority-files sample.
	pctx := samplePlanContext(ctx.WorkingDir, 6, 2000)
	for path, content := range ctx.FilesRead {
		if len(pctx) >= 8 {
			break
		}
		// Use relative path if possible so the planner key matches
		// what the agent will pass to read_file/edit_file later.
		rel := path
		if strings.HasPrefix(path, ctx.WorkingDir+"/") {
			rel = strings.TrimPrefix(path, ctx.WorkingDir+"/")
		}
		s := content
		if len(s) > 2000 {
			s = s[:2000] + "\n... (truncated)"
		}
		pctx[rel] = s
	}

	req := V3PlanRequest{
		UserMessage:    noted,
		WorkingDir:     ctx.WorkingDir,
		ProjectContext: pctx,
		NCandidates:    3,
	}
	plan, err := callV3PlanStreaming(ctx.V3URL, req, func(stage, detail string, data map[string]interface{}) {
		switch stage {
		case "token", "llm_start", "llm_end":
			return
		}
		payload := map[string]interface{}{"stage": stage, "detail": detail, "revision": ctx.PlanRevisions + 1}
		for k, v := range data {
			payload[k] = v
		}
		ctx.Stream("v3_plan", payload)
	})
	ctx.PlanRevisions++
	if err != nil || plan == nil {
		log.Printf("[agent] plan revision failed: %v — continuing with previous plan", err)
		return
	}
	ctx.Plan = plan
	ctx.PlanStepsSatisfied = make([]bool, len(plan.Steps))
	ctx.PlanOffStreak = 0

	// Re-emit the full plan structure so renderers replace the
	// previous plan view with the revised one. Same shape as the
	// initial generatePlan emission so consumers can use one code
	// path for both.
	planPayload := map[string]interface{}{
		"steps":         plan.Steps,
		"verify_step":   plan.VerifyStep,
		"rationale":     plan.Rationale,
		"winning_score": plan.WinningScore,
		"revision":      ctx.PlanRevisions,
	}
	ctx.Stream("plan_loaded", planPayload)
}

func countTrue(bs []bool) int {
	n := 0
	for _, b := range bs {
		if b {
			n++
		}
	}
	return n
}
