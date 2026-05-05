package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func mkPlan(steps ...PlanStep) *Plan {
	p := &Plan{Steps: steps}
	if len(steps) > 0 {
		p.VerifyStep = steps[len(steps)-1].ID
	}
	return p
}

func mkArgs(t *testing.T, v interface{}) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestActionMatchesTool(t *testing.T) {
	cases := []struct {
		action, tool string
		want         bool
	}{
		// Direct + prefix-stem matching against the canonical tool
		// names. The planner is prompted to produce tool names
		// verbatim, so we don't need to handle freeform descriptive
		// actions ("verify with curl") — those count as off-plan.
		{"read_file", "read_file", true},
		{"read", "read_file", true},
		{"write_file", "write_file", true},
		{"edit", "edit_file", true},
		{"run_command", "run_command", true},
		{"run", "run_command", true},
		{"investigate the bug", "read_file", false},
		{"verify with curl", "run_command", false},
		{"write_file", "edit_file", false},
		{"", "read_file", false},
		{"read", "", false},
	}
	for _, tc := range cases {
		if got := actionMatchesTool(tc.action, tc.tool); got != tc.want {
			t.Errorf("actionMatchesTool(%q, %q) = %v, want %v",
				tc.action, tc.tool, got, tc.want)
		}
	}
}

func TestTargetsOverlap(t *testing.T) {
	cases := []struct {
		a, b string
		want bool
	}{
		{"app.py", "app.py", true},
		{"app.py", "/workspace/app.py", true},
		{"templates/index.html", "/workspace/templates/index.html", true},
		{"./app.py", "app.py", true},
		{"app.py", "tests/app.py", true},
		{"app.py", "src/main.go", false},
		{"curl http://localhost:5000/", "curl http://localhost:5000/hello", true},
		{"", "app.py", false},
	}
	for _, tc := range cases {
		if got := targetsOverlap(tc.a, tc.b); got != tc.want {
			t.Errorf("targetsOverlap(%q, %q) = %v, want %v",
				tc.a, tc.b, got, tc.want)
		}
	}
}

func TestMatchPlanStepFirstUnsatisfied(t *testing.T) {
	plan := mkPlan(
		PlanStep{ID: "s1", Action: "read_file", Target: "templates/index.html"},
		PlanStep{ID: "s2", Action: "edit_file", Target: "templates/index.html"},
		PlanStep{ID: "s3", Action: "run_command", Target: "curl http://localhost:5000/"},
	)
	satisfied := []bool{false, false, false}

	// read_file on the templates path matches s1.
	args := mkArgs(t, map[string]string{"path": "/workspace/templates/index.html"})
	if got := matchPlanStep(plan, satisfied, "read_file", args); got != 0 {
		t.Errorf("matchPlanStep first call = %d, want 0", got)
	}

	// Mark s1 satisfied. read_file again should NOT re-match s1 (already done).
	satisfied[0] = true
	if got := matchPlanStep(plan, satisfied, "read_file", args); got != -1 {
		t.Errorf("matchPlanStep after s1 satisfied = %d, want -1", got)
	}

	// edit_file on the same path matches s2.
	editArgs := mkArgs(t, map[string]string{"path": "templates/index.html"})
	if got := matchPlanStep(plan, satisfied, "edit_file", editArgs); got != 1 {
		t.Errorf("matchPlanStep edit_file = %d, want 1", got)
	}

	// run_command with curl matches s3 even with extra path components.
	runArgs := mkArgs(t, map[string]string{"command": "curl http://localhost:5000/"})
	if got := matchPlanStep(plan, satisfied, "run_command", runArgs); got != 2 {
		t.Errorf("matchPlanStep run_command = %d, want 2", got)
	}
}

func TestMatchPlanStepNoMatchOffPlan(t *testing.T) {
	plan := mkPlan(
		PlanStep{ID: "s1", Action: "read_file", Target: "app.py"},
	)
	satisfied := []bool{false}

	// list_directory isn't in the plan — should not match.
	args := mkArgs(t, map[string]string{"path": "."})
	if got := matchPlanStep(plan, satisfied, "list_directory", args); got != -1 {
		t.Errorf("off-plan list_directory matched step %d, want -1", got)
	}

	// read_file on a different file shouldn't match a target-specific step.
	args = mkArgs(t, map[string]string{"path": "tests/test_app.py"})
	if got := matchPlanStep(plan, satisfied, "read_file", args); got != -1 {
		t.Errorf("read_file on wrong file matched step %d, want -1", got)
	}
}

func TestMatchPlanStepNilPlanReturnsMinusOne(t *testing.T) {
	if got := matchPlanStep(nil, nil, "read_file", nil); got != -1 {
		t.Errorf("nil plan = %d, want -1", got)
	}
}

func TestRecordPlanAdherenceUpdatesState(t *testing.T) {
	plan := mkPlan(
		PlanStep{ID: "s1", Action: "read_file", Target: "app.py"},
		PlanStep{ID: "s2", Action: "edit_file", Target: "app.py"},
	)
	ctx := &AgentContext{Plan: plan}
	ctx.StreamFn = func(string, interface{}) {} // /dev/null sink

	// On-plan read → satisfied[0], streak resets.
	revise := recordPlanAdherence(ctx, "read_file",
		mkArgs(t, map[string]string{"path": "app.py"}), true)
	if revise {
		t.Error("first call shouldn't trigger revise")
	}
	if !ctx.PlanStepsSatisfied[0] {
		t.Error("step 0 not marked satisfied")
	}
	if ctx.PlanOffStreak != 0 {
		t.Errorf("off_streak = %d, want 0 after on-plan call", ctx.PlanOffStreak)
	}

	// Off-plan list_directory → streak goes to 1.
	revise = recordPlanAdherence(ctx, "list_directory",
		mkArgs(t, map[string]string{"path": "."}), true)
	if revise {
		t.Error("streak=1 shouldn't trigger revise")
	}
	if ctx.PlanOffStreak != 1 {
		t.Errorf("off_streak = %d, want 1", ctx.PlanOffStreak)
	}

	// Two more off-plan → streak hits threshold.
	recordPlanAdherence(ctx, "list_directory",
		mkArgs(t, map[string]string{"path": "."}), true)
	revise = recordPlanAdherence(ctx, "list_directory",
		mkArgs(t, map[string]string{"path": "."}), true)
	if !revise {
		t.Errorf("streak=%d should trigger revise (threshold=%d)",
			ctx.PlanOffStreak, planAutoReviseThreshold)
	}
}

func TestRecordPlanAdherenceFailedCallsDontSatisfy(t *testing.T) {
	plan := mkPlan(
		PlanStep{ID: "s1", Action: "run_command", Target: "pytest"},
	)
	ctx := &AgentContext{Plan: plan}
	ctx.StreamFn = func(string, interface{}) {}

	// Failed run_command shouldn't tick off the verify step.
	recordPlanAdherence(ctx, "run_command",
		mkArgs(t, map[string]string{"command": "pytest"}), false)
	if ctx.PlanStepsSatisfied[0] {
		t.Error("failed call shouldn't satisfy plan step")
	}
	if ctx.PlanOffStreak != 1 {
		t.Errorf("failed call should extend streak, got %d", ctx.PlanOffStreak)
	}
}

func TestRecordPlanAdherenceNoOpWithoutPlan(t *testing.T) {
	ctx := &AgentContext{}
	if recordPlanAdherence(ctx, "read_file", nil, true) {
		t.Error("nil plan shouldn't trigger revise")
	}
	if ctx.PlanStepsSatisfied != nil {
		t.Error("nil plan shouldn't allocate satisfied tracking")
	}
}

func TestRecordPlanAdherenceCapsRevisions(t *testing.T) {
	plan := mkPlan(PlanStep{ID: "s1", Action: "read_file", Target: "a.py"})
	ctx := &AgentContext{
		Plan:          plan,
		PlanRevisions: planMaxRevisions, // already at the cap
	}
	ctx.StreamFn = func(string, interface{}) {}

	// Hammer with off-plan calls — past the cap, recordPlanAdherence
	// must NOT request a revise (we'd thrash forever otherwise).
	for i := 0; i < 10; i++ {
		if recordPlanAdherence(ctx, "list_directory",
			mkArgs(t, map[string]string{"path": "."}), true) {
			t.Fatalf("revise triggered past cap at i=%d", i)
		}
	}
}

func TestBuildSystemPromptIncludesPlan(t *testing.T) {
	plan := mkPlan(
		PlanStep{ID: "s1", Action: "read_file", Target: "app.py", Why: "inspect current routes"},
		PlanStep{ID: "s2", Action: "edit_file", Target: "app.py", Why: "add /hello route"},
		PlanStep{ID: "s3", Action: "run_command", Target: "curl http://localhost:5000/hello", Why: "verify"},
	)
	plan.Rationale = "investigate, change, verify."
	ctx := &AgentContext{
		WorkingDir: "/workspace",
		Plan:       plan,
	}
	prompt := buildSystemPrompt(ctx)

	// Plan section header present.
	if !strings.Contains(prompt, "## Plan") {
		t.Error("system prompt missing ## Plan header")
	}
	// All three step actions surfaced.
	for _, s := range []string{"read_file", "edit_file", "run_command"} {
		if !strings.Contains(prompt, s) {
			t.Errorf("system prompt missing step action %q", s)
		}
	}
	// Verify step marker present so model knows which step is "done"-gate.
	if !strings.Contains(prompt, "verify step (s3)") {
		t.Error("system prompt doesn't call out the verify step")
	}
	if !strings.Contains(prompt, "investigate, change, verify") {
		t.Error("system prompt doesn't include rationale")
	}
}

func TestBuildSystemPromptOmitsPlanSectionWhenNoPlan(t *testing.T) {
	ctx := &AgentContext{WorkingDir: "/workspace"}
	prompt := buildSystemPrompt(ctx)
	if strings.Contains(prompt, "## Plan") {
		t.Error("system prompt has Plan section when ctx.Plan is nil")
	}
}

func TestExtractToolTargetReadsCommonShapes(t *testing.T) {
	cases := []struct {
		tool string
		args interface{}
		want string
	}{
		{"read_file", map[string]string{"path": "app.py"}, "app.py"},
		{"write_file", map[string]string{"path": "x.py", "content": "..."}, "x.py"},
		{"edit_file", map[string]string{"path": "y.py", "old_str": "a", "new_str": "b"}, "y.py"},
		{"run_command", map[string]string{"command": "pytest tests/"}, "pytest tests/"},
		{"list_directory", map[string]string{"path": "src"}, "src"},
		{"plan_revise", map[string]string{"reason": "x"}, ""}, // unknown tool → empty
	}
	for _, tc := range cases {
		args := mkArgs(t, tc.args)
		if got := extractToolTarget(tc.tool, args); got != tc.want {
			t.Errorf("extractToolTarget(%s) = %q, want %q", tc.tool, got, tc.want)
		}
	}
}
