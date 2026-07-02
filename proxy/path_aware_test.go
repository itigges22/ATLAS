package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// May 10 2026 path-aware error-loop breaker + plan-progress reminder.
// Locks both against regression.

func TestExtractFailurePath(t *testing.T) {
	cases := []struct {
		tool string
		args string
		want string
	}{
		{"read_file", `{"path":"app.py","offset":0,"limit":100}`, "app.py"},
		{"write_file", `{"path":"new.py","content":"hello"}`, "new.py"},
		{"edit_file", `{"path":"a.py","old_str":"x","new_str":"y"}`, "a.py"},
		{"ast_edit", `{"path":"t.html","selector":"<body>","content":"..."}`, "t.html"},
		{"delete_file", `{"path":"old.py"}`, "old.py"},
		{"find_file", `{"pattern":".*test.*\\.py$"}`, ""},                 // no path field
		{"find_file", `{"pattern":"x","path":"src/"}`, "src/"},            // optional path
		{"list_directory", `{"path":"templates"}`, "templates"},
		{"search_files", `{"pattern":"TODO","path":"src/"}`, "src/"},
		{"run_command", `{"command":"python app.py"}`, ""}, // no path applicable
		{"run_background", `{"command":"flask run"}`, ""},
	}
	for _, tc := range cases {
		t.Run(tc.tool, func(t *testing.T) {
			got := extractFailurePath(tc.tool, json.RawMessage(tc.args))
			if got != tc.want {
				t.Errorf("extractFailurePath(%q, %q) = %q, want %q", tc.tool, tc.args, got, tc.want)
			}
		})
	}
}

func TestBuildPlanReminderNoPlan(t *testing.T) {
	ctx := &AgentContext{}
	if got := buildPlanReminder(ctx); got != "" {
		t.Errorf("no plan should return empty, got %q", got)
	}
}

func TestBuildPlanReminderRendersProgress(t *testing.T) {
	ctx := &AgentContext{
		Plan: &Plan{
			Steps: []PlanStep{
				{ID: "s1", Action: "read_file", Target: "app.py"},
				{ID: "s2", Action: "ast_edit", Target: "templates/dashboard.html"},
				{ID: "s3", Action: "run_command", Target: "curl localhost:5000/dashboard"},
			},
			VerifyStep: "s3",
		},
		PlanStepsSatisfied: []bool{true, false, false},
	}
	got := buildPlanReminder(ctx)
	if got == "" {
		t.Fatal("expected reminder, got empty")
	}
	for _, s := range []string{"[system note]: plan progress", "1/3", "s2", "ast_edit templates/dashboard.html", "Done: s1", "Remaining: s2, s3"} {
		if !strings.Contains(got, s) {
			t.Errorf("reminder missing %q: %s", s, got)
		}
	}
}

func TestBuildPlanReminderAllSatisfied(t *testing.T) {
	ctx := &AgentContext{
		Plan: &Plan{
			Steps:      []PlanStep{{ID: "s1"}, {ID: "s2"}},
			VerifyStep: "s2",
		},
		PlanStepsSatisfied: []bool{true, true},
	}
	got := buildPlanReminder(ctx)
	for _, s := range []string{"plan complete", "2/2", "s2"} {
		if !strings.Contains(got, s) {
			t.Errorf("complete-plan reminder missing %q: %s", s, got)
		}
	}
}

func TestBuildPlanReminderLazyInitsSatisfied(t *testing.T) {
	// PlanStepsSatisfied can be nil at first turn — the reminder
	// should lazily initialize it so it doesn't panic.
	ctx := &AgentContext{
		Plan: &Plan{
			Steps: []PlanStep{{ID: "s1", Action: "read_file", Target: "app.py"}},
		},
	}
	got := buildPlanReminder(ctx)
	if got == "" {
		t.Fatal("expected reminder, got empty")
	}
	if ctx.PlanStepsSatisfied == nil || len(ctx.PlanStepsSatisfied) != 1 {
		t.Errorf("reminder should have lazily initialized PlanStepsSatisfied; got %v", ctx.PlanStepsSatisfied)
	}
}
