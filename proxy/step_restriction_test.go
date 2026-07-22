package main

import (
	"strings"
	"testing"
)

// May 2026 BiasBusters #2/#3 — locks the trigger that activates the
// per-step grammar restriction. The restriction must fire exactly when
// the model is about to pick a write_file/edit_file retry on an existing
// .py/.html file (the bias case from the May 7 flask session). It must
// NOT fire on stale rejections, non-Python/HTML files, or non-write_file
// failures.
func TestStepExclusions(t *testing.T) {
	cases := []struct {
		name      string
		messages  []AgentMessage
		wantTools []string
		wantExt   string
	}{
		{
			name: "fresh write_file rejection on .html — fires",
			messages: []AgentMessage{
				{Role: "system", Content: "sys"},
				{Role: "user", Content: "expand dashboard.html"},
				{Role: "assistant", Content: `{"type":"tool_call","name":"write_file","args":{"path":"templates/dashboard.html","content":"..."}}`},
				{Role: "tool", ToolName: "write_file", Content: `{"success":false,"error":"File templates/dashboard.html already exists (87 lines). write_file is for creating new files..."}`},
			},
			wantTools: []string{"edit_file", "write_file"},
			wantExt:   ".html",
		},
		{
			name: "fresh write_file rejection on .py — fires",
			messages: []AgentMessage{
				{Role: "tool", ToolName: "write_file", Content: `{"success":false,"error":"File app.py already exists (42 lines). write_file is for creating new files..."}`},
			},
			wantTools: []string{"edit_file", "write_file"},
			wantExt:   ".py",
		},
		{
			name: "fresh write_file rejection on .htm — fires",
			messages: []AgentMessage{
				{Role: "tool", ToolName: "write_file", Content: `{"success":false,"error":"File legacy.htm already exists (200 lines). write_file is for creating new files..."}`},
			},
			wantTools: []string{"edit_file", "write_file"},
			wantExt:   ".htm",
		},
		{
			name: "rejection on .css — does not fire (not a tree-sitter target)",
			messages: []AgentMessage{
				{Role: "tool", ToolName: "write_file", Content: `{"success":false,"error":"File styles.css already exists (50 lines). write_file is for creating new files..."}`},
			},
			wantTools: nil,
			wantExt:   "",
		},
		{
			name: "stale rejection — assistant has already corrected — does not fire",
			messages: []AgentMessage{
				{Role: "tool", ToolName: "write_file", Content: `{"success":false,"error":"File app.py already exists (42 lines)..."}`},
				{Role: "assistant", Content: `{"type":"tool_call","name":"structural_edit","args":{...}}`},
				{Role: "tool", ToolName: "structural_edit", Content: `{"success":true}`},
			},
			wantTools: nil,
			wantExt:   "",
		},
		{
			name: "edit_file rejection (wrong tool, not the trigger) — does not fire",
			messages: []AgentMessage{
				{Role: "tool", ToolName: "edit_file", Content: `{"success":false,"error":"old_str not found in app.py"}`},
			},
			wantTools: nil,
			wantExt:   "",
		},
		{
			name: "ephemeral system note already injected last turn — still fires",
			messages: []AgentMessage{
				{Role: "tool", ToolName: "write_file", Content: `{"success":false,"error":"File templates/index.html already exists (150 lines)..."}`},
				{Role: "user", Content: "[system note]: For this single decision, edit_file and write_file are unavailable..."},
			},
			wantTools: []string{"edit_file", "write_file"},
			wantExt:   ".html",
		},
		{
			name:      "empty conversation — does not fire",
			messages:  []AgentMessage{},
			wantTools: nil,
			wantExt:   "",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := &AgentContext{Messages: tc.messages}
			gotTools, gotExt := stepExclusions(ctx)
			if !equalStringSlices(gotTools, tc.wantTools) {
				t.Errorf("stepExclusions() tools = %v, want %v", gotTools, tc.wantTools)
			}
			if gotExt != tc.wantExt {
				t.Errorf("stepExclusions() ext = %q, want %q", gotExt, tc.wantExt)
			}
		})
	}
}

// Locks the GBNF restriction shape: the banned tool names must be absent
// from the tool-name production. Without this guard, a future refactor
// of buildGBNFGrammarForTools could silently drop the exclusion logic
// and leave the model free to pick edit_file again.
//
// GBNF quotes a tool name as `"\"edit_file\""` (a literal JSON string
// token), so we match against the raw escaped sequence rather than
// `"edit_file"`.
func TestBuildGBNFGrammarForToolsExcludes(t *testing.T) {
	const editFileTok = `"\"edit_file\""`
	const writeFileTok = `"\"write_file\""`
	const structuralEditTok = `"\"structural_edit\""`
	const readFileTok = `"\"read_file\""`

	all := buildGBNFGrammarForTools(nil)
	if !strings.Contains(all, editFileTok) {
		t.Fatalf("baseline grammar should contain %s before exclusion; grammar=\n%s", editFileTok, all)
	}
	restricted := buildGBNFGrammarForTools([]string{"edit_file", "write_file"})
	if strings.Contains(restricted, editFileTok) {
		t.Errorf("restricted grammar still contains %s", editFileTok)
	}
	if strings.Contains(restricted, writeFileTok) {
		t.Errorf("restricted grammar still contains %s", writeFileTok)
	}
	if !strings.Contains(restricted, structuralEditTok) {
		t.Errorf("restricted grammar must keep %s available", structuralEditTok)
	}
	if !strings.Contains(restricted, readFileTok) {
		t.Errorf("restricted grammar must keep %s available", readFileTok)
	}
}

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
