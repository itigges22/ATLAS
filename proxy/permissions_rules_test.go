// Tests for needsPermission — the mode/approval logic behind interactive
// permission prompts.

package main

import (
	"encoding/json"
	"testing"
)

func TestNeedsPermission(t *testing.T) {
	args := json.RawMessage(`{}`)

	t.Run("yolo mode approves everything", func(t *testing.T) {
		for _, ctx := range []*AgentContext{
			{YoloMode: true},
			{PermissionMode: PermissionYolo},
		} {
			if needsPermission(ctx, "run_command", args) {
				t.Errorf("yolo context still prompts for run_command")
			}
		}
	})

	t.Run("unknown tool always prompts", func(t *testing.T) {
		if !needsPermission(&AgentContext{}, "no_such_tool", args) {
			t.Errorf("unknown tool did not prompt")
		}
	})

	t.Run("read-only tools never prompt", func(t *testing.T) {
		if needsPermission(&AgentContext{}, "read_file", args) {
			t.Errorf("read_file prompted in default mode")
		}
	})

	t.Run("destructive tools prompt in default mode", func(t *testing.T) {
		if !needsPermission(&AgentContext{}, "run_command", args) {
			t.Errorf("run_command did not prompt in default mode")
		}
		if !needsPermission(&AgentContext{}, "write_file", args) {
			t.Errorf("write_file did not prompt in default mode")
		}
	})

	t.Run("session-approved tool skips the prompt", func(t *testing.T) {
		ctx := &AgentContext{}
		ctx.allowToolForTurn("run_command")
		if needsPermission(ctx, "run_command", args) {
			t.Errorf("session-approved run_command still prompts")
		}
		// Approval is per-tool, not global.
		if !needsPermission(ctx, "write_file", args) {
			t.Errorf("write_file inherited run_command's approval")
		}
	})

	t.Run("accept-edits auto-approves edits but not commands", func(t *testing.T) {
		ctx := &AgentContext{PermissionMode: PermissionAcceptEdits}
		for _, tool := range []string{"write_file", "edit_file", "ast_edit", "move_file"} {
			if needsPermission(ctx, tool, args) {
				t.Errorf("%s prompted in accept-edits mode", tool)
			}
		}
		if !needsPermission(ctx, "run_command", args) {
			t.Errorf("run_command did not prompt in accept-edits mode")
		}
	})
}
