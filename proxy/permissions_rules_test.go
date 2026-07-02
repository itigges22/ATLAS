// Tests for the user-configurable permission rules engine
// (checkPermissionRules / matchPattern) and the needsPermission
// mode/approval logic behind interactive approval.

package main

import (
	"encoding/json"
	"testing"
)

func TestMatchPattern(t *testing.T) {
	cases := []struct {
		name    string
		pattern string
		value   string
		want    bool
	}{
		{"exact match", "npm install", "npm install", true},
		{"exact mismatch is still contains-checked", "npm install", "npm", false},
		{"glob star suffix", "npm *", "npm install", true},
		{"prefix wildcard", "git*", "git push origin dev", true},
		{"prefix wildcard no match", "git*", "cargo build", false},
		{"path glob", "docs/*.md", "docs/SETUP.md", true},
		// filepath.Match's * does not cross /, but the prefix-wildcard
		// branch takes over for nested paths.
		{"trailing star crosses path separators", "docs/*", "docs/lang/ja/README.md", true},
		// Substring matching is the documented fallback for command
		// patterns: a bare word matches anywhere in the command line.
		{"substring fallback", "pytest", "python -m pytest tests/", true},
		{"substring fallback negative", "pytest", "python -m unittest", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := matchPattern(tc.pattern, tc.value); got != tc.want {
				t.Errorf("matchPattern(%q, %q) = %v, want %v",
					tc.pattern, tc.value, got, tc.want)
			}
		})
	}
}

func TestCheckPermissionRules(t *testing.T) {
	runArgs := func(cmd string) json.RawMessage {
		b, _ := json.Marshal(map[string]string{"command": cmd})
		return b
	}
	writeArgs := func(path string) json.RawMessage {
		b, _ := json.Marshal(map[string]string{"path": path, "content": "x"})
		return b
	}

	t.Run("no rules yields no decision", func(t *testing.T) {
		if got := checkPermissionRules(nil, "run_command", runArgs("ls")); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})

	t.Run("rule for a different tool does not fire", func(t *testing.T) {
		rules := []PermissionRule{{Tool: "write_file", Pattern: "*", Action: "deny"}}
		if got := checkPermissionRules(rules, "run_command", runArgs("ls")); got != "" {
			t.Errorf("write_file rule fired for run_command: got %q", got)
		}
	})

	t.Run("command pattern allows matching command", func(t *testing.T) {
		rules := []PermissionRule{{Tool: "run_command", Pattern: "npm *", Action: "allow"}}
		if got := checkPermissionRules(rules, "run_command", runArgs("npm install")); got != "allow" {
			t.Errorf("got %q, want allow", got)
		}
		if got := checkPermissionRules(rules, "run_command", runArgs("cargo build")); got != "" {
			t.Errorf("non-matching command decided %q, want empty", got)
		}
	})

	t.Run("first matching rule wins", func(t *testing.T) {
		rules := []PermissionRule{
			{Tool: "run_command", Pattern: "git push*", Action: "deny"},
			{Tool: "run_command", Pattern: "git *", Action: "allow"},
		}
		if got := checkPermissionRules(rules, "run_command", runArgs("git push origin main")); got != "deny" {
			t.Errorf("got %q, want deny (earlier rule should win)", got)
		}
		if got := checkPermissionRules(rules, "run_command", runArgs("git status")); got != "allow" {
			t.Errorf("got %q, want allow", got)
		}
	})

	t.Run("path rules match write_file targets", func(t *testing.T) {
		rules := []PermissionRule{{Tool: "write_file", Pattern: "docs/*", Action: "allow"}}
		if got := checkPermissionRules(rules, "write_file", writeArgs("docs/SETUP.md")); got != "allow" {
			t.Errorf("got %q, want allow", got)
		}
		if got := checkPermissionRules(rules, "write_file", writeArgs("main.go")); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})

	t.Run("malformed args yield no decision instead of panicking", func(t *testing.T) {
		rules := []PermissionRule{{Tool: "run_command", Pattern: "*", Action: "allow"}}
		if got := checkPermissionRules(rules, "run_command", json.RawMessage(`{"command": 42}`)); got != "" {
			t.Errorf("got %q, want empty for malformed args", got)
		}
	})

	t.Run("tool without a match-value extractor never matches", func(t *testing.T) {
		rules := []PermissionRule{{Tool: "read_file", Pattern: "*", Action: "deny"}}
		if got := checkPermissionRules(rules, "read_file", writeArgs("x")); got != "" {
			t.Errorf("got %q, want empty (read_file has no extractor)", got)
		}
	})
}

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
