package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Effect classification is a property of the tool, declared on its ToolDef,
// so a new tool cannot be registered without deciding what it can do to the
// workspace. A parallel map keyed by name would drift; this cannot.
//
// The classes describe CAPABILITY, not outcome. A direct mutator still
// reports applied/refused/failed/none per branch; the effect only says the
// tool is capable of mutating and therefore owes a local classification.
func TestEveryRegisteredToolDeclaresAnEffect(t *testing.T) {
	if len(toolRegistry) == 0 {
		t.Fatal("tool registry is empty; this test would prove nothing")
	}
	var missing []string
	for name, def := range toolRegistry {
		if def.Effect == ToolEffectUnknown {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("tools registered without an effect class: %v", missing)
	}
}

// The verified distribution, pinned so a reclassification has to be
// deliberate. stop_background is CommandUnobserved rather than read-only or
// control: it sends SIGTERM/SIGKILL and performs no filesystem comparison, so
// a process killed mid-write may leave partial bytes nobody measured.
func TestEffectDistributionMatchesTheVerifiedAudit(t *testing.T) {
	want := map[string]ToolEffect{
		"read_file": ToolEffectReadOnly, "search_files": ToolEffectReadOnly,
		"list_directory": ToolEffectReadOnly, "find_file": ToolEffectReadOnly,
		"outline_file": ToolEffectReadOnly, "tail_background": ToolEffectReadOnly,

		"write_file": ToolEffectDirectMutation, "edit_file": ToolEffectDirectMutation,
		"structural_edit": ToolEffectDirectMutation, "delete_file": ToolEffectDirectMutation,
		"move_file": ToolEffectDirectMutation, "insert_after": ToolEffectDirectMutation,
		"replace_lines": ToolEffectDirectMutation,

		"run_command":     ToolEffectCommandUnobserved,
		"run_background":  ToolEffectCommandUnobserved,
		"stop_background": ToolEffectCommandUnobserved,
	}
	for name, expect := range want {
		def, ok := toolRegistry[name]
		if !ok {
			t.Errorf("expected tool %q is not registered", name)
			continue
		}
		if def.Effect != expect {
			t.Errorf("%s: effect = %q, want %q", name, def.Effect, expect)
		}
	}
	for name := range toolRegistry {
		if _, known := want[name]; !known {
			t.Errorf("tool %q is registered but absent from the verified audit; "+
				"classify it deliberately", name)
		}
	}
}

func TestDirectMutatorsAreNotClassifiableAtTheBoundary(t *testing.T) {
	for name, def := range toolRegistry {
		if def.Effect != ToolEffectDirectMutation {
			continue
		}
		// A direct mutator's outcome is only knowable at the branch that
		// performed it, so the boundary must never invent one.
		if def.Effect.BoundaryClassifiable() {
			t.Errorf("%s: a direct mutator must not be classifiable at the "+
				"shared boundary", name)
		}
	}
}

// Boundary classification through the REAL executeToolCall, not a mirror.
func TestBoundaryClassifiesReadOnlyAndCommandTools(t *testing.T) {
	dir := t.TempDir()
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.StreamFn = func(string, interface{}) {}

	cases := []struct {
		name string
		tool string
		args string
		want MutationStatus
	}{
		// Read-only: mutates nothing whether it succeeds or fails.
		{"read success", "read_file", `{"path":"f.txt"}`, MutationNone},
		{"read missing file", "read_file", `{"path":"nope.txt"}`, MutationNone},
		{"list", "list_directory", `{"path":"."}`, MutationNone},
		// Pre-dispatch refusals prove no handler ran.
		{"unknown tool", "no_such_tool", `{}`, MutationNone},
		{"missing args", "read_file", ``, MutationNone},
	}
	os.WriteFile(filepath.Join(dir, "f.txt"), []byte("hello\n"), 0o644)
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			res := executeToolCall(c.tool, json.RawMessage(c.args), ctx)
			if res.MutationStatus != c.want {
				t.Errorf("MutationStatus = %q, want %q", res.MutationStatus, c.want)
			}
			if !res.Classified() {
				t.Errorf("result not fully classified: %+v", res)
			}
			if res.ValidationStatus.Passed() {
				t.Error("validation was synthesized as passed")
			}
		})
	}
}

// A command tool that reached its handler is Unobserved regardless of exit
// status: a subprocess may have written before failing.
func TestCommandToolsReportUnobservedIncludingFailure(t *testing.T) {
	dir := t.TempDir()
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.TrustMode = trustFullyTrusted
	ctx.VerifyOnHost = true
	ctx.StreamFn = func(string, interface{}) {}

	for _, c := range []struct{ name, cmd string }{
		{"success", "true"},
		{"non-zero exit", "false"},
		{"writes then fails", "echo x > side_effect.txt; exit 3"},
	} {
		t.Run(c.name, func(t *testing.T) {
			args, _ := json.Marshal(map[string]string{"command": c.cmd})
			res := executeToolCall("run_command", args, ctx)
			if res.MutationStatus != MutationUnobserved {
				t.Errorf("MutationStatus = %q, want unobserved", res.MutationStatus)
			}
			if res.MutationStatus.Applied() {
				t.Error("unobserved must never read as applied")
			}
			if !res.Classified() {
				t.Errorf("result not classified: %+v", res)
			}
		})
	}
}

// The boundary must never invent a direct mutator's outcome.
func TestBoundaryDoesNotClassifyDirectMutators(t *testing.T) {
	if ToolEffectDirectMutation.BoundaryClassifiable() {
		t.Fatal("direct mutation must not be boundary-classifiable")
	}
	for _, e := range []ToolEffect{ToolEffectReadOnly, ToolEffectCommandUnobserved} {
		if !e.BoundaryClassifiable() {
			t.Errorf("%q should be boundary-classifiable", e)
		}
	}
	if ToolEffectUnknown.BoundaryClassifiable() {
		t.Fatal("unknown effect must not be boundary-classifiable")
	}
}

// Intermediate-state safety check. The boundary now declines to classify
// direct mutators, and "declines to classify" must mean exactly that: the
// result passes through untouched and the mutation still happens. If
// BoundaryClassifiable were ever read as an authorization check, write_file
// would start failing operationally while every unit test still passed.
func TestDirectMutatorStillMutatesWhileUnclassified(t *testing.T) {
	dir := t.TempDir()
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.StreamFn = func(string, interface{}) {}

	args, _ := json.Marshal(map[string]string{
		"path": "new.py", "content": "VALUE = 1\n"})
	res := executeToolCall("write_file", args, ctx)

	// 1. The mutation actually occurred.
	got, err := os.ReadFile(filepath.Join(dir, "new.py"))
	if err != nil {
		t.Fatalf("write_file no longer writes: %v", err)
	}
	if string(got) != "VALUE = 1\n" {
		t.Fatalf("bytes on disk = %q", string(got))
	}
	// 2. Legacy Success is untouched.
	if !res.Success {
		t.Fatalf("legacy Success regressed to false: %+v", res)
	}
	// 3. The boundary did not invent a classification for it.
	if res.MutationStatus != MutationUnknown {
		t.Errorf("boundary classified a direct mutator as %q; only the local "+
			"branch may do that", res.MutationStatus)
	}
	// 4. Unknown is a pending-migration state, not an operational rejection.
	if res.Error != "" {
		t.Errorf("unclassified direct mutator was rejected: %q", res.Error)
	}
}

// The same for a direct mutator that legitimately fails: still no boundary
// classification, still the real error, still no invented facts.
func TestDirectMutatorFailurePassesThroughUnclassified(t *testing.T) {
	dir := t.TempDir()
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.StreamFn = func(string, interface{}) {}

	args, _ := json.Marshal(map[string]string{"path": "missing.txt"})
	res := executeToolCall("delete_file", args, ctx)
	if res.Success {
		t.Fatal("deleting a missing file unexpectedly succeeded")
	}
	if res.MutationStatus != MutationUnknown {
		t.Errorf("boundary classified a failing direct mutator as %q",
			res.MutationStatus)
	}
	if res.Error == "" {
		t.Error("the real error was suppressed")
	}
}
