package main

import "testing"

// The active edit-test-fix loop: after a successful write of foo.py and a
// FAILED run referencing it, the next write must fast-path (skip V3).
func TestIsActiveDebugIteration(t *testing.T) {
	ctx := &AgentContext{
		SessionWrites: map[string]bool{"foo.py": true},
		Messages: []AgentMessage{
			{Role: "assistant", Content: "..."},
			{Role: "tool", ToolName: "run_command",
				Content: `{"success":false,"error":"File \"foo.py\", line 3\nSyntaxError"}`},
		},
	}
	if !isActiveDebugIteration(ctx, "foo.py") {
		t.Error("failed run of an already-written file should be active iteration")
	}
	// First write of a file (not in SessionWrites) → V3, not fast-path.
	if isActiveDebugIteration(ctx, "bar.py") {
		t.Error("first write of a file must not fast-path")
	}
	// Last tool action was a read, not a failing run → not iterating.
	ctx.Messages[1] = AgentMessage{Role: "tool", ToolName: "read_file", Content: "..."}
	if isActiveDebugIteration(ctx, "foo.py") {
		t.Error("a read as the last action is not an edit-test-fix loop")
	}
	// Run succeeded → task likely done, not iterating.
	ctx.Messages[1] = AgentMessage{Role: "tool", ToolName: "run_command",
		Content: `{"success":true,"data":{"stdout":"ok"}}`}
	if isActiveDebugIteration(ctx, "foo.py") {
		t.Error("a passing run is not an active fix loop")
	}
}

// isBinaryContent: NUL-bearing data is binary; clean text is not.
func TestIsBinaryContent(t *testing.T) {
	if !isBinaryContent([]byte("\x7fELF\x02\x01\x00\x00garbage")) {
		t.Error("ELF header with NULs should be binary")
	}
	if isBinaryContent([]byte("def foo():\n    return 1\n")) {
		t.Error("plain source text must not be flagged binary")
	}
	if isBinaryContent([]byte("")) {
		t.Error("empty file is not binary")
	}
}
