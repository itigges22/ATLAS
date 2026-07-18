package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestRepeatDetectorFiresOnIdenticalCalls(t *testing.T) {
	ctx := &AgentContext{}
	args := json.RawMessage(`{"path":"app.py","offset":0,"limit":100}`)
	for i := 0; i < 2; i++ {
		if _, repeating := recordToolCall(ctx, "read_file", args); repeating {
			t.Fatalf("fired at call %d, want threshold 3", i+1)
		}
	}
	msg, repeating := recordToolCall(ctx, "read_file", args)
	if !repeating {
		t.Fatal("identical call 3x must fire")
	}
	if !strings.Contains(msg, "read_file") {
		t.Fatalf("corrective doesn't name the tool: %q", msg)
	}
}

func TestRepeatDetectorCanonicalizesJSONFormatting(t *testing.T) {
	ctx := &AgentContext{}
	recordToolCall(ctx, "run_command", json.RawMessage(`{"command":"pytest","timeout":30}`))
	recordToolCall(ctx, "run_command", json.RawMessage(`{"timeout":30,"command":"pytest"}`))
	_, repeating := recordToolCall(ctx, "run_command", json.RawMessage(`{ "command" : "pytest", "timeout" : 30 }`))
	if !repeating {
		t.Fatal("key order / whitespace variations of the same call must match")
	}
}

func TestWriteFileRepeatKeyedOnPathNotContent(t *testing.T) {
	// The 2026-07-18 loop: the model fully rewrote app.py five times
	// with slightly different content each attempt while V3 kept
	// writing the verified expansion. Content-varying rewrites of the
	// same path must still count as repetition.
	ctx := &AgentContext{}
	for i := 0; i < 2; i++ {
		args := json.RawMessage(fmt.Sprintf(
			`{"path":"app.py","content":"from flask import Flask # attempt %d"}`, i))
		if _, repeating := recordToolCall(ctx, "write_file", args); repeating {
			t.Fatalf("fired at write %d, want threshold 3", i+1)
		}
	}
	msg, repeating := recordToolCall(ctx, "write_file",
		json.RawMessage(`{"path":"app.py","content":"completely different third draft"}`))
	if !repeating {
		t.Fatal("3 rewrites of the same path with different content must fire")
	}
	if !strings.Contains(msg, "app.py") || !strings.Contains(msg, "rewritten") {
		t.Fatalf("write-loop corrective should name the path and the rewrite pattern: %q", msg)
	}
}

func TestWriteFileDifferentPathsDoNotFire(t *testing.T) {
	ctx := &AgentContext{}
	for i, p := range []string{"app.py", "static/game.js", "templates/index.html"} {
		args := json.RawMessage(fmt.Sprintf(`{"path":"%s","content":"x"}`, p))
		if _, repeating := recordToolCall(ctx, "write_file", args); repeating {
			t.Fatalf("multi-file scaffolding flagged as a loop at write %d (%s)", i+1, p)
		}
	}
}

func TestEditFileKeepsFullArgsSignature(t *testing.T) {
	// Distinct surgical edits to one file in close succession are
	// legitimate iteration — only identical edits are a loop.
	ctx := &AgentContext{}
	for i := 0; i < 4; i++ {
		args := json.RawMessage(fmt.Sprintf(
			`{"path":"app.py","old_str":"v%d","new_str":"v%d"}`, i, i+1))
		if _, repeating := recordToolCall(ctx, "edit_file", args); repeating {
			t.Fatalf("distinct edits to one path flagged as a loop at edit %d", i+1)
		}
	}
}

func TestWriteFileRepeatOutsideWindowDoesNotFire(t *testing.T) {
	ctx := &AgentContext{}
	wf := json.RawMessage(`{"path":"app.py","content":"draft"}`)
	recordToolCall(ctx, "write_file", wf)
	// Eight unrelated calls push the first write out of the window.
	for i := 0; i < toolRepeatWindow; i++ {
		recordToolCall(ctx, "read_file",
			json.RawMessage(fmt.Sprintf(`{"path":"f%d.py"}`, i)))
	}
	recordToolCall(ctx, "write_file", wf)
	if _, repeating := recordToolCall(ctx, "write_file", wf); repeating {
		t.Fatal("two in-window writes must not fire (threshold 3)")
	}
}
