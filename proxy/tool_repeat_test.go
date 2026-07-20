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

func TestWriteFileReassertionKeyedOnPathAndContent(t *testing.T) {
	// The 2026-07-18 loop: the model reasserted the SAME app.py draft
	// while V3 wrote the verified expansion. Reassertion = same logical
	// content (whitespace/formatting aside) rewritten to the same path;
	// it must still fire at the threshold. (Materially different content
	// is iteration — TestWriteFileIterationNotRepeat — and must NOT fire.)
	ctx := &AgentContext{}
	for i := 0; i < 2; i++ {
		// Same code — only TRAILING whitespace/CR differs, which is noise and
		// must still collide. (Leading indentation is semantic in Python and
		// is deliberately NOT collapsed — TestWriteFileIndentationChangeIsIteration.)
		args := json.RawMessage(fmt.Sprintf(
			`{"path":"app.py","content":"from flask import Flask\napp = Flask(__name__)%s"}`,
			strings.Repeat(" ", i)))
		if _, repeating := recordToolCall(ctx, "write_file", args); repeating {
			t.Fatalf("fired at write %d, want threshold 3", i+1)
		}
	}
	msg, repeating := recordToolCall(ctx, "write_file",
		json.RawMessage(`{"path":"app.py","content":"from flask import Flask\napp = Flask(__name__)"}`))
	if !repeating {
		t.Fatal("reassertion of the same logical content must fire")
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

// Iteration must NOT be flagged as repetition: rewriting the same file
// with materially different content (fixing successive compiler errors)
// produces different signatures, so the detector stays silent. Regression
// for TB2 2026-07-19 (polyglot killed mid-fix by the path-only key).
func TestWriteFileIterationNotRepeat(t *testing.T) {
	ctx := &AgentContext{}
	versions := []string{
		`{"path":"main.py.c","content":"int main(){ return 0; }"}`,
		`{"path":"main.py.c","content":"int main(){ printf(\"x\"); return 0; }"}`,
		`{"path":"main.py.c","content":"#include <stdio.h>\nint main(){ printf(\"x\"); return 0; }"}`,
	}
	for i, v := range versions {
		_, repeating := recordToolCall(ctx, "write_file", json.RawMessage(v))
		if repeating {
			t.Errorf("version %d: iteration flagged as repetition", i)
		}
	}
}

// Reassertion IS still caught: rewriting the same file with identical
// code — only trailing whitespace / line-ending noise differs — collides
// on the fingerprint and fires at the threshold. Protects the 2026-07-18 case.
func TestWriteFileReassertionStillCaught(t *testing.T) {
	ctx := &AgentContext{}
	// Same code + same leading indentation; only trailing whitespace/CR varies.
	versions := []string{
		`{"path":"app.py","content":"def f():\n    return 1"}`,
		`{"path":"app.py","content":"def f():\n    return 1  "}`,
		`{"path":"app.py","content":"def f():\r\n    return 1\r"}`,
	}
	fired := false
	for _, v := range versions {
		if _, r := recordToolCall(ctx, "write_file", json.RawMessage(v)); r {
			fired = true
		}
	}
	if !fired {
		t.Error("reassertion of the same logical content was not caught")
	}
}

// An indentation-only change is a REAL change in Python (iteration), so it
// must NOT collide as reassertion (#147 review finding #13).
func TestWriteFileIndentationChangeIsIteration(t *testing.T) {
	ctx := &AgentContext{}
	// A common fix: correcting a wrongly-indented body line. Different
	// leading indentation -> different fingerprint -> not flagged.
	versions := []string{
		`{"path":"m.py","content":"def f():\nreturn 1"}`,        // broken indent
		`{"path":"m.py","content":"def f():\n    return 1"}`,    // fixed (4)
		`{"path":"m.py","content":"def f():\n        return 1"}`, // 8-space
	}
	for i, v := range versions {
		if _, r := recordToolCall(ctx, "write_file", json.RawMessage(v)); r {
			t.Fatalf("indentation change at write %d flagged as reassertion", i+1)
		}
	}
}
