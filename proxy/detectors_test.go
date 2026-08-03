package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRepeatDetectorFiresOnIdenticalCalls(t *testing.T) {
	ctx := &AgentContext{}
	args := json.RawMessage(`{"path":"app.py","offset":0,"limit":100}`)
	for i := 0; i < 2; i++ {
		if _, _, repeating := recordToolCall(ctx, "read_file", args); repeating {
			t.Fatalf("fired at call %d, want threshold 3", i+1)
		}
	}
	msg, obs, repeating := recordToolCall(ctx, "read_file", args)
	if !repeating {
		t.Fatal("identical call 3x must fire")
	}
	if !strings.Contains(msg, "read_file") {
		t.Fatalf("corrective doesn't name the tool: %q", msg)
	}
	// The observation carries the streak the detector just erased.
	if obs.Count != toolRepeatThreshold {
		t.Errorf("observation count = %d, want %d", obs.Count, toolRepeatThreshold)
	}
	// The detector owns the reset: the window is empty on return, so the
	// same streak can't fire a second corrective.
	if len(ctx.RecentToolCalls) != 0 {
		t.Errorf("firing must clear the window, got %d entries", len(ctx.RecentToolCalls))
	}
	if _, _, again := recordToolCall(ctx, "read_file", args); again {
		t.Error("the call right after a fire must start a fresh streak")
	}
}

func TestRepeatDetectorCanonicalizesJSONFormatting(t *testing.T) {
	ctx := &AgentContext{}
	recordToolCall(ctx, "run_command", json.RawMessage(`{"command":"pytest","timeout":30}`))
	recordToolCall(ctx, "run_command", json.RawMessage(`{"timeout":30,"command":"pytest"}`))
	_, _, repeating := recordToolCall(ctx, "run_command", json.RawMessage(`{ "command" : "pytest", "timeout" : 30 }`))
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
		if _, _, repeating := recordToolCall(ctx, "write_file", args); repeating {
			t.Fatalf("fired at write %d, want threshold 3", i+1)
		}
	}
	msg, _, repeating := recordToolCall(ctx, "write_file",
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
		if _, _, repeating := recordToolCall(ctx, "write_file", args); repeating {
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
		if _, _, repeating := recordToolCall(ctx, "edit_file", args); repeating {
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
	if _, _, repeating := recordToolCall(ctx, "write_file", wf); repeating {
		t.Fatal("two in-window writes must not fire (threshold 3)")
	}
}

// Iteration must NOT be flagged as repetition: rewriting the same file
// with materially different content (fixing successive compiler errors)
// produces different signatures, so the detector stays silent. Regression
// for 2026-07-19 (a polyglot task killed mid-fix by the path-only key).
func TestWriteFileIterationNotRepeat(t *testing.T) {
	ctx := &AgentContext{}
	versions := []string{
		`{"path":"main.py.c","content":"int main(){ return 0; }"}`,
		`{"path":"main.py.c","content":"int main(){ printf(\"x\"); return 0; }"}`,
		`{"path":"main.py.c","content":"#include <stdio.h>\nint main(){ printf(\"x\"); return 0; }"}`,
	}
	for i, v := range versions {
		_, _, repeating := recordToolCall(ctx, "write_file", json.RawMessage(v))
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
		if _, _, r := recordToolCall(ctx, "write_file", json.RawMessage(v)); r {
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
		`{"path":"m.py","content":"def f():\nreturn 1"}`,         // broken indent
		`{"path":"m.py","content":"def f():\n    return 1"}`,     // fixed (4)
		`{"path":"m.py","content":"def f():\n        return 1"}`, // 8-space
	}
	for i, v := range versions {
		if _, _, r := recordToolCall(ctx, "write_file", json.RawMessage(v)); r {
			t.Fatalf("indentation change at write %d flagged as reassertion", i+1)
		}
	}
}

// May 10 2026 BiasBusters #30 — locks the reasoning-repetition
// detector against regression. Prefix-match similarity over normalized
// reasoning openings; ≥2 consecutive identical openings triggers
// intervention. Single-turn repeats and prose-free turns must NOT fire.

func TestRecordReasoningTriggersOnConsecutiveRepeat(t *testing.T) {
	ctx := &AgentContext{}
	// Turn 1: first reasoning. No intervention.
	if msg, _, fired := recordReasoning(ctx, "Now I need to read the file to understand the structure."); fired || msg != "" {
		t.Fatalf("turn 1: expected no fire, got fired=%v msg=%q", fired, msg)
	}
	// Turn 2: same opening prefix. count=1 (not yet at threshold of 2).
	if msg, _, fired := recordReasoning(ctx, "Now I need to read the file to understand the structure."); fired || msg != "" {
		t.Fatalf("turn 2: expected no fire (count=1, threshold=2), got fired=%v msg=%q", fired, msg)
	}
	// Turn 3: same opening prefix again. count=2. FIRES.
	msg, obs, fired := recordReasoning(ctx, "Now I need to read the file to understand the structure.")
	if !fired {
		t.Fatalf("turn 3: expected intervention, got no fire")
	}
	if !strings.Contains(msg, "Reasoning repetition") {
		t.Errorf("intervention message missing canonical prefix: %s", msg)
	}
	if !strings.Contains(msg, "3 consecutive turns") {
		t.Errorf("intervention should report 3 consecutive turns, got: %s", msg)
	}
	// The observation is what the caller renders its log line and its
	// agent_reasoning_intervention payload from, so it must carry the
	// same count the message reports and the snippet that repeated.
	if obs.Count != 3 {
		t.Errorf("observation count = %d, want 3 consecutive turns", obs.Count)
	}
	if obs.Snippet != normalizeReasoningSnippet("Now I need to read the file to understand the structure.") {
		t.Errorf("observation snippet = %q, want the normalized repeated opening", obs.Snippet)
	}
	// The detector owns the reset. Reading the streak back off ctx now
	// sees the cleared values — which is exactly why the observation is
	// returned instead.
	if ctx.ConsecutiveReasoningRepeats != 0 || ctx.LastReasoningSnippet != "" {
		t.Errorf("firing must clear the streak; got repeats=%d snippet=%q",
			ctx.ConsecutiveReasoningRepeats, ctx.LastReasoningSnippet)
	}
	// A fourth identical turn starts over rather than re-firing the
	// same loop immediately.
	if _, _, again := recordReasoning(ctx, "Now I need to read the file to understand the structure."); again {
		t.Error("the turn right after a fire must start a fresh streak")
	}
}

func TestRecordReasoningResetOnDivergence(t *testing.T) {
	ctx := &AgentContext{}
	recordReasoning(ctx, "Now I need to read the file.")
	recordReasoning(ctx, "Now I need to read the file.")
	// Turn 3: model commits to a different thought — counter resets.
	if _, _, fired := recordReasoning(ctx, "I have the file content. Now let me write the new version."); fired {
		t.Error("divergent reasoning should reset the counter, no intervention expected")
	}
	if ctx.ConsecutiveReasoningRepeats != 0 {
		t.Errorf("counter should reset to 0 after divergence, got %d", ctx.ConsecutiveReasoningRepeats)
	}
	// Turn 4: similar to turn 3 (the new pattern). count=1.
	if _, _, fired := recordReasoning(ctx, "I have the file content. Now let me write the new version."); fired {
		t.Error("turn 4 should be count=1 (one repeat), no fire yet")
	}
	// Turn 5: third identical → FIRES.
	if _, _, fired := recordReasoning(ctx, "I have the file content. Now let me write the new version."); !fired {
		t.Error("turn 5 should fire (count=2 of new pattern)")
	}
}

func TestRecordReasoningIgnoresEmptyTurns(t *testing.T) {
	ctx := &AgentContext{}
	recordReasoning(ctx, "Now I need to read the file.")
	// Turn 2: empty reasoning (model committed straight to action). Counter resets.
	if _, _, fired := recordReasoning(ctx, ""); fired {
		t.Error("empty reasoning should not fire")
	}
	if ctx.ConsecutiveReasoningRepeats != 0 || ctx.LastReasoningSnippet != "" {
		t.Errorf("empty reasoning should reset state; got repeats=%d snippet=%q",
			ctx.ConsecutiveReasoningRepeats, ctx.LastReasoningSnippet)
	}
	// Turn 3: same prose as turn 1, but the empty turn 2 broke the streak.
	// Should be treated as a fresh start, not count=2.
	recordReasoning(ctx, "Now I need to read the file.")
	if ctx.ConsecutiveReasoningRepeats != 0 {
		t.Errorf("post-empty: counter should be 0 (fresh start), got %d", ctx.ConsecutiveReasoningRepeats)
	}
}

func TestRecordReasoningNormalizesWhitespace(t *testing.T) {
	ctx := &AgentContext{}
	recordReasoning(ctx, "  Now I  need\nto    read the file.\n")
	recordReasoning(ctx, "now i need to read the file.")
	// Both should normalize to the same prefix → count=1.
	if ctx.ConsecutiveReasoningRepeats != 1 {
		t.Errorf("normalized whitespace+case should match; got count=%d, snippet=%q",
			ctx.ConsecutiveReasoningRepeats, ctx.LastReasoningSnippet)
	}
}

func TestRecordReasoningRespectsPrefixLength(t *testing.T) {
	// Two reasonings that share the first 80 chars but diverge later
	// should still match — that's the design (we want the OPENING to
	// be the signal). Let me confirm the prefix-match behavior.
	a := "Looking at the existing dashboard.html, I see the basic Flask template that needs to be transformed into a metrics view."
	b := "Looking at the existing dashboard.html, I see the basic Flask template that needs to be expanded with three KPI cards."
	ctx := &AgentContext{}
	recordReasoning(ctx, a)
	recordReasoning(ctx, b)
	// First 80 chars match → count should advance.
	if ctx.ConsecutiveReasoningRepeats == 0 {
		t.Errorf("expected prefix match to advance counter; got count=0, snippet=%q",
			ctx.LastReasoningSnippet)
	}
}

func TestRecordReasoningDoesNotFireOnSingleRepeat(t *testing.T) {
	ctx := &AgentContext{}
	recordReasoning(ctx, "Looking at the file...")
	if _, _, fired := recordReasoning(ctx, "Looking at the file..."); fired {
		t.Error("single repeat (turn 2 = turn 1) should not fire — needs 2 consecutive repeats")
	}
}

func TestNormalizeReasoningSnippet(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"   \n  ", ""},
		{"Hello World", "hello world"},
		{"  HELLO\n\tWORLD  ", "hello world"},
		{strings.Repeat("a", 200), strings.Repeat("a", 80)},
	}
	for _, tc := range cases {
		if got := normalizeReasoningSnippet(tc.in); got != tc.want {
			t.Errorf("normalize(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// The uninstalled-dependency loop: `python3 -m flask run` fails because flask
// isn't in the sandbox. The steer must name the package and tell it to install.
func TestMissingModuleSteerPythonDashM(t *testing.T) {
	dir := t.TempDir()
	ctx := &AgentContext{WorkingDir: dir}
	out := "/usr/local/bin/python3: No module named flask\n"
	steer := missingModuleSteer(ctx, out)
	if !strings.Contains(steer, "flask") || !strings.Contains(steer, "pip install") {
		t.Errorf("expected install steer naming flask, got: %q", steer)
	}
}

// ModuleNotFoundError (import form), and a top-level package is extracted from
// a dotted submodule.
func TestMissingModuleSteerImportForm(t *testing.T) {
	dir := t.TempDir()
	ctx := &AgentContext{WorkingDir: dir}
	out := "ModuleNotFoundError: No module named 'flask.cli'\n"
	steer := missingModuleSteer(ctx, out)
	if !strings.Contains(steer, "pip install flask") {
		t.Errorf("expected `pip install flask` (top-level pkg), got: %q", steer)
	}
}

// When a requirements.txt exists, prefer installing the whole manifest.
func TestMissingModuleSteerPrefersRequirements(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "requirements.txt"), []byte("flask\n"), 0644); err != nil {
		t.Fatal(err)
	}
	ctx := &AgentContext{WorkingDir: dir}
	out := "No module named flask\n"
	steer := missingModuleSteer(ctx, out)
	if !strings.Contains(steer, "pip install -r requirements.txt") {
		t.Errorf("expected requirements.txt steer, got: %q", steer)
	}
}

func TestMissingModuleSteerNoModuleError(t *testing.T) {
	ctx := &AgentContext{WorkingDir: t.TempDir()}
	if s := missingModuleSteer(ctx, "Total: 42\n"); s != "" {
		t.Errorf("expected empty steer for unrelated output, got: %q", s)
	}
}

// The case-typo loop: ran `pip install -r Requirements.txt` while the real
// file is `requirements.txt`. The steer must name the actual file.
func TestMissingFileSteerCaseMismatch(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "requirements.txt"), []byte("flask\n"), 0644); err != nil {
		t.Fatal(err)
	}
	ctx := &AgentContext{WorkingDir: dir}
	out := "ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'Requirements.txt'\n"
	steer := missingFileSteer(ctx, out)
	if !strings.Contains(steer, "requirements.txt") || !strings.Contains(steer, "case") {
		t.Errorf("expected case-mismatch steer naming requirements.txt, got: %q", steer)
	}
}

// A genuinely absent file (no case-variant) must NOT produce a steer — we
// never invent an anchor for a file that doesn't exist.
func TestMissingFileSteerNoVariant(t *testing.T) {
	dir := t.TempDir()
	ctx := &AgentContext{WorkingDir: dir}
	out := "cat: nope.txt: No such file or directory\n"
	if s := missingFileSteer(ctx, out); s != "" {
		t.Errorf("expected no steer when no case-variant exists, got: %q", s)
	}
}

// Shell-style error (filename before the colon) is also recognized.
func TestMissingFileSteerShellShape(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "main.py"), []byte("print(1)\n"), 0644); err != nil {
		t.Fatal(err)
	}
	ctx := &AgentContext{WorkingDir: dir}
	out := "python: Main.py: No such file or directory\n"
	steer := missingFileSteer(ctx, out)
	if !strings.Contains(steer, "main.py") {
		t.Errorf("expected steer naming main.py, got: %q", steer)
	}
}

func TestTracebackSteerNamesFixSite(t *testing.T) {
	ctx := &AgentContext{WorkingDir: "/workspace"}
	out := "Traceback (most recent call last):\n" +
		"  File \"/workspace/_agenttest/app.py\", line 14, in get_item\n" +
		"    return jsonify(items[item_id + 1])\n" +
		"IndexError: list index out of range\n"
	steer := tracebackSteer(ctx, out)
	for _, want := range []string{"get_item", "line 14", "IndexError", "function:get_item"} {
		if !strings.Contains(steer, want) {
			t.Errorf("steer missing %q:\n%s", want, steer)
		}
	}
}

func TestTracebackSteerNoTraceback(t *testing.T) {
	ctx := &AgentContext{WorkingDir: "/workspace"}
	if s := tracebackSteer(ctx, "Total inventory value: $237\n"); s != "" {
		t.Errorf("expected empty steer for non-traceback output, got: %s", s)
	}
}

// Environment errors (missing package) aren't code-localization targets —
// steering/banning would loop on an unfixable import.
func TestTracebackSteerSkipsModuleNotFound(t *testing.T) {
	ctx := &AgentContext{WorkingDir: "/workspace"}
	out := "Traceback (most recent call last):\n" +
		"  File \"/workspace/snake_game.py\", line 1, in <module>\n" +
		"    import pygame\n" +
		"ModuleNotFoundError: No module named 'pygame'\n"
	if s := tracebackSteer(ctx, out); s != "" {
		t.Errorf("should not steer on ModuleNotFoundError, got: %s", s)
	}
}

// The deepest frame is usually stdlib; the fix site is the deepest PROJECT
// frame (the user line that called into the library).
func TestTracebackSteerSkipsStdlib(t *testing.T) {
	ctx := &AgentContext{WorkingDir: "/workspace"}
	out := "Traceback (most recent call last):\n" +
		"  File \"/workspace/app.py\", line 5, in main\n" +
		"    data = json.loads(raw)\n" +
		"  File \"/usr/lib/python3.9/json/__init__.py\", line 346, in loads\n" +
		"    return _default_decoder.decode(s)\n" +
		"ValueError: Expecting value\n"
	steer := tracebackSteer(ctx, out)
	if !strings.Contains(steer, "app.py") || !strings.Contains(steer, "function:main") {
		t.Errorf("should pick project frame app.py:main, got: %s", steer)
	}
	if strings.Contains(steer, "json/__init__") {
		t.Errorf("should NOT point at stdlib, got: %s", steer)
	}
}

// The missing-binary loop (observed 2026-07-18): `git clone ...` in a
// sandbox without git. The steer must name the binary, state that
// apt-get can't work (non-root, read-only), and point at alternatives.
func TestMissingCommandSteerBashForm(t *testing.T) {
	out := "bash: line 1: git: command not found\n"
	steer := missingCommandSteer(out)
	if !strings.Contains(steer, "`git`") || !strings.Contains(steer, "CANNOT be installed") {
		t.Errorf("expected missing-command steer naming git, got: %q", steer)
	}
	if strings.Contains(steer, "apt-get install") {
		t.Errorf("steer must not suggest apt-get install (impossible in sandbox): %q", steer)
	}
}

// dash/sh abbreviates: "sh: 1: sqlite3: not found".
func TestMissingCommandSteerShForm(t *testing.T) {
	out := "sh: 1: sqlite3: not found\n"
	steer := missingCommandSteer(out)
	if !strings.Contains(steer, "`sqlite3`") {
		t.Errorf("expected steer naming sqlite3, got: %q", steer)
	}
}

// A full path is reduced to its basename.
func TestMissingCommandSteerPathBasename(t *testing.T) {
	out := "bash: line 3: /usr/local/bin/terraform: command not found\n"
	steer := missingCommandSteer(out)
	if !strings.Contains(steer, "`terraform`") {
		t.Errorf("expected basename terraform, got: %q", steer)
	}
}

// Bare "<name>: not found" without an sh prefix must NOT fire — program
// output legitimately prints "config.yaml: not found" shapes.
func TestMissingCommandSteerNoFalsePositive(t *testing.T) {
	if s := missingCommandSteer("config.yaml: not found\n"); s != "" {
		t.Errorf("expected no steer for non-shell not-found line, got: %q", s)
	}
	if s := missingCommandSteer("all tests passed\n"); s != "" {
		t.Errorf("expected no steer for clean output, got: %q", s)
	}
}

// The broken-verification-command loop (observed 2026-07-19, regex-chess): the
// model verifies with `python3 -c "...; def f(): ..."` — a multi-statement
// script that can't parse on a -c line — and the SyntaxError is in the
// command, not the solution. Steer must move the test to a file.
func TestBrokenInlineScriptSteerFires(t *testing.T) {
	cmd := `python3 -c "import json, re; def all_legal_next_positions(fen): return []"`
	out := "  File \"<string>\", line 1\n    import json, re; def all_legal\n                     ^\nSyntaxError: invalid syntax"
	steer := brokenInlineScriptSteer(cmd, out)
	if steer == "" || !strings.Contains(steer, "inline `-c`") || !strings.Contains(steer, ".py") {
		t.Errorf("expected broken-inline-script steer, got: %q", steer)
	}
}

// A syntax error in a REAL file (not a -c one-liner) must NOT match — that's
// a solution bug tracebackSteer localizes, not a broken verify command.
func TestBrokenInlineScriptSteerIgnoresRealFile(t *testing.T) {
	cmd := `python3 solution.py`
	out := "  File \"solution.py\", line 12\n    def f(\n         ^\nSyntaxError: invalid syntax"
	if s := brokenInlineScriptSteer(cmd, out); s != "" {
		t.Errorf("expected no steer for a real-file syntax error, got: %q", s)
	}
}

// No SyntaxError at all → no steer.
func TestBrokenInlineScriptSteerNoSyntaxError(t *testing.T) {
	if s := brokenInlineScriptSteer(`python3 -c "print(1)"`, "1\n"); s != "" {
		t.Errorf("expected no steer on clean output, got: %q", s)
	}
}

// Truncation robustness: the sandbox clipped the output before the
// "SyntaxError:" line, leaving only the "<string>" frame. The steer must
// still fire (2026-07-19 regression — the keyword gate missed this).
func TestBrokenInlineScriptSteerTruncatedOutput(t *testing.T) {
	cmd := `python3 -c "import json, re; def all_legal(fen): return []"`
	out := `  File "<string>", line 1` + "\n" + `    import json, re; def all_legal(fen): return []`
	if s := brokenInlineScriptSteer(cmd, out); s == "" {
		t.Error("steer must fire on a <string> frame even when SyntaxError is truncated away")
	}
}

// #147 review #9: the bash command-not-found steer must require a real bash
// diagnostic prefix, not fire on the phrase in ordinary program output.
func TestMissingCommandSteerRequiresShellPrefix(t *testing.T) {
	if s := missingCommandSteer("bash: line 1: git: command not found\n"); s == "" {
		t.Error("real bash diagnostic must fire")
	}
	if s := missingCommandSteer("bash: git: command not found\n"); s == "" {
		t.Error("bash diagnostic without line-number must fire")
	}
	// Program output that merely prints the phrase must NOT fire.
	if s := missingCommandSteer(`print("mytool: command not found")` + "\nmytool: command not found\n"); s != "" {
		t.Errorf("must not fire on program output: %q", s)
	}
}

// #147 review #11: don't misfire on `python -c "exec(open('f').read())"` —
// the SyntaxError is in file f, not the one-liner.
func TestBrokenInlineScriptSteerSkipsExec(t *testing.T) {
	cmd := `python3 -c "exec(open('solution.py').read())"`
	out := "  File \"<string>\", line 1\n    def broken(\nSyntaxError: invalid syntax"
	if s := brokenInlineScriptSteer(cmd, out); s != "" {
		t.Errorf("must not fire when -c execs external code: %q", s)
	}
}

// The live failure: three structural_edit calls on function:index in a Flask
// app whose HTML/JS template is a module-level constant. index() is two lines
// and holds none of the JavaScript being fixed, so every attempt was doomed —
// but each carried a different body, so a whole-args signature saw three
// distinct calls and the repeat detector stayed silent until the failure cap
// killed the turn.
func TestStructuralEditRepeatsKeyOnSelectorNotContent(t *testing.T) {
	ctx := &AgentContext{}
	args := func(content string) json.RawMessage {
		return json.RawMessage(`{"path":"app.py","selector":"function:index","content":` +
			mustJSONString(content) + `}`)
	}

	var msg string
	var fired bool
	for i, body := range []string{"first attempt", "second, rewritten", "third, different again"} {
		msg, _, fired = recordToolCall(ctx, "structural_edit", args(body))
		if fired {
			if i == 0 {
				t.Fatal("fired on the first call — that is not a repeat")
			}
			break
		}
	}
	if !fired {
		t.Fatal("three attempts at the same selector never registered as repetition")
	}
	// The message has to name the selector as the cause, or the model reads it
	// as "try again differently" and rewrites the body a fourth time.
	for _, want := range []string{"function:index", "SELECTOR", "edit_file", "insert_after"} {
		if !strings.Contains(msg, want) {
			t.Errorf("escalation missing %q:\n%s", want, msg)
		}
	}
}

// A different selector is a different attempt and must not be conflated.
func TestStructuralEditDifferentSelectorsAreNotRepeats(t *testing.T) {
	ctx := &AgentContext{}
	for _, sel := range []string{"function:a", "function:b", "function:c", "class:D"} {
		args := json.RawMessage(`{"path":"app.py","selector":"` + sel + `","content":"x"}`)
		if _, _, fired := recordToolCall(ctx, "structural_edit", args); fired {
			t.Fatalf("distinct selector %q was treated as a repeat", sel)
		}
	}
}

func mustJSONString(s string) string {
	b, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// Run 9 (2026-08-02) emitted the same replace_lines call on turns 2 and 3
// against a rejection that named the file, the line, the cause and two
// concrete fixes. The repetition detector needs three occurrences in its
// window and only steers the NEXT turn, so an identical pair never reached
// it and the run died on the three-strike breaker with the file untouched.
func TestAnIdenticalResendOfARejectedCallIsRefused(t *testing.T) {
	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	args := json.RawMessage(`{"path":"app.py","start_line":201,"end_line":201,"content":"x"}`)

	if refusal := identicalRetryRefusal(ctx, "replace_lines", args); refusal != "" {
		t.Fatalf("a first attempt must run: %s", refusal)
	}
	recordFailedToolCall(ctx, "replace_lines", args, "stops a render loop: `draw` now runs once")

	refusal := identicalRetryRefusal(ctx, "replace_lines", args)
	if refusal == "" {
		t.Fatal("the identical re-send must be refused before it executes")
	}
	// The original rejection has to come back with it — the model needs the
	// reason, not just the fact that it repeated itself.
	if !strings.Contains(refusal, "stops a render loop") {
		t.Errorf("refusal drops the original reason:\n%s", refusal)
	}
	if !strings.Contains(refusal, "read_file") {
		t.Errorf("refusal gives no way forward:\n%s", refusal)
	}
}

func TestTheRefusalIsScopedToCallsThatFailed(t *testing.T) {
	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	read := json.RawMessage(`{"path":"app.py"}`)

	// Re-reading a file after editing it is byte-identical and correct.
	if refusal := identicalRetryRefusal(ctx, "read_file", read); refusal != "" {
		t.Errorf("a repeated successful call was refused: %s", refusal)
	}
	// A different call to the same tool is not the same call.
	args := json.RawMessage(`{"path":"a.py","start_line":1,"end_line":1,"content":"x"}`)
	recordFailedToolCall(ctx, "replace_lines", args, "boom")
	other := json.RawMessage(`{"path":"a.py","start_line":2,"end_line":2,"content":"x"}`)
	if refusal := identicalRetryRefusal(ctx, "replace_lines", other); refusal != "" {
		t.Errorf("a different call was refused: %s", refusal)
	}
}

func TestASucceedingCallClearsItsOwnRejection(t *testing.T) {
	// An edit rejected for a stale range works after a re-read. The memory
	// must not outlive the condition that caused it.
	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	args := json.RawMessage(`{"path":"a.py","start_line":1,"end_line":1,"content":"x"}`)
	recordFailedToolCall(ctx, "replace_lines", args, "stale range")
	clearFailedToolCall(ctx, "replace_lines", args)
	if refusal := identicalRetryRefusal(ctx, "replace_lines", args); refusal != "" {
		t.Errorf("rejection outlived the failure: %s", refusal)
	}
}

// The refusal path incremented consecutiveErrors and appended the failure
// path, then returned — while both stopping rules that read those live inside
// the post-execution failure branch it skips. Observed: four consecutive
// refusals of the same structural_edit in one run, no breaker, no ceiling; the
// model was refused cheaply and forever.
func TestStuckOnOnePathIsTheSharedBreakerCondition(t *testing.T) {
	if !stuckOnOnePath([]string{"todo.py", "todo.py", "todo.py"}) {
		t.Error("three failures on one file must count as stuck")
	}
	for _, paths := range [][]string{
		{"a.py", "b.py", "c.py"}, // grinding through multi-file work
		{"a.py", "a.py"},         // not yet three
		{"", "", ""},             // unnamed target proves nothing
		{"a.py", "a.py", "b.py"},
	} {
		if stuckOnOnePath(paths) {
			t.Errorf("wrongly reported stuck: %v", paths)
		}
	}
}

func TestRepeatedRefusalTellsTheUserRetryingWontHelp(t *testing.T) {
	msg := repeatedRefusalSummary("structural_edit", "todo.py", false)
	for _, want := range []string{"re-sent after being refused", "todo.py", "Nothing was written", "same wall"} {
		if !strings.Contains(msg, want) {
			t.Errorf("summary missing %q:\n%s", want, msg)
		}
	}
	if wrote := repeatedRefusalSummary("edit_file", "", true); !strings.Contains(wrote, "did land on disk") {
		t.Errorf("a run that wrote must say so:\n%s", wrote)
	}
}
