package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

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

// The missing-binary loop (TB2 bench 2026-07-18): `git clone ...` in a
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
