package main

import (
	"strings"
	"testing"
)

func TestSanitizeFileContentStripsMarkdownWrapper(t *testing.T) {
	// The exact failure mode from /home/isaac/snake/templates/index.html:
	// LLM prose preamble + ```html fence + actual HTML + closing fence +
	// numbered-list explanation containing literal {{ url_for(...) }}.
	in := strings.Join([]string{
		"Looking at the task, I need to create a complete index.html file.",
		"",
		"```html",
		"<!DOCTYPE html>",
		"<html><body>hi</body></html>",
		"```",
		"",
		"This file:",
		"1. Renders correctly",
		"2. **Includes Jinja syntax** ({{ url_for(...) }})",
	}, "\n")
	got, sanitized := sanitizeFileContent("templates/index.html", in)
	if !sanitized {
		t.Fatal("sanitized=false, want true")
	}
	want := "<!DOCTYPE html>\n<html><body>hi</body></html>"
	if got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestSanitizeFileContentLeavesCleanCodeAlone(t *testing.T) {
	in := "def foo():\n    return 1\n"
	got, sanitized := sanitizeFileContent("foo.py", in)
	if sanitized {
		t.Errorf("sanitized=true on clean input; should be no-op")
	}
	if got != in {
		t.Errorf("got %q, want %q (no fences → no change)", got, in)
	}
}

func TestSanitizeFileContentLeavesMarkdownFilesAlone(t *testing.T) {
	// Fences are legitimate content in .md files.
	in := "# Title\n\n```python\nprint('hi')\n```\n"
	got, sanitized := sanitizeFileContent("README.md", in)
	if sanitized {
		t.Errorf("sanitized=true on .md; should pass through")
	}
	if got != in {
		t.Errorf("content changed for .md file")
	}
}

func TestSanitizeFileContentHandlesUnmatchedFence(t *testing.T) {
	// Truncated response: opener but no closer. Take everything after
	// the opener (better than discarding the file).
	in := "Here's the code:\n\n```python\ndef foo():\n    return 1\n"
	got, sanitized := sanitizeFileContent("foo.py", in)
	if !sanitized {
		t.Fatal("sanitized=false, want true (opener present)")
	}
	if !strings.Contains(got, "def foo()") {
		t.Errorf("lost the code body: %q", got)
	}
	if strings.Contains(got, "Here's the code") {
		t.Errorf("kept the prose preamble: %q", got)
	}
}

func TestSanitizeFileContentPreservesTrailingNewline(t *testing.T) {
	in := "```python\ndef foo():\n    pass\n```\n"
	got, sanitized := sanitizeFileContent("foo.py", in)
	if !sanitized {
		t.Fatal("sanitized=false, want true")
	}
	if !strings.HasSuffix(got, "\n") {
		t.Errorf("dropped trailing newline: %q", got)
	}
}

func TestValidateShellCommandBlocksDestructiveVerbs(t *testing.T) {
	cases := []struct {
		name string
		cmd  string
	}{
		{"plain rm", "rm /workspace/foo.py"},
		{"rm -rf", "rm -rf templates"},
		{"mv", "mv templates venv/templates"},
		{"cp overwrite", "cp old.py new.py"},
		{"chained mv", "cd /workspace && mv app.py venv/"},
		{"find -delete", "find . -name '*.tmp' -delete"},
		{"find -exec rm", "find . -type f -exec rm {} \\;"},
		{"truncating redirect", "echo bad > /workspace/app.py"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := validateShellCommand(tc.cmd); got == "" {
				t.Errorf("validateShellCommand(%q) = empty, want rejection", tc.cmd)
			}
		})
	}
}

func TestValidateShellCommandAllowsBuildAndTest(t *testing.T) {
	cases := []string{
		"python app.py",
		"pytest tests/",
		"npm run build",
		"go test ./...",
		"cd /workspace && python -m flask run",
		"ls -la templates/",
		"cat app.py",
		"curl -s http://localhost:5000/",
		"grep -r 'TODO' src/",
		"echo 'progress' > /dev/null",
		"python app.py >> server.log",
		"pytest -v 2> errors.log",
	}
	for _, cmd := range cases {
		if got := validateShellCommand(cmd); got != "" {
			t.Errorf("validateShellCommand(%q) rejected: %s", cmd, got)
		}
	}
}

func TestValidateShellCommandAllowsDevNullRedirect(t *testing.T) {
	// /dev/null is the "discard output" idiom; never user-data.
	if got := validateShellCommand("python -c 'print(1)' > /dev/null"); got != "" {
		t.Errorf("rejected /dev/null redirect: %s", got)
	}
}

func TestValidateShellCommandBlocksBashCBypass(t *testing.T) {
	// The deny-list is bypassable if the model wraps the destructive
	// verb inside `bash -c "..."`. Roo Code's regression test case.
	cases := []string{
		`bash -c "rm -rf foo"`,
		`sh -c 'mv templates venv/'`,
		`zsh -c "echo malicious"`,
		`dash -c "find . -delete"`,
		`eval "rm -rf $HOME"`,
		`eval $command`,
	}
	for _, cmd := range cases {
		if got := validateShellCommand(cmd); got == "" {
			t.Errorf("validateShellCommand(%q) = empty, want rejection", cmd)
		}
	}
}

func TestValidateShellCommandStillAllowsLegitShellWork(t *testing.T) {
	// `bash -c` is the bypass; bash with no -c (or other flags) is fine.
	// `python -c` is a common, legit verification idiom and should pass.
	allowed := []string{
		"bash --version",
		"python -c 'import flask; print(flask.__version__)'",
		"node -e 'console.log(1+1)'",
		"git log -c",
	}
	for _, cmd := range allowed {
		if got := validateShellCommand(cmd); got != "" {
			t.Errorf("validateShellCommand(%q) rejected: %s", cmd, got)
		}
	}
}

func TestIsFixIntentMessage(t *testing.T) {
	fixIntents := []string{
		"fix the bug in app.py",
		"the form submission is broken",
		"why isn't this rendering",
		"the page won't load",
		"I'm getting an error",
		"can you verify it works",
	}
	for _, m := range fixIntents {
		if !isFixIntentMessage(m) {
			t.Errorf("isFixIntentMessage(%q) = false, want true", m)
		}
	}
	notFix := []string{
		"add a logout button to the header",
		"create a new flask route for /admin",
		"write a test for the login function",
		"hi", // doesn't trip — bare greeting
	}
	for _, m := range notFix {
		if isFixIntentMessage(m) {
			t.Errorf("isFixIntentMessage(%q) = true, want false", m)
		}
	}
}

func TestIsVerificationCommand(t *testing.T) {
	verifies := []string{
		"pytest tests/",
		"python app.py",
		"python3 -m pytest",
		"go test ./...",
		"go build",
		"cargo test",
		"npm test",
		"npm run build",
		"curl http://localhost:5000/",
		"make test",
		"ruff check src/",
		"mypy app.py",
	}
	for _, cmd := range verifies {
		if !isVerificationCommand(cmd) {
			t.Errorf("isVerificationCommand(%q) = false, want true", cmd)
		}
	}
	recon := []string{
		"ls -la",
		"cat app.py",
		"grep -r TODO src/",
		"find . -name '*.py'",
		"echo hello",
		"pip install flask",
	}
	for _, cmd := range recon {
		if isVerificationCommand(cmd) {
			t.Errorf("isVerificationCommand(%q) = true, want false (recon, not verification)", cmd)
		}
	}
}

func TestSplitShellSegmentsRespectsQuotes(t *testing.T) {
	// `;` inside single quotes shouldn't split.
	got := splitShellSegments(`echo 'a;b'; rm foo`)
	if len(got) != 2 {
		t.Fatalf("got %d segments, want 2: %v", len(got), got)
	}
	if !strings.Contains(got[0], "a;b") {
		t.Errorf("first segment lost the quoted body: %q", got[0])
	}
}
