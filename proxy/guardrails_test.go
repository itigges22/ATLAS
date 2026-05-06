package main

import (
	"os"
	"path/filepath"
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

func TestValidateShellCommandAllowsStderrRedirects(t *testing.T) {
	// stderr→stdout merge (`2>&1`), stderr→file, and `&>` are all
	// standard verification idioms. The early version of the regex
	// treated any `>` as a "truncating redirect" and rejected
	// `python app.py 2>&1` — confirmed in May 2026 user logs where
	// every verification attempt with `2>&1` was bounced. Regression
	// tests for each shape so it doesn't drift back.
	allowed := []string{
		"python app.py 2>&1",
		"python3 -c 'import flask' 2>&1",
		"pytest -v 2> errors.log",
		"curl http://localhost:5000/ 2>/dev/null",
		"node app.js >& output.log",        // bash &> shorthand variant
		"go test ./... 2>&1 | tee out.log", // pipe + merge
	}
	for _, cmd := range allowed {
		if got := validateShellCommand(cmd); got != "" {
			t.Errorf("validateShellCommand(%q) rejected: %s", cmd, got)
		}
	}
}

func TestValidateShellCommandAllowsLogRedirectWithTrailingFlags(t *testing.T) {
	// Confirmed in May 2026 user logs: the model issued
	// `python app.py > flask.log 2>&1 &` to background a flask server
	// for verification, and the guardrail rejected it. Root cause was
	// a too-greedy tail extraction that pulled in the `2>&1 &` after
	// the destination, defeating the .log/.out suffix exception.
	// Regression: every shape below has a build-artefact destination
	// followed by trailing flags that must NOT bleed into the path.
	allowed := []string{
		"python app.py > flask.log 2>&1 &",
		"python app.py > server.out 2>&1",
		"python app.py >flask.log 2>&1",
		"node app.js > app.log 2>&1 &",
		"go run main.go > out.log 2>/dev/null",
		"python app.py > /dev/null 2>&1 &",
	}
	for _, cmd := range allowed {
		if got := validateShellCommand(cmd); got != "" {
			t.Errorf("validateShellCommand(%q) rejected: %s", cmd, got)
		}
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

func TestResolveAgentPathTranslatesHostPrefix(t *testing.T) {
	ctx := &AgentContext{
		WorkingDir:     "/workspace",
		HostWorkingDir: "/home/isaac/snake",
	}
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"absolute host path → container", "/home/isaac/snake/app.py", "/workspace/app.py"},
		{"absolute host path nested", "/home/isaac/snake/templates/index.html", "/workspace/templates/index.html"},
		{"host root itself", "/home/isaac/snake", "/workspace"},
		{"host path with trailing slash", "/home/isaac/snake/", "/workspace"},
		{"relative path → joined", "app.py", "/workspace/app.py"},
		{"absolute non-host path passes through", "/etc/passwd", "/etc/passwd"},
		{"host-prefix lookalike does not match", "/home/isaac/snakebar/app.py", "/home/isaac/snakebar/app.py"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := resolveAgentPath(ctx, tc.in); got != tc.want {
				t.Errorf("resolveAgentPath(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestResolveAgentPathNoHostMappingFallsBack(t *testing.T) {
	// Without HostWorkingDir set (dev/test mode), absolute paths
	// pass through and relative paths join against WorkingDir —
	// matching the original resolvePath behaviour.
	ctx := &AgentContext{WorkingDir: "/tmp/proj"}
	if got := resolveAgentPath(ctx, "/home/x/file.py"); got != "/home/x/file.py" {
		t.Errorf("got %q, want pass-through", got)
	}
	if got := resolveAgentPath(ctx, "src/x.py"); got != "/tmp/proj/src/x.py" {
		t.Errorf("got %q, want joined", got)
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

func TestLooksLikeStubHTMLPlaceholder(t *testing.T) {
	// Exactly the shape the model emitted in the May 6 flask run.
	stub := "<!DOCTYPE html>\n<html>\n<head>\n    <title>Pricing</title>\n</head>\n<body>\n    <h1>Pricing Page</h1>\n</body>\n</html>"
	if got := looksLikeStub("templates/pricing.html", stub); got == "" {
		t.Error("stub HTML should be rejected")
	}
}

func TestLooksLikeStubAcceptsRealHTML(t *testing.T) {
	// A real templated page with content — not a stub.
	real := `<!DOCTYPE html>
<html><head><title>Pricing</title></head>
<body>
  <h1>Pricing Page</h1>
  <p>Choose the plan that fits your needs.</p>
  <ul>
    <li>Free — $0/mo: 1 project, 100 calls/day</li>
    <li>Pro — $20/mo: unlimited projects, 10k calls/day</li>
    <li>Team — $80/mo: SSO, audit log, priority support</li>
  </ul>
  <p>All plans include a 14-day trial.</p>
</body></html>`
	if got := looksLikeStub("templates/pricing.html", real); got != "" {
		t.Errorf("real HTML rejected as stub: %s", got)
	}
}

func TestLooksLikeStubPython(t *testing.T) {
	if got := looksLikeStub("widget.py", "pass"); got == "" {
		t.Error("`pass`-only file should be rejected")
	}
	if got := looksLikeStub("widget.py", "# TODO: implement\n"); got == "" {
		t.Error("`# TODO`-only file should be rejected")
	}
	// Real one-liner — not a stub.
	if got := looksLikeStub("widget.py", "from flask import Blueprint\nbp = Blueprint('widget', __name__)\n"); got != "" {
		t.Errorf("real one-liner rejected: %s", got)
	}
}

func TestLooksLikeStubEmpty(t *testing.T) {
	if got := looksLikeStub("a.txt", ""); got == "" {
		t.Error("empty content should be rejected")
	}
}

func TestLooksLikeStubAcceptsShortShellScript(t *testing.T) {
	// Short content is fine if it has substance.
	if got := looksLikeStub("scripts/probe.sh", "#!/bin/sh\nexec curl -sf http://localhost:8080/health\n"); got != "" {
		t.Errorf("real shell one-liner rejected: %s", got)
	}
}

func TestPatternMatchHintNewFileInDirOfSiblings(t *testing.T) {
	dir := t.TempDir()
	// 3 siblings with the same extension.
	for _, name := range []string{"index.html", "about.html", "contact.html"} {
		os.WriteFile(filepath.Join(dir, name), []byte("<html/>"), 0o644)
	}
	target := filepath.Join(dir, "pricing.html")
	if got := patternMatchHint(target, "<html/>"); got == "" {
		t.Error("expected hint when no siblings have been read")
	}
	// Now register a sibling read — hint should disappear.
	patternReadTracker.add(filepath.Join(dir, "index.html"))
	if got := patternMatchHint(target, "<html/>"); got != "" {
		t.Errorf("hint should disappear after sibling read: %s", got)
	}
}

func TestPatternMatchHintSkipsExistingFiles(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.html"), []byte("a"), 0o644)
	os.WriteFile(filepath.Join(dir, "b.html"), []byte("b"), 0o644)
	target := filepath.Join(dir, "a.html") // exists already → not a new write
	if got := patternMatchHint(target, "x"); got != "" {
		t.Errorf("editing existing file should not trip hint: %s", got)
	}
}

func TestPatternMatchHintSkipsLonelyDir(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "only.go"), []byte("package x"), 0o644)
	target := filepath.Join(dir, "new.go")
	if got := patternMatchHint(target, "package x"); got != "" {
		t.Errorf("single-sibling dir shouldn't trip hint: %s", got)
	}
}

func TestResolveVerifyTargetEnvAndConfig(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ATLAS_VERIFY_IN", "")
	if got := resolveVerifyTarget(dir); got != "sandbox" {
		t.Errorf("default = %q, want sandbox", got)
	}
	t.Setenv("ATLAS_VERIFY_IN", "host")
	if got := resolveVerifyTarget(dir); got != "host" {
		t.Errorf("env-host = %q, want host", got)
	}
	// Per-project config wins.
	os.MkdirAll(filepath.Join(dir, ".atlas"), 0o755)
	os.WriteFile(filepath.Join(dir, ".atlas", "config.toml"),
		[]byte("[execution]\ntarget = \"sandbox\"\n"), 0o644)
	if got := resolveVerifyTarget(dir); got != "sandbox" {
		t.Errorf("config override = %q, want sandbox", got)
	}
}
