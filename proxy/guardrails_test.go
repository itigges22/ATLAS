package main

import (
	"fmt"
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

func TestSanitizeFileContentLeavesInteriorFenceAlone(t *testing.T) {
	// A fenced usage example inside a docstring partway through a real
	// module. The fence is legitimate content — stripping to the fence
	// body would discard the code before and after it.
	var b strings.Builder
	for i := 0; i < 38; i++ {
		fmt.Fprintf(&b, "def fn_%d():\n", i)
	}
	b.WriteString("def frobnicate(x):\n")
	b.WriteString("    \"\"\"Frobnicate x.\n\n")
	b.WriteString("    Example:\n\n")
	b.WriteString("    ```python\n")
	b.WriteString("    frobnicate(1)\n")
	b.WriteString("    ```\n")
	b.WriteString("    \"\"\"\n")
	b.WriteString("    return x + 1\n")
	for i := 0; i < 70; i++ {
		fmt.Fprintf(&b, "def tail_%d():\n", i)
	}
	in := b.String()
	got, sanitized := sanitizeFileContent("frob.py", in)
	if sanitized {
		t.Error("sanitized=true for interior docstring fence; should pass through")
	}
	if got != in {
		t.Errorf("content changed for interior fence:\n%q", got)
	}
}

func TestSanitizeFileContentLeavesTopDocstringFenceAlone(t *testing.T) {
	// Fence near the top of the file, but inside a module docstring —
	// the docstring marker before the fence disqualifies the wrapper
	// interpretation, and the code after the closing fence must survive.
	in := strings.Join([]string{
		`"""Frobnicate.`,
		"",
		"```python",
		"frob()",
		"```",
		`"""`,
		"def frob():",
		"    return 1",
	}, "\n")
	got, sanitized := sanitizeFileContent("frob.py", in)
	if sanitized {
		t.Error("sanitized=true for docstring-wrapped fence; should pass through")
	}
	if got != in {
		t.Errorf("content changed:\n%q", got)
	}
}

func TestSanitizeFileContentLeavesInlineDocstringFenceAlone(t *testing.T) {
	// The opening docstring line has code before the delimiter
	// (DOC = """...), so the fence inside it is legitimate content and the
	// assignment, closing delimiter, and trailing function must all survive.
	in := strings.Join([]string{
		`DOC = """Usage example:`,
		"```python",
		"x = 1",
		"```",
		`end"""`,
		"",
		"def f():",
		"    return 1",
	}, "\n")
	got, sanitized := sanitizeFileContent("example.py", in)
	if sanitized {
		t.Error("sanitized=true for inline-docstring-wrapped fence; should pass through")
	}
	if got != in {
		t.Errorf("content changed (data loss):\n%q", got)
	}
}

func TestSanitizeFileContentStripsWrapperWithProseCommentMention(t *testing.T) {
	// A genuine whole-file wrapper whose intro prose merely mentions a
	// comment marker must still be stripped (the marker is not at line start).
	in := strings.Join([]string{
		"Here's app.js with the /* config */ block rewritten:",
		"```javascript",
		"const x = 1;",
		"export default x;",
		"```",
	}, "\n")
	got, sanitized := sanitizeFileContent("app.js", in)
	if !sanitized {
		t.Fatal("sanitized=false; the wrapper should have been stripped")
	}
	if strings.Contains(got, "```") || strings.Contains(got, "Here's app.js") {
		t.Errorf("wrapper not fully stripped:\n%q", got)
	}
}

func TestSanitizeFileContentLeavesFenceFollowedByCodeAlone(t *testing.T) {
	// Opening fence at the top, but substantial content after the last
	// bare fence — not a whole-file wrapper.
	var b strings.Builder
	b.WriteString("```\n")
	b.WriteString("example()\n")
	b.WriteString("```\n")
	for i := 0; i < 20; i++ {
		fmt.Fprintf(&b, "line_%d = %d\n", i, i)
	}
	in := b.String()
	got, sanitized := sanitizeFileContent("data.py", in)
	if sanitized {
		t.Error("sanitized=true with 20 content lines after the fence; should pass through")
	}
	if got != in {
		t.Errorf("content changed:\n%q", got)
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

// Catastrophic commands stay blocked even though run_command is otherwise
// permissive — these would wipe the whole project, fork-bomb the sandbox, or
// destroy a block device, none of which the project-folder jail protects
// against on its own.
func TestValidateShellCommandBlocksCatastrophic(t *testing.T) {
	cases := []struct {
		name string
		cmd  string
	}{
		{"rm -rf root", "rm -rf /"},
		{"rm -rf workspace root", "rm -rf /workspace"},
		{"rm -rf workspace glob", "rm -rf /workspace/*"},
		{"rm -rf home", "rm -rf $HOME"},
		{"rm -rf dot", "rm -rf ."},
		{"rm -rf star", "rm -rf *"},
		{"rm -rf chained at root", "cd /workspace && rm -rf ."},
		{"find -delete", "find . -name '*.tmp' -delete"},
		{"find -exec rm", "find . -type f -exec rm {} \\;"},
		{"fork bomb", ":(){ :|:& };:"},
		{"mkfs", "mkfs.ext4 /dev/sda1"},
		{"dd to device", "dd if=/dev/zero of=/dev/sda bs=1M"},
		{"redirect to device", "echo x > /dev/sda"},
		{"bash -c wrapping rm -rf /", `bash -c "rm -rf /"`},
		{"eval wrapping rm -rf home", `eval "rm -rf $HOME"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := validateShellCommand(tc.cmd); got == "" {
				t.Errorf("validateShellCommand(%q) = empty, want rejection", tc.cmd)
			}
		})
	}
}

// File-management commands the OLD policy blocked are now allowed — the model
// shouldn't reinvent shell, and the sandbox jail bounds the blast radius to
// the project folder. This is the regression guard for the 2026-06 loosening.
func TestValidateShellCommandAllowsFileManagement(t *testing.T) {
	allowed := []string{
		"mv index.html templates/",
		"mv templates venv/templates",
		"cp old.py new.py",
		"cd /workspace && mv app.py src/",
		"rm app.py",                 // delete a specific file
		"rm -f stale.pyc",           // forced, but not recursive
		"rm -rf __pycache__",        // recursive, but a named subdir
		"rm -rf node_modules build", // named subdirs
		"mkdir -p static/js",
		"chmod +x run.sh",
		"sed -i 's/foo/bar/' app.py", // in-place content edit via shell
		"echo bad > app.py",          // truncating redirect into a project file
		"ln -s ../shared lib",
	}
	for _, cmd := range allowed {
		if got := validateShellCommand(cmd); got != "" {
			t.Errorf("validateShellCommand(%q) rejected: %s", cmd, got)
		}
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

func TestValidateShellCommandUnwrapsBashCForCatastrophic(t *testing.T) {
	// `bash -c "..."` / `eval "..."` are allowed wrappers now, but they must
	// not smuggle a catastrophic command past the denylist — we unwrap one
	// layer and re-check.
	blocked := []string{
		`bash -c "rm -rf /"`,
		`sh -c 'rm -rf /workspace'`,
		`dash -c "find . -delete"`,
		`eval "rm -rf $HOME"`,
		`bash -c ":(){ :|:& };:"`,
	}
	for _, cmd := range blocked {
		if got := validateShellCommand(cmd); got == "" {
			t.Errorf("validateShellCommand(%q) = empty, want rejection", cmd)
		}
	}
}

func TestValidateShellCommandAllowsLegitShellWork(t *testing.T) {
	// bash -c wrapping a benign command is fine now; `python -c` / `node -e`
	// verification idioms must pass.
	allowed := []string{
		"bash --version",
		`bash -c "python app.py"`,
		`sh -c 'pytest -q'`,
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

// May 8 2026 flask test surfaced this: model drifted from the real
// project root (/home/isaac/snake) to a phantom /workspace cwd in
// run_background, burning turns 8-11. The guard below catches the
// drift one turn earlier with a rejection that names the actual
// workingDir, so the model can self-correct in a single round-trip.
func TestValidateWorkingDirReferenceRejectsPhantomWorkspace(t *testing.T) {
	const wd = "/home/isaac/snake"
	cases := []string{
		"cd /workspace && python app.py",
		"cd /workspace && pip install flask",
		"python /workspace/app.py",
		"ls /workspace",
		"pytest /workspace/tests/",
		// trailing path components — must still flag
		"cd /workspace/templates && tree",
	}
	for _, cmd := range cases {
		t.Run(cmd, func(t *testing.T) {
			if got := validateWorkingDirReference(cmd, wd); got == "" {
				t.Errorf("validateWorkingDirReference(%q, %q) = empty, want rejection", cmd, wd)
			}
		})
	}
}

func TestValidateWorkingDirReferenceAllowsLegitWorkspaceProject(t *testing.T) {
	// When the project actually IS at /workspace (e.g. the docker-compose
	// default deployment), the guard must NOT reject — false rejects
	// would break legit setups.
	cases := []struct{ wd, cmd string }{
		{"/workspace", "cd /workspace && python app.py"},
		{"/workspace/myproject", "cd /workspace/myproject && pytest"},
		{"/workspace", "ls /workspace/templates"},
	}
	for _, tc := range cases {
		t.Run(tc.cmd, func(t *testing.T) {
			if got := validateWorkingDirReference(tc.cmd, tc.wd); got != "" {
				t.Errorf("validateWorkingDirReference(%q, wd=%q) rejected: %s", tc.cmd, tc.wd, got)
			}
		})
	}
}

func TestValidateWorkingDirReferenceIgnoresUnrelatedPaths(t *testing.T) {
	// The /workspace check must be precise — substring matches inside
	// other paths (e.g. /home/foo_workspace) and non-/workspace commands
	// must pass through untouched. Empty workingDir is also a no-op
	// (during early bootstrap before AgentContext is fully populated).
	const wd = "/home/isaac/snake"
	cases := []struct{ name, cmd string }{
		{"unrelated path with workspace substring", "ls /home/isaac/foo_workspace_dir/"},
		{"workspace word, no slash", "echo workspace"},
		{"no workspace at all", "python app.py"},
		{"build command", "pytest tests/"},
		{"npm command", "npm run build"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := validateWorkingDirReference(tc.cmd, wd); got != "" {
				t.Errorf("validateWorkingDirReference(%q, wd=%q) rejected: %s", tc.cmd, wd, got)
			}
		})
	}
	// Empty workingDir = no-op
	if got := validateWorkingDirReference("cd /workspace && python app.py", ""); got != "" {
		t.Errorf("validateWorkingDirReference with empty workingDir rejected: %s", got)
	}
}

func TestValidateRunCommandChainsBothGates(t *testing.T) {
	const wd = "/home/isaac/snake"
	// Shell-mutation gate fires first (more specific message). Use a
	// catastrophic command, since targeted file ops are now allowed.
	if got := validateRunCommand("rm -rf /", wd); got == "" || !strings.Contains(got, "rm") {
		t.Errorf("expected shell-mutation rejection mentioning rm, got %q", got)
	}
	// /workspace gate fires when shell-mutation is clean.
	if got := validateRunCommand("cd /workspace && python app.py", wd); got == "" || !strings.Contains(got, "/workspace") {
		t.Errorf("expected workspace rejection, got %q", got)
	}
	// Both clean → empty.
	if got := validateRunCommand("python app.py", wd); got != "" {
		t.Errorf("expected pass-through, got rejection %q", got)
	}
}

// May 9 2026 ast_edit destructive-stub case: model emitted only
// "<!DOCTYPE html>\n" (16B) for an entire <html>-element rewrite of a
// 120B file, ast_edit "succeeded", file destroyed, model declared
// "done". Guard catches this exact shape without false-rejecting
// realistic small replacements.
func TestValidateNotSuspiciouslyShrunkRejectsDestructiveStub(t *testing.T) {
	// Today's exact case.
	if got := validateNotSuspiciouslyShrunk("ast_edit", "templates/index.html", 120, 16); got == "" {
		t.Error("expected rejection for 120B → 16B replacement")
	}
	// Larger original, larger stub — still flagged.
	if got := validateNotSuspiciouslyShrunk("ast_edit", "app.py", 5000, 20); got == "" {
		t.Error("expected rejection for 5000B → 20B replacement")
	}
	// edit_file path covered by the same guard.
	if got := validateNotSuspiciouslyShrunk("edit_file", "app.py", 200, 8); got == "" {
		t.Error("expected rejection for edit_file 200B → 8B")
	}
	// May 10 2026: the 32B-just-passes failure that motivated bumping
	// the floor from 32 to 128. Model emitted exactly 32B for an HTML
	// body rewrite — clearly a stub but slipped past the v1 guard.
	if got := validateNotSuspiciouslyShrunk("ast_edit", "templates/dashboard.html", 2199, 32); got == "" {
		t.Error("expected rejection for 2199B → 32B (the May 10 boundary case)")
	}
	// 60B replacement of 2KB original — under the 64B floor.
	if got := validateNotSuspiciouslyShrunk("ast_edit", "templates/index.html", 2000, 60); got == "" {
		t.Error("expected rejection for 2000B → 60B (below 64B floor)")
	}
}

func TestValidateNotSuspiciouslyShrunkAllowsLegitEdits(t *testing.T) {
	cases := []struct {
		name     string
		old, new int
	}{
		{"original below threshold (line edit)", 50, 5},
		{"genuine small change", 200, 150},
		{"replace_all collapsing duplicates", 800, 400},
		{"new content >= 64B (above threshold)", 1500, 64},
		{"new content well above threshold", 200, 200},
		{"refactor to one-liner with reasonable body (5KB → 80B)", 5000, 80},
		{"both small (below 100B trigger)", 80, 20},
		{"empty original (new file via ast_edit-ish path)", 0, 16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := validateNotSuspiciouslyShrunk("ast_edit", "x.py", tc.old, tc.new); got != "" {
				t.Errorf("validateNotSuspiciouslyShrunk(%d, %d) rejected: %s", tc.old, tc.new, got)
			}
		})
	}
}

func TestValidateNotSuspiciouslyShrunkRejectionMessage(t *testing.T) {
	// The rejection text must (a) name the tool so the model knows what
	// to retry, (b) report old/new sizes so the model can see it WAS
	// truncated, and (c) tell it to re-emit the FULL body.
	got := validateNotSuspiciouslyShrunk("ast_edit", "templates/index.html", 120, 16)
	if got == "" {
		t.Fatal("expected rejection")
	}
	for _, s := range []string{"ast_edit refused", "16B", "120B", "FULL", "templates/index.html"} {
		if !strings.Contains(got, s) {
			t.Errorf("rejection missing %q: %s", s, got)
		}
	}
}

// May 8 2026 flask test: dashboard.html ended up with two consecutive
// <!DOCTYPE html> lines after a successful ast_edit. Root cause: model
// included <!DOCTYPE html> in `content` when selector was <html>, but
// ast_edit's <html> selector replaces only the html element — the
// existing doctype declaration above it was untouched, producing a
// duplicated doctype. This test locks the strip behaviour.
func TestStripLeadingDoctype(t *testing.T) {
	cases := []struct {
		name     string
		in       string
		want     string
		stripped bool
	}{
		{
			name:     "html5 doctype",
			in:       "<!DOCTYPE html>\n<html><body></body></html>",
			want:     "<html><body></body></html>",
			stripped: true,
		},
		{
			name:     "html5 doctype lowercase",
			in:       "<!doctype html>\n<html></html>",
			want:     "<html></html>",
			stripped: true,
		},
		{
			name:     "doctype with leading whitespace",
			in:       "  \n<!DOCTYPE html>\n<html></html>",
			want:     "<html></html>",
			stripped: true,
		},
		{
			name:     "no doctype",
			in:       "<html><body></body></html>",
			want:     "<html><body></body></html>",
			stripped: false,
		},
		{
			name:     "doctype not at start (after content)",
			in:       "<html><!DOCTYPE html><body></body></html>",
			want:     "<html><!DOCTYPE html><body></body></html>",
			stripped: false,
		},
		{
			name:     "verbose html4 doctype",
			in:       `<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">` + "\n<html></html>",
			want:     "<html></html>",
			stripped: true,
		},
		{
			name:     "empty content",
			in:       "",
			want:     "",
			stripped: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := stripLeadingDoctype(tc.in)
			if got != tc.want {
				t.Errorf("stripLeadingDoctype(%q) = %q, want %q", tc.in, got, tc.want)
			}
			if ok != tc.stripped {
				t.Errorf("stripLeadingDoctype(%q) stripped=%v, want %v", tc.in, ok, tc.stripped)
			}
		})
	}
}

// May 10 2026 false-success: action-intent prompts ("rewrite X", "add Y")
// were slipping past the fix-intent verification gate, letting the model
// declare done without making any actual edit. Lock the action-intent
// vocabulary so this gate stays armed across prompt phrasings.
func TestIsActionIntentMessage(t *testing.T) {
	actionIntents := []string{
		"rewrite templates/dashboard.html",
		"create a new flask blueprint",
		"add a logout button to the header",
		"implement a /health endpoint",
		"build a metrics page",
		"refactor app.py to use blueprints",
		"replace the hero section with a new one",
		"update the dashboard to show three KPI cards",
		"modify the User model to track login_at",
		"change the dashboard layout to flex",
		"convert this to TypeScript",
		"redesign templates/index.html for SaaS",
		// May 10 prompt that motivated this gate:
		"Rewrite templates/dashboard.html to display a clean SaaS-style metrics dashboard",
		// TB2 round-2 prompt that motivated gating the text exit: the
		// model narrated "I will now proceed to sanitize..." as a text
		// response and quit with zero edits. Must classify as action.
		"Please help sanitize my github repository of all API keys. Please find and remove all such information and replace it with placeholder values",
	}
	for _, m := range actionIntents {
		if !isActionIntentMessage(m) {
			t.Errorf("isActionIntentMessage(%q) = false, want true", m)
		}
	}

	notAction := []string{
		"hi",
		"thanks",
		"what does this code do",
		"explain the dashboard route",
		"is the server running",
		"why isn't this working", // fix-intent, not action-intent
		"how do I curl the api",
	}
	for _, m := range notAction {
		if isActionIntentMessage(m) {
			t.Errorf("isActionIntentMessage(%q) = true, want false", m)
		}
	}
}

func TestActionWithoutProductiveChangeMessage(t *testing.T) {
	// Sanity: rejection text must (a) tell the model NOT to declare done,
	// (b) name the missing tools, (c) mention verification ≠ task.
	got := actionWithoutProductiveChangeMessage("rewrite templates/dashboard.html...")
	for _, s := range []string{"Cannot declare `done`", "write_file", "edit_file", "ast_edit", "Verification", "NOT the task"} {
		if !strings.Contains(got, s) {
			t.Errorf("rejection missing %q: %s", s, got)
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
		"markdownlint README.md",
		"shellcheck scripts/setup.sh",
		"golangci-lint run ./...",
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

func TestResolveAgentPathStripsWorkspacePrefix(t *testing.T) {
	// PC-198 — model frequently emits `workspace/X` (no leading slash)
	// when it means the project root. resolveAgentPath must strip
	// the prefix instead of joining it onto cwd, which would
	// produce `/workspace/workspace/X` and 404.
	ctx := &AgentContext{WorkingDir: "/workspace"}
	cases := []struct {
		in, want string
	}{
		{"workspace/app.py", "/workspace/app.py"},
		{"workspace", "/workspace"},
		{"./workspace/app.py", "/workspace/app.py"},
		{"app.py", "/workspace/app.py"}, // no prefix → unchanged
		{"src/main.go", "/workspace/src/main.go"},
	}
	for _, tc := range cases {
		if got := resolveAgentPath(ctx, tc.in); got != tc.want {
			t.Errorf("resolveAgentPath(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestLooksCorruptedOnDiskDetectsProsePreamble(t *testing.T) {
	// The exact corruption pattern from /home/isaac/snake/templates/index.html.
	// Without PC-201, write_file gets blocked by the >5-line gate and the
	// model loops forever trying to clean it via edit_file.
	corrupt := "Looking at the task, I need to create a complete `index.html` file...\n\n```html\n<!DOCTYPE html>\n<html><body>hi</body></html>\n```\n\nThis file:\n1. Renders correctly\n"
	if !looksCorruptedOnDisk("templates/index.html", corrupt) {
		t.Error("expected corrupted prose+fence file to be detected")
	}

	clean := "<!DOCTYPE html>\n<html><body>hi</body></html>\n"
	if looksCorruptedOnDisk("templates/index.html", clean) {
		t.Error("clean HTML file should NOT be flagged as corrupted")
	}

	// Markdown files legitimately contain fences — never flagged.
	md := "# Title\n\n```python\nprint('hi')\n```\n"
	if looksCorruptedOnDisk("README.md", md) {
		t.Error("markdown file with fence should NOT be flagged")
	}
}

// HARNESS-12: extract the task's named output file(s) from the prompt,
// and ignore input filenames (no write-verb nearby).
func TestExpectedOutputPaths(t *testing.T) {
	cases := map[string][]string{
		"Please save your solution in the file sol.sql":              {"sol.sql"},
		"Recover the rows and create a JSON file in recover.json":    {"recover.json"},
		"Write the output to out.txt":                                {"out.txt"},
		"create a file called out.html that survives the filter":     {"out.html"},
		"read the file input.txt and write the result to output.txt": {"output.txt"},
		"I have a sqlite database in trunc.db that was corrupted":    nil,
		"optimize the query in my-sql-query.sql, save it in sol.sql": {"sol.sql"},
	}
	for prompt, want := range cases {
		got := expectedOutputPaths(prompt)
		// output.txt case: input.txt must be excluded, output.txt included.
		if prompt[:4] == "read" {
			if len(got) != 1 || got[0] != "output.txt" {
				t.Errorf("%q: got %v, want [output.txt] (input.txt must be excluded)", prompt, got)
			}
			continue
		}
		if want == nil {
			if len(got) != 0 {
				t.Errorf("%q: got %v, want none (input file, no write verb)", prompt, got)
			}
			continue
		}
		found := false
		for _, g := range got {
			if g == want[0] {
				found = true
			}
		}
		if !found {
			t.Errorf("%q: got %v, want to include %v", prompt, got, want)
		}
	}
}

func TestExpectedOutputMissingMessage(t *testing.T) {
	m := expectedOutputMissingMessage([]string{"sol.sql"})
	if !strings.Contains(m, "`sol.sql`") || !strings.Contains(m, "does not exist") {
		t.Errorf("message should name the file and say it's missing: %q", m)
	}
}

// HARNESS-12 refinement: "the file X must exist / must contain" names a
// deliverable without a write verb (TB2 merge-diff).
func TestExpectedOutputMustExistPhrasing(t *testing.T) {
	got := expectedOutputPaths("the final repository must include repo/algo.py. The file repo/algo.py must exist in the merged result and must contain a function named map")
	found := false
	for _, g := range got {
		if g == "repo/algo.py" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected repo/algo.py from must-exist phrasing, got %v", got)
	}
}
