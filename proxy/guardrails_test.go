package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
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

// May 9 2026 structural_edit destructive-stub case: model emitted only
// "<!DOCTYPE html>\n" (16B) for an entire <html>-element rewrite of a
// 120B file, structural_edit "succeeded", file destroyed, model declared
// "done". Guard catches this exact shape without false-rejecting
// realistic small replacements.
func TestValidateNotSuspiciouslyShrunkRejectsDestructiveStub(t *testing.T) {
	// Today's exact case.
	if got := validateNotSuspiciouslyShrunk("structural_edit", "templates/index.html", 120, 16); got == "" {
		t.Error("expected rejection for 120B → 16B replacement")
	}
	// Larger original, larger stub — still flagged.
	if got := validateNotSuspiciouslyShrunk("structural_edit", "app.py", 5000, 20); got == "" {
		t.Error("expected rejection for 5000B → 20B replacement")
	}
	// edit_file path covered by the same guard.
	if got := validateNotSuspiciouslyShrunk("edit_file", "app.py", 200, 8); got == "" {
		t.Error("expected rejection for edit_file 200B → 8B")
	}
	// May 10 2026: the 32B-just-passes failure that motivated bumping
	// the floor from 32 to 128. Model emitted exactly 32B for an HTML
	// body rewrite — clearly a stub but slipped past the v1 guard.
	if got := validateNotSuspiciouslyShrunk("structural_edit", "templates/dashboard.html", 2199, 32); got == "" {
		t.Error("expected rejection for 2199B → 32B (the May 10 boundary case)")
	}
	// 60B replacement of 2KB original — under the 64B floor.
	if got := validateNotSuspiciouslyShrunk("structural_edit", "templates/index.html", 2000, 60); got == "" {
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
		{"empty original (new file via structural_edit-ish path)", 0, 16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := validateNotSuspiciouslyShrunk("structural_edit", "x.py", tc.old, tc.new); got != "" {
				t.Errorf("validateNotSuspiciouslyShrunk(%d, %d) rejected: %s", tc.old, tc.new, got)
			}
		})
	}
}

func TestValidateNotSuspiciouslyShrunkRejectionMessage(t *testing.T) {
	// The rejection text must (a) name the tool so the model knows what
	// to retry, (b) report old/new sizes so the model can see it WAS
	// truncated, and (c) tell it to re-emit the FULL body.
	got := validateNotSuspiciouslyShrunk("structural_edit", "templates/index.html", 120, 16)
	if got == "" {
		t.Fatal("expected rejection")
	}
	for _, s := range []string{"structural_edit refused", "16B", "120B", "FULL", "templates/index.html"} {
		if !strings.Contains(got, s) {
			t.Errorf("rejection missing %q: %s", s, got)
		}
	}
}

// May 8 2026 flask test: dashboard.html ended up with two consecutive
// <!DOCTYPE html> lines after a successful structural_edit. Root cause: model
// included <!DOCTYPE html> in `content` when selector was <html>, but
// structural_edit's <html> selector replaces only the html element — the
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
		// The prompt that motivated gating the text exit: the
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
	for _, s := range []string{"Cannot declare `done`", "write_file", "edit_file", "structural_edit", "Verification", "NOT the task"} {
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
	if got := patternMatchHint(target, nil); got == "" {
		t.Error("expected hint when no siblings have been read")
	}
	// A sibling in the session read-cache — hint should disappear.
	read := map[string]string{filepath.Join(dir, "index.html"): "<html/>"}
	if got := patternMatchHint(target, read); got != "" {
		t.Errorf("hint should disappear after sibling read: %s", got)
	}
}

func TestPatternMatchHintSkipsExistingFiles(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.html"), []byte("a"), 0o644)
	os.WriteFile(filepath.Join(dir, "b.html"), []byte("b"), 0o644)
	target := filepath.Join(dir, "a.html") // exists already → not a new write
	if got := patternMatchHint(target, nil); got != "" {
		t.Errorf("editing existing file should not trip hint: %s", got)
	}
}

func TestPatternMatchHintSkipsLonelyDir(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "only.go"), []byte("package x"), 0o644)
	target := filepath.Join(dir, "new.go")
	if got := patternMatchHint(target, nil); got != "" {
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
	// The model frequently emits `workspace/X` (no leading slash)
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
	// Without the corruption exemption, write_file gets blocked by the
	// >5-line gate and the model loops forever trying to clean it via
	// edit_file.
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

// Extract the task's named output file(s) from the prompt,
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

// "the file X must exist / must contain" names a deliverable without a
// write verb (a merge-diff prompt).
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

// --- verification gate: observed state vs message shape ---------------------

// The 2026-07-21 dogfooding failure. "build an API with a couple tests" holds
// no repair-shaped word, so message-shape analysis alone cannot gate it — yet
// the model ran pytest, watched it fail, diagnosed the fix in prose, and
// exited without applying it.
func TestVerificationGateFiresOnFailedVerificationWithoutFixIntent(t *testing.T) {
	msg := "build me a small API with a couple tests"

	if isFixIntentMessage(msg) {
		t.Fatalf("precondition failed: %q should not read as fix-intent — "+
			"if it did, this test would not be exercising the observed-state path", msg)
	}

	userWantsVerification := isFixIntentMessage(msg)
	sawFailedVerification := true // pytest ran and exited non-zero
	verifiedThisLoop := false

	if !((userWantsVerification || sawFailedVerification) && !verifiedThisLoop) {
		t.Fatal("gate did not fire after a verification command failed — a red test must block done regardless of how the request was worded")
	}
}

// A green run after a red one must clear the latch, or the loop can never
// legitimately finish.
func TestVerificationGateClearsAfterPassingRun(t *testing.T) {
	userWantsVerification := false
	sawFailedVerification := false // reset when the command later succeeded
	verifiedThisLoop := true

	if (userWantsVerification || sawFailedVerification) && !verifiedThisLoop {
		t.Fatal("gate still firing after verification passed")
	}
}

// No verification attempted and no repair-shaped request: nothing to gate.
// This is what keeps the gate off pure feature work.
func TestVerificationGateStaysOffWhenNothingWasVerified(t *testing.T) {
	userWantsVerification := isFixIntentMessage("add a logout button to the navbar")
	sawFailedVerification := false
	verifiedThisLoop := false

	if (userWantsVerification || sawFailedVerification) && !verifiedThisLoop {
		t.Fatal("gate fired on a plain feature request with no verification attempted")
	}
}

func TestVerificationRejectionMessageNamesTheRedCommand(t *testing.T) {
	failed := verificationRejectionMessage(true)
	if !strings.Contains(failed, "FAILED") {
		t.Fatalf("failed-verification message should say a command failed, got: %s", failed)
	}
	// The model has already seen the output; it needs to act, not be taught.
	if !strings.Contains(failed, "Describing the fix is not applying it") {
		t.Fatalf("failed-verification message should push for the edit, got: %s", failed)
	}

	plain := verificationRejectionMessage(false)
	if !strings.Contains(plain, "fix/repair request") {
		t.Fatalf("fix-intent message should describe the request, got: %s", plain)
	}
	if failed == plain {
		t.Fatal("both triggers produced the same message — the gate must say which one fired")
	}
}

func TestGateTriggerPrefersTheConcreteSignal(t *testing.T) {
	if got := gateTrigger(true, true); got != "failed-verification" {
		t.Fatalf("gateTrigger(true,true) = %q, want failed-verification (a red command outranks message shape)", got)
	}
	if got := gateTrigger(true, false); got != "fix-intent" {
		t.Fatalf("gateTrigger(true,false) = %q, want fix-intent", got)
	}
	if got := gateTrigger(false, false); got != "none" {
		t.Fatalf("gateTrigger(false,false) = %q, want none", got)
	}
}

// --- tool-choice boundaries in descriptions --------------------------------

// The 2026-07-21 port-conflict loop: the model started a blocking dev server
// with run_command, it was killed at the timeout, and the identical command
// was reissued twice more. run_background's description already named
// `python app.py`; run_command's said only "Execute a shell command", so
// nothing marked the boundary and the generic tool won. Descriptions are
// surfaced to the model by buildToolDescriptionsExcluding, so this is the
// text it actually reads when choosing.
func TestRunCommandDescriptionRedirectsLongRunningWork(t *testing.T) {
	docs := buildToolDescriptionsExcluding(nil)

	for _, want := range []string{"run_background", "doesn't exit"} {
		if !strings.Contains(docs, want) {
			t.Fatalf("tool docs missing %q — run_command must mark the boundary it does not cover", want)
		}
	}
	if !strings.Contains(docs, "WAIT for it to exit") {
		t.Fatal("run_command description must state that it blocks, which is what makes it wrong for servers")
	}
}

// Both tools must not claim the same job.
func TestRunBackgroundDescriptionOwnsServers(t *testing.T) {
	docs := buildToolDescriptionsExcluding(nil)
	if !strings.Contains(docs, "python app.py") {
		t.Fatal("run_background should keep naming a concrete server command — that specificity is what makes it selectable")
	}
}

// --- done-without-action gate: observed engagement vs verb list ------------

// The gap the verb list left. actionIntentWords carries "create"/"add"/"make"
// but not "remove"/"delete", so this prompt armed no gate and the model could
// close the turn having deleted nothing.
func TestWantsStateChangeCatchesRemovalWithoutAVerbListEntry(t *testing.T) {
	msg := "remove the debug logging from app.py"

	if isActionIntentMessage(msg) {
		t.Fatalf("precondition failed: %q now matches actionIntentWords, so this test "+
			"no longer exercises the observed-engagement path", msg)
	}
	if !wantsStateChange(msg, Tier2Medium, true) {
		t.Fatal("removal request with the workspace inspected must arm the gate")
	}
}

// The user's stated worst case: model investigates, finds the bug, narrates it,
// never applies the fix.
func TestWantsStateChangeCatchesDiagnoseWithoutFix(t *testing.T) {
	msg := "the sidebar overlaps the content on narrow screens"
	if !wantsStateChange(msg, classifyAgentTier(msg), true) {
		t.Fatal("bug report + files opened + nothing written must arm the gate")
	}
}

// Reading files is also how a question is answered; those must stay ungated.
func TestWantsStateChangeIgnoresQuestionsThatReadFiles(t *testing.T) {
	msg := "why does the game store direction as a string"
	if got := classifyAgentTier(msg); got != Tier0Conversational {
		t.Fatalf("precondition: classifyAgentTier(%q) = %v, want T0", msg, got)
	}
	if wantsStateChange(msg, Tier0Conversational, true) {
		t.Fatal("a question that opened files must not be gated — writing nothing is the correct outcome")
	}
}

// Acknowledgements are longer than the T0 length floor but never touch the
// project, so engagement is what keeps them out.
func TestWantsStateChangeIgnoresAcknowledgements(t *testing.T) {
	for _, msg := range []string{"thanks, that looks great", "ok that makes sense to me"} {
		if wantsStateChange(msg, classifyAgentTier(msg), false) {
			t.Errorf("acknowledgement %q armed the gate with no workspace inspection", msg)
		}
	}
}

// Explicit action wording still arms the gate on its own, before any tool runs.
func TestWantsStateChangeHonoursExplicitActionIntent(t *testing.T) {
	if !wantsStateChange("add a logout button to the navbar", Tier2Medium, false) {
		t.Fatal("explicit action intent must arm the gate without needing inspection")
	}
}

// One gate must not be able to spend the whole bounce allowance and silence
// the rest. Reproduces an observed session: the verification gate bounced the
// exit three times, and with a single shared counter exitGates then returned
// early forever — so the done-without-action gate never ran and the model
// exited having changed nothing while claiming the work was already present.
func TestExitGatesOneGateCannotStarveAnother(t *testing.T) {
	dir := t.TempDir()
	ctx := &AgentContext{WorkingDir: dir, Tier: Tier2Medium}
	// "fix ..." is repair intent, so the verification gate arms; no
	// verification ran and no productive change landed, so both gates hold.
	const msg = "fix the pause toggle in app.py so the spacebar works"
	st := &runState{
		userWantsVerification: true,
		inspectedWorkspace:    true,
	}

	for i := 1; i <= maxGateBounces; i++ {
		gate, _ := st.exitGates(ctx, msg, "done")
		if gate != "verification_gate" {
			t.Fatalf("bounce %d: got gate %q, want verification_gate", i, gate)
		}
	}
	// Verification's budget is now spent. The action gate has said nothing
	// yet and must still get to.
	gate, rejection := st.exitGates(ctx, msg, "done")
	if gate != "action_gate" {
		t.Fatalf("after verification exhausted its budget, got gate %q, want action_gate", gate)
	}
	if rejection == "" {
		t.Error("action gate bounced with an empty rejection")
	}
	if st.gateBounces["verification_gate"] != maxGateBounces {
		t.Errorf("verification gate overspent: %d", st.gateBounces["verification_gate"])
	}
}

// Each gate still stops repeating itself once its own budget is gone.
func TestExitGatesPerGateBudgetIsEnforced(t *testing.T) {
	dir := t.TempDir()
	ctx := &AgentContext{WorkingDir: dir, Tier: Tier2Medium}
	const msg = "add a pause toggle to app.py"
	st := &runState{inspectedWorkspace: true}

	for i := 1; i <= maxGateBounces; i++ {
		if gate, _ := st.exitGates(ctx, msg, "done"); gate != "action_gate" {
			t.Fatalf("bounce %d: got %q, want action_gate", i, gate)
		}
	}
	if gate, _ := st.exitGates(ctx, msg, "done"); gate != "" {
		t.Errorf("action gate exceeded its budget: got %q, want the exit to pass", gate)
	}
}

// The tiers exist so V3 does not run on everything. A question asking for an
// explanation must not be classified as work — measured live: both
// conversational probes ran the V3 pipeline AND wrote to files the user had
// explicitly asked it to leave alone.
func TestExplainOnlyQuestionsClassifyAsChat(t *testing.T) {
	for _, msg := range []string{
		"In orders.py, what does find_duplicates do, and what is its time complexity? Just explain — do not change any code.",
		"In orders.py, apply_discount(100, 10) returns 90.0 but a colleague says it should return 90. Explain what is going on here and whether it is actually a bug. Do not change the code.",
		"Explain how the retry logic works, without editing anything.",
	} {
		if got := classifyAgentTier(msg); got != Tier0Conversational {
			t.Errorf("want T0 for a question, got %v: %.60s", got, msg)
		}
	}
}

// The negation must not swallow genuine work. "don't change the API" is a
// constraint on a real task, not a request to keep hands off the codebase.
func TestWorkStillClassifiesAsWork(t *testing.T) {
	for _, msg := range []string{
		"add a pause toggle to app.py",
		"fix the login bug",
		"fix the parser but don't change the public API",
		"refactor store.py without changing its behaviour",
	} {
		if got := classifyAgentTier(msg); got == Tier0Conversational {
			t.Errorf("want work tier, got T0: %.60s", msg)
		}
	}
}

// A question mark mid-message, with the question qualified afterwards, is how
// people actually write. A suffix-only check read it as not-a-question.
func TestQuestionDetectedMidMessage(t *testing.T) {
	if !isQuestionMessage("what does find_duplicates do? Just explain.") {
		t.Error("a question with a trailing qualifier must still be a question")
	}
	if !isQuestionMessage("In orders.py, what does find_duplicates do") {
		t.Error("a question starter after a clause boundary must count")
	}
	if isQuestionMessage("add a pause toggle to app.py") {
		t.Error("a plain instruction is not a question")
	}
}

func TestNegatedActionWordIsNotActionIntent(t *testing.T) {
	if isActionIntentMessage("please explain this, do not change any code") {
		t.Error("a negated action word must not read as action intent")
	}
	if !isActionIntentMessage("create a new module for this") {
		t.Error("a plain action word must still read as action intent")
	}
}

// A `text` reply ends the turn, so a model that announces a tool call instead
// of making one stops with the right intent and no action. Observed on a
// question about code: "I need to read orders.py — I'll start by outlining
// the file to locate the function", then the turn ended.
func TestAnnouncedToolUseIsDetected(t *testing.T) {
	for _, s := range []string{
		"I need to read the `orders.py` file to explain what find_duplicates does.",
		"I'll start by outlining the file to locate the function.",
		"Let me look at the implementation first.",
		"I'm going to check store.py before answering.",
	} {
		if !announcesImminentToolUse(s) {
			t.Errorf("should detect an announced tool call: %q", s)
		}
	}
}

// An ANSWER that happens to mention reading must not be mistaken for one.
func TestRealAnswersAreNotTreatedAsAnnouncements(t *testing.T) {
	for _, s := range []string{
		"find_duplicates is O(n^2): it reads the list inside a nested loop.",
		"The function opens the file and returns its contents.",
		"apply_discount returns a float because / is true division in Python 3.",
		"",
	} {
		if announcesImminentToolUse(s) {
			t.Errorf("real answer misread as an announcement: %q", s)
		}
	}
}

// A reply that signs off promising the actual answer leaves the user with half
// of one. Observed on a bug-find task: the model named the file, described the
// symptom, then ended "I will now provide the specific location and the
// incorrect comparison as requested" — and the turn ended there.
func TestPromisedContentIsDetected(t *testing.T) {
	for _, s := range []string{
		"The bug is in planning.py. I will now provide the specific location and the incorrect comparison as requested.",
		"I've analysed the file. Let me give you the exact line.",
		"Here's the summary so far. I'll now show the failing comparison.",
	} {
		if !promisesMoreContent(s) {
			t.Errorf("should detect an undelivered promise: %q", s)
		}
	}
}

// A complete answer must not be bounced, including one that mentions
// providing something earlier in the reply and then does.
func TestCompleteAnswersAreNotBounced(t *testing.T) {
	for _, s := range []string{
		"planning.py breaks ties with n_steps > best_steps, so a tie keeps the longer plan.",
		"I'll provide the details: the comparison on line 314 uses > where it should use <.",
		"find_duplicates is O(n^2) because of the nested loop.",
		"",
	} {
		if promisesMoreContent(s) {
			t.Errorf("complete answer wrongly flagged: %q", s)
		}
	}
}

// The live failure: "Please list the files in the directory you are in."
// list_directory succeeded, which set inspectedWorkspace, which made the
// action gate demand a productive change before `done`. The model was refused
// its own completion and told the user it was "unable to complete the 'done'
// state as per the system's requirements" — a rule nothing states.
func TestAReadOnlyRequestDoesNotDemandAWrite(t *testing.T) {
	readOnly := []string{
		"Please list the files in the directory you are in.",
		"list files",
		"show me the config",
		"what files are in src/",
		"where is the router defined",
		"print the contents of app.py",
	}
	for _, m := range readOnly {
		// inspectedWorkspace=true is the point: the read that answered the
		// question is the same read the gate used as evidence of work.
		if wantsStateChange(m, Tier1Simple, true) {
			t.Errorf("read-only request still demands a write: %q", m)
		}
	}

	mustStillDemandWork := []string{
		"add a list of files to the README",   // action verb wins
		"find and fix the bug in the parser",  // fix intent wins
		"show me the bug and repair it",       // fix intent wins
		"the file list is broken",             // fix intent wins
		"refactor the directory listing code", // action verb wins
	}
	for _, m := range mustStillDemandWork {
		if !wantsStateChange(m, Tier1Simple, true) {
			t.Errorf("work request was wrongly exempted: %q", m)
		}
	}
}

// The live failure: the model wanted app.py, sent find_file {"path":"app.py"}
// with no pattern, got "pattern cannot be empty", and immediately sent
// {"path":".*app\\.py.*"} — moving the regex into `path` rather than into
// `pattern`, because the message named the field it had left blank and never
// the one it had filled.
func TestFindFileArgSwapIsNamed(t *testing.T) {
	hint := findFileArgSwapHint("app.py")
	if hint == "" {
		t.Fatal("a filename in `path` must be recognised as the swapped argument")
	}
	// Doubled backslash on purpose: the hint is what the model pastes into
	// JSON, where \\ is how a literal backslash is written.
	for _, want := range []string{"app.py", "pattern", "path", `app\\.py$`} {
		if !strings.Contains(hint, want) {
			t.Errorf("hint missing %q:\n%s", want, hint)
		}
	}

	// Real directories must fall through to the generic message.
	for _, dir := range []string{"", "src", "./", "scripts/fixtures", "/workspace", "a/b.py"} {
		if h := findFileArgSwapHint(dir); h != "" {
			t.Errorf("path %q is a directory or unclear, must not claim an arg swap: %s", dir, h)
		}
	}
}

// The suggested pattern has to be a valid regex that matches the file, or the
// retry fails too.
func TestFindFileArgSwapSuggestsAWorkingPattern(t *testing.T) {
	hint := findFileArgSwapHint("snake_app.py")
	start := strings.Index(hint, `{"pattern":"`)
	if start < 0 {
		t.Fatalf("no concrete retry in hint: %s", hint)
	}
	// The hint renders the pattern with %q, so it carries JSON/Go escaping —
	// which is exactly what the model must paste into its tool args. Unquote
	// it to get the regex the proxy will actually receive.
	raw := hint[start+len(`{"pattern":`):]
	raw = raw[:strings.LastIndex(raw, `}`)]
	suggested, err := strconv.Unquote(raw)
	if err != nil {
		t.Fatalf("suggested pattern %q is not a valid quoted string: %v", raw, err)
	}
	re, err := regexp.Compile(suggested)
	if err != nil {
		t.Fatalf("suggested pattern %q does not compile: %v", suggested, err)
	}
	if !re.MatchString("snake_app.py") {
		t.Errorf("suggested pattern %q does not match the filename it came from", suggested)
	}
	// The dot must be escaped, or it matches unintended names.
	if re.MatchString("snakeXapp.py") {
		t.Errorf("suggested pattern %q left the dot unescaped", suggested)
	}
}

// The live failure: the model sent
//
//	if(headX === food.x ℘ headY ===food.y) {
//
// for a line whose operator is `&&`. U+2118 SCRIPT CAPITAL P is a decode
// artefact, not a typo, and the rejection it got said the old_str was "9 lines
// long" — true, and not why it failed.
func TestCorruptedCharacterIsNamed(t *testing.T) {
	file := "        if(headX === food.x && headY === food.y) {\n            score++;\n"

	bad := foreignRunes("if(headX === food.x ℘ headY ===food.y) {", file)
	if len(bad) != 1 || bad[0] != '℘' {
		t.Fatalf("expected the corrupted rune, got %q", string(bad))
	}
	desc := describeForeignRunes(bad)
	if !strings.Contains(desc, "U+2118") {
		t.Errorf("description must carry the codepoint, got %q", desc)
	}

	// Must NOT fire on ASCII mis-copies — the closest-line hint owns those.
	if r := foreignRunes("if(headX == food.x && headY === food.y) {", file); len(r) != 0 {
		t.Errorf("ASCII mis-copy wrongly flagged as corruption: %q", string(r))
	}
	// Must NOT fire on non-ASCII the file genuinely contains.
	withUnicode := "label = 'café ☕'\n"
	if r := foreignRunes("label = 'café ☕'", withUnicode); len(r) != 0 {
		t.Errorf("legitimate non-ASCII flagged: %q", string(r))
	}
	// A non-ASCII char absent from the file IS corruption even amid other text.
	if r := foreignRunes("x = 1 — y", "x = 1 - y\n"); len(r) != 1 || r[0] != '—' {
		t.Errorf("em-dash substitution not caught: %q", string(r))
	}
	// Empty inputs are not corruption.
	if r := foreignRunes("", file); len(r) != 0 {
		t.Errorf("empty old_str flagged: %q", string(r))
	}
}

// The verification gate bounces `done` three times and then, out of bounces,
// lets the model's summary through unchanged. Three runs on 2026-08-02 ended
// with a confident "I verified..." over a broken file — once over a Flask app
// whose only @app.route had been deleted.
func TestAnUnverifiedRunDoesNotShipTheModelsClaim(t *testing.T) {
	claim := "I updated the snake game logic and verified that the page loads."

	wrote := unverifiedSummary(true, claim)
	for _, want := range []string{"NOTHING in this run verified", "Run it yourself", "UNVERIFIED"} {
		if !strings.Contains(wrote, want) {
			t.Errorf("summary missing %q:\n%s", want, wrote)
		}
	}
	// The model's account is kept — it usually describes the intended change
	// correctly; it is the verification claim inside it that is unsupported.
	if !strings.Contains(wrote, claim) {
		t.Errorf("dropped the agent's account entirely:\n%s", wrote)
	}

	nothing := unverifiedSummary(false, claim)
	if !strings.Contains(nothing, "Nothing was written to disk") {
		t.Errorf("a run that wrote nothing must say so:\n%s", nothing)
	}
}

func TestVerificationDemandedAndUnmetIsIndependentOfTheBounceBudget(t *testing.T) {
	s := &runState{sawFailedVerification: true, gateBounces: map[string]int{}}
	if !s.verificationDemandedAndUnmet() {
		t.Fatal("a red verification with nothing green must report unmet")
	}
	// Exhausting the bounces means the gate stopped blocking, not that
	// anything got verified.
	for i := 0; i < maxGateBounces+2; i++ {
		s.chargeBounce("verification_gate")
	}
	if !s.verificationDemandedAndUnmet() {
		t.Error("running out of bounces was mistaken for a passing verification")
	}
	s.verifiedThisLoop = true
	if s.verificationDemandedAndUnmet() {
		t.Error("a green verification still reported unmet")
	}
}

// Run 11 was refused three times — selector-unreachable, span-too-large,
// stale-range — each attempt responding to the previous error, and the
// error-loop breaker killed it with the file untouched while it was closing
// in. Three DIFFERENT rejections is a model converging, not one looping.
func TestRejectionClassSeparatesConvergingFromLooping(t *testing.T) {
	selector := rejectionClass("`draw` exists in app.py, but NOT as a node any selector can reach: it is javascript at lines 77-202")
	tooLarge := rejectionClass("replace_lines: 126 lines is too large a range (limit 60).")
	stale := rejectionClass("replace_lines: line 202 of app.py is not what you expected, so the range is wrong")

	for _, pair := range [][2]string{{selector, tooLarge}, {tooLarge, stale}, {selector, stale}} {
		if pair[0] == pair[1] {
			t.Errorf("distinct rejections collapsed into one class:\n%q\n%q", pair[0], pair[1])
		}
	}

	// The same failure on different lines of different files IS one class —
	// that is the loop the breaker exists for.
	a := rejectionClass("replace_lines: line 202 of app.py is not what you expected, so the range is wrong")
	b := rejectionClass("replace_lines: line 87 of main.py is not what you expected, so the range is wrong")
	if a != b {
		t.Errorf("the same failure was split by its variable parts:\n%q\n%q", a, b)
	}
	if rejectionClass("") != "" {
		t.Error("an empty error must not produce a class")
	}
}

// Run 4 built the variable-delay loop it was asked for, never added the
// per-food decrement, and emitted `done`. The plan named both; the reminder
// showing progress every turn was ignored; nothing checked it at the exit.
func TestDoneIsRefusedWhilePlanStepsRemain(t *testing.T) {
	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	ctx.Plan = &Plan{
		WinningScore: 0.8,
		Steps: []PlanStep{
			{ID: "s1", Action: "replace setInterval with a variable-delay loop"},
			{ID: "s2", Action: "decrement the delay by 3ms per food eaten"},
		},
	}
	ctx.PlanStepsSatisfied = []bool{true, false}

	msg := planIncompleteMessage(ctx)
	if msg == "" {
		t.Fatal("done must be refused while a planned step has never landed")
	}
	if !strings.Contains(msg, "s2") || !strings.Contains(msg, "decrement the delay") {
		t.Errorf("the unsatisfied step is not named:\n%s", msg)
	}
	if !strings.Contains(msg, "1 of 2") {
		t.Errorf("progress not stated:\n%s", msg)
	}

	ctx.PlanStepsSatisfied = []bool{true, true}
	if planIncompleteMessage(ctx) != "" {
		t.Error("a fully satisfied plan still blocked done")
	}
}

// A bad plan blocking a finished task is worse than no gate.
func TestThePlanGateStandsDownWhenThePlanIsNotEvidence(t *testing.T) {
	base := func() *AgentContext {
		ctx := NewAgentContext(t.TempDir(), Tier2Medium)
		ctx.Plan = &Plan{WinningScore: 0.8, Steps: []PlanStep{
			{ID: "s1", Action: "a"}, {ID: "s2", Action: "b"}}}
		ctx.PlanStepsSatisfied = []bool{true, false}
		return ctx
	}
	low := base()
	low.Plan.WinningScore = 0.2
	if planIncompleteMessage(low) != "" {
		t.Error("gated on a plan the planner itself rates as weak")
	}
	// Nothing matched: the matcher is not tracking this task, which is not
	// evidence the model did nothing.
	untracked := base()
	untracked.PlanStepsSatisfied = []bool{false, false}
	if planIncompleteMessage(untracked) != "" {
		t.Error("gated when no step matched at all")
	}
	single := base()
	single.Plan.Steps = single.Plan.Steps[:1]
	single.PlanStepsSatisfied = []bool{false}
	if planIncompleteMessage(single) != "" {
		t.Error("gated on a one-step plan, which carries no multi-part signal")
	}
	none := base()
	none.Plan = nil
	if planIncompleteMessage(none) != "" {
		t.Error("gated with no plan at all")
	}
}

// Asked "how does the contact form work?" on a fresh workspace, the loop hit
// its 5-turn conversational cap during recon and ended with an `error` event
// and nothing else — no answer, no partial, zero bytes. A blank reply is the
// worst thing the harness can produce: the user cannot tell whether ATLAS is
// broken, still thinking, or ignoring them.
func TestRunningOutOfTurnsStillSaysSomething(t *testing.T) {
	dir := t.TempDir()
	ctx := NewAgentContext(dir, Tier0Conversational)
	ctx.RecordFileRead(filepath.Join(dir, "index.html"), "<html></html>")
	ctx.RecordFileRead(filepath.Join(dir, "script.js"), "const f = 1;")

	msg := outOfTurnsSummary(ctx, false)
	for _, want := range []string{"ran out of turns", "Nothing was written", "index.html", "script.js", "narrower request"} {
		if !strings.Contains(msg, want) {
			t.Errorf("summary missing %q:\n%s", want, msg)
		}
	}
	if wrote := outOfTurnsSummary(ctx, true); !strings.Contains(wrote, "written to disk") {
		t.Errorf("a run that wrote must say so:\n%s", wrote)
	}
}

// A server started in the foreground burns the whole sandbox timeout and then
// reports a failure that says nothing about the code. Observed on the
// first-contact path: `python3 -m http.server 8000` cost 30s of a 3m39s run
// before the model reached for run_background on its own.
func TestForegroundServerStartsAreRedirectedBeforeTheyRun(t *testing.T) {
	for _, cmd := range []string{
		"python3 -m http.server 8000",
		"python -m http.server",
		"npm run dev",
		"npm start",
		"flask run --port 5001",
		"uvicorn main:app --reload",
		"php -S localhost:8000",
		"npx vite",
		"cd site && python3 -m http.server 9000",
	} {
		if foregroundServerRejection(cmd) == "" {
			t.Errorf("not redirected: %q", cmd)
		}
	}
	// Narrow on purpose. These exit on their own, and refusing them would
	// block a legitimate verification.
	for _, cmd := range []string{
		"python app.py", // just as likely a script that exits
		"pytest tests/",
		"curl -I http://localhost:8000",
		"npm run build",
		"npm test",
		"go build ./...",
		"ls -la",
	} {
		if r := foregroundServerRejection(cmd); r != "" {
			t.Errorf("wrongly redirected %q:\n%s", cmd, r)
		}
	}
	// The rejection has to carry the exact replacement call.
	r := foregroundServerRejection("python3 -m http.server 8000")
	for _, want := range []string{"run_background", "python3 -m http.server 8000", "curl", "stop_background"} {
		if !strings.Contains(r, want) {
			t.Errorf("rejection missing %q:\n%s", want, r)
		}
	}
}

// `text` is a terminal exit, so an announcement that slips this check ends the
// turn and the user gets a promise instead of an answer. Observed on a fresh
// workspace: "How does the contact form work?" came back as "I'll look into
// the contact form's implementation to see how it handles submissions and
// where the data is sent." — then done, with no tool calls and no answer.
// "look at" was in the verb list; "look into" was not.
func TestAnnouncementDetectorCoversTheWaysModelsStall(t *testing.T) {
	for _, s := range []string{
		"I'll look into the contact form's implementation to see how it handles submissions and where the data is sent.",
		"Let me look through the source to find where that is set.",
		"I'm going to investigate how routing works here.",
		"I need to take a look at the config first.",
		"First, I will read app.py to understand the structure.",
		"I'll dig into the handler and report back.",
	} {
		if !announcesImminentToolUse(s) {
			t.Errorf("announcement not detected: %q", s)
		}
	}
	// Real answers that merely mention these words must pass through.
	for _, s := range []string{
		"The contact form does not send anything — script.js calls preventDefault and only shows a status message.",
		"It reads the config at startup and caches it for the process lifetime.",
		"I looked at both files: routing is handled by Flask's @app.route decorators.",
		"Yes. The form posts to /contact, which is defined in app.py at line 42.",
	} {
		if announcesImminentToolUse(s) {
			t.Errorf("real answer wrongly flagged as an announcement: %q", s)
		}
	}
}
