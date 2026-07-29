// Guardrails for the agent loop. Centralises the checks that bounce
// model output before it touches disk or the host filesystem.
//
// Why a separate file: the rules accumulate (output sanitisation,
// shell-op blocking, protected paths) and live downstream of multiple
// tool handlers. Keeping them together makes the policy auditable —
// reviewers don't have to chase three call sites to know what we
// reject.
//
// Background: ATLAS runs against compact local coding models that are
// weaker than the API frontier models. Claude-Code-style "trust the
// model + permission prompts" doesn't hold for us; the model will
// reliably emit markdown-fenced code with prose preamble and reach
// for shell `mv`/`rm` against source files mid-task. Server-side
// gates are how we keep the workspace usable.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

// sanitizeFileContent strips markdown wrappers and prose preamble from
// content destined for disk. The local model frequently emits:
//
//	Looking at the task, I need to create a complete index.html...
//
//	```html
//	<!DOCTYPE html>
//	...
//	```
//
//	This file does X, Y, Z.
//
// Without this strip, the whole markdown wrapper lands on disk
// verbatim — Jinja chokes on `{{ url_for(...) }}` fragments inside a
// numbered-list explanation, the user sees a 500, debugging starts.
//
// The function returns (cleaned, modified). modified=true means a
// fence/prose was stripped — the caller should log it so we can spot
// repeat offenders. .md / .markdown / .rst files are passed through
// unchanged because fences are legitimate content there.
//
// Only a WHOLE-FILE wrapper is stripped: the opening fence must sit at
// the very top of the content (preceded by at most a few prose lines),
// and the closing fence may be followed only by a short prose trailer.
// A fence deeper in the file — e.g. a fenced example inside a docstring
// — is legitimate content and passes through unchanged.
func sanitizeFileContent(filePath, content string) (string, bool) {
	ext := strings.ToLower(filepath.Ext(filePath))
	switch ext {
	case ".md", ".markdown", ".rst", ".txt":
		return content, false
	}

	lines := strings.Split(content, "\n")

	// Locate the opening fence within the preamble allowance. More than
	// a few non-empty lines before the first fence — or any line that
	// opens a docstring/comment block — means the fence is interior
	// content, not a wrapper.
	const maxWrapperProseLines = 5
	openIdx := -1
	preambleProse := 0
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			openIdx = i
			break
		}
		if trimmed == "" {
			continue
		}
		preambleProse++
		if preambleProse > maxWrapperProseLines || lineSignalsRealContent(trimmed) {
			return content, false
		}
	}
	if openIdx < 0 {
		return content, false
	}

	closeIdx := -1
	for i := len(lines) - 1; i > openIdx; i-- {
		if strings.TrimSpace(lines[i]) == "```" {
			closeIdx = i
			break
		}
	}

	// Same whole-file requirement on the way out: after the closing
	// fence only a short prose trailer ("This file: 1. ... 2. ...") is
	// allowed. Substantial content or docstring/comment markers after
	// the fence mean the pair is interior — pass through unchanged.
	if closeIdx > openIdx {
		const maxTrailerProseLines = 8
		trailerProse := 0
		for _, line := range lines[closeIdx+1:] {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}
			trailerProse++
			if trailerProse > maxTrailerProseLines || lineSignalsRealContent(trimmed) {
				return content, false
			}
		}
	}

	var extracted []string
	if closeIdx > openIdx {
		extracted = lines[openIdx+1 : closeIdx]
	} else {
		// Unmatched closing fence — model probably truncated. Take
		// everything after the opener; better than discarding the
		// whole file or keeping the prose preamble.
		extracted = lines[openIdx+1:]
	}

	cleaned := strings.Join(extracted, "\n")
	// Preserve a single trailing newline if the original had one — POSIX
	// text files conventionally end with \n.
	if strings.HasSuffix(content, "\n") && !strings.HasSuffix(cleaned, "\n") {
		cleaned += "\n"
	}
	return cleaned, true
}

// docstringDelimiters mark a Python/multiline string. When one appears
// anywhere on a preamble or trailer line, the line is real string content
// (e.g. `DOC = """usage:`) and a fence around it is legitimate — so the
// content is not a whole-file wrapper. These are matched with Contains
// because code commonly precedes the delimiter on the opening line.
var docstringDelimiters = []string{`"""`, "'''"}

// commentBlockOpeners mark a comment block. These are matched by prefix so
// that model prose merely mentioning a marker (e.g. "the /* config */
// block:") does not disqualify a genuine whole-file wrapper, while a line
// that actually opens a comment block does.
var commentBlockOpeners = []string{"/*", "*/", "<!--", "-->"}

// lineSignalsRealContent reports whether a trimmed line indicates the text
// around a fence is real file content (a docstring or comment block) rather
// than model prose wrapping the file.
func lineSignalsRealContent(trimmed string) bool {
	for _, d := range docstringDelimiters {
		if strings.Contains(trimmed, d) {
			return true
		}
	}
	for _, m := range commentBlockOpeners {
		if strings.HasPrefix(trimmed, m) {
			return true
		}
	}
	return false
}

// run_command executes inside the sandbox container, which is already a
// project-folder jail: read-only rootfs, no-new-privileges, ONLY the project
// dir bind-mounted writable at /workspace, and the /shell endpoint forces cwd
// under /workspace. So the model cannot touch the host — the blast radius of
// any shell command is the project folder (recoverable via git). Given that,
// the old "block every mutating verb" policy was overbroad: it made the model
// reinvent mv/cp/rm as bespoke tools and loop when it couldn't (e.g. "mv
// index.html templates/" refused → mkdir loop → stuck). Policy now (2026-06):
// allow shell to manage files freely; block ONLY the few commands that are
// catastrophic even inside the jail — wiping the whole project, fork-bombing
// the sandbox, or destroying a block device. Content edits are still nudged
// toward write_file/edit_file by the system prompt (that's where V3 + the lens
// add value), but they are no longer hard-refused at the shell.

// shellFindDeleteRe catches `find ... -delete` / `find ... -exec rm` — a
// recursive delete whose target is usually `.` (the project root), so its
// blast radius is the whole workspace. Kept blocked; targeted deletes use
// `rm <file>` or delete_file.
var shellFindDeleteRe = regexp.MustCompile(
	`\bfind\b.*?(-delete\b|-exec\s+rm\b)`)

// shellForkBombRe matches the classic fork bomb and close variants: a function
// whose body pipes to itself and backgrounds (`| … &`) then invokes itself.
// The `&` (background spawn) inside the braces is the signature that separates
// a bomb from a benign `f() { ls | grep x; }`.
var shellForkBombRe = regexp.MustCompile(`\(\)\s*\{[^}]*\|[^}]*&[^}]*\}\s*;`)

// shellDeviceWriteRe matches filesystem/device destruction: mkfs/wipefs, `dd`
// onto a device, or a redirect straight onto a block device.
var shellDeviceWriteRe = regexp.MustCompile(
	`\b(mkfs\S*|wipefs)\b|\bdd\b[^|;&]*\bof=/dev/|(^|\s)>\s*/dev/(sd|nvme|mmcblk|vd|hd|xvd)`)

// shellWrapperRe matches a `bash -c "…"` / `sh -c '…'` / `eval …` prefix so we
// can unwrap it and run the catastrophic checks against the REAL command — a
// model that wraps `rm -rf /` in `bash -c` must not slip past the denylist.
var shellWrapperRe = regexp.MustCompile(`^\s*(?:(?:bash|sh|zsh|dash|ksh)\s+-c|eval)\s+`)

// unwrapShellWrapper strips one `bash -c "…"` / `eval "…"` layer (and the
// surrounding quotes) so catastrophic-pattern checks see the inner command.
func unwrapShellWrapper(seg string) string {
	loc := shellWrapperRe.FindStringIndex(seg)
	if loc == nil {
		return seg
	}
	inner := strings.TrimSpace(seg[loc[1]:])
	if len(inner) >= 2 {
		if (inner[0] == '"' && inner[len(inner)-1] == '"') ||
			(inner[0] == '\'' && inner[len(inner)-1] == '\'') {
			inner = inner[1 : len(inner)-1]
		}
	}
	return inner
}

// validateShellCommand returns a non-empty rejection reason ONLY for a command
// that is catastrophic even inside the sandbox jail (whole-project wipe, fork
// bomb, device destruction). Everything else — mv, cp, mkdir, rm of specific
// files, chmod, sed -i, > redirects, build/test/run — is allowed.
func validateShellCommand(cmd string) string {
	stripped := strings.TrimSpace(cmd)
	if stripped == "" {
		return ""
	}
	// Whole-command checks (survive segment splitting / wrapper quoting).
	unwrapped := unwrapShellWrapper(stripped)
	if shellForkBombRe.MatchString(stripped) || shellForkBombRe.MatchString(unwrapped) {
		return "run_command refused: that is a fork bomb — it would exhaust the sandbox's process table. If you need to spawn processes, run them one at a time."
	}
	if shellDeviceWriteRe.MatchString(stripped) || shellDeviceWriteRe.MatchString(unwrapped) {
		return "run_command refused: writing to a block device or formatting a filesystem (dd/mkfs/wipefs) is blocked. Work with files under the project directory instead."
	}

	for _, seg := range splitShellSegments(stripped) {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}
		seg = unwrapShellWrapper(seg)
		if msg := catastrophicRm(seg); msg != "" {
			return msg
		}
		if shellFindDeleteRe.MatchString(seg) {
			return "run_command refused: `find ... -delete` / `-exec rm` recursively deletes from the search root (usually the whole project). Delete specific files with `rm <file>` or the delete_file tool."
		}
	}
	return ""
}

// catastrophicRm flags a recursive `rm` whose target would wipe the whole
// project (or root / home). A targeted recursive delete of a subdirectory
// (`rm -rf __pycache__`, `rm -rf node_modules`, `rm -rf build`) is allowed —
// only roots and glob-everything targets are catastrophic.
func catastrophicRm(seg string) string {
	fields := strings.Fields(seg)
	i := 0
	for i < len(fields) && (fields[i] == "sudo" || strings.Contains(fields[i], "=")) {
		i++ // skip a sudo / leading VAR=val env prefix
	}
	if i >= len(fields) || filepath.Base(fields[i]) != "rm" {
		return ""
	}
	recursive := false
	var targets []string
	for _, f := range fields[i+1:] {
		if strings.HasPrefix(f, "--") {
			if f == "--recursive" {
				recursive = true
			}
			continue
		}
		if strings.HasPrefix(f, "-") {
			if strings.ContainsAny(f, "rR") {
				recursive = true
			}
			continue
		}
		targets = append(targets, f)
	}
	if !recursive {
		return "" // `rm file` / `rm -f file` is fine; only recursive wipes are gated
	}
	for _, t := range targets {
		if isCatastrophicDeleteTarget(t) {
			return "run_command refused: `rm -r` of " + t + " would wipe the whole project (or root). Delete a specific subdirectory by name instead (e.g. `rm -rf build`), or use delete_file."
		}
	}
	return ""
}

// isCatastrophicDeleteTarget reports whether a recursive-rm target is a root /
// home / project-root / glob-everything path.
func isCatastrophicDeleteTarget(t string) bool {
	t = strings.Trim(t, `"'`)
	switch t {
	case "/", "/*", "~", "~/", "~/*", "$HOME", "${HOME}", "$HOME/*", "${HOME}/*",
		".", "./", "./*", "*", "..", "../", "../*",
		"/workspace", "/workspace/", "/workspace/*":
		return true
	}
	return false
}

// workspaceRefRe matches `/workspace` as a path component (preceded by
// non-word char or line start, followed by /, whitespace, end, or
// non-word char). Avoids false matches inside e.g. `/home/foo_workspace`.
var workspaceRefRe = regexp.MustCompile(`(^|[^a-zA-Z0-9_])/workspace(/|\s|$|[^a-zA-Z0-9_])`)

// validateWorkingDirReference rejects shell commands that reference
// `/workspace` when /workspace is not the project's working directory.
//
// Coding models often have a training-data prior toward `/workspace` as a
// generic project sandbox path — coding-assistant fine-tunes use it
// heavily. The system prompt explicitly warns against absolute paths
// but the prior leaks through under conversation pressure. May 8 2026
// flask test: model emitted a correct `cd /home/isaac/snake && python
// app.py` at turn 7, then drifted at turn 9 to `cd /workspace && python
// app.py` and burned three turns retrying that wrong path. This guard
// catches the drift one turn earlier with a rejection that names the
// actual workingDir, so the model can self-correct in one round-trip.
//
// Returns "" if (a) workingDir is empty, (b) cmd doesn't reference
// /workspace, (c) the actual project IS at /workspace (no false reject),
// or (d) the /workspace mention is a substring of an unrelated path
// (`/home/foo_workspace`). Otherwise returns a rejection string.
func validateWorkingDirReference(cmd, workingDir string) string {
	if workingDir == "" {
		return ""
	}
	if !strings.Contains(cmd, "/workspace") {
		return ""
	}
	if workingDir == "/workspace" || strings.HasPrefix(workingDir, "/workspace/") {
		return ""
	}
	if !workspaceRefRe.MatchString(cmd) {
		return ""
	}
	return fmt.Sprintf(
		"command refused: references /workspace, which is not your project root. Working directory is %s — `cd %s && ...` for shell commands, or use relative paths from there. /workspace is a generic training-data prior, not this project's path.",
		workingDir, workingDir)
}

// validateRunCommand chains the shell-mutation gate and the workingDir
// gate. Used by both run_command and run_background paths in the agent
// loop. Empty return = command is allowed.
func validateRunCommand(cmd, workingDir string) string {
	if r := validateShellCommand(cmd); r != "" {
		return r
	}
	if r := validateWorkingDirReference(cmd, workingDir); r != "" {
		return r
	}
	return ""
}

// validateNotSuspiciouslyShrunk rejects writes that replace a
// substantial original with a tiny new payload. May 9 2026 structural_edit
// failure: model emitted only `<!DOCTYPE html>\n` (16B) for an entire
// <html>-element rewrite of a 120B file; the on-disk result was a
// destroyed file passed off as a successful "done". The model usually
// produces this shape when its response stops mid-output (json_object
// grammar + length bias converging on minimal valid
// JSON) — the parser sees a syntactically clean tool_call with empty
// content, no truncation marker fires, the recovery path doesn't
// engage, and the destructive write lands.
//
// Heuristic: skip the check when the original was already small
// (line-level edits often legitimately shrink), reject when the new
// payload is clearly a stub. Threshold history:
//
//	v1 (May 9 2026): newSize < 32 — model slipped a 32B stub past it
//	v2 (May 10 morning): bumped to 128 — false-rejected legit
//	  "5KB function refactored to 80B one-liner" case
//	v3 (current): 64 — catches today's 32B destructive stubs and any
//	  "doctype-only" outputs while leaving room for real one-liner
//	  refactors. Subtler cases (legitimate-shape but bad code) are
//	  V3's job now that structural_edit always routes through it.
func validateNotSuspiciouslyShrunk(toolName, path string, oldSize, newSize int) string {
	if oldSize < 100 {
		return ""
	}
	if newSize >= 64 {
		return ""
	}
	return fmt.Sprintf(
		"%s refused: replacement is suspiciously small (%dB) for an existing %dB target at %s. The model usually emits this shape when its response was cut off mid-output or stopped after only the doctype/scaffolding. Re-emit %s with the FULL replacement body — don't ship a stub for a real rewrite.",
		toolName, newSize, oldSize, path, toolName)
}

// leadingDoctypeRe matches an HTML5 <!DOCTYPE ...> declaration at the
// very start of a string (allowing whitespace before it). Case-insensitive
// per spec.
var leadingDoctypeRe = regexp.MustCompile(`(?i)^\s*<!DOCTYPE[^>]*>\s*\n?`)

// stripLeadingDoctype removes a leading <!DOCTYPE> declaration from
// content. Returns the stripped content and true if a doctype was
// present, the original content and false otherwise. Used by structural_edit
// when the selector is <html> to prevent duplicated doctypes (the
// element selector replaces only <html>...</html>, not the preceding
// doctype).
func stripLeadingDoctype(content string) (string, bool) {
	if loc := leadingDoctypeRe.FindStringIndex(content); loc != nil {
		return content[loc[1]:], true
	}
	return content, false
}

// fixIntentWords tracks vocabulary that signals "the user wants
// something repaired or verified." Reused by the verification gate
// to decide when "done" needs a build/test/run before it passes.
// Kept in sync with classifyAgentTier's fix-intent list.
var fixIntentWords = []string{
	"fix", "broken", "doesn't work", "doesn't", "does not work", "does not",
	"not working", "isn't working", "isn't", "is not", "aren't", "wasn't",
	"didn't", "won't", "can't", "bug", "issue", "problem", "error",
	"failed", "fails", "failing", "incorrect", "wrong", "verify",
	"render", "renders", "rendering", "load", "loads", "loading",
}

// isFixIntentMessage returns true when the user prompt looks like a
// repair/verification request. The verification gate uses this to
// decide whether `done` requires a real verification step. Pure
// feature requests ("add a logout button") don't trip the gate —
// adding code doesn't always need a curl/test to declare done.
func isFixIntentMessage(msg string) bool {
	lower := strings.ToLower(msg)
	for _, w := range fixIntentWords {
		if strings.Contains(lower, w) {
			return true
		}
	}
	return false
}

// actionIntentWords tracks verbs that signal "the user wants something
// CREATED, MODIFIED, or REPLACED on disk." Distinct from
// fixIntentWords (which is about repair/verification) — these match
// feature-build prompts where the model must emit a write_file /
// edit_file / structural_edit / delete_file before `done` is honest.
//
// May 10 2026 false-success case that motivated this: prompt was
// "Rewrite templates/dashboard.html to display a clean SaaS-style
// metrics dashboard..." Model spent 6 turns starting servers and
// curling the placeholder, never edited anything, declared `done`.
// The fix-intent gate didn't fire because "rewrite" isn't a
// fix-intent word — but it IS clearly an action-intent word that
// should have required a productive write.
var actionIntentWords = []string{
	"rewrite", "rewriting", "rewritten",
	"create", "creates", "creating", "created",
	"add", "adds", "adding", "added",
	"implement", "implements", "implementing", "implemented",
	"build", "builds", "building", "built",
	"write", "writes", "writing", "wrote",
	"refactor", "refactors", "refactoring", "refactored",
	"replace", "replaces", "replacing", "replaced",
	"update", "updates", "updating", "updated",
	"modify", "modifies", "modifying", "modified",
	"change", "changes", "changing", "changed",
	"make a", "make the", "make it",
	"convert", "converts", "converting", "converted",
	"redesign", "redesigning", "redesigned",
}

// reOutputFilenameTok matches a filename-looking token: an optional
// leading path, then name.ext (1-6 char extension). Captures group 1.
var reOutputFilenameTok = regexp.MustCompile("[`\"']?((?:[~./]|\\.\\./)?[\\w./-]*\\.[A-Za-z][A-Za-z0-9]{0,5})[`\"']?")

// reOutputWriteVerb matches the stems of verbs that mean "produce this
// file" — used to tell a prompt's OUTPUT file from an INPUT file. `read`
// is deliberately absent (it names an input).
var reOutputWriteVerb = regexp.MustCompile(`(?i)\b(sav|writ|creat|output|generat|stor|produc|recover|dump)`)

// reMustProduce matches the "<file> must exist / must contain" requirement
// phrasing (TB2 merge-diff: "the file algo.py must exist in the merged
// result"), which names a deliverable without a write verb. Checked in a
// window AFTER the filename.
var reMustProduce = regexp.MustCompile(`(?i)^\s*(must (exist|contain|include|be (creat|writt|present|generat)))`)

// expectedOutputPaths extracts the file(s) a task prompt explicitly asks
// the model to produce: a filename token preceded within ~70 chars by a
// write/save/create/output verb. Grounded in the task text (many TB2 and
// real prompts say "save your solution in X", "write the output to Y",
// "create a JSON file Z"), so it can be checked against disk at the end.
// Bounded to the first 2 to avoid over-steering on a chatty prompt.
func expectedOutputPaths(msg string) []string {
	var out []string
	seen := map[string]bool{}
	for _, m := range reOutputFilenameTok.FindAllStringSubmatchIndex(msg, -1) {
		path := msg[m[2]:m[3]]
		if path == "" || strings.Count(path, ".") == len(path) {
			continue
		}
		start := m[0] - 70
		if start < 0 {
			start = 0
		}
		afterEnd := m[1] + 40
		if afterEnd > len(msg) {
			afterEnd = len(msg)
		}
		// Output signal: a write verb within ~70 chars before the filename,
		// OR "must exist/contain" requirement phrasing right after it.
		if !reOutputWriteVerb.MatchString(msg[start:m[0]]) &&
			!reMustProduce.MatchString(msg[m[1]:afterEnd]) {
			continue // input/incidental filename
		}
		if !seen[path] {
			seen[path] = true
			out = append(out, path)
			if len(out) >= 2 {
				break
			}
		}
	}
	return out
}

// missingExpectedOutputs returns the expected output files that do not
// exist on disk. Checks the resolved path with os.Stat so it counts a
// file created by ANY means (write_file OR a run_command that
// redirected/generated it), not just write_file. Stat probes are
// contained to known roots — the workspace, plus the system temp dir
// (host-verify tasks legitimately name /tmp outputs). A path outside
// both is skipped: the gate only enforces deliverables it can check
// without probing arbitrary prompt-derived paths.
func missingExpectedOutputs(ctx *AgentContext, expected []string) []string {
	var missing []string
	roots := []string{filepath.Clean(ctx.WorkingDir), filepath.Clean(os.TempDir())}
	for _, p := range expected {
		resolved := resolveAgentPath(ctx, p)
		for _, root := range roots {
			rel, err := filepath.Rel(root, resolved)
			if err != nil || !filepath.IsLocal(rel) {
				continue
			}
			if _, err := os.Stat(filepath.Join(root, rel)); err != nil {
				missing = append(missing, p)
			}
			break // first containing root decides
		}
	}
	return missing
}

// logPath escapes CR/LF in a request-derived value so a crafted name
// can't forge additional log lines; logPaths is the slice form.
func logPath(p string) string {
	p = strings.ReplaceAll(p, "\n", `\n`)
	return strings.ReplaceAll(p, "\r", `\r`)
}

func logPaths(paths []string) []string {
	out := make([]string, len(paths))
	for i, p := range paths {
		out[i] = logPath(p)
	}
	return out
}

// isActionIntentMessage returns true when the prompt clearly asks
// for a state change on disk (create/rewrite/refactor/etc.). The
// done-without-action gate uses this to bounce a `done` that wasn't
// preceded by any productive write — which would otherwise pass
// through silently because the fix-intent gate ignores feature work.
func isActionIntentMessage(msg string) bool {
	lower := strings.ToLower(msg)
	for _, w := range actionIntentWords {
		if strings.Contains(lower, w) {
			return true
		}
	}
	return false
}

// expectedOutputMissingMessage tells the model the task's named output
// file doesn't exist yet — the deliverable, not just "some change." Names
// the file(s) so the steer is concrete and grounded in the task text.
func expectedOutputMissingMessage(missing []string) string {
	quoted := make([]string, len(missing))
	for i, p := range missing {
		quoted[i] = "`" + p + "`"
	}
	return "Before you finish — the task names " +
		strings.Join(quoted, " and ") +
		" as a deliverable, but it does not exist on disk yet. If your code PRODUCES it when run, run your code now to generate it (do NOT hand-write a fabricated stand-in). If it is a file you author directly, write your solution to it. If you have genuinely already produced it elsewhere or it is not actually required, you may proceed."
}

// actionWithoutProductiveChangeMessage tells the model to actually do
// the work the user asked for before declaring done. Concrete and
// directive — points at the missing tool call, not abstract "you
// haven't done enough." Mirror of verificationRejectionMessage's
// shape.
func actionWithoutProductiveChangeMessage(userMsg string) string {
	return "Cannot declare `done` yet — the user asked you to make a change on disk (rewrite/create/add/implement/refactor/etc.) and you haven't emitted any successful write_file / edit_file / structural_edit / delete_file in this loop. Verification (running the server, curling the page) is NOT the task — it's how you confirm AFTER the change. Re-read the user's request, identify what file needs to change, and emit the appropriate edit tool. Then verify, then done."
}

// verificationCommandRe matches the leading token of commands that
// actually verify something (build, test, run, fetch). Used by the
// verification gate to recognise when the model has done due
// diligence before declaring done. ls/cat/grep/echo deliberately
// excluded — those are recon, not verification.
var verificationCommandRe = regexp.MustCompile(
	`^\s*(` +
		// Test runners
		`pytest|python\s+-m\s+pytest|nose|tox|` +
		// Build / type-check / static analysis
		`mypy|ruff|pylint|tsc|eslint|gofmt|vet|markdownlint|stylelint|` +
		`shellcheck|hadolint|flake8|rubocop|golangci-lint|` +
		// Run-the-thing
		`python|python3|node|deno|bun|ruby|cargo\s+run|cargo\s+test|cargo\s+check|cargo\s+build|` +
		`go\s+run|go\s+test|go\s+build|go\s+vet|` +
		`npm\s+(test|run|start)|yarn\s+(test|run|start)|pnpm\s+(test|run|start)|` +
		`make(\s+|$)|just(\s+|$)|` +
		// HTTP probes
		`curl|wget|http\b|httpie\b` +
		`)`)

// isVerificationCommand returns true when a run_command call counts
// as proof the agent verified its work. Recon (ls, cat, grep, find)
// returns false — listing a directory doesn't tell you the code
// works. Build/test/run/curl returns true: those exercise the code
// path and a clean exit means something.
func isVerificationCommand(cmd string) bool {
	return verificationCommandRe.MatchString(strings.TrimSpace(cmd))
}

// wantsStateChange reports whether `done` should be blocked when no write,
// edit, or delete succeeded in this run.
//
// actionIntentWords alone was the test, and it is an open vocabulary that
// cannot be completed: it lists "create"/"add"/"make" but not
// "remove"/"delete", so "remove the debug logging from app.py" armed no
// gate and the model could close the turn having deleted nothing. Adding
// those two words leaves the next verb missing.
//
// The second signal is observed instead of guessed: a read-only tool
// succeeded, so the model opened the project rather than answering from the
// message alone. That covers any phrasing, including verbs no list has.
//
// It needs the tier to stay honest, because reading files is also how a
// question gets answered. "why does the game store direction as a string"
// opens the file and correctly writes nothing; classifyAgentTier calls that
// conversational, and conversational messages are never gated. What remains
// is the case worth blocking: a non-conversational message, the model went
// into the project, and nothing changed on disk.
func wantsStateChange(userMessage string, tier Tier, inspectedWorkspace bool) bool {
	if isActionIntentMessage(userMessage) {
		return true
	}
	return inspectedWorkspace && tier != Tier0Conversational
}

// gateTrigger names why the verification gate fired, for the log line. A
// red command outranks message shape: it is the concrete signal, and when
// both hold it is the one that describes what actually happened.
func gateTrigger(userWantsVerification, sawFailedVerification bool) string {
	switch {
	case sawFailedVerification:
		return "failed-verification"
	case userWantsVerification:
		return "fix-intent"
	default:
		return "none"
	}
}

// verificationRejectionMessage tells the model exactly what's
// missing and what to run. We prefer concrete suggestions over
// abstract "verify your work" prompts — the model is more likely to
// pick a sensible command when given a category.
//
// sawFailedVerification distinguishes the two ways this gate fires. When a
// verification command has actually gone red in this loop, the run holds
// concrete evidence of breakage, so the message says that rather than
// describing the request — the model has already seen the failure and needs
// to act on it, not be told what verification is.
func verificationRejectionMessage(sawFailedVerification bool) string {
	if sawFailedVerification {
		return "Cannot declare `done` — a test or build command you ran in this session FAILED and nothing has passed since. You have already seen the failure output. Apply the fix with `edit_file`, `structural_edit`, or `write_file`, then re-run the same command and confirm it exits clean. Describing the fix is not applying it: if you know what the problem is, make the edit now. Declaring done over a red test reports a broken result as a working one."
	}
	return "Cannot declare `done` yet — this is a fix/repair request and you haven't verified the change works. Before emitting `done`, run a verification command and confirm it succeeded. Examples: `python app.py` to start a server, `curl http://localhost:5000/` to probe a route, `pytest tests/` to run tests, `npm test` for Node, `go test ./...` for Go. \"Done\" without a clean verification exit is a guess, not a fix."
}

// splitShellSegments splits a command line on `&&`, `||`, `;`, `|`
// while ignoring those characters when they appear inside single
// or double quotes. Best-effort, not a real shell parser — but enough
// for the model-emitted commands we want to gate.
func splitShellSegments(cmd string) []string {
	var out []string
	var cur strings.Builder
	inSingle, inDouble := false, false
	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		switch c {
		case '\'':
			if !inDouble {
				inSingle = !inSingle
			}
		case '"':
			if !inSingle {
				inDouble = !inDouble
			}
		}
		if !inSingle && !inDouble {
			if c == '&' && i+1 < len(cmd) && cmd[i+1] == '&' {
				out = append(out, cur.String())
				cur.Reset()
				i++
				continue
			}
			if c == '|' && i+1 < len(cmd) && cmd[i+1] == '|' {
				out = append(out, cur.String())
				cur.Reset()
				i++
				continue
			}
			if c == ';' || c == '|' {
				out = append(out, cur.String())
				cur.Reset()
				continue
			}
		}
		cur.WriteByte(c)
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}

// isNewWrite returns true when the resolved path doesn't yet exist on
// disk. Used by stub-detection / pattern-reflex gates to scope their
// rejection logic to genuinely new files — modifying an existing file
// is a different shape and the V3 / surgical-edit gate handles those.
func isNewWrite(resolvedPath string) bool {
	_, err := os.Stat(resolvedPath)
	return os.IsNotExist(err)
}

// stubHTMLRe catches `<h1>Foo Page</h1>` / `<h1>Bar Section</h1>` —
// the exact shape the model emits when it gives up and ships a
// placeholder. Matches inside <body>, allows whitespace.
var stubHTMLRe = regexp.MustCompile(
	`(?is)<h\d>\s*[A-Za-z]+\s+(page|section|title|content|view)\s*</h\d>`)

// looksLikeStub returns a non-empty rejection string when the content
// looks like a placeholder/stub. PC-195. The model's lazy-completion
// failure mode is to ship 8-line skeletons that pass syntactic gates
// but ship the absolute minimum content to claim "done." Catches the
// most egregious shapes per file type; deliberately conservative —
// short content that has REAL substance (one-liner shell scripts,
// minimal Dockerfiles, single-import test files) passes through.
//
// The fix is to either model the file from a sibling (templates/index.html
// usually has the right scaffold) or — if the user really did ask for
// a placeholder — say so in the response so the user knows.
func looksLikeStub(displayPath, content string) string {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return "write_file refused: content is empty. If you mean to create an empty file, write a meaningful starting structure or `touch` it via run_command."
	}

	ext := strings.ToLower(filepath.Ext(displayPath))
	lineCount := strings.Count(trimmed, "\n") + 1

	switch ext {
	case ".html", ".htm":
		// 200 chars is the cliff — full pages don't fit under that.
		if len(trimmed) < 200 && stubHTMLRe.MatchString(trimmed) {
			return stubRejectionMessage(displayPath,
				"the body is just `<h1>X Page</h1>` with no real content")
		}
	case ".py":
		// Functions whose body is `pass` or a single TODO comment.
		if lineCount <= 5 && (regexp.MustCompile(`(?m)^\s*pass\s*$`).MatchString(trimmed) ||
			regexp.MustCompile(`(?im)^\s*#\s*TODO\b.*$`).MatchString(trimmed)) {
			if !strings.Contains(trimmed, "import ") && !strings.Contains(trimmed, "def ") && !strings.Contains(trimmed, "class ") {
				return stubRejectionMessage(displayPath,
					"the file body is just `pass` / `# TODO` with no real implementation")
			}
		}
	case ".md", ".markdown":
		if len(trimmed) < 100 && (strings.Contains(strings.ToLower(trimmed), "todo") ||
			strings.Contains(strings.ToLower(trimmed), "placeholder")) {
			return stubRejectionMessage(displayPath,
				"the document is just a TODO/placeholder marker")
		}
	case ".js", ".ts", ".tsx", ".jsx":
		// React component / module that's just an empty fragment or
		// a `<div>Page</div>` placeholder.
		if len(trimmed) < 200 && regexp.MustCompile(`(?is)return\s*\(?\s*<[a-z0-9]+>\s*[A-Za-z]+\s+(page|section|view)\s*</[a-z0-9]+>\s*\)?`).MatchString(trimmed) {
			return stubRejectionMessage(displayPath,
				"the component just returns `<X>Foo Page</X>` with no real markup")
		}
	}
	return ""
}

func stubRejectionMessage(path, why string) string {
	return fmt.Sprintf(
		"write_file refused: %s looks like a placeholder stub — %s. Either (a) read a sibling file in the same directory to model the structure (the project's other %s files almost certainly have the right scaffold), or (b) if the user explicitly asked for an empty placeholder, acknowledge that in your response so they know the file needs to be filled in. Don't ship stubs and call the task done.",
		path, why, strings.TrimPrefix(filepath.Ext(path), "."))
}

// patternMatchHint returns a non-empty rejection string when the model
// is creating a NEW file in a directory that already contains files of
// the same extension AND it hasn't read any of those siblings in this
// session. PC-194. Forces the "model from existing patterns" reflex
// instead of generating from scratch — a NEW route handler should
// match the project's existing route handlers, a new test should match
// the existing test conventions, etc.
//
// Only fires when:
//   - The target path doesn't exist (genuinely new file, not an edit)
//   - The parent directory contains ≥1 sibling with the same extension
//   - ctx.FilesRead doesn't include any of those siblings
//
// Soft-coupled to AgentContext via the FilesRead snapshot we pass in;
// keeps the helper testable without dragging the whole context type in.
func patternMatchHint(resolvedPath, _ string) string {
	if !isNewWrite(resolvedPath) {
		return ""
	}
	dir := filepath.Dir(resolvedPath)
	ext := strings.ToLower(filepath.Ext(resolvedPath))
	if ext == "" {
		return ""
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	var siblings []string
	for _, e := range entries {
		if e.IsDir() || strings.ToLower(filepath.Ext(e.Name())) != ext {
			continue
		}
		full := filepath.Join(dir, e.Name())
		if full == resolvedPath {
			continue
		}
		siblings = append(siblings, e.Name())
	}
	// Need a meaningful neighborhood — single-sibling dirs are too noisy
	// (one-off configs, isolated entry points). Two or more is enough
	// to call it a "pattern."
	if len(siblings) < 2 {
		return ""
	}
	// Don't gate when no session context is available — would block
	// the first write of every new project. The agent loop wires this
	// to ctx.FilesRead via the read-tracker; absence here means we
	// can't reason about it.
	read := patternReadTracker.snapshot()
	for _, s := range siblings {
		if _, ok := read[filepath.Join(dir, s)]; ok {
			return ""
		}
	}
	preview := siblings
	if len(preview) > 3 {
		preview = preview[:3]
	}
	return fmt.Sprintf(
		"write_file deferred: you're creating a new %s file in %s, which already contains %d sibling %s files (e.g. %s). Read at least one of those first so this new file follows the project's existing conventions (style, imports, structure). Then re-issue the write_file call.",
		ext, dir, len(siblings), ext, strings.Join(preview, ", "))
}

// patternReadTracker is a tiny indirection so patternMatchHint can ask
// "did the agent read any of these sibling files?" without taking an
// AgentContext dep that would force the function into the agent
// package's import cycle. The agent loop populates this on every
// successful read_file via patternReadTracker.add(absPath); the gate
// reads from snapshot(). Cleared per-session via reset().
//
// Tradeoff: the tracker is process-global, so concurrent sessions in
// one proxy share it. That's fine — a sibling read by ANY recent
// session is signal that the model's been there. We bound memory by
// capping at 200 entries (LRU-ish via insertion order).
var patternReadTracker = newReadTracker()

type readTracker struct {
	mu    sync.Mutex
	items map[string]struct{}
}

func newReadTracker() *readTracker { return &readTracker{items: map[string]struct{}{}} }

func (r *readTracker) add(absPath string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.items) > 200 {
		// Cheap eviction: drop the whole map. Anything still
		// relevant will be re-added on next read.
		r.items = map[string]struct{}{}
	}
	r.items[absPath] = struct{}{}
}

func (r *readTracker) snapshot() map[string]struct{} {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[string]struct{}, len(r.items))
	for k := range r.items {
		out[k] = struct{}{}
	}
	return out
}

// looksCorruptedOnDisk returns true when the file at displayPath has
// the markdown-fence-with-prose corruption pattern that
// sanitizeFileContent strips on input. PC-201.
//
// The corruption shape is what `<model> generated` left behind in
// May 2026 templates: prose preamble ("Looking at the task, I need
// to create..."), then a ```html fence, then real HTML, then a
// closing fence with trailing commentary. Once on disk, this file
// is unparseable to Jinja/the browser, but PC-159's surgical-edit
// gate blocks write_file from cleaning it up. This helper tells the
// agent loop "the file is broken, let write_file overwrite it."
//
// Mechanism: re-runs the same sanitizer that filters write_file
// inputs against the existing on-disk content. If sanitizing would
// change anything, the file is corrupted in the way we know how to
// recognize. False positives are bounded — sanitizeFileContent only
// strips when a fence is present, so a clean file (no fence) always
// returns false here.
func looksCorruptedOnDisk(displayPath, existing string) bool {
	cleaned, sanitized := sanitizeFileContent(displayPath, existing)
	return sanitized && cleaned != existing
}
