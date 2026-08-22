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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
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
// Two properties hold over the whole operation, both established by fuzzing
// (a single pass satisfied neither):
//
//   - It never empties a file. One pass on content whose only fence is an
//     unmatched opener took "everything after the opener" — nothing — and
//     returned "", so a generation truncated right after ```python would
//     have landed on disk as an empty file with modified=true.
//   - It is idempotent. One pass strips one layer, so doubly-wrapped content
//     came back still carrying a ``` line, which is a syntax error in every
//     language this runs on. Stripping to a fixpoint means the result never
//     needs another pass.
func sanitizeFileContent(filePath, content string) (string, bool) {
	// Strip to a fixpoint rather than to a fixed number of layers: any fixed
	// bound is a case where the result still needs another pass, which is
	// the non-idempotence this is here to avoid. Termination is not a
	// question of taste — every successful strip consumes at least the
	// opener line, so the content strictly shrinks, and the line count is a
	// hard upper bound on how many layers can exist at all.
	cleaned := content
	modified := false
	maxLayers := strings.Count(content, "\n") + 2
	for i := 0; i < maxLayers; i++ {
		next, changed := stripOneFenceLayer(filePath, cleaned)
		if !changed || next == cleaned {
			break
		}
		// A sanitizer that empties a file is destroying the write, not
		// cleaning it. Keep the last non-empty form.
		if strings.TrimSpace(next) == "" && strings.TrimSpace(cleaned) != "" {
			break
		}
		cleaned, modified = next, true
	}
	return cleaned, modified
}

// isDocumentAsset reports whether a path is prose rather than code.
//
// This is the set stripOneFenceLayer has always used to decide that a file's
// fences are its CONTENT, not a wrapper around it -- a markdown document full
// of code blocks is still a document. Naming it lets the completion decision
// ask the same question, and the answer has to be an allowlist: an unknown
// extension holding source-like logic, or an unsupported language like .rs,
// must fall outside it, because "no checker ran" means something entirely
// different for those than it does for a text file.
func isDocumentAsset(filePath string) bool {
	switch strings.ToLower(filepath.Ext(filePath)) {
	case ".md", ".markdown", ".rst", ".txt":
		return true
	}
	return false
}

// stripOneFenceLayer removes a single whole-file markdown wrapper. Callers
// want sanitizeFileContent, which drives this to a fixpoint.
func stripOneFenceLayer(filePath, content string) (string, bool) {
	if isDocumentAsset(filePath) {
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
	if r := foregroundServerRejection(cmd); r != "" {
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
// promisesMoreContent reports an answer that ends by promising content it
// never delivers — "I will now provide the specific location", "let me give
// you the exact comparison".
//
// Distinct from announcesImminentToolUse: that one catches announcing a TOOL
// call before any work has happened. This catches a reply that has done the
// work, then signs off promising the actual answer. Observed on a bug-find
// task: the model named the file, described the symptom, and ended with "I
// will now provide the specific location and the incorrect comparison as
// requested" — and the turn ended there, leaving the user a half-answer.
//
// Requires the promise to be at the END, because "I'll explain why below"
// followed by the explanation is fine.
func promisesMoreContent(text string) bool {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return false
	}
	lower := strings.ToLower(trimmed)
	for _, phrase := range []string{
		"i will now provide", "i'll now provide", "i will now give",
		"i'll now give", "i will provide the", "i'll provide the",
		"let me provide the", "let me give you the", "i will now show",
		"i'll now show", "here is what i will", "i will now list",
	} {
		at := strings.LastIndex(lower, phrase)
		if at < 0 {
			continue
		}
		// A promise that is FOLLOWED by the thing promised is fine —
		// "I'll provide the details: line 314 uses > where it should use <"
		// delivers in the same breath. What is broken is a promise with
		// nothing concrete after it. Digits, operators and backticks are the
		// cheap signal for "concrete", and the remaining prose after an
		// undelivered promise ("...as requested.") has none of them.
		rest := lower[at+len(phrase):]
		return !strings.ContainsAny(rest, "0123456789`=<>+*/(){}[]")
	}
	return false
}

// announcesImminentToolUse reports first-person narration of a tool call the
// model is about to make — "I need to read X", "let me look at Y", "I'll
// start by outlining".
//
// A `text` reply ends the turn, so a model that announces instead of acting
// stops with the right intent and no action. Deliberately narrow: it needs a
// first-person subject AND an action verb aimed at inspecting the workspace,
// so an ANSWER that merely mentions reading ("this function reads the file")
// does not match.
func announcesImminentToolUse(text string) bool {
	lower := strings.ToLower(strings.TrimSpace(text))
	if lower == "" {
		return false
	}
	subjects := []string{"i need to ", "i'll ", "i will ", "let me ", "i am going to ",
		"i'm going to ", "i should ", "first, i ", "next, i "}
	// "look into" was missing and "look at" was not enough: observed on a
	// fresh workspace, "How does the contact form work?" was answered with
	// "I'll look into the contact form's implementation..." and nothing else.
	// `text` is a terminal exit, so an announcement that slips this check ends
	// the turn and the user gets a promise instead of an answer.
	verbs := []string{"read", "look at", "look into", "look through", "look over",
		"open", "inspect", "examine", "outline", "check", "search", "list",
		"start by", "investigate", "dig into", "trace through", "review the",
		"take a look"}
	for _, sub := range subjects {
		at := strings.Index(lower, sub)
		if at < 0 {
			continue
		}
		// Look only just past the subject: "I need to read" matches, while
		// "I need to explain why the code reads a file" does not.
		window := lower[at:]
		if len(window) > 80 {
			window = window[:80]
		}
		for _, v := range verbs {
			if strings.Contains(window, v) {
				return true
			}
		}
	}
	return false
}

// inlineProgramFlagRe matches an interpreter invoked with a program passed
// inline on the command line rather than as a file.
var inlineProgramFlagRe = regexp.MustCompile(`\b(python3?|node|perl|ruby)\s+-(c|e)\b`)

// shellParseFailureSignatures are the messages bash emits when it cannot parse
// the command at all, so nothing ran. Distinct from a command that ran and
// failed: there is no output to reason about and no partial effect to undo.
var shellParseFailureSignatures = []string{
	"syntax error near unexpected token",
	"unexpected EOF while looking for matching",
	"syntax error: unexpected end of file",
}

// shellQuotingHint explains a command the shell could not parse, when the
// cause is a program quoted inline after -c or -e.
//
// Measured across runs 14-16: run_command is the least reliable tool in the
// harness at 86 failures in 156 calls, and 53 of those trace to this one
// shape. Eight commands failed to parse and the model re-sent them 45 times.
// The raw bash error names a token and a column, which says nothing the model
// can act on, so it changed nothing and sent the identical line again. One
// session did this twelve times before writing the code to a file instead,
// which worked first try.
//
// Nesting is why the shape fails rather than any one typo: the outer shell
// quotes, the quotes inside the program, and f-string braces all have to
// agree. The observed commands were not merely mis-shell-quoted, they were
// also invalid Python (`print(f'...{...}")` opens on a quote and closes on a
// double quote), so no amount of re-quoting by the harness could rescue them.
// The fix is to stop nesting: put the program in a file.
func shellQuotingHint(command, errMsg string) string {
	if command == "" || errMsg == "" {
		return ""
	}
	parseFailed := false
	for _, sig := range shellParseFailureSignatures {
		if strings.Contains(errMsg, sig) {
			parseFailed = true
			break
		}
	}
	if !parseFailed || !inlineProgramFlagRe.MatchString(command) {
		return ""
	}
	return "\n\nThe shell could not parse this, so nothing ran and re-sending it " +
		"unchanged fails identically. Quoting a program inline after -c is fragile: " +
		"the shell's quotes, the quotes inside your code, and any braces all have to " +
		"nest correctly. Write the snippet to a file with write_file (say check.py) " +
		"and run it with `python3 check.py`. Then the quoting is only the language's " +
		"problem, and whatever comes back is a real error in the code."
}

// fileSHA256 hashes the current on-disk bytes of a session path, "" when
// the file can't be read.
func fileSHA256(ctx *AgentContext, path string) string {
	data, err := os.ReadFile(resolveAgentPath(ctx, path))
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// sessionWriteHashes snapshots the sha256 of every file this session wrote,
// keyed by the path as the model sent it. Taken at the moment a verifying
// run succeeds, it records WHICH bytes that run vouched for.
func sessionWriteHashes(ctx *AgentContext) map[string]string {
	if ctx == nil || len(ctx.SessionWrites) == 0 {
		return nil
	}
	out := make(map[string]string, len(ctx.SessionWrites))
	for p := range ctx.SessionWrites {
		if h := fileSHA256(ctx, p); h != "" {
			out[p] = h
		}
	}
	return out
}

// commandNamesPath reports whether the command line names the file as a
// whole token (path, basename, or */basename) in any shell segment. Within
// a command already classified as verification, a named file participated
// in what was verified — passed to an interpreter, a test runner, or a
// grader as an argument. Substring matching is deliberately avoided:
// "solve.py" must not match "solve.py.bak".
func commandNamesPath(command, path string) bool {
	base := filepath.Base(path)
	for _, segment := range splitShellSegments(command) {
		for _, tok := range strings.Fields(segment) {
			tok = strings.Trim(tok, `"'`)
			if tok == path || tok == base || strings.HasSuffix(tok, "/"+base) {
				return true
			}
		}
	}
	return false
}

// --- work-contract verification demand ---------------------------------------
//
// verificationDemandedAndUnmet asks one session-wide question: did anything
// pass. It cannot say WHAT was verified or at WHICH bytes, so `echo ok` clears
// it and a rewrite after a green run does not re-arm it. The evidence that can
// answer already exists -- ctx.VerificationEvidence records, per green command,
// the sha256 of each file the command actually NAMED -- and until now its only
// consumer was lens labelling. A completion decision is exactly the place that
// evidence was built for.
//
// Scope is deliberately narrow: a client that declared task_mode work. Nothing
// here touches contractless callers, questions, documents, deletion, moves,
// debt, hazards, tombstones, permission or timeouts.

type verificationDemand struct {
	Required bool
	Met      bool
	// Missing names the first deliverable that has no current, relevant,
	// green evidence, or the first declared command that did not run against
	// the final bytes. Empty when Met.
	Missing string
}

// codeDeliverablesFor is the set this demand covers: paths the client declared
// plus paths the session wrote, restricted to extensions the syntax gate knows.
// Documents keep the existing exact-current-hash rule and are not included.
func codeDeliverablesFor(ctx *AgentContext, expected []string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(rel string) {
		if rel == "" {
			return
		}
		resolved := resolveAgentPath(ctx, rel)
		// Executability, not registry membership: a static or declarative
		// artifact has a checker but nothing to run, and demanding an
		// execution that names it is an obligation nothing can discharge.
		// Its bytes are still held to the existing validation contract.
		if meta, gated := syntaxGateLanguages[strings.ToLower(filepath.Ext(resolved))]; !gated ||
			!meta.Executable {
			return
		}
		if seen[resolved] {
			return
		}
		seen[resolved] = true
		out = append(out, resolved)
	}
	for _, rel := range expected {
		add(rel)
	}
	if ctx != nil {
		ctx.LedgerMu.Lock()
		for key, d := range ctx.Ledger {
			if d.Tombstoned || d.Generation == 0 {
				continue
			}
			add(key)
		}
		ctx.LedgerMu.Unlock()
	}
	sort.Strings(out)
	return out
}

// evidenceIsCurrent reports whether a green record still describes the bytes on
// disk for every path it covered. A later mutation to any covered path makes
// the record stale, which is the verify-then-modify hole stated as a rule.
// The record is keyed by the path the model wrote, the deliverable set by the
// path the client declared or the ledger owns. Both are put through
// resolveAgentPath -- the one canonicalisation rule -- so "solve.py" and
// "./solve.py" are the same file without a second normalizer.
func evidenceIsCurrent(ctx *AgentContext, rec VerificationRecord) (map[string]string, bool) {
	if len(rec.Covered) == 0 {
		return nil, false
	}
	out := make(map[string]string, len(rec.Covered))
	for p, h := range rec.Covered {
		if fileSHA256(ctx, p) != h {
			return nil, false
		}
		out[resolveAgentPath(ctx, p)] = h
	}
	return out, true
}

// contractRequiresCommand reports whether the client declared this exact
// command. isVerificationCommand recognises builds, tests, probes and runners
// by shape, and a client's own requirement may be none of those -- a linter, a
// schema check, a project script. Without this, a declared command could be
// impossible to satisfy: the demand would require it and nothing would ever
// record it running.
//
// It changes what gets RECORDED, never how strongly the record counts. The
// evidence is the same execution, bound to the same hashes; declaring a command
// says who required it, not that its passing proves more.
func contractRequiresCommand(ctx *AgentContext, command string) bool {
	if ctx == nil || ctx.TaskContract == nil {
		return false
	}
	for _, want := range ctx.TaskContract.Verification {
		if want == command {
			return true
		}
	}
	return false
}

// decideVerificationDemand is the single owner of the work-contract demand.
//
// Fails closed everywhere: no evidence, stale evidence, evidence that names a
// different path, or a declared command that never ran against the final bytes
// all leave the demand unmet. A successful command that names nothing covers
// nothing, so `true` and `echo ok` cannot satisfy it -- not because they are
// recognised, but because they carry no binding.
func decideVerificationDemand(ctx *AgentContext, tc *TaskContract, expected []string) verificationDemand {
	if ctx == nil || tc == nil || tc.TaskMode != TaskModeWork {
		return verificationDemand{}
	}
	paths := codeDeliverablesFor(ctx, expected)
	// A declared command is a requirement in its own right. It survives when
	// the run produced nothing executable -- a client that asked for
	// `htmlhint index.html` asked for it regardless of what the registry
	// thinks can be run.
	if len(paths) == 0 && len(tc.Verification) == 0 {
		return verificationDemand{}
	}
	type liveRecord struct {
		command string
		covered map[string]string
	}
	current := make([]liveRecord, 0, len(ctx.VerificationEvidence))
	for _, rec := range ctx.VerificationEvidence {
		if covered, ok := evidenceIsCurrent(ctx, rec); ok {
			current = append(current, liveRecord{command: rec.Command, covered: covered})
		}
	}
	for _, p := range paths {
		h := fileSHA256(ctx, p)
		if h == "" {
			return verificationDemand{Required: true, Missing: p}
		}
		covered := false
		for _, rec := range current {
			if rec.covered[p] == h {
				covered = true
				break
			}
		}
		if !covered {
			return verificationDemand{Required: true, Missing: p}
		}
	}
	// Declared commands are matched by exact recorded identity. No shell
	// parsing, no equivalence: "python3  solve.py" is not "python3 solve.py".
	for _, want := range tc.Verification {
		ran := false
		for _, rec := range current {
			if rec.command == want {
				ran = true
				break
			}
		}
		if !ran {
			return verificationDemand{Required: true, Missing: want}
		}
	}
	return verificationDemand{Required: true, Met: true}
}

// driftedSinceVerification names the first session-written file whose bytes
// no longer match the verified snapshot, or "" when everything still does.
// A file written AFTER the snapshot (absent from it) is drift by definition:
// it has never been executed.
func driftedSinceVerification(ctx *AgentContext, verified map[string]string) string {
	if ctx == nil {
		return ""
	}
	for p := range ctx.SessionWrites {
		want, seen := verified[p]
		if !seen {
			return p
		}
		data, err := os.ReadFile(resolveAgentPath(ctx, p))
		if err != nil {
			return p
		}
		sum := sha256.Sum256(data)
		if hex.EncodeToString(sum[:]) != want {
			return p
		}
	}
	return ""
}

// silentRunWhenOutputPromised reports a verification run that exited 0 while
// printing nothing, on a task whose prompt demands printed output.
//
// A file whose tail has been swallowed by a comment (one drifting "#" line
// eating the solve() call) still parses, runs, and exits 0. Empty stdout is
// then indistinguishable from success unless someone asks whether output was
// promised. Only run_commands that execute a program are held to it —
// build/compile steps legitimately print nothing.
func silentRunWhenOutputPromised(ctx *AgentContext, userMessage, command string, data json.RawMessage) bool {
	lower := strings.ToLower(userMessage)
	if !strings.Contains(lower, "print") && !strings.Contains(lower, "output") {
		return false
	}
	cmd := strings.TrimSpace(command)
	runsProgram := strings.Contains(cmd, "python") || strings.Contains(cmd, "node ") ||
		strings.Contains(cmd, "go run") || strings.HasPrefix(cmd, "./")
	if !runsProgram {
		return false
	}
	var out struct {
		Stdout string `json:"stdout"`
	}
	if json.Unmarshal(data, &out) != nil {
		return false
	}
	return strings.TrimSpace(out.Stdout) == ""
}

// stdinRedirectRe matches a shell stdin redirect from a plain filename
// anywhere in a segment: `python3 solve.py < input.txt`, including with
// trailing redirections after it (`< input.txt > out.txt`) — the
// trailing-only anchor missed those and the contract gate stayed silent
// (audit finding). `<<` heredocs and `<(...)` process substitution are not
// the shape this is about and are excluded by the caller.
var stdinRedirectRe = regexp.MustCompile(`(?:^|[^<>])<\s*([A-Za-z0-9_./-]+)`)

// pipeCatRe matches the pipe idiom that feeds a file to the next command's
// stdin: `cat input.txt | prog`. Same contract as `prog < input.txt`.
var pipeCatRe = regexp.MustCompile(`(?:^|&&|\|\||;)\s*cat\s+([A-Za-z0-9_./-]+)\s*\|[^|]`)

// stdinRedirectSource names the file a command pipes into a program's stdin,
// or "" when it does not.
//
// A program run as `prog < data` is being verified under a contract the
// caller may never use. Measured on the AoC tasks, whose prompt says the
// program must read input.txt: 7 of 10 failures wrote a program that reads
// stdin, ran it as `python3 solve.py < input.txt`, and got a successful
// result — so the model had every reason to believe it was done. The checker
// then ran `python solve.py` with no redirect and got 0. None of the sessions
// that verified this way passed.
//
// The same model with no shell never does this: it writes code that opens the
// file, because piping is not available to it. The tool is what makes the
// wrong shape reachable, so the harness is what has to notice.
func stdinRedirectSource(command string) string {
	cmd := strings.TrimSpace(command)
	if cmd == "" {
		return ""
	}
	// `cat file | prog` feeds prog's stdin exactly like `prog < file`.
	if m := pipeCatRe.FindStringSubmatch(cmd); m != nil {
		return m[1]
	}
	for _, seg := range splitShellSegments(cmd) {
		if strings.Contains(seg, "<<") || strings.Contains(seg, "<(") {
			continue
		}
		if m := stdinRedirectRe.FindStringSubmatch(seg); m != nil {
			return m[1]
		}
	}
	return ""
}

// redirectOnlyVerificationMessage tells the model to run the artifact the way
// its caller will.
func redirectOnlyVerificationMessage(source string) string {
	return fmt.Sprintf(
		"Every time you ran the program you piped a file into it: `< %s`. That "+
			"verifies it as a filter reading stdin, which is not how it will be "+
			"run. %s is sitting in the working directory, so run it standalone "+
			"— `python3 <yourfile>` with no `<` — and make it open %s itself. "+
			"If it prints nothing or 0 that way, it is reading stdin and needs "+
			"to read the file instead.",
		source, source, source)
}

// fileCitationRe matches a filename as it appears in prose: `scoring.py`,
// planning.py, src/app/main.go. The extension must start with a letter and run
// 1-5 characters, so version strings ("V3.2") and decimals ("0.34") are not
// read as paths. Anything it over-matches is discarded by the existence check
// in unreadFileCitations.
var fileCitationRe = regexp.MustCompile(`[A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z][A-Za-z0-9]{0,4}`)

// maxCitedPaths caps how many unread files one rejection names. Listing every
// one turns the correction into a chore; the model needs the shape and a
// couple of concrete targets.
const maxCitedPaths = 3

// unreadFileCitations returns files in the workspace that the reply makes a
// claim about without the run ever having been shown their contents.
//
// A reply that names a file it never opened is guessing, and the guess reads
// exactly like knowledge. Measured on a diagnostic question across three
// modules: 12 of 12 sessions ran list_directory, outlined ONE file, and
// answered. Which file they guessed decided the outcome — scoring.py wrong
// 11/11, planning.py right 1/1 — because the prompt said "scored" and the
// filename matched. One session cited "lines 134-142" of a file whose body it
// had never seen.
//
// The predicate is existence plus absence of evidence, not a judgement about
// the claim: the file has to be real, and the run has to have never read it.
// A file the model wrote is evidence enough, since it authored the contents.
func unreadFileCitations(ctx *AgentContext, text string) []string {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	var out []string
	seen := map[string]bool{}
	for _, m := range fileCitationRe.FindAllString(text, -1) {
		name := strings.Trim(m, "./-")
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		resolved := resolveAgentPath(ctx, name)
		if info, err := os.Stat(resolved); err != nil || info.IsDir() {
			continue
		}
		if ctx.WasBodySeen(resolved) {
			continue
		}
		out = append(out, name)
		if len(out) == maxCitedPaths {
			break
		}
	}
	return out
}

// unreadCitationMessage tells the model to look before it answers.
//
// It names outline_file explicitly because that is the tool the failure runs
// through: an outline lists signatures and line ranges with no bodies, which
// is enough scaffolding to state a confident, specific, wrong claim about what
// those lines contain.
func unreadCitationMessage(paths []string) string {
	var b strings.Builder
	b.WriteString("Your reply makes a claim about ")
	for i, p := range paths {
		switch {
		case i == 0:
		case i == len(paths)-1:
			b.WriteString(" and ")
		default:
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "`%s`", p)
	}
	subject, object := "that file", "it"
	if len(paths) > 1 {
		subject, object = "those files", "each of them"
	}
	fmt.Fprintf(&b, ", but this session has never seen the contents of %s. ", subject)
	fmt.Fprintf(&b, "outline_file lists signatures and line ranges only — it shows you no code, "+
		"so anything you say about what those lines do is a guess. "+
		"Call read_file on %s now, and on any other file you are about to name as the cause, "+
		"then answer from what the code actually says. ", object)
	b.WriteString(
		"If you are not sure which file holds the problem, search_files for the relevant symbol " +
			"across the whole directory rather than picking the file whose name matches the question.")
	return b.String()
}

// isExplainOnlyMessage reports an explicit "tell me, do not touch it"
// instruction: an explain/describe request paired with a no-edit directive.
//
// Position-based negation cannot catch this. In "…whether it is actually a
// bug. Do not change the code." the intent word comes BEFORE the directive,
// so a backward scan finds nothing — yet the instruction plainly governs the
// whole message. Measured: that prompt classified T2, ran the V3 pipeline,
// and wrote to files the user had just asked it to leave alone.
//
// Both halves are required. "fix the bug but don't change the public API" has
// the directive and is still real work; without the explain half it stays
// action intent.
func isExplainOnlyMessage(lower string) bool {
	explain := false
	for _, w := range []string{"explain", "describe", "walk me through",
		"what does", "what is", "how does", "why does", "tell me"} {
		if strings.Contains(lower, w) {
			explain = true
			break
		}
	}
	if !explain {
		return false
	}
	for _, d := range []string{
		"do not change", "don't change", "dont change",
		"do not edit", "don't edit", "dont edit",
		"do not modify", "don't modify", "dont modify",
		"do not write", "don't write",
		"without changing", "without editing", "without modifying",
		"no code changes", "just explain", "only explain", "explain only",
	} {
		if strings.Contains(lower, d) {
			return true
		}
	}
	return false
}

func isFixIntentMessage(msg string) bool {
	lower := strings.ToLower(msg)
	if isExplainOnlyMessage(lower) {
		return false
	}
	for _, w := range fixIntentWords {
		idx := 0
		for {
			i := strings.Index(lower[idx:], w)
			if i < 0 {
				break
			}
			at := idx + i
			// Same negation rule as isActionIntentMessage: "explain whether
			// it is a bug, do not change the code" is a question about a
			// defect, not a request to repair one, and reading it as repair
			// intent handed it the write pipeline.
			if !negatedAt(lower, at) {
				return true
			}
			idx = at + len(w)
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
// phrasing (a merge-diff prompt: "the file algo.py must exist in the
// merged result"), which names a deliverable without a write verb.
// Checked in a window AFTER the filename.
var reMustProduce = regexp.MustCompile(`(?i)^\s*(must (exist|contain|include|be (creat|writt|present|generat)))`)

// expectedOutputPaths extracts the file(s) a task prompt explicitly asks
// the model to produce: a filename token preceded within ~70 chars by a
// write/save/create/output verb. Grounded in the task text (many bench and
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
	if isExplainOnlyMessage(lower) {
		return false
	}
	for _, w := range actionIntentWords {
		idx := 0
		for {
			i := strings.Index(lower[idx:], w)
			if i < 0 {
				break
			}
			at := idx + i
			if !negatedAt(lower, at) {
				return true
			}
			idx = at + len(w)
		}
	}
	return false
}

// negatedAt reports whether the action word at `at` is inside a negation —
// "do not change any code", "without editing", "no need to fix it".
//
// A plain substring scan reads "do not change any code" as a request to
// change code, so a question carrying that clause was classified T2 and got
// the whole write pipeline. Measured: the identical question scored T0
// without the clause and T2 with it, and the T2 run edited files the user had
// explicitly asked it to leave alone. Telling ATLAS not to touch anything
// made it more likely to.
//
// Scans a short window back rather than parsing: the negation always sits
// within a few words in the phrasings people actually use, and a wider window
// would start swallowing unrelated clauses ("I fixed the parser, now don't
// worry about X" must still read as action intent).
func negatedAt(lower string, at int) bool {
	const window = 24
	start := at - window
	if start < 0 {
		start = 0
	}
	before := lower[start:at]
	for _, neg := range []string{
		"do not ", "don't ", "dont ", "never ", "without ",
		"no need to ", "rather than ", "instead of ", "avoid ",
	} {
		if strings.Contains(before, neg) {
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
	c := strings.TrimSpace(cmd)
	if !verificationCommandRe.MatchString(c) {
		return false
	}
	// A probe that never retrieves a body proves the SERVER is up, not that
	// the artifact works. Measured on a "build me a snake game" session:
	// `curl -I http://localhost:8000` was recorded as the verification, the
	// gate opened, and `done` shipped an index.html containing JavaScript
	// and no HTML at all. A static file server answers 200 for a directory
	// listing, so a HEAD request cannot distinguish a working page from a
	// broken one.
	if headOnlyProbeRe.MatchString(c) {
		return false
	}
	return true
}

// headOnlyProbeRe matches curl/wget invocations that fetch headers only:
// `curl -I`, `curl --head`, `wget --spider`. Long-form and clustered short
// flags (-sI) both count.
var headOnlyProbeRe = regexp.MustCompile(`(?i)\b(?:curl\b[^|;&]*?\s-{1,2}(?:I\b|head\b)|curl\b[^|;&]*?\s-[a-zA-Z]*I[a-zA-Z]*\b|wget\b[^|;&]*?--spider\b)`)

// actionDemandSource names why an action-demand decision came out as it did.
// It exists so evidence can say which authority answered, not merely what the
// answer was.
type actionDemandSource string

const (
	actionDemandContractWork     actionDemandSource = "contract_work"
	actionDemandContractQuestion actionDemandSource = "contract_question"
	actionDemandLegacy           actionDemandSource = "legacy"
	// contract_invalid_failed_closed cannot be produced by a validated
	// request; it exists so an internally malformed mode fails toward
	// requiring work rather than silently reading as a question.
	actionDemandContractInvalid actionDemandSource = "contract_invalid_failed_closed"
)

// actionDemand is one decision about whether a request demands a state change,
// together with the authority that made it and the legacy heuristic's own
// answer for comparison.
type actionDemand struct {
	Required bool
	Source   actionDemandSource
	// Legacy is what wantsStateChange said. It is always reported and is
	// authoritative ONLY when no contract is present.
	Legacy bool
}

// decideActionDemand is the single owner of "does this request demand a state
// change". Both live action-demand sites consume it and nothing else calls the
// heuristic for a live decision.
//
// Where the client declared a task mode, that mode decides. Step 3B measured
// the alternative on a frozen 105-case corpus: 25 of 101 evaluable requests
// disagreed with the client's own declaration, 19 of them work that the wording
// alone read as a question, and 110 of 115 gate-bearing requests were decided
// with inspectedWorkspace false -- i.e. on phrasing. A client that states what
// it asked for is better evidence than a guess about its English.
//
// Where no contract was sent, nothing changes: the heuristic decides exactly as
// before. That corpus says nothing about contractless clients, so it authorises
// nothing for them.
//
// The mode establishes an OBLIGATION only. It never authorises completion,
// mutation, deletion, permission or verification -- those keep their own
// evidence, and a question contract does not erase debt, hazards or broken
// deliverables.
//
// Pure: no model output, no workspace state beyond the inspection flag it is
// handed, no shadow state, and the same answer whether or not capture is on.
func decideActionDemand(tc *TaskContract, userMessage string, tier Tier,
	inspectedWorkspace bool) actionDemand {
	// Evaluated exactly once, here, for every path. Reporting it costs
	// nothing because the heuristic is pure, and having it always present
	// keeps the shadow record identical whether or not capture is enabled.
	legacy := wantsStateChange(userMessage, tier, inspectedWorkspace)
	if tc == nil {
		return actionDemand{Required: legacy, Source: actionDemandLegacy, Legacy: legacy}
	}
	switch tc.TaskMode {
	case TaskModeWork:
		return actionDemand{Required: true, Source: actionDemandContractWork, Legacy: legacy}
	case TaskModeQuestion:
		return actionDemand{Required: false, Source: actionDemandContractQuestion, Legacy: legacy}
	default:
		return actionDemand{Required: true, Source: actionDemandContractInvalid, Legacy: legacy}
	}
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
	// A request to LOOK is satisfied by looking. inspectedWorkspace turns on
	// as soon as a read-only tool succeeds, so without this "list the files"
	// bounced its own `done`: the read that answered the question was the same
	// read the gate treated as evidence that work had started. Observed live —
	// the model listed the directory, was refused `done`, and told the user it
	// was "unable to complete the 'done' state as per the system's
	// requirements", a rule nothing in the prompt states.
	if isReadOnlyRequest(userMessage) {
		return false
	}
	return inspectedWorkspace && tier != Tier0Conversational
}

// isReadOnlyRequest matches asks that are ANSWERED by reading: list, show,
// find, print. Deliberately narrow — it only decides whether a completed read
// is allowed to end the turn, so a false positive lets a real edit request
// finish without editing.
//
// Callers check isActionIntentMessage first, which covers "add a list to the
// page". The fix-intent exclusion below covers the other overlap: "find and
// fix the bug" opens with a read verb but is not read-only work.
func isReadOnlyRequest(msg string) bool {
	lower := strings.ToLower(msg)
	if isFixIntentMessage(msg) || isActionIntentMessage(msg) {
		return false
	}
	for _, w := range []string{
		"list the file", "list files", "list all", "list the director",
		"show me", "show the", "what files", "which files", "what's in",
		"whats in", "what is in", "find the file", "where is", "where are",
		"print the", "display the", "read the",
	} {
		if strings.Contains(lower, w) {
			return true
		}
	}
	return false
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
// blockedServerStart reports whether a failed verification command failed
// because it is a long-running process rather than because the code is
// broken: it never exited (the sandbox timeout fired) or it could not bind
// because something is already serving that port.
//
// The distinction decides what the verification gate says next. Treating it
// as a red test tells the model to fix its code and re-run the command, and
// re-running a blocking server start can never exit clean — an observed
// session started the server correctly with run_background, was told to
// "re-run the same command and confirm it exits clean", and spent its three
// remaining bounces re-sending `done` because nothing it could do satisfied
// that.
func blockedServerStart(output string) bool {
	low := strings.ToLower(output)
	return strings.Contains(low, "execution timed out") ||
		strings.Contains(low, "address already in use") ||
		strings.Contains(low, "is in use by another program")
}

func verificationRejectionMessage(sawFailedVerification bool) string {
	return verificationRejection(sawFailedVerification, false, "")
}

// verificationRejection is verificationRejectionMessage with the two facts
// that change the advice: whether the red command was a blocking server
// start, and the job id if one is already running in the background.
// rewriteThreshold is how many consecutive red verifications exhaust the
// benefit of incremental edits. The no-tool retry baseline regenerates from
// scratch on every failure and scores 78% where incremental nibbling
// plateaued at 66-68: past this streak the advice flips from "apply the fix"
// to "rewrite the file from a clean sheet".
const rewriteThreshold = 2

// executorNames are commands that run a file handed to them as an argument.
// Wrappers like timeout/env don't qualify on their own — the real interpreter
// still has to appear between them and the file.
var executorNames = map[string]bool{
	"python": true, "python3": true, "python2": true, "py": true,
	"node": true, "nodejs": true, "deno": true, "bun": true,
	"ruby": true, "perl": true, "php": true, "lua": true,
	"bash": true, "sh": true, "zsh": true, "dash": true,
	"pytest": true, "go": true, "cargo": true, "java": true, "dotnet": true,
}

// executionAttempt reports whether the command actually attempts to RUN the
// file at path — an interpreter invocation (`python3 solve.py`, with or
// without wrapper/flag tokens in between) or direct execution (`./solve.py`).
// Merely naming the file proves nothing about its runtime behavior: `cat`,
// `grep`, `ls`, `wc` all name it and were all discharging the warned-run
// mark under the old substring rule (audit finding).
func executionAttempt(command, path string) bool {
	base := filepath.Base(path)
	for _, segment := range splitShellSegments(command) {
		toks := strings.Fields(segment)
		fileAt := -1
		for i, tok := range toks {
			tok = strings.Trim(tok, `"'`)
			if tok == path || tok == base || strings.HasSuffix(tok, "/"+base) {
				if strings.HasPrefix(tok, "./") {
					return true // direct execution
				}
				fileAt = i
				break
			}
		}
		if fileAt <= 0 {
			continue // absent, or the file itself is the first token without ./
		}
		for _, prev := range toks[:fileAt] {
			prev = strings.Trim(prev, `"'`)
			if executorNames[filepath.Base(prev)] {
				return true
			}
		}
	}
	return false
}

// freshRewriteAdvice is the start-over guidance shared by the done-gate
// rejection and the immediate mid-loop corrective. One source of truth so
// the model hears the same instruction at the crossing and at the gate.
func freshRewriteAdvice(redStreak int) string {
	return fmt.Sprintf(
		"the verification command has now failed %d times in a row and your incremental edits are not converging. Stop patching. Rewrite the file from scratch with write_file: re-read the task statement, take a fresh approach, and keep it simple. A clean rewrite finds bugs that ten small edits walk past.",
		redStreak)
}

// verificationRejectionWithStreak is verificationRejection plus the red-run
// streak that decides between edit-the-fix and start-over advice.
func verificationRejectionWithStreak(sawFailedVerification, serverBlocked bool, bgJobID string, redStreak int) string {
	if !serverBlocked && sawFailedVerification && redStreak > rewriteThreshold {
		return "Cannot declare `done` — " + freshRewriteAdvice(redStreak)
	}
	return verificationRejection(sawFailedVerification, serverBlocked, bgJobID)
}

func verificationRejection(sawFailedVerification, serverBlocked bool, bgJobID string) string {
	if serverBlocked {
		probe := "Start it with `run_background` (it returns a job_id), then probe it with " +
			"`run_command(\"curl http://localhost:<port>/\")`."
		if bgJobID != "" {
			probe = "It is ALREADY running as background job " + bgJobID +
				" — do not start another copy. Probe it now with " +
				"`run_command(\"curl http://localhost:<port>/\")`, which is the command that verifies it."
		}
		return "Cannot declare `done` yet — nothing has verified this change. The command that " +
			"failed is a long-running server: it did not exit because servers do not exit, so " +
			"re-running it in the foreground can never succeed and its failure says nothing about " +
			"your code. " + probe + " A clean curl is the verification this gate wants."
	}
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
// looks like a placeholder/stub. The model's lazy-completion
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
// session. Forces the "model from existing patterns" reflex
// instead of generating from scratch — a NEW route handler should
// match the project's existing route handlers, a new test should match
// the existing test conventions, etc.
//
// Only fires when:
//   - The target path doesn't exist (genuinely new file, not an edit)
//   - The parent directory contains ≥1 sibling with the same extension
//   - ctx.FilesRead doesn't include any of those siblings
//
// Soft-coupled to AgentContext via the FilesRead snapshot we pass in
// (ctx.SnapshotFilesRead() at the call site); keeps the helper testable
// without dragging the whole context type in.
func patternMatchHint(resolvedPath string, filesRead map[string]string) string {
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
	for _, s := range siblings {
		if _, ok := filesRead[filepath.Join(dir, s)]; ok {
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

// looksCorruptedOnDisk returns true when the file at displayPath has
// the markdown-fence-with-prose corruption pattern that
// sanitizeFileContent strips on input.
//
// The corruption shape is what `<model> generated` left behind in
// May 2026 templates: prose preamble ("Looking at the task, I need
// to create..."), then a ```html fence, then real HTML, then a
// closing fence with trailing commentary. Once on disk, this file
// is unparseable to Jinja/the browser, but the surgical-edit
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

// findFileArgSwapHint catches find_file called with the filename in `path`
// and `pattern` empty, and returns the corrected call.
//
// Observed live: the model sent {"path":"app.py"}, got "pattern cannot be
// empty", and sent {"path":".*app\\.py.*"} — moving the regex into `path`
// rather than into `pattern`, because the error named the field it had left
// blank and not the one it had filled. A `path` that carries an extension and
// no separator is a filename, so say which argument it belongs in.
func findFileArgSwapHint(path string) string {
	p := strings.TrimSpace(path)
	if p == "" || strings.ContainsAny(p, `/\`) {
		return "" // a real directory, or nothing to go on
	}
	ext := filepath.Ext(p)
	if ext == "" || len(ext) > 6 {
		return ""
	}
	// Escape it into the regex the caller meant.
	quoted := regexp.QuoteMeta(p)
	return fmt.Sprintf(
		"find_file: `pattern` is empty and `path` is %q, which is a filename, not a directory. "+
			"`path` is WHERE to search (a directory, default the project root) and `pattern` is a "+
			"regex matching the FILENAME. Retry as: find_file {\"pattern\":%q}",
		p, quoted+"$")
}

// foreignRunes reports characters in old_str that appear NOWHERE in the file,
// restricted to non-ASCII. A mismatch caused by one corrupted character is
// invisible in a diff of two long strings, and every other hint sends the model
// to re-copy text it already copied correctly.
//
// Observed live: the model sent `if(headX === food.x ℘ headY ===food.y)` for a
// line reading `&&`. U+2118 SCRIPT CAPITAL P is not a typo, it is a decode
// artefact, and the rejection it got ("your old_str is 9 lines long") was true
// and useless — the length was not why it failed.
//
// ASCII is excluded deliberately: a wrong ASCII character is an ordinary
// mis-copy the closest-line hint already handles. A non-ASCII rune that the
// file does not contain anywhere is nearly always corruption.
func foreignRunes(oldStr, fileContent string) []rune {
	var out []rune
	seen := map[rune]bool{}
	for _, r := range oldStr {
		if r < 0x80 || seen[r] {
			continue
		}
		seen[r] = true
		if !strings.ContainsRune(fileContent, r) {
			out = append(out, r)
		}
	}
	return out
}

// describeForeignRunes renders foreignRunes for the model: the character, its
// codepoint, and where it sits, so the fix is mechanical rather than a re-copy.
func describeForeignRunes(runes []rune) string {
	parts := make([]string, 0, len(runes))
	for _, r := range runes {
		parts = append(parts, fmt.Sprintf("%q (U+%04X)", r, r))
	}
	return strings.Join(parts, ", ")
}

// unverifiedSummary replaces a completion claim the run's own evidence cannot
// support.
//
// The verification gate bounces a `done` three times and then, out of
// bounces, lets it through — so the model's summary reaches the user
// unchanged. Observed 2026-08-02 across three runs: "I updated the snake game
// logic... I also verified that the page loads", over a file whose only
// @app.route had been deleted. The gate had done its job and the claim
// shipped anyway.
//
// Rewriting the summary is mechanical and needs nothing from the model, which
// is why it is done here rather than by asking the model to be more careful.
// Making `done` ungrammatical would be stronger, but that needs strict
// schema-GBNF and Gemma-family models require the loose grammar (a strict
// schema makes them emit `done` instead of calling tools at all).
//
// The model's own words are kept, labelled, because they usually do describe
// the intended change accurately — it is the verification claim inside them
// that is unsupported.
func unverifiedSummary(wrote bool, claim string) string {
	var sb strings.Builder
	if wrote {
		sb.WriteString("Changes were written to disk, but NOTHING in this run verified them — " +
			"no build, test, or probe command completed successfully. Run it yourself before " +
			"relying on it.")
	} else {
		sb.WriteString("Nothing was written to disk in this run, and no verification command " +
			"completed successfully.")
	}
	if c := strings.TrimSpace(claim); c != "" {
		sb.WriteString("\n\nThe agent's own account, which is UNVERIFIED and may describe work " +
			"that did not land:\n")
		sb.WriteString(truncateStr(c, 1200))
	}
	return sb.String()
}

// planIncompleteMessage names the plan steps that never got a matching tool
// call, or "" to let `done` through.
//
// The plan is generated up front, PlanStepsSatisfied tracks which steps have
// been hit, and buildPlanReminder shows the model its progress every turn —
// but nothing checked it at the exit. Run 4 built the variable-delay loop it
// was asked for, never added the per-food decrement, and emitted `done`: two
// required edits, one delivered. The reminder is an instruction and was
// ignored; this is the same fact used as a gate.
//
// Gated on plan quality, because a bad plan blocking a finished task is worse
// than no plan at all. A low-scoring plan, or one whose steps the matcher
// could not track, is not evidence of anything and is skipped.
func planIncompleteMessage(ctx *AgentContext) string {
	if ctx == nil || ctx.Plan == nil || len(ctx.Plan.Steps) < 2 {
		return ""
	}
	if ctx.Plan.WinningScore < planGateMinScore {
		return ""
	}
	if len(ctx.PlanStepsSatisfied) != len(ctx.Plan.Steps) {
		return ""
	}
	// Nothing matched at all means the matcher is not tracking this task, not
	// that the model did nothing.
	if countTrue(ctx.PlanStepsSatisfied) == 0 {
		return ""
	}
	var missing []string
	for i, step := range ctx.Plan.Steps {
		if !ctx.PlanStepsSatisfied[i] {
			missing = append(missing, fmt.Sprintf("  %s: %s", step.ID, step.Action))
		}
	}
	if len(missing) == 0 {
		return ""
	}
	return fmt.Sprintf(
		"Cannot declare `done` yet — %d of %d planned steps have landed, and these have not:\n%s\n"+
			"Each one needs its own tool call. Do the next one now. If a step is genuinely "+
			"unnecessary or already satisfied by an edit you made under a different step, say "+
			"which in your next `done` summary rather than leaving it silent.",
		countTrue(ctx.PlanStepsSatisfied), len(ctx.Plan.Steps), strings.Join(missing, "\n"))
}

// reForegroundServer matches commands that serve until killed. Deliberately
// narrow: only forms that cannot be anything else. `python app.py` is
// excluded because it is just as likely to be a script that exits, and
// refusing it would block a legitimate verification.
var reForegroundServer = regexp.MustCompile(`(?i)(^|\s|&&|;)\s*(` +
	`python3?\s+-m\s+http\.server` +
	`|php\s+-S\b` +
	`|(python3?\s+-m\s+)?(uvicorn|gunicorn|waitress-serve)\b` +
	`|flask\s+run\b` +
	`|(npm|yarn|pnpm)\s+(start|run\s+(dev|serve|start|preview))\b` +
	`|(npx\s+)?(vite|next\s+dev|http-server|serve)\b` +
	`|rails\s+s(erver)?\b` +
	`|jekyll\s+serve\b` +
	`)`)

// foregroundServerRejection redirects a server start from run_command to
// run_background before it executes.
//
// Observed on the first-contact path — an empty workspace, "create a simple
// portfolio website": the model wrote three files, then ran
// `python3 -m http.server 8000` in the foreground, waited out the full 30s
// sandbox timeout, and only then reached for run_background. That is 30
// seconds of a 3m39s run, every time, on the most common way anyone will
// first try ATLAS.
//
// The guidance already says to use run_background for servers and the model
// still does this, which is the usual result for an instruction. The harness
// can see the command before it runs, so it stops being a suggestion.
// A script the user wrote is only recognisable as a server from its
// contents. These calls do not return until the process is killed, so
// finding one in the file the command runs is evidence rather than a guess
// from the command text — `python app.py` and `python solve.py` are
// indistinguishable otherwise, and refusing on the name would block
// ordinary verification.
var serverLoopMarkers = []string{
	"app.run(", ".serve_forever(", "uvicorn.run(", "socketserver.",
	"httpd.serve", "app.listen(", "server.listen(", "web.run_app(",
}

// reServerScript pulls the workspace file out of an interpreter invocation:
// `python app.py`, `python3 -u srv/app.py`, `node server.js`.
var reServerScript = regexp.MustCompile(
	`(?i)\b(?:python3?|node|ruby|php)\b[^|;&]*?\s([\w./-]+\.(?:py|js|mjs|rb|php))\b`)

// runsAServerLoop reports whether the file this command executes blocks
// forever. readFile returns workspace contents; a miss says no, since
// refusing on a filename alone is the guess this avoids.
func runsAServerLoop(cmd string, readFile func(string) (string, bool)) bool {
	if readFile == nil {
		return false
	}
	m := reServerScript.FindStringSubmatch(cmd)
	if m == nil {
		return false
	}
	src, ok := readFile(m[1])
	if !ok {
		return false
	}
	low := strings.ToLower(src)
	for _, marker := range serverLoopMarkers {
		if strings.Contains(low, marker) {
			return true
		}
	}
	return false
}

func foregroundServerRejection(cmd string) string {
	return foregroundServerRejectionWithSource(cmd, nil)
}

// foregroundServerRejectionWithSource is foregroundServerRejection plus the
// workspace reader, so a script that serves is caught alongside the
// launchers that always do.
//
// The pattern list covers the well-known launchers (`http.server`,
// `npm run dev`, `flask run`). It cannot cover a file the user wrote:
// measured 2026-08-04 on flask_pause, `python app.py` sat through the full
// 30s sandbox timeout before the verification gate said anything, and that
// is the shape a real project takes.
func foregroundServerRejectionWithSource(cmd string,
	readFile func(string) (string, bool)) string {
	// A trailing & already detaches, so the call returns immediately and
	// there is nothing to redirect. (`nohup` alone does not: it only
	// ignores SIGHUP, and the command still holds the foreground.)
	if strings.HasSuffix(strings.TrimSpace(cmd), "&") {
		return ""
	}
	if !reForegroundServer.MatchString(cmd) && !runsAServerLoop(cmd, readFile) {
		return ""
	}
	return fmt.Sprintf(
		"`%s` serves until it is killed, so run_command would sit on it until the timeout "+
			"and report a failure that says nothing about your code. Start it with "+
			"run_background instead — it returns a job_id immediately:\n"+
			"  run_background {\"command\": %q}\n"+
			"Then probe it with run_command (`curl -I http://localhost:<port>/`), and "+
			"stop_background when you are done.",
		truncateStr(cmd, 80), cmd)
}

// outOfTurnsSummary is what the user reads when the loop hits its turn cap.
//
// The cap used to end the turn with an `error` event and nothing else, so a
// question whose recon ran long came back blank. A blank reply is the worst
// outcome the harness can produce: the user cannot tell whether ATLAS is
// broken, still thinking, or ignoring them. Say what happened, say what was
// learned, and name the next move.
func outOfTurnsSummary(ctx *AgentContext, wrote bool) string {
	var sb strings.Builder
	sb.WriteString("I ran out of turns for this request before finishing.")
	if wrote {
		sb.WriteString(" Changes were written to disk — check them before relying on them.")
	} else {
		sb.WriteString(" Nothing was written to disk.")
	}
	if files := ctx.SnapshotFilesRead(); len(files) > 0 {
		names := make([]string, 0, len(files))
		for p := range files {
			if rel, err := filepath.Rel(ctx.WorkingDir, p); err == nil && rel != "" {
				names = append(names, rel)
			} else {
				names = append(names, filepath.Base(p))
			}
		}
		sort.Strings(names)
		fmt.Fprintf(&sb, "\n\nI did get to look at: %s.", strings.Join(names, ", "))
	}
	sb.WriteString("\n\nAsk again and point me at the specific file or function you care " +
		"about — a narrower request finishes inside the budget.")
	return sb.String()
}

// repeatedRefusalSummary ends a run that kept re-sending one rejected call.
//
// The user needs to know the tool call was refused rather than attempted, and
// that re-running the same prompt will do the same thing — otherwise the
// obvious response is to try again verbatim.
func repeatedRefusalSummary(tool, path string, wrote bool) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Stopped: the same `%s` call was re-sent after being refused, and kept being "+
		"re-sent without changing.", tool)
	if path != "" {
		fmt.Fprintf(&sb, " Target: %s.", path)
	}
	if wrote {
		sb.WriteString(" Earlier changes in this run did land on disk — check them.")
	} else {
		sb.WriteString(" Nothing was written to disk.")
	}
	sb.WriteString("\n\nThe refusal reason is in the per-turn errors above. Re-running this " +
		"prompt unchanged will hit the same wall; say specifically what to change and where, " +
		"or point at a different file.")
	return sb.String()
}

// repeatTerminalSummary is the terminal for the repeat detector, and the one
// place that decides whether a run may say its change landed.
//
// `wrote` (madeProductiveChange) is a PROGRESS HINT. It records that some
// mutation succeeded earlier in the run; it says nothing about the bytes on
// disk now, and it may never authorize a completion claim. Measured on the
// seed-20260901 confirmation, task debounce5: an accepted write set the hint,
// the model then repeated a failing verification, and the terminal read "Made
// your change ... the change is on disk" over a file containing a
// SyntaxError. One false success in 50 sessions, and the only terminal in
// that run which misreported its own outcome.
//
// Completion is therefore derived from a FRESH observation of the declared
// deliverable's CURRENT bytes, through the same syntax contract the write
// path uses. Existence is not validity, an earlier success is not this
// content, and anything short of an explicit pass -- not_run, not_applicable,
// unknown, unreadable, or nothing declared -- is undemonstrated and stops.
// recovered carries the per-path outcomes of Phase 3B restoration, which the
// caller performs before composing this summary. It is disclosed as its own
// clause: recovery changes what is on disk, never whether the run finished.
func repeatTerminalSummary(ctx *AgentContext, expected []string, wrote bool,
	recovered []restoreDecision) string {
	var sb strings.Builder
	sb.WriteString("Stopped: the same tool call kept repeating without making progress")
	// Validation status alters DISCLOSURE only. A repeat-breaker is an
	// operational failure whatever the bytes look like: syntax is not task
	// completion, and the run stopped without finishing its verification.
	// Neither branch may read as a completion claim.
	switch {
	case !wrote:
		sb.WriteString(", and nothing was written to disk")
	case deliverablesDemonstrablyValid(ctx, expected):
		sb.WriteString(". Your work is on disk and parses, but the " +
			"verification did not complete, so this run cannot say the task is done")
	default:
		sb.WriteString(". Earlier changes did land on disk, but the current " +
			"contents were not shown to be valid — treat them as unverified")
	}
	sb.WriteString(". Try a more specific instruction (e.g. name the file and " +
		"the exact change).")
	// Appended last, and never in place of the stop: a restored file is a
	// safer starting point, not a completed task.
	sb.WriteString(restorationDisclosure(recovered))
	return sb.String()
}

// deliverablesDemonstrablyValid answers the completion question and nothing
// else: does every declared deliverable, as it exists RIGHT NOW, pass the
// syntax contract? With nothing declared there is nothing demonstrated, so
// the answer is no.
func deliverablesDemonstrablyValid(ctx *AgentContext, expected []string) bool {
	if len(expected) == 0 {
		return false
	}
	for _, rel := range expected {
		resolved := resolveAgentPath(ctx, rel)
		content, err := os.ReadFile(resolved)
		if err != nil {
			return false
		}
		status := fallbackSyntaxOutcomeFor(ctx, resolved, string(content)).WholeFile.Status
		if status == ValidationPassed {
			continue
		}
		// A document has no syntax to pass. Requiring one meant a valid
		// notes.txt could never demonstrate anything, however current its
		// bytes were. What stands in for the pass is existence plus currency:
		// the checker reports that nothing applies, the path is prose rather
		// than unsupported code, and the ledger's record still describes the
		// bytes that are there. This is not a syntax pass and is never
		// relabelled as one.
		if status == ValidationNotApplicable && documentDeliverableCurrent(ctx, resolved, content) {
			continue
		}
		return false
	}
	return true
}

// documentDeliverableCurrent is the existence-and-currency evidence a non-code
// deliverable can offer in place of a syntax pass.
//
// Every clause is required. Prose only, so an unsupported language or an
// unknown extension holding logic is excluded; the ledger must already own the
// path, so nothing the session never wrote qualifies; its recorded verdict
// must be exactly "checked nothing because nothing applies"; and that record
// has to describe the bytes on disk right now.
func documentDeliverableCurrent(ctx *AgentContext, resolved string, content []byte) bool {
	if !isDocumentAsset(resolved) {
		return false
	}
	key := ledgerKey(ctx, resolved)
	ctx.LedgerMu.Lock()
	d := ctx.Ledger[key]
	var kind ValidationKind
	var status ValidationStatus
	var tombstoned bool
	var current string
	if d != nil {
		kind, status = d.CurrentValidation()
		tombstoned, current = d.Tombstoned, d.CurrentHash
	}
	ctx.LedgerMu.Unlock()
	if d == nil || tombstoned {
		return false
	}
	if kind != ValidationKindNone || status != ValidationNotApplicable {
		return false
	}
	return current == hashBytes(content)
}

// inferenceFailureSummary is what the user reads when the model call itself
// fails — the stream cannot continue, so this is the last thing they get.
//
// The context-size 400 is called out by name because it is actionable and
// because it was the deterministic killer: `aoc_sonar` hit it in both reps at
// turn 3, and the run ended with an `error` event and no outcome at all.
func inferenceFailureSummary(err error, wrote bool) string {
	msg := ""
	if err != nil {
		msg = err.Error()
	}
	var sb strings.Builder
	if strings.Contains(msg, "exceed_context_size") || strings.Contains(msg, "exceeds the available context size") {
		sb.WriteString("Stopped: the conversation outgrew the model's context window, so the " +
			"request was refused before it ran. This usually means a large file was read " +
			"into the session. Start a fresh request naming just the file and change you " +
			"want, or raise the server's context size.")
	} else {
		sb.WriteString("Stopped: the model call failed, so the run could not continue.")
		if msg != "" {
			fmt.Fprintf(&sb, "\n\n%s", truncateStr(msg, 300))
		}
	}
	if wrote {
		sb.WriteString("\n\nChanges made earlier in this run are on disk — check them before re-running.")
	} else {
		sb.WriteString("\n\nNothing was written to disk.")
	}
	return sb.String()
}

// nothingWrittenSummary states that a run which was asked to change
// something finished without changing anything.
//
// The action gate bounces this while it can, then stops — its bounces are
// capped so an exhausted gate cannot loop. Past the cap the exit is
// unremarked, and the worst version is a summary that looks like success:
// observed on smallrung_toml, a refused structural_edit, the model giving
// up on tools and emitting the replacement as chat text, and that code
// arriving as the run's summary. A user reading it has no way to tell the
// change was never applied.
//
// Prefixed rather than replacing: whatever the model said may still be
// useful, it just cannot stand as a completion claim.
func nothingWrittenSummary(original string) string {
	const lead = "Nothing was written — no file was created or changed in this run. " +
		"Any code below is a proposal, not something on disk."
	if strings.TrimSpace(original) == "" {
		return lead
	}
	return lead + "\n\n" + original
}

// verifiedPhase reports whether a V3 phase_solved value means a candidate
// actually passed verification.
//
// The field is initialised to "none" and only overwritten when something
// passes, so "not empty" is not the same question as "verified" — "none" is
// a perfectly good non-empty string. Listing the phases that mean success is
// the check that cannot drift: a new phase added upstream is unverified here
// until it is named, which is the safe direction.
func verifiedPhase(phase string) bool {
	switch phase {
	case "probe", "phase1", "pr_cot", "refinement", "budget":
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// Literal-content contracts: the model plans, the harness copies.
//
// A quantized model cannot be trusted to transcribe bytes it was given —
// measured live and deterministic: told to write exactly `BANNER = "ready"`,
// it emits `BANNER = " ready"` under greedy AND default sampling, because the
// space-prefixed BPE token for the word outranks the bare one after a quote
// (the leading-whitespace artifact; arXiv:2502.14969). The literature's
// remedy for the whole class is to treat the LLM as a planner and use a
// deterministic channel for exact emission (arXiv:2601.03640, 2604.18170).
//
// Here that channel is the user's own message: when the request carries the
// intended bytes explicitly, they are recorded as contracts, and a landed
// write is verified against them. A near-miss whose only divergence is
// whitespace is repaired mechanically — the user's bytes are definitionally
// the correct rendering, so substituting them cannot be wrong. Anything
// beyond whitespace is left alone: a bolder repair could mask a legitimate
// transformation the model was asked to make.

// literalMinBytes is the smallest contract worth tracking. Below this,
// prose fragments ("x", "42") would false-positive all over the artifact.
const literalMinBytes = 8

// literalExactlyRe captures the single line following an "exactly ...:"
// marker: `containing exactly one line:\nBANNER = "ready"`. Deliberately
// only ONE line — multi-line literals in prose are what fenced blocks are
// for, and guessing where a prose literal ends is how false positives start.
var literalExactlyRe = regexp.MustCompile(`(?i)(?:exactly|verbatim|precisely)[^:\n]*:[ \t]*\n([^\n]+)`)

// literalFenceRe captures every fenced block in the task text.
var literalFenceRe = regexp.MustCompile("(?s)```(?:[a-zA-Z0-9+#._-]+)?[ \\t]*\\r?\\n(.*?)```")

// extractLiteralBlocks pulls the byte-exact content contracts out of a human
// request. Only explicit forms count: fenced blocks, and the single line
// after an "exactly:"-style marker.
func extractLiteralBlocks(task string) []string {
	var out []string
	seen := map[string]bool{}
	add := func(s string) {
		s = strings.Trim(s, "\r\n")
		if len(strings.TrimSpace(s)) < literalMinBytes || seen[s] {
			return
		}
		seen[s] = true
		out = append(out, s)
	}
	for _, m := range literalFenceRe.FindAllStringSubmatch(task, -1) {
		add(m[1])
	}
	for _, m := range literalExactlyRe.FindAllStringSubmatch(task, -1) {
		add(m[1])
	}
	return out
}

// stripAllWhitespace is the near-miss equivalence: two renderings whose
// non-whitespace bytes agree differ only in spacing, and the user's literal
// is by definition the correct spacing.
func stripAllWhitespace(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r != ' ' && r != '\t' && r != '\r' && r != '\n' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// repairLiteralDrift returns content with every absent literal whose
// whitespace-insensitive rendering IS present replaced by the literal's
// exact bytes. The bool reports whether anything changed.
func repairLiteralDrift(content string, literals []string) (string, []string, bool) {
	var repaired []string
	for _, lit := range literals {
		if strings.Contains(content, lit) {
			continue // contract already satisfied byte-exact
		}
		litLines := strings.Split(lit, "\n")
		litKey := stripAllWhitespace(lit)
		lines := strings.Split(content, "\n")
		for i := 0; i+len(litLines) <= len(lines); i++ {
			window := strings.Join(lines[i:i+len(litLines)], "\n")
			if stripAllWhitespace(window) == litKey {
				lines = append(lines[:i], append(litLines, lines[i+len(litLines):]...)...)
				content = strings.Join(lines, "\n")
				repaired = append(repaired, lit)
				break
			}
		}
	}
	return content, repaired, len(repaired) > 0
}

// toolBanNote tells the model a tool is gone for a file and names what is
// left. Written as a fact rather than a suggestion, because the suggestion
// form was measured to be ignored: an explicit "re-sending will not help,
// use structural_edit" was followed by the identical call on the next turn.
func toolBanNote(tool, path string) string {
	alt := "`replace_lines` (assert only the first and last line of the range) or `write_file` with the complete new contents"
	if ext := strings.ToLower(filepath.Ext(path)); ext == ".py" || ext == ".html" || ext == ".htm" {
		alt = "`structural_edit` (a selector such as `function:update` plus the new body — no old_str to reproduce) or `write_file` with the complete new contents"
	}
	return fmt.Sprintf(
		"%s is no longer available for %s in this session: it was sent and rejected unchanged, so it is not a path to a working edit here. Use %s.",
		tool, path, alt)
}
