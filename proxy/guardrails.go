// Guardrails for the agent loop. Centralises the checks that bounce
// model output before it touches disk or the host filesystem.
//
// Why a separate file: the rules accumulate (output sanitisation,
// shell-op blocking, protected paths) and live downstream of multiple
// tool handlers. Keeping them together makes the policy auditable —
// reviewers don't have to chase three call sites to know what we
// reject.
//
// Background: ATLAS runs against a local qwen-coder model that's
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
//   Looking at the task, I need to create a complete index.html...
//
//   ```html
//   <!DOCTYPE html>
//   ...
//   ```
//
//   This file does X, Y, Z.
//
// Without this strip, the whole markdown wrapper lands on disk
// verbatim — Jinja chokes on `{{ url_for(...) }}` fragments inside a
// numbered-list explanation, the user sees a 500, debugging starts.
//
// The function returns (cleaned, modified). modified=true means a
// fence/prose was stripped — the caller should log it so we can spot
// repeat offenders. .md / .markdown / .rst files are passed through
// unchanged because fences are legitimate content there.
func sanitizeFileContent(filePath, content string) (string, bool) {
	ext := strings.ToLower(filepath.Ext(filePath))
	switch ext {
	case ".md", ".markdown", ".rst", ".txt":
		return content, false
	}

	lines := strings.Split(content, "\n")

	openIdx := -1
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			openIdx = i
			break
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

// shellDestructiveRe matches the leading token of a destructive
// filesystem command. We split on `&&`, `||`, `;`, `|` first, so each
// segment can be checked in isolation.
var shellDestructiveRe = regexp.MustCompile(
	`^\s*(rm|mv|cp|rmdir|chmod|chown|truncate)(\s+|$)`)

// shellFindDeleteRe catches `find ... -delete` and `find ... -exec rm`.
// These bypass the leading-token check above because the destructive
// verb is buried deep in the command line.
var shellFindDeleteRe = regexp.MustCompile(
	`\bfind\b.*?(-delete\b|-exec\s+rm\b)`)

// shellTruncatingRedirectRe catches `> path` (overwrite) but excludes:
//   - `>>`            append (handled by [^>] predecessor)
//   - `2>`, `1>`      stderr/stdout fd redirect (handled by [^0-9])
//   - `>&1`, `>&2`    fd dup (handled by [^&] in dest leading char)
//   - `> /dev/null`   discard, allowed downstream
//
// The destination's leading char is split out so `&` (fd duplication)
// and `>` (would be `>>`, already excluded) are rejected without
// double-checking. Composite redirects like `2>&1` are entirely
// non-truncating and must not trip this — the previous regex did,
// breaking every legit `python app.py 2>&1` verification call.
var shellTruncatingRedirectRe = regexp.MustCompile(
	`(^|[^>0-9])>\s*([^>&\s][^>\s]*)`)

// validateShellCommand returns a non-empty rejection reason if the
// command would mutate user files via the shell. Build/test/lint
// commands (python, npm, go, cargo, pytest, make, ls, cat, grep…)
// are all fine — only the destructive filesystem verbs trigger this.
//
// Today's behaviour we're trying to prevent: agent loops responding
// to a "fix this" prompt by running `mv templates venv/templates`,
// `rm -rf old/`, or piping a heredoc with `>` over an existing source
// file. The native edit_file / write_file / delete_file tools are
// the supported mutation path — their content goes through V3 and
// the surgical-edit gate; shell mutation bypasses both.
func validateShellCommand(cmd string) string {
	stripped := strings.TrimSpace(cmd)
	if stripped == "" {
		return ""
	}

	// Split on shell separators. Quoted segments aren't perfectly
	// honoured but model-emitted commands rarely contain quoted ; or |.
	segments := splitShellSegments(stripped)
	for _, seg := range segments {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}
		// `cd ...` segments are pass-throughs, not destructive — but
		// they prepend to the working directory of subsequent
		// segments, which we already analyse independently.
		if strings.HasPrefix(seg, "cd ") || seg == "cd" {
			continue
		}
		if shellDestructiveRe.MatchString(seg) {
			verb := strings.Fields(seg)[0]
			return shellRejectionMessage(verb, "the leading verb is " + verb)
		}
		if shellFindDeleteRe.MatchString(seg) {
			return shellRejectionMessage("find -delete",
				"`find` with -delete or -exec rm")
		}
		if shellHiddenCommandRe.MatchString(seg) {
			return shellRejectionMessage("bash -c / sh -c / eval",
				"a shell wrapper (bash -c, sh -c, eval, …) — these hide arbitrary commands inside a quoted argument and bypass the per-segment safety check")
		}
		// Truncating redirect: `... > some/path`. We only reject when
		// the target isn't /dev/null and isn't an obvious build artefact
		// suffix (.log, .out) — those are usually intentional.
		//
		// Use FindStringSubmatch so we read just the destination path
		// (capture group 2), not "everything after the >". Trailing
		// flags like `2>&1 &` would otherwise glue onto the path and
		// break the suffix exception (see PC-189).
		if sm := shellTruncatingRedirectRe.FindStringSubmatch(seg); sm != nil {
			dest := sm[2]
			if dest == "/dev/null" || dest == "/dev/stderr" {
				continue
			}
			lowerDest := strings.ToLower(dest)
			if strings.HasSuffix(lowerDest, ".log") || strings.HasSuffix(lowerDest, ".out") {
				continue
			}
			return shellRejectionMessage("> redirect",
				"a truncating redirect into "+dest)
		}
	}
	return ""
}

// shellRejectionMessage formats a directive that points the model
// at the right native tool. The model's next turn sees this as the
// tool_result and (in practice) re-emits the operation as edit_file
// or delete_file.
func shellRejectionMessage(verb, detail string) string {
	return "run_command refused: " + detail + ". Modify files with the dedicated tools — `edit_file` (old_str/new_str) for content changes, `write_file` for brand-new files, `delete_file` for removal. Shell `" + verb + "` bypasses the surgical-edit gate, the V3 pipeline, and audit logging, and will be rejected."
}

// shellHiddenCommandRe catches `bash -c "..."` / `sh -c "..."` /
// `zsh -c "..."` / `dash -c "..."` / `eval ...`. These wrappers can
// hide arbitrary destructive commands inside a quoted argument that
// the leading-token check above can't see — Roo Code's bypass test
// case is `bash -c "rm -rf foo"`. We don't try to parse the inner
// command (that's a recursive-shell-parser rabbit hole); we reject
// the wrapper itself. Build/test commands that need shell features
// (pipes, redirects, env vars) work fine without `bash -c`.
var shellHiddenCommandRe = regexp.MustCompile(
	`^\s*(bash|sh|zsh|dash|ksh)\s+-c\b|^\s*eval(\s+|$)`)

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
		`mypy|ruff|pylint|tsc|eslint|gofmt|vet|` +
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

// verificationRejectionMessage tells the model exactly what's
// missing and what to run. We prefer concrete suggestions over
// abstract "verify your work" prompts — the model is more likely to
// pick a sensible command when given a category.
func verificationRejectionMessage(userMsg string) string {
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
