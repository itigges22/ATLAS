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
	"path/filepath"
	"regexp"
	"strings"
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

// shellTruncatingRedirectRe catches `> path` (overwrite) but not `>>`
// (append) and not `> /dev/null` (the model uses this to silence
// output, which is fine).
var shellTruncatingRedirectRe = regexp.MustCompile(
	`(^|[^>])>\s*(?:[^>\s]+)`)

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
		if m := shellTruncatingRedirectRe.FindStringIndex(seg); m != nil {
			tail := seg[m[0]:]
			tail = strings.TrimLeft(tail, " >")
			tail = strings.TrimSpace(tail)
			if tail == "/dev/null" || tail == "/dev/stderr" {
				continue
			}
			lowerTail := strings.ToLower(tail)
			if strings.HasSuffix(lowerTail, ".log") || strings.HasSuffix(lowerTail, ".out") {
				continue
			}
			return shellRejectionMessage("> redirect",
				"a truncating redirect into "+tail)
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
