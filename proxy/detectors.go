package main

import (
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// Tool-call repetition detector. Catches the structural-loop case the
// PC-207 lens scoring doesn't see: model calls the SAME (tool, args)
// pair multiple times in close succession (e.g. read_file('app.py')
// 4 times in 6 turns, or run_command('curl localhost:5000/...') three
// times after the server already returned the same error each time).
//
// This is complementary to the lens-as-PRM intervention in agent.go:
// lens scores GENERATED CONTENT semantically; this detector scores
// CALL SHAPES structurally. Together they cover most stuck patterns:
//   - lens catches "model produced low-quality content" (stub writes)
//   - this catches "model emitted the same tool call again" (read loops)

const (
	// toolRepeatWindow is the number of recent tool calls to remember.
	// 8 is enough to span a typical recon → action → verify → recon
	// pattern (4-6 turns) plus margin for re-tries, while staying small
	// enough that a long-ago repeated call doesn't keep firing
	// interventions on a different topic.
	toolRepeatWindow = 8

	// toolRepeatThreshold is the number of times the same call signature
	// must appear in the window before we intervene. 3 is the minimum
	// that's clearly a pattern (1 = normal, 2 = retry); 4+ would miss
	// the kind of stub-loop case where the model only got 3 attempts in
	// before something else broke the chain.
	toolRepeatThreshold = 3
)

// recordToolCall pushes a (tool_name, args) signature into ctx's
// rolling window and returns the corrective message + true when the
// same signature has appeared toolRepeatThreshold times within the
// last toolRepeatWindow entries. Returns ("", false) otherwise.
//
// Caller is responsible for resetting ctx.RecentToolCalls after acting
// on the corrective so we don't re-fire on the same crash on the next
// iteration.
func recordToolCall(ctx *AgentContext, toolName string, args json.RawMessage) (string, bool) {
	sig := toolCallSignature(toolName, args)
	ctx.RecentToolCalls = append(ctx.RecentToolCalls, sig)
	if len(ctx.RecentToolCalls) > toolRepeatWindow {
		ctx.RecentToolCalls = ctx.RecentToolCalls[len(ctx.RecentToolCalls)-toolRepeatWindow:]
	}

	count := 0
	for _, s := range ctx.RecentToolCalls {
		if s == sig {
			count++
		}
	}
	if count < toolRepeatThreshold {
		return "", false
	}
	if toolName == "write_file" {
		if p := writeFilePath(args); p != "" {
			return fmt.Sprintf(
				"⚠ You have fully rewritten `%s` %d times in the last %d tool calls. Each write_file replaces the "+
					"whole file, and the on-disk version is the verified result of your previous write — rewriting it "+
					"from memory just loops. Read the file to see what is actually there, then either make one targeted "+
					"change with edit_file or structural_edit, or respond with done if the request is satisfied.",
				p, count, toolRepeatWindow), true
		}
	}
	return fmt.Sprintf(
		"⚠ Tool-call repetition detected: you've called `%s` with these exact arguments %d times in the last %d turns. "+
			"The same call won't produce a different result. Try a different approach: (a) use different arguments to "+
			"discover what's actually there (different path, broader regex, list_directory before read_file), "+
			"(b) try a sibling tool — find_file if a path is unclear, run_command if a tool is failing in a confusing "+
			"way, (c) declare done if you've already gathered enough information, or (d) ask the user for clarification "+
			"if the task is ambiguous.",
		toolName, count, toolRepeatWindow), true
}

// writeFilePath extracts the path argument from write_file args ("" on
// any parse failure).
func writeFilePath(args json.RawMessage) string {
	var wf struct {
		Path string `json:"path"`
	}
	if json.Unmarshal(args, &wf) != nil {
		return ""
	}
	return wf.Path
}

// writeFileContentFingerprint returns a hash of the write_file content
// with ALL whitespace removed, or "" if there's no content. Whitespace-
// stripping makes the fingerprint stable across trivial reformatting (so
// reasserting the same draft with cosmetic changes still collides) while
// treating any material code change as different (so iterating toward a
// fix — polyglot rewriting main.py.c to clear a line-30 syntax error —
// produces a DIFFERENT fingerprint and is not counted as repetition).
func writeFileContentFingerprint(args json.RawMessage) string {
	var wf struct {
		Content string `json:"content"`
	}
	if json.Unmarshal(args, &wf) != nil || wf.Content == "" {
		return ""
	}
	// Normalize each line to its LEADING indentation + trailing-trimmed
	// body, then join with "\n". Leading whitespace is PRESERVED because in
	// Python it is semantic: an indentation-only fix is a real change and
	// must produce a different fingerprint, or it is misclassified as
	// reassertion and the loop breaker kills a legitimate iteration (#147
	// review finding #13). Trailing whitespace and CR are dropped as noise.
	lines := strings.Split(wf.Content, "\n")
	for i, ln := range lines {
		lines[i] = strings.TrimRight(ln, " \t\r")
	}
	h := sha1.Sum([]byte(strings.Join(lines, "\n")))
	return hex.EncodeToString(h[:])
}

// toolCallSignature computes a stable hash of a (tool_name, args)
// tuple. Re-marshals args through encoding/json to canonicalize key
// order and whitespace — important because the model sometimes emits
// the same logical call with slightly different JSON formatting that
// would defeat naive string-equality detection.
//
// write_file signatures are keyed on target path + a whitespace-stripped
// content fingerprint. Rewriting the SAME path with the SAME logical
// content is reassertion (a real loop — observed 2026-07-18: the model
// reasserted its ~25-line app.py draft five times while V3 wrote the
// verified expansion). Rewriting the same path with MATERIALLY DIFFERENT
// content is iteration (observed 2026-07-19 TB2: polyglot rewriting
// main.py.c three times to clear successive compiler errors, killed by
// the path-only key as if it were a loop). The content fingerprint
// separates the two: reassertion collides, iteration diverges. Falls back
// to path-only when there's no content to fingerprint.
// edit_file/structural_edit keep full-args signatures: distinct surgical
// edits to one file in close succession are legitimate iteration.
func toolCallSignature(toolName string, args json.RawMessage) string {
	if toolName == "write_file" {
		if p := writeFilePath(args); p != "" {
			key := toolName + "|path:" + p
			if fp := writeFileContentFingerprint(args); fp != "" {
				key += "|c:" + fp
			}
			h := sha1.Sum([]byte(key))
			return hex.EncodeToString(h[:])
		}
	}
	var v interface{}
	canonical := []byte(args)
	if err := json.Unmarshal(args, &v); err == nil {
		if b, err := json.Marshal(v); err == nil {
			canonical = b
		}
	}
	h := sha1.Sum([]byte(toolName + "|" + string(canonical)))
	return hex.EncodeToString(h[:])
}

// Reasoning-repetition detector. May 10 2026 BiasBusters follow-up #30.
//
// Sibling to:
//   - tool_repeat.go (structural call-shape repetition)
//   - lens_score.go's agentLensRegression (semantic content quality)
//
// The pattern this catches: the model emits reasoning_content that
// rehashes the same opening prose across consecutive turns ("Now I
// need to look at the file" / "Let me check the file" / similar)
// without committing to action. Tool-call repetition won't fire
// because the eventual tool calls may differ; lens scoring won't fire
// because the LANDED content (write_file/edit_file output) may be
// fine. The bug is in the THINKING, not in the action or the artifact.
//
// Detection: normalize the first reasoningSnippetLen chars of each
// turn's reasoning_content (lowercase + collapsed whitespace), compare
// to the previous turn's snippet. Identical snippets across
// reasoningRepeatThreshold consecutive turns triggers an intervention.
//
// Why prefix-match instead of full-text similarity: prose preambles
// rehash strongly at the opening ("Now I need to..." dominates the
// reasoning even when later sentences vary). Catching the OPENING is
// what distinguishes "model is stuck" from "model is on a different
// thought now." Embedding-based similarity (cosine over reasoning
// embeddings) is a future refinement; prefix-match handles the
// dominant repetition shape we've actually seen in user logs.

const (
	// reasoningRepeatThreshold is the number of consecutive identical
	// snippets that fires intervention. 2 = the SECOND consecutive
	// repetition (i.e. three turns total with the same opening). 1
	// would be the first repetition, which is too eager — the model
	// may have legitimately needed a second pass on the same topic.
	reasoningRepeatThreshold = 2

	// reasoningSnippetLen is the prefix length used for similarity
	// comparison. 80 chars is roughly the first 1-2 sentences of
	// the model's reasoning preamble, which is where the rehash pattern
	// shows. Longer would over-match (later sentences naturally vary
	// even within a stuck loop); shorter would under-match (the model's
	// boilerplate openings collide on unrelated tasks).
	reasoningSnippetLen = 80
)

// recordReasoning updates ctx with the current turn's reasoning
// snippet and returns the corrective message + true when the same
// snippet has appeared reasoningRepeatThreshold consecutive times.
// Returns ("", false) otherwise. Empty reasoning resets the counter
// (the detector only flags STUCK thinking, not absence of thinking).
//
// Caller should reset ctx.ConsecutiveReasoningRepeats and
// ctx.LastReasoningSnippet to "" after acting on the corrective so
// the same loop doesn't re-fire on the next iteration.
func recordReasoning(ctx *AgentContext, reasoning string) (string, bool) {
	snippet := normalizeReasoningSnippet(reasoning)
	if snippet == "" {
		// No reasoning emitted (or pure whitespace) — break the streak.
		// A single reasoning-free turn means the model committed to
		// action without preamble, which is exactly what we want to
		// reward, not flag as a continuation.
		ctx.ConsecutiveReasoningRepeats = 0
		ctx.LastReasoningSnippet = ""
		return "", false
	}

	if ctx.LastReasoningSnippet != "" && snippet == ctx.LastReasoningSnippet {
		ctx.ConsecutiveReasoningRepeats++
	} else {
		ctx.ConsecutiveReasoningRepeats = 0
	}
	ctx.LastReasoningSnippet = snippet

	if ctx.ConsecutiveReasoningRepeats < reasoningRepeatThreshold {
		return "", false
	}

	return fmt.Sprintf(
		"⚠ Reasoning repetition detected: your reasoning has opened with the same prose for %d consecutive turns "+
			"(\"%s...\"). The same opening won't change the outcome. Either commit to the next action — emit a "+
			"different tool call, run a verification command, or emit `done` if the task is complete — OR change "+
			"the investigation direction (read a different file, try a different selector, ask the user for "+
			"clarification). Don't rephrase the same thought.",
		ctx.ConsecutiveReasoningRepeats+1,
		truncateForCorrective(reasoning, 60),
	), true
}

// normalizeReasoningSnippet lowercases, collapses whitespace, and
// truncates to reasoningSnippetLen so similar openings compare equal
// across minor formatting differences. Empty input → empty output.
func normalizeReasoningSnippet(reasoning string) string {
	s := strings.TrimSpace(reasoning)
	if s == "" {
		return ""
	}
	s = strings.ToLower(s)
	// Collapse all whitespace runs to a single space.
	var b strings.Builder
	prevSpace := false
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
			}
			continue
		}
		b.WriteRune(r)
		prevSpace = false
	}
	out := strings.TrimSpace(b.String())
	if len(out) > reasoningSnippetLen {
		out = out[:reasoningSnippetLen]
	}
	return out
}

// truncateForCorrective returns the first n runes of s, suitable for
// embedding in a model-facing corrective string (preserves UTF-8).
func truncateForCorrective(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	// Byte-truncate then trim back to a rune boundary.
	cut := s[:n]
	if !strings.ContainsRune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?;:'-", rune(cut[len(cut)-1])) {
		// Lazy — just take whole bytes; if model preamble has Unicode
		// punctuation right at byte n we may chop a multi-byte char.
		// Safe-but-lossy: trim one extra byte.
		if len(cut) > 1 {
			cut = cut[:len(cut)-1]
		}
	}
	return cut
}

// Option 3 (issue #39 / Dmitri thesis): convert localization the model is bad
// at into the directed edit it is good at. When run_command surfaces a Python
// traceback, the deepest in-project frame names the exact fix site — so instead
// of leaving a weak model to "find the bug" (where it hallucinates symbols and
// edits the wrong function), the harness mechanically extracts file:line:
// function from the traceback and hands the model a directed instruction.
//
// Verified failure this addresses: a traceback pointing at draw():95 / get_item
// line N, after which the model edited an unrelated function. The stack frame IS
// the localization — no LLM reasoning required to read it.

var reTraceFrame = regexp.MustCompile(`File "([^"]+)", line (\d+), in (\S+)`)

// Patterns that name a file the OS couldn't open. Two shapes cover the
// common cases: Python/pip quote the name ("No such file or directory:
// 'Requirements.txt'"); shell tools put it before the colon ("cat:
// Requirements.txt: No such file or directory").
var (
	reMissingQuoted = regexp.MustCompile(`No such file or directory:?\s*'([^']+)'`)
	reMissingShell  = regexp.MustCompile(`(?m)([^\s:'"]+): No such file or directory`)
)

// reMissingModule matches both shapes of "package not installed": the
// `python -m` form (`/usr/local/bin/python3: No module named flask`) and the
// import form (`ModuleNotFoundError: No module named 'flask'`).
var reMissingModule = regexp.MustCompile(`No module named '?([A-Za-z0-9_.]+)'?`)

// Missing-binary shapes. bash spells it out ("bash: line 1: git: command
// not found"); dash/sh abbreviates ("sh: 1: git: not found") — the sh form
// requires the `sh: N:` prefix so a stray "<file>: not found" in program
// output can't false-positive.
var (
	// Anchored to a real bash diagnostic — `bash: [line N: ]<cmd>: command
	// not found` — so it can't fire on the string "X: command not found"
	// appearing in ordinary program output (#147 review finding #9). Path
	// prefixes on the shell name (/usr/bin/bash) and the `line N:` clause
	// are both optional.
	reCmdNotFoundBash = regexp.MustCompile(`(?m)(?:^|\s)(?:[\w./-]*/)?bash: (?:line \d+: )?([A-Za-z0-9._/+-]+): command not found`)
	reCmdNotFoundSh   = regexp.MustCompile(`(?m)(?:^|\s)(?:/bin/)?sh: \d+: ([A-Za-z0-9._/+-]+): not found`)
)

// missingModuleSteer catches the uninstalled-dependency loop: the model runs
// `python -m flask run` (or `python app.py`), the sandbox reports the package
// isn't installed, and the model re-runs the identical command until the
// repetition breaker kills the session (observed: flask run 3× then
// run_background flask run 3× → stuck). tracebackSteer deliberately ignores
// ModuleNotFoundError (it's not a code bug to localize), but ignoring it left
// NO positive guidance. This provides it: the sandbox ships no app libraries,
// so the fix is to install the package first. Returns "" when the output names
// no missing module.
func missingModuleSteer(ctx *AgentContext, output string) string {
	if !strings.Contains(output, "No module named") {
		return ""
	}
	m := reMissingModule.FindStringSubmatch(output)
	if m == nil {
		return ""
	}
	mod := m[1]
	if i := strings.Index(mod, "."); i > 0 {
		mod = mod[:i] // top-level package (flask.cli → flask)
	}
	// Prefer the project's own dependency manifest when one is present — it
	// pins the right versions and installs everything in one shot.
	hasReqs := false
	if entries, err := readWorkspaceDir(ctx, "."); err == nil {
		for _, e := range entries {
			switch strings.ToLower(e.Name()) {
			case "requirements.txt", "pyproject.toml", "pipfile":
				hasReqs = true
			}
		}
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "[system note]: The command failed because the Python package `%s` is not installed in the sandbox (it ships no app libraries — install what the project needs). ", mod)
	if hasReqs {
		sb.WriteString("Install the project's dependencies first with `pip install -r requirements.txt`, then re-run. ")
	} else {
		fmt.Fprintf(&sb, "Install it first with `pip install %s`, then re-run. ", mod)
	}
	sb.WriteString("Re-running the command before installing will fail exactly the same way.")
	return sb.String()
}

// missingCommandSteer catches the missing-binary loop: the model runs a
// command whose binary isn't in the sandbox image (`git clone ...` →
// "bash: line 1: git: command not found"), then either re-runs it
// identically into the repetition breaker or gives up outright (both
// observed on the TB2 bench, 2026-07-18: git and sqlite3). The sandbox
// runs non-root on a read-only base fs, so `apt-get install` can NEVER
// work at runtime — without this steer the model has no way to know
// that, and suggesting apt-get would just start a different loop. The
// steer states the constraint and points at the escape hatches that DO
// work: pip-installable equivalents (~/.local is writable, `python3 -m X`
// avoids PATH issues) or a different approach with the preinstalled
// toolchains. Returns "" when the output names no missing command.
func missingCommandSteer(output string) string {
	var cmd string
	if m := reCmdNotFoundBash.FindStringSubmatch(output); m != nil {
		cmd = m[1]
	} else if m := reCmdNotFoundSh.FindStringSubmatch(output); m != nil {
		cmd = m[1]
	}
	if cmd == "" {
		return ""
	}
	cmd = filepath.Base(cmd) // "/usr/bin/foo: command not found" → foo
	var sb strings.Builder
	fmt.Fprintf(&sb, "[system note]: The command failed because `%s` is not installed in the sandbox, and system packages CANNOT be installed at runtime (non-root, read-only base — apt-get/sudo will not work). Re-running the same command will fail identically. ", cmd)
	sb.WriteString("Instead: if a Python equivalent exists, `pip install <package>` works (invoke it as `python3 -m <module>` to avoid PATH issues); otherwise use one of the preinstalled toolchains (python3/pip, node/npm, go, cargo, ruby, php, java) or accomplish the step a different way.")
	return sb.String()
}

// brokenInlineScriptSteer catches the broken-verification-command loop: the
// model tries to verify its solution with `python -c "<multi-statement
// script>"` — a script containing a `def`/`for`/`if`/`class` body that can't
// live on a single -c line — so the command fails with a SyntaxError in the
// `-c` argument ITSELF, not in the file being tested. The model then re-runs
// the same malformed command (observed TB2 2026-07-19, regex-chess: the
// solution file re.json may be fine; the verify one-liner had `def` inline
// and never parsed) until the repetition breaker ends the session with the
// solution unverified. Steer it to move the test into a file. Keyed on a
// syntax error in code compiled from a string ("<string>") plus an inline
// -c/-command invocation, so a genuine syntax error in a real .py file (which
// tracebackSteer handles) doesn't match. Returns "" otherwise.
func brokenInlineScriptSteer(command, output string) string {
	// Signal the inline script is the error site: Python attributes errors
	// in code compiled from a string to "<string>"/"<stdin>" (a real file
	// error names the file). This is robust to output truncation — the
	// "<string>" frame is printed BEFORE the "SyntaxError:" line, so a
	// clipped sandbox result keeps the frame but may drop the keyword
	// (observed TB2 2026-07-19: the SyntaxError line was truncated away and
	// the keyword-gated check missed the loop).
	fromString := strings.Contains(output, `File "<string>"`) ||
		strings.Contains(output, `File "<stdin>"`)
	inlineFlag := strings.Contains(command, " -c ") ||
		strings.Contains(command, " -c\"") ||
		strings.Contains(command, "\t-c ")
	if !fromString || !inlineFlag {
		return ""
	}
	// If the -c script execs external code (exec(open(f).read()),
	// eval/compile), a SyntaxError in THAT code is also attributed to
	// "<string>" — the bug is in the exec'd file, not the one-liner, so
	// "move your test to a file" is wrong advice (#147 review finding #11).
	if strings.Contains(command, "exec(") || strings.Contains(command, "eval(") ||
		strings.Contains(command, "compile(") {
		return ""
	}
	// If a REAL file frame also appears, the error is in a module the -c
	// script imported, not the inline script itself — tracebackSteer
	// localizes that. Don't misfire "move your test to a file" onto a
	// genuine solution bug.
	if reRealFileFrame.MatchString(output) {
		return ""
	}
	return "[system note]: The error is in your inline `-c` script itself, not in the file you are testing — a multi-statement script (with a `def`/`for`/`if`/`class` body) cannot be written on a single `python -c` line. Your solution file may be correct; only the verification command is malformed. Write the test to a `.py` file with write_file, then run it with `run_command`: `python3 <testfile>.py`. Re-running the same `-c` one-liner will fail the same way."
}

// reRealFileFrame matches a traceback frame naming a real file (not the
// <string>/<stdin> pseudo-files that -c/exec/eval produce).
var reRealFileFrame = regexp.MustCompile(`File "[^<][^"]*"`)

// missingFileSteer catches the case-typo loop: the model writes
// `requirements.txt` then runs `pip install -r Requirements.txt`, gets "No
// such file or directory", and re-runs the identical wrong command (observed:
// 5× until the repetition breaker fired). When the missing name differs from a
// real workspace file only by case, the harness names the correct file so the
// model re-runs with the right name instead of looping. Returns "" when there
// is no missing-file error or no case-variant exists (so we never invent an
// anchor for a genuinely absent file).
func missingFileSteer(ctx *AgentContext, output string) string {
	if !strings.Contains(output, "No such file or directory") {
		return ""
	}
	// Collect candidate missing names from both error shapes.
	var cands []string
	for _, m := range reMissingQuoted.FindAllStringSubmatch(output, -1) {
		cands = append(cands, m[1])
	}
	for _, m := range reMissingShell.FindAllStringSubmatch(output, -1) {
		cands = append(cands, m[1])
	}
	seen := map[string]bool{}
	for _, cand := range cands {
		if cand == "" || seen[cand] {
			continue
		}
		seen[cand] = true
		base := filepath.Base(cand)
		dir := filepath.Dir(cand) // "." when cand is a bare filename
		entries, err := readWorkspaceDir(ctx, dir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			name := e.Name()
			if name != base && strings.EqualFold(name, base) {
				actual := name
				if dir != "." && dir != "" {
					actual = filepath.Join(dir, name)
				}
				return fmt.Sprintf("[system note]: There is no file `%s`, but `%s` exists — the name differs only in case. Re-run the command with the exact name `%s`. Do not re-run it unchanged.", cand, actual, actual)
			}
		}
	}
	return ""
}

// tracebackExclusion is the grammar-level counterpart to runBlockAfterTraceback.
// If the most recent tool result is a crashed run with a parseable traceback,
// it returns the run tools to ban from the next decision's GBNF tool-name enum
// plus a directed [system note]. The soft block returns an error the model can
// ignore (observed: it re-emitted run_command 6×); banning the tool name makes
// re-running *physically unemittable*, forcing the model to edit the named fix
// site. The restriction is scoped to one decision and clears once the model
// acts (the most recent tool result is then the edit, not the crash).
func tracebackExclusion(ctx *AgentContext) ([]string, string) {
	for i := len(ctx.Messages) - 1; i >= 0; i-- {
		m := ctx.Messages[i]
		if m.Role != "tool" {
			continue
		}
		if m.ToolName != "run_command" && m.ToolName != "run_background" {
			return nil, ""
		}
		var r struct {
			Data struct {
				Stdout string `json:"stdout"`
				Stderr string `json:"stderr"`
			} `json:"data"`
		}
		_ = json.Unmarshal([]byte(m.Content), &r)
		steer := tracebackSteer(ctx, r.Data.Stderr+"\n"+r.Data.Stdout)
		if steer == "" {
			return nil, ""
		}
		note := "[system note]: For this single decision, run_command and run_background are unavailable — the code is unchanged, so running it again only reproduces the crash. Make the edit now. " + steer
		return []string{"run_command", "run_background"}, note
	}
	return nil, ""
}

// runBlockAfterTraceback prevents the run-it-again loop. A weak model, handed a
// crash + a directed "fix function X" steer, often just re-emits the identical
// run_command instead of editing (observed: 6 identical runs, no edit). If the
// most recent tool result was a run that crashed with a traceback, block the
// next run and return the directed steer as the result — the code is unchanged,
// so re-running can only crash the same way. The block clears itself naturally:
// once the model edits, the most recent tool result is the edit, not the crash.
func runBlockAfterTraceback(ctx *AgentContext) *ToolResult {
	for i := len(ctx.Messages) - 1; i >= 0; i-- {
		m := ctx.Messages[i]
		if m.Role != "tool" {
			continue
		}
		if m.ToolName != "run_command" && m.ToolName != "run_background" {
			return nil // most recent tool wasn't a run (e.g. an edit) — don't block
		}
		var r struct {
			Data struct {
				Stdout string `json:"stdout"`
				Stderr string `json:"stderr"`
			} `json:"data"`
			Error string `json:"error"`
		}
		_ = json.Unmarshal([]byte(m.Content), &r)
		steer := tracebackSteer(ctx, r.Data.Stderr+"\n"+r.Data.Stdout)
		if steer == "" {
			return nil
		}
		return &ToolResult{Success: false, Error: "Re-running is blocked: the code is unchanged, so it will crash exactly the same way. Edit the code FIRST, then run. " + steer}
	}
	return nil
}

// tracebackSteer scans tool output for a Python traceback and returns a
// directed steer naming the exact fix site, or "" when there is no parseable
// in-project frame. ctx is used to read the offending line from disk
// (best-effort) so the steer can quote it.
func tracebackSteer(ctx *AgentContext, output string) string {
	if !strings.Contains(output, "Traceback (most recent call last)") {
		return ""
	}
	frames := reTraceFrame.FindAllStringSubmatch(output, -1)
	if len(frames) == 0 {
		return ""
	}

	// Walk frames outermost→deepest; keep the LAST one that's a project file
	// (skip stdlib / site-packages / <string> / <frozen ...> frames — the bug
	// is in the user's code, not the library it called).
	var file, fn string
	var lineNo int
	for _, f := range frames {
		p := f[1]
		if strings.Contains(p, "site-packages") || strings.Contains(p, "/usr/lib/") ||
			strings.Contains(p, "/lib/python") || strings.HasPrefix(p, "<") {
			continue
		}
		n, err := strconv.Atoi(f[2])
		if err != nil {
			continue
		}
		file, lineNo, fn = p, n, f[3]
	}
	if file == "" || lineNo == 0 {
		return ""
	}

	// Exception summary = last non-indented, non-"Traceback" line.
	exc := ""
	for _, l := range strings.Split(strings.TrimRight(output, "\n"), "\n") {
		if l == "" || strings.HasPrefix(l, " ") || strings.HasPrefix(l, "\t") ||
			strings.HasPrefix(l, "Traceback") {
			continue
		}
		exc = strings.TrimSpace(l)
	}

	// Don't fire on environment errors. A missing top-level package
	// (ModuleNotFoundError: pygame) is not a code bug the model can fix by
	// editing the function the frame points at — the "fix" is installing the
	// dependency. Steering + banning runs here would force the model to "edit"
	// an unfixable import and loop. Let the normal flow handle it (the model
	// can choose to install or switch libraries).
	if strings.HasPrefix(exc, "ModuleNotFoundError") || strings.HasPrefix(exc, "ImportError") {
		return ""
	}

	// Best-effort: read the offending line so the steer can quote real bytes.
	// Also record the read: the steer hands the model this file's content, and
	// the very next thing we want it to do is edit it — but edit_file/structural_edit
	// require a prior read_file (the blind-edit guard). Without recording it,
	// the model's correct directed edit bounces with "file not read yet," it
	// loops, and gets stopped (the 2/3 variance). The harness HAS read the file
	// to build this steer, so the edit is grounded, not blind.
	exact := ""
	if data, resolved, err := readWorkspaceFile(ctx, file); err == nil {
		lines := strings.Split(string(data), "\n")
		if lineNo >= 1 && lineNo <= len(lines) {
			exact = strings.TrimSpace(lines[lineNo-1])
		}
		ctx.RecordFileRead(resolved, string(data))
	}

	rel := file
	if i := strings.Index(rel, "/workspace/"); i >= 0 {
		rel = rel[i+len("/workspace/"):]
	}

	var sb strings.Builder
	sb.WriteString("[system note]: The traceback points at the exact bug location — ")
	fmt.Fprintf(&sb, "%s line %d, in function `%s`", rel, lineNo, fn)
	if exc != "" {
		sb.WriteString(" (" + exc + ")")
	}
	sb.WriteString(". ")
	// Directed, MINIMAL-edit instruction. The model fixes the right function
	// reliably now, but rewriting the whole node via structural_edit makes it typo
	// the unchanged parts (observed: items -> Items, a fresh NameError it then
	// repeats). When we have the exact line, tell it to change ONLY that line
	// with edit_file: old_str = the verbatim line (so the match is exact and
	// the model isn't recalling it), new_str = the same line with only the bug
	// fixed. That shrinks the model's text generation to a one-line delta and
	// removes the collateral-typo surface.
	if exact != "" {
		fmt.Fprintf(&sb, "The buggy line is EXACTLY:\n%s\n", exact)
		sb.WriteString("Fix it with a MINIMAL edit_file: set old_str to that exact line (copy it character-for-character) and new_str to the SAME line with only the bug corrected. ")
		sb.WriteString("Change nothing else on the line — do not rename variables, do not re-case anything, do not rewrite the whole function, and do not hardcode a value. Change only what causes the error.")
	} else if fn != "<module>" && fn != "" {
		fmt.Fprintf(&sb, "Fix the bug in `%s` with structural_edit selector `function:%s`. Keep every identifier exactly as it already appears (same spelling and case) — change only the buggy logic. Do not edit other functions or hardcode a value.", fn, fn)
	} else {
		fmt.Fprintf(&sb, "Fix the code at line %d — change only the buggy logic, keep all other identifiers exactly as written.", lineNo)
	}
	return sb.String()
}
