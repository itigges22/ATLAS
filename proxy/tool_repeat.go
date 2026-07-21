package main

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

import (
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

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
					"change with edit_file or ast_edit, or respond with done if the request is satisfied.",
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
// edit_file/ast_edit keep full-args signatures: distinct surgical
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
