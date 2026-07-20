package main

// Syntax gate for unverified fallback writes. When a V3 call fails or
// times out, the fallback used to write the model's raw baseline to disk
// with success=true — and a truncated tool call (content cut mid-string)
// landed as a file with a SyntaxError while the agent believed the write
// succeeded. Observed twice in the 2026-07-18 mini-bench (t06, t09):
// V3 hit its 3-minute cap, the fallback wrote a 362-byte truncated
// baseline, the follow-up run failed, and the loop breakers stopped a
// session whose "productive change" was a broken file.
//
// The gate routes fallback content through the sandbox's /syntax-check
// (the same checker V3's smoke pass uses). Fail-open by design: if the
// sandbox is unreachable or the file type unsupported, the write
// proceeds — the gate only blocks KNOWN-broken content.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// syntaxGateLanguages maps extensions to the sandbox's language names.
// Only types the sandbox's /syntax-check actually verifies are listed —
// anything else passes through ungated.
var syntaxGateLanguages = map[string]string{
	".py":   "python",
	".js":   "javascript",
	".ts":   "typescript",
	".go":   "go",
	".java": "java",
	".kt":   "kotlin",
	".rb":   "ruby",
	".php":  "php",
	".sh":   "bash",
	".json": "json",
	".yaml": "yaml",
	".yml":  "yaml",
	".html": "html",
	".htm":  "html",
	".xml":  "xml",
}

// checkFallbackSyntax returns ("", true) when `content` is safe to write
// as a fallback: it parsed cleanly, or it could not be checked (sandbox
// down, unsupported extension). Returns (firstError, false) when the
// sandbox confirmed the content does not parse.
func checkFallbackSyntax(ctx *AgentContext, path, content string) (string, bool) {
	if ctx == nil || ctx.SandboxURL == "" {
		return "", true
	}
	lang, gated := syntaxGateLanguages[strings.ToLower(filepath.Ext(path))]
	if !gated {
		return "", true
	}
	body, err := json.Marshal(map[string]string{
		"code":     content,
		"language": lang,
	})
	if err != nil {
		return "", true
	}
	client := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequest("POST", ctx.SandboxURL+"/syntax-check", bytes.NewReader(body))
	if err != nil {
		return "", true
	}
	req.Header.Set("Content-Type", "application/json")
	if serviceToken != "" {
		req.Header.Set("Authorization", "Bearer "+serviceToken)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", true // fail-open: gate only blocks confirmed-broken content
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", true
	}
	var out struct {
		Valid  bool     `json:"valid"`
		Errors []string `json:"errors"`
	}
	if json.NewDecoder(resp.Body).Decode(&out) != nil {
		return "", true
	}
	if out.Valid {
		return "", true
	}
	first := "syntax error"
	if len(out.Errors) > 0 {
		first = out.Errors[0]
	}
	return first, false
}

// reSyntaxLineNo pulls a 1-based line number out of a Python syntax error
// message ("... (file, line 13)" or "at line 13"), when present.
var reSyntaxLineNo = regexp.MustCompile(`line (\d+)`)

// fallbackSyntaxRejection builds the tool error handed back to the model
// when the gate blocks a write. It DISTINGUISHES the two failure shapes,
// because the old one-size message ("truncated — resend complete content")
// is actively wrong for a genuine syntax bug in COMPLETE content and made
// the model reassert the same broken text (TB2 2026-07-20,
// pytorch-model-recovery: an f-string with nested quotes resent verbatim 5×):
//   - truncation shape (unterminated string / unexpected EOF / "never
//     closed") → the content really is cut off; resend it complete.
//   - a mid-content syntax bug → point at the offending line (quoted from
//     `content` when the error carries a line number) and tell the model to
//     FIX that line, explicitly forbidding an identical resend.
func fallbackSyntaxRejection(path, content, syntaxErr string) string {
	low := strings.ToLower(syntaxErr)
	truncationShape := strings.Contains(low, "unexpected eof") ||
		strings.Contains(low, "was never closed") ||
		strings.Contains(low, "unterminated") ||
		strings.Contains(low, "expected an indented block")
	if truncationShape {
		return fmt.Sprintf(
			"Your content for %s does not parse (%s) — this looks like a "+
				"truncated tool call (content cut off mid-way). Retry write_file "+
				"with the COMPLETE file content; if it is long, write it in full, "+
				"not in fragments.", path, truncateStr(syntaxErr, 200))
	}
	// Genuine syntax bug: quote the offending line if we can locate it.
	quoted := ""
	if m := reSyntaxLineNo.FindStringSubmatch(syntaxErr); m != nil {
		if n, err := strconv.Atoi(m[1]); err == nil && n >= 1 {
			if lines := strings.Split(content, "\n"); n <= len(lines) {
				quoted = fmt.Sprintf(" The offending line %d is:\n%s\n", n, strings.TrimRight(lines[n-1], " \t"))
			}
		}
	}
	return fmt.Sprintf(
		"Your content for %s has a syntax error (%s) — it was NOT written. The "+
			"content is NOT truncated; it is complete but INVALID.%s Fix THAT "+
			"specific error (e.g. a common cause is nested double-quotes inside "+
			"an f-string — use single quotes for the inner string, or a temp "+
			"variable). Do NOT resend the same content unchanged; it will fail "+
			"identically.", path, truncateStr(syntaxErr, 200), quoted)
}
