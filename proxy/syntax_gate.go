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

// fallbackSyntaxRejection builds the tool error handed back to the model
// when the gate blocks a fallback write. Names the likely cause
// (truncated tool call) and the recovery (resend complete content).
func fallbackSyntaxRejection(path, syntaxErr string) string {
	return fmt.Sprintf(
		"V3 verification was unavailable and your content for %s does not "+
			"parse (%s). The file was NOT written — this usually means the "+
			"tool call was truncated mid-content. Retry write_file with the "+
			"COMPLETE file content (if it is long, write the file in full, "+
			"not in fragments).", path, truncateStr(syntaxErr, 200))
}
