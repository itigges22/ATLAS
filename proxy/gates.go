// Gates: the checks the agent loop runs at its decision points — before a
// write reaches disk, and before a `done` is accepted.
//
// In file order:
//
//	Completion-claim verification — bounces a `done` whose summary claims
//	  universal success ("all routes work") while the workspace still shows
//	  a structural gap. Needs both halves, so a narrow summary or a clean
//	  workspace passes untouched.
//	Structural-unresolved gate — asks v3-service whether a write or edit
//	  leaves an undefined name behind, and rejects only the unresolved names
//	  that this change introduced.
//	Syntax gate — routes fallback writes through the sandbox's
//	  /syntax-check. These are the writes that skipped the V3 pipeline, so
//	  nothing else in the loop has parsed them.
//	Embedded-script gate — parses the JS/CSS inside <script>/<style> blocks
//	  in HTML files and in Python string literals, which every other gate is
//	  structurally blind to.
//	Plan adherence — matches each tool call against the pre-flight plan and
//	  counts the off-plan streak, regenerating the plan once the streak runs
//	  long. Advisory: it never blocks a call.
//	Plan-progress reminder — renders the compact step-progress block
//	  injected ahead of each LLM call so a long multi-file task doesn't
//	  lose track of what's left.
//	Asset-graph lint — cross-file coherence for small web projects: a
//	  template no route renders, an href to a file that isn't there, a fetch
//	  to a route that doesn't exist. Advisory notes, deduped per session.
//
// They share a shape, not a subject. Each one inspects agent output or the
// workspace at a single point in the loop and returns a string; the blocking
// four return it as a rejection the model must answer, the advisory three
// return it as a [system note]. They hold no state in common — this is a
// policy surface, not a pipeline, and gates can be read and changed one at a
// time.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Completion-claim verification.
//
// Background: the agent's done.summary often makes universal claims
// ("all routes work", "fixed all bugs", "verified everything") that
// we can structurally check against the workspace. The May 2026 flask
// run had the model claim "All routes are functioning properly" while
// only 3 of 7 needed templates existed. The verification gate only
// checks "did you run a verification command at all?" — it does
// NOT check whether the claim in the summary matches reality.
//
// Two-stage filter:
//   1. claimsUniversal(summary) — does the wording make a global
//      assertion? Quiet pass for narrow done summaries ("added /admin
//      route" — model said nothing about the rest of the app).
//   2. verifyCompletionClaims(workingDir) — cheap structural
//      checks for the failure modes we know about. Returns a directive
//      to the model when a gap is found, "" otherwise.
//
// Conservative on false positives. Universal claims with no gap pass
// silently; narrow claims pass even when there ARE gaps elsewhere.
// Only the AND case (claim + gap) bounces. The model can override by
// using narrower wording or by calling out the gap explicitly.

// multiIssueWords trips when the USER PROMPT (not the summary) signals
// "fix multiple things." Models bypass claimsUniversal by writing a
// narrow done summary ("fixed the product route") even when the user
// clearly asked for the whole app to work. Catching it on the prompt
// side handles that.
var multiIssueWords = []string{
	"lots of", "ton of", "tons of", "many", "multiple", "several",
	"all bugs", "all the bugs", "all issues", "all the issues",
	"all errors", "all the errors", "all routes", "all endpoints",
	"all tests", "all pages", "all the routes", "all the endpoints",
	"fix all", "fix everything", "fix the bugs", "fix the issues",
	"all of the", "everything", "nothing works",
	"it does not work", "doesn't work", "isn't working",
}

// promptIsMultiIssue returns true when the user explicitly framed the
// task as "fix multiple things" or "make this whole thing work."
// Used as an alternate trigger for the claim-check gate so a narrow
// done summary still gets verified against the workspace.
func promptIsMultiIssue(prompt string) bool {
	lower := strings.ToLower(prompt)
	for _, w := range multiIssueWords {
		if strings.Contains(lower, w) {
			return true
		}
	}
	return false
}

// claimWords trips the universal-claim filter. The phrases below are
// what models actually emit when they oversummarize ("all", "every",
// "no errors", "everything works", "fully functional", etc.).
var claimWords = []string{
	"all routes", "all endpoints", "all pages", "all tests",
	"all bugs", "all issues", "all errors",
	"every route", "every endpoint", "every page",
	"all routes are", "all endpoints are",
	"fully functional", "fully working", "fully operational",
	"completely fixed", "completely working", "completely done",
	"no errors", "no issues", "no bugs", "no problems",
	"everything works", "everything is working", "everything is fixed",
	"fixed all", "verified all", "tested all",
	"functioning properly", "functioning correctly", "working properly",
}

// claimsUniversal returns true when the summary contains a global
// assertion the structural checks should validate. Case-insensitive.
func claimsUniversal(summary string) bool {
	lower := strings.ToLower(summary)
	for _, w := range claimWords {
		if strings.Contains(lower, w) {
			return true
		}
	}
	return false
}

// verifyCompletionClaims returns a non-empty directive when the model's
// universal claim doesn't match reality. The directive is shaped as
// a tool-result error, so it lands back in the model's context as
// "your done was bounced because X."
//
// The structural evidence comes from assetLintFindings — the same
// bounded workspace walk the advisory lint uses — filtered down to the
// hard gaps: template references (render_template('X') in .py,
// {% extends/include %} in templates) whose target does not exist.
// Those are blocking because a missing render_template target is a
// guaranteed 500 at runtime; the rest of the lint stays advisory.
func verifyCompletionClaims(workingDir string) string {
	if workingDir == "" {
		return ""
	}
	var gaps []string
	for _, f := range assetLintFindings(workingDir) {
		if strings.Contains(f, "references template ") {
			gaps = append(gaps, f)
		}
	}
	if len(gaps) == 0 {
		return ""
	}
	return fmt.Sprintf(
		"Your `done` summary claims the work is complete, but a structural check of the workspace found gaps:\n\n%s\n\nFix the missing files (or correct your summary to acknowledge what's not done) before declaring done.",
		strings.Join(gaps, "\n"))
}

// Structural gate for the edit and write paths (issue #147). The V3
// structural veto hard-rejects generated candidates whose direct-identifier
// calls resolve to no local def, import, or builtin — but the edit path
// (improveContentWithV3) frequently sent no project_context, so the
// in-pipeline veto was gated off, and even when it fired the pipeline's
// baseline fallback resurrected the model's own edit. Result observed in
// 2026-07-18 dogfooding: a structural_edit replaced a route with a body calling
// render_template while the file imported only render_template_string; it
// passed V3 verification, landed as verified, and every request 500'd
// (NameError). structural_edit had no syntax gate at all; edit_file's syntax gate
// catches parse failures but a NameError parses fine.
//
// This proxy-side gate closes the hole where it can't be bypassed: it
// resolves the COMPOSED post-change file through v3-service's structural
// checker and refuses landing content that INTRODUCES an unresolved direct
// call — the same healthy->broken rule as the syntax gate (a change that
// leaves a pre-existing unresolved name in place, i.e. a repair-in-
// progress, is allowed). Wired into edit_file, structural_edit, and every
// write_file branch (V3 winner, V3-error fallback, iteration fast-path,
// T0/T1 direct); under BypassV3 only the non-iterating T0/T1 direct
// write_file skips the gate (so the demo baseline pane shows the raw
// model) — the edit paths and the iteration fast-path stay gated in all
// modes. Python-only and fail-open: if v3-service is unreachable, the file
// isn't .py, or tree-sitter is unavailable, the write proceeds — the gate
// only blocks on a POSITIVE, newly-introduced unresolved call.

// checkStructuralUnresolved returns the direct-identifier calls in
// `content` that resolve to nothing (no local def, import, builtin, or
// supplied project symbol), and ok=true only when the check actually ran.
// Fail-open: (nil, false) for a non-.py file, an empty V3 URL, a network
// failure, or a tree-sitter/parse error on the far side.
func checkStructuralUnresolved(ctx *AgentContext, path, content string) ([]string, bool) {
	if ctx == nil || ctx.V3URL == "" {
		return nil, false
	}
	if strings.ToLower(filepath.Ext(path)) != ".py" {
		return nil, false
	}
	payload := map[string]interface{}{"path": path, "source": content}
	// Pass the OTHER files the model has read as project context so a call
	// to a symbol defined elsewhere in the project is credited (more
	// lenient = fewer false blocks). Crucially, EXCLUDE the file being
	// edited: SnapshotFilesRead still holds its PRE-EDIT body, which would
	// credit a top-level def the edit just deleted and let a genuine
	// NameError through (#147 review finding #2). The edited file's current
	// symbols come from `source`, which structural_score parses directly.
	cleanTarget := filepath.Clean(path)
	rel := make(map[string]string)
	addContext := func(p, c string) {
		if filepath.Clean(p) == cleanTarget {
			return // don't credit the pre-edit self
		}
		r, err := filepath.Rel(ctx.WorkingDir, p)
		if err != nil || r == "" {
			r = p
		}
		// Only .py files carry resolvable symbols, and entries are
		// truncated like the V3 request builders — read_file snapshots
		// can be 200 KB each, and this body is POSTed per gated write.
		if strings.ToLower(filepath.Ext(r)) != ".py" {
			return
		}
		if len(c) > 4000 {
			c = c[:4000] + "\n... (truncated)"
		}
		rel[r] = c
	}
	for p, c := range ctx.SnapshotFilesRead() {
		addContext(p, c)
	}
	// Files the session WROTE are leniency context too — write_file paths
	// never RecordFileRead, so without these a sibling the session just
	// created is invisible here while the in-pipeline veto (which merges
	// SessionWrites) credits it, making this gate strictly stricter than
	// the veto it backstops. Disk content wins over any stale snapshot.
	for w := range ctx.SessionWrites {
		if w == "" || strings.ToLower(filepath.Ext(w)) != ".py" {
			continue // only .py carries symbols — skip the disk read otherwise
		}
		abs := resolveAgentPath(ctx, w)
		if filepath.Clean(abs) == cleanTarget {
			continue // pre-edit self; skip the read
		}
		if data, err := os.ReadFile(abs); err == nil {
			addContext(abs, string(data))
		}
	}
	if len(rel) > 0 {
		payload["project_context"] = rel
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, false
	}
	// ctx.Ctx may be nil on paths constructed without a request context;
	// the gate must fail open (or keep working), never panic.
	base := ctx.Ctx
	if base == nil {
		base = context.Background()
	}
	reqCtx, cancel := context.WithTimeout(base, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, "POST",
		ctx.V3URL+"/internal/structural_check", bytes.NewReader(body))
	if err != nil {
		return nil, false
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, false // fail-open
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, false
	}
	var r struct {
		OK         bool     `json:"ok"`
		Unresolved []string `json:"unresolved"`
	}
	if json.Unmarshal(raw, &r) != nil || !r.OK {
		return nil, false // parse error / tree-sitter missing -> fail-open
	}
	return r.Unresolved, true
}

// editIntroducesUnresolved returns the names an edit NEWLY makes
// unresolved — present in the edited file's unresolved set but not the
// original's. Mirrors the syntax gate's healthy->broken rule: an edit must
// not INTRODUCE a NameError, but a pre-existing unresolved name (the model
// is mid-repair) is allowed to remain. Returns nil when the check couldn't
// run (fail-open) or nothing new was introduced.
func editIntroducesUnresolved(ctx *AgentContext, path, original, edited string) []string {
	editedUnres, ok := checkStructuralUnresolved(ctx, path, edited)
	if !ok || len(editedUnres) == 0 {
		return nil
	}
	origUnres, ok := checkStructuralUnresolved(ctx, path, original)
	if !ok {
		// One retry: the edited-side call just succeeded, so a failure
		// here is a transient blip on the second back-to-back request.
		origUnres, ok = checkStructuralUnresolved(ctx, path, original)
	}
	if !ok {
		// The original-side check couldn't run (transient service failure;
		// tree-sitter-missing would have failed the edited side first, and
		// malformed Python does NOT trigger this — tree-sitter parses it
		// tolerantly and returns a partial extraction). Without a baseline
		// the healthy->broken comparison is meaningless, and counting
		// EVERY unresolved name as newly introduced would block the model
		// from fixing one error at a time — fail open instead.
		return nil
	}
	was := make(map[string]bool, len(origUnres))
	for _, n := range origUnres {
		was[n] = true
	}
	var introduced []string
	for _, n := range editedUnres {
		if !was[n] {
			introduced = append(introduced, n)
		}
	}
	return introduced
}

// readOriginalForGate returns the on-disk original for the healthy->broken
// comparison. A missing file is a first write (empty original — every
// unresolved call counts as introduced). Any OTHER read failure means the
// original is unknowable, so the caller must skip the gate (fail open)
// rather than treat the file as empty and count pre-existing unresolved
// calls as newly introduced.
func readOriginalForGate(path string) (string, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", true
		}
		return "", false
	}
	return string(data), true
}

// structuralRejection builds the tool error handed back when the gate
// blocks an edit — names the offending calls and the recovery.
// structuralWriteRejection is the write_file variant: the recovery must
// name the operation the model actually issued — an "edit" steer on a
// blocked NEW-file write sends the model to edit_file against a file
// that doesn't exist.
func structuralRejection(path string, introduced []string) string {
	return fmt.Sprintf(
		"edit for %s calls %s, which the file neither imports, defines, nor "+
			"gets from builtins — running it would raise NameError. The file was "+
			"NOT modified. Add the missing import (or correct the name to one that "+
			"IS in scope), then re-issue the edit.",
		path, quoteNames(introduced))
}

func structuralWriteRejection(path string, introduced []string) string {
	return fmt.Sprintf(
		"write_file for %s calls %s, which the file neither imports, defines, "+
			"nor gets from builtins — running it would raise NameError. Nothing "+
			"was written. Add the missing import (or correct the name to one that "+
			"IS in scope), then re-issue the write_file with the full corrected "+
			"content.",
		path, quoteNames(introduced))
}

func quoteNames(names []string) string {
	quoted := make([]string, len(names))
	for i, n := range names {
		quoted[i] = "`" + n + "`"
	}
	return strings.Join(quoted, ", ")
}

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
// down, unsupported extension). Returns (firstError, false) when a checker
// confirmed the content does not parse.
//
// Two checkers run, whole-file first: the sandbox parses the file in its own
// language, then checkEmbeddedScript parses the JavaScript/CSS that lives
// INSIDE it (a <script> block in an .html file, or in a Python string handed
// to render_template_string). The sandbox's checker sees the Python or the
// markup only, so a stray `)` in embedded JavaScript passes it.
func checkFallbackSyntax(ctx *AgentContext, path, content string) (string, bool) {
	if ctx == nil {
		return "", true
	}
	if msg, ok := checkSandboxSyntax(ctx, path, content); !ok {
		return msg, false
	}
	return checkEmbeddedScript(ctx, path, content)
}

// checkSandboxSyntax is the whole-file half of checkFallbackSyntax: the
// sandbox's /syntax-check in the file's own language.
func checkSandboxSyntax(ctx *AgentContext, path, content string) (string, bool) {
	if ctx.SandboxURL == "" {
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

// ---------------------------------------------------------------------------
// Embedded-script gate
// ---------------------------------------------------------------------------
//
// 2026-08-01 dogfooding: the model edited a Flask app whose whole UI is one
// HTML string (`HTML_TEMPLATE = """..."""` → render_template_string) and left
// a stray closing paren inside its <script> block:
//
//	else if(key === 'ArrowDown' && direction !== 'UP') nextDirection = 'DOWN');
//
// Every gate on the write path was structurally blind to it. The Python
// compiles (the JavaScript is string content), so pycheck and the sandbox's
// /syntax-check pass; the server starts and `curl /` returns 200, so the
// verification gate passes and `done` is accepted — while the game is dead in
// the browser. Nothing in the loop ever parses the JavaScript.
//
// checkEmbeddedScript closes that by routing the content through
// v3-service /internal/embedded_script_check, which tree-sitter-parses the
// JS/CSS inside <script>/<style> blocks — in .html/.htm/.jinja/.jinja2 files
// and in Python string literals.
//
// Fail-soft everywhere: no V3 URL, an unreachable service, a missing grammar,
// a parse timeout or an unsupported file type all mean "no finding", never a
// blocked write. The far side is conservative in the same direction — an
// ambiguous block (template statement tags, `<script src>`, a non-JS `type`,
// an escaped Python string) reports nothing rather than guessing.

// embeddedScriptErrPrefix marks a checkFallbackSyntax error as an
// embedded-script finding. The message is pre-formatted for the model, so
// callers hand it back verbatim instead of wrapping it in generic
// "does not parse / check your old_str" advice that would be wrong here.
const embeddedScriptErrPrefix = "embedded-script: "

// embeddedScriptExts are the file types that can CARRY an embedded script.
// Anything else short-circuits before any network call.
var embeddedScriptExts = map[string]bool{
	".py": true, ".html": true, ".htm": true, ".jinja": true, ".jinja2": true,
}

// embeddedScriptFinding mirrors one entry of the v3-service response.
type embeddedScriptFinding struct {
	Line    int    `json:"line"`
	Column  int    `json:"column"`
	Kind    string `json:"kind"`    // "javascript" | "css"
	Where   string `json:"where"`   // "the <script> block inside the Python string HTML_TEMPLATE"
	Message string `json:"message"` // "unexpected `)`"
	Hint    string `json:"hint"`    // how to fix it
	Text    string `json:"text"`    // the offending source line
}

// checkEmbeddedScript returns ("", true) when `content` has no broken embedded
// script, or when the check could not run. Returns (prefixed rejection, false)
// when v3-service confirms the embedded JavaScript/CSS does not parse.
func checkEmbeddedScript(ctx *AgentContext, path, content string) (string, bool) {
	if ctx == nil || ctx.V3URL == "" {
		return "", true
	}
	if !embeddedScriptExts[strings.ToLower(filepath.Ext(path))] {
		return "", true
	}
	// Cheap local pre-filter: no <script/<style anywhere means no network
	// call. Most gated writes never touch the service because of this.
	low := strings.ToLower(content)
	if !strings.Contains(low, "<script") && !strings.Contains(low, "<style") {
		return "", true
	}
	body, err := json.Marshal(map[string]string{"path": path, "source": content})
	if err != nil {
		return "", true
	}
	base := ctx.Ctx
	if base == nil {
		base = context.Background()
	}
	reqCtx, cancel := context.WithTimeout(base, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, "POST",
		ctx.V3URL+"/internal/embedded_script_check", bytes.NewReader(body))
	if err != nil {
		return "", true
	}
	req.Header.Set("Content-Type", "application/json")
	if serviceToken != "" {
		req.Header.Set("Authorization", "Bearer "+serviceToken)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", true // fail-soft: unreachable service never blocks a write
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", true
	}
	var out struct {
		OK       bool                    `json:"ok"`
		Findings []embeddedScriptFinding `json:"findings"`
	}
	if json.NewDecoder(resp.Body).Decode(&out) != nil || !out.OK {
		return "", true // grammar missing / non-UTF-8 -> fail-soft
	}
	if len(out.Findings) == 0 {
		return "", true
	}
	return embeddedScriptErrPrefix + formatEmbeddedScriptRejection(path, out.Findings[0]), false
}

// embeddedScriptGate applies the healthy->broken rule the other write gates
// use and returns the rejection text, or "" to allow the write. A file whose
// embedded script was ALREADY broken before the change is a repair-in-progress
// and is left alone; only a change that newly breaks it is blocked. Fail-soft:
// "" whenever the check couldn't run.
func embeddedScriptGate(ctx *AgentContext, path, original, edited string) string {
	synErr, ok := checkEmbeddedScript(ctx, path, edited)
	if ok {
		return ""
	}
	if _, wasHealthy := checkEmbeddedScript(ctx, path, original); !wasHealthy {
		return ""
	}
	msg, _ := embeddedScriptRejectionFor(synErr)
	return msg
}

// embeddedScriptRejectionFor unwraps a checkFallbackSyntax error that turned
// out to be an embedded-script finding. (message, true) when it is one — the
// message is already model-ready — else ("", false).
func embeddedScriptRejectionFor(syntaxErr string) (string, bool) {
	if !strings.HasPrefix(syntaxErr, embeddedScriptErrPrefix) {
		return "", false
	}
	return strings.TrimPrefix(syntaxErr, embeddedScriptErrPrefix), true
}

// formatEmbeddedScriptRejection names the file, the line, the offending
// construct and the fix — and spells out WHY the model's usual verification
// missed it, because "I ran it and curled the page" is exactly the evidence
// that made the broken snake game look done.
func formatEmbeddedScriptRejection(path string, f embeddedScriptFinding) string {
	lang, block, breakage := "JavaScript", "<script>", "the browser stops running the script at that point, so the page loads but nothing on it responds"
	if f.Kind == "css" {
		lang, block, breakage = "CSS", "<style>", "the browser drops the rest of the stylesheet, so the page loads unstyled"
	}
	host := "HTML"
	if strings.ToLower(filepath.Ext(path)) == ".py" {
		host = "Python"
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "%s has a %s syntax error in %s — it was NOT written.\n",
		path, lang, f.Where)
	fmt.Fprintf(&sb, "line %d: %s\n", f.Line, f.Message)
	if f.Text != "" {
		fmt.Fprintf(&sb, "  %d | %s\n", f.Line, f.Text)
	}
	if f.Hint != "" {
		fmt.Fprintf(&sb, "%s\n", f.Hint)
	}
	fmt.Fprintf(&sb,
		"Running the file will NOT surface this: the %s syntax is valid, the server "+
			"still starts and the page still returns 200 — but %s. Fix line %d inside "+
			"the %s block and re-send the corrected content; do NOT resend it unchanged.",
		host, breakage, f.Line, block)
	return sb.String()
}

// reSyntaxLineNo pulls a 1-based line number out of a Python syntax error
// message ("... (file, line 13)" or "at line 13"), when present.
var reSyntaxLineNo = regexp.MustCompile(`line (\d+)`)

// fallbackSyntaxRejection builds the tool error handed back to the model
// when the gate blocks a write. It DISTINGUISHES the two failure shapes,
// because the old one-size message ("truncated — resend complete content")
// is actively wrong for a genuine syntax bug in COMPLETE content and made
// the model reassert the same broken text (observed 2026-07-20 on a
// pytorch-model-recovery task: an f-string with nested quotes resent 5×):
//   - truncation shape (unterminated string / unexpected EOF / "never
//     closed") → the content really is cut off; resend it complete.
//   - a mid-content syntax bug → point at the offending line (quoted from
//     `content` when the error carries a line number) and tell the model to
//     FIX that line, explicitly forbidding an identical resend.
func fallbackSyntaxRejection(path, content, syntaxErr string) string {
	// An embedded-script finding arrives pre-formatted (it already names the
	// line and the fix); the truncation/syntax-bug fork below is about the
	// host file and would give wrong advice for JavaScript inside a string.
	if msg, isEmbedded := embeddedScriptRejectionFor(syntaxErr); isEmbedded {
		return msg
	}
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

// ---------------------------------------------------------------------------
// Plan adherence — track tool calls against the pre-flight plan
// ---------------------------------------------------------------------------
//
// Adherence is advisory by default: we record which planned steps a tool
// call satisfies and emit metric events, but we don't block the model.
// Hard-blocking off-plan calls would be brittle when the plan was
// suboptimal — the model often discovers correct work the planner
// missed.
//
// What we DO actively do: count the off-plan streak, and once it
// crosses planAutoReviseThreshold we regenerate the plan with whatever
// context the agent has discovered so far. That's the "plan_revise
// escape" — the agent doesn't have to know about a plan_revise tool;
// the loop notices the divergence and re-plans for it.
//
// Adherence rules:
//   - A tool call satisfies the FIRST unsatisfied plan step whose
//     action verb matches the tool name (read_file ↔ "read_file" or
//     "read", run_command ↔ "run_command" or "run").
//   - If the planned step has a target, we additionally require the
//     tool's path/command to mention that target. Loose substring
//     match — paths normalize to basename so /workspace/app.py and
//     ./app.py both match a step targeting "app.py".
//   - Steps are matched in order (first unsatisfied wins) so the
//     model can revisit a planned action without re-satisfying earlier
//     steps. Out-of-order is fine; off-plan is what we count.

const (
	// planAutoReviseThreshold is the number of consecutive off-plan
	// tool calls before we auto-revise the plan. Bumped 3→5 alongside
	// the recon-tool neutrality fix below: even with recon excluded,
	// 3 was firing on routine exploration patterns (the May 6 session
	// hit it twice on read_file/list_directory chains for templates
	// the model was hunting). 5 unmatched non-recon calls is a real
	// off-plan signal; 3 was thrashing on normal agent behavior.
	planAutoReviseThreshold = 5

	// planMaxRevisions caps how many times we'll regenerate per loop.
	// After this we give up and run plan-free for the remainder.
	planMaxRevisions = 2
)

// isReconTool returns true for tools that gather information without
// taking action. These calls are neutral for plan adherence — they
// neither satisfy plan steps (a plan rarely lists "read_file app.py"
// as a step) nor count as off-plan (recon between planned actions is
// expected and shouldn't burn the off-streak counter).
//
// Without this, the agent's natural "look around before changing
// anything" pattern triggered plan revisions purely from exploratory
// reads — visible in the May 6 session as 2 revisions fired purely
// from read_file/list_directory chains.
func isReconTool(name string) bool {
	switch name {
	case "read_file", "list_directory", "find_file", "search_files":
		return true
	}
	return false
}

// matchPlanStep returns the index of the first unsatisfied plan step
// that the tool call (toolName, args) satisfies, or -1 if no match.
// satisfied must be the same length as plan.Steps.
func matchPlanStep(plan *Plan, satisfied []bool, toolName string, args json.RawMessage) int {
	if plan == nil || len(plan.Steps) == 0 {
		return -1
	}
	if len(satisfied) != len(plan.Steps) {
		return -1
	}
	target := extractToolTarget(toolName, args)
	for i, step := range plan.Steps {
		if satisfied[i] {
			continue
		}
		if !actionMatchesTool(step.Action, toolName) {
			continue
		}
		// Target match is advisory — if the step has no target field
		// or the tool args don't carry an obvious target, the action
		// match alone is enough.
		if step.Target != "" && target != "" {
			if !targetsOverlap(step.Target, target) {
				continue
			}
		}
		return i
	}
	return -1
}

// actionMatchesTool reports whether step.Action describes the same
// operation as a tool call named toolName. We check both directions
// (action→tool and tool→action) and normalize underscores so plans
// written as "read file" or "read_file" both match read_file.
func actionMatchesTool(action, toolName string) bool {
	if action == "" || toolName == "" {
		return false
	}
	a := strings.ToLower(strings.ReplaceAll(action, "_", " "))
	t := strings.ToLower(strings.ReplaceAll(toolName, "_", " "))
	if a == t || strings.Contains(a, t) {
		return true
	}
	// Also allow the verb stem ("read" matches "read_file" tool).
	verb := strings.SplitN(t, " ", 2)[0]
	if verb != "" && strings.HasPrefix(a, verb) {
		return true
	}
	return false
}

// targetsOverlap reports whether two paths/targets refer to the same
// thing. For paths: equality or path-suffix match (so
// "templates/index.html" matches "/workspace/templates/index.html").
// For commands (anything with a space or non-path char): loose
// substring match so "curl http://localhost:5000/" matches a plan
// target of "curl http://localhost:5000/hello".
//
// Path-shaped strings require a path-component boundary: without it,
// "app.py" would match "tests/test_app.py" and reads of the test file
// would tick off the source-file plan step.
func targetsOverlap(planTarget, toolTarget string) bool {
	a := strings.ToLower(strings.TrimSpace(planTarget))
	b := strings.ToLower(strings.TrimSpace(toolTarget))
	if a == "" || b == "" {
		return false
	}
	if a == b {
		return true
	}
	a = strings.TrimPrefix(a, "./")
	b = strings.TrimPrefix(b, "./")
	if a == b {
		return true
	}
	// Path-suffix match: basename or last-N-components alignment.
	if strings.HasSuffix(b, "/"+a) || strings.HasSuffix(a, "/"+b) {
		return true
	}
	// Heuristic: anything with a space looks like a command rather
	// than a filename. Allow substring there so partial command
	// matches still count.
	if strings.ContainsAny(a, " \t") || strings.ContainsAny(b, " \t") {
		return strings.Contains(a, b) || strings.Contains(b, a)
	}
	return false
}

// extractToolTarget returns the most useful "target" string for a
// tool call: file path for file tools, command string for run_command,
// path for list_directory. Empty when the tool has no clear target
// (e.g. plan_revise itself).
func extractToolTarget(toolName string, args json.RawMessage) string {
	switch toolName {
	case "read_file", "delete_file":
		var x struct {
			Path string `json:"path"`
		}
		if json.Unmarshal(args, &x) == nil {
			return x.Path
		}
	case "write_file":
		var x WriteFileInput
		if json.Unmarshal(args, &x) == nil {
			return x.Path
		}
	case "edit_file":
		var x struct {
			Path string `json:"path"`
		}
		if json.Unmarshal(args, &x) == nil {
			return x.Path
		}
	case "run_command":
		var x RunCommandInput
		if json.Unmarshal(args, &x) == nil {
			return x.Command
		}
	case "list_directory":
		var x struct {
			Path string `json:"path"`
		}
		if json.Unmarshal(args, &x) == nil {
			return x.Path
		}
	}
	return ""
}

// recordPlanAdherence is called from the agent loop after each
// tool-call dispatch. It updates ctx.PlanStepsSatisfied and
// ctx.PlanOffStreak, emits a "plan_adherence" metric, and returns
// true if the off-streak crossed the auto-revise threshold (caller
// should regenerate the plan).
func recordPlanAdherence(ctx *AgentContext, toolName string, args json.RawMessage, success bool) bool {
	if ctx.Plan == nil {
		return false
	}
	if ctx.PlanStepsSatisfied == nil {
		ctx.PlanStepsSatisfied = make([]bool, len(ctx.Plan.Steps))
	}

	idx := matchPlanStep(ctx.Plan, ctx.PlanStepsSatisfied, toolName, args)

	// Only successful tool calls count toward step satisfaction.
	// A failed run_command shouldn't tick off the verify_step.
	if idx >= 0 && success {
		ctx.PlanStepsSatisfied[idx] = true
		ctx.PlanOffStreak = 0
		ctx.Stream("plan_adherence", map[string]interface{}{
			"matched":     true,
			"step_index":  idx,
			"step_id":     ctx.Plan.Steps[idx].ID,
			"step_action": ctx.Plan.Steps[idx].Action,
			"satisfied":   countTrue(ctx.PlanStepsSatisfied),
			"total":       len(ctx.PlanStepsSatisfied),
		})
		return false
	}

	// Recon tools (read_file / list_directory / find_file / search_files)
	// are neutral: they don't satisfy steps but they don't extend the
	// off-streak either. The agent's natural exploration pattern
	// shouldn't trigger plan revisions.
	if isReconTool(toolName) {
		ctx.Stream("plan_adherence", map[string]interface{}{
			"matched":    false,
			"neutral":    true,
			"tool":       toolName,
			"off_streak": ctx.PlanOffStreak, // unchanged
			"satisfied":  countTrue(ctx.PlanStepsSatisfied),
			"total":      len(ctx.PlanStepsSatisfied),
		})
		return false
	}

	// No matching step (or the call failed) — extend the off-streak.
	ctx.PlanOffStreak++
	ctx.Stream("plan_adherence", map[string]interface{}{
		"matched":    false,
		"tool":       toolName,
		"off_streak": ctx.PlanOffStreak,
		"satisfied":  countTrue(ctx.PlanStepsSatisfied),
		"total":      len(ctx.PlanStepsSatisfied),
	})

	// Threshold check — caller should auto-revise when this returns
	// true. We also cap at planMaxRevisions so a chronically
	// off-plan run doesn't loop forever calling /v3/plan.
	if ctx.PlanOffStreak >= planAutoReviseThreshold && ctx.PlanRevisions < planMaxRevisions {
		return true
	}
	return false
}

// revisePlan regenerates the plan with whatever the agent has
// discovered since the original plan was made. The user message
// passed in is the ORIGINAL one (the goal hasn't changed); we
// suffix a short note explaining why we're re-planning so the
// planner can adjust shape.
func revisePlan(ctx *AgentContext, originalUserMessage string, reason string) {
	if ctx.Plan == nil || ctx.PlanRevisions >= planMaxRevisions {
		return
	}
	// Compose a revision-aware user message. The planner prompt is
	// goal-oriented, so we keep the user's original goal verbatim
	// and append a "what we learned" note. This lets the planner
	// re-shape the plan around the new info rather than starting
	// from zero.
	noted := originalUserMessage
	if reason != "" {
		noted = fmt.Sprintf("%s\n\n[Re-planning context: %s]", originalUserMessage, reason)
	}
	log.Printf("[agent] revising plan (revision %d/%d): %s",
		ctx.PlanRevisions+1, planMaxRevisions, reason)
	ctx.Stream("plan_revise", map[string]interface{}{
		"reason":   reason,
		"revision": ctx.PlanRevisions + 1,
	})

	// Carry forward what the agent has read so far — it's the most
	// concrete signal of "what the agent knows now" beyond the
	// original priority-files sample.
	pctx := samplePlanContext(ctx.WorkingDir, 6, 2000)
	for path, content := range ctx.SnapshotFilesRead() {
		if len(pctx) >= 8 {
			break
		}
		// Use relative path if possible so the planner key matches
		// what the agent will pass to read_file/edit_file later.
		rel := path
		if strings.HasPrefix(path, ctx.WorkingDir+"/") {
			rel = strings.TrimPrefix(path, ctx.WorkingDir+"/")
		}
		s := content
		if len(s) > 2000 {
			s = s[:2000] + "\n... (truncated)"
		}
		pctx[rel] = s
	}

	req := V3PlanRequest{
		UserMessage:    noted,
		WorkingDir:     ctx.WorkingDir,
		ProjectContext: pctx,
		NCandidates:    3,
	}
	plan, err := callV3PlanStreaming(ctx.Ctx, ctx.V3URL, req, func(stage, detail string, data map[string]interface{}) {
		switch stage {
		case "token", "llm_start", "llm_end":
			return
		}
		payload := map[string]interface{}{"stage": stage, "detail": detail, "revision": ctx.PlanRevisions + 1}
		for k, v := range data {
			payload[k] = v
		}
		ctx.Stream("v3_plan", payload)
	})
	ctx.PlanRevisions++
	if err != nil || plan == nil {
		log.Printf("[agent] plan revision failed: %v — continuing with previous plan", err)
		return
	}
	ctx.Plan = plan
	ctx.PlanStepsSatisfied = make([]bool, len(plan.Steps))
	ctx.PlanOffStreak = 0

	// Re-emit the full plan structure so renderers replace the
	// previous plan view with the revised one. Same shape as the
	// initial generatePlan emission so consumers can use one code
	// path for both.
	planPayload := map[string]interface{}{
		"steps":         plan.Steps,
		"verify_step":   plan.VerifyStep,
		"rationale":     plan.Rationale,
		"winning_score": plan.WinningScore,
		"revision":      ctx.PlanRevisions,
	}
	ctx.Stream("plan_loaded", planPayload)
}

func countTrue(bs []bool) int {
	n := 0
	for _, b := range bs {
		if b {
			n++
		}
	}
	return n
}

// Plan-progress reminder injection. May 10 2026.
//
// Long multi-file tasks (e.g. "redo all 10 templates to match a SaaS
// design") lose sight of the plan once conversation trimming kicks in.
// The plan is generated up front via /v3/plan and stashed on ctx.Plan,
// and PlanStepsSatisfied tracks which steps have been hit — but neither
// surfaces back to the model after the original plan-rendering message
// drops out of the trim window.
//
// Fix: at the START of each LLM call we render a compact plan-progress
// "[system note]: ..." line and prepend it to the messages slice
// passed to callLLMOnce. The note is EPHEMERAL — it's not appended to
// ctx.Messages, so it doesn't accumulate or get re-trimmed. Every
// turn, the model sees: "step 3 of 7 — currently working on edit
// templates/dashboard.html; done: index.html, contact.html; remaining:
// pricing.html, services.html, ...".
//
// Cost: ~150 chars per turn. Cheap compared to letting the model
// re-read all the templates to remember what's done.

// buildPlanReminder returns a one-line "[system note]" string with
// plan progress, or "" if no plan is active. The caller prepends this
// to the messages slice passed to a single LLM call — it's not added
// to ctx.Messages, so it doesn't bloat history.
func buildPlanReminder(ctx *AgentContext) string {
	if ctx.Plan == nil || len(ctx.Plan.Steps) == 0 {
		return ""
	}
	if ctx.PlanStepsSatisfied == nil {
		ctx.PlanStepsSatisfied = make([]bool, len(ctx.Plan.Steps))
	}

	total := len(ctx.Plan.Steps)
	doneCount := 0
	doneIDs := make([]string, 0, total)
	remainingIDs := make([]string, 0, total)
	var current *PlanStep

	for i := range ctx.Plan.Steps {
		step := &ctx.Plan.Steps[i]
		if i < len(ctx.PlanStepsSatisfied) && ctx.PlanStepsSatisfied[i] {
			doneCount++
			doneIDs = append(doneIDs, step.ID)
		} else {
			if current == nil {
				current = step
			}
			remainingIDs = append(remainingIDs, step.ID)
		}
	}

	if current == nil {
		// All steps satisfied — the model should be on the verify step
		// or about to emit done. Surface that explicitly.
		return fmt.Sprintf(
			"[system note]: plan complete (%d/%d steps satisfied). Verify your work via `%s` if you haven't already, then emit `done` with a summary of what landed.",
			doneCount, total, planVerifyHint(ctx.Plan))
	}

	doneFrag := "none yet"
	if len(doneIDs) > 0 {
		doneFrag = strings.Join(doneIDs, ", ")
	}
	return fmt.Sprintf(
		"[system note]: plan progress %d/%d — currently on step %q (%s %s). Done: %s. Remaining: %s. Stay on the current step until it's complete; don't jump ahead and don't re-explore finished work.",
		doneCount, total, current.ID, current.Action, current.Target,
		doneFrag, strings.Join(remainingIDs, ", "),
	)
}

func planVerifyHint(p *Plan) string {
	if p == nil || p.VerifyStep == "" {
		return "the appropriate test/curl/run command"
	}
	return p.VerifyStep
}

// Asset-graph lint: cross-file coherence checks for small web projects.
// The sandbox verifies each file in isolation (compile, run, HTTP 200),
// so a project can pass every check while its files ignore each other —
// a template no route renders, a static script no page loads, an href
// to a file that doesn't exist. All three shapes appeared in the
// 2026-07-18 snake-game session. Findings are advisory text handed back
// to the model as [system note]s; nothing here blocks a write or a
// done — the stuck-pattern detectors stay the only loop-breakers.

const (
	// assetLintMaxFiles bounds the workspace walk. Past this the project
	// is no longer "small", reference search gets quadratic-ish, and a
	// framework's asset pipeline makes textual matching wrong anyway.
	assetLintMaxFiles = 400
	// assetLintMaxFileBytes skips huge files during content search.
	assetLintMaxFileBytes = 256 * 1024
)

var (
	// src/href values worth resolving as local paths. Skips externals,
	// anchors, data URIs, protocol-relative, and templated values.
	reSrcHref = regexp.MustCompile(`(?i)\b(?:src|href)\s*=\s*["']([^"']+)["']`)
	// url_for('static', filename='x.js') — Flask's canonical static ref.
	reURLFor = regexp.MustCompile(`url_for\(\s*['"]static['"]\s*,\s*filename\s*=\s*['"]([^'"]+)['"]`)
	// render_template('name.html') — referenced template must exist.
	reRenderTemplate = regexp.MustCompile(`render_template\(\s*['"]([^'"]+)['"]`)
	// {% extends "base.html" %} / {% include "nav.html" %}.
	reJinjaRef = regexp.MustCompile(`\{%\s*(?:extends|include)\s+['"]([^'"]+)['"]`)
	// fetch('/path'...) with a local absolute path (quote or backtick).
	reFetchURL = regexp.MustCompile("fetch\\(\\s*[`'\"](/[^`'\"?#{]*)")
	// <form action="/path">.
	reFormAction = regexp.MustCompile(`(?i)\baction\s*=\s*["'](/[^"'?#{]*)["']`)
	// @app.route('/path') / @bp.route(...).
	reFlaskRoute = regexp.MustCompile(`@\w+\.route\(\s*['"]([^'"]+)['"]`)
)

// assetLintFindings walks the project under workingDir and returns
// advisory findings about the template/static/reference graph. Returns
// nil for big projects (bounded walk) and on any filesystem trouble —
// this is a best-effort advisory pass, never a blocker.
func assetLintFindings(workingDir string) []string {
	type entry struct {
		rel     string
		content string
	}
	var files []entry
	count := 0
	filepath.Walk(workingDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		name := info.Name()
		if info.IsDir() {
			if strings.HasPrefix(name, ".") || name == "node_modules" ||
				name == "venv" || name == "__pycache__" {
				return filepath.SkipDir
			}
			return nil
		}
		count++
		if count > assetLintMaxFiles {
			return fmt.Errorf("project too large")
		}
		if info.Size() > assetLintMaxFileBytes {
			return nil
		}
		switch strings.ToLower(filepath.Ext(name)) {
		case ".py", ".html", ".htm", ".js", ".css", ".jinja", ".jinja2":
			data, rerr := os.ReadFile(path)
			if rerr != nil {
				return nil
			}
			rel, rerr2 := filepath.Rel(workingDir, path)
			if rerr2 != nil {
				return nil
			}
			files = append(files, entry{rel: filepath.ToSlash(rel), content: string(data)})
		}
		return nil
	})
	if count > assetLintMaxFiles {
		return nil
	}

	var findings []string
	allOther := func(self string) string {
		var b strings.Builder
		for _, f := range files {
			if f.rel != self {
				b.WriteString(f.content)
				b.WriteByte('\n')
			}
		}
		return b.String()
	}
	htmlCount := 0
	routeSet := []*regexp.Regexp{}
	routeRaw := []string{}
	for _, f := range files {
		ext := strings.ToLower(filepath.Ext(f.rel))
		if ext == ".html" || ext == ".htm" {
			htmlCount++
		}
		if ext == ".py" {
			for _, m := range reFlaskRoute.FindAllStringSubmatch(f.content, -1) {
				raw := strings.TrimSuffix(m[1], "/")
				if raw == "" {
					raw = "/"
				}
				// '<int:id>'-style segments match any single path segment.
				var b strings.Builder
				b.WriteString("^")
				for _, seg := range strings.Split(raw, "/") {
					if seg == "" {
						continue
					}
					b.WriteString("/")
					if strings.HasPrefix(seg, "<") && strings.HasSuffix(seg, ">") {
						b.WriteString("[^/]+")
					} else {
						b.WriteString(regexp.QuoteMeta(seg))
					}
				}
				if b.String() == "^" {
					b.WriteString("/")
				}
				b.WriteString("/?$")
				if re, err := regexp.Compile(b.String()); err == nil {
					routeSet = append(routeSet, re)
					routeRaw = append(routeRaw, m[1])
				}
			}
		}
	}
	routeMatches := func(target string) bool {
		t := strings.TrimSuffix(target, "/")
		if t == "" {
			t = "/"
		}
		for _, re := range routeSet {
			if re.MatchString(t) {
				return true
			}
		}
		return false
	}

	for _, f := range files {
		switch {
		case strings.HasPrefix(f.rel, "templates/"):
			base := filepath.Base(f.rel)
			if others := allOther(f.rel); !strings.Contains(others, base) {
				msg := fmt.Sprintf(
					"%s is referenced by nothing (no render_template call or include names %q).",
					f.rel, base)
				// render_template_string elsewhere is the smell that
				// pairs with an orphaned template (2026-07-18 snake
				// session: model inlined the page and orphaned both
				// the template and its static script).
				if strings.Contains(others, "render_template_string") {
					msg += " A .py file builds its page inline with render_template_string instead — either render this template or delete it."
				}
				findings = append(findings, msg)
			}
		case strings.HasPrefix(f.rel, "static/"):
			base := filepath.Base(f.rel)
			relUnderStatic := strings.TrimPrefix(f.rel, "static/")
			others := allOther(f.rel)
			if !strings.Contains(others, base) && !strings.Contains(others, relUnderStatic) {
				findings = append(findings, fmt.Sprintf(
					"%s is referenced by nothing (no <script src>, <link href>, or url_for('static', ...) names it).",
					f.rel))
			}
		default:
			// Flat-layout orphans: a .js/.css living beside .html files
			// (no templates/static dirs) is subject to the same rule —
			// three mini-bench tasks inlined a duplicate <script> and
			// orphaned the companion file, invisible to the prefix-keyed
			// rules above. Only fires when the project has HTML at all
			// (a pure node/python lib's entry file is legitimately
			// unreferenced).
			ext := strings.ToLower(filepath.Ext(f.rel))
			if (ext == ".js" || ext == ".css") && htmlCount > 0 {
				if !strings.Contains(allOther(f.rel), filepath.Base(f.rel)) {
					findings = append(findings, fmt.Sprintf(
						"%s is referenced by nothing — if a page should load it, add the <script src>/<link href>; if its content was inlined instead, delete the file.",
						f.rel))
				}
			}
		}

		// Referenced-but-missing templates: render_template('x') in .py,
		// {% extends/include %} in templates. The snake fix session
		// shipped an errorhandler rendering templates/404.html that did
		// not exist — every 404 became a 500.
		ext := strings.ToLower(filepath.Ext(f.rel))
		var tmplRefs []string
		if ext == ".py" {
			for _, m := range reRenderTemplate.FindAllStringSubmatch(f.content, -1) {
				tmplRefs = append(tmplRefs, m[1])
			}
		}
		if ext == ".html" || ext == ".htm" || ext == ".jinja" || ext == ".jinja2" {
			for _, m := range reJinjaRef.FindAllStringSubmatch(f.content, -1) {
				tmplRefs = append(tmplRefs, m[1])
			}
		}
		for _, name := range tmplRefs {
			if strings.Contains(name, "{{") {
				continue
			}
			rel := filepath.FromSlash(name)
			// A name escaping templates/ is dangling by definition (Jinja
			// loaders refuse traversal) — report it WITHOUT the Stat probe,
			// which stays contained to workingDir.
			dangling := !filepath.IsLocal(rel)
			if !dangling {
				_, err := os.Stat(filepath.Join(workingDir, "templates", rel))
				dangling = err != nil
			}
			if dangling {
				findings = append(findings, fmt.Sprintf(
					"%s references template %q, but templates/%s does not exist.",
					f.rel, name, name))
			}
		}

		// Route-contract check: fetch()/form-action URLs must correspond
		// to a declared Flask route. Mini-bench t01 generated a JS
		// frontend calling REST endpoints in a style the backend half
		// implemented differently — page loads, halves can't talk.
		if len(routeSet) > 0 && (ext == ".js" || ext == ".html" || ext == ".htm") {
			seen := map[string]bool{}
			for _, re := range []*regexp.Regexp{reFetchURL, reFormAction} {
				for _, m := range re.FindAllStringSubmatch(f.content, -1) {
					target := m[1]
					if seen[target] || strings.HasPrefix(target, "/static/") {
						continue
					}
					seen[target] = true
					if !routeMatches(target) {
						findings = append(findings, fmt.Sprintf(
							"%s calls %q, but no Flask route matches it (routes: %s).",
							f.rel, target, strings.Join(routeRaw, ", ")))
					}
				}
			}
		}
	}

	// Dangling local references: src/href/url_for pointing at files that
	// don't exist in the workspace.
	seenDangling := map[string]bool{}
	for _, f := range files {
		for _, m := range reSrcHref.FindAllStringSubmatch(f.content, -1) {
			target := m[1]
			if strings.Contains(target, "://") || strings.HasPrefix(target, "//") ||
				strings.HasPrefix(target, "#") || strings.HasPrefix(target, "data:") ||
				strings.HasPrefix(target, "mailto:") || strings.Contains(target, "{{") ||
				strings.Contains(target, "{%") {
				continue
			}
			target = strings.SplitN(target, "?", 2)[0]
			target = strings.SplitN(target, "#", 2)[0]
			if target == "" || target == "/" {
				continue
			}
			rel := filepath.FromSlash(strings.TrimPrefix(target, "/"))
			// A target escaping the workspace can't be served from it —
			// report as dangling without the Stat probe (contained to
			// workingDir).
			dangling := !filepath.IsLocal(rel)
			if !dangling {
				_, err := os.Stat(filepath.Join(workingDir, rel))
				dangling = err != nil
			}
			if dangling {
				key := f.rel + "→" + target
				if !seenDangling[key] {
					seenDangling[key] = true
					findings = append(findings, fmt.Sprintf(
						"%s references %q, which does not exist in the workspace.", f.rel, target))
				}
			}
		}
		for _, m := range reURLFor.FindAllStringSubmatch(f.content, -1) {
			rel := filepath.FromSlash(m[1])
			// A filename escaping static/ 404s at runtime (Flask refuses
			// traversal) — report as dangling without the Stat probe.
			dangling := !filepath.IsLocal(rel)
			if !dangling {
				_, err := os.Stat(filepath.Join(workingDir, filepath.Join("static", rel)))
				dangling = err != nil
			}
			if dangling {
				key := f.rel + "→" + m[1]
				if !seenDangling[key] {
					seenDangling[key] = true
					findings = append(findings, fmt.Sprintf(
						"%s references url_for('static', filename=%q), but static/%s does not exist.",
						f.rel, m[1], m[1]))
				}
			}
		}
	}

	sort.Strings(findings)
	return findings
}

// assetLintNote runs the lint and formats findings the model has not
// seen yet as one [system note] body ("" when quiet). Dedup state lives
// in ctx.AssetLintSeen so a persistent orphan is mentioned once, not
// after every subsequent write.
func assetLintNote(ctx *AgentContext) string {
	findings := assetLintFindings(ctx.WorkingDir)
	if len(findings) == 0 {
		return ""
	}
	if ctx.AssetLintSeen == nil {
		ctx.AssetLintSeen = make(map[string]bool)
	}
	var fresh []string
	for _, f := range findings {
		if !ctx.AssetLintSeen[f] {
			ctx.AssetLintSeen[f] = true
			fresh = append(fresh, f)
		}
	}
	if len(fresh) == 0 {
		return ""
	}
	return "Project structure check: " + strings.Join(fresh, " ") +
		" This is advisory — fix it if these files are meant to work together."
}
