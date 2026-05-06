// Completion-claim verification (PC-197).
//
// Background: the agent's done.summary often makes universal claims
// ("all routes work", "fixed all bugs", "verified everything") that
// we can structurally check against the workspace. The May 2026 flask
// run had the model claim "All routes are functioning properly" while
// only 3 of 7 needed templates existed. The verification gate (PC-179)
// only checks "did you run a verification command at all?" — it does
// NOT check whether the claim in the summary matches reality.
//
// Two-stage filter:
//   1. claimsUniversal(summary) — does the wording make a global
//      assertion? Quiet pass for narrow done summaries ("added /admin
//      route" — model said nothing about the rest of the app).
//   2. verifyCompletionClaims(workingDir, summary) — cheap structural
//      checks for the failure modes we know about. Returns a directive
//      to the model when a gap is found, "" otherwise.
//
// Conservative on false positives. Universal claims with no gap pass
// silently; narrow claims pass even when there ARE gaps elsewhere.
// Only the AND case (claim + gap) bounces. The model can override by
// using narrower wording or by calling out the gap explicitly.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

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

// verifyCompletionClaims runs cheap structural checks against
// workingDir and returns a non-empty directive when the model's
// universal claim doesn't match reality. The directive is shaped as
// a tool-result error, so it lands back in the model's context as
// "your done was bounced because X."
//
// Checks (additive — first gap wins, but caller sees one combined
// message when multiple are flagged):
//   - Flask/Jinja template references (render_template('X')) → templates/X exists?
//   - Django render() / get_template() → matching template exists?
//   - Express render() / res.render('X') → views/X exists?
//
// Each check is gated on the file types it's relevant to (no
// node_modules walk for a Python-only project, etc.). Bounded walk
// depth and per-file size limits so we don't spend the user's
// session-end latency on a 50-MB monorepo scan.
func verifyCompletionClaims(workingDir, summary string) string {
	if workingDir == "" {
		return ""
	}
	gaps := []string{}

	if g := checkTemplateReferences(workingDir); g != "" {
		gaps = append(gaps, g)
	}

	if len(gaps) == 0 {
		return ""
	}
	return fmt.Sprintf(
		"Your `done` summary claims the work is complete, but a structural check of the workspace found gaps:\n\n%s\n\nFix the missing files (or correct your summary to acknowledge what's not done) before declaring done.",
		strings.Join(gaps, "\n\n"))
}

// renderTemplateRe matches `render_template('X.html')` in Python
// (Flask/Jinja). The pattern handles single quotes, double quotes,
// and an optional whitespace inside the parens. Captures the template
// path. Doesn't try to handle dynamic args (`render_template(name)`)
// — those can't be statically verified.
var renderTemplateRe = regexp.MustCompile(
	`render_template\(\s*['"]([^'"\n]+)['"]`)

// expressRenderRe matches `res.render('view')` / `res.render("view")`
// in Express/Node. The view name might or might not include an
// extension; we check the full path AND extensionless variants.
var expressRenderRe = regexp.MustCompile(
	`\bres\.render\(\s*['"]([^'"\n]+)['"]`)

// checkTemplateReferences walks workingDir for source files,
// extracts template references, and reports any that don't resolve
// to a file under templates/ (Flask) or views/ (Express). Bounded
// to ~200 source files and ignores noise dirs.
func checkTemplateReferences(workingDir string) string {
	skipDirs := map[string]bool{
		"venv": true, ".venv": true, "env": true, "node_modules": true,
		".git": true, "__pycache__": true, "dist": true, "build": true,
		"target": true, "vendor": true, ".dart_tool": true,
	}
	const maxFiles = 200
	const maxBytes = 64 * 1024 // 64K per file is plenty for top-level grep

	var missingFlask []missing
	var missingExpress []missing

	count := 0
	err := filepath.Walk(workingDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			if skipDirs[info.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if count >= maxFiles {
			return filepath.SkipDir
		}
		ext := strings.ToLower(filepath.Ext(path))
		var re *regexp.Regexp
		var resolveTo []string
		switch ext {
		case ".py":
			re = renderTemplateRe
			// Flask resolves render_template against templates/ AND
			// any blueprint-registered template_folder. We only
			// check the conventional templates/ here — false negatives
			// in custom layouts are acceptable.
			resolveTo = []string{filepath.Join(workingDir, "templates")}
		case ".js", ".ts", ".mjs", ".cjs":
			re = expressRenderRe
			// Express defaults views/ for app.set('views', ...).
			resolveTo = []string{filepath.Join(workingDir, "views")}
		default:
			return nil
		}
		count++
		// Read up to maxBytes — render_template calls usually appear
		// in the first KB of small route files anyway.
		f, err := os.Open(path)
		if err != nil {
			return nil
		}
		defer f.Close()
		buf := make([]byte, maxBytes)
		n, _ := f.Read(buf)
		body := string(buf[:n])
		matches := re.FindAllStringSubmatch(body, -1)
		for _, m := range matches {
			tmpl := m[1]
			// Empty or path-traversal-y references — skip.
			if tmpl == "" || strings.Contains(tmpl, "..") {
				continue
			}
			if !templateResolves(tmpl, resolveTo, ext) {
				rel, _ := filepath.Rel(workingDir, path)
				rec := missing{ref: tmpl, from: rel}
				if ext == ".py" {
					missingFlask = append(missingFlask, rec)
				} else {
					missingExpress = append(missingExpress, rec)
				}
			}
		}
		return nil
	})
	if err != nil {
		return ""
	}

	var lines []string
	if len(missingFlask) > 0 {
		lines = append(lines,
			"**Flask templates referenced but missing:** "+formatMissing(missingFlask, "templates/"))
	}
	if len(missingExpress) > 0 {
		lines = append(lines,
			"**Express views referenced but missing:** "+formatMissing(missingExpress, "views/"))
	}
	return strings.Join(lines, "\n")
}

// templateResolves returns true when tmpl exists under any of the
// candidate dirs. For JS/TS we also try common view-engine extensions
// (.ejs, .pug, .hbs) when tmpl has none.
func templateResolves(tmpl string, candidates []string, srcExt string) bool {
	probe := func(p string) bool {
		_, err := os.Stat(p)
		return err == nil
	}
	for _, dir := range candidates {
		if probe(filepath.Join(dir, tmpl)) {
			return true
		}
		// JS view engines often elide the extension.
		if srcExt != ".py" && filepath.Ext(tmpl) == "" {
			for _, ext := range []string{".ejs", ".pug", ".hbs", ".html"} {
				if probe(filepath.Join(dir, tmpl+ext)) {
					return true
				}
			}
		}
	}
	return false
}

// formatMissing builds a compact list "<dir>X (in <src>), <dir>Y (in <src>)"
// for the rejection message. Capped at 8 to keep the prompt back-pressure
// digestible.
func formatMissing(items []missing, dirPrefix string) string {
	if len(items) > 8 {
		items = items[:8]
	}
	parts := make([]string, len(items))
	for i, m := range items {
		parts[i] = fmt.Sprintf("`%s%s` (referenced in %s)", dirPrefix, m.ref, m.from)
	}
	return strings.Join(parts, ", ")
}

// missing is a single template/view reference that doesn't resolve.
// Local to this file because no other check produces the same shape.
type missing struct {
	ref  string
	from string
}
