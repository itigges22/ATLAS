package main

// Asset-graph lint: cross-file coherence checks for small web projects.
// The sandbox verifies each file in isolation (compile, run, HTTP 200),
// so a project can pass every check while its files ignore each other —
// a template no route renders, a static script no page loads, an href
// to a file that doesn't exist. All three shapes appeared in the
// 2026-07-18 snake-game session. Findings are advisory text handed back
// to the model as [system note]s; nothing here blocks a write or a
// done — the stuck-pattern detectors stay the only loop-breakers.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

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
	// render_template_string with a sizeable inline literal — the smell
	// that pairs with an orphaned template.
	reInlineTemplate = regexp.MustCompile(`render_template_string\s*\(`)
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
	hasInlineTemplateUse := false
	for _, f := range files {
		if strings.HasSuffix(f.rel, ".py") && reInlineTemplate.MatchString(f.content) {
			hasInlineTemplateUse = true
			break
		}
	}

	for _, f := range files {
		switch {
		case strings.HasPrefix(f.rel, "templates/"):
			base := filepath.Base(f.rel)
			if !strings.Contains(allOther(f.rel), base) {
				msg := fmt.Sprintf(
					"%s is referenced by nothing (no render_template call or include names %q).",
					f.rel, base)
				if hasInlineTemplateUse {
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
			candidate := strings.TrimPrefix(target, "/")
			if _, err := os.Stat(filepath.Join(workingDir, filepath.FromSlash(candidate))); err != nil {
				key := f.rel + "→" + target
				if !seenDangling[key] {
					seenDangling[key] = true
					findings = append(findings, fmt.Sprintf(
						"%s references %q, which does not exist in the workspace.", f.rel, target))
				}
			}
		}
		for _, m := range reURLFor.FindAllStringSubmatch(f.content, -1) {
			candidate := filepath.Join("static", filepath.FromSlash(m[1]))
			if _, err := os.Stat(filepath.Join(workingDir, candidate)); err != nil {
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
