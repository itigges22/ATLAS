package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTree(t *testing.T, root string, files map[string]string) {
	t.Helper()
	for rel, content := range files {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func findingsContaining(findings []string, substr string) bool {
	for _, f := range findings {
		if strings.Contains(f, substr) {
			return true
		}
	}
	return false
}

func TestAssetLintFlagsSnakeGameShape(t *testing.T) {
	// The 2026-07-18 session verbatim: template and static script
	// written, then an app.py that inlines everything and references
	// neither.
	root := t.TempDir()
	writeTree(t, root, map[string]string{
		"app.py": "from flask import Flask, render_template_string\n" +
			"app = Flask(__name__)\n" +
			"@app.route('/')\ndef index():\n" +
			"    return render_template_string(\"<html>inline</html>\")\n",
		"templates/index.html": "<html><body>snake</body></html>",
		"static/game.js":       "console.log('game');",
	})
	findings := assetLintFindings(root)
	if !findingsContaining(findings, "templates/index.html is referenced by nothing") {
		t.Fatalf("orphan template not flagged: %v", findings)
	}
	if !findingsContaining(findings, "render_template_string") {
		t.Fatalf("inline-template hint missing: %v", findings)
	}
	if !findingsContaining(findings, "static/game.js is referenced by nothing") {
		t.Fatalf("orphan static not flagged: %v", findings)
	}
}

func TestAssetLintQuietOnWiredProject(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{
		"app.py": "from flask import Flask, render_template\n" +
			"@app.route('/')\ndef index():\n" +
			"    return render_template('index.html')\n",
		"templates/index.html": "<html><script src=\"/static/game.js\"></script></html>",
		"static/game.js":       "console.log('game');",
	})
	if findings := assetLintFindings(root); len(findings) != 0 {
		t.Fatalf("wired project should be quiet, got: %v", findings)
	}
}

func TestAssetLintDanglingReferences(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{
		"templates/index.html": "<html>" +
			"<script src=\"/static/missing.js\"></script>" +
			"<link href=\"https://cdn.example.com/x.css\">" + // external: skip
			"<a href=\"#top\">top</a>" + // anchor: skip
			"<img src=\"{{ asset_path }}\">" + // unresolvable templated value: skip
			"</html>",
		"app.py": "from flask import render_template\n" +
			"render_template('index.html')\n" +
			"x = url_for('static', filename='also-missing.css')\n",
	})
	findings := assetLintFindings(root)
	if !findingsContaining(findings, `"/static/missing.js"`) {
		t.Fatalf("dangling src not flagged: %v", findings)
	}
	if !findingsContaining(findings, "also-missing.css") {
		t.Fatalf("dangling url_for not flagged: %v", findings)
	}
	for _, f := range findings {
		if strings.Contains(f, "cdn.example.com") || strings.Contains(f, "#top") ||
			strings.Contains(f, "asset_path") {
			t.Fatalf("external/anchor/templated ref flagged: %q", f)
		}
	}
}

func TestAssetLintSkipsLargeProjects(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{"templates/orphan.html": "<html></html>"}
	for i := 0; i < assetLintMaxFiles+5; i++ {
		files[fmt.Sprintf("pkg/f%d.py", i)] = "x = 1\n"
	}
	writeTree(t, root, files)
	if findings := assetLintFindings(root); findings != nil {
		t.Fatalf("large project must be skipped, got: %v", findings)
	}
}

func TestAssetLintNoteDedupes(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{
		"templates/index.html": "<html></html>",
		"app.py":               "print('no render')\n",
	})
	ctx := &AgentContext{WorkingDir: root}
	first := assetLintNote(ctx)
	if first == "" || !strings.Contains(first, "templates/index.html") {
		t.Fatalf("first note should carry the finding: %q", first)
	}
	if again := assetLintNote(ctx); again != "" {
		t.Fatalf("unchanged findings must not repeat, got: %q", again)
	}
}

func TestSessionManifestNoteAnnouncesOncePerFile(t *testing.T) {
	ctx := &AgentContext{SessionWrites: map[string]bool{"app.py": true}}
	if note := sessionManifestNote(ctx); note != "" {
		t.Fatalf("single file needs no manifest, got %q", note)
	}
	ctx.SessionWrites["templates/index.html"] = true
	note := sessionManifestNote(ctx)
	if !strings.Contains(note, "app.py") || !strings.Contains(note, "templates/index.html") {
		t.Fatalf("manifest should list both files: %q", note)
	}
	if again := sessionManifestNote(ctx); again != "" {
		t.Fatalf("no new files → no repeat, got %q", again)
	}
	ctx.SessionWrites["static/game.js"] = true
	if third := sessionManifestNote(ctx); !strings.Contains(third, "static/game.js") {
		t.Fatalf("new file should re-announce full set: %q", third)
	}
}

func TestAssetLintFlatLayoutOrphan(t *testing.T) {
	// Mini-bench t03/t07: flat layout (no templates/static), html inlines a
	// duplicate <script> and orphans the companion .js.
	root := t.TempDir()
	writeTree(t, root, map[string]string{
		"index.html": "<html><script>function calculate(){}</script></html>",
		"calc.js":    "function calculate(expression) { return 1; }",
		"style.css":  "body { margin: 0; }",
	})
	findings := assetLintFindings(root)
	if !findingsContaining(findings, "calc.js is referenced by nothing") {
		t.Fatalf("flat orphan js not flagged: %v", findings)
	}
	if !findingsContaining(findings, "style.css is referenced by nothing") {
		t.Fatalf("flat orphan css not flagged: %v", findings)
	}
}

func TestAssetLintFlatLayoutQuietWhenWiredOrNoHTML(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{
		"index.html": "<html><script src=\"calc.js\"></script><link href=\"style.css\"></html>",
		"calc.js":    "function calculate(){}",
		"style.css":  "body{}",
	})
	if f := assetLintFindings(root); len(f) != 0 {
		t.Fatalf("wired flat project should be quiet: %v", f)
	}
	// A pure library with no HTML must not flag its entry file.
	root2 := t.TempDir()
	writeTree(t, root2, map[string]string{"index.js": "module.exports = 1;"})
	if f := assetLintFindings(root2); len(f) != 0 {
		t.Fatalf("no-HTML project should be quiet: %v", f)
	}
}

func TestAssetLintMissingTemplateReference(t *testing.T) {
	// Snake fix session: errorhandler renders templates/404.html which
	// doesn't exist — every 404 became a 500.
	root := t.TempDir()
	writeTree(t, root, map[string]string{
		"app.py": "from flask import Flask, render_template\n" +
			"@app.route('/')\ndef i(): return render_template('index.html')\n" +
			"@app.errorhandler(404)\ndef nf(e): return render_template('404.html'), 404\n",
		"templates/index.html": "<html>{% extends \"base.html\" %}</html>",
	})
	findings := assetLintFindings(root)
	if !findingsContaining(findings, `references template "404.html"`) {
		t.Fatalf("missing py-referenced template not flagged: %v", findings)
	}
	if !findingsContaining(findings, `references template "base.html"`) {
		t.Fatalf("missing jinja extends target not flagged: %v", findings)
	}
}

func TestAssetLintRouteContractMismatch(t *testing.T) {
	// Mini-bench t01: JS calls REST endpoints the backend half never
	// declared; page loads, halves can't talk.
	root := t.TempDir()
	writeTree(t, root, map[string]string{
		"app.py": "from flask import Flask, render_template\n" +
			"@app.route('/')\ndef i(): return render_template('index.html')\n" +
			"@app.route('/add', methods=['POST'])\ndef a(): return ''\n" +
			"@app.route('/delete/<int:todo_id>', methods=['POST'])\ndef d(todo_id): return ''\n",
		"templates/index.html": "<html><script src=\"/static/todo.js\"></script>" +
			"<form action=\"/add\"></form></html>",
		"static/todo.js": "fetch('/api/todos');\nfetch(`/delete/${id}`, {method:'POST'});\n",
	})
	findings := assetLintFindings(root)
	if !findingsContaining(findings, `"/api/todos"`) {
		t.Fatalf("unrouted fetch target not flagged: %v", findings)
	}
	for _, f := range findings {
		if strings.Contains(f, `"/add"`) || strings.Contains(f, `"/delete/`) {
			t.Fatalf("legitimately routed target flagged: %q", f)
		}
	}
}
