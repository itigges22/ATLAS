package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestClaimsUniversalCatchesGlobalAssertions(t *testing.T) {
	yes := []string{
		"All routes are functioning properly.",
		"Fixed all bugs in the routing layer.",
		"Everything works as expected.",
		"No errors remaining.",
		"Tested all endpoints — all green.",
		"App is fully functional.",
	}
	for _, s := range yes {
		if !claimsUniversal(s) {
			t.Errorf("claimsUniversal(%q) = false, want true", s)
		}
	}
	no := []string{
		"Added /admin route. Run the test suite to confirm the rest still works.",
		"Created the missing template for /pricing.",
		"Updated the readme.",
		"",
	}
	for _, s := range no {
		if claimsUniversal(s) {
			t.Errorf("claimsUniversal(%q) = true, want false", s)
		}
	}
}

func TestVerifyCompletionClaimsCatchesMissingFlaskTemplates(t *testing.T) {
	dir := t.TempDir()
	// app.py references 4 templates; only index.html exists.
	app := `from flask import Flask, render_template
app = Flask(__name__)
@app.route('/')
def index(): return render_template('index.html')
@app.route('/pricing')
def pricing(): return render_template('pricing.html')
@app.route('/contact')
def contact(): return render_template('contact.html')
@app.route('/admin')
def admin(): return render_template('admin.html')
`
	if err := os.WriteFile(filepath.Join(dir, "app.py"), []byte(app), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "templates"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "templates", "index.html"), []byte("ok"), 0o644); err != nil {
		t.Fatal(err)
	}

	got := verifyCompletionClaims(dir, "All routes are functioning properly.")
	if got == "" {
		t.Fatal("expected gap report, got empty")
	}
	for _, want := range []string{"pricing.html", "contact.html", "admin.html"} {
		if !strings.Contains(got, want) {
			t.Errorf("gap report missing %q\n%s", want, got)
		}
	}
	if strings.Contains(got, "index.html") {
		t.Errorf("gap report includes index.html (which exists):\n%s", got)
	}
}

func TestVerifyCompletionClaimsAllPresentReturnsEmpty(t *testing.T) {
	dir := t.TempDir()
	app := `from flask import Flask, render_template
app = Flask(__name__)
@app.route('/')
def index(): return render_template('index.html')
`
	os.WriteFile(filepath.Join(dir, "app.py"), []byte(app), 0o644)
	os.MkdirAll(filepath.Join(dir, "templates"), 0o755)
	os.WriteFile(filepath.Join(dir, "templates", "index.html"), []byte("ok"), 0o644)

	if got := verifyCompletionClaims(dir, "All routes work."); got != "" {
		t.Errorf("expected empty (no gaps), got: %s", got)
	}
}

func TestVerifyCompletionClaimsExpressMissingViews(t *testing.T) {
	dir := t.TempDir()
	srv := `const express = require('express');
const app = express();
app.get('/', (req, res) => res.render('home'));
app.get('/about', (req, res) => res.render('about'));
`
	os.WriteFile(filepath.Join(dir, "server.js"), []byte(srv), 0o644)
	os.MkdirAll(filepath.Join(dir, "views"), 0o755)
	os.WriteFile(filepath.Join(dir, "views", "home.ejs"), []byte("<%= 1 %>"), 0o644)
	// about.* missing

	got := verifyCompletionClaims(dir, "All endpoints are working.")
	if !strings.Contains(got, "about") {
		t.Errorf("expected `about` in gap, got: %s", got)
	}
}

func TestVerifyCompletionClaimsSkipsNoiseDirs(t *testing.T) {
	dir := t.TempDir()
	// A render_template inside venv/ should NOT be parsed.
	os.MkdirAll(filepath.Join(dir, "venv", "lib"), 0o755)
	os.WriteFile(filepath.Join(dir, "venv", "lib", "junk.py"),
		[]byte(`render_template('ghost.html')`), 0o644)
	if got := verifyCompletionClaims(dir, "All routes work."); got != "" {
		t.Errorf("noise dir tripped check: %s", got)
	}
}

func TestVerifyCompletionClaimsHandlesDynamicRenderArgs(t *testing.T) {
	// `render_template(name)` — variable arg, can't statically check.
	// Should NOT produce a gap report.
	dir := t.TempDir()
	app := `from flask import render_template
def view(name): return render_template(name)
`
	os.WriteFile(filepath.Join(dir, "app.py"), []byte(app), 0o644)
	if got := verifyCompletionClaims(dir, "All routes work."); got != "" {
		t.Errorf("dynamic render tripped check: %s", got)
	}
}
