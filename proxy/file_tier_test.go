// File-tier classifier tests. Covers the post-inversion behaviour where
// V3 should fire much more often: small-but-routed code files (flask,
// express) used to slip through to T1 because hasLogicIndicators needed
// 3+ patterns and the <50-line short-circuit fired first.

package main

import "testing"

func TestClassifyFileTierFlaskAppPyIsT2(t *testing.T) {
	// 33-line flask routing module — exactly the file the user was
	// debugging when V3 never fired. Must be T2 now.
	content := `from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/product')
def product():
    return render_template('product.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
`
	if got := classifyFileTier("app.py", content); got < Tier2Medium {
		t.Errorf("flask app.py = %v, want >= T2 (was T1 under old <50-line rule)", got)
	}
}

func TestClassifyFileTierTinyConfigStaysT1(t *testing.T) {
	// Single-line .gitignore / 5-line shell wrapper — V3 has nothing to
	// diversify on, must stay T1 to avoid wasted pipeline cost.
	if got := classifyFileTier(".gitignore", "node_modules\n.env\n"); got != Tier1Simple {
		t.Errorf(".gitignore = %v, want T1", got)
	}
	if got := classifyFileTier("foo.sh", "#!/bin/sh\nexec node app.js\n"); got != Tier1Simple {
		t.Errorf("tiny shell script = %v, want T1", got)
	}
}

func TestClassifyFileTierConfigByName(t *testing.T) {
	// Config files by name are always T1 regardless of line count.
	configFiles := []string{"package.json", "go.mod", "Dockerfile", "requirements.txt"}
	for _, f := range configFiles {
		if got := classifyFileTier(f, "x\n\n\n\n\n\n\n\n\n\n\n\n\n"); got != Tier1Simple {
			t.Errorf("classifyFileTier(%q) = %v, want T1", f, got)
		}
	}
}

func TestClassifyFileTierCodeExtBenefitOfDoubt(t *testing.T) {
	// 15-line python file with no recognisable logic patterns — naming
	// it .py is enough to get T2 now, because V3 helps with code shape
	// even when the model didn't tag any specific framework idiom.
	content := `# generated module
NAME = "atlas"
VERSION = "0.1.0"
AUTHOR = "team"
LICENSE = "MIT"
DESCRIPTION = "demo"
KEYWORDS = ["a", "b"]
EXTRAS = {}
DEPS = []
DEV_DEPS = []
URL = "https://example.com"
HOMEPAGE = URL
DOWNLOADS = URL + "/dl"
`
	if got := classifyFileTier("constants.py", content); got != Tier2Medium {
		t.Errorf("constants.py = %v, want T2 (code-ext fallback)", got)
	}
}

func TestClassifyFileTierTinyCodeFileStaysT1(t *testing.T) {
	// 4 lines — below the new 10-line floor, even .py is T1.
	if got := classifyFileTier("hello.py", "print('hi')\n"); got != Tier1Simple {
		t.Errorf("1-line script = %v, want T1", got)
	}
}

func TestClassifyFileTierMidSizedHtmlIsT2(t *testing.T) {
	// 90-line flask template — used to fall through to T1 because the
	// markup branch only fired at >=150 lines. Now .html is in codeExts
	// so V3 fires on real templates instead of only on huge mockups.
	content := ""
	for i := 0; i < 90; i++ {
		content += "<p>row " + string(rune('a'+i%26)) + "</p>\n"
	}
	if got := classifyFileTier("templates/index.html", content); got != Tier2Medium {
		t.Errorf("90-line index.html = %v, want T2 (was T1 under <150-line markup veto)", got)
	}
}

func TestHasLogicIndicatorsFlaskAppHits(t *testing.T) {
	// Verifies the threshold drop + flask-pattern additions. A flask
	// app.py used to register only "def " (1 indicator); now the route
	// decorators count, putting it well over the new threshold of 2.
	content := `from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
`
	if !hasLogicIndicators(content) {
		t.Errorf("hasLogicIndicators(flask snippet) = false, want true")
	}
}
