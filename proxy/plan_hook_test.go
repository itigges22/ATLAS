package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSamplePlanContextPicksUpPriorityFiles(t *testing.T) {
	dir := t.TempDir()
	// Lay down a typical flask app shape.
	if err := os.WriteFile(filepath.Join(dir, "app.py"),
		[]byte("from flask import Flask\napp = Flask(__name__)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "templates"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "templates", "index.html"),
		[]byte("<html><body>hi</body></html>"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A noisy unrelated file that shouldn't appear.
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"),
		[]byte("unrelated"), 0o644); err != nil {
		t.Fatal(err)
	}

	got := samplePlanContext(dir, 6, 2000)
	if _, ok := got["app.py"]; !ok {
		t.Errorf("expected app.py in context, got keys %v", keys(got))
	}
	if _, ok := got["templates/index.html"]; !ok {
		t.Errorf("expected templates/index.html in context, got keys %v", keys(got))
	}
	if _, ok := got["notes.txt"]; ok {
		t.Errorf("notes.txt leaked into priority context")
	}
}

func TestSamplePlanContextTruncatesLargeFiles(t *testing.T) {
	// File between maxBytes (1000) and the hard-skip ceiling (4×maxBytes
	// = 4000) should pass the size gate and get truncated. Files above
	// 4000 are skipped wholesale to avoid yanking a 50KB README into
	// the planner.
	dir := t.TempDir()
	big := make([]byte, 3000)
	for i := range big {
		big[i] = 'a'
	}
	if err := os.WriteFile(filepath.Join(dir, "main.py"), big, 0o644); err != nil {
		t.Fatal(err)
	}

	got := samplePlanContext(dir, 6, 1000)
	content, ok := got["main.py"]
	if !ok {
		t.Fatal("main.py missing from sampled context")
	}
	// 1000 bytes of body + "\n... (truncated)" marker.
	if len(content) > 1100 {
		t.Errorf("content %d bytes — sampler should truncate to ~1000", len(content))
	}
	if len(content) < 1000 {
		t.Errorf("content %d bytes — sampler shouldn't truncate below maxBytes", len(content))
	}
}

func TestSamplePlanContextSkipsHugeFiles(t *testing.T) {
	// Files >4×maxBytes are skipped wholesale to keep the planner
	// budget small. Verifies the hard-skip ceiling.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "app.py"),
		[]byte("from flask import Flask\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	huge := make([]byte, 10_000)
	if err := os.WriteFile(filepath.Join(dir, "README.md"), huge, 0o644); err != nil {
		t.Fatal(err)
	}

	got := samplePlanContext(dir, 6, 1000)
	if _, ok := got["README.md"]; ok {
		t.Errorf("10KB README should be skipped at maxBytes=1000")
	}
	if _, ok := got["app.py"]; !ok {
		t.Errorf("small app.py should still be picked up")
	}
}

func TestSamplePlanContextFallsBackToShallowWalk(t *testing.T) {
	// No priority files — sampler should still pick up source files
	// from a shallow read of the working dir.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "weird_entry.go"),
		[]byte("package main\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ignored.dat"),
		[]byte("binary"), 0o644); err != nil {
		t.Fatal(err)
	}

	got := samplePlanContext(dir, 6, 2000)
	if _, ok := got["weird_entry.go"]; !ok {
		t.Errorf("expected weird_entry.go in fallback walk, got %v", keys(got))
	}
	if _, ok := got["ignored.dat"]; ok {
		t.Errorf(".dat file leaked through extension filter")
	}
}

func TestSamplePlanContextEmptyOnMissingDir(t *testing.T) {
	got := samplePlanContext("", 5, 1000)
	if got != nil {
		t.Errorf("expected nil for empty workingDir, got %v", got)
	}
}

func TestSamplePlanContextWalksSubdirsForPriorityFiles(t *testing.T) {
	// May 2026 user case: workspace root has no app.py, but a
	// snake/ subdir does. Sampler should pick up snake/app.py with
	// the path keyed as "snake/app.py" so the planner emits tool
	// calls using that exact path.
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "snake", "templates"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "snake", "app.py"),
		[]byte("from flask import Flask\napp=Flask(__name__)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "snake", "templates", "index.html"),
		[]byte("<html>hi</html>"), 0o644); err != nil {
		t.Fatal(err)
	}

	got := samplePlanContext(dir, 6, 2000)
	if _, ok := got["snake/app.py"]; !ok {
		t.Errorf("expected snake/app.py via subdir walk, got keys %v", keys(got))
	}
	if _, ok := got["snake/templates/index.html"]; !ok {
		t.Errorf("expected snake/templates/index.html via subdir walk, got keys %v", keys(got))
	}
}

func TestSamplePlanContextSkipsNoiseDirs(t *testing.T) {
	// venv/ and node_modules/ shouldn't be walked even if they
	// contain a priority filename — these are cache/vendor dirs.
	dir := t.TempDir()
	for _, junk := range []string{"venv", "node_modules", ".git", "__pycache__"} {
		if err := os.MkdirAll(filepath.Join(dir, junk), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, junk, "app.py"),
			[]byte("# noise"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got := samplePlanContext(dir, 6, 2000)
	for _, junk := range []string{"venv/app.py", "node_modules/app.py", ".git/app.py", "__pycache__/app.py"} {
		if _, ok := got[junk]; ok {
			t.Errorf("noise path %q leaked into context", junk)
		}
	}
}

func TestShouldGeneratePlanGates(t *testing.T) {
	cases := []struct {
		name string
		tier Tier
		msg  string
		want bool
	}{
		{"T0 trivial chat", Tier0Conversational, "thanks man", false},
		{"short ack", Tier2Medium, "yes", false},
		{"borderline short", Tier2Medium, "fix it", false}, // 6 chars
		{"real fix request", Tier2Medium, "fix the index.html template", true},
		{"feature add", Tier2Medium, "add a /hello route to app.py", true},
		{"T3 architectural", Tier3Hard, "build a flask app with auth and a database", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := &AgentContext{Tier: tc.tier}
			if got := shouldGeneratePlan(ctx, tc.msg); got != tc.want {
				t.Errorf("shouldGeneratePlan(%q, tier=%v) = %v, want %v",
					tc.msg, tc.tier, got, tc.want)
			}
		})
	}
}

func keys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func TestDetectProjectVenvPythonFindsCommonShapes(t *testing.T) {
	cases := []struct {
		name    string
		layout  []string // relative paths to create as files
		wantRel string   // expected venv-python relative path
	}{
		{"venv/bin/python", []string{"venv/bin/python"}, "venv/bin/python"},
		{".venv/bin/python", []string{".venv/bin/python"}, ".venv/bin/python"},
		{"env/bin/python3", []string{"env/bin/python3"}, "env/bin/python3"},
		{"prefers-venv-over-.venv", []string{"venv/bin/python", ".venv/bin/python"}, "venv/bin/python"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			for _, rel := range tc.layout {
				abs := filepath.Join(dir, rel)
				if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(abs, []byte("#!/bin/sh\n"), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			got := detectProjectVenvPython(dir)
			want := filepath.Join(dir, tc.wantRel)
			if got != want {
				t.Errorf("detectProjectVenvPython() = %q, want %q", got, want)
			}
		})
	}
}

func TestDetectProjectVenvPythonReturnsEmptyWhenAbsent(t *testing.T) {
	dir := t.TempDir()
	// No venv layout — just a stray app.py.
	if err := os.WriteFile(filepath.Join(dir, "app.py"), []byte("print(1)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := detectProjectVenvPython(dir); got != "" {
		t.Errorf("detectProjectVenvPython() = %q, want empty", got)
	}
}

func TestDetectProjectVenvPythonRejectsDirectoryNamedPython(t *testing.T) {
	// Edge case: a directory called `venv/bin/python/` (not a file)
	// must NOT be treated as the python binary. Prevents false
	// positives when a venv has been corrupted or scaffolded weirdly.
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "venv", "bin", "python"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := detectProjectVenvPython(dir); got != "" {
		t.Errorf("detectProjectVenvPython() = %q, want empty (directory not file)", got)
	}
}

func TestDetectProjectToolchainsPython(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "requirements.txt"), []byte("flask\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tcs := detectProjectToolchains(dir)
	if len(tcs) != 1 || tcs[0].Name != "python" {
		t.Fatalf("got %+v, want one python toolchain", tcs)
	}
	if tcs[0].InstallCommand != "pip install -r requirements.txt" {
		t.Errorf("install = %q, want pip install -r", tcs[0].InstallCommand)
	}
}

func TestDetectProjectToolchainsPolyglot(t *testing.T) {
	// React frontend + Django backend + Rust core in one repo.
	dir := t.TempDir()
	for _, f := range []string{"package.json", "tsconfig.json", "pyproject.toml", "Cargo.toml", "Cargo.lock"} {
		if err := os.WriteFile(filepath.Join(dir, f), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	tcs := detectProjectToolchains(dir)
	names := map[string]bool{}
	for _, tc := range tcs {
		names[tc.Name] = true
	}
	for _, want := range []string{"python", "node", "rust"} {
		if !names[want] {
			t.Errorf("missing toolchain %q in %v", want, names)
		}
	}
	// Node with tsconfig.json should pick the tsx runner.
	for _, tc := range tcs {
		if tc.Name == "node" && tc.Runner != "tsx" {
			t.Errorf("node runner = %q, want tsx (tsconfig present)", tc.Runner)
		}
	}
}

func TestDetectProjectToolchainsNodePkgManager(t *testing.T) {
	cases := []struct {
		lockfile string
		wantPM   string
	}{
		{"pnpm-lock.yaml", "pnpm"},
		{"yarn.lock", "yarn"},
		{"bun.lockb", "bun"},
		{"package-lock.json", "npm"},
	}
	for _, tc := range cases {
		t.Run(tc.lockfile, func(t *testing.T) {
			dir := t.TempDir()
			os.WriteFile(filepath.Join(dir, "package.json"), []byte("{}"), 0o644)
			os.WriteFile(filepath.Join(dir, tc.lockfile), []byte(""), 0o644)
			tcs := detectProjectToolchains(dir)
			if len(tcs) != 1 || tcs[0].PackageManager != tc.wantPM {
				t.Fatalf("got pkgManager %q, want %q", tcs[0].PackageManager, tc.wantPM)
			}
		})
	}
}

func TestProbeToolchainReadyPythonVenv(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "requirements.txt"), []byte("flask"), 0o644)
	tcs := detectProjectToolchains(dir)
	if got := probeToolchainReady(dir, tcs[0]); !strings.Contains(got, "NOT installed") {
		t.Errorf("no venv yet, got %q", got)
	}
	// Now scaffold a populated venv.
	sp := filepath.Join(dir, "venv", "lib", "python3.11", "site-packages", "flask")
	os.MkdirAll(sp, 0o755)
	if got := probeToolchainReady(dir, tcs[0]); !strings.Contains(got, "appear installed") {
		t.Errorf("populated venv not detected: %q", got)
	}
}

func TestProbeToolchainReadyNodeModules(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "package.json"), []byte("{}"), 0o644)
	tcs := detectProjectToolchains(dir)
	if got := probeToolchainReady(dir, tcs[0]); !strings.Contains(got, "missing") {
		t.Errorf("no node_modules, got %q", got)
	}
	os.MkdirAll(filepath.Join(dir, "node_modules", "express"), 0o755)
	if got := probeToolchainReady(dir, tcs[0]); !strings.Contains(got, "populated") {
		t.Errorf("populated node_modules not detected: %q", got)
	}
}

func TestHasUserPackagesIgnoresPipOnly(t *testing.T) {
	dir := t.TempDir()
	for _, p := range []string{"pip", "setuptools", "wheel", "pkg_resources"} {
		os.MkdirAll(filepath.Join(dir, p), 0o755)
	}
	if hasUserPackages(dir) {
		t.Error("pip-only site-packages should not count as 'user packages'")
	}
	os.MkdirAll(filepath.Join(dir, "flask"), 0o755)
	if !hasUserPackages(dir) {
		t.Error("flask present should count as user package")
	}
}
