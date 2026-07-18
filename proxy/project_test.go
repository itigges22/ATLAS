package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDetectNodeJSUsesDeclaredScripts(t *testing.T) {
	dir := t.TempDir()
	packageJSON := `{
		"scripts": {
			"build": "vite build",
			"dev": "vite --host 0.0.0.0",
			"test": "vitest run"
		},
		"dependencies": {
			"react": "latest"
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "package.json"), []byte(packageJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	info := detectProjectInfo(dir)
	if info == nil {
		t.Fatal("detectProjectInfo returned nil")
	}
	if info.BuildCommand != "npm run build" {
		t.Fatalf("BuildCommand = %q, want npm run build", info.BuildCommand)
	}
	if info.DevCommand != "npm run dev" {
		t.Fatalf("DevCommand = %q, want npm run dev", info.DevCommand)
	}
	if info.TestCommand != "npm test" {
		t.Fatalf("TestCommand = %q, want npm test", info.TestCommand)
	}
}

func TestDetectNodeJSDoesNotInventMissingBuildScript(t *testing.T) {
	dir := t.TempDir()
	packageJSON := `{
		"scripts": {
			"start": "node server.js"
		},
		"dependencies": {
			"express": "latest"
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "package.json"), []byte(packageJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	info := detectProjectInfo(dir)
	if info == nil {
		t.Fatal("detectProjectInfo returned nil")
	}
	if info.BuildCommand != "" {
		t.Fatalf("BuildCommand = %q, want empty when package has no build script", info.BuildCommand)
	}
	if info.TestCommand != "" {
		t.Fatalf("TestCommand = %q, want empty when package has no test script", info.TestCommand)
	}
}

func TestDetectNodeJSUsesLockfilePackageManagerForScripts(t *testing.T) {
	dir := t.TempDir()
	packageJSON := `{
		"scripts": {
			"build": "vite build",
			"test": "vitest run"
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "package.json"), []byte(packageJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "pnpm-lock.yaml"), []byte("lockfileVersion: '9.0'\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	info := detectProjectInfo(dir)
	if info == nil {
		t.Fatal("detectProjectInfo returned nil")
	}
	if info.BuildCommand != "pnpm run build" {
		t.Fatalf("BuildCommand = %q, want pnpm run build", info.BuildCommand)
	}
	if info.TestCommand != "pnpm run test" {
		t.Fatalf("TestCommand = %q, want pnpm run test", info.TestCommand)
	}
}

func TestDetectNodeJSUsesCurrentBunLockfile(t *testing.T) {
	dir := t.TempDir()
	packageJSON := `{
		"scripts": {
			"build": "vite build",
			"test": "bun test"
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "package.json"), []byte(packageJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bun.lock"), []byte("# bun lockfile\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	info := detectProjectInfo(dir)
	if info == nil {
		t.Fatal("detectProjectInfo returned nil")
	}
	if info.BuildCommand != "bun run build" {
		t.Fatalf("BuildCommand = %q, want bun run build", info.BuildCommand)
	}
	if info.TestCommand != "bun run test" {
		t.Fatalf("TestCommand = %q, want bun run test", info.TestCommand)
	}
}

func TestDetectNodeJSUsesPackageManagerFieldWithoutLockfile(t *testing.T) {
	dir := t.TempDir()
	packageJSON := `{
		"packageManager": "yarn@4.10.3",
		"scripts": {
			"build": "vite build"
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "package.json"), []byte(packageJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	info := detectProjectInfo(dir)
	if info == nil {
		t.Fatal("detectProjectInfo returned nil")
	}
	if info.BuildCommand != "yarn build" {
		t.Fatalf("BuildCommand = %q, want yarn build", info.BuildCommand)
	}
}

func TestDetectNextJSFallsBackToNextBuild(t *testing.T) {
	dir := t.TempDir()
	packageJSON := `{
		"dependencies": {
			"next": "latest",
			"react": "latest",
			"react-dom": "latest"
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "package.json"), []byte(packageJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	info := detectProjectInfo(dir)
	if info == nil {
		t.Fatal("detectProjectInfo returned nil")
	}
	if info.Framework != "nextjs" {
		t.Fatalf("Framework = %q, want nextjs", info.Framework)
	}
	if info.BuildCommand != "npx next build" {
		t.Fatalf("BuildCommand = %q, want npx next build", info.BuildCommand)
	}
}
