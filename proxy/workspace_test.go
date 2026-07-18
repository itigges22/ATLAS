package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveWorkspacePathRejectsTraversalAndAbsolutePaths(t *testing.T) {
	root := t.TempDir()
	ctx := &AgentContext{WorkingDir: root}
	for _, input := range []string{"../outside.txt", "/etc/passwd"} {
		if _, err := resolveWorkspacePath(ctx, input); err == nil {
			t.Errorf("resolveWorkspacePath(%q) succeeded, want rejection", input)
		}
	}
}

func TestResolveWorkspacePathAllowsHostPathTranslation(t *testing.T) {
	root := t.TempDir()
	ctx := &AgentContext{WorkingDir: root, HostWorkingDir: "/Users/test/project"}
	got, err := resolveWorkspacePath(ctx, "/Users/test/project/src/main.go")
	if err != nil {
		t.Fatalf("resolveWorkspacePath: %v", err)
	}
	want := filepath.Join(root, "src", "main.go")
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestResolveWorkspacePathRejectsSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(root, "escape")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	ctx := &AgentContext{WorkingDir: root}
	if _, err := resolveWorkspacePath(ctx, "escape/file.txt"); err == nil {
		t.Fatal("symlink escape succeeded, want rejection")
	}
}

func TestReadWorkspaceFileRejectsSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	secret := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(secret, []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "escape")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}

	ctx := &AgentContext{WorkingDir: root}
	if _, _, err := readWorkspaceFile(ctx, "escape/secret.txt"); err == nil {
		t.Fatal("readWorkspaceFile followed a symlink outside the workspace")
	}
}

func TestReadWorkspaceFileReadsRegularWorkspaceFile(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "inside.txt"), []byte("inside"), 0o600); err != nil {
		t.Fatal(err)
	}

	got, _, err := readWorkspaceFile(&AgentContext{WorkingDir: root}, "inside.txt")
	if err != nil {
		t.Fatalf("readWorkspaceFile: %v", err)
	}
	if string(got) != "inside" {
		t.Fatalf("readWorkspaceFile = %q, want inside", got)
	}
}

func TestExecuteToolCallRejectsWorkspaceEscape(t *testing.T) {
	root := t.TempDir()
	ctx := &AgentContext{WorkingDir: root}
	res := executeToolCall("read_file", json.RawMessage(`{"path":"../secret"}`), ctx)
	if res.Success || !strings.Contains(res.Error, "outside the workspace") {
		t.Fatalf("result = %+v, want workspace rejection", res)
	}
}
