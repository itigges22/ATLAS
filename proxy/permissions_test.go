// Tests for the safety deny-list applied in executeToolCall.

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newPermCtx(dir string) *AgentContext {
	return &AgentContext{
		WorkingDir:    dir,
		FilesRead:     map[string]string{},
		FileReadTimes: map[string]time.Time{},
		SessionWrites: map[string]bool{},
	}
}

// write_file to a sensitive target (.env) must be refused in every mode,
// and nothing may land on disk.
func TestExecuteToolCallDeniesEnvWrite(t *testing.T) {
	dir := t.TempDir()
	ctx := newPermCtx(dir)
	res := executeToolCall("write_file", json.RawMessage(`{"path":".env","content":"SECRET=1"}`), ctx)
	if res.Success {
		t.Fatalf("write_file to .env succeeded, want denial: %+v", res)
	}
	if !strings.Contains(res.Error, "blocked by safety rule") {
		t.Errorf("error %q does not mention the safety rule", res.Error)
	}
	if _, err := os.Stat(filepath.Join(dir, ".env")); !os.IsNotExist(err) {
		t.Errorf(".env exists on disk after denied write")
	}
}

// Sensitive key material is refused, including in subdirectories.
func TestExecuteToolCallDeniesKeyMaterialWrites(t *testing.T) {
	dir := t.TempDir()
	ctx := newPermCtx(dir)
	for _, path := range []string{
		"server.pem", "id_rsa.key", "aws_credentials.json",
		"certs/server.pem", "keys/id_rsa.key", "config/aws_credentials.json",
	} {
		input, _ := json.Marshal(map[string]string{"path": path, "content": "secret"})
		res := executeToolCall("write_file", json.RawMessage(input), ctx)
		if res.Success {
			t.Errorf("write_file to %q succeeded, want denial", path)
		}
	}
}

// Files whose names merely resemble a sensitive one must NOT be blocked.
func TestDenyWritePathAllowsLookalikes(t *testing.T) {
	for _, path := range []string{
		".env.example", ".envrc", "staging.env", "deploy/production.env",
		"src/app.envoy.yaml", "docs/environment.md", "pemphigus.txt",
	} {
		if reason := denyWritePathReason(path); reason != "" {
			t.Errorf("denyWritePathReason(%q) = %q, want allowed", path, reason)
		}
	}
	for _, path := range []string{".env", "certs/tls.pem", "a/b/c/service.key"} {
		if reason := denyWritePathReason(path); reason == "" {
			t.Errorf("denyWritePathReason(%q) allowed, want denied", path)
		}
	}
}

// Only destructive root-scoped commands are blocked; in-workspace commands
// and commands that merely mention a dangerous string are allowed.
func TestDenyCommandReason(t *testing.T) {
	denied := []string{
		"rm -rf /", "rm -rf /*", "rm -fr / ", "sudo rm -rf /",
		"mkfs.ext4 /dev/sda1", "dd if=/dev/zero of=/dev/sda",
	}
	for _, cmd := range denied {
		if denyCommandReason(cmd) == "" {
			t.Errorf("denyCommandReason(%q) allowed, want denied", cmd)
		}
	}
	allowed := []string{
		"rm -rf /workspace/build", "rm -rf ./node_modules", "rm -rf /tmp/scratch",
		"git clean -fdx", "echo 'rm -rf /' > warn.txt", "make", "npm run build",
		"grep mkfs docs.txt", "dd if=input.bin of=output.bin",
	}
	for _, cmd := range allowed {
		if reason := denyCommandReason(cmd); reason != "" {
			t.Errorf("denyCommandReason(%q) = %q, want allowed", cmd, reason)
		}
	}
}

// A normal write is not affected by the deny-list.
func TestExecuteToolCallAllowsNormalWrite(t *testing.T) {
	dir := t.TempDir()
	ctx := newPermCtx(dir)
	res := executeToolCall("write_file", json.RawMessage(`{"path":"notes.txt","content":"grocery list:\n- apples\n- flour\n"}`), ctx)
	if !res.Success {
		t.Fatalf("normal write_file failed: %+v", res)
	}
	data, err := os.ReadFile(filepath.Join(dir, "notes.txt"))
	if err != nil || !strings.Contains(string(data), "apples") {
		t.Errorf("notes.txt missing or wrong content: %q err=%v", string(data), err)
	}
}
