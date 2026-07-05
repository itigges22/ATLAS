// Credential-file read exclusion (P0: sensitive values must not flow
// into model context by default). All paths are synthetic fixtures.

package main

import (
	"encoding/json"
	"os"
	"testing"
)

func TestDenyReadPathReason(t *testing.T) {
	os.Unsetenv("ATLAS_ALLOW_CREDENTIAL_READS")

	blocked := []string{
		".env",
		"subdir/.env",
		".env.production",
		".netrc",
		".npmrc",
		".pypirc",
		"certs/server.pem",
		"secrets/signing.key",
		".ssh/id_rsa",
		"id_ed25519",
		"/home/user/.ssh/id_ecdsa",
		".aws/credentials",
		".aws/config",
		".kube/config",
		".docker/config.json",
		"secrets/service-token",
		"secrets/api-keys.json",
		"gcp-credentials.json",
	}
	for _, p := range blocked {
		if reason := denyReadPathReason(p); reason == "" {
			t.Errorf("read of %q should be blocked", p)
		}
	}

	allowed := []string{
		".env.example", // template, documented
		"main.go",
		"config.yaml",
		"src/environment.ts", // unrelated name
		"staging.envrc.sample",
		".ssh/id_rsa.pub",     // public half
		"docs/kube/config.md", // .kube parent match is exact-dir only
		"README.md",
	}
	for _, p := range allowed {
		if reason := denyReadPathReason(p); reason != "" {
			t.Errorf("read of %q wrongly blocked: %s", p, reason)
		}
	}
}

func TestDenyReadOverride(t *testing.T) {
	os.Setenv("ATLAS_ALLOW_CREDENTIAL_READS", "1")
	defer os.Unsetenv("ATLAS_ALLOW_CREDENTIAL_READS")
	if reason := denyReadPathReason(".env"); reason != "" {
		t.Fatalf("override not honored: %s", reason)
	}
}

func TestShouldDenyToolCallReadFile(t *testing.T) {
	os.Unsetenv("ATLAS_ALLOW_CREDENTIAL_READS")
	args, _ := json.Marshal(map[string]string{"path": ".netrc"})
	denied, reason := shouldDenyToolCall("read_file", args)
	if !denied {
		t.Fatal("read_file .netrc not denied")
	}
	if reason == "" || !containsStr(reason, "ATLAS_ALLOW_CREDENTIAL_READS") {
		t.Fatalf("refusal must name the documented override: %q", reason)
	}

	denied, _ = shouldDenyToolCall("outline_file", args)
	if !denied {
		t.Fatal("outline_file .netrc not denied")
	}

	ok, _ := json.Marshal(map[string]string{"path": "main.py"})
	denied, reason = shouldDenyToolCall("read_file", ok)
	if denied {
		t.Fatalf("normal read denied: %s", reason)
	}
}

func containsStr(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
