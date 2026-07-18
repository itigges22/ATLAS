package main

import (
	"os"
	"testing"
)

func TestResolveTrustModeDefault(t *testing.T) {
	os.Unsetenv("ATLAS_TRUST_MODE")
	if m := resolveTrustMode(); m != trustTrusted {
		t.Fatalf("default trust mode = %q, want trusted", m)
	}
}

func TestResolveTrustModeValues(t *testing.T) {
	cases := map[string]trustMode{
		"untrusted":     trustUntrusted,
		"trusted":       trustTrusted,
		"fully-trusted": trustFullyTrusted,
		"fully_trusted": trustFullyTrusted,
		"FULLY-TRUSTED": trustFullyTrusted,
		"":              trustTrusted,
		"nonsense":      trustTrusted, // unrecognized → safe default
	}
	for in, want := range cases {
		os.Setenv("ATLAS_TRUST_MODE", in)
		if got := resolveTrustMode(); got != want {
			t.Errorf("ATLAS_TRUST_MODE=%q → %q, want %q", in, got, want)
		}
	}
	os.Unsetenv("ATLAS_TRUST_MODE")
}

func TestCommandsAllowed(t *testing.T) {
	if trustUntrusted.commandsAllowed() {
		t.Error("untrusted must not allow commands")
	}
	if !trustTrusted.commandsAllowed() {
		t.Error("trusted must allow commands")
	}
	if !trustFullyTrusted.commandsAllowed() {
		t.Error("fully-trusted must allow commands")
	}
}

func TestHostExecutionAllowed(t *testing.T) {
	if trustUntrusted.hostExecutionAllowed() {
		t.Error("untrusted must not allow host execution")
	}
	if trustTrusted.hostExecutionAllowed() {
		t.Error("trusted must NOT allow host execution (sandbox only)")
	}
	if !trustFullyTrusted.hostExecutionAllowed() {
		t.Error("fully-trusted must allow host execution")
	}
}

func TestRunCommandRefusedWhenUntrusted(t *testing.T) {
	tool := runCommandTool()
	ctx := &AgentContext{TrustMode: trustUntrusted, WorkingDir: "/tmp"}
	res, err := tool.Execute([]byte(`{"command":"echo hi"}`), ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Success {
		t.Fatal("run_command should be refused under untrusted mode")
	}
	if res.Error != untrustedRefusal {
		t.Fatalf("expected untrusted refusal, got: %q", res.Error)
	}
}
