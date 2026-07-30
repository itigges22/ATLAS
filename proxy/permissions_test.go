// Tests for the safety deny-list applied in executeToolCall.

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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

func permCtx(sessionID string) (*AgentContext, context.CancelFunc) {
	ctx := context.Background()
	reqCtx, cancel := context.WithCancel(ctx)
	return &AgentContext{
		PassID:        sessionID,
		Ctx:           reqCtx,
		FilesRead:     map[string]string{},
		FileReadTimes: map[string]time.Time{},
		SessionWrites: map[string]bool{},
	}, cancel
}

func postDecision(t *testing.T, body string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/v1/permission", strings.NewReader(body))
	w := httptest.NewRecorder()
	handlePermission(w, r)
	return w
}

// An allow decision unblocks awaitPermission and returns true.
func TestAwaitPermissionAllow(t *testing.T) {
	ctx, cancel := permCtx("sess-allow")
	defer cancel()
	done := make(chan bool, 1)
	go func() { done <- awaitPermission(ctx, "run_command", "call_1", json.RawMessage(`{"command":"ls"}`)) }()

	// Wait for the pending entry to register, then answer it.
	waitForPending(t, "sess-allow", "call_1")
	w := postDecision(t, `{"session_id":"sess-allow","tool_call_id":"call_1","decision":"allow","scope":"once"}`)
	if w.Code != http.StatusOK {
		t.Fatalf("decision POST status = %d, want 200", w.Code)
	}
	if got := <-done; !got {
		t.Error("awaitPermission returned false for an allow decision")
	}
}

// A deny decision returns false and does not whitelist the tool.
func TestAwaitPermissionDeny(t *testing.T) {
	ctx, cancel := permCtx("sess-deny")
	defer cancel()
	done := make(chan bool, 1)
	go func() { done <- awaitPermission(ctx, "delete_file", "call_2", json.RawMessage(`{"path":"x"}`)) }()

	waitForPending(t, "sess-deny", "call_2")
	postDecision(t, `{"session_id":"sess-deny","tool_call_id":"call_2","decision":"deny"}`)
	if got := <-done; got {
		t.Error("awaitPermission returned true for a deny decision")
	}
	if ctx.isToolAllowed("delete_file") {
		t.Error("deny should not whitelist the tool")
	}
}

// A session-scoped allow whitelists the tool for the rest of the turn.
func TestAwaitPermissionSessionScopeWhitelists(t *testing.T) {
	ctx, cancel := permCtx("sess-scope")
	defer cancel()
	done := make(chan bool, 1)
	go func() { done <- awaitPermission(ctx, "run_command", "call_3", json.RawMessage(`{"command":"ls"}`)) }()

	waitForPending(t, "sess-scope", "call_3")
	postDecision(t, `{"session_id":"sess-scope","tool_call_id":"call_3","decision":"allow","scope":"session"}`)
	if got := <-done; !got {
		t.Fatal("awaitPermission returned false for an allow decision")
	}
	if !ctx.isToolAllowed("run_command") {
		t.Error("session-scope allow should whitelist run_command for the turn")
	}
}

// Cancelling the request context (client disconnect or /cancel) denies.
func TestAwaitPermissionCancelDenies(t *testing.T) {
	ctx, cancel := permCtx("sess-cancel")
	done := make(chan bool, 1)
	go func() { done <- awaitPermission(ctx, "run_command", "call_4", json.RawMessage(`{}`)) }()
	waitForPending(t, "sess-cancel", "call_4")
	cancel()
	if got := <-done; got {
		t.Error("awaitPermission should deny when the context is cancelled")
	}
}

// A decision for an unknown/already-answered key is a 404 no-op.
func TestHandlePermissionUnknownKey(t *testing.T) {
	w := postDecision(t, `{"session_id":"nope","tool_call_id":"call_9","decision":"allow"}`)
	if w.Code != http.StatusNotFound {
		t.Errorf("unknown key status = %d, want 404", w.Code)
	}
}

// A caller without a session id proceeds without prompting (non-TUI path).
func TestAwaitPermissionNoSessionDenies(t *testing.T) {
	// Fail closed: with no session_id there is no channel to answer a
	// prompt, and proceeding would make mode:"default" yolo-equivalent
	// for any client that omits the field. Unattended clients opt in
	// explicitly via mode:"yolo" or session_allowed_tools.
	ctx, cancel := permCtx("")
	defer cancel()
	if awaitPermission(ctx, "run_command", "call_5", json.RawMessage(`{}`)) {
		t.Error("awaitPermission must deny when there is no session id")
	}
}

func waitForPending(t *testing.T, sessionID, callID string) {
	t.Helper()
	key := permKey(sessionID, callID)
	for i := 0; i < 200; i++ {
		if _, ok := pendingPermissions.Load(key); ok {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("pending permission %q never registered", key)
}

// Tests for needsPermission — the mode/approval logic behind interactive
// permission prompts.

func TestNeedsPermission(t *testing.T) {
	args := json.RawMessage(`{}`)

	t.Run("yolo mode approves everything", func(t *testing.T) {
		for _, ctx := range []*AgentContext{
			{YoloMode: true},
			{PermissionMode: PermissionYolo},
		} {
			if needsPermission(ctx, "run_command", args) {
				t.Errorf("yolo context still prompts for run_command")
			}
		}
	})

	t.Run("unknown tool always prompts", func(t *testing.T) {
		if !needsPermission(&AgentContext{}, "no_such_tool", args) {
			t.Errorf("unknown tool did not prompt")
		}
	})

	t.Run("read-only tools never prompt", func(t *testing.T) {
		if needsPermission(&AgentContext{}, "read_file", args) {
			t.Errorf("read_file prompted in default mode")
		}
	})

	t.Run("destructive tools prompt in default mode", func(t *testing.T) {
		if !needsPermission(&AgentContext{}, "run_command", args) {
			t.Errorf("run_command did not prompt in default mode")
		}
		if !needsPermission(&AgentContext{}, "write_file", args) {
			t.Errorf("write_file did not prompt in default mode")
		}
	})

	t.Run("session-approved tool skips the prompt", func(t *testing.T) {
		ctx := &AgentContext{}
		ctx.allowToolForTurn("run_command")
		if needsPermission(ctx, "run_command", args) {
			t.Errorf("session-approved run_command still prompts")
		}
		// Approval is per-tool, not global.
		if !needsPermission(ctx, "write_file", args) {
			t.Errorf("write_file inherited run_command's approval")
		}
	})

	t.Run("accept-edits auto-approves edits but not commands", func(t *testing.T) {
		ctx := &AgentContext{PermissionMode: PermissionAcceptEdits}
		for _, tool := range []string{"write_file", "edit_file", "structural_edit", "move_file"} {
			if needsPermission(ctx, tool, args) {
				t.Errorf("%s prompted in accept-edits mode", tool)
			}
		}
		if !needsPermission(ctx, "run_command", args) {
			t.Errorf("run_command did not prompt in accept-edits mode")
		}
	})
}

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
