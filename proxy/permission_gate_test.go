package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

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
func TestAwaitPermissionNoSessionProceeds(t *testing.T) {
	ctx, cancel := permCtx("")
	defer cancel()
	if !awaitPermission(ctx, "run_command", "call_5", json.RawMessage(`{}`)) {
		t.Error("awaitPermission should proceed when there is no session id")
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
