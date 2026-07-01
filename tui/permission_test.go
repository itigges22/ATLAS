// Tests for the interactive permission approval modal — the
// "permission_request" SSE handler, the y/a/n input gating, and the
// /v1/permission post-back.

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// permReqMsg builds a chatStreamMsg carrying a permission_request event.
func permReqMsg(tool, message, callID string) chatStreamMsg {
	data, _ := json.Marshal(map[string]interface{}{
		"tool_name":    tool,
		"message":      message,
		"tool_call_id": callID,
		"args":         json.RawMessage(`{"command":"npm install"}`),
	})
	return chatStreamMsg{ev: chatEvent{Type: "permission_request", Data: data}}
}

// A permission_request for a tool NOT on the session allowlist raises the
// modal and captures the turn's session id.
func TestPermissionRequestRaisesModal(t *testing.T) {
	m := sized(80, 30)
	m.turnSessionID = "turn-abc"
	updated, _ := m.Update(permReqMsg("run_command", "Run command: npm install", "call_3"))
	mu := updated.(tuiModel)
	if mu.pendingPerm == nil {
		t.Fatal("pendingPerm is nil — modal should be up")
	}
	if mu.pendingPerm.toolName != "run_command" {
		t.Errorf("toolName = %q, want run_command", mu.pendingPerm.toolName)
	}
	if mu.pendingPerm.message != "Run command: npm install" {
		t.Errorf("message = %q", mu.pendingPerm.message)
	}
	if mu.pendingPerm.toolCallID != "call_3" {
		t.Errorf("toolCallID = %q, want call_3", mu.pendingPerm.toolCallID)
	}
	if mu.pendingPerm.sessionID != "turn-abc" {
		t.Errorf("sessionID = %q, want the turn's session id", mu.pendingPerm.sessionID)
	}
}

// A tool already approved "for session" auto-answers allow without a modal.
func TestPermissionRequestAutoAllowsWhitelistedTool(t *testing.T) {
	m := sized(80, 30)
	m.turnSessionID = "turn-abc"
	m.sessionAllowedTools["run_command"] = true
	updated, _ := m.Update(permReqMsg("run_command", "Run command: npm install", "call_3"))
	mu := updated.(tuiModel)
	if mu.pendingPerm != nil {
		t.Fatal("whitelisted tool should auto-allow without raising the modal")
	}
	if len(mu.chat) != 1 || !strings.Contains(mu.chat[0].Body, "auto-allowed") {
		t.Fatalf("chat = %+v, want an auto-allowed system row", mu.chat)
	}
	if !mu.chat[0].Echo {
		t.Error("decision record should be Echo=true so it stays out of agent history")
	}
}

// permServer captures the last /v1/permission body for assertions.
func permServer(t *testing.T, got *map[string]string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/permission" {
			http.Error(w, "wrong path", 404)
			return
		}
		var body map[string]string
		_ = json.NewDecoder(r.Body).Decode(&body)
		*got = body
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]bool{"delivered": true})
	}))
}

func TestPermissionModalKeyDecisions(t *testing.T) {
	cases := []struct {
		key            string
		wantDecision   string
		wantScope      string
		wantAllowlist  bool
		wantBodyPrefix string // "" = no local transcript row expected
	}{
		{"y", "allow", "once", false, "allowed"},
		{"a", "allow", "session", true, "allowed"},
		// Deny adds no local row — the proxy's permission_denied event renders it.
		{"n", "deny", "once", false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.key, func(t *testing.T) {
			var got map[string]string
			srv := permServer(t, &got)
			defer srv.Close()

			m := sized(80, 30)
			m.proxyURL = srv.URL
			m.pendingPerm = &permPrompt{
				toolName: "run_command", message: "Run command: npm install",
				toolCallID: "call_3", sessionID: "turn-abc",
			}
			var msg tea.KeyMsg
			if tc.key == "esc" {
				msg = tea.KeyMsg(tea.Key{Type: tea.KeyEsc})
			} else {
				msg = keyMsg(tc.key)
			}
			updated, cmd := m.Update(msg)
			mu := updated.(tuiModel)
			if mu.pendingPerm != nil {
				t.Fatal("pendingPerm should be cleared after a decision")
			}
			if tc.wantAllowlist && !mu.sessionAllowedTools["run_command"] {
				t.Error("'a' should add run_command to the session allowlist")
			}
			if !tc.wantAllowlist && tc.key == "y" && mu.sessionAllowedTools["run_command"] {
				t.Error("'y' should not add to the session allowlist")
			}
			if tc.wantBodyPrefix == "" {
				if len(mu.chat) != 0 {
					t.Errorf("expected no local row, got %+v", mu.chat)
				}
			} else if len(mu.chat) == 0 || !strings.HasPrefix(mu.chat[len(mu.chat)-1].Body, tc.wantBodyPrefix) {
				t.Errorf("chat tail = %+v, want a %q record", mu.chat, tc.wantBodyPrefix)
			}
			if cmd == nil {
				t.Fatal("decision must return a post-back Cmd")
			}
			cmd() // fire the POST synchronously
			if got == nil {
				t.Fatal("no /v1/permission POST received")
			}
			if got["decision"] != tc.wantDecision {
				t.Errorf("decision = %q, want %q", got["decision"], tc.wantDecision)
			}
			if got["scope"] != tc.wantScope {
				t.Errorf("scope = %q, want %q", got["scope"], tc.wantScope)
			}
			if got["session_id"] != "turn-abc" || got["tool_call_id"] != "call_3" {
				t.Errorf("correlation = %+v, want session turn-abc / call_3", got)
			}
		})
	}
}

// esc denies, same as 'n'.
func TestPermissionModalEscDenies(t *testing.T) {
	var got map[string]string
	srv := permServer(t, &got)
	defer srv.Close()
	m := sized(80, 30)
	m.proxyURL = srv.URL
	m.pendingPerm = &permPrompt{toolName: "run_command", toolCallID: "call_9", sessionID: "s1"}
	_, cmd := m.Update(tea.KeyMsg(tea.Key{Type: tea.KeyEsc}))
	if cmd == nil {
		t.Fatal("esc must return a post-back Cmd")
	}
	cmd()
	if got["decision"] != "deny" {
		t.Errorf("esc decision = %q, want deny", got["decision"])
	}
}

// While the modal is up, non-decision keys are swallowed and never reach the
// textarea.
func TestPermissionModalSwallowsOtherKeys(t *testing.T) {
	m := sized(80, 30)
	m.pendingPerm = &permPrompt{toolName: "run_command", toolCallID: "call_3", sessionID: "s1"}
	updated, _ := m.Update(keyMsg("q"))
	mu := updated.(tuiModel)
	if mu.pendingPerm == nil {
		t.Error("a non-decision key must not resolve the modal")
	}
	if mu.input.Value() != "" {
		t.Errorf("input = %q, want empty (key swallowed, not typed)", mu.input.Value())
	}
}

// The rendered frame is exactly terminal-height even with the modal up.
func TestViewHeightMatchesTerminalWithModal(t *testing.T) {
	m := withStages(t, 100, 30, 2)
	m.pendingPerm = &permPrompt{
		toolName: "run_command", message: "Run command: npm install",
		toolCallID: "call_3", sessionID: "s1",
	}
	out := m.View()
	if got := lipgloss.Height(out); got != 30 {
		t.Errorf("view height with modal = %d, want 30", got)
	}
	if !strings.Contains(out, "Permission required") {
		t.Error("modal title not rendered")
	}
	if !strings.Contains(out, "[y] allow once") {
		t.Error("modal key legend not rendered")
	}
}

// After an 'a' (allow for session) the next /v1/agent request carries the tool
// in session_allowed_tools so the proxy skips re-prompting.
func TestSessionAllowedToolsRideRequestBody(t *testing.T) {
	var gotReq agentRequest
	srv := fakeAgentServer(t, []chatEvent{
		mkChatEvent("done", map[string]string{"summary": "ok"}),
	}, &gotReq)
	defer srv.Close()

	out := make(chan chatEvent, 8)
	if err := sendChatOpts(context.Background(), srv.URL, "go", "/w", "default", "s1", nil,
		demoOpts{allowedTools: []string{"run_command", "write_file"}}, out); err != nil {
		t.Fatalf("sendChatOpts: %v", err)
	}
	close(out)
	if len(gotReq.SessionAllowedTools) != 2 ||
		gotReq.SessionAllowedTools[0] != "run_command" ||
		gotReq.SessionAllowedTools[1] != "write_file" {
		t.Errorf("session_allowed_tools = %v, want [run_command write_file]",
			gotReq.SessionAllowedTools)
	}
}

func TestSortedAllowedTools(t *testing.T) {
	got := sortedAllowedTools(map[string]bool{"zed": true, "abc": true, "off": false})
	if strings.Join(got, ",") != "abc,zed" {
		t.Errorf("sortedAllowedTools = %v, want [abc zed] (false entries dropped)", got)
	}
	if sortedAllowedTools(nil) != nil {
		t.Error("empty allowlist should yield nil")
	}
}

// Ctrl+C while the modal is up cancels the turn (clearing the modal) rather
// than being swallowed, so the user can always abort mid-prompt.
func TestPermissionModalCtrlCCancelsTurn(t *testing.T) {
	m := sized(80, 30)
	_, cancel := context.WithCancel(context.Background())
	m.turnActive = true
	m.turnCancel = cancel
	m.pendingPerm = &permPrompt{
		toolName: "run_command", toolCallID: "call_1", sessionID: "turn-x",
	}
	updated, _ := m.Update(keyMsg("ctrl+c"))
	mu := updated.(tuiModel)
	if mu.pendingPerm != nil {
		t.Error("Ctrl+C should clear the permission modal")
	}
	if mu.turnActive {
		t.Error("Ctrl+C should cancel the in-flight turn")
	}
}
