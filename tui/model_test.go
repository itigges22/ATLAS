// Tests for the Bubbletea model — drive Update directly with synthetic
// messages and assert on the resulting state. Skips teatest's harness
// because direct Update calls give the same coverage with less plumbing.

package main

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// keyMsg builds a tea.KeyMsg matching what the runtime sends for a
// named key (e.g. "enter", "ctrl+l").
func keyMsg(s string) tea.KeyMsg {
	switch s {
	case "enter":
		return tea.KeyMsg(tea.Key{Type: tea.KeyEnter})
	case "ctrl+l":
		return tea.KeyMsg(tea.Key{Type: tea.KeyCtrlL})
	case "ctrl+t":
		return tea.KeyMsg(tea.Key{Type: tea.KeyCtrlT})
	case "ctrl+r":
		return tea.KeyMsg(tea.Key{Type: tea.KeyCtrlR})
	case "ctrl+c":
		return tea.KeyMsg(tea.Key{Type: tea.KeyCtrlC})
	}
	// Default: treat as a single rune.
	return tea.KeyMsg(tea.Key{Type: tea.KeyRunes, Runes: []rune(s)})
}

// sized returns a model that has already received a WindowSizeMsg, so
// View() doesn't return the placeholder string.
func sized(width, height int) tuiModel {
	m := newTUIModel("http://test")
	updated, _ := m.Update(tea.WindowSizeMsg{Width: width, Height: height})
	return updated.(tuiModel)
}

func TestEmptyEnterDoesNothing(t *testing.T) {
	m := sized(80, 30)
	updated, _ := m.Update(keyMsg("enter"))
	mu := updated.(tuiModel)
	if len(mu.chat) != 0 {
		t.Errorf("empty enter should not append to chat (got %d rows)", len(mu.chat))
	}
	if mu.turnActive {
		t.Errorf("empty enter should not start a turn")
	}
}

func TestCtrlLClearsChat(t *testing.T) {
	m := sized(80, 30)
	m.chat = []chatMessage{
		{Role: roleUser, Body: "hi"},
		{Role: roleAssistant, Body: "hello"},
	}
	updated, _ := m.Update(keyMsg("ctrl+l"))
	mu := updated.(tuiModel)
	if len(mu.chat) != 0 {
		t.Errorf("ctrl+l should clear chat (got %d rows)", len(mu.chat))
	}
}

func TestCtrlTCyclesMode(t *testing.T) {
	m := sized(80, 30)
	want := []string{"accept-edits", "yolo", "default"}
	for i, w := range want {
		updated, _ := m.Update(keyMsg("ctrl+t"))
		m = updated.(tuiModel)
		if m.mode != w {
			t.Errorf("cycle %d: mode = %q, want %q", i, m.mode, w)
		}
	}
}

func TestEnvelopeMsgUpdatesPipelineState(t *testing.T) {
	m := sized(80, 30)
	updated, _ := m.Update(envelopeMsg{ev: Envelope{
		EventID: "e1", Type: EvtStageStart, Stage: "phase2",
		Timestamp: 1.0, Payload: map[string]interface{}{},
	}})
	m = updated.(tuiModel)
	stages := m.state.stages()
	if len(stages) != 1 || stages[0].Name != "phase2" {
		t.Fatalf("envelope didn't reach state: stages = %v", stages)
	}
	if len(m.envelope) != 1 {
		t.Errorf("envelope log length = %d, want 1", len(m.envelope))
	}
}

func TestChatStreamMsgAppendsAssistantText(t *testing.T) {
	m := sized(80, 30)
	payload, _ := json.Marshal(map[string]string{"content": "hello world"})
	updated, _ := m.Update(chatStreamMsg{ev: chatEvent{
		Type: "text", Data: payload,
	}})
	m = updated.(tuiModel)
	if len(m.chat) != 1 || m.chat[0].Role != roleAssistant {
		t.Fatalf("chat = %+v, want one assistant row", m.chat)
	}
	if m.chat[0].Body != "hello world" {
		t.Errorf("body = %q, want %q", m.chat[0].Body, "hello world")
	}
}

func TestChatStreamMsgToolResultMarksSuccess(t *testing.T) {
	m := sized(80, 30)
	payload, _ := json.Marshal(map[string]interface{}{
		"tool":    "read_file",
		"success": true,
		"data":    json.RawMessage(`{"content": "42 lines"}`),
	})
	updated, _ := m.Update(chatStreamMsg{ev: chatEvent{
		Type: "tool_result", Data: payload,
	}})
	m = updated.(tuiModel)
	if len(m.chat) != 1 || !m.chat[0].Success {
		t.Errorf("expected one successful tool row; got %+v", m.chat)
	}
}

func TestTurnDoneSentinelClearsActive(t *testing.T) {
	m := sized(80, 30)
	m.turnActive = true
	payload, _ := json.Marshal(map[string]string{"err": ""})
	updated, _ := m.Update(chatStreamMsg{ev: chatEvent{
		Type: "__turn_done__", Data: payload,
	}})
	m = updated.(tuiModel)
	if m.turnActive {
		t.Errorf("turnActive should be false after __turn_done__")
	}
}

func TestTurnDoneWithErrorAppendsSystemMsg(t *testing.T) {
	m := sized(80, 30)
	m.turnActive = true
	payload, _ := json.Marshal(map[string]string{"err": "boom"})
	updated, _ := m.Update(chatStreamMsg{ev: chatEvent{
		Type: "__turn_done__", Data: payload,
	}})
	m = updated.(tuiModel)
	if len(m.chat) != 1 || m.chat[0].Role != roleSystem {
		t.Fatalf("expected one system error row; got %+v", m.chat)
	}
	if m.chat[0].Body != "boom" {
		t.Errorf("body = %q", m.chat[0].Body)
	}
}

func TestTickAdvancesSpinner(t *testing.T) {
	m := sized(80, 30)
	for i := 0; i < 3; i++ {
		updated, _ := m.Update(tickMsg(time.Now()))
		m = updated.(tuiModel)
	}
	if m.spinnerFrame != 3 {
		t.Errorf("spinnerFrame = %d, want 3", m.spinnerFrame)
	}
}

func TestViewContainsPaneTitlesWhenSized(t *testing.T) {
	m := sized(120, 40)
	out := m.View()
	for _, want := range []string{"Pipeline", "Chat", "Events", "Message"} {
		if !strings.Contains(out, want) {
			t.Errorf("view missing pane title %q", want)
		}
	}
}

func TestViewBeforeWindowSizeRendersWithSafeDefaults(t *testing.T) {
	// Some terminals don't reliably emit an initial WindowSizeMsg —
	// View must render the actual UI with safe defaults so the user
	// isn't stuck on a placeholder screen.
	m := newTUIModel("http://test")
	out := m.View()
	for _, want := range []string{"Pipeline", "Chat", "Message"} {
		if !strings.Contains(out, want) {
			t.Errorf("pre-size view missing %q; got %q", want, out)
		}
	}
}
