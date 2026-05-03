// Tests for slash command dispatch (PC-062 step 4).
//
// handleSlash is a pure-ish function from input string → (state mutation,
// tea.Cmd). Tests pin the dispatch table and the local-state mutations
// (context add/drop, help/quit signaling). Shell-out commands are not
// executed here — that's covered by step 7's integration tests.

package main

import (
	"strings"
	"testing"
)

func newTestModel() *tuiModel {
	m := newTUIModel("http://localhost:8090")
	return &m
}

func TestSlashHelpEchoesHelpText(t *testing.T) {
	m := newTestModel()
	consumed, cmd, quit := m.handleSlash("/help")
	if !consumed {
		t.Fatal("consumed = false, want true")
	}
	if cmd != nil {
		t.Errorf("/help should not return a tea.Cmd")
	}
	if quit {
		t.Errorf("/help should not signal quit")
	}
	// Two messages: the echo of "/help" and the help-body.
	if len(m.chat) != 2 {
		t.Fatalf("chat length = %d, want 2", len(m.chat))
	}
	if !strings.Contains(m.chat[1].Body, "Slash commands") {
		t.Errorf("help body missing header: %q", m.chat[1].Body)
	}
}

func TestSlashQuitSignalsQuit(t *testing.T) {
	m := newTestModel()
	consumed, cmd, quit := m.handleSlash("/quit")
	if !consumed || !quit {
		t.Errorf("consumed=%v quit=%v, want both true", consumed, quit)
	}
	if cmd == nil {
		t.Errorf("expected tea.Quit cmd")
	}
}

func TestSlashAddPopulatesContext(t *testing.T) {
	m := newTestModel()
	m.handleSlash("/add foo.go bar.go")
	if !m.contextFiles["foo.go"] || !m.contextFiles["bar.go"] {
		t.Errorf("contextFiles = %v", m.contextFiles)
	}
	// Adding the same files again should report "no new files added"
	// without duplicating entries.
	m.handleSlash("/add foo.go")
	count := 0
	for range m.contextFiles {
		count++
	}
	if count != 2 {
		t.Errorf("contextFiles size = %d, want 2 (no dup)", count)
	}
}

func TestSlashDropRemovesContext(t *testing.T) {
	m := newTestModel()
	m.handleSlash("/add foo.go bar.go")
	m.handleSlash("/drop foo.go")
	if m.contextFiles["foo.go"] {
		t.Errorf("foo.go should be dropped")
	}
	if !m.contextFiles["bar.go"] {
		t.Errorf("bar.go should remain")
	}
}

func TestContextSuffixOmitsHintWhenEmpty(t *testing.T) {
	m := newTestModel()
	if got := m.contextSuffix(); got != "" {
		t.Errorf("empty context suffix = %q, want empty", got)
	}
	m.handleSlash("/add foo.go")
	got := m.contextSuffix()
	if !strings.Contains(got, "foo.go") {
		t.Errorf("suffix = %q, missing foo.go", got)
	}
	if !strings.Contains(got, "atlas-tui context") {
		t.Errorf("suffix = %q, missing marker tag", got)
	}
}

func TestSlashUnknownReportsErrorNotPassthrough(t *testing.T) {
	m := newTestModel()
	consumed, _, _ := m.handleSlash("/diffx")
	if !consumed {
		t.Fatal("unknown slash should still be consumed (not passed to agent)")
	}
	if len(m.chat) < 2 {
		t.Fatalf("expected echo + error, got %d msgs", len(m.chat))
	}
	if !strings.Contains(m.chat[1].Body, "unknown command") {
		t.Errorf("missing unknown-command notice: %q", m.chat[1].Body)
	}
}

func TestNonSlashInputNotConsumed(t *testing.T) {
	m := newTestModel()
	consumed, _, _ := m.handleSlash("fix the snake game")
	if consumed {
		t.Errorf("plain input should not be consumed by slash handler")
	}
	if len(m.chat) != 0 {
		t.Errorf("plain input should not append to chat from slash handler")
	}
}

func TestSlashRunRequiresArgument(t *testing.T) {
	m := newTestModel()
	consumed, cmd, _ := m.handleSlash("/run")
	if !consumed {
		t.Fatal("consumed = false")
	}
	if cmd != nil {
		t.Errorf("/run with no arg should not run anything")
	}
	// echo + error message
	if len(m.chat) != 2 || !strings.Contains(m.chat[1].Body, "/run requires") {
		t.Errorf("chat = %v, want error about missing arg", m.chat)
	}
}
