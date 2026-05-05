package main

import (
	"testing"
)

func TestTrimMessagesPinsRecentUser(t *testing.T) {
	// system + user instruction + 10 tool exchanges. keepLast=8 means the
	// raw tail starts at index 3 — the user instruction (index 1) is gone.
	// Pin must restore it.
	msgs := []AgentMessage{{Role: "system", Content: "sys"}}
	msgs = append(msgs, AgentMessage{Role: "user", Content: "fix the bug"})
	for i := 0; i < 10; i++ {
		msgs = append(msgs,
			AgentMessage{Role: "assistant", Content: "tool call"},
			AgentMessage{Role: "tool", Content: "result"})
	}

	got := trimMessages(msgs, 8)

	if got[0].Role != "system" {
		t.Fatalf("got[0].Role = %q, want system", got[0].Role)
	}
	if got[1].Role != "user" || got[1].Content != "fix the bug" {
		t.Fatalf("got[1] = %+v, want pinned user 'fix the bug'", got[1])
	}
	if len(got) != 1+1+8 {
		t.Errorf("len(got) = %d, want 10 (system + pin + 8 tail)", len(got))
	}
}

func TestTrimMessagesNoDuplicateWhenPinInWindow(t *testing.T) {
	// Short conversation: user instruction is already in the tail window.
	// Don't duplicate it.
	msgs := []AgentMessage{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "u1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "u2 — current"},
		{Role: "assistant", Content: "a2"},
		{Role: "tool", Content: "t1"},
		{Role: "assistant", Content: "a3"},
		{Role: "tool", Content: "t2"},
		{Role: "assistant", Content: "a4"},
		{Role: "tool", Content: "t3"},
		{Role: "assistant", Content: "a5"},
		{Role: "tool", Content: "t4"},
		{Role: "assistant", Content: "a6"},
	}
	// 13 messages, keepLast=8 → tailStart=5. Most-recent user is at idx 3,
	// outside window → gets pinned.
	got := trimMessages(msgs, 8)
	userCount := 0
	for _, m := range got {
		if m.Role == "user" {
			userCount++
		}
	}
	if userCount != 1 {
		t.Errorf("user count = %d, want 1 (no duplicate pin)", userCount)
	}
	if got[1].Content != "u2 — current" {
		t.Errorf("pinned msg = %q, want most-recent user 'u2 — current'", got[1].Content)
	}
}

func TestTrimMessagesPinAlreadyInTailNoDuplicate(t *testing.T) {
	// User msg is inside tail window — function shouldn't pin (would dup).
	msgs := []AgentMessage{
		{Role: "system", Content: "sys"},
		{Role: "assistant", Content: "old1"},
		{Role: "tool", Content: "old2"},
		{Role: "assistant", Content: "old3"},
		{Role: "user", Content: "current ask"},
		{Role: "assistant", Content: "tail1"},
		{Role: "tool", Content: "tail2"},
		{Role: "assistant", Content: "tail3"},
		{Role: "tool", Content: "tail4"},
		{Role: "assistant", Content: "tail5"},
		{Role: "tool", Content: "tail6"},
		{Role: "assistant", Content: "tail7"},
	}
	// 12 messages, keepLast=8 → tailStart=4. User at idx 4 → in window.
	got := trimMessages(msgs, 8)
	userCount := 0
	for _, m := range got {
		if m.Role == "user" {
			userCount++
		}
	}
	if userCount != 1 {
		t.Errorf("user count = %d, want 1 (already in tail, no dup)", userCount)
	}
	if len(got) != 1+8 {
		t.Errorf("len(got) = %d, want 9 (system + 8 tail, no pin)", len(got))
	}
}

func TestTrimMessagesShortConversationUnchanged(t *testing.T) {
	msgs := []AgentMessage{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}
	got := trimMessages(msgs, 8)
	if len(got) != 3 {
		t.Errorf("len(got) = %d, want 3 (under threshold, no trim)", len(got))
	}
}

func TestTrimMessagesPriorHistoryDoesNotConfusePin(t *testing.T) {
	// Reproduces the bug: PriorHistory put a prior-turn user msg at idx 1.
	// Hardcoded ctx.Messages[1] would have pinned the WRONG user message.
	// trimMessages scans backwards, so it picks the current-turn user.
	msgs := []AgentMessage{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "PRIOR turn ask"},      // from PriorHistory
		{Role: "assistant", Content: "PRIOR turn reply"}, // from PriorHistory
		{Role: "user", Content: "CURRENT turn ask"},
	}
	for i := 0; i < 10; i++ {
		msgs = append(msgs,
			AgentMessage{Role: "assistant", Content: "tool call"},
			AgentMessage{Role: "tool", Content: "result"})
	}

	got := trimMessages(msgs, 8)
	if got[1].Content != "CURRENT turn ask" {
		t.Errorf("pinned = %q, want 'CURRENT turn ask' (most-recent, not idx-1)", got[1].Content)
	}
}

func TestClassifyAgentTierTrivialChatStaysT0(t *testing.T) {
	for _, msg := range []string{
		"hi", "Hello", "hey", "thanks", "thank you", "ok", "yes", "no",
		"perfect", "got it", "cool", "bye",
	} {
		if got := classifyAgentTier(msg); got != Tier0Conversational {
			t.Errorf("classifyAgentTier(%q) = %v, want T0", msg, got)
		}
	}
}

func TestClassifyAgentTierEmptyOrSubFiveCharsStaysT0(t *testing.T) {
	for _, msg := range []string{"", " ", "  \n", "abc", "a"} {
		if got := classifyAgentTier(msg); got != Tier0Conversational {
			t.Errorf("classifyAgentTier(%q) = %v, want T0", msg, got)
		}
	}
}

func TestClassifyAgentTierRealTaskDefaultsToT2(t *testing.T) {
	// These were T1 under the old cascade — too quick a fall-through.
	// New rule: anything that isn't trivial chat is T2 minimum, so V3
	// has a chance to fire on every real task.
	t2Prompts := []string{
		"fix the issues in the flask web app",
		"why isn't this rendering",
		"add a button to the page",
		"the form submission is broken",
		"can you check what's wrong",
		"make a quick utility",
		"refactor this function",
		"remove the unused import",
	}
	for _, msg := range t2Prompts {
		if got := classifyAgentTier(msg); got < Tier2Medium {
			t.Errorf("classifyAgentTier(%q) = %v, want >= T2", msg, got)
		}
	}
}

func TestClassifyAgentTierMultiComponentStaysT3(t *testing.T) {
	// T3 budget should still trip on explicit multi-component prompts.
	t3Prompts := []string{
		"build a full application with frontend and backend authentication",
		"set up middleware, database, and authentication for the api",
	}
	for _, msg := range t3Prompts {
		if got := classifyAgentTier(msg); got != Tier3Hard {
			t.Errorf("classifyAgentTier(%q) = %v, want T3", msg, got)
		}
	}
}

