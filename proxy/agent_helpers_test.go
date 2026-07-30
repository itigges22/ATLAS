package main

import (
	"fmt"
	"strings"
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
		{Role: "user", Content: "PRIOR turn ask"},        // from PriorHistory
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

func TestClassifyAgentTierMultiComponentIsTreatedAsWork(t *testing.T) {
	// This asserted Tier3Hard when classifyAgentTier had a multi-component
	// branch. That branch was removed because T3 was indistinguishable from
	// T2 at every consumer: TierMaxTurns leaves T1/T2/T3 uncapped alike,
	// shouldGeneratePlan tests only Tier0Conversational, and v3-service
	// reads the tier into a log line without branching on it. The contract
	// that actually holds is that these are work, not chat.
	workPrompts := []string{
		"build a full application with frontend and backend authentication",
		"set up middleware, database, and authentication for the api",
	}
	for _, msg := range workPrompts {
		if got := classifyAgentTier(msg); got == Tier0Conversational {
			t.Errorf("classifyAgentTier(%q) = T0 — a multi-component build must not be capped at 5 turns", msg)
		}
	}
}

// The 2026-07-21 dogfooding regression. This message names no file, matches
// no task-verb list, and is not a question — it describes desired behaviour.
// Classified T0 it was capped at 5 turns and produced a zero-tool-call
// non-answer instead of an edit.
func TestClassifyAgentTierBehaviourDescriptionIsWork(t *testing.T) {
	msgs := []string{
		"the snake is still moving way too fast, please slow it down significantly",
		"the sidebar overlaps the content on narrow screens",
		"it crashes when the list is empty",
	}
	for _, msg := range msgs {
		if got := classifyAgentTier(msg); got == Tier0Conversational {
			t.Errorf("classifyAgentTier(%q) = T0 — describing broken behaviour is a task; "+
				"absence of a recognized task word is not evidence of chat", msg)
		}
	}
}

func TestClassifyAgentTierGreetingsAndQuestionsAreConversational(t *testing.T) {
	for _, msg := range []string{"hi", "thanks", "ok", "yep", "  hello  "} {
		if got := classifyAgentTier(msg); got != Tier0Conversational {
			t.Errorf("classifyAgentTier(%q) = %v, want T0", msg, got)
		}
	}
	for _, msg := range []string{
		"why does the game store direction as a string",
		"what does the lens actually score here",
		"is the sandbox mounted read-only?",
	} {
		if got := classifyAgentTier(msg); got != Tier0Conversational {
			t.Errorf("classifyAgentTier(%q) = %v, want T0 (question shape)", msg, got)
		}
	}
}

// A question that is also a request must be treated as the request.
func TestClassifyAgentTierTaskWinsOverQuestionShape(t *testing.T) {
	for _, msg := range []string{
		"can you fix the login bug?",
		"could you add a logout button to the navbar?",
		"why is it broken - can you repair the parser?",
	} {
		if got := classifyAgentTier(msg); got != Tier2Medium {
			t.Errorf("classifyAgentTier(%q) = %v, want T2 — task intent outranks question shape", msg, got)
		}
	}
}

func TestClassifyParseFailureTruncatedEditFile(t *testing.T) {
	// Real shape from the May 2026 user logs: tool_call with edit_file
	// + huge old_str cut mid-string. Feedback should call out
	// truncation explicitly so the model shrinks the next attempt.
	raw := `{"type":"tool_call","name":"edit_file","args":{"path":"snake/app.py","old_str":"@app.route('/')\ndef index():\n    return render_template('index.html')\n\n@app.route('/product')\ndef product():\n    return render_template('product.html')\n\n@app.route('/solutions')\ndef solutions():\n    return render_template('solutions.html')\n\n@app.route('/pricing"`

	_, got := classifyParseFailure(raw)
	if !strings.Contains(got, "TRUNCATED") {
		t.Errorf("expected TRUNCATED callout, got %q", got)
	}
	if !strings.Contains(got, "shrink") && !strings.Contains(got, "smaller") {
		t.Errorf("expected actionable shrink advice, got %q", got)
	}
}

func TestClassifyParseFailureEmptyResponse(t *testing.T) {
	if _, got := classifyParseFailure(""); !strings.Contains(got, "empty") {
		t.Errorf("empty input should mention empty, got %q", got)
	}
	if _, got := classifyParseFailure("   \n\t "); !strings.Contains(got, "empty") {
		t.Errorf("whitespace-only should be treated as empty, got %q", got)
	}
}

func TestClassifyParseFailureMalformedToolCall(t *testing.T) {
	// Looks like a tool_call but ends cleanly — different feedback
	// than truncation.
	raw := `{"type":"tool_call","name":"read_file","args":{"path":"app.py",}}`
	_, got := classifyParseFailure(raw)
	if strings.Contains(got, "TRUNCATED") {
		t.Errorf("clean-ending malformed shouldn't say TRUNCATED, got %q", got)
	}
}

func TestClassifyParseFailureProse(t *testing.T) {
	raw := "Here's what I'll do: I'll read the file first..."
	_, got := classifyParseFailure(raw)
	if !strings.Contains(got, "JSON") {
		t.Errorf("prose response should get JSON-only nudge, got %q", got)
	}
}

// budgetedKeepLast must COUNT the pinned messages (recent user + recent
// file read) — trimMessages re-injects them even when they're outside
// the tail, so ignoring them under-budgets the real prompt (observed
// live as a llama-server exceed_context_size 400: 32844 > 32768).
func TestBudgetedKeepLastCountsPinnedFile(t *testing.T) {
	t.Setenv("ATLAS_CTX_SIZE", "131072")
	t.Setenv("ATLAS_PARALLEL_SLOTS", "4")
	// Budget ≈ 32768 - 8192 - 2048 - 4096 = 18432 tokens.
	bigFile := strings.Repeat("x", 40000) // ~10k tokens, pinned read_file
	msgs := []AgentMessage{{Role: "system", Content: "sys"}}
	msgs = append(msgs, AgentMessage{Role: "user", Content: "task"})
	msgs = append(msgs, AgentMessage{Role: "tool", ToolName: "read_file", Content: bigFile})
	// 40 tool exchanges of ~1k tokens each — far beyond budget together
	// with the pinned file.
	chunk := strings.Repeat("y", 4000)
	for i := 0; i < 40; i++ {
		msgs = append(msgs,
			AgentMessage{Role: "assistant", Content: "call"},
			AgentMessage{Role: "tool", Content: chunk})
	}

	keep := budgetedKeepLast(msgs)
	kept := trimMessages(msgs, keep)

	// Sum the estimate over what trimMessages actually keeps; it must
	// fit the budget (the old bug: pinned file was re-injected on top
	// of an already-full tail).
	total := 0
	for _, m := range kept {
		total += estTokens(m.Content)
	}
	if budget := conversationTokenBudget(); total > budget {
		t.Errorf("kept set estimates %d tokens > budget %d — pins not counted", total, budget)
	}
	if keep < 8 {
		t.Errorf("keep = %d, floor is 8", keep)
	}
}

// Overflow detection keys on llama.cpp's error type string and message.
func TestIsContextOverflow(t *testing.T) {
	err := fmt.Errorf(`LLM returned 400: {"error":{"code":400,"message":"request (32844 tokens) exceeds the available context size (32768 tokens), try increasing it","type":"exceed_context_size_error"}}`)
	if !isContextOverflow(err) {
		t.Error("expected overflow detection on exceed_context_size_error")
	}
	if isContextOverflow(fmt.Errorf("LLM returned 500: upstream crashed")) {
		t.Error("false positive on unrelated LLM error")
	}
	if isContextOverflow(nil) {
		t.Error("false positive on nil")
	}
}

// --- repetition sampling ---------------------------------------------------

func TestApplyRepetitionSamplingDefaultsEnableDry(t *testing.T) {
	body := map[string]interface{}{}
	applyRepetitionSampling(body)

	if body["dry_multiplier"] != 0.8 {
		t.Fatalf("dry_multiplier = %v, want 0.8 (DRY must be on by default — "+
			"llama-server ships every repetition control disabled)", body["dry_multiplier"])
	}
	// Above llama.cpp's default of 2: 3-token runs are ordinary in source.
	if body["dry_allowed_length"] != 6 {
		t.Fatalf("dry_allowed_length = %v, want 6", body["dry_allowed_length"])
	}
	if body["dry_penalty_last_n"] != 2048 {
		t.Fatalf("dry_penalty_last_n = %v, want 2048 (bounded lookback, not -1)",
			body["dry_penalty_last_n"])
	}
	// repeat_penalty scores individual tokens and mangles code indentation;
	// it must stay off unless explicitly opted into.
	if _, ok := body["repeat_penalty"]; ok {
		t.Fatalf("repeat_penalty must not be set by default, got %v", body["repeat_penalty"])
	}
}

func TestApplyRepetitionSamplingDryDisabledByEnv(t *testing.T) {
	t.Setenv("ATLAS_DRY_MULTIPLIER", "0")
	body := map[string]interface{}{}
	applyRepetitionSampling(body)

	for _, k := range []string{"dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n"} {
		if _, ok := body[k]; ok {
			t.Fatalf("%s set despite ATLAS_DRY_MULTIPLIER=0", k)
		}
	}
}

func TestApplyRepetitionSamplingRepeatPenaltyOptIn(t *testing.T) {
	t.Setenv("ATLAS_REPEAT_PENALTY", "1.1")
	body := map[string]interface{}{}
	applyRepetitionSampling(body)

	if body["repeat_penalty"] != 1.1 {
		t.Fatalf("repeat_penalty = %v, want 1.1", body["repeat_penalty"])
	}
	if body["repeat_last_n"] != 64 {
		t.Fatalf("repeat_last_n = %v, want 64", body["repeat_last_n"])
	}
}

func TestEnvFloatOrAndEnvIntOrFallBackOnGarbage(t *testing.T) {
	t.Setenv("ATLAS_TEST_FLOAT", "not-a-float")
	t.Setenv("ATLAS_TEST_INT", "not-an-int")
	if got := envFloatOr("ATLAS_TEST_FLOAT", 1.75); got != 1.75 {
		t.Fatalf("envFloatOr on garbage = %v, want fallback 1.75", got)
	}
	if got := envIntOr("ATLAS_TEST_INT", 6); got != 6 {
		t.Fatalf("envIntOr on garbage = %v, want fallback 6", got)
	}
}
