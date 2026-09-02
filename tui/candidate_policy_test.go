package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The candidate policy is a control the user operates, like /ask and ctrl+t:
// a named selection from the proxy's typed values, never something read off
// the words they typed.

func TestCandidatePolicyCommandIsRecognised(t *testing.T) {
	m := &tuiModel{}
	consumed, _, quit := m.handleSlash("/candidate-policy automatic")
	if !consumed || quit {
		t.Fatalf("consumed=%v quit=%v", consumed, quit)
	}
	last := m.chat[len(m.chat)-1]
	if strings.Contains(last.Body, "unknown command") {
		t.Fatalf("/candidate-policy is not a command the TUI knows: %q", last.Body)
	}
	if m.candidatePolicy != candidatePolicyAutomatic {
		t.Fatalf("selection recorded as %q, want %q", m.candidatePolicy, candidatePolicyAutomatic)
	}
	if !strings.Contains(last.Body, "automatic") {
		t.Errorf("the confirmation does not describe the selection: %q", last.Body)
	}
}

func TestCandidatePolicyWordsMapToTheTypedValues(t *testing.T) {
	for word, want := range map[string]string{
		"strict": "strict", "advisory": "advisory", "automatic": "automatic_v3", "automatic_v3": "automatic_v3",
	} {
		got, ok := candidatePolicyValue(word)
		if !ok || got != want {
			t.Errorf("%q -> %q ok=%v, want %q", word, got, ok, want)
		}
	}
	for _, bad := range []string{"", "auto", "confirm", "AUTOMATIC", "yes"} {
		if _, ok := candidatePolicyValue(bad); ok {
			t.Errorf("%q was accepted as a policy", bad)
		}
	}
	m := &tuiModel{}
	m.handleSlash("/candidate-policy confirm")
	if m.candidatePolicy != "" || !strings.Contains(m.chat[len(m.chat)-1].Body, "one of") {
		t.Errorf("an unknown word changed the selection or was not refused: %q / %q",
			m.candidatePolicy, m.chat[len(m.chat)-1].Body)
	}
}

// The selection is explicit on the wire, strict included. An omitted field is
// the proxy's cue to apply the operator default, so a TUI that displayed
// strict while sending nothing could be running under whatever the server
// chose. What the header shows is what the request says.
func TestCandidatePolicyIsAlwaysExplicitOnTheWire(t *testing.T) {
	// A fresh model, before any command: strict, and it says so.
	fresh := &tuiModel{}
	fresh.startNewSession()
	if fresh.candidatePolicy != candidatePolicyStrict {
		t.Fatalf("a new session selected %q, want %q", fresh.candidatePolicy, candidatePolicyStrict)
	}
	body := captureContractWithPolicy(t, taskModeWork, fresh.candidatePolicy, "Create app.py.")
	if got := contractOf(t, body)["candidate_policy"]; got != "strict" {
		t.Errorf("a new session sent candidate_policy=%v, want \"strict\"", got)
	}
	// Explicitly chosen strict: the same explicit value.
	m := &tuiModel{}
	m.handleSlash("/candidate-policy strict")
	if m.candidatePolicy != candidatePolicyStrict {
		t.Errorf("selecting strict stored %q, want %q", m.candidatePolicy, candidatePolicyStrict)
	}
	body = captureContractWithPolicy(t, taskModeWork, m.candidatePolicy, "Create app.py.")
	if got := contractOf(t, body)["candidate_policy"]; got != "strict" {
		t.Errorf("explicit strict sent candidate_policy=%v, want \"strict\"", got)
	}
	// And the request path itself never sends an empty selection: a caller
	// that set nothing still declares strict.
	body = captureContractWithPolicy(t, taskModeWork, "", "Create app.py.")
	if got := contractOf(t, body)["candidate_policy"]; got != "strict" {
		t.Errorf("an unset selection sent candidate_policy=%v, want \"strict\"", got)
	}
}

// captureContractWithPolicy is captureContract with the session selection set.
func captureContractWithPolicy(t *testing.T, mode taskMode, policy, msg string) map[string]interface{} {
	t.Helper()
	var got map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"done\",\"summary\":\"ok\"}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	out := make(chan chatEvent, 16)
	if err := sendChatOpts(context.Background(), srv.URL, msg, t.TempDir(),
		"default", "s", nil, demoOpts{taskMode: mode, candidatePolicy: policy}, out); err != nil {
		t.Fatal(err)
	}
	return got
}

func TestCandidatePolicyIsSentInTheTypedContract(t *testing.T) {
	for _, policy := range []string{candidatePolicyAdvisory, candidatePolicyAutomatic} {
		body := captureContractWithPolicy(t, taskModeWork, policy, "Make solve fast.")
		tc := contractOf(t, body)
		if tc["candidate_policy"] != policy {
			t.Errorf("sent candidate_policy=%v, want %q", tc["candidate_policy"], policy)
		}
		if tc["task_mode"] != "work" {
			t.Errorf("task_mode=%v, want work", tc["task_mode"])
		}
		// Still no fabricated obligations: the TUI knows nothing structured
		// about files or commands, whatever the policy.
		if _, ok := tc["expected_outputs"]; ok {
			t.Errorf("policy %q made the TUI invent expected_outputs", policy)
		}
		if _, ok := tc["verification"]; ok {
			t.Errorf("policy %q made the TUI invent verification", policy)
		}
	}
}

// The control never edits the user's message, and a question stays a
// question: the proxy refuses mutation authority to a question whatever the
// policy, and the TUI sends the truthful declaration for it to refuse.
func TestCandidatePolicyLeavesTheMessageAndTheModeAlone(t *testing.T) {
	const msg = "Rewrite app.py from scratch."
	for _, mode := range []taskMode{taskModeWork, taskModeQuestion} {
		body := captureContractWithPolicy(t, mode, candidatePolicyAutomatic, msg)
		if body["message"] != msg {
			t.Errorf("mode %q: the message changed to %q", mode, body["message"])
		}
		tc := contractOf(t, body)
		if tc["task_mode"] != string(mode) {
			t.Errorf("mode %q was sent as %v", mode, tc["task_mode"])
		}
	}
}

// Session-wide, and only session-wide.
func TestCandidatePolicyPersistsForTheSessionAndResetsWithANewOne(t *testing.T) {
	m := &tuiModel{}
	m.handleSlash("/candidate-policy automatic")
	if m.candidatePolicy != candidatePolicyAutomatic {
		t.Fatal("selection not recorded")
	}
	// Another command, another message: the selection stays.
	m.handleSlash("/candidate-policy")
	if m.candidatePolicy != candidatePolicyAutomatic {
		t.Error("showing the selection changed it")
	}
	if !strings.Contains(m.chat[len(m.chat)-1].Body, "automatic") {
		t.Errorf("/candidate-policy did not report the current selection: %q", m.chat[len(m.chat)-1].Body)
	}
	m.startNewSession()
	if m.candidatePolicy != candidatePolicyStrict {
		t.Errorf("a new session selected %q, want strict", m.candidatePolicy)
	}
}

// Visible in state and help, in words.
func TestCandidatePolicyIsVisibleInHeaderAndHelp(t *testing.T) {
	if !strings.Contains(slashCommandHelp, "/candidate-policy") {
		t.Error("help does not list /candidate-policy")
	}
	for policy, want := range map[string]string{
		"": "candidates:strict", candidatePolicyAdvisory: "candidates:advisory", candidatePolicyAutomatic: "candidates:automatic",
	} {
		if got := candidatePolicyHeader(policy); got != want {
			t.Errorf("header for %q = %q, want %q", policy, got, want)
		}
	}
	h := renderHeader("http://p", "/w", "default · "+candidatePolicyHeader(candidatePolicyAutomatic), false, 0, 200)
	if !strings.Contains(h, "candidates:automatic") {
		t.Error("the rendered header does not show the selection")
	}
	for _, banned := range []string{"v3.1", "V3.1", "version"} {
		if strings.Contains(candidatePolicyLabel(candidatePolicyAutomatic), banned) {
			t.Errorf("the label carries version language: %q", banned)
		}
	}
}
