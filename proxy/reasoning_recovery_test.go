package main

import "testing"

func TestRecoverStructuredReasoningAcceptsWhitespaceFormattedText(t *testing.T) {
	raw := "{\n  \"type\": \"text\",\n  \"content\": \"agent-ok\"\n}"
	recovered, ok := recoverStructuredReasoning(raw)
	if !ok {
		t.Fatal("valid text envelope in reasoning_content was discarded")
	}
	parsed, err := extractModelResponse(recovered)
	if err != nil || parsed.Type != "text" || parsed.Content != "agent-ok" {
		t.Fatalf("recovered response = %#v, err=%v", parsed, err)
	}
}

func TestRecoverStructuredReasoningRejectsNarration(t *testing.T) {
	if recovered, ok := recoverStructuredReasoning("I should inspect the repository first."); ok {
		t.Fatalf("pure narration recovered as agent response: %q", recovered)
	}
}

func TestRecoverStructuredReasoningAcceptsDoneAndToolCall(t *testing.T) {
	for _, raw := range []string{
		`{"type": "done", "summary": "finished"}`,
		`{"args": {"path": "."}, "name": "list_directory", "type": "tool_call"}`,
	} {
		if _, ok := recoverStructuredReasoning(raw); !ok {
			t.Fatalf("valid structured response was discarded: %s", raw)
		}
	}
}
