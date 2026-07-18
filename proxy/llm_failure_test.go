// Tests for callLLMOnce's failure handling against a fake llama-server:
// HTTP errors, unreachable server, truncated streams, mid-stream
// cancellation, and the reasoning_content fallback. The agent loop's
// resilience depends on these paths returning promptly with a
// classifiable error instead of hanging or panicking.

package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func llmTestCtx(url string) *AgentContext {
	return &AgentContext{
		InferenceURL: url,
		Ctx:          context.Background(),
		Messages:     []AgentMessage{{Role: "user", Content: "hi"}},
	}
}

func sseWrite(w http.ResponseWriter, lines ...string) {
	fl, _ := w.(http.Flusher)
	for _, l := range lines {
		io.WriteString(w, l+"\n\n")
		if fl != nil {
			fl.Flush()
		}
	}
}

func TestCallLLMOnce_HTTPErrorSurfacesStatusAndBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, `{"error":{"message":"model loading"}}`,
				http.StatusInternalServerError)
		}))
	defer srv.Close()

	ctx := llmTestCtx(srv.URL)
	_, _, err := callLLMOnce(ctx, ctx.Messages, 0.3)
	if err == nil {
		t.Fatal("500 from llama-server produced no error")
	}
	if !strings.Contains(err.Error(), "LLM returned 500") {
		t.Errorf("error %q does not name the status code", err)
	}
	// The response body is part of the error so the agent loop (and the
	// user-facing failure event) can say WHY llama-server refused.
	if !strings.Contains(err.Error(), "model loading") {
		t.Errorf("error %q drops the server's explanation", err)
	}
}

func TestCallLLMOnce_UnreachableServerFailsFast(t *testing.T) {
	// A server that existed and is gone — connection refused territory.
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close()

	ctx := llmTestCtx(url)
	start := time.Now()
	_, _, err := callLLMOnce(ctx, ctx.Messages, 0.3)
	if err == nil {
		t.Fatal("dead llama-server produced no error")
	}
	if !strings.Contains(err.Error(), "LLM request failed") {
		t.Errorf("error %q is not the request-failure classification", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("failure took %v — should fail fast, not wait on a timeout", elapsed)
	}
}

func TestCallLLMOnce_TruncatedStreamReturnsPartialContent(t *testing.T) {
	// Server streams one delta then ends the response without [DONE] —
	// the shape of a llama-server crash mid-generation. Current contract:
	// clean EOF ends the scan without error and the partial content is
	// returned, so the caller's parse step decides what to do with it.
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			sseWrite(w, `data: {"choices":[{"delta":{"content":"{\"type\":\"do"}}]}`)
			// no [DONE], no usage — connection just ends
		}))
	defer srv.Close()

	ctx := llmTestCtx(srv.URL)
	content, _, err := callLLMOnce(ctx, ctx.Messages, 0.3)
	if err != nil {
		t.Fatalf("clean-EOF truncation returned error: %v", err)
	}
	if content != `{"type":"do` {
		t.Errorf("partial content = %q, want the streamed prefix", content)
	}
}

func TestCallLLMOnce_ContextCancelAbortsStalledStream(t *testing.T) {
	// Server sends one token then stalls forever. Cancelling the agent
	// context (what /cancel does) must abort the read promptly with the
	// stream-read classification, returning whatever was accumulated.
	// The `release` channel unblocks the handler after the call returns —
	// srv.Close() waits for active handlers, and server-side disconnect
	// detection isn't reliable enough to end the stall on its own.
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			sseWrite(w, `data: {"choices":[{"delta":{"content":"partial"}}]}`)
			select {
			case <-r.Context().Done():
			case <-release:
			}
		}))
	defer srv.Close()
	// LIFO: this runs BEFORE srv.Close(), releasing the stalled handler
	// so Close doesn't wait forever on it.
	defer close(release)

	cancelCtx, cancel := context.WithCancel(context.Background())
	ctx := llmTestCtx(srv.URL)
	ctx.Ctx = cancelCtx

	go func() {
		time.Sleep(150 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	content, _, err := callLLMOnce(ctx, ctx.Messages, 0.3)
	if err == nil {
		t.Fatal("cancelled stream returned no error")
	}
	if !strings.Contains(err.Error(), "read LLM stream") {
		t.Errorf("error %q is not the stream-read classification", err)
	}
	if content != "partial" {
		t.Errorf("accumulated content = %q, want %q", content, "partial")
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Errorf("cancel took %v to unblock the call", elapsed)
	}
}

func TestCallLLMOnce_ReasoningOnlyStreamRecoversToolCall(t *testing.T) {
	// Model ignores enable_thinking=false and streams everything as
	// reasoning_content, including the JSON tool call. The empty-content
	// fallback must recover the structured envelope.
	payload := `{\"type\":\"tool_call\",\"name\":\"read_file\",\"args\":{\"path\":\"main.go\"}}`
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			sseWrite(w,
				`data: {"choices":[{"delta":{"reasoning_content":"I should read the file. "}}]}`,
				`data: {"choices":[{"delta":{"reasoning_content":"`+payload+`"}}]}`,
				`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":9}}`,
				`data: [DONE]`)
		}))
	defer srv.Close()

	ctx := llmTestCtx(srv.URL)
	content, tokens, err := callLLMOnce(ctx, ctx.Messages, 0.3)
	if err != nil {
		t.Fatalf("reasoning-only stream errored: %v", err)
	}
	if !strings.Contains(content, `"read_file"`) {
		t.Errorf("recovered content %q lost the tool call", content)
	}
	if tokens != 9 {
		t.Errorf("total tokens = %d, want 9 from the usage block", tokens)
	}
}

func TestCallLLMOnce_ReasoningOnlyProseReturnsEmpty(t *testing.T) {
	// Pure narration with no tool call must come back EMPTY so the
	// caller's re-prompt fires — returning the prose would just
	// parse-error and waste the turn.
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			sseWrite(w,
				`data: {"choices":[{"delta":{"reasoning_content":"Now I need to look at the file and think about it more."}}]}`,
				`data: [DONE]`)
		}))
	defer srv.Close()

	ctx := llmTestCtx(srv.URL)
	content, _, err := callLLMOnce(ctx, ctx.Messages, 0.3)
	if err != nil {
		t.Fatalf("prose-only stream errored: %v", err)
	}
	if content != "" {
		t.Errorf("prose-only reasoning returned %q, want empty so the caller re-prompts", content)
	}
	if ctx.LastTurnReasoning == "" {
		t.Error("reasoning was not stashed on ctx for the repetition detector")
	}
}
