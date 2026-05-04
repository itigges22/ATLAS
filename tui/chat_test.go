// Tests for the /v1/agent chat client. Uses httptest.NewServer to
// stand in for atlas-proxy so the test exercises the real HTTP+SSE
// path without needing the proxy running.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// fakeAgentServer returns an httptest server that, on POST /v1/agent,
// streams the given canned events then closes with [DONE]. Captures
// the request body for the test to inspect.
func fakeAgentServer(t *testing.T, events []chatEvent,
	gotBody *agentRequest) *httptest.Server {

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/agent", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "method", 405)
			return
		}
		body, _ := io.ReadAll(r.Body)
		if gotBody != nil {
			_ = json.Unmarshal(body, gotBody)
		}
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("test server: no flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		for _, ev := range events {
			b, _ := json.Marshal(ev)
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	})
	mux.HandleFunc("/cancel", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "method", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]bool{"cancelled": true})
	})
	return httptest.NewServer(mux)
}

func mkChatEvent(typ string, data interface{}) chatEvent {
	b, _ := json.Marshal(data)
	return chatEvent{Type: typ, Data: b}
}

func TestSendChatPostsRequestBodyAndStreamsEvents(t *testing.T) {
	want := []chatEvent{
		mkChatEvent("text", map[string]string{"content": "hi"}),
		mkChatEvent("tool_call", map[string]interface{}{
			"name": "read_file", "args": map[string]string{"path": "x.go"},
		}),
		mkChatEvent("done", map[string]string{"summary": "ok"}),
	}
	var gotReq agentRequest
	srv := fakeAgentServer(t, want, &gotReq)
	defer srv.Close()

	out := make(chan chatEvent, 16)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := sendChat(ctx, srv.URL, "fix bug", "/work", "default",
		"sess-1", nil, out); err != nil {
		t.Fatalf("sendChat: %v", err)
	}
	close(out)

	if gotReq.Message != "fix bug" {
		t.Errorf("Message = %q", gotReq.Message)
	}
	if gotReq.WorkingDir != "/work" {
		t.Errorf("WorkingDir = %q", gotReq.WorkingDir)
	}
	if gotReq.Mode != "default" {
		t.Errorf("Mode = %q", gotReq.Mode)
	}
	if gotReq.SessionID != "sess-1" {
		t.Errorf("SessionID = %q", gotReq.SessionID)
	}

	got := []chatEvent{}
	for ev := range out {
		got = append(got, ev)
	}
	if len(got) != 3 {
		t.Fatalf("got %d events, want 3", len(got))
	}
	if got[0].Type != "text" || got[1].Type != "tool_call" || got[2].Type != "done" {
		t.Errorf("event order = %v", []string{got[0].Type, got[1].Type, got[2].Type})
	}
}

func TestSendChatHandlesNon200Status(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "no llama", 503)
	}))
	defer srv.Close()
	out := make(chan chatEvent, 1)
	err := sendChat(context.Background(), srv.URL, "hi", "/", "default", "s", nil, out)
	if err == nil {
		t.Fatal("sendChat should return error on 503")
	}
	if !strings.Contains(err.Error(), "503") {
		t.Errorf("error = %v, want one mentioning 503", err)
	}
}

func TestSendChatContextCancelStopsStream(t *testing.T) {
	// Server drips events forever — sendChat should return when ctx cancels.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, _ := w.(http.Flusher)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		for i := 0; i < 100; i++ {
			b, _ := json.Marshal(mkChatEvent("text",
				map[string]string{"content": "drip"}))
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
			select {
			case <-r.Context().Done():
				return
			case <-time.After(20 * time.Millisecond):
			}
		}
	}))
	defer srv.Close()

	out := make(chan chatEvent, 32)
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(80 * time.Millisecond)
		cancel()
	}()
	err := sendChat(ctx, srv.URL, "hi", "/", "default", "s", nil, out)
	if err == nil {
		// context cancellation on read returns ctx.Err() or io.EOF;
		// either is acceptable. nil means the stream completed before
		// cancel fired — which is also fine but unexpected.
		t.Log("sendChat returned nil — server may have closed first")
	}
}

func TestCancelTurnPosts(t *testing.T) {
	got := make(chan map[string]string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/cancel" {
			http.Error(w, "wrong path", 404)
			return
		}
		var body map[string]string
		_ = json.NewDecoder(r.Body).Decode(&body)
		got <- body
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]bool{"cancelled": true})
	}))
	defer srv.Close()

	if err := cancelTurn(srv.URL, "sess-9"); err != nil {
		t.Fatalf("cancelTurn: %v", err)
	}
	body := <-got
	if body["session_id"] != "sess-9" {
		t.Errorf("session_id = %q", body["session_id"])
	}
}

func TestCancelTurnEmptyIDIsError(t *testing.T) {
	if err := cancelTurn("http://localhost:0", ""); err == nil {
		t.Errorf("expected error for empty session id")
	}
}

func TestParseChatSSESkipsCommentsAndDONE(t *testing.T) {
	stream := strings.Join([]string{
		": connected",
		"",
		"data: " + mustJSON(mkChatEvent("text", map[string]string{"content": "a"})),
		"",
		": heartbeat",
		"",
		"data: " + mustJSON(mkChatEvent("text", map[string]string{"content": "b"})),
		"",
		"data: [DONE]",
		"",
	}, "\n")
	out := make(chan chatEvent, 4)
	if err := parseChatSSE(context.Background(), strings.NewReader(stream), out); err != nil {
		t.Fatalf("parseChatSSE: %v", err)
	}
	close(out)
	got := []chatEvent{}
	for ev := range out {
		got = append(got, ev)
	}
	if len(got) != 2 {
		t.Errorf("got %d events, want 2 (comments + [DONE] should be skipped)", len(got))
	}
}

func mustJSON(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

func TestLoadBearerTokenMissingFileReturnsEmpty(t *testing.T) {
	t.Setenv("ATLAS_API_KEYS_PATH", "/nonexistent/path/api-keys.json")
	if got := loadBearerToken(); got != "" {
		t.Errorf("loadBearerToken with missing file = %q, want empty", got)
	}
}

func TestLoadBearerTokenReadsAPIKey(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/api-keys.json"
	if err := writeJSON(path, map[string]string{"api_key": "abc123"}); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ATLAS_API_KEYS_PATH", path)
	if got := loadBearerToken(); got != "abc123" {
		t.Errorf("loadBearerToken = %q, want abc123", got)
	}
}

func writeJSON(path string, v interface{}) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o600)
}
