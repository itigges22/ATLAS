// PC-062: chat client — POSTs user messages to atlas-proxy /v1/agent
// and consumes the SSE response stream.
//
// Two SSE protocols flow into this TUI:
//
//   /events  — typed Envelope stream (consumer.go) — pipeline visibility
//   /v1/agent — {type, data} chat stream (this file) — assistant reply
//
// The chat stream is request-scoped: each user message opens a fresh
// SSE connection that closes when the agent loop returns [DONE]. This
// is intentional — chat turns are RPC-like, not a persistent feed.

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// chatEvent mirrors the SSEEvent shape in proxy/types.go.
// Different from Envelope — this is the older, simpler chat protocol.
type chatEvent struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

// historyMessage is one prior-turn user/assistant text row sent to the
// proxy on each /v1/agent call. Field shape MUST match the historyMsg
// struct in proxy/agent.go's handleAgent.
type historyMessage struct {
	Role    string `json:"role"`    // "user" or "assistant"
	Content string `json:"content"`
}

// agentRequest is the POST body for /v1/agent. Field tags MUST match
// the anonymous struct in proxy/agent.go's handleAgent.
type agentRequest struct {
	Message           string           `json:"message"`
	WorkingDir        string           `json:"working_dir"`
	Mode              string           `json:"mode"`       // "default" | "accept-edits" | "yolo"
	SessionID         string           `json:"session_id"` // PC-062: required so /cancel can target this turn
	History           []historyMessage `json:"history,omitempty"`
	BypassV3          bool             `json:"bypass_v3,omitempty"`          // /demo raw-pane flag — proxy short-circuits V3 calls
	DisableFreshSlot  bool             `json:"disable_fresh_slot,omitempty"` // /demo flag — skip PC-045 so pre-warm survives
	SandboxSubdir     string           `json:"sandbox_subdir,omitempty"`     // /demo flag — write files into this subdir of the workspace
}

// cancelTurn POSTs /cancel for a session_id. Best-effort: returns
// immediately on connection failure. The TCP-disconnect path in the
// chat client is the primary cancel mechanism; this is defense-in-depth
// for cases where a reverse proxy buffers the disconnect.
func cancelTurn(proxyURL, sessionID string) error {
	if sessionID == "" {
		return fmt.Errorf("empty session id")
	}
	body, _ := json.Marshal(map[string]string{"session_id": sessionID})
	req, err := http.NewRequest("POST",
		strings.TrimRight(proxyURL, "/")+"/cancel", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if tok := loadBearerToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// sendChat opens an SSE POST to /v1/agent and forwards each parsed
// event to out. Returns nil on clean [DONE], err otherwise. Caller is
// responsible for closing the channel after this returns.
//
// history is the prior-turn user/assistant transcript so the agent can
// answer follow-ups. Pass nil for the first turn of a session.
func sendChat(ctx context.Context, proxyURL, message, workingDir, mode,
	sessionID string, history []historyMessage, out chan<- chatEvent) error {
	return sendChatOpts(ctx, proxyURL, message, workingDir, mode,
		sessionID, history, demoOpts{}, out)
}

// demoOpts bundles the per-request flags the /demo split-pane needs.
// Held in a struct rather than added as positional args to keep
// sendChatOpts readable as the demo grows.
type demoOpts struct {
	bypassV3         bool
	disableFreshSlot bool
	sandboxSubdir    string
}

// sendChatOpts is sendChat with the demo flags exposed; used by the
// /demo split-pane to drive a raw-9B session next to the V3 one
// against the same proxy (bypassV3), keep PC-045 from wiping the
// pre-warmed prefix cache (disableFreshSlot), and write files into
// per-side sandbox subdirs so the two panes never clobber each other
// (sandboxSubdir).
func sendChatOpts(ctx context.Context, proxyURL, message, workingDir, mode,
	sessionID string, history []historyMessage,
	opts demoOpts, out chan<- chatEvent) error {

	body, err := json.Marshal(agentRequest{
		Message:          message,
		WorkingDir:       workingDir,
		Mode:             mode,
		SessionID:        sessionID,
		History:          history,
		BypassV3:         opts.bypassV3,
		DisableFreshSlot: opts.disableFreshSlot,
		SandboxSubdir:    opts.sandboxSubdir,
	})
	if err != nil {
		return fmt.Errorf("encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST",
		strings.TrimRight(proxyURL, "/")+"/v1/agent", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if tok := loadBearerToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}

	// No overall timeout — agent turns can run minutes for long
	// generations. Connection-level timeout only.
	client := &http.Client{
		Transport: &http.Transport{
			ResponseHeaderTimeout: 30 * time.Second,
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("connect: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("status %d: %s", resp.StatusCode,
			strings.TrimSpace(string(b)))
	}
	return parseChatSSE(ctx, resp.Body, out)
}

// parseChatSSE reads the chat-protocol SSE stream. Returns nil when it
// sees `data: [DONE]` (clean end-of-turn) or io.EOF.
func parseChatSSE(ctx context.Context, r io.Reader, out chan<- chatEvent) error {
	scanner := bufio.NewScanner(r)
	// tool_result frames carry diff blobs; bump the buffer like
	// consumer.go does for the same reason.
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			return nil
		}
		var ev chatEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue // skip malformed frame, don't kill the turn
		}
		if ev.Type == "" {
			continue
		}
		select {
		case out <- ev:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return scanner.Err()
}

// loadBearerToken returns the bearer token for /v1/agent if a keys
// file is configured. The proxy doesn't currently enforce auth (see
// docs/CONFIGURATION.md — API_KEYS_PATH is "optional"), but the file
// is created by `atlas init` for forward compatibility.
//
// Search order: $ATLAS_API_KEYS_PATH, then ./secrets/api-keys.json
// relative to cwd. Returns "" if no token is found — caller must
// handle the empty case (don't set the header).
func loadBearerToken() string {
	path := os.Getenv("ATLAS_API_KEYS_PATH")
	if path == "" {
		wd, err := os.Getwd()
		if err == nil {
			cand := filepath.Join(wd, "secrets", "api-keys.json")
			if _, err := os.Stat(cand); err == nil {
				path = cand
			}
		}
	}
	if path == "" {
		return ""
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	// File shape (per `atlas init`): {"api_key": "..."} or
	// {"keys": {"tui": "..."}}. Try both.
	var single struct {
		APIKey string `json:"api_key"`
	}
	if err := json.Unmarshal(data, &single); err == nil && single.APIKey != "" {
		return single.APIKey
	}
	var grouped struct {
		Keys map[string]string `json:"keys"`
	}
	if err := json.Unmarshal(data, &grouped); err == nil {
		if v, ok := grouped.Keys["tui"]; ok && v != "" {
			return v
		}
	}
	return ""
}
