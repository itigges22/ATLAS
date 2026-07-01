// PC-062: chat clients for the ATLAS agent stream and the demo's one-shot
// raw OpenAI-compatible stream.
//
// Two SSE protocols flow into this TUI:
//
//   /events  — typed Envelope stream (consumer.go) — pipeline visibility
//   /v1/agent — {type, data} chat stream (this file) — assistant reply
//   /v1/chat/completions — standard OpenAI SSE used only by the raw demo pane
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
	"sort"
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
	Role    string `json:"role"` // "user" or "assistant"
	Content string `json:"content"`
}

// agentRequest is the POST body for /v1/agent. Field tags MUST match
// the anonymous struct in proxy/agent.go's handleAgent.
type agentRequest struct {
	Message          string           `json:"message"`
	WorkingDir       string           `json:"working_dir"`
	Mode             string           `json:"mode"`       // "default" | "accept-edits" | "yolo"
	SessionID        string           `json:"session_id"` // PC-062: required so /cancel can target this turn
	History          []historyMessage `json:"history,omitempty"`
	DisableFreshSlot bool             `json:"disable_fresh_slot,omitempty"` // /demo flag — skip PC-045 so pre-warm survives
	SandboxSubdir    string           `json:"sandbox_subdir,omitempty"`     // /demo flag — write files into this subdir of the workspace
}

type rawChatRequest struct {
	Model              string              `json:"model"`
	Messages           []map[string]string `json:"messages"`
	Temperature        float64             `json:"temperature"`
	MaxTokens          int                 `json:"max_tokens"`
	Stream             bool                `json:"stream"`
	StreamOptions      map[string]bool     `json:"stream_options"`
	ChatTemplateKwargs map[string]bool     `json:"chat_template_kwargs"`
}

// sendRawChat makes one direct OpenAI-compatible model request. It never uses
// /v1/agent, so the raw demo pane has no planner, tool loop, candidate search,
// filesystem access, or other orchestration layer.
func sendRawChat(ctx context.Context, proxyURL, modelID, message string,
	out chan<- chatEvent) error {
	if strings.TrimSpace(modelID) == "" {
		modelID = "local-model"
	}
	promptTokens := len(message) / 4
	out <- makeChatEvent("llm_call_start", map[string]int{"prompt_tokens": promptTokens})

	body, err := json.Marshal(rawChatRequest{
		Model:              modelID,
		Messages:           []map[string]string{{"role": "user", "content": message}},
		Temperature:        0.3,
		MaxTokens:          8192,
		Stream:             true,
		StreamOptions:      map[string]bool{"include_usage": true},
		ChatTemplateKwargs: map[string]bool{"enable_thinking": false},
	})
	if err != nil {
		return fmt.Errorf("encode raw request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(proxyURL, "/")+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build raw request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if tok := loadBearerToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}

	started := time.Now()
	resp, err := (&http.Client{Transport: &http.Transport{
		ResponseHeaderTimeout: 60 * time.Second,
	}}).Do(req)
	if err != nil {
		return fmt.Errorf("raw model request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		responseBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("raw model returned %d: %s", resp.StatusCode,
			strings.TrimSpace(string(responseBody)))
	}

	var content strings.Builder
	var reasoning strings.Builder
	firstToken := true
	totalTokens := 0
	completionTokens := 0
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
			} `json:"choices"`
			Usage struct {
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage"`
		}
		if json.Unmarshal([]byte(payload), &chunk) != nil {
			continue
		}
		if chunk.Usage.TotalTokens > 0 {
			totalTokens = chunk.Usage.TotalTokens
			completionTokens = chunk.Usage.CompletionTokens
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta
		if firstToken && (delta.Content != "" || delta.ReasoningContent != "") {
			firstToken = false
			out <- makeChatEvent("llm_first_token", map[string]int64{
				"prompt_ms": time.Since(started).Milliseconds(),
			})
		}
		if delta.ReasoningContent != "" {
			reasoning.WriteString(delta.ReasoningContent)
			out <- makeChatEvent("reasoning_token", map[string]string{"text": delta.ReasoningContent})
		}
		if delta.Content != "" {
			content.WriteString(delta.Content)
			out <- makeChatEvent("llm_token", map[string]string{"text": delta.Content})
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read raw model stream: %w", err)
	}
	if content.Len() == 0 && reasoning.Len() > 0 {
		content.WriteString(reasoning.String())
	}
	if completionTokens == 0 {
		completionTokens = len(content.String()) / 4
	}
	if totalTokens == 0 {
		totalTokens = promptTokens + completionTokens
	}
	out <- makeChatEvent("llm_call_end", map[string]interface{}{
		"tokens":       totalTokens,
		"total_tokens": totalTokens,
		"chars":        content.Len(),
		"ms":           time.Since(started).Milliseconds(),
	})
	if content.Len() > 0 {
		out <- makeChatEvent("text", map[string]string{"content": content.String()})
	}
	return nil
}

func makeChatEvent(eventType string, payload interface{}) chatEvent {
	data, _ := json.Marshal(payload)
	return chatEvent{Type: eventType, Data: data}
}

// cancelTurn POSTs /cancel for a session_id. Best-effort: returns
// immediately on connection failure. The TCP-disconnect path in the
// chat client is the primary cancel mechanism; this is defense-in-depth
// for cases where a reverse proxy buffers the disconnect.
// fileVerdict is a per-file accept/deny for the post-pass review. Shape MUST
// match the proxy's /feedback `files` entries.
type fileVerdict struct {
	Path    string `json:"path"`
	Verdict string `json:"verdict"` // "accept" | "deny"
}

// submitFeedback posts a pass verdict to the proxy, which turns the pass's
// writes into labeled lens-training samples. `thumbs` is the pass-level 👍/👎;
// `files` carries any per-file accept/deny (deny → confident negative; the
// rest ride the thumbs weight). Returns the number of samples recorded.
func submitFeedback(proxyURL, sessionID, thumbs string, files []fileVerdict) (int, error) {
	if sessionID == "" {
		return 0, fmt.Errorf("no completed pass to rate yet")
	}
	payload := map[string]interface{}{"session_id": sessionID, "thumbs": thumbs}
	if len(files) > 0 {
		payload["files"] = files
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequest("POST",
		strings.TrimRight(proxyURL, "/")+"/feedback", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if tok := loadBearerToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := (&http.Client{Timeout: 5 * time.Second}).Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return 0, fmt.Errorf("feedback returned %d: %s", resp.StatusCode,
			strings.TrimSpace(string(b)))
	}
	var r struct {
		Recorded int `json:"recorded"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&r)
	return r.Recorded, nil
}

// trainingStatus mirrors the proxy's /v1/lens/training-status payload.
type trainingStatus struct {
	Total            int    `json:"total"`
	Good             int    `json:"good"`
	Bad              int    `json:"bad"`
	Threshold        int    `json:"threshold"`
	RetrainAvailable bool   `json:"retrain_available"`
	Command          string `json:"command"`
}

// lensRetrainStatusMsg carries a training-status poll result back to Update
// (after a pass completes) so the model can surface the "retrain available"
// banner without blocking on the HTTP call.
type lensRetrainStatusMsg struct{ status trainingStatus }

// fetchTrainingStatus asks the proxy how many labeled samples have accumulated
// and whether a retrain is worth offering. Best-effort: errors are returned so
// the caller can simply skip the banner.
func fetchTrainingStatus(proxyURL string) (trainingStatus, error) {
	var ts trainingStatus
	req, err := http.NewRequest("GET",
		strings.TrimRight(proxyURL, "/")+"/v1/lens/training-status", nil)
	if err != nil {
		return ts, err
	}
	if tok := loadBearerToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := (&http.Client{Timeout: 3 * time.Second}).Do(req)
	if err != nil {
		return ts, err
	}
	defer resp.Body.Close()
	err = json.NewDecoder(resp.Body).Decode(&ts)
	return ts, err
}

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
	disableFreshSlot bool
	sandboxSubdir    string
}

// sendChatOpts is sendChat with the demo flags exposed; used by the
// /demo split-pane's V3 side to keep PC-045 from wiping the
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
	// File shapes: {"api_key": "..."}, {"keys": {"tui": "..."}}, or the
	// `atlas init` shape {"sk-…": {"user": "local", …}} where each top-
	// level key IS a token mapping to a metadata object. Try in order.
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
	// Token-keyed shape ({"sk-…": {"user": "local", …}}): accept only
	// entries whose key looks like a token or whose value carries key
	// metadata, so an unrelated object-of-objects (e.g. {"keys": {...}})
	// is not mistaken for a token map. Use the lexicographically first
	// qualifying token so repeated loads pick the same one.
	var keyed map[string]map[string]interface{}
	if err := json.Unmarshal(data, &keyed); err == nil && len(keyed) > 0 {
		tokens := make([]string, 0, len(keyed))
		for k, meta := range keyed {
			_, hasUser := meta["user"]
			_, hasCreatedBy := meta["created_by"]
			if strings.HasPrefix(k, "sk-") || hasUser || hasCreatedBy {
				tokens = append(tokens, k)
			}
		}
		if len(tokens) > 0 {
			sort.Strings(tokens)
			return tokens[0]
		}
	}
	return ""
}
