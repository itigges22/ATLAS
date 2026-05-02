// PC-062: SSE event consumer for the TUI.
//
// Mirrors the Envelope struct in atlas-proxy/events.go and the Python
// dataclass in atlas/cli/events.py. The schema contract is documented
// in docs/PROTOCOL.md — any change here MUST be made in lockstep
// across all three implementations.
//
// The TUI receives Envelope events on a channel and feeds them into
// the Bubbletea model via tea.Cmd / tea.Msg. The connection is
// resumable: if /events drops, we reconnect with exponential backoff.

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Envelope is the wire-format event. Field tags MUST match
// atlas/cli/events.py and atlas-proxy/events.go exactly.
type Envelope struct {
	EventID    string                 `json:"event_id"`
	Timestamp  float64                `json:"timestamp"`
	Type       string                 `json:"type"`
	Stage      string                 `json:"stage"`
	Payload    map[string]interface{} `json:"payload"`
	ParentID   string                 `json:"parent_id,omitempty"`
	DurationMS int64                  `json:"duration_ms,omitempty"`
}

// Legal types — mirror atlas.cli.events.EVENT_TYPES.
const (
	EvtStageStart = "stage_start"
	EvtStageEnd   = "stage_end"
	EvtToolCall   = "tool_call"
	EvtToolResult = "tool_result"
	EvtMetric     = "metric"
	EvtError      = "error"
	EvtDone       = "done"
)

// streamEvents connects to the /events SSE endpoint and pushes parsed
// envelopes onto out. Returns when ctx is cancelled or the connection
// dies unrecoverably. The TUI's reconnect loop wraps this.
func streamEvents(ctx context.Context, eventsURL string, out chan<- Envelope) error {
	req, err := http.NewRequestWithContext(ctx, "GET", eventsURL, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Accept", "text/event-stream")

	client := &http.Client{
		// No timeout on the response body — SSE streams indefinitely.
		// Connect timeout is the only meaningful deadline.
		Transport: &http.Transport{
			ResponseHeaderTimeout: 5 * time.Second,
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("connect: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %d", resp.StatusCode)
	}

	return parseSSE(ctx, resp.Body, out)
}

// parseSSE reads SSE frames from r and pushes parsed envelopes onto out.
// Skips comment lines (`: connected`, `: heartbeat`) and named events
// (`event: result`) since those are control frames, not envelopes —
// see docs/PROTOCOL.md "SSE control frames".
func parseSSE(ctx context.Context, r io.Reader, out chan<- Envelope) error {
	scanner := bufio.NewScanner(r)
	// SSE frames can contain large JSON blobs. The default 64KB buffer
	// is too small for tool_result payloads with embedded file diffs.
	scanner.Buffer(make([]byte, 0, 64*1024), 1*1024*1024)

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, ":") {
			continue // blank line = frame boundary; `:` = SSE comment
		}
		if strings.HasPrefix(line, "event:") {
			continue // named events (`event: result`) are legacy v3-service framing
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}

		var ev Envelope
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			// Malformed envelope — skip rather than killing the stream.
			// Could be the legacy `{stage, detail}` shape from v3-service
			// emitting in dual-mode without our opt-in header. We just
			// don't render those.
			continue
		}
		if ev.Type == "" || ev.EventID == "" {
			continue // legacy or unrecognized
		}

		select {
		case out <- ev:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return scanner.Err()
}

// streamEventsWithReconnect wraps streamEvents with exponential backoff
// reconnection. Useful when the proxy restarts mid-session — TUI keeps
// running rather than dying on the first dropped connection.
func streamEventsWithReconnect(ctx context.Context, eventsURL string, out chan<- Envelope) {
	backoff := 500 * time.Millisecond
	maxBackoff := 30 * time.Second
	for {
		err := streamEvents(ctx, eventsURL, out)
		if ctx.Err() != nil {
			return // user quit / TUI shutdown
		}
		// Connection died; back off and retry.
		dlog("conn", "events_disconnected", map[string]interface{}{
			"err": fmt.Sprintf("%v", err), "backoff_ms": backoff.Milliseconds(),
		})
		select {
		case <-time.After(backoff):
		case <-ctx.Done():
			return
		}
		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}
