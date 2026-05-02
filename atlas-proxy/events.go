// PC-061: typed event protocol for atlas-proxy.
//
// Mirrors the schema defined in atlas/cli/events.py so the same Python
// consumer (atlas/cli/events.py) can read events from both v3-service
// and atlas-proxy. The envelope shape is documented in docs/PROTOCOL.md.
//
// Architecture: a global pub/sub broker. Producers (agent loop, tool
// dispatch, etc.) call Emit() with an Envelope. Subscribers (SSE clients
// connected to /events) get a buffered channel; slow consumers are
// dropped from individual events rather than blocking producers.
//
// SSE-only transport. Cancellation is out of scope for this ticket
// (separate POST /cancel will follow when the TUI lands).

package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// Envelope is the wire-format event. JSON tags match the Python dataclass
// in atlas/cli/events.py exactly — any change here MUST also be made there.
type Envelope struct {
	EventID    string                 `json:"event_id"`
	Timestamp  float64                `json:"timestamp"`
	Type       string                 `json:"type"`
	Stage      string                 `json:"stage"`
	Payload    map[string]interface{} `json:"payload"`
	ParentID   string                 `json:"parent_id,omitempty"`
	DurationMS int64                  `json:"duration_ms,omitempty"`
}

// Legal Type values. Mirror atlas.cli.events.EVENT_TYPES.
const (
	EvtStageStart  = "stage_start"
	EvtStageEnd    = "stage_end"
	EvtToolCall    = "tool_call"
	EvtToolResult  = "tool_result"
	EvtMetric      = "metric"
	EvtError       = "error"
	EvtDone        = "done"
)

// NewEventID returns a short, log-readable, session-unique id. Same format
// as the Python helper: "evt_" + 8 hex chars.
func NewEventID() string {
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		// fall back to time-based to never block — collisions are not a
		// correctness concern (event_ids are for parent_id pairing only)
		return fmt.Sprintf("evt_%08x", time.Now().UnixNano()&0xffffffff)
	}
	return "evt_" + hex.EncodeToString(b[:])
}

// NewEnvelope builds a well-formed envelope with sensible defaults.
func NewEnvelope(typ, stage string, payload map[string]interface{}) Envelope {
	if payload == nil {
		payload = map[string]interface{}{}
	}
	return Envelope{
		EventID:   NewEventID(),
		Timestamp: float64(time.Now().UnixNano()) / 1e9,
		Type:      typ,
		Stage:     stage,
		Payload:   payload,
	}
}

// ---------------------------------------------------------------------------
// Broker
// ---------------------------------------------------------------------------

const subscriberBuffer = 256 // events buffered per slow subscriber before drops kick in

type broker struct {
	mu          sync.Mutex
	subscribers map[chan Envelope]struct{}
}

var defaultBroker = &broker{
	subscribers: map[chan Envelope]struct{}{},
}

func (b *broker) subscribe() chan Envelope {
	ch := make(chan Envelope, subscriberBuffer)
	b.mu.Lock()
	b.subscribers[ch] = struct{}{}
	b.mu.Unlock()
	return ch
}

func (b *broker) unsubscribe(ch chan Envelope) {
	b.mu.Lock()
	delete(b.subscribers, ch)
	b.mu.Unlock()
	close(ch)
}

func (b *broker) emit(ev Envelope) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for ch := range b.subscribers {
		// Non-blocking send. Slow consumers lose events — never block
		// the agent loop on a wedged TUI.
		select {
		case ch <- ev:
		default:
		}
	}
}

// Emit publishes an envelope to all subscribers. Safe from any goroutine.
func Emit(ev Envelope) {
	defaultBroker.emit(ev)
}

// EmitSimple is a convenience for the common stage_start / stage_end /
// metric cases — saves callers from constructing the map themselves.
func EmitSimple(typ, stage string, detail string) {
	payload := map[string]interface{}{}
	if detail != "" {
		payload["detail"] = detail
	}
	Emit(NewEnvelope(typ, stage, payload))
}

// ---------------------------------------------------------------------------
// HTTP handler
// ---------------------------------------------------------------------------

// handleEvents serves a continuous SSE stream of envelope events. Closes
// when the client disconnects (Request.Context().Done()).
func handleEvents(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	ch := defaultBroker.subscribe()
	defer defaultBroker.unsubscribe(ch)

	// Heartbeat ticker — keeps proxies / load balancers from idling out
	// the connection during quiet stretches in the agent loop.
	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case ev, open := <-ch:
			if !open {
				return
			}
			line, err := json.Marshal(ev)
			if err != nil {
				continue
			}
			if _, err := fmt.Fprintf(w, "data: %s\n\n", line); err != nil {
				return
			}
			flusher.Flush()
		case <-heartbeat.C:
			if _, err := fmt.Fprint(w, ": heartbeat\n\n"); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}
