// Tests for the typed event broker + envelope shape (PC-061).

package main

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestNewEventIDFormat(t *testing.T) {
	id := NewEventID()
	if !strings.HasPrefix(id, "evt_") {
		t.Fatalf("event id missing 'evt_' prefix: %q", id)
	}
	if len(id) != len("evt_")+8 {
		t.Fatalf("event id wrong length: %q (len=%d)", id, len(id))
	}
}

func TestNewEventIDsAreUnique(t *testing.T) {
	seen := map[string]struct{}{}
	for i := 0; i < 1000; i++ {
		id := NewEventID()
		if _, dup := seen[id]; dup {
			t.Fatalf("duplicate event id at iteration %d: %q", i, id)
		}
		seen[id] = struct{}{}
	}
}

func TestNewEnvelopeFieldsAreSet(t *testing.T) {
	ev := NewEnvelope(EvtStageStart, "phase2", map[string]interface{}{"detail": "x"})
	if ev.Type != EvtStageStart {
		t.Fatalf("type = %q, want %q", ev.Type, EvtStageStart)
	}
	if ev.Stage != "phase2" {
		t.Fatalf("stage = %q, want phase2", ev.Stage)
	}
	if ev.Payload["detail"] != "x" {
		t.Fatalf("payload not set: %+v", ev.Payload)
	}
	if ev.EventID == "" {
		t.Fatal("event id empty")
	}
	if ev.Timestamp <= 0 {
		t.Fatalf("timestamp = %v, want > 0", ev.Timestamp)
	}
}

func TestEnvelopeJSONShapeMatchesPythonSchema(t *testing.T) {
	// The Python consumer (atlas/cli/events.py) requires these fields
	// in this exact spelling. Schema mismatch here breaks the consumer
	// silently — pin the field names.
	ev := NewEnvelope(EvtStageEnd, "phase2", map[string]interface{}{
		"success": true,
	})
	ev.ParentID = "evt_aabbccdd"
	ev.DurationMS = 4523

	raw, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var roundtrip map[string]interface{}
	if err := json.Unmarshal(raw, &roundtrip); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, key := range []string{"event_id", "timestamp", "type", "stage",
		"payload", "parent_id", "duration_ms"} {
		if _, ok := roundtrip[key]; !ok {
			t.Errorf("missing required field %q in JSON: %s", key, raw)
		}
	}
	if roundtrip["type"] != "stage_end" {
		t.Errorf("type field misspelled: %v", roundtrip["type"])
	}
}

func TestEnvelopeOmitsEmptyOptionalFields(t *testing.T) {
	// When parent_id and duration_ms aren't set, they should be omitted
	// from the JSON (mirrors Python `to_dict()` behavior).
	ev := NewEnvelope(EvtStageStart, "phase2", nil)
	raw, _ := json.Marshal(ev)
	var rt map[string]interface{}
	_ = json.Unmarshal(raw, &rt)
	if _, has := rt["parent_id"]; has {
		t.Errorf("parent_id should be omitted when empty, got %v", rt["parent_id"])
	}
	if _, has := rt["duration_ms"]; has {
		t.Errorf("duration_ms should be omitted when zero, got %v", rt["duration_ms"])
	}
}

// ---------------------------------------------------------------------------
// Broker
// ---------------------------------------------------------------------------

func TestBrokerSubscribersReceiveEmittedEvents(t *testing.T) {
	b := &broker{subscribers: map[chan Envelope]struct{}{}}
	ch := b.subscribe()
	defer b.unsubscribe(ch)

	go b.emit(NewEnvelope(EvtMetric, "lens", map[string]interface{}{"name": "gx", "value": 0.83}))

	select {
	case ev := <-ch:
		if ev.Type != EvtMetric {
			t.Fatalf("got %q, want %q", ev.Type, EvtMetric)
		}
	case <-time.After(time.Second):
		t.Fatal("subscriber didn't receive event in 1s")
	}
}

func TestBrokerFanOutToMultipleSubscribers(t *testing.T) {
	b := &broker{subscribers: map[chan Envelope]struct{}{}}
	subs := []chan Envelope{b.subscribe(), b.subscribe(), b.subscribe()}
	defer func() {
		for _, ch := range subs {
			b.unsubscribe(ch)
		}
	}()

	b.emit(NewEnvelope(EvtStageStart, "phase2", nil))

	for i, ch := range subs {
		select {
		case ev := <-ch:
			if ev.Stage != "phase2" {
				t.Errorf("subscriber %d: stage = %q, want phase2", i, ev.Stage)
			}
		case <-time.After(time.Second):
			t.Errorf("subscriber %d didn't receive event", i)
		}
	}
}

func TestBrokerUnsubscribeStopsDelivery(t *testing.T) {
	b := &broker{subscribers: map[chan Envelope]struct{}{}}
	ch := b.subscribe()
	b.unsubscribe(ch)

	// After unsubscribe, the channel is closed. Emitting must not
	// panic by trying to send on a closed channel.
	b.emit(NewEnvelope(EvtMetric, "x", nil))

	// And the channel must drain to closed (not block).
	for range ch {
	}
}

func TestBrokerSlowConsumerDoesntBlockProducer(t *testing.T) {
	b := &broker{subscribers: map[chan Envelope]struct{}{}}
	slow := b.subscribe() // never reads
	defer b.unsubscribe(slow)

	// Emit more events than the buffer can hold. Producer must not block.
	done := make(chan struct{})
	go func() {
		for i := 0; i < subscriberBuffer*4; i++ {
			b.emit(NewEnvelope(EvtMetric, "x", nil))
		}
		close(done)
	}()
	select {
	case <-done:
		// Drain slow consumer to verify it got at least the buffered events.
		count := 0
	loop:
		for {
			select {
			case <-slow:
				count++
			default:
				break loop
			}
		}
		if count == 0 {
			t.Error("slow consumer got nothing — drop logic too aggressive")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("producer blocked on slow consumer")
	}
}

func TestBrokerConcurrentSubscribeUnsubscribeIsRaceFree(t *testing.T) {
	// Run with `go test -race` in CI to actually catch a race; this test
	// just ensures the operations don't crash under concurrent load.
	b := &broker{subscribers: map[chan Envelope]struct{}{}}
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ch := b.subscribe()
			b.emit(NewEnvelope(EvtMetric, "x", nil))
			b.unsubscribe(ch)
		}()
	}
	wg.Wait()
}

// ---------------------------------------------------------------------------
// EmitSimple convenience
// ---------------------------------------------------------------------------

func TestEmitSimpleProducesValidEnvelope(t *testing.T) {
	// Subscribe to the default broker first so EmitSimple's event lands somewhere.
	ch := defaultBroker.subscribe()
	defer defaultBroker.unsubscribe(ch)

	EmitSimple(EvtStageStart, "agent", "starting")

	select {
	case ev := <-ch:
		if ev.Type != EvtStageStart {
			t.Fatalf("type = %q", ev.Type)
		}
		if ev.Payload["detail"] != "starting" {
			t.Fatalf("detail not propagated: %+v", ev.Payload)
		}
	case <-time.After(time.Second):
		t.Fatal("EmitSimple event didn't reach subscriber")
	}
}

func TestEmitSimpleOmitsEmptyDetailFromPayload(t *testing.T) {
	ch := defaultBroker.subscribe()
	defer defaultBroker.unsubscribe(ch)

	EmitSimple(EvtStageStart, "agent", "")

	select {
	case ev := <-ch:
		if _, has := ev.Payload["detail"]; has {
			t.Errorf("empty detail leaked into payload: %+v", ev.Payload)
		}
	case <-time.After(time.Second):
		t.Fatal("event didn't arrive")
	}
}

// ---------------------------------------------------------------------------
// /events handler — integration test
// ---------------------------------------------------------------------------

// This test exists because a unit test on the broker missed a real bug:
// handleEvents called WriteHeader but never flushed, so clients with
// connect-timeout shorter than the 15s heartbeat saw "no response".
// The bug was caught by hand-running curl. This test reproduces that
// scenario in-process.
func TestHandleEventsSendsHeadersImmediately(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(handleEvents))
	defer srv.Close()

	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(srv.URL)
	if err != nil {
		t.Fatalf("GET /events failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
}

func TestHandleEventsEmitsConnectedSentinelWithinOneSecond(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(handleEvents))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()

	// Read the first chunk on a goroutine so we can time-bound it.
	got := make(chan string, 1)
	go func() {
		r := bufio.NewReader(resp.Body)
		line, _ := r.ReadString('\n')
		got <- strings.TrimRight(line, "\r\n")
	}()

	select {
	case line := <-got:
		if !strings.Contains(line, "connected") {
			t.Errorf("first line = %q, want containing 'connected'", line)
		}
	case <-time.After(time.Second):
		t.Fatal("no body byte arrived within 1s — handler is likely buffering its first write")
	}
}

func TestHandleEventsStreamsEmittedEnvelopeToConnectedClient(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(handleEvents))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()

	r := bufio.NewReader(resp.Body)
	// Drain the `: connected` sentinel + its blank line.
	_, _ = r.ReadString('\n')
	_, _ = r.ReadString('\n')

	// Now Emit something. The handler is mid-loop on r.Context, so it
	// should receive on its broker channel and write the data: line.
	go func() {
		// Tiny delay so the handler's select loop is parked.
		time.Sleep(50 * time.Millisecond)
		Emit(NewEnvelope(EvtMetric, "lens", map[string]interface{}{
			"name": "gx_score", "value": 0.83,
		}))
	}()

	got := make(chan string, 1)
	go func() {
		// Read until we see a `data:` line.
		for {
			line, err := r.ReadString('\n')
			if err != nil {
				got <- ""
				return
			}
			if strings.HasPrefix(line, "data:") {
				got <- strings.TrimSpace(strings.TrimPrefix(line, "data:"))
				return
			}
		}
	}()

	select {
	case line := <-got:
		if line == "" {
			t.Fatal("stream closed before data: line arrived")
		}
		var ev Envelope
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("data line not valid envelope JSON: %v\n%s", err, line)
		}
		if ev.Type != EvtMetric || ev.Stage != "lens" {
			t.Errorf("got %+v", ev)
		}
		if ev.Payload["name"] != "gx_score" {
			t.Errorf("payload missing or wrong: %+v", ev.Payload)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("emitted envelope didn't reach connected client within 2s")
	}
}
