// Tests for the typed event broker + envelope shape (PC-061).

package main

import (
	"encoding/json"
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
