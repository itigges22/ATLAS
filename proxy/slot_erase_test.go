package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// llama-server handles an erase on its main loop, so a slot that is
// mid-decode answers only once it frees up. Measured 2026-08-03: a single
// attempt lost one slot in half the sessions of a 26-session run, and the
// slot it lost is the one still holding the previous session's KV — the
// cross-session bleed the erase exists to prevent.
func TestEraseRetriesABusySlot(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			http.Error(w, "slot is processing", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client := &http.Client{Timeout: 2 * time.Second}
	if !eraseOneSlot(context.Background(), client, srv.URL, 1) {
		t.Error("a slot that frees up on the second attempt was reported stale")
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("expected a retry, got %d call(s)", got)
	}
}

func TestEraseReportsASlotItCouldNotClear(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "still busy", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	client := &http.Client{Timeout: 2 * time.Second}
	if eraseOneSlot(context.Background(), client, srv.URL, 2) {
		t.Error("a slot that never cleared was reported clear")
	}
}

func TestEraseStopsWhenTheRequestIsCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "busy", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	client := &http.Client{Timeout: 2 * time.Second}

	start := time.Now()
	if eraseOneSlot(ctx, client, srv.URL, 0) {
		t.Error("a cancelled erase was reported clear")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("cancellation should stop the retries promptly, took %v", elapsed)
	}
}
