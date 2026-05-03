// Tests for the PC-062 /cancel endpoint.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCancelEndpointAbortsSession(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	activeSessions.Store("sess-abc", context.CancelFunc(cancel))

	body, _ := json.Marshal(map[string]string{"session_id": "sess-abc"})
	req := httptest.NewRequest("POST", "/cancel", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	handleCancel(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200; body = %s", rec.Code, rec.Body.String())
	}
	var resp map[string]bool
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !resp["cancelled"] {
		t.Errorf("cancelled = false, want true")
	}
	// Session should be removed (idempotent — second cancel returns 404).
	if _, ok := activeSessions.Load("sess-abc"); ok {
		t.Errorf("session not removed from map after cancel")
	}
	// And the cancel func must have actually fired.
	select {
	case <-ctx.Done():
		// good
	default:
		t.Errorf("context not cancelled")
	}
}

func TestCancelEndpointUnknownSessionReturns404(t *testing.T) {
	body, _ := json.Marshal(map[string]string{"session_id": "does-not-exist"})
	req := httptest.NewRequest("POST", "/cancel", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	handleCancel(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", rec.Code)
	}
}

func TestCancelEndpointRejectsGet(t *testing.T) {
	req := httptest.NewRequest("GET", "/cancel", nil)
	rec := httptest.NewRecorder()
	handleCancel(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

func TestCancelEndpointRequiresSessionID(t *testing.T) {
	body, _ := json.Marshal(map[string]string{})
	req := httptest.NewRequest("POST", "/cancel", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	handleCancel(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}
