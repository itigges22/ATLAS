package main

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
)

func TestWriteErrorEnvelope(t *testing.T) {
	rec := httptest.NewRecorder()
	writeError(rec, 400, ErrInvalidInput, "bad thing")
	if rec.Code != 400 {
		t.Fatalf("status = %d", rec.Code)
	}
	var env ErrorEnvelope
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	if env.Error != "invalid_input" {
		t.Errorf("code = %q", env.Error)
	}
	if env.APIVersion != APIVersion {
		t.Errorf("api_version = %q", env.APIVersion)
	}
}

func TestAllErrorCodesUnique(t *testing.T) {
	seen := map[ErrorCode]bool{}
	for _, c := range AllErrorCodes {
		if seen[c] {
			t.Fatalf("duplicate error code %q", c)
		}
		seen[c] = true
	}
	if len(AllErrorCodes) != 12 {
		t.Fatalf("expected 12 error codes, got %d", len(AllErrorCodes))
	}
}
