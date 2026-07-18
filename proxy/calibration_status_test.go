package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func lensHealthServer(t *testing.T, lens map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":     "healthy",
			"subsystems": map[string]any{"lens": lens},
		})
	}))
}

func compatibleLensHealth() map[string]any {
	return map[string]any{
		"enabled":           true,
		"cost_field_loaded": true,
		"cost_field_dim":    3840,
		"embed_dim":         3840,
		"gx_loaded":         true,
		"cx_calibrated":     true,
		"gx_calibrated":     true,
		"self_test_pass":    true,
	}
}

func TestProbeLensStatusRequiresSelectedModelCalibration(t *testing.T) {
	health := compatibleLensHealth()
	health["cx_calibrated"] = false
	srv := lensHealthServer(t, health)
	defer srv.Close()

	got := probeLensStatus(context.Background(), srv.URL)
	if got.Verdict != "uncalibrated" {
		t.Fatalf("verdict = %q, want uncalibrated", got.Verdict)
	}
}

func TestProbeLensStatusSurfacesArtifactIdentityMismatch(t *testing.T) {
	health := compatibleLensHealth()
	health["cost_field_loaded"] = false
	health["self_test_error"] = "artifacts are for model-a, selected model is model-b"
	srv := lensHealthServer(t, health)
	defer srv.Close()

	got := probeLensStatus(context.Background(), srv.URL)
	if got.Hint != health["self_test_error"] {
		t.Fatalf("hint = %q, want identity mismatch", got.Hint)
	}
}

func TestProbeLensStatusRequiresCompleteArtifacts(t *testing.T) {
	health := compatibleLensHealth()
	health["gx_loaded"] = false
	srv := lensHealthServer(t, health)
	defer srv.Close()

	got := probeLensStatus(context.Background(), srv.URL)
	if got.Verdict != "incomplete-artifacts" {
		t.Fatalf("verdict = %q, want incomplete-artifacts", got.Verdict)
	}
}

func TestProbeLensStatusSupportsCalibratedMatchingArtifacts(t *testing.T) {
	srv := lensHealthServer(t, compatibleLensHealth())
	defer srv.Close()

	got := probeLensStatus(context.Background(), srv.URL)
	if got.Verdict != "supported" {
		t.Fatalf("verdict = %q, want supported (%s)", got.Verdict, got.Hint)
	}
}

func TestProbeASAStatusRequiresMatchingModelMarker(t *testing.T) {
	dir := t.TempDir()
	vector := filepath.Join(dir, "ast_edit_steering.gguf")
	if err := os.WriteFile(vector, []byte("vector"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ATLAS_CONTROL_VECTOR", vector)
	t.Setenv("ATLAS_MODEL_NAME", "selected-model")

	if got := probeASAStatus(); got.Verdict != "unverified" {
		t.Fatalf("without marker verdict = %q, want unverified", got.Verdict)
	}
	if err := os.WriteFile(vector+".model", []byte("other-model\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := probeASAStatus(); got.Verdict != "incompatible" {
		t.Fatalf("wrong marker verdict = %q, want incompatible", got.Verdict)
	}
	if err := os.WriteFile(vector+".model", []byte("selected-model\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := probeASAStatus(); got.Verdict != "supported" {
		t.Fatalf("matching marker verdict = %q, want supported", got.Verdict)
	}
}
