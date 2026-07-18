package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModelsFallsBackToConfiguredModel(t *testing.T) {
	previous := modelName
	modelName = "configured-model"
	t.Cleanup(func() { modelName = previous })
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
	}))
	t.Cleanup(upstream.Close)
	withInferenceURL(t, upstream.URL)

	for _, path := range []string{"/v1/models", "/models"} {
		t.Run(path, func(t *testing.T) {
			rec := httptest.NewRecorder()
			newProxyMux().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
			}
			var payload struct {
				Data []struct {
					ID string `json:"id"`
				} `json:"data"`
			}
			if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if len(payload.Data) != 1 || payload.Data[0].ID != modelName {
				t.Fatalf("model data = %#v, want id %q", payload.Data, modelName)
			}
		})
	}
}

func TestModelsPrefersLoadedModelOverStaleConfig(t *testing.T) {
	previous := modelName
	modelName = "stale-config-model"
	t.Cleanup(func() { modelName = previous })
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("upstream path = %q, want /v1/models", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"loaded-runtime-model"}]}`))
	}))
	t.Cleanup(upstream.Close)
	withInferenceURL(t, upstream.URL)

	rec := httptest.NewRecorder()
	newProxyMux().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(payload.Data) != 1 || payload.Data[0].ID != "loaded-runtime-model" {
		t.Fatalf("model data = %#v, want loaded runtime model", payload.Data)
	}
}

func TestHealthAdvertisesRawDemoCapability(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(upstream.Close)

	previousInference, previousLens, previousSandbox := inferenceURL, lensURL, sandboxURL
	inferenceURL, lensURL, sandboxURL = upstream.URL, upstream.URL, upstream.URL
	t.Cleanup(func() {
		inferenceURL, lensURL, sandboxURL = previousInference, previousLens, previousSandbox
	})

	rec := httptest.NewRecorder()
	newProxyMux().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	var payload struct {
		Capabilities []string `json:"capabilities"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	found := false
	for _, capability := range payload.Capabilities {
		if capability == demoRawCapability {
			found = true
		}
	}
	if !found {
		t.Fatalf("health capabilities = %v, want %q", payload.Capabilities, demoRawCapability)
	}
}

func withInferenceURL(t *testing.T, url string) {
	t.Helper()
	previous := inferenceURL
	inferenceURL = url
	t.Cleanup(func() {
		inferenceURL = previous
	})
}

func TestPassthroughPreservesRequestURI(t *testing.T) {
	var gotMethod string
	var gotRequestURI string
	var gotBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotRequestURI = r.URL.RequestURI()
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read upstream body: %v", err)
		}
		gotBody = string(body)
		w.Header().Set("X-Upstream", "seen")
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte("ok"))
	}))
	t.Cleanup(upstream.Close)
	withInferenceURL(t, upstream.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions?stream=true&n=2", strings.NewReader("payload"))
	req.Header.Set("X-Client", "atlas-test")
	rec := httptest.NewRecorder()

	newProxyMux().ServeHTTP(rec, req)

	if rec.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want %d; body=%q", rec.Code, http.StatusAccepted, rec.Body.String())
	}
	if gotMethod != http.MethodPost {
		t.Fatalf("method = %q, want %q", gotMethod, http.MethodPost)
	}
	if gotRequestURI != "/v1/chat/completions?stream=true&n=2" {
		t.Fatalf("request URI = %q", gotRequestURI)
	}
	if gotBody != "payload" {
		t.Fatalf("body = %q", gotBody)
	}
	if rec.Header().Get("X-Upstream") != "seen" {
		t.Fatalf("upstream header was not copied")
	}
}

func TestPassthroughRejectsOversizedBody(t *testing.T) {
	upstreamCalled := false
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalled = true
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(upstream.Close)
	withInferenceURL(t, upstream.URL)

	handler := http.MaxBytesHandler(newProxyMux(), 1)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("too large"))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want %d; body=%q", rec.Code, http.StatusRequestEntityTooLarge, rec.Body.String())
	}
	if upstreamCalled {
		t.Fatalf("oversized request reached upstream")
	}
}
