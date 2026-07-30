package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
	if len(AllErrorCodes) != 6 {
		t.Fatalf("expected 6 error codes, got %d", len(AllErrorCodes))
	}
}

// Tests for internal service authentication. All tokens are synthetic
// test fixtures.

func withToken(t *testing.T, tok string) func() {
	t.Helper()
	prev := serviceToken
	serviceToken = tok
	return func() { serviceToken = prev }
}

func TestRequireServiceToken(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	t.Run("no token configured passes everything through", func(t *testing.T) {
		defer withToken(t, "")()
		h := requireServiceToken(inner)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/agent", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("open mode rejected: %d", rec.Code)
		}
	})

	t.Run("correct token accepted", func(t *testing.T) {
		defer withToken(t, "atlas-st-testfixture")()
		h := requireServiceToken(inner)
		req := httptest.NewRequest("POST", "/v1/agent", nil)
		req.Header.Set("Authorization", "Bearer atlas-st-testfixture")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("valid token rejected: %d", rec.Code)
		}
	})

	t.Run("missing token rejected", func(t *testing.T) {
		defer withToken(t, "atlas-st-testfixture")()
		h := requireServiceToken(inner)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/agent", nil))
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("missing token got %d, want 401", rec.Code)
		}
		// The response must never echo token material.
		if got := rec.Body.String(); strings.Contains(got, "testfixture") {
			t.Fatalf("401 body leaks token material: %q", got)
		}
	})

	t.Run("wrong token rejected", func(t *testing.T) {
		defer withToken(t, "atlas-st-testfixture")()
		h := requireServiceToken(inner)
		req := httptest.NewRequest("POST", "/cancel", nil)
		req.Header.Set("Authorization", "Bearer atlas-st-WRONG")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("wrong token got %d, want 401", rec.Code)
		}
	})

	t.Run("health and ready stay open", func(t *testing.T) {
		defer withToken(t, "atlas-st-testfixture")()
		h := requireServiceToken(inner)
		for _, p := range []string{"/health", "/ready"} {
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, httptest.NewRequest("GET", p, nil))
			if rec.Code != http.StatusOK {
				t.Fatalf("%s rejected (%d) — compose healthchecks are headerless", p, rec.Code)
			}
		}
	})
}

func TestTokenTransportInjection(t *testing.T) {
	defer withToken(t, "atlas-st-inject")()
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			gotAuth = r.Header.Get("Authorization")
		}))
	defer srv.Close()

	client := &http.Client{Transport: &tokenTransport{}}

	t.Run("injects when absent", func(t *testing.T) {
		req, _ := http.NewRequest("GET", srv.URL, nil)
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if gotAuth != "Bearer atlas-st-inject" {
			t.Fatalf("header not injected: %q", gotAuth)
		}
	})

	t.Run("caller-set header wins (passthrough forwarding)", func(t *testing.T) {
		req, _ := http.NewRequest("GET", srv.URL, nil)
		req.Header.Set("Authorization", "Bearer client-own-key")
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if gotAuth != "Bearer client-own-key" {
			t.Fatalf("caller header overridden: %q", gotAuth)
		}
	})
}

func TestWithRequestIDGenerates(t *testing.T) {
	var seen string
	h := withRequestID(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			seen = requestIDFromContext(r.Context())
		}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/health", nil))
	if seen == "" || !strings.HasPrefix(seen, "req-") {
		t.Fatalf("no generated request id in context: %q", seen)
	}
	if rec.Header().Get(requestIDHeader) != seen {
		t.Fatalf("response header %q != context %q",
			rec.Header().Get(requestIDHeader), seen)
	}
}

func TestWithRequestIDHonorsClientID(t *testing.T) {
	var seen string
	h := withRequestID(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			seen = requestIDFromContext(r.Context())
		}))
	req := httptest.NewRequest("GET", "/health", nil)
	req.Header.Set(requestIDHeader, "req-client-supplied")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if seen != "req-client-supplied" {
		t.Fatalf("client id not honored: %q", seen)
	}
}

func TestTokenTransportForwardsRequestID(t *testing.T) {
	defer withToken(t, "")() // token off — isolate the ID-forwarding path
	var got string
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			got = r.Header.Get(requestIDHeader)
		}))
	defer srv.Close()
	client := &http.Client{Transport: &tokenTransport{}}
	req, _ := http.NewRequest("GET", srv.URL, nil)
	ctx := context.WithValue(req.Context(), requestIDKey, "req-trace-99")
	resp, err := client.Do(req.WithContext(ctx))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if got != "req-trace-99" {
		t.Fatalf("request id not forwarded downstream: %q", got)
	}
}

func TestLogEventJSONMode(t *testing.T) {
	var buf strings.Builder
	oldOut := log.Writer()
	log.SetOutput(&buf)
	defer log.SetOutput(oldOut)
	prev := logJSON
	logJSON = true
	defer func() { logJSON = prev }()
	logEvent("info", "hello", "req-7", map[string]interface{}{"k": "v"})
	out := buf.String()
	idx := strings.Index(out, "{")
	if idx < 0 {
		t.Fatalf("no JSON object in output: %q", out)
	}
	var rec map[string]interface{}
	if err := json.Unmarshal([]byte(out[idx:]), &rec); err != nil {
		t.Fatalf("not JSON: %v (%q)", err, out)
	}
	if rec["level"] != "info" || rec["msg"] != "hello" ||
		rec["request_id"] != "req-7" || rec["k"] != "v" {
		t.Fatalf("bad record: %v", rec)
	}
}

func TestSafeLogFieldEscapesRecordSeparators(t *testing.T) {
	got := safeLogField("first\nforged\r\x00entry", 200)
	if strings.ContainsAny(got, "\r\n\x00") {
		t.Fatalf("safeLogField emitted a raw record separator: %q", got)
	}
	for _, escaped := range []string{`\n`, `\r`, `\x00`} {
		if !strings.Contains(got, escaped) {
			t.Fatalf("safeLogField(%q) missing %q", got, escaped)
		}
	}
}

func TestSafeLogFieldBoundsUntrustedText(t *testing.T) {
	got := safeLogField(strings.Repeat("x", 100), 12)
	if len(got) > 20 {
		t.Fatalf("bounded log field remained unexpectedly large: %q", got)
	}
}

// Corpus-driven tests for private-value filtering. Every fixture value
// is synthetic (see tests/fixtures/private_value_fixtures.json).

type pvCase struct {
	Name           string   `json:"name"`
	Input          string   `json:"input"`
	MustNotContain []string `json:"must_not_contain"`
	MustContain    []string `json:"must_contain"`
	MustEqualInput bool     `json:"must_equal_input"`
}

type pvCorpus struct {
	Placeholder   string   `json:"placeholder"`
	Cases         []pvCase `json:"cases"`
	NegativeCases []pvCase `json:"negative_cases"`
}

func loadCorpus(t *testing.T) pvCorpus {
	t.Helper()
	path := filepath.Join("..", "tests", "fixtures",
		"private_value_fixtures.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("fixture corpus missing: %v", err)
	}
	var c pvCorpus
	if err := json.Unmarshal(data, &c); err != nil {
		t.Fatalf("fixture corpus unparsable: %v", err)
	}
	return c
}

func TestFilterPrivateValuesCorpus(t *testing.T) {
	corpus := loadCorpus(t)
	if corpus.Placeholder != privateValuePlaceholder {
		t.Fatalf("placeholder drift: corpus %q vs code %q",
			corpus.Placeholder, privateValuePlaceholder)
	}
	for _, c := range corpus.Cases {
		got := filterPrivateValues(c.Input)
		for _, bad := range c.MustNotContain {
			if strings.Contains(got, bad) {
				t.Errorf("%s: %q survived filtering: %q", c.Name, bad, got)
			}
		}
		for _, keep := range c.MustContain {
			if !strings.Contains(got, keep) {
				t.Errorf("%s: context %q lost: %q", c.Name, keep, got)
			}
		}
		if len(c.MustNotContain) > 0 &&
			!strings.Contains(got, privateValuePlaceholder) {
			t.Errorf("%s: no placeholder in output: %q", c.Name, got)
		}
	}
	for _, c := range corpus.NegativeCases {
		if got := filterPrivateValues(c.Input); got != c.Input {
			t.Errorf("%s: benign input modified: %q -> %q",
				c.Name, c.Input, got)
		}
	}
}

func TestFilteringWriterOnLogger(t *testing.T) {
	var buf bytes.Buffer
	lg := log.New(filteringWriter{w: &buf}, "", 0)
	lg.Printf("turn failed: EXAMPLE_API_TOKEN=not-a-real-token status=500")
	out := buf.String()
	if strings.Contains(out, "not-a-real-token") {
		t.Fatalf("fixture value reached the log sink: %q", out)
	}
	if !strings.Contains(out, "status=500") {
		t.Fatalf("benign context lost: %q", out)
	}
	if !strings.Contains(out, privateValuePlaceholder) {
		t.Fatalf("placeholder missing: %q", out)
	}
}
