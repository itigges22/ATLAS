package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
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

// Every generation request leaving the proxy carries an explicit
// completion bound, so a client disconnect can't leave an unbounded zombie
// generation holding a llama slot.
func TestClampGenerationBody(t *testing.T) {
	get := func(body []byte, key string) (float64, bool) {
		var m map[string]interface{}
		if err := json.Unmarshal(body, &m); err != nil {
			t.Fatalf("clamped body is not JSON: %v", err)
		}
		v, ok := m[key].(float64)
		return v, ok
	}

	t.Run("missing max_tokens gets the default ceiling", func(t *testing.T) {
		out := clampGenerationBody("/v1/chat/completions",
			[]byte(`{"messages":[{"role":"user","content":"hi"}]}`))
		if v, ok := get(out, "max_tokens"); !ok || v != 8192 {
			t.Fatalf("want max_tokens=8192, got %v (present=%v)", v, ok)
		}
	})

	t.Run("in-range value passes through untouched", func(t *testing.T) {
		in := []byte(`{"max_tokens":512}`)
		out := clampGenerationBody("/v1/chat/completions", in)
		if string(out) != string(in) {
			t.Fatalf("in-range body modified: %s", out)
		}
	})

	t.Run("unlimited sentinel -1 is clamped", func(t *testing.T) {
		out := clampGenerationBody("/v1/completions", []byte(`{"max_tokens":-1}`))
		if v, _ := get(out, "max_tokens"); v != 8192 {
			t.Fatalf("want clamp to 8192, got %v", v)
		}
	})

	t.Run("above-ceiling value is clamped", func(t *testing.T) {
		out := clampGenerationBody("/v1/chat/completions", []byte(`{"max_tokens":900000}`))
		if v, _ := get(out, "max_tokens"); v != 8192 {
			t.Fatalf("want clamp to 8192, got %v", v)
		}
	})

	t.Run("llama-native endpoints use n_predict", func(t *testing.T) {
		out := clampGenerationBody("/completion", []byte(`{"prompt":"x"}`))
		if v, ok := get(out, "n_predict"); !ok || v != 8192 {
			t.Fatalf("want n_predict=8192, got %v (present=%v)", v, ok)
		}
	})

	t.Run("env override sets the ceiling", func(t *testing.T) {
		t.Setenv("ATLAS_MAX_COMPLETION_TOKENS", "1024")
		out := clampGenerationBody("/v1/chat/completions", []byte(`{}`))
		if v, _ := get(out, "max_tokens"); v != 1024 {
			t.Fatalf("want max_tokens=1024, got %v", v)
		}
		out = clampGenerationBody("/v1/chat/completions", []byte(`{"max_tokens":2048}`))
		if v, _ := get(out, "max_tokens"); v != 1024 {
			t.Fatalf("want clamp to 1024, got %v", v)
		}
	})

	t.Run("non-generation paths and bad JSON pass through", func(t *testing.T) {
		for _, c := range []struct{ path, body string }{
			{"/health", `{}`},
			{"/slots", `{"max_tokens":-1}`},
			{"/v1/chat/completions", `not json`},
		} {
			out := clampGenerationBody(c.path, []byte(c.body))
			if string(out) != c.body {
				t.Fatalf("%s: body modified: %s", c.path, out)
			}
		}
	})
}

// --- The structured task contract --------------------------------------------
//
// ATLAS infers obligations from English today: which verbs the user used
// decides whether work was demanded, which filenames near a write verb become
// deliverables, which phrases demand verification. This is the first piece of
// the replacement -- the client saying, in typed fields, what it already knows.
//
// It decides nothing yet. This commit adds the shape, validates it, and stores
// it; a guard proves no production decision reads it.

func TestTaskContractWireShape(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "app.py"), []byte("A = 1\n"), 0o644)

	for _, c := range []struct {
		name    string
		body    string
		present bool
		valid   bool
		mode    TaskMode
		outputs []string
		verify  []string
	}{
		{"absent contract", `{}`, false, false, "", nil, nil},
		{"work with outputs and verification",
			`{"task_contract":{"task_mode":"work","expected_outputs":["app.py"],"verification":["go test ./..."]}}`,
			true, true, TaskModeWork, []string{"app.py"}, []string{"go test ./..."}},
		{"question", `{"task_contract":{"task_mode":"question"}}`,
			true, true, TaskModeQuestion, nil, nil},
		{"unknown mode", `{"task_contract":{"task_mode":"explore"}}`, true, false, "", nil, nil},
		{"empty mode", `{"task_contract":{"task_mode":""}}`, true, false, "", nil, nil},
		{"one bad path rejects the whole contract",
			`{"task_contract":{"task_mode":"work","expected_outputs":["app.py","../../etc/passwd"]}}`,
			true, false, "", nil, nil},
		{"workspace escape", `{"task_contract":{"task_mode":"work","expected_outputs":["../x.py"]}}`,
			true, false, "", nil, nil},
		{"empty output entry", `{"task_contract":{"task_mode":"work","expected_outputs":[""]}}`,
			true, false, "", nil, nil},
		{"empty verification entry",
			`{"task_contract":{"task_mode":"work","verification":["  "]}}`, true, false, "", nil, nil},
		{"aliases deduplicate canonically",
			`{"task_contract":{"task_mode":"work","expected_outputs":["app.py","./app.py"]}}`,
			true, true, TaskModeWork, []string{"app.py"}, nil},
		{"duplicate verification deduplicates",
			`{"task_contract":{"task_mode":"work","verification":["go test ./...","go test ./..."]}}`,
			true, true, TaskModeWork, nil, []string{"go test ./..."}},
	} {
		t.Run(c.name, func(t *testing.T) {
			var wire struct {
				TaskContract *TaskContract `json:"task_contract,omitempty"`
			}
			if err := json.Unmarshal([]byte(c.body), &wire); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if (wire.TaskContract != nil) != c.present {
				t.Fatalf("present=%v want %v", wire.TaskContract != nil, c.present)
			}
			if !c.present {
				return
			}
			got, err := validateTaskContract(wire.TaskContract, dir)
			if (err == nil) != c.valid {
				t.Fatalf("valid=%v (err=%v) want %v", err == nil, err, c.valid)
			}
			if !c.valid {
				if got != nil {
					t.Error("an invalid contract produced a stored value; it must be all or nothing")
				}
				return
			}
			if got.TaskMode != c.mode {
				t.Errorf("mode=%q want %q", got.TaskMode, c.mode)
			}
			if strings.Join(got.ExpectedOutputs, "|") != strings.Join(c.outputs, "|") {
				t.Errorf("outputs=%v want %v", got.ExpectedOutputs, c.outputs)
			}
			if strings.Join(got.Verification, "|") != strings.Join(c.verify, "|") {
				t.Errorf("verification=%v want %v", got.Verification, c.verify)
			}
		})
	}
}

// Bounds use the ceiling the rest of the session state already uses.
func TestTaskContractBounds(t *testing.T) {
	dir := t.TempDir()
	mk := func(n int) *TaskContract {
		c := &TaskContract{TaskMode: TaskModeWork}
		for i := 0; i < n; i++ {
			c.ExpectedOutputs = append(c.ExpectedOutputs, fmt.Sprintf("f%d.py", i))
		}
		return c
	}
	if _, err := validateTaskContract(mk(maxTaskContractEntries), dir); err != nil {
		t.Errorf("at the bound: %v", err)
	}
	got, err := validateTaskContract(mk(maxTaskContractEntries+1), dir)
	if err == nil {
		t.Error("over the bound was accepted")
	}
	if got != nil {
		t.Error("overflow produced a partial contract")
	}
	v := &TaskContract{TaskMode: TaskModeWork}
	for i := 0; i <= maxTaskContractEntries; i++ {
		v.Verification = append(v.Verification, fmt.Sprintf("cmd %d", i))
	}
	if _, err := validateTaskContract(v, dir); err == nil {
		t.Error("verification overflow was accepted")
	}
}

// Round-trip: a valid contract serialises back to the same fields.
func TestTaskContractRoundTrip(t *testing.T) {
	dir := t.TempDir()
	in := &TaskContract{
		TaskMode:        TaskModeWork,
		ExpectedOutputs: []string{"b.py", "a.py"},
		Verification:    []string{"pytest", "go vet ./..."},
	}
	got, err := validateTaskContract(in, dir)
	if err != nil {
		t.Fatal(err)
	}
	// Stable ordering, so two equivalent requests never disagree.
	if strings.Join(got.ExpectedOutputs, "|") != "a.py|b.py" {
		t.Errorf("outputs not stably ordered: %v", got.ExpectedOutputs)
	}
	if strings.Join(got.Verification, "|") != "go vet ./...|pytest" {
		t.Errorf("verification not stably ordered: %v", got.Verification)
	}
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var back TaskContract
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back.TaskMode != got.TaskMode ||
		strings.Join(back.ExpectedOutputs, "|") != strings.Join(got.ExpectedOutputs, "|") ||
		strings.Join(back.Verification, "|") != strings.Join(got.Verification, "|") {
		t.Errorf("round trip lost fields: %+v vs %+v", back, got)
	}
}

// The contract decides nothing. One definition, one validator, no consumer.
func TestTaskContractHasNoDecisionConsumer(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var defs, validators, readers []string
	for _, e := range entries {
		n := e.Name()
		if e.IsDir() || !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, n, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", n, err)
		}
		ast.Inspect(f, func(node ast.Node) bool {
			switch v := node.(type) {
			case *ast.TypeSpec:
				if v.Name != nil && v.Name.Name == "TaskContract" {
					defs = append(defs, fmt.Sprintf("%s:%d", n, fset.Position(v.Pos()).Line))
				}
			case *ast.FuncDecl:
				if v.Name != nil && v.Name.Name == "validateTaskContract" {
					validators = append(validators, fmt.Sprintf("%s:%d", n, fset.Position(v.Pos()).Line))
				}
			case *ast.SelectorExpr:
				if v.Sel != nil && v.Sel.Name == "TaskContract" {
					readers = append(readers, fmt.Sprintf("%s:%d", n, fset.Position(v.Pos()).Line))
				}
			}
			return true
		})
	}
	t.Logf("definitions=%v validators=%v reads=%v", defs, validators, readers)
	if len(defs) != 1 {
		t.Errorf("%d TaskContract definitions, want exactly one: %v", len(defs), defs)
	}
	if len(validators) != 1 {
		t.Errorf("%d validators, want exactly one: %v", len(validators), validators)
	}
	for _, r := range readers {
		if !strings.HasPrefix(r, "agent.go:") {
			t.Errorf("%s reads the task contract outside the request boundary", r)
		}
	}
	body, _ := os.ReadFile("agent.go")
	for _, fn := range []string{"wantsStateChange", "classifyAgentTier", "terminalCompletionAllowed",
		"finalizeCompletion", "blockingTombstone", "needsPermission", "honestTerminalSummary"} {
		i := strings.Index(string(body), "func "+fn)
		if i < 0 {
			continue
		}
		end := strings.Index(string(body)[i+1:], "\nfunc ")
		if end < 0 {
			end = len(body) - i - 1
		}
		if strings.Contains(string(body)[i:i+1+end], "TaskContract") {
			t.Errorf("%s consults the task contract; this commit adds no decision", fn)
		}
	}
	guard, _ := os.ReadFile("guardrails.go")
	if strings.Contains(string(guard), "TaskContract") {
		t.Error("guardrails.go consults the task contract")
	}
}

// The contract is inert: a run with one and a run without produce the same
// events, the same terminal, and the same disk.
func TestTaskContractIsInert(t *testing.T) {
	run := func(t *testing.T, contract *TaskContract, prompt string,
		plan func(i int) map[string]interface{}) (string, string, []string) {
		t.Helper()
		dir := t.TempDir()
		turns := 0
		var mu sync.Mutex
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch {
			case strings.HasPrefix(r.URL.Path, "/v3/"), strings.HasPrefix(r.URL.Path, "/internal/"):
				http.Error(w, "unavailable", http.StatusServiceUnavailable)
				return
			case strings.HasSuffix(r.URL.Path, "/syntax-check"):
				json.NewEncoder(w).Encode(map[string]interface{}{"valid": true})
				return
			case strings.HasSuffix(r.URL.Path, "/execute"):
				var in struct{ Code string }
				json.NewDecoder(r.Body).Decode(&in)
				if strings.Contains(in.Code, ".atlas-mount-probe") {
					b, _ := os.ReadFile(filepath.Join(dir, ".atlas-mount-probe"))
					json.NewEncoder(w).Encode(map[string]interface{}{
						"success": true, "stdout": string(b), "exit_code": 0})
					return
				}
				json.NewEncoder(w).Encode(map[string]interface{}{
					"success": true, "stdout": "", "exit_code": 0})
				return
			case !strings.HasSuffix(r.URL.Path, "/v1/chat/completions"):
				http.NotFound(w, r)
				return
			}
			io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "text/event-stream")
			mu.Lock()
			i := turns
			turns++
			mu.Unlock()
			call, _ := json.Marshal(plan(i))
			d, _ := json.Marshal(map[string]interface{}{
				"choices": []map[string]interface{}{
					{"delta": map[string]string{"content": string(call)}}}})
			fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", d)
		}))
		defer srv.Close()

		ctx := NewAgentContext(dir, Tier2Medium)
		ctx.InferenceURL, ctx.SandboxURL, ctx.V3URL = srv.URL, srv.URL, srv.URL
		ctx.PermissionMode = PermissionYolo
		ctx.TrustMode = trustFullyTrusted
		ctx.VerifyOnHost = true
		ctx.MaxTurns = 0
		ctx.TaskContract = contract // the only difference between the two runs
		var events []string
		terminal := map[string]string{}
		ctx.StreamFn = func(et string, data interface{}) {
			b, _ := json.Marshal(data)
			mu.Lock()
			defer mu.Unlock()
			// Timings differ run to run; everything else is compared.
			line := et + "|" + string(b)
			var m map[string]interface{}
			if json.Unmarshal(b, &m) == nil {
				for _, k := range []string{"elapsed", "prompt_ms", "elapsed_ms",
					"duration_ms", "wall_s", "ms"} {
					delete(m, k)
				}
				c, _ := json.Marshal(m)
				line = et + "|" + string(c)
			}
			events = append(events, line)
			if et == "done" {
				var m map[string]string
				json.Unmarshal(b, &m)
				for k, v := range m {
					terminal[k] = v
				}
			}
		}
		runAgentLoop(ctx, prompt)
		var disk []string
		filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() {
				return nil
			}
			rel, _ := filepath.Rel(dir, p)
			if !strings.HasPrefix(rel, ".") {
				b, _ := os.ReadFile(p)
				disk = append(disk, rel+"="+hashBytes(b)[:12])
			}
			return nil
		})
		sort.Strings(disk)
		return strings.Join(events, "\n"),
			terminal["status"] + "/" + terminal["reason"], disk
	}

	contract := &TaskContract{
		TaskMode:        TaskModeWork,
		ExpectedOutputs: []string{"never_written.py"},
		Verification:    []string{"go test ./..."},
	}
	for _, c := range []struct {
		name, prompt string
		plan         func(i int) map[string]interface{}
	}{
		{"work request", "Create app.py.", func(i int) map[string]interface{} {
			if i == 0 {
				return map[string]interface{}{"type": "tool_call", "name": "write_file",
					"args": map[string]string{"path": "app.py", "content": "A = 1\n"}}
			}
			return map[string]interface{}{"type": "done", "summary": "wrote app.py"}
		}},
		{"question", "What does this repository do?", func(i int) map[string]interface{} {
			return map[string]interface{}{"type": "done", "summary": "it is empty"}
		}},
		{"prose only", "Create app.py.", func(i int) map[string]interface{} {
			return map[string]interface{}{"type": "text", "content": "here is what I would do"}
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			evA, termA, diskA := run(t, nil, c.prompt, c.plan)
			evB, termB, diskB := run(t, contract, c.prompt, c.plan)
			t.Logf("%s: terminal without=%q with=%q", c.name, termA, termB)
			if termA != termB {
				t.Errorf("terminal differs: %q vs %q", termA, termB)
			}
			if strings.Join(diskA, ",") != strings.Join(diskB, ",") {
				t.Errorf("disk differs:\n  %v\n  %v", diskA, diskB)
			}
			if evA != evB {
				a, b := strings.Split(evA, "\n"), strings.Split(evB, "\n")
				for i := 0; i < len(a) || i < len(b); i++ {
					var x, y string
					if i < len(a) {
						x = a[i]
					}
					if i < len(b) {
						y = b[i]
					}
					if x != y {
						t.Errorf("first difference at event %d:\n  without: %.200s\n  with:    %.200s", i, x, y)
						break
					}
				}
			}
		})
	}
}

// A caller that declares nothing keeps behaving exactly as before. This is the
// external/legacy path, and it must stay open: the contract is additive, not a
// new requirement.
func TestLegacyRequestWithoutContractIsAccepted(t *testing.T) {
	var wire struct {
		Message      string        `json:"message"`
		TaskContract *TaskContract `json:"task_contract,omitempty"`
	}
	const legacy = `{"message":"Create app.py.","working_dir":"/w","mode":"default"}`
	if err := json.Unmarshal([]byte(legacy), &wire); err != nil {
		t.Fatalf("a legacy body was rejected: %v", err)
	}
	if wire.TaskContract != nil {
		t.Error("a body with no task_contract decoded one anyway")
	}
	got, err := validateTaskContract(wire.TaskContract, t.TempDir())
	if err != nil || got != nil {
		t.Errorf("absent contract validated to %v (err=%v); absent must stay absent", got, err)
	}
	// And a present-but-invalid one is still a hard refusal, never a silent
	// omission that would look like a client which declared nothing.
	bad := &TaskContract{TaskMode: "explore"}
	if _, err := validateTaskContract(bad, t.TempDir()); err == nil {
		t.Error("an unsupported mode was accepted")
	}
}

// --- Bounded graceful shutdown -----------------------------------------------
//
// main() blocked in ListenAndServe and died on log.Fatalf, so SIGTERM killed
// the process where it stood: an active agent request was cut mid-turn, the
// TUI's permanent /events subscriber had no coordinated close, and no ordered
// cleanup had anywhere to run. A diagnostic sink that must drain and write a
// footer needs that landing site to exist.

func TestShutdownBudgetArithmetic(t *testing.T) {
	for _, c := range []struct {
		name       string
		sessionSec string
		graceSec   string
		wantErr    bool
	}{
		{"defaults", "", "", false}, // 600 + 10 = 610 < 650
		{"explicit shipped values", "600", "650", false},
		{"session raised without grace", "3600", "650", true},
		{"session raised with grace", "3600", "3700", false},
		{"grace exactly equal to need", "600", "610", true}, // strict <
		{"grace one second above need", "600", "611", false},
		{"malformed grace", "", "abc", true},
		{"zero grace", "", "0", true},
		{"negative grace", "", "-5", true},
	} {
		t.Run(c.name, func(t *testing.T) {
			if c.sessionSec != "" {
				t.Setenv("ATLAS_AGENT_SESSION_TIMEOUT_SEC", c.sessionSec)
			}
			if c.graceSec != "" {
				t.Setenv("ATLAS_SHUTDOWN_GRACE_SEC", c.graceSec)
			}
			budget, err := shutdownBudget()
			if (err != nil) != c.wantErr {
				t.Fatalf("err=%v, wantErr=%v", err, c.wantErr)
			}
			if err != nil {
				// The refusal must show the whole sum, not just complain.
				for _, want := range []string{"ATLAS_SHUTDOWN_GRACE_SEC"} {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error does not name %s: %v", want, err)
					}
				}
				return
			}
			total, _ := sessionBudget()
			if budget.drain != total {
				t.Errorf("drain=%v, want the session total %v", budget.drain, total)
			}
			if budget.hookMargin != shutdownHookMargin {
				t.Errorf("hookMargin=%v", budget.hookMargin)
			}
		})
	}
}

// The shipped compose grace must agree with the shipped default.
func TestComposeGraceAgreesWithTheDefault(t *testing.T) {
	b, err := os.ReadFile("../docker-compose.yml")
	if err != nil {
		t.Skipf("compose unavailable: %v", err)
	}
	if !strings.Contains(string(b), "stop_grace_period: 650s") {
		t.Error("docker-compose.yml does not pin stop_grace_period: 650s to match " +
			"the shipped ATLAS_SHUTDOWN_GRACE_SEC default")
	}
}

// The production path: a real listener, a real in-flight request, a real
// /events subscriber, and a signal.
func TestRunServerDrainsAndRunsHooks(t *testing.T) {
	prev := defaultBroker
	defaultBroker = &broker{subscribers: map[chan Envelope]struct{}{}}
	defer func() { defaultBroker = prev }()

	released := make(chan struct{})
	started := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/events", handleEvents)
	mux.HandleFunc("/slow", func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-released // still running when the signal arrives
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "finished")
	})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: mux}
	base := "http://" + ln.Addr().String()

	var hookRuns int32
	hooks := []closeHook{{
		name: "test-hook",
		fn: func(ctx context.Context) error {
			atomic.AddInt32(&hookRuns, 1)
			return nil
		},
	}}
	signals := make(chan os.Signal, 2)
	result := make(chan shutdownResult, 1)
	go func() {
		r, _ := runServer(srv, ln, signals, hooks,
			shutdownBudgetValues{drain: 10 * time.Second, hookMargin: 2 * time.Second})
		result <- r
	}()

	// A permanent /events subscriber, exactly as the TUI keeps open.
	evResp, err := http.Get(base + "/events")
	if err != nil {
		t.Fatal(err)
	}
	defer evResp.Body.Close()
	evBuf := make([]byte, 32)
	evResp.Body.Read(evBuf)

	// An in-flight request.
	slowDone := make(chan int, 1)
	go func() {
		resp, err := http.Get(base + "/slow")
		if err != nil {
			slowDone <- 0
			return
		}
		defer resp.Body.Close()
		io.ReadAll(resp.Body)
		slowDone <- resp.StatusCode
	}()
	<-started

	signals <- syscall.SIGTERM
	signals <- syscall.SIGTERM // repeated signals must not double-run hooks

	// The subscriber unblocks without consuming the drain budget.
	evClosed := make(chan struct{})
	go func() { io.ReadAll(evResp.Body); close(evClosed) }()
	select {
	case <-evClosed:
	case <-time.After(5 * time.Second):
		t.Fatal("/events held the shutdown open")
	}

	// The in-flight request still finishes on its own terms.
	close(released)
	if code := <-slowDone; code != http.StatusOK {
		t.Errorf("the in-flight request was cut off: status %d", code)
	}

	select {
	case r := <-result:
		if r.forced {
			t.Errorf("a clean drain was reported as forced: %+v", r)
		}
		if !r.hooksRan {
			t.Error("close hooks did not run")
		}
	case <-time.After(15 * time.Second):
		t.Fatal("runServer never returned")
	}
	if n := atomic.LoadInt32(&hookRuns); n != 1 {
		t.Errorf("hooks ran %d times, want exactly one", n)
	}
	// New work is refused once draining began.
	if _, err := http.Get(base + "/slow"); err == nil {
		t.Error("the server accepted a new request after draining")
	}
}

// A request that outlives the drain budget takes the forced-close path, and
// that is reported as its own outcome -- not as a listener failure.
func TestRunServerForcedCloseIsClassified(t *testing.T) {
	prev := defaultBroker
	defaultBroker = &broker{subscribers: map[chan Envelope]struct{}{}}
	defer func() { defaultBroker = prev }()

	block := make(chan struct{})
	defer close(block)
	mux := http.NewServeMux()
	mux.HandleFunc("/stuck", func(w http.ResponseWriter, r *http.Request) { <-block })
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	srv := &http.Server{Handler: mux}
	signals := make(chan os.Signal, 1)
	result := make(chan shutdownResult, 1)
	go func() {
		r, _ := runServer(srv, ln, signals, nil,
			shutdownBudgetValues{drain: 300 * time.Millisecond, hookMargin: 200 * time.Millisecond})
		result <- r
	}()
	go http.Get("http://" + ln.Addr().String() + "/stuck")
	time.Sleep(100 * time.Millisecond)
	signals <- syscall.SIGTERM

	select {
	case r := <-result:
		if !r.forced {
			t.Error("a request outliving the drain budget was not classified as forced")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("runServer never returned")
	}
}

// A listener that cannot start returns immediately rather than waiting for a
// signal, and ErrServerClosed is never an error.
func TestRunServerStartupFailureReturns(t *testing.T) {
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	ln.Close() // already closed: Serve fails at once
	srv := &http.Server{Handler: http.NewServeMux()}
	done := make(chan error, 1)
	go func() {
		_, err := runServer(srv, ln, make(chan os.Signal, 1), nil,
			shutdownBudgetValues{drain: time.Second, hookMargin: time.Second})
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Error("a listener failure was reported as success")
		}
		if errors.Is(err, http.ErrServerClosed) {
			t.Error("ErrServerClosed leaked out as a failure")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("runServer waited for a signal after a startup failure")
	}
}

// A hook that hangs is cut off by its own margin, and a failing hook is
// reported without hanging the process.
func TestCloseHooksAreBounded(t *testing.T) {
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	srv := &http.Server{Handler: http.NewServeMux()}
	signals := make(chan os.Signal, 1)
	hooks := []closeHook{
		{name: "hangs", fn: func(ctx context.Context) error { <-ctx.Done(); return ctx.Err() }},
		{name: "fails", fn: func(ctx context.Context) error { return errors.New("boom") }},
	}
	result := make(chan shutdownResult, 1)
	go func() {
		r, _ := runServer(srv, ln, signals, hooks,
			shutdownBudgetValues{drain: time.Second, hookMargin: 200 * time.Millisecond})
		result <- r
	}()
	time.Sleep(50 * time.Millisecond)
	signals <- syscall.SIGTERM
	select {
	case r := <-result:
		if len(r.hookErrors) != 2 {
			t.Errorf("hook errors=%v, want both reported", r.hookErrors)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("a blocked hook hung the shutdown")
	}
}

// --- The private shadow capture ----------------------------------------------
//
// ATLAS still decides what the user demanded by reading their English. The
// client now declares it structurally, and nothing compares the two. This is
// that comparison, written to a private file and read by nobody: it decides
// nothing, reaches no wire, and exists only so a later corpus can say how often
// the two disagree and why.

func shadowEnv(t *testing.T, dir, name string) {
	t.Helper()
	t.Setenv("ATLAS_DIAGNOSTIC_DIR", dir)
	t.Setenv("ATLAS_SHADOW_CAPTURE", name)
}

func readShadowRecords(t *testing.T, path string) []map[string]interface{} {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("capture unreadable: %v", err)
	}
	var out []map[string]interface{}
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var m map[string]interface{}
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("malformed JSONL line %q: %v", line, err)
		}
		out = append(out, m)
	}
	return out
}

// Disabled is the deployed default: nothing opens, nothing is written.
func TestShadowSinkDisabledByDefault(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ATLAS_DIAGNOSTIC_DIR", dir)
	// ATLAS_SHADOW_CAPTURE deliberately unset.
	sink, err := openShadowSink()
	if err != nil {
		t.Fatalf("disabled must not error: %v", err)
	}
	if sink != nil {
		t.Fatal("a sink was opened with no capture configured")
	}
	if !sink.enabled() { // nil receiver must answer false, not panic
		// expected
	} else {
		t.Error("a nil sink reports enabled")
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Errorf("disabled mode created %d file(s)", len(entries))
	}
}

// Initialization refuses anything it cannot own: an existing destination, or
// one outside the capture root.
func TestShadowSinkInitializationRefusals(t *testing.T) {
	t.Run("existing destination", func(t *testing.T) {
		dir := t.TempDir()
		os.WriteFile(filepath.Join(dir, "run.jsonl"), []byte("old\n"), 0o644)
		shadowEnv(t, dir, "run.jsonl")
		if _, err := openShadowSink(); err == nil {
			t.Error("an existing capture was accepted; a mixed run would look like one run")
		}
	})
	for _, escape := range []string{"../outside.jsonl", "/etc/passwd", "sub/../../out.jsonl"} {
		t.Run("escape "+escape, func(t *testing.T) {
			dir := t.TempDir()
			shadowEnv(t, dir, escape)
			if _, err := openShadowSink(); err == nil {
				t.Errorf("%q escaped the capture root and was accepted", escape)
			}
		})
	}
}

// A full queue drops, counts, and never blocks the caller.
func TestShadowSinkDropsWhenFull(t *testing.T) {
	dir := t.TempDir()
	f, err := os.Create(filepath.Join(dir, "run.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	// The writer is not started, so nothing drains the queue while it fills.
	sink := newShadowSink(f)
	for i := 0; i < shadowQueueDepth*4; i++ {
		sink.submit(map[string]interface{}{"record_kind": "x", "i": i})
	}
	go sink.run()
	if err := sink.close(context.Background(), 5*time.Second); err != nil {
		t.Fatalf("close: %v", err)
	}
	recs := readShadowRecords(t, filepath.Join(dir, "run.jsonl"))
	footer := recs[len(recs)-1]
	if footer["record_kind"] != "task_contract_shadow_footer" {
		t.Fatalf("last record is not a footer: %v", footer["record_kind"])
	}
	if footer["dropped"].(float64) <= 0 {
		t.Error("a saturated queue reported no drops")
	}
	if footer["accepted"].(float64) != float64(shadowQueueDepth*4) {
		t.Errorf("accepted=%v", footer["accepted"])
	}
}

// Duplicate request IDs are a capture defect, never merged.
func TestShadowSinkFlagsDuplicateRequestIDs(t *testing.T) {
	dir := t.TempDir()
	shadowEnv(t, dir, "run.jsonl")
	sink, _ := openShadowSink()
	sink.noteRequest("req-1")
	sink.noteRequest("req-2")
	sink.noteRequest("req-1") // same id twice in one capture
	sink.close(context.Background(), 5*time.Second)
	recs := readShadowRecords(t, filepath.Join(dir, "run.jsonl"))
	footer := recs[len(recs)-1]
	if footer["duplicate_request_ids"].(float64) != 1 {
		t.Errorf("duplicate_request_ids=%v, want 1", footer["duplicate_request_ids"])
	}
}

// A capture with no footer is detectable as incomplete.
func TestShadowCaptureWithoutFooterIsIncomplete(t *testing.T) {
	dir := t.TempDir()
	shadowEnv(t, dir, "run.jsonl")
	sink, _ := openShadowSink()
	sink.submit(map[string]interface{}{"record_kind": "task_contract_shadow_request"})
	sink.abandonForTest() // ungraceful: writer stops, no footer
	recs := readShadowRecords(t, filepath.Join(dir, "run.jsonl"))
	for _, r := range recs {
		if r["record_kind"] == "task_contract_shadow_footer" {
			t.Error("an abandoned capture wrote a footer")
		}
	}
}

// shadowLoopRun drives the real agent loop with capture optionally enabled and
// returns the emitted records plus everything a causal comparison needs.
func shadowLoopRun(t *testing.T, capture bool, contract *TaskContract, prompt string,
	plan func(i int) map[string]interface{}) ([]map[string]interface{}, string, []string, string, []string) {
	t.Helper()
	dir := t.TempDir()
	capDir := t.TempDir()
	if capture {
		shadowEnv(t, capDir, "run.jsonl")
		sink, err := openShadowSink()
		if err != nil {
			t.Fatalf("sink: %v", err)
		}
		activeShadowSink.Store(sink)
		t.Cleanup(func() {
			sink.close(context.Background(), 5*time.Second)
			activeShadowSink.Store(nil)
		})
	} else {
		activeShadowSink.Store(nil)
	}

	turns := 0
	var mu sync.Mutex
	var events []string
	var modelBodies []string
	terminal := map[string]string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/v3/"), strings.HasPrefix(r.URL.Path, "/internal/"):
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		case strings.HasSuffix(r.URL.Path, "/syntax-check"):
			json.NewEncoder(w).Encode(map[string]interface{}{"valid": true})
			return
		case strings.HasSuffix(r.URL.Path, "/execute"):
			var in struct{ Code string }
			json.NewDecoder(r.Body).Decode(&in)
			if strings.Contains(in.Code, ".atlas-mount-probe") {
				b, _ := os.ReadFile(filepath.Join(dir, ".atlas-mount-probe"))
				json.NewEncoder(w).Encode(map[string]interface{}{
					"success": true, "stdout": string(b), "exit_code": 0})
				return
			}
			json.NewEncoder(w).Encode(map[string]interface{}{
				"success": true, "stdout": "", "exit_code": 0})
			return
		case !strings.HasSuffix(r.URL.Path, "/v1/chat/completions"):
			http.NotFound(w, r)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		mu.Lock()
		modelBodies = append(modelBodies, string(raw))
		i := turns
		turns++
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		call, _ := json.Marshal(plan(i))
		d, _ := json.Marshal(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": string(call)}}}})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", d)
	}))
	defer srv.Close()

	reqCtx := context.WithValue(context.Background(), requestIDKey, "req-shadow-1")
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.Ctx = reqCtx
	ctx.InferenceURL, ctx.SandboxURL, ctx.V3URL = srv.URL, srv.URL, srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.TrustMode = trustFullyTrusted
	ctx.VerifyOnHost = true
	ctx.MaxTurns = 0
	ctx.TaskContract = contract
	ctx.StreamFn = func(et string, data interface{}) {
		b, _ := json.Marshal(data)
		mu.Lock()
		defer mu.Unlock()
		var m map[string]interface{}
		line := et + "|" + string(b)
		if json.Unmarshal(b, &m) == nil {
			for _, k := range []string{"elapsed", "prompt_ms", "ms", "elapsed_ms",
				"duration_ms", "wall_s"} {
				delete(m, k)
			}
			c, _ := json.Marshal(m)
			line = et + "|" + string(c)
		}
		events = append(events, line)
		if et == "done" {
			var mm map[string]string
			json.Unmarshal(b, &mm)
			for k, v := range mm {
				terminal[k] = v
			}
		}
	}
	runAgentLoop(ctx, prompt)

	// Each run gets its own TempDir, so the workspace path is not a property of
	// the run; canonicalise it before anything compares two runs.
	for i, e := range events {
		events[i] = strings.ReplaceAll(e, dir, "<ws>")
	}
	for i, b := range modelBodies {
		modelBodies[i] = strings.ReplaceAll(b, dir, "<ws>")
	}

	var disk []string
	filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(dir, p)
		if !strings.HasPrefix(rel, ".") {
			b, _ := os.ReadFile(p)
			disk = append(disk, rel+"="+hashBytes(b)[:12])
		}
		return nil
	})
	sort.Strings(disk)

	var recs []map[string]interface{}
	if capture {
		if s := activeShadowSink.Load(); s != nil {
			s.close(context.Background(), 5*time.Second)
			recs = readShadowRecords(t, filepath.Join(capDir, "run.jsonl"))
		}
	}
	norm := make([]string, 0, len(modelBodies))
	for _, b := range modelBodies {
		norm = append(norm, canonPromptBody(b))
	}
	return recs, strings.Join(events, "\n"), disk,
		terminal["status"] + "/" + terminal["reason"], norm
}

// canonPromptBody puts one upstream request body into a form two runs can be
// compared byte for byte in.
//
// Three places already build the wire by ranging toolRegistry, a Go map, so
// their order is randomised per request independently of anything under test:
// the response_format tool-name enum (tools.go buildToolCallSchemaForTools),
// the GBNF tool-name alternation (buildToolCallGrammar), and the "### <tool>"
// documentation blocks in the system prompt (buildToolsDoc). Exactly those
// three orderings are canonicalised here and nothing else: any added, removed
// or altered prompt byte still shows up as a difference, and the ordering of
// the messages array, which carries meaning, is left alone.
func canonPromptBody(raw string) string {
	var body interface{}
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		return raw
	}
	var walk func(v interface{}) interface{}
	walk = func(v interface{}) interface{} {
		switch t := v.(type) {
		case map[string]interface{}:
			for k, e := range t {
				t[k] = walk(e)
			}
			return t
		case []interface{}:
			allStrings := len(t) > 0
			for i, e := range t {
				t[i] = walk(e)
				if _, ok := e.(string); !ok {
					allStrings = false
				}
			}
			if allStrings {
				sort.Slice(t, func(i, j int) bool {
					return t[i].(string) < t[j].(string)
				})
			}
			return t
		case string:
			return canonPromptString(t)
		}
		return v
	}
	out, err := json.Marshal(walk(body))
	if err != nil {
		return raw
	}
	return string(out)
}

func canonPromptString(s string) string {
	if strings.Contains(s, "\n### ") {
		parts := strings.Split(s, "\n### ")
		// Whatever follows the last tool block belongs to the section after
		// the tool list, not to the block it happens to trail; detach it so
		// sorting cannot carry it to a random position.
		tail := ""
		for i, seg := range parts[1:] {
			if idx := strings.Index(seg, "\n## "); idx >= 0 {
				tail, parts[i+1] = seg[idx:], seg[:idx]
			}
		}
		sort.Strings(parts[1:])
		s = strings.Join(parts, "\n### ") + tail
	}
	if !strings.Contains(s, "::=") || !strings.Contains(s, " | ") {
		return s
	}
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		head, body, ok := strings.Cut(line, "::=")
		if !ok || !strings.Contains(body, " | ") {
			continue
		}
		alts := strings.Split(body, " | ")
		sort.Strings(alts)
		lines[i] = head + "::=" + strings.Join(alts, " | ")
	}
	return strings.Join(lines, "\n")
}

func shadowWorkPlan(i int) map[string]interface{} {
	switch i {
	case 0:
		return map[string]interface{}{"type": "tool_call", "name": "list_directory",
			"args": map[string]string{"path": "."}}
	case 1:
		return map[string]interface{}{"type": "tool_call", "name": "write_file",
			"args": map[string]string{"path": "app.py", "content": "A = 1\n"}}
	}
	return map[string]interface{}{"type": "done", "summary": "wrote app.py"}
}

// The two record kinds, the two gate sites, and the value production consumed.
func TestShadowRecordsCaptureBothGateSites(t *testing.T) {
	contract := &TaskContract{TaskMode: TaskModeWork}
	recs, _, _, term, _ := shadowLoopRun(t, true, contract, "Create app.py.", shadowWorkPlan)
	t.Logf("terminal=%s records=%d", term, len(recs))

	var snapshots, gates []map[string]interface{}
	sites := map[string]bool{}
	seqs := map[float64]bool{}
	for _, r := range recs {
		switch r["record_kind"] {
		case "task_contract_shadow_request":
			snapshots = append(snapshots, r)
		case "task_contract_shadow_gate":
			gates = append(gates, r)
			sites[r["call_site"].(string)] = true
			seqs[r["gate_seq"].(float64)] = true
		}
	}
	if len(snapshots) != 1 {
		t.Fatalf("%d request snapshots, want exactly one", len(snapshots))
	}
	if len(gates) == 0 {
		t.Fatal("no gate evaluations were observed")
	}
	for _, r := range append(snapshots, gates...) {
		if r["influences_live_decision"] != false {
			t.Error("a record does not declare itself inert")
		}
		if r["request_id"] != "req-shadow-1" {
			t.Errorf("request_id=%v", r["request_id"])
		}
	}
	// Gate sequences are per-request and strictly increasing.
	if len(seqs) != len(gates) {
		t.Errorf("gate_seq collided across %d gates", len(gates))
	}
	// Every gate carries the live boolean and a closed comparison.
	for _, g := range gates {
		if _, ok := g["legacy_wants_state_change"].(bool); !ok {
			t.Error("a gate record has no live decision")
		}
		switch g["comparison"] {
		case shadowAgreeWork, shadowAgreeQuestion, shadowContractWorkLegacyQuestion,
			shadowContractQuestionLegacyWork, shadowUnmeasured:
		default:
			t.Errorf("comparison %q is outside the closed vocabulary", g["comparison"])
		}
	}
	t.Logf("gate sites observed: %v", sites)
}

// A contractless request is unmeasured, never inferred.
func TestShadowContractlessGateIsUnmeasured(t *testing.T) {
	recs, _, _, _, _ := shadowLoopRun(t, true, nil, "Create app.py.", shadowWorkPlan)
	saw := false
	for _, r := range recs {
		if r["record_kind"] != "task_contract_shadow_gate" {
			continue
		}
		saw = true
		if r["comparison"] != shadowUnmeasured {
			t.Errorf("a contractless gate was classified %q", r["comparison"])
		}
	}
	if !saw {
		t.Fatal("no gate records")
	}
	for _, r := range recs {
		if r["record_kind"] == "task_contract_shadow_request" &&
			r["contract_present"] != false {
			t.Error("a contractless request claimed a contract")
		}
	}
}

// Undeclared lists stay non-comparable; verification never claims equality.
func TestShadowUndeclaredListsAreNonComparable(t *testing.T) {
	recs, _, _, _, _ := shadowLoopRun(t, true, &TaskContract{TaskMode: TaskModeWork},
		"Create app.py.", shadowWorkPlan)
	for _, r := range recs {
		if r["record_kind"] != "task_contract_shadow_request" {
			continue
		}
		if r["output_comparison"] != shadowNotDeclared {
			t.Errorf("output_comparison=%v, want contract_not_declared", r["output_comparison"])
		}
		if r["verification_comparison"] != shadowNotDeclared {
			t.Errorf("verification_comparison=%v", r["verification_comparison"])
		}
	}
	// A declared verification never reports command equality with a boolean.
	recs2, _, _, _, _ := shadowLoopRun(t, true,
		&TaskContract{TaskMode: TaskModeWork, Verification: []string{"go test ./..."}},
		"Fix the failing test.", shadowWorkPlan)
	for _, r := range recs2 {
		if r["record_kind"] != "task_contract_shadow_request" {
			continue
		}
		got := r["verification_comparison"]
		if got != shadowVerifyLegacyRequires && got != shadowVerifyLegacyDoesNot {
			t.Errorf("verification_comparison=%v", got)
		}
		if strings.Contains(fmt.Sprint(got), "exact") {
			t.Error("verification claimed exact command agreement with a boolean heuristic")
		}
	}
}

// Aliases collapse to one canonical identity through the existing resolver.
func TestShadowOutputAliasesShareIdentity(t *testing.T) {
	recs, _, _, _, _ := shadowLoopRun(t, true,
		&TaskContract{TaskMode: TaskModeWork, ExpectedOutputs: []string{"app.py", "./app.py"}},
		"Create app.py.", shadowWorkPlan)
	for _, r := range recs {
		if r["record_kind"] != "task_contract_shadow_request" {
			continue
		}
		hashes := r["output_hashes"].([]interface{})
		if len(hashes) != 1 {
			t.Errorf("aliases produced %d identities, want 1: %v", len(hashes), hashes)
		}
	}
}

// No raw prose, path, command, or token reaches the file.
func TestShadowRecordsCarryNoRawText(t *testing.T) {
	const secret = "Create app.py with the SECRETMARKER inside."
	recs, _, _, _, _ := shadowLoopRun(t, true,
		&TaskContract{TaskMode: TaskModeWork, ExpectedOutputs: []string{"app.py"},
			Verification: []string{"pytest --marker=SECRETCMD"}},
		secret, shadowWorkPlan)
	blob, _ := json.Marshal(recs)
	for _, banned := range []string{"SECRETMARKER", "SECRETCMD", "pytest", "app.py",
		"Create app.py"} {
		if strings.Contains(string(blob), banned) {
			t.Errorf("the capture leaked %q", banned)
		}
	}
}

// The whole point: enabling capture changes nothing the user can observe.
func TestShadowCaptureIsCausallyInert(t *testing.T) {
	contract := &TaskContract{TaskMode: TaskModeWork,
		ExpectedOutputs: []string{"app.py"}, Verification: []string{"pytest"}}
	_, evOff, diskOff, termOff, promptOff := shadowLoopRun(t, false, contract, "Create app.py.", shadowWorkPlan)
	recs, evOn, diskOn, termOn, promptOn := shadowLoopRun(t, true, contract, "Create app.py.", shadowWorkPlan)
	t.Logf("off=%s on=%s records=%d", termOff, termOn, len(recs))
	if len(recs) == 0 {
		t.Fatal("capture was enabled but recorded nothing")
	}
	if termOff != termOn {
		t.Errorf("terminal differs: %q vs %q", termOff, termOn)
	}
	if strings.Join(diskOff, ",") != strings.Join(diskOn, ",") {
		t.Errorf("disk differs:\n  %v\n  %v", diskOff, diskOn)
	}
	if len(promptOff) != len(promptOn) {
		t.Fatalf("turn count differs: %d vs %d", len(promptOff), len(promptOn))
	}
	for i := range promptOff {
		if promptOff[i] != promptOn[i] {
			k := 0
			for k < len(promptOff[i]) && k < len(promptOn[i]) && promptOff[i][k] == promptOn[i][k] {
				k++
			}
			lo := k - 60
			if lo < 0 {
				lo = 0
			}
			clip := func(v string) string {
				hi := k + 100
				if hi > len(v) {
					hi = len(v)
				}
				return v[lo:hi]
			}
			t.Fatalf("model prompt bytes differ on turn %d at byte %d:\n  off: %s\n  on:  %s",
				i, k, clip(promptOff[i]), clip(promptOn[i]))
		}
	}
	if evOff != evOn {
		a, b := strings.Split(evOff, "\n"), strings.Split(evOn, "\n")
		for i := 0; i < len(a) || i < len(b); i++ {
			var x, y string
			if i < len(a) {
				x = a[i]
			}
			if i < len(b) {
				y = b[i]
			}
			if x != y {
				j := 0
				for j < len(x) && j < len(y) && x[j] == y[j] {
					j++
				}
				lo := j - 80
				if lo < 0 {
					lo = 0
				}
				clip := func(v string) string {
					hi := j + 120
					if hi > len(v) {
						hi = len(v)
					}
					if lo > len(v) {
						return "<short>"
					}
					return v[lo:hi]
				}
				t.Fatalf("event/model stream differs at entry %d byte %d:\n  off: %s\n  on:  %s",
					i, j, clip(x), clip(y))
			}
		}
	}
}

// Capture off costs a run nothing: no sink, no goroutine, no allocation on the
// gate path, and the gate still returns exactly what production asked for.
func TestShadowDisabledCostsNothing(t *testing.T) {
	activeShadowSink.Store(nil)
	before := runtime.NumGoroutine()
	_, _, _, term, _ := shadowLoopRun(t, false, &TaskContract{TaskMode: TaskModeWork},
		"Create app.py.", shadowWorkPlan)
	t.Logf("terminal=%s", term)
	deadline := time.Now().Add(2 * time.Second)
	after := runtime.NumGoroutine()
	for after > before && time.Now().Before(deadline) {
		runtime.Gosched()
		after = runtime.NumGoroutine()
	}
	if after > before {
		t.Errorf("goroutines grew with capture off: %d -> %d", before, after)
	}
	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	st := &runState{}
	for _, live := range []bool{true, false} {
		if got := observeStateChangeGate(ctx, st, shadowGateActionGate, live); got != live {
			t.Errorf("the disabled gate returned %v for %v", got, live)
		}
	}
	allocs := testing.AllocsPerRun(100, func() {
		observeStateChangeGate(ctx, st, shadowGateActionGate, true)
	})
	if allocs != 0 {
		t.Errorf("the disabled gate allocated %v per call", allocs)
	}
	if st.shadowGate.n != 0 {
		t.Errorf("the disabled gate advanced its sequence to %d", st.shadowGate.n)
	}
}

// The recorded value follows the workspace-inspection state production was
// holding at the moment of evaluation, not the request text.
//
// Both gate sites are evaluated at the completion boundary, so a single
// request cannot straddle the transition: by the time the gate runs, any
// inspection this turn has already happened. The transition is therefore shown
// where production has it, across two runs of the identical neutral request
// that differ only in whether the workspace was inspected.
func TestShadowGateFollowsInspectionNotRequestText(t *testing.T) {
	const neutral = "app.py numbers"
	noInspect := func(i int) map[string]interface{} {
		return map[string]interface{}{"type": "done", "summary": "nothing to do"}
	}
	report := func(recs []map[string]interface{}) []string {
		var out []string
		for _, r := range recs {
			if r["record_kind"] != "task_contract_shadow_gate" {
				continue
			}
			out = append(out, fmt.Sprintf("%v/live=%v/insp=%v/%v",
				r["gate_seq"], r["legacy_wants_state_change"], r["inspected_workspace"],
				r["comparison"]))
			// The comparison is derived from the value production consumed.
			want := shadowContractWorkLegacyQuestion
			if r["legacy_wants_state_change"].(bool) {
				want = shadowAgreeWork
			}
			if r["comparison"] != want {
				t.Errorf("live=%v recorded comparison %q, want %q",
					r["legacy_wants_state_change"], r["comparison"], want)
			}
			// The gate records inspection state and the live value together,
			// so a disagreement between them is visible in the capture.
			if r["legacy_wants_state_change"] != r["inspected_workspace"] {
				t.Errorf("live=%v with inspected=%v on a neutral request",
					r["legacy_wants_state_change"], r["inspected_workspace"])
			}
		}
		if len(out) == 0 {
			t.Fatal("no gate records")
		}
		return out
	}
	contract := &TaskContract{TaskMode: TaskModeWork}
	cold, _, _, termCold, _ := shadowLoopRun(t, true, contract, neutral, noInspect)
	warm, _, _, termWarm, _ := shadowLoopRun(t, true, contract, neutral, shadowWorkPlan)
	coldSeq, warmSeq := report(cold), report(warm)
	t.Logf("uninspected %s: %v", termCold, coldSeq)
	t.Logf("inspected   %s: %v", termWarm, warmSeq)
	for _, e := range coldSeq {
		if !strings.Contains(e, "live=false") {
			t.Errorf("an uninspected run recorded %s", e)
		}
	}
	for _, e := range warmSeq {
		if !strings.Contains(e, "live=true") {
			t.Errorf("an inspected run recorded %s", e)
		}
	}
}

// A request rejected at the boundary never produces a snapshot claiming a
// contract the run does not have.
func TestShadowRejectedContractProducesNoSnapshot(t *testing.T) {
	capDir := t.TempDir()
	shadowEnv(t, capDir, "reject.jsonl")
	sink, err := openShadowSink()
	if err != nil {
		t.Fatalf("sink: %v", err)
	}
	activeShadowSink.Store(sink)
	defer activeShadowSink.Store(nil)

	dir := t.TempDir()
	for _, bad := range []string{
		`{"task_mode":"explore"}`,
		`{"task_mode":"work","expected_outputs":["../../etc/passwd"]}`,
		`{"task_mode":""}`,
	} {
		var in TaskContract
		if err := json.Unmarshal([]byte(bad), &in); err != nil {
			t.Fatalf("fixture %s: %v", bad, err)
		}
		if _, err := validateTaskContract(&in, dir); err == nil {
			t.Fatalf("%s was accepted", bad)
		}
	}
	sink.close(context.Background(), 5*time.Second)
	for _, r := range readShadowRecords(t, filepath.Join(capDir, "reject.jsonl")) {
		if r["record_kind"] != "task_contract_shadow_footer" {
			t.Errorf("a rejected contract produced %v", r["record_kind"])
		}
	}
}

// Structural: the shadow path cannot reach a live decision.
//
// Three properties, each read off the syntax tree rather than trusted:
//   - observeStateChangeGate returns its live argument and nothing else, and
//     never assigns to it, so instrumenting a gate cannot change its value.
//   - wantsStateChange is called exactly once per live site, and each call is
//     an argument to observeStateChangeGate, so the recorded value is the
//     value production consumed and the heuristic runs once.
//   - outside the shadow implementation itself, no shadow identifier appears
//     anywhere in production code, so no shadow state can be branched on.
func TestShadowHasNoPathIntoPolicy(t *testing.T) {
	fset := token.NewFileSet()
	files := map[string]*ast.File{}
	entries, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range entries {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		files[name] = f
	}

	// 1. the observer is an identity function on its live argument
	var observer *ast.FuncDecl
	for _, f := range files {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == "observeStateChangeGate" {
				observer = fd
			}
		}
	}
	if observer == nil {
		t.Fatal("observeStateChangeGate is gone")
	}
	returns := 0
	ast.Inspect(observer.Body, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.ReturnStmt:
			returns++
			if len(v.Results) != 1 {
				t.Errorf("a return yields %d values", len(v.Results))
				return true
			}
			id, ok := v.Results[0].(*ast.Ident)
			if !ok || id.Name != "live" {
				t.Errorf("a return does not yield the live value: %T", v.Results[0])
			}
		case *ast.AssignStmt:
			for _, lhs := range v.Lhs {
				if id, ok := lhs.(*ast.Ident); ok && id.Name == "live" {
					t.Error("the observer assigns to the live value")
				}
			}
		}
		return true
	})
	if returns == 0 {
		t.Error("the observer never returns")
	}

	// 2. every live wantsStateChange call is an argument to the observer
	sites := 0
	for name, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			id, ok := call.Fun.(*ast.Ident)
			if !ok {
				return true
			}
			switch id.Name {
			case "wantsStateChange":
				sites++
			case "observeStateChangeGate":
				inner := 0
				for _, a := range call.Args {
					ast.Inspect(a, func(m ast.Node) bool {
						if c, ok := m.(*ast.CallExpr); ok {
							if ci, ok := c.Fun.(*ast.Ident); ok && ci.Name == "wantsStateChange" {
								inner++
							}
						}
						return true
					})
				}
				if inner != 1 {
					t.Errorf("%s: an observed gate wraps %d wantsStateChange calls, want 1",
						name, inner)
				}
			}
			return true
		})
	}
	if sites != 2 {
		t.Errorf("%d live wantsStateChange calls, want the 2 observed gate sites", sites)
	}

	// 3. shadow identifiers stay inside the shadow implementation
	allowed := map[string]bool{
		"observeStateChangeGate":    true,
		"emitShadowRequestSnapshot": true,
		"contractOutputs":           true,
		"shadowCanonicalSet":        true,
		"shadowHashes":              true,
		"shadowCompareSets":         true,
		"shadowHash":                true,
		"openShadowSink":            true,
		"newShadowSink":             true,
		"shadowCaptureRoot":         true,
		"main":                      true,
		"enabled":                   true,
		"run":                       true,
		"submit":                    true,
		"noteRequest":               true,
		"close":                     true,
		"pauseWriterForTest":        true,
		"resumeWriterForTest":       true,
		"abandonForTest":            true,
	}
	// The two call sites and the snapshot call may name the entry points.
	entryPoints := map[string]bool{
		"observeStateChangeGate":    true,
		"emitShadowRequestSnapshot": true,
		"shadowGateActionDemanded":  true,
		"shadowGateActionGate":      true,
	}
	for name, f := range files {
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || allowed[fd.Name.Name] || fd.Body == nil {
				continue
			}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				id, ok := n.(*ast.Ident)
				if !ok || !strings.HasPrefix(strings.ToLower(id.Name), "shadow") {
					return true
				}
				if entryPoints[id.Name] {
					return true
				}
				t.Errorf("%s: %s reads shadow state %q outside the shadow path",
					name, fd.Name.Name, id.Name)
				return true
			})
		}
	}
}

// abandonForTest stops the writer without a footer, the way a SIGKILL would.
func (s *shadowSink) abandonForTest() {
	close(s.queue)
	<-s.done
	s.f.Close()
}
