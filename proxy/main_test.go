package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
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
