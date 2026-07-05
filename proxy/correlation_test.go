package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

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
	defer withToken(t, "")()  // token off — isolate the ID-forwarding path
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
