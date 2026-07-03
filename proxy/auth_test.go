// Tests for internal service authentication. All tokens are synthetic
// test fixtures.

package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

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
