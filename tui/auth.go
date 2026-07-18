package main

// Internal service auth, client side. The `atlas` launcher
// (atlas/cli/commands/tui.py) resolves the checkout's
// secrets/service-token and passes its path via
// ATLAS_SERVICE_TOKEN_FILE; a cwd-relative secrets/service-token is
// the fallback for direct binary runs from the checkout. No token =>
// no header (works against an auth-disabled proxy).

import (
	"net/http"
	"os"
	"strings"
)

var serviceToken = loadServiceToken()

func loadServiceToken() string {
	path := os.Getenv("ATLAS_SERVICE_TOKEN_FILE")
	if path == "" {
		path = "secrets/service-token"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

type tokenTransport struct {
	base http.RoundTripper
}

func (t *tokenTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if serviceToken != "" && req.Header.Get("Authorization") == "" {
		req = req.Clone(req.Context())
		req.Header.Set("Authorization", "Bearer "+serviceToken)
	}
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(req)
}

// installTokenTransport covers every nil-Transport client in the TUI
// (chat SSE, permission/cancel/feedback POSTs, calibration probe,
// events stream) through the process default transport.
func installTokenTransport() {
	if serviceToken == "" {
		return
	}
	http.DefaultTransport = &tokenTransport{base: http.DefaultTransport}
}
