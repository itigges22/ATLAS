package main

// Correlation IDs + structured logging.
//
// Every inbound request gets an X-ATLAS-Request-ID (read from the client
// or generated), echoed in the response and stored in the request
// context. Outbound calls to llama/v3/lens/sandbox forward the same ID
// (tokenTransport reads it from the request context), so one turn is
// traceable across services.
//
// Log format is line-oriented by default; ATLAS_LOG_FORMAT=json emits
// one JSON object per line with stable fields. Both paths still pass
// through the private-value filter (main() wraps the log writer).

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

const requestIDHeader = "X-ATLAS-Request-ID"

type ctxKey string

const requestIDKey ctxKey = "atlas-request-id"

func newRequestID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return "req-unknown"
	}
	return "req-" + hex.EncodeToString(b)
}

func requestIDFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(requestIDKey).(string); ok {
		return v
	}
	return ""
}

// withRequestID wraps a handler so every request carries a correlation
// ID (client-provided or generated), echoed back and put in the context.
func withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimSpace(r.Header.Get(requestIDHeader))
		if id == "" {
			id = newRequestID()
		}
		w.Header().Set(requestIDHeader, id)
		ctx := context.WithValue(r.Context(), requestIDKey, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// --- structured logging --------------------------------------------------

var logJSON = strings.EqualFold(os.Getenv("ATLAS_LOG_FORMAT"), "json")

// logEvent emits one structured record. In json mode it's a JSON object
// with stable fields; otherwise a readable line. request_id is included
// when present. Fields beyond the standard set are passed as kv pairs.
func logEvent(level, msg, requestID string, kv map[string]interface{}) {
	if logJSON {
		rec := map[string]interface{}{
			"ts":      time.Now().UTC().Format(time.RFC3339Nano),
			"level":   level,
			"service": "atlas-proxy",
			"version": APIVersion,
			"msg":     msg,
		}
		if requestID != "" {
			rec["request_id"] = requestID
		}
		for k, v := range kv {
			rec[k] = v
		}
		b, err := json.Marshal(rec)
		if err != nil {
			log.Printf("%s: %s", level, msg)
			return
		}
		log.Printf("%s", b)
		return
	}
	// line mode
	if requestID != "" {
		log.Printf("[%s] [%s] %s", level, requestID, msg)
	} else {
		log.Printf("[%s] %s", level, msg)
	}
}

// jsonLineWriter converts each (already private-value-filtered) log line
// into the same JSON record shape logEvent emits, so ATLAS_LOG_FORMAT=json
// covers every log call in the process, not only logEvent call sites.
// Lines that are already JSON objects (logEvent's json-mode output) pass
// through unchanged.
type jsonLineWriter struct {
	w io.Writer
}

func (j jsonLineWriter) Write(p []byte) (int, error) {
	line := bytes.TrimRight(p, "\n")
	if len(line) > 0 && line[0] == '{' && json.Valid(line) {
		return j.w.Write(p)
	}
	rec := map[string]interface{}{
		"ts":      time.Now().UTC().Format(time.RFC3339Nano),
		"level":   "info",
		"service": "atlas-proxy",
		"version": APIVersion,
		"msg":     string(line),
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return j.w.Write(p)
	}
	b = append(b, '\n')
	if _, err := j.w.Write(b); err != nil {
		return 0, err
	}
	return len(p), nil
}
