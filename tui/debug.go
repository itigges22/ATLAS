// Debug log — append-only file capturing everything that flows through
// the TUI so the live view isn't the only artifact. Bubbletea takes
// over the terminal in alt-screen mode, which makes copy/inspect hard;
// the log gives the operator (and Claude) a flat record to read after
// the fact.
//
// Enabled by --log <path> on the CLI or $ATLAS_TUI_LOG. Disabled by
// default — emitting events to a file always-on isn't free and would
// surprise users with mystery files.

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

var (
	dlogMu sync.Mutex
	dlogW  io.Writer = io.Discard
)

// initDebugLog opens path for append. Empty path → no-op (logger
// stays at io.Discard). Returns the close func or nil.
func initDebugLog(path string) (func(), error) {
	if path == "" {
		return nil, nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open log %s: %w", path, err)
	}
	dlogMu.Lock()
	dlogW = f
	dlogMu.Unlock()
	dlog("session", "started", map[string]interface{}{
		"pid": os.Getpid(),
	})
	return func() {
		dlog("session", "ended", nil)
		dlogMu.Lock()
		dlogW = io.Discard
		dlogMu.Unlock()
		_ = f.Close()
	}, nil
}

// dlog writes one timestamped line. category groups events (chat,
// event, user, slash, turn, conn); subject is a short tag; fields is
// optional structured data dumped as JSON.
//
// Format: `2026-05-02T17:03:21.123Z chat:text {"content":"Hi!"}`
func dlog(category, subject string, fields map[string]interface{}) {
	dlogMu.Lock()
	defer dlogMu.Unlock()
	if dlogW == io.Discard {
		return
	}
	ts := time.Now().UTC().Format("2006-01-02T15:04:05.000Z")
	line := fmt.Sprintf("%s %s:%s", ts, category, subject)
	if len(fields) > 0 {
		b, _ := json.Marshal(fields)
		line += " " + string(b)
	}
	_, _ = fmt.Fprintln(dlogW, line)
}
