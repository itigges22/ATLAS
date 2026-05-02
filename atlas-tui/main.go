// PC-062: Bubbletea TUI for ATLAS — main entry point.
//
// Connects to atlas-proxy /events (typed envelope SSE stream from
// PC-061) and renders a 3-pane layout: pipeline progress, event log,
// stats + chat input. Replaces Aider as the canonical chat UI for
// users who opt in via `atlas tui`.
//
// Bubbletea model is in model.go; pane rendering in panes.go;
// chat/agent client in chat.go; SSE consumer in consumer.go.

package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
)

const (
	defaultProxyURL = "http://localhost:8090"
)

func main() {
	proxyURL := flag.String("proxy", envOr("ATLAS_PROXY_URL", defaultProxyURL),
		"atlas-proxy base URL (default: $ATLAS_PROXY_URL or http://localhost:8090)")
	flag.Parse()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	model := newTUIModel(*proxyURL)

	// SSE consumer goroutine: pushes envelopes onto a channel that
	// the Bubbletea program drains via a tea.Cmd.
	go streamEventsWithReconnect(ctx, *proxyURL+"/events", model.events)

	prog := tea.NewProgram(model,
		tea.WithAltScreen(),       // alt-screen so quitting restores the terminal
		tea.WithMouseCellMotion(), // cheap to enable; future panes might want it
	)

	if _, err := prog.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "atlas-tui: %v\n", err)
		os.Exit(1)
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
