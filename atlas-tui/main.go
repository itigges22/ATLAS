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
	logPath := flag.String("log", envOr("ATLAS_TUI_LOG", ""),
		"append-only debug log path (default: off; alt-screen makes copy hard, "+
			"so tail this file to see what the TUI saw)")
	flag.Parse()

	if closer, err := initDebugLog(*logPath); err != nil {
		fmt.Fprintf(os.Stderr, "atlas-tui: %v\n", err)
		os.Exit(1)
	} else if closer != nil {
		defer closer()
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	model := newTUIModel(*proxyURL)

	// Surface the Python wrapper's startup warning (workspace mismatch
	// etc.) inside the TUI. Without this the warning prints to stderr
	// and is immediately covered by alt-screen.
	if note := os.Getenv("ATLAS_TUI_STARTUP_NOTE"); note != "" {
		model.chat = append(model.chat, chatMessage{
			Role: roleSystem, Meta: "startup", Body: note,
		})
	}

	// SSE consumer goroutine: pushes envelopes onto a channel that
	// the Bubbletea program drains via a tea.Cmd.
	go streamEventsWithReconnect(ctx, *proxyURL+"/events", model.events)

	// Mouse cell-motion capture so the wheel scrolls the chat pane.
	// Cell-motion (vs all-motion) only captures when buttons are held
	// or pressed, which keeps idle text selection working on most
	// modern terminals. iTerm2/Kitty/WezTerm let users hold Option/
	// Shift while dragging to override capture entirely; that's the
	// recommended escape hatch when copy/paste is needed.
	prog := tea.NewProgram(model, tea.WithAltScreen(),
		tea.WithMouseCellMotion())

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
