// PC-062: pane renderers — pure functions from state → string.
//
// Panes:
//   pipelinePane — stage table with status icons + durations
//   eventsPane   — scrolling event log (raw envelope stream)
//   chatPane     — chat history (user + assistant + tool calls)
//   statsPane    — one-line counter strip
//   inputPane    — textarea (rendered by Bubbles, this file just frames it)

package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
)

var (
	bordStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("240"))

	bordStyleFocused = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("117"))

	titleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("117")).
			Bold(true)

	dimStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))

	okStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("42"))
	failStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
	runStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
	idleStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))

	chatUserStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("117")).
			Bold(true)

	chatAssistantStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("231"))

	chatToolStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("214"))

	chatSystemStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("245")).
			Italic(true)
)

// renderPipelinePane returns the pipeline pane content (no border).
// Caller wraps in a bordered box at the right size.
func renderPipelinePane(p *pipelineState, width int) string {
	stages := p.stages()
	if len(stages) == 0 {
		return dimStyle.Render("waiting for events…")
	}
	rows := make([]string, 0, len(stages))
	for _, s := range stages {
		rows = append(rows, renderPipelineRow(s, width))
	}
	return strings.Join(rows, "\n")
}

func renderPipelineRow(s *stageStatus, width int) string {
	icon, style := stageIcon(s)
	name := lipgloss.NewStyle().Width(16).Render(truncate(s.Name, 16))
	status := style.Width(8).Render(stageStatusLabel(s))
	dur := dimStyle.Width(8).Render(formatDuration(s.Duration()))
	detail := dimStyle.Render(truncate(s.Detail, max(0, width-40)))
	return fmt.Sprintf("%s  %s %s %s %s", icon, name, status, dur, detail)
}

func stageIcon(s *stageStatus) (string, lipgloss.Style) {
	if s.Running() {
		return runStyle.Render("⚙"), runStyle
	}
	if s.Success {
		return okStyle.Render("✓"), okStyle
	}
	return failStyle.Render("✗"), failStyle
}

func stageStatusLabel(s *stageStatus) string {
	if s.Running() {
		return "RUN"
	}
	if s.Success {
		return "OK"
	}
	return "FAIL"
}

func formatDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

// renderEventsPane returns the event log pane content. `height` is the
// usable inside height (caller has already accounted for the border).
func renderEventsPane(events []Envelope, height, width int) string {
	if height <= 0 {
		return ""
	}
	start := 0
	if len(events) > height {
		start = len(events) - height
	}
	lines := make([]string, 0, height)
	for _, ev := range events[start:] {
		lines = append(lines, formatEventLine(ev, width))
	}
	for len(lines) < height {
		lines = append(lines, "")
	}
	return strings.Join(lines, "\n")
}

// renderChatPane returns the chat history rendered for `height` rows.
// Renders bottom-up so the latest message stays in view; older messages
// scroll off the top. Markdown rendering for assistant text uses the
// passed glamour renderer (may be nil — falls back to plain).
func renderChatPane(chat []chatMessage, renderer *glamour.TermRenderer,
	height, width int) string {
	if height <= 0 {
		return ""
	}
	if len(chat) == 0 {
		return dimStyle.Render(
			"Type a message and press Enter to send it to the agent.")
	}

	// Render each message into a block of lines, then trim from the
	// front until total height fits. Bottom-anchored.
	blocks := make([][]string, 0, len(chat))
	total := 0
	for _, msg := range chat {
		block := renderChatMessage(msg, renderer, width)
		blocks = append(blocks, block)
		total += len(block) + 1 // +1 for blank-line separator
	}

	// Drop full blocks from the front until we fit.
	for total > height && len(blocks) > 1 {
		total -= len(blocks[0]) + 1
		blocks = blocks[1:]
	}

	// Stitch with single blank lines between blocks.
	out := []string{}
	for i, block := range blocks {
		if i > 0 {
			out = append(out, "")
		}
		out = append(out, block...)
	}

	// Trim from the top of the final assembled view if it's still
	// taller than `height` (one block was bigger than the pane).
	if len(out) > height {
		out = out[len(out)-height:]
	}
	// Pad to height so the box doesn't collapse upward.
	for len(out) < height {
		out = append([]string{""}, out...)
	}
	return strings.Join(out, "\n")
}

// renderChatMessage formats one chat row into a list of display lines.
func renderChatMessage(m chatMessage, renderer *glamour.TermRenderer,
	width int) []string {
	switch m.Role {
	case roleUser:
		header := chatUserStyle.Render("you")
		body := wrapPlain(m.Body, width-2)
		return prependPrefix(header, body)

	case roleAssistant:
		header := chatAssistantStyle.Bold(true).Render("agent")
		body := renderMarkdown(m.Body, renderer)
		return prependPrefix(header, body)

	case roleTool:
		mark := okStyle.Render("✓")
		if !m.Success && m.Body != "" && !looksLikeToolCall(m.Body) {
			mark = failStyle.Render("✗")
		}
		// tool_call has no Success, body is the args summary; tool_result
		// has Success and body is summary/error. Mark distinguishes.
		header := chatToolStyle.Render(fmt.Sprintf("%s tool · %s", mark, m.Meta))
		body := wrapPlain(m.Body, width-2)
		return prependPrefix(header, body)

	case roleSystem:
		tag := m.Meta
		if tag == "" {
			tag = "system"
		}
		header := chatSystemStyle.Render(fmt.Sprintf("· %s", tag))
		body := wrapPlain(m.Body, width-2)
		return prependPrefix(header, body)
	}
	return []string{m.Body}
}

// looksLikeToolCall returns true for a tool_call body (args summary,
// no success field set yet). Used to keep the args-line uncolored
// rather than marked failed.
func looksLikeToolCall(body string) bool {
	// Heuristic: tool_call summaries start with "path=", "command=", etc.
	// tool_result summaries are stdout/messages. Not perfect but the
	// alternative is plumbing an "isCall" bit through chatMessage.
	prefixes := []string{"path=", "command=", "old=", "new="}
	for _, p := range prefixes {
		if strings.HasPrefix(body, p) {
			return true
		}
	}
	return false
}

func renderMarkdown(body string, r *glamour.TermRenderer) []string {
	if r == nil {
		return wrapPlain(body, 80)
	}
	out, err := r.Render(body)
	if err != nil {
		return wrapPlain(body, 80)
	}
	out = strings.TrimRight(out, "\n")
	return strings.Split(out, "\n")
}

// wrapPlain hard-wraps long lines at the given width. Nothing fancy —
// just splits at whitespace when possible.
func wrapPlain(s string, width int) []string {
	if width < 8 {
		width = 8
	}
	out := []string{}
	for _, line := range strings.Split(s, "\n") {
		for len(line) > width {
			cut := width
			if idx := strings.LastIndex(line[:width], " "); idx > width/2 {
				cut = idx
			}
			out = append(out, line[:cut])
			line = strings.TrimLeft(line[cut:], " ")
		}
		out = append(out, line)
	}
	return out
}

func prependPrefix(header string, body []string) []string {
	out := make([]string, 0, len(body)+1)
	out = append(out, header)
	for _, line := range body {
		out = append(out, "  "+line)
	}
	return out
}

// renderStatsPane is a one-line summary of pipeline counters + active
// stage. Plain text — no border (rendered between chat and input).
func renderStatsPane(p *pipelineState, width int) string {
	parts := []string{}
	if active := p.activeStage(); active != nil {
		parts = append(parts, runStyle.Render(fmt.Sprintf("● %s", active.Name)))
	}
	if p.currentTurn > 0 {
		parts = append(parts, fmt.Sprintf("turn:%d", p.currentTurn))
	}
	parts = append(parts,
		fmt.Sprintf("tools:%d✓/%d✗", p.toolSuccesses, p.toolFailures),
		fmt.Sprintf("events:%d", p.totalEvents))
	if p.errors > 0 {
		parts = append(parts, failStyle.Render(fmt.Sprintf("errors:%d", p.errors)))
	}
	if p.done {
		mark := okStyle.Render("✓ done")
		if !p.doneSuccess {
			mark = failStyle.Render("✗ done")
		}
		parts = append(parts, mark, fmt.Sprintf("total:%dms", p.totalMS))
	}
	line := dimStyle.Render(strings.Join(parts, "  "))
	if lipgloss.Width(line) > width {
		// best-effort truncation by raw bytes (lipgloss styled strings
		// shouldn't overflow much in practice).
		line = line[:width]
	}
	return line
}

// layoutFullScreen stitches header + pipeline + chat + events + stats
// + input into a full-screen view.
//
// Vertical budget (per row, including borders):
//   header   1
//   pipeline up to 10  (8 inner + 2 border, capped)
//   chat     fills (≥5)
//   events   5  (3 inner + 2 border)
//   stats    1  (no border)
//   input    5  (3 inner + 2 border)
func layoutFullScreen(p *pipelineState, events []Envelope, chat []chatMessage,
	inputView string, renderer *glamour.TermRenderer, header string,
	width, height int) string {

	if width <= 0 || height <= 0 {
		return ""
	}

	const (
		headerH = 1
		eventsH = 5 // 3 inner + 2 border
		statsH  = 1
		inputH  = 5 // 3 inner + 2 border
	)
	pipelineRows := len(p.stages())
	if pipelineRows < 1 {
		pipelineRows = 1
	}
	pipelineH := pipelineRows + 2 // borders
	if pipelineH > 10 {
		pipelineH = 10
	}
	chatH := height - headerH - pipelineH - eventsH - statsH - inputH
	if chatH < 5 {
		// Squeeze the pipeline first, then events, before going below 5.
		shortfall := 5 - chatH
		if pipelineH-shortfall >= 3 {
			pipelineH -= shortfall
			chatH = 5
		} else {
			chatH = max(3, height-headerH-pipelineH-statsH-inputH-3)
		}
	}

	innerW := width - 2 // border consumes 2 cols on each box

	pipelineBox := bordStyle.Width(innerW).Render(
		titleStyle.Render(" Pipeline ") + "\n" +
			renderPipelinePane(p, innerW))

	chatBox := bordStyle.Width(innerW).Render(
		titleStyle.Render(" Chat ") + "\n" +
			renderChatPane(chat, renderer, chatH-3, innerW-2))

	eventsBox := bordStyle.Width(innerW).Render(
		titleStyle.Render(" Events ") + "\n" +
			renderEventsPane(events, eventsH-3, innerW))

	statsLine := renderStatsPane(p, width)

	inputBox := bordStyleFocused.Width(innerW).Render(
		titleStyle.Render(" Message ") + "\n" + inputView)

	return lipgloss.JoinVertical(lipgloss.Left,
		header,
		pipelineBox,
		chatBox,
		eventsBox,
		statsLine,
		inputBox,
	)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
