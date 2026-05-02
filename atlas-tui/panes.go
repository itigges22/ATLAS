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
	"github.com/charmbracelet/x/ansi"
)

// thinkingVerbs cycles through quirky phrases while the model is
// generating, so a long turn doesn't look stuck on a single word.
// Refreshed every ~3s based on spinnerFrame (which ticks at 150ms).
var thinkingVerbs = []string{
	"Thinking", "Pondering", "Cogitating", "Brewing", "Concocting",
	"Conjuring", "Synthesizing", "Mulling", "Reasoning", "Plotting",
	"Distilling", "Crafting", "Weaving", "Marinating", "Percolating",
	"Crystallizing", "Untangling", "Hypothesizing", "Sleuthing",
	"Decoding", "Composing", "Sculpting", "Architecting", "Brainstorming",
	"Wrangling", "Spelunking", "Unraveling", "Deliberating",
}

var (
	bordStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("240"))

	bordStyleFocused = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("117"))

	// Bash mode: leading "!" routes input to a shell command. Red
	// border signals "this WILL execute on your machine".
	bordStyleBash = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("196"))

	// Slash mode: leading "/" is a TUI command (no agent / shell).
	// Purple border + completion hint above the box.
	bordStyleSlash = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("141"))

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

	chatTurnStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("141")).
			Bold(true)

	// chatLLMStyle is intentionally dim — these rows are "machine
	// internals" (encoding/decoding/streaming JSON) and should sit
	// behind the brighter assistant/tool output rows.
	chatLLMStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("242")).
			Italic(true)

	chatV3Style = lipgloss.NewStyle().
			Foreground(lipgloss.Color("99"))
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
// scroll is the number of rows scrolled UP from the bottom; 0 means
// "follow the latest message". The value is clamped against the total
// line count so over-scrolling is harmless.
//
// Returns (rendered, atTop, atBottom, totalLines, viewStart) so the
// caller can show a scroll indicator in the chat title.
func renderChatPane(chat []chatMessage, renderer *glamour.TermRenderer,
	height, width, scroll int) (string, bool, bool, int, int) {
	if height <= 0 {
		return "", true, true, 0, 0
	}
	if len(chat) == 0 {
		empty := dimStyle.Render(
			"Type a message and press Enter to send it to the agent.")
		return empty, true, true, 0, 0
	}

	// Flatten every message into a list of display lines with blank
	// separators. We then take a (height)-tall window from the bottom
	// shifted up by `scroll` rows.
	allLines := []string{}
	for i, msg := range chat {
		if i > 0 {
			allLines = append(allLines, "")
		}
		allLines = append(allLines, renderChatMessage(msg, renderer, width)...)
	}
	// Defense in depth: hard-truncate any line wider than `width` using
	// ANSI-aware truncation. Glamour and other renderers should respect
	// our wrap settings, but a pathological line (long URL, code block
	// that didn't break, runaway styling) would otherwise expand the
	// chat box past the right column and cover the sidebar. lipgloss
	// Width() is a *minimum* — without this clamp, the box grows.
	for i, ln := range allLines {
		if lipgloss.Width(ln) > width {
			allLines[i] = ansi.Truncate(ln, width, "")
		}
	}
	total := len(allLines)

	// Clamp scroll so the user can't drift off either end.
	maxScroll := total - height
	if maxScroll < 0 {
		maxScroll = 0
	}
	if scroll > maxScroll {
		scroll = maxScroll
	}
	if scroll < 0 {
		scroll = 0
	}

	end := total - scroll
	start := end - height
	if start < 0 {
		start = 0
	}
	// Snapshot the full flattened chat for the highlight-to-copy
	// handler. Strings are ANSI-styled; the consumer strips ANSI
	// before pushing to clipboard.
	lastChatLines = allLines
	lastChatViewStart = start
	out := append([]string(nil), allLines[start:end]...)
	// Pad short content to height (newest stays at bottom). When we
	// pad ABOVE, the visual row → flat-line index relation shifts:
	// the first padded row maps to line 0 (start = 0), but the user's
	// mouse Y still indexes from the chat pane's top. The handler
	// guards against negative line indices, so this is OK.
	for len(out) < height {
		out = append([]string{""}, out...)
	}
	return strings.Join(out, "\n"), start == 0, scroll == 0, total, start
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
		// Meta convention from appendChatEvent:
		//   "→ <tool>" — pending call, no mark
		//   "← <tool>" — returned, render ✓ or ✗ based on m.Success
		var header string
		if strings.HasPrefix(m.Meta, "→") {
			header = chatToolStyle.Render("tool · " + m.Meta)
		} else {
			mark := okStyle.Render("✓")
			if !m.Success {
				mark = failStyle.Render("✗")
			}
			header = chatToolStyle.Render(fmt.Sprintf("%s tool · %s", mark, m.Meta))
		}
		body := wrapPlain(m.Body, width-2)
		return prependPrefix(header, body)

	case roleSystem:
		tag := m.Meta
		if tag == "" {
			tag = "system"
		}
		// Per-tag styling so the user can scan the chat at a glance:
		// turn separators in purple, LLM call markers in cyan italic,
		// V3 progress in violet, everything else in dim grey italic.
		var header string
		switch m.Meta {
		case "turn":
			header = chatTurnStyle.Render(fmt.Sprintf("── %s ──", m.Body))
			return []string{header}
		case "llm", "v3-llm":
			// Multi-line streaming bodies get a header line + indented
			// dim body. Single-line stats fit on one row. v3-llm uses
			// the same layout but inherits the V3 violet tint via the
			// switch below — pick the style first.
			lineStyle := chatLLMStyle
			if m.Meta == "v3-llm" {
				lineStyle = chatV3Style.Italic(true)
				tag = "v3"
			}
			if !strings.Contains(m.Body, "\n") {
				return []string{lineStyle.Render(fmt.Sprintf("· %s · %s", tag, m.Body))}
			}
			parts := strings.SplitN(m.Body, "\n", 2)
			out := []string{lineStyle.Render(fmt.Sprintf("· %s · %s", tag, parts[0]))}
			for _, ln := range wrapPlain(parts[1], width-4) {
				out = append(out, lineStyle.Render("  "+ln))
			}
			return out
		case "V3":
			header = chatV3Style.Render(fmt.Sprintf("· %s · %s", tag, m.Body))
			return []string{header}
		}
		header = chatSystemStyle.Render(fmt.Sprintf("· %s", tag))
		body := wrapPlain(m.Body, width-2)
		return prependPrefix(header, body)
	}
	return []string{m.Body}
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
func renderStatsPane(p *pipelineState, width int,
	lastTurnTokens, totalTokens, maxTokens int) string {
	parts := []string{}
	if active := p.activeStage(); active != nil {
		parts = append(parts, runStyle.Render(fmt.Sprintf("● %s", active.Name)))
	}
	if p.currentTurn > 0 {
		parts = append(parts, fmt.Sprintf("turn:%d", p.currentTurn))
	}
	// Token usage / context-window pressure. lastTurnTokens reflects the
	// most recent llm_call_end's total (Qwen3.5 reports prompt+completion
	// for the whole call). When that approaches maxTokens, the next turn
	// will start truncating history — flag in red over 80% utilization.
	if lastTurnTokens > 0 && maxTokens > 0 {
		pct := float64(lastTurnTokens) / float64(maxTokens) * 100
		ctxStr := fmt.Sprintf("ctx:%s/%s (%.0f%%)",
			formatTokens(lastTurnTokens), formatTokens(maxTokens), pct)
		switch {
		case pct >= 80:
			parts = append(parts, failStyle.Render(ctxStr))
		case pct >= 50:
			parts = append(parts, runStyle.Render(ctxStr))
		default:
			parts = append(parts, ctxStr)
		}
	}
	if totalTokens > 0 {
		parts = append(parts, fmt.Sprintf("session:%s",
			formatTokens(totalTokens)))
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
		line = ansi.Truncate(line, width, "")
	}
	return line
}

// formatTokens renders a token count in a compact form (e.g. 12k, 1.4k).
// Used by the stats line so a 32768-token max doesn't blow the width
// budget on narrow terminals.
// slashCommandList drives the slash-mode autocomplete hint. Order is
// the order users see; keep frequent commands first.
var slashCommandList = []string{
	"/help", "/clear", "/compact",
	"/add", "/drop", "/context",
	"/diff", "/commit", "/undo",
	"/run", "/hide", "/show",
	"/mouse", "/copy", "/quit",
}

// renderSlashHint shows a one-line list of slash commands matching
// the current prefix. Renders empty when the prefix matches nothing
// (e.g. "/foobar" — let the unknown-command error explain itself).
func renderSlashHint(value string, width int) string {
	prefix := strings.TrimSpace(value)
	matches := []string{}
	for _, c := range slashCommandList {
		if strings.HasPrefix(c, prefix) {
			matches = append(matches, c)
		}
	}
	if len(matches) == 0 {
		return ""
	}
	header := lipgloss.NewStyle().
		Foreground(lipgloss.Color("141")).
		Bold(true).
		Render("commands:")
	body := dimStyle.Render(strings.Join(matches, "  "))
	line := header + " " + body
	if lipgloss.Width(line) > width {
		line = ansi.Truncate(line, width, "")
	}
	return line
}

// renderHelpHint shows the full slashCommandHelp body above the input
// box while the user is in "?" mode. Live hint, no Enter required —
// matches how /slash shows its command dropdown. Pressing Enter still
// routes through /help (so the help also lands in chat scrollback).
func renderHelpHint(width int) string {
	header := lipgloss.NewStyle().
		Foreground(lipgloss.Color("220")).
		Bold(true).
		Render("help · press Enter to dismiss, or just keep typing")
	body := dimStyle.Render(slashCommandHelp)
	out := header + "\n" + body
	// Width-clamp each rendered line so a wide help body doesn't blow
	// past the input column on narrow terminals.
	lines := strings.Split(out, "\n")
	for i, line := range lines {
		if lipgloss.Width(line) > width {
			lines[i] = ansi.Truncate(line, width, "")
		}
	}
	return strings.Join(lines, "\n")
}

// renderBashHint shows a single warning row above the bash input box.
// Red text; brief enough to fit a single row at any reasonable width.
func renderBashHint(_ string, width int) string {
	header := lipgloss.NewStyle().
		Foreground(lipgloss.Color("196")).
		Bold(true).
		Render("bash:")
	body := dimStyle.Render(
		"Enter runs as a shell command in the working dir.")
	line := header + " " + body
	if lipgloss.Width(line) > width {
		line = ansi.Truncate(line, width, "")
	}
	return line
}

// formatTokens renders a token count compactly. Below 10,000 we keep
// the raw integer so the user sees per-token movement during
// streaming (1234 → 1235 → 1236…). At 10k+ the count changes too
// quickly for raw digits to be readable, so we abbreviate.
func formatTokens(n int) string {
	switch {
	case n >= 100000:
		return fmt.Sprintf("%dk", n/1000)
	case n >= 10000:
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

// layoutFullScreen stitches header + sidebar + pipeline + chat +
// events + stats + input into a full-screen view.
//
// Horizontal budget:
//   files    sidebar — 28 cols when terminal ≥ 110 cols, hidden below
//   right    everything else
//
// Vertical budget (per row, including borders), right column:
//   header   1
//   pipeline up to 10  (8 inner + 2 border, capped)
//   chat     fills (≥5)
//   events   5  (3 inner + 2 border)
//   stats    1  (no border)
//   input    5  (3 inner + 2 border)
func layoutFullScreen(p *pipelineState, events []Envelope, chat []chatMessage,
	inputView, inputValue, inputMode string,
	renderer *glamour.TermRenderer, header string,
	turnActive bool, spinnerFrame, chatScroll int,
	files []fileEntry, modified map[string]bool, fileScroll int, fileRoot string,
	lastTurnTokens, totalTokens, maxTokens int,
	hideFiles, hidePipeline, hideEvents bool,
	width, height int) (string, int) {

	if width <= 0 || height <= 0 {
		return "", 0
	}

	const (
		headerH = 1
		statsH  = 1
	)
	// Input box: 5 rows by default (3 inner + 2 border). bash/slash
	// mode adds a one-row hint banner above, so reserve one extra row.
	// help mode shows the full multi-line slashCommandHelp body — count
	// its lines so the box doesn't bleed into the chat pane.
	inputH := 5
	switch inputMode {
	case "bash", "slash":
		inputH = 6
	case "help":
		// +1 header + N body lines + 1 spacer row before the input box.
		// Cap at half the terminal height so the chat pane stays
		// visible on small terminals — the help body is dense but
		// scrollable in the user's terminal so a clip is acceptable.
		helpLines := strings.Count(slashCommandHelp, "\n") + 1
		inputH = 5 + 1 + helpLines + 1
		if cap := height / 2; cap > 8 && inputH > cap {
			inputH = cap
		}
	}
	eventsH := 5 // 3 inner + 2 border
	if hideEvents {
		eventsH = 0
	}
	pipelineH := 0
	if !hidePipeline {
		pipelineRows := len(p.stages())
		if pipelineRows < 1 {
			pipelineRows = 1
		}
		pipelineH = pipelineRows + 2 // borders
		if pipelineH > 10 {
			pipelineH = 10
		}
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

	// Sidebar gets a fixed 26 cols when there's enough room. Dropped
	// on truly narrow terminals (<90 cols) so the chat/events panes
	// don't get squeezed unreadably small. The previous 110-col
	// threshold meant the safe-default render (width=100, used before
	// the first WindowSizeMsg) hid the sidebar entirely — making it
	// look like the feature was broken at startup.
	sidebarW := 0
	if width >= 90 && !hideFiles {
		sidebarW = 26
	}
	rightW := width - sidebarW
	if rightW < 20 {
		// Pathological narrow case — a 10-col terminal still has to
		// produce *some* output. Force a minimum so lipgloss doesn't
		// see negative widths.
		rightW = 20
		sidebarW = 0
	}
	innerW := rightW - 2 // border consumes 2 cols on each box
	if innerW < 10 {
		innerW = 10
	}

	pipelineBox := ""
	if !hidePipeline {
		pipelineBox = bordStyle.Width(innerW).Render(
			titleStyle.Render(" Pipeline ") + "\n" +
				renderPipelinePane(p, innerW))
	}

	// Chat: render history into the available rows, then append a
	// "thinking…" footer row below the content (still inside the box)
	// when a turn is in flight. Footer-anchored so the user's eye
	// follows naturally from the latest message down to the indicator,
	// instead of jumping back to the title at the top.
	chatContentH := chatH - 3 // -3 = title row + 2 border rows
	thinkingRow := ""
	if turnActive {
		chatContentH -= 1 // reserve one row for the indicator
		mark := spinnerFrames[spinnerFrame%len(spinnerFrames)]
		// Verb cycles every ~3s. spinnerFrame ticks at 150ms, so divide
		// by 20 for a 3s rotation. Adds a sense of progress on long turns.
		verb := thinkingVerbs[(spinnerFrame/20)%len(thinkingVerbs)]
		thinkingRow = "\n" + runStyle.Render(
			fmt.Sprintf("  %s %s…  (Ctrl+C to cancel)", mark, verb))
	}
	chatPane, atTop, atBottom, totalLines, viewStart := renderChatPane(
		chat, renderer, chatContentH, innerW-2, chatScroll)
	// Chat title shows scroll state. When following, just " Chat ".
	// When scrolled, show the position so the user knows where they
	// are and how to get back to live.
	chatTitle := " Chat "
	if !atBottom {
		chatTitle = fmt.Sprintf(" Chat — ↑ %d/%d (PgDn / Ctrl+End to follow) ",
			viewStart, totalLines)
		if atTop {
			chatTitle = fmt.Sprintf(" Chat — top %d/%d (PgDn to scroll down) ",
				viewStart, totalLines)
		}
	}
	chatBox := bordStyle.Width(innerW).Render(
		titleStyle.Render(chatTitle) + "\n" +
			chatPane +
			thinkingRow)

	// Snapshot chat pane bounds so the highlight-to-copy mouse handler
	// can map screen Y → chat-line index. Chat box layout (top→bottom):
	//   row 0          top border
	//   row 1          title
	//   row 2..        chatPane content (chatContentH rows)
	//   row 2+content  thinking row (if turnActive) — also "in" the
	//                  chat pane visually but not part of allLines, so
	//                  excluded from selection range.
	//   row 2+content+(thinking?) bottom border
	// Right column starts at screen Y = headerH; pipelineBox precedes
	// chatBox if shown.
	pipelineRowOffset := 0
	if !hidePipeline {
		pipelineRowOffset = pipelineH
	}
	lastChatTopY = headerH + pipelineRowOffset + 2
	lastChatBottomY = lastChatTopY + chatContentH - 1
	lastChatLeftX = 1                       // 1 = inside the box's left border
	lastChatRightX = innerW - 2             // -2 = right border + safety

	eventsBox := ""
	if !hideEvents {
		eventsBox = bordStyle.Width(innerW).Render(
			titleStyle.Render(" Events ") + "\n" +
				renderEventsPane(events, eventsH-3, innerW))
	}

	statsLine := renderStatsPane(p, width, lastTurnTokens, totalTokens, maxTokens)

	// Mode-driven input box: red for bash, purple for slash, cyan for
	// the default chat mode. The title also flips so the user has two
	// signals (color + label) for the active mode.
	style := bordStyleFocused
	title := " Message "
	hint := ""
	switch inputMode {
	case "bash":
		style = bordStyleBash
		title = " Bash · executes on your machine "
		hint = renderBashHint(inputValue, innerW)
	case "slash":
		style = bordStyleSlash
		title = " Command "
		hint = renderSlashHint(inputValue, innerW)
	case "help":
		style = bordStyleSlash
		title = " ? · help "
		hint = renderHelpHint(innerW)
	}
	inputBox := style.Width(innerW).Render(
		titleStyle.Render(title) + "\n" + inputView)
	if hint != "" {
		inputBox = hint + "\n" + inputBox
	}

	// Build the right column out of only the panes that are still
	// enabled — JoinVertical is fine with an empty string but the
	// concise list reads better and keeps lipgloss off the empty rows.
	rightParts := []string{}
	if pipelineBox != "" {
		rightParts = append(rightParts, pipelineBox)
	}
	rightParts = append(rightParts, chatBox)
	if eventsBox != "" {
		rightParts = append(rightParts, eventsBox)
	}
	rightParts = append(rightParts, statsLine, inputBox)
	rightCol := lipgloss.JoinVertical(lipgloss.Left, rightParts...)

	// No sidebar on narrow terminals — return the right column at full
	// width with the header on top.
	if sidebarW == 0 {
		return lipgloss.JoinVertical(lipgloss.Left, header, rightCol), totalLines
	}

	// Sidebar runs the full body height (everything below the header).
	// Total box rows = bodyH (matches right column). Inside the box:
	// 1 title row + filesPane rows + 2 border rows = bodyH, so
	// filesPane gets bodyH - 3 rows of content.
	bodyH := height - headerH
	sidebarInnerW := sidebarW - 2
	filesContentH := bodyH - 3
	if filesContentH < 1 {
		filesContentH = 1
	}
	filesPane := renderFilesPane(files, modified, fileRoot,
		filesContentH, sidebarInnerW, fileScroll)
	filesBox := bordStyle.Width(sidebarInnerW).Height(bodyH - 2).Render(
		titleStyle.Render(" Files ") + "\n" + filesPane)

	// Sidebar on the right (after rightCol) per user preference —
	// keeps the chat anchored on the left where the eye starts.
	body := lipgloss.JoinHorizontal(lipgloss.Top, rightCol, filesBox)
	return lipgloss.JoinVertical(lipgloss.Left, header, body), totalLines
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
