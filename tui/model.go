// PC-062: Bubbletea model — owns Envelope channel + pipeline state +
// chat history + textarea + rendered view.
//
// Two SSE streams feed the model:
//   /events   → envelopeMsg → state.apply()  (always-on visibility)
//   /v1/agent → chatStreamMsg → chat history (per-turn, on Enter)

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
)

type envelopeMsg struct {
	ev Envelope
}

// tickMsg fires every second to refresh durations on running stages.
type tickMsg time.Time

// chatStreamMsg is one event from a /v1/agent SSE turn.
type chatStreamMsg struct {
	ev chatEvent
}

// chatTurnDoneMsg signals that the current /v1/agent turn finished
// (clean [DONE] or error). err is nil on clean completion.
type chatTurnDoneMsg struct {
	err error
}

type chatRole int

const (
	roleUser chatRole = iota
	roleAssistant
	roleTool
	roleSystem
)

// chatMessage is one row in the chat history.
type chatMessage struct {
	Role chatRole
	Body string
	// Meta — for tool: the tool name; for system: severity tag.
	Meta string
	// Success — only meaningful for tool rows. Drives the icon color.
	Success bool
}

type tuiModel struct {
	proxyURL string
	events   chan Envelope

	// Visible state
	width  int
	height int

	// Derived state — pipeline + counters from the event stream.
	state    pipelineState
	envelope []Envelope
	maxLines int

	// Chat
	input          textarea.Model
	chat           []chatMessage
	chatEvents     chan chatEvent
	turnActive     bool
	turnCancel     context.CancelFunc
	turnSessionID  string
	chatRenderer   *glamour.TermRenderer

	// Set when the user presses Ctrl+C mid-turn so the trailing flurry
	// of error/llm_call_end/__turn_done__ events render as "cancelled"
	// rather than misleading "FAIL"/"ERROR" rows. Cleared when the next
	// turn starts. The proxy's context-cancelled errors come through
	// the same SSE channel as real errors, so the flag is the only way
	// to distinguish "user aborted" from "real failure".
	userCancelled bool

	// Highlight-to-copy state. Press inside the chat pane sets selStart;
	// motion while held updates selEnd; release computes the line range
	// covered and pushes those lines to the clipboard via copyToClipboard
	// (OSC52 fallback works over SSH). Cell coords are screen-relative.
	// We only copy when there was a real drag (non-zero delta), so a
	// pure click doesn't trigger a copy.
	selecting          bool
	selPane            string // "chat" / "events" / "pipeline" / "files"
	selStartX, selStartY int
	selEndX, selEndY     int

	// Files added via /add — appended as a hint to each /v1/agent message.
	contextFiles map[string]bool

	// Working dir + permission mode for /v1/agent payloads.
	workingDir string
	mode       string

	// Polish state — spinner phase, last-sent message for Ctrl+R.
	spinnerFrame int
	lastUserMsg  string

	// Token accounting from llm_call_end events. lastTurnTokens is the
	// usage reported on the most recent LLM call (Qwen3.5 reports the
	// FULL prompt+completion total, not a delta — that's the value we
	// compare against maxContextTokens to gauge "how full is the
	// window"). totalTokensSession sums per-call deltas across the
	// whole session, used for the "tokens used overall" indicator.
	lastTurnTokens     int
	totalTokensSession int
	maxContextTokens   int

	// Per-LLM-call streaming state. While the model is decoding, every
	// llm_token event appends to streamingLLMText, and the trailing
	// "· llm ·" row is rewritten with header + tail so the user can
	// watch the JSON tool call come together token-by-token. Cleared
	// on llm_call_end.
	streamingLLM       bool
	streamingLLMText   string
	streamingLLMHeader string

	// Prompt-eval progress. While llama-server is encoding the prompt
	// (before the first decoded token arrives), the proxy polls /slots
	// every 250ms and emits llm_prompt_progress with processed/total/pct.
	// We render this as the body of the streaming row instead of a static
	// "encoding prompt…" line. Cleared on llm_first_token / llm_call_end.
	promptProcessed int
	promptTotal     int
	promptPct       float64
	// Set on llm_call_start, zeroed on llm_first_token / llm_call_end.
	// While non-zero the spinner ticker rewrites the streaming row at
	// 100ms cadence so the elapsed timer keeps moving even when the
	// proxy's progress poller is between emits (or /slots is silent).
	promptEvalStart time.Time

	// Same idea, but for V3's *internal* LLM calls (candidate gen,
	// scoring). Tracked separately so a v3_token doesn't overwrite the
	// agent loop's row and vice versa.
	streamingV3     bool
	streamingV3Text string

	// Chat scroll offset — number of rows scrolled UP from the bottom.
	// 0 means "follow the latest" (auto-scroll on new messages); >0
	// freezes the view at a position N rows above the latest. PgUp/PgDn
	// /mouse-wheel adjust; End jumps back to follow. lastChatTotal is
	// the line count from the most recent render (used to clamp scroll
	// at the top so PgUp/wheel-up stops growing once you hit the start
	// of history — without this, 100 PgUps requires 100 PgDns to undo).
	chatScroll     int
	eventsScroll   int
	pipelineScroll int
	lastChatTotal  int

	// Hide-pane toggles. Slash commands /hide files / pipeline / events
	// drop the corresponding pane; /show <name> brings it back.
	hideFiles    bool
	hidePipeline bool
	hideEvents   bool

	// Input mode derived from leading char of the textarea value.
	// "" / "bash" / "slash" — drives input-box border color and the
	// completion hint above the box.
	inputMode string

	// Spinner verb cycle — every ~3s the "thinking" word changes so
	// long generations don't feel static. Index advances based on
	// spinnerFrame ticks rather than a separate timer.
	thinkingVerbIdx int

	// Sidebar file tree — flat list of entries scanned from workingDir,
	// re-scanned every fileScanInterval and after every write/edit/
	// delete tool result. modifiedFiles is the set of relative paths
	// the agent has touched this session (highlighted in the sidebar).
	fileEntries    []fileEntry
	modifiedFiles  map[string]bool
	lastFileScan   time.Time
	fileScanScroll int

	// Toast notifications. Transient overlay messages that auto-decay
	// (e.g. "✓ copied 1234 chars"). Pruned every tick (100ms) by the
	// tickMsg handler. Rendered in View() as a banner spliced into the
	// header row — not a chat message, so it doesn't pollute history.
	toasts []toast

	// Lifecycle
	quitting bool
}

// toast is one transient notification. ExpiresAt is checked every tick
// against time.Now(); expired entries get dropped from m.toasts.
type toast struct {
	Body      string
	ExpiresAt time.Time
}

// showToast queues a transient overlay message that auto-dismisses
// after 2.5s. Used for "copied N chars from <pane>" style feedback —
// fire-and-forget UX hints that shouldn't pollute chat history.
func (m *tuiModel) showToast(body string) {
	m.toasts = append(m.toasts, toast{
		Body:      body,
		ExpiresAt: time.Now().Add(2500 * time.Millisecond),
	})
}

// scrollChat adjusts m.chatScroll by `delta` rows (positive = scroll
// up toward older messages, negative = scroll down). Clamps to
// [0, lastChatTotalRendered] so unbounded PgUp / wheel-up doesn't
// accumulate state that requires equal-and-opposite PgDns to clear.
func (m *tuiModel) scrollChat(delta int) {
	m.chatScroll += delta
	if max := lastChatTotalRendered; m.chatScroll > max {
		m.chatScroll = max
	}
	if m.chatScroll < 0 {
		m.chatScroll = 0
	}
}

// replaceV3LLMRow rewrites the most recent v3-llm row's body. Used by
// the v3_token / v3_llm_end handlers so a single row tracks the live
// stream instead of spawning a fresh chat row per token.
func (m *tuiModel) replaceV3LLMRow(body string) {
	for i := len(m.chat) - 1; i >= 0; i-- {
		if m.chat[i].Role == roleSystem && m.chat[i].Meta == "v3-llm" {
			m.chat[i].Body = body
			return
		}
	}
	m.chat = append(m.chat, chatMessage{
		Role: roleSystem, Meta: "v3-llm", Body: body,
	})
}

// replaceLLMRow rewrites the body of the most recent system/llm row.
// If no such row exists (shouldn't happen — llm_call_start always
// inserts one — but defensive), append a fresh one. Used by every
// llm_* event to keep one anchor row per LLM call rather than spawning
// a new chat row per token.
func (m *tuiModel) replaceLLMRow(body string) {
	for i := len(m.chat) - 1; i >= 0; i-- {
		if m.chat[i].Role == roleSystem && m.chat[i].Meta == "llm" {
			m.chat[i].Body = body
			return
		}
	}
	m.chat = append(m.chat, chatMessage{
		Role: roleSystem, Meta: "llm", Body: body,
	})
}

func newTUIModel(proxyURL string) tuiModel {
	ta := textarea.New()
	ta.Placeholder = "Type a message · ! for bash · / for command · ? for help"
	// No per-line prompt — bubbles renders Prompt on EVERY soft-wrapped
	// line, which made multi-line input look noisy ("> > > >"). The
	// mode indicator lives in the input box's border color now.
	ta.Prompt = ""
	// Same reason we drop line numbers: bubbles defaults ShowLineNumbers
	// to true, so a one-liner shows a stray "1" gutter that confuses
	// users into thinking the input is a code editor.
	ta.ShowLineNumbers = false
	ta.CharLimit = 8000
	ta.SetWidth(80)
	ta.SetHeight(3)
	ta.Focus()

	wd, _ := os.Getwd()

	// Glamour renderer for assistant markdown. We avoid WithAutoStyle()
	// here: it sends an OSC 11 background-color query to the terminal,
	// and that query's response (e.g. `\e]11;rgb:...\e\\`) can leak
	// into the user's view as visible "0x1b ]11;..." escape garbage if
	// the terminal responds before Bubbletea's input parser is fully
	// attached — exactly the symptom reported at startup. Standard
	// "dark" works for the common case (dark terminals); users who want
	// a different style can set $GLAMOUR_STYLE before launch.
	style := os.Getenv("GLAMOUR_STYLE")
	if style == "" {
		style = "dark"
	}
	// Initial wrap is conservative — gets rebuilt on the first
	// WindowSizeMsg with the actual chat width (terminal width minus
	// sidebar minus border overhead). Anything wider than the chat box
	// causes lipgloss to expand the box, hiding the sidebar.
	renderer, _ := glamour.NewTermRenderer(
		glamour.WithStandardStyle(style),
		glamour.WithWordWrap(60),
	)

	return tuiModel{
		proxyURL:         proxyURL,
		events:           make(chan Envelope, 256),
		state:            newPipelineState(),
		maxLines:         1000,
		input:            ta,
		chatEvents:       make(chan chatEvent, 64),
		chatRenderer:     renderer,
		workingDir:       wd,
		mode:             "default",
		maxContextTokens: 32768, // Qwen3.5-9B context size; matches llama-server config
		// File scan is dispatched async from Init() — see scanFilesCmd.
		// Doing it synchronously here blocked tea.NewProgram from
		// entering its event loop, during which the user's keystrokes
		// hit the bare TTY (not the TUI), and the terminal's startup
		// capability-query responses leaked through as visible
		// escape sequences (the "0x1b ]]" the user reported).
		fileEntries:   nil,
		modifiedFiles: map[string]bool{},
		lastFileScan:  time.Time{},
	}
}

// scanFilesMsg carries the result of an async file scan back to the
// model's Update loop. Triggered initially from Init() and again
// after every write/edit/delete tool result + on the slow tick.
type scanFilesMsg struct {
	entries []fileEntry
	at      time.Time
}

func scanFilesCmd(root string) tea.Cmd {
	return func() tea.Msg {
		return scanFilesMsg{
			entries: scanFiles(root, 2, 500),
			at:      time.Now(),
		}
	}
}

func (m tuiModel) Init() tea.Cmd {
	return tea.Batch(
		waitForEnvelope(m.events),
		waitForChatEvent(m.chatEvents),
		tickEvery(100*time.Millisecond),
		textarea.Blink,
		// Run the initial file-tree scan off the main thread so it
		// doesn't block the event loop. The empty sidebar shows for
		// the ~10–50ms it takes scanFiles to complete on a typical
		// project; results arrive via scanFilesMsg.
		scanFilesCmd(m.workingDir),
		// Ask Bubbletea to send a WindowSizeMsg right away. Some
		// terminals/multiplexers (tmux, screen) delay or skip the
		// initial resize event, leaving us rendering with safe
		// defaults (width=100) longer than necessary — which hides
		// the sidebar (threshold 90) and looks broken at startup.
		tea.WindowSize(),
	)
}

func waitForEnvelope(ch <-chan Envelope) tea.Cmd {
	return func() tea.Msg {
		ev, ok := <-ch
		if !ok {
			return nil
		}
		return envelopeMsg{ev: ev}
	}
}

func waitForChatEvent(ch <-chan chatEvent) tea.Cmd {
	return func() tea.Msg {
		ev, ok := <-ch
		if !ok {
			return nil
		}
		return chatStreamMsg{ev: ev}
	}
}

func tickEvery(d time.Duration) tea.Cmd {
	return tea.Tick(d, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

// buildChatHistory packs prior user/assistant text rows from m.chat
// into the wire shape /v1/agent expects. Excludes:
//   - the most recent roleUser entry (that's the message being sent
//     this turn — handleAgent pairs it with PriorHistory, so sending
//     it twice would duplicate)
//   - tool / system rows (within-turn machinery, not conversation)
//   - empty bodies
//
// Cap at the last 40 rows; the proxy trims further if needed. Returns
// nil when there's no prior history (first turn of a session) so the
// JSON payload omits the field entirely.
func (m *tuiModel) buildChatHistory() []historyMessage {
	if len(m.chat) == 0 {
		return nil
	}
	// Locate the last user row — that's the just-appended new message.
	lastUserIdx := -1
	for i := len(m.chat) - 1; i >= 0; i-- {
		if m.chat[i].Role == roleUser {
			lastUserIdx = i
			break
		}
	}

	out := make([]historyMessage, 0, len(m.chat))
	for i, row := range m.chat {
		if i == lastUserIdx {
			continue
		}
		if row.Body == "" {
			continue
		}
		var role, content string
		switch row.Role {
		case roleUser:
			role = "user"
			content = row.Body
		case roleAssistant:
			role = "assistant"
			// CRITICAL: wrap the assistant's prior text in the JSON
			// envelope shape the model is supposed to emit. m.chat
			// stores only the extracted .content, but the LLM saw a
			// full {"type":"text","content":"..."} when it generated
			// this turn. Sending raw text here teaches the model the
			// format is plain text — next turn it emits raw text and
			// the proxy parse fails. Re-wrap to keep the format
			// signal consistent across turns.
			env, err := json.Marshal(map[string]string{
				"type":    "text",
				"content": row.Body,
			})
			if err != nil {
				continue
			}
			content = string(env)
		default:
			continue // tool / system rows: skip
		}
		out = append(out, historyMessage{Role: role, Content: content})
	}
	if len(out) == 0 {
		return nil
	}
	if len(out) > 40 {
		out = out[len(out)-40:]
	}
	return out
}

// sendChatCmd kicks off a /v1/agent turn. Runs sendChat in a goroutine
// because Bubbletea Cmds should be quick — the goroutine pumps events
// onto m.chatEvents which the model drains via waitForChatEvent.
func (m *tuiModel) sendChatCmd(message string) tea.Cmd {
	ctx, cancel := context.WithCancel(context.Background())
	sessionID := newSessionID()
	m.turnCancel = cancel
	m.turnSessionID = sessionID
	m.turnActive = true
	m.userCancelled = false // fresh turn — clear the cancel sticky flag

	proxyURL := m.proxyURL
	workingDir := m.workingDir
	mode := m.mode
	out := m.chatEvents
	history := m.buildChatHistory()

	return func() tea.Msg {
		go func() {
			err := sendChat(ctx, proxyURL, message, workingDir, mode, sessionID, history, out)
			// Signal turn end via the same channel using a sentinel
			// chatEvent (type="__turn_done__") — keeps the event
			// ordering: all messages drain before the done marker.
			payload, _ := json.Marshal(map[string]string{
				"err": errString(err),
			})
			out <- chatEvent{Type: "__turn_done__", Data: payload}
		}()
		return nil
	}
}

// newSessionID returns a fresh hex token for tagging an /v1/agent turn
// so /cancel can target it. Cryptographic randomness is overkill but
// trivially cheap and avoids any chance of collision across concurrent
// TUI sessions hitting the same proxy.
func newSessionID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func (m tuiModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			if m.turnActive && m.turnCancel != nil {
				// First Ctrl+C cancels the in-flight turn; second quits.
				// Belt-and-suspenders: cancel locally (closes TCP) AND
				// POST /cancel so the proxy aborts even when buffered.
				m.userCancelled = true
				m.turnCancel()
				sid := m.turnSessionID
				proxyURL := m.proxyURL
				m.turnActive = false
				m.chat = append(m.chat, chatMessage{
					Role: roleSystem, Meta: "cancelled",
					Body: "turn cancelled",
				})
				return m, func() tea.Msg {
					_ = cancelTurn(proxyURL, sid)
					return nil
				}
			}
			m.quitting = true
			return m, tea.Quit
		case "ctrl+d":
			m.quitting = true
			return m, tea.Quit

		case "ctrl+l":
			m.chat = nil
			return m, nil

		case "ctrl+t":
			// Cycle permission mode. Visible in header.
			switch m.mode {
			case "default":
				m.mode = "accept-edits"
			case "accept-edits":
				m.mode = "yolo"
			default:
				m.mode = "default"
			}
			m.chat = append(m.chat, chatMessage{
				Role: roleSystem, Meta: "mode",
				Body: fmt.Sprintf("mode → %s", m.mode),
			})
			return m, nil

		case "ctrl+r":
			if !m.turnActive && m.lastUserMsg != "" {
				m.chat = append(m.chat, chatMessage{
					Role: roleUser, Body: m.lastUserMsg,
				})
				return m, m.sendChatCmd(m.lastUserMsg + m.contextSuffix())
			}
			return m, nil

		case "pgup":
			m.scrollChat(10)
			return m, nil
		case "pgdown":
			m.scrollChat(-10)
			return m, nil
		case "ctrl+home":
			m.scrollChat(1 << 30) // clamped to lastChatTotal
			return m, nil
		case "ctrl+end":
			m.chatScroll = 0
			return m, nil
		case "enter":
			// Enter sends; Shift+Enter (or Alt+Enter) inserts newline.
			// textarea handles Shift+Enter as KeyShiftEnter ("shift+enter").
			if !m.turnActive {
				text := strings.TrimSpace(m.input.Value())
				if text == "" {
					return m, nil
				}
				m.input.Reset()
				dlog("user", "input", map[string]interface{}{"text": text})
				// Bash mode: leading "!" runs as a shell command in the
				// working dir, output appears as a system row. Same path
				// as /run but with the conversational shorthand devs
				// expect from Claude Code.
				if strings.HasPrefix(text, "!") {
					cmdStr := strings.TrimSpace(text[1:])
					if cmdStr == "" {
						m.chat = append(m.chat, chatMessage{
							Role: roleSystem, Meta: "error",
							Body: "Bash mode: type ! followed by a command.",
						})
						return m, nil
					}
					m.chat = append(m.chat, chatMessage{
						Role: roleUser, Body: "! " + cmdStr,
					})
					return m, runShellCmd(m.workingDir, "!"+cmdStr,
						[]string{"bash", "-lc", cmdStr})
				}
				// "?" alone (or with trailing whitespace) is a shorthand
				// for /help — same convention as Claude Code so users
				// don't have to remember the slash form.
				if text == "?" {
					text = "/help"
				}
				// Slash commands intercepted before agent send.
				if consumed, slashCmd, quit := m.handleSlash(text); consumed {
					dlog("slash", "dispatched", map[string]interface{}{
						"input": text, "quit": quit,
					})
					if quit {
						m.quitting = true
					}
					if slashCmd != nil {
						cmds = append(cmds, slashCmd)
					}
					return m, tea.Batch(cmds...)
				}
				// Plain message → send to agent. Append context-files
				// hint so the agent knows the user's chosen scope.
				m.chat = append(m.chat, chatMessage{
					Role: roleUser, Body: text,
				})
				m.lastUserMsg = text
				dlog("turn", "started", map[string]interface{}{
					"session_id": "(set in sendChatCmd)",
					"len":        len(text),
				})
				cmds = append(cmds, m.sendChatCmd(text+m.contextSuffix()))
				return m, tea.Batch(cmds...)
			}
		}

	case tea.MouseMsg:
		// Wheel routes to whichever pane the cursor is over so events,
		// pipeline, files all scroll independently — not just chat.
		if msg.Action == tea.MouseActionPress {
			switch msg.Button {
			case tea.MouseButtonWheelUp:
				m.scrollPaneAt(msg.X, msg.Y, 3)
				return m, nil
			case tea.MouseButtonWheelDown:
				m.scrollPaneAt(msg.X, msg.Y, -3)
				return m, nil
			}
		}
		// Highlight-to-copy in any pane. Press finds the pane under
		// (X,Y); motion updates the end; release extracts text from
		// that pane's snapshot and copies via OSC52 / CLI tool.
		switch msg.Action {
		case tea.MouseActionPress:
			if msg.Button == tea.MouseButtonLeft {
				if pane := findPane(msg.X, msg.Y); pane != nil {
					m.selecting = true
					m.selPane = pane.name
					m.selStartX, m.selStartY = msg.X, msg.Y
					m.selEndX, m.selEndY = msg.X, msg.Y
					dlog("mouse", "press", map[string]interface{}{
						"x": msg.X, "y": msg.Y, "pane": pane.name,
						"paneTopY": pane.topY, "paneBottomY": pane.bottomY,
						"viewStart": pane.viewStart, "lines": len(pane.lines),
					})
				}
			}
		case tea.MouseActionMotion:
			if m.selecting {
				m.selEndX, m.selEndY = msg.X, msg.Y
			}
		case tea.MouseActionRelease:
			if m.selecting {
				selPane := m.selPane
				m.selecting = false
				dy := m.selEndY - m.selStartY
				if dy < 0 {
					dy = -dy
				}
				dx := m.selEndX - m.selStartX
				if dx < 0 {
					dx = -dx
				}
				// Pure click (no drag) → no copy.
				if dy == 0 && dx < 2 {
					return m, nil
				}
				text := extractPaneSelection(selPane,
					m.selStartY, m.selEndY,
					m.selStartX, m.selEndX)
				dlog("mouse", "release", map[string]interface{}{
					"pane":     selPane,
					"startX":   m.selStartX, "startY": m.selStartY,
					"endX":     m.selEndX, "endY": m.selEndY,
					"text_len": len(text),
					"preview":  truncate(text, 60),
				})
				if text == "" {
					m.showToast("nothing to copy")
					return m, nil
				}
				if err := copyToClipboard(text); err != nil {
					m.showToast(fmt.Sprintf("copy failed: %v", err))
					return m, nil
				}
				m.showToast(fmt.Sprintf("✓ copied %d chars from %s pane",
					len(text), selPane))
				return m, nil
			}
		}
		return m, nil

	case tea.WindowSizeMsg:
		// Drag-resizing modern terminals fires WindowSizeMsg dozens of
		// times in quick succession. Glamour init isn't free (it loads
		// styles + builds a renderer); doing it on every event was
		// queueing slow Updates behind a flood of resize messages.
		// Skip the rebuild when only the height changed, and skip
		// duplicate-width events entirely.
		widthChanged := msg.Width != m.width
		if msg.Width == m.width && msg.Height == m.height {
			return m, nil
		}
		m.width = msg.Width
		m.height = msg.Height
		m.input.SetWidth(max(20, msg.Width-2))
		if widthChanged {
			style := os.Getenv("GLAMOUR_STYLE")
			if style == "" {
				style = "dark"
			}
			// Glamour wrap MUST match the chat box's content width or
			// lipgloss expands the box past where the sidebar sits.
			// Mirror panes.go's layout: sidebar 26 cols when W>=90,
			// chat box border (2) + indent (2) on either side.
			wrap := msg.Width - 6
			if msg.Width >= 90 {
				wrap = msg.Width - 26 - 6
			}
			if wrap < 20 {
				wrap = 20
			}
			if wrap > 100 {
				wrap = 100 // cap for readability — long lines hurt scanning
			}
			if r, err := glamour.NewTermRenderer(
				glamour.WithStandardStyle(style),
				glamour.WithWordWrap(wrap),
			); err == nil {
				m.chatRenderer = r
			}
		}
		// Force a full repaint of the alt-screen so leftover content
		// from the prior size doesn't bleed through. Without this,
		// shrinking the terminal can leave stale rows on screen and
		// growing it can leave the new edges blank until the next
		// natural redraw.
		return m, tea.ClearScreen

	case envelopeMsg:
		// While the user has cancelled, drop the trailing flurry of
		// cancellation-shaped envelopes (LLM error, stage_end with
		// success=false) so the events pane and pipeline pane don't
		// surface a misleading FAIL/ERROR row. The chat already shows
		// "turn cancelled" — that's the single user-visible signal.
		if m.userCancelled && envelopeLooksCancelled(msg.ev) {
			dlog("event", "suppressed_cancel", map[string]interface{}{
				"type": msg.ev.Type, "stage": msg.ev.Stage,
			})
			return m, waitForEnvelope(m.events)
		}
		m.state.apply(msg.ev)
		m.envelope = append(m.envelope, msg.ev)
		if len(m.envelope) > m.maxLines {
			m.envelope = m.envelope[len(m.envelope)-m.maxLines:]
		}
		dlog("event", msg.ev.Type, map[string]interface{}{
			"stage": msg.ev.Stage, "payload": msg.ev.Payload,
		})
		return m, waitForEnvelope(m.events)

	case chatStreamMsg:
		if msg.ev.Type == "__turn_done__" {
			m.turnActive = false
			var p struct {
				Err string `json:"err"`
			}
			_ = json.Unmarshal(msg.ev.Data, &p)
			if p.Err != "" && !m.userCancelled && !looksCancelled(p.Err) {
				m.chat = append(m.chat, chatMessage{
					Role: roleSystem, Meta: "error",
					Body: p.Err,
				})
			}
			dlog("turn", "ended", map[string]interface{}{"err": p.Err})
		} else {
			// Skip dlog for llm_token — at ~30 tok/s a long generation
			// produces thousands of entries and crowds out actually
			// interesting events when reading the file.
			if msg.ev.Type != "llm_token" {
				dlog("chat", msg.ev.Type, map[string]interface{}{
					"data": json.RawMessage(msg.ev.Data),
				})
			}
			m.appendChatEvent(msg.ev)
		}
		return m, waitForChatEvent(m.chatEvents)

	case slashResultMsg:
		body := msg.output
		if msg.err != nil {
			if body == "" {
				body = msg.err.Error()
			} else {
				body = body + "\n[error: " + msg.err.Error() + "]"
			}
		}
		if body == "" {
			body = "(no output)"
		}
		role := roleSystem
		if msg.err != nil {
			role = roleSystem
		}
		m.chat = append(m.chat, chatMessage{
			Role: role, Meta: msg.command, Body: body,
			Success: msg.err == nil,
		})
		dlog("slash", "result", map[string]interface{}{
			"command": msg.command, "ok": msg.err == nil,
			"output_len": len(msg.output),
		})
		return m, nil

	case tickMsg:
		m.spinnerFrame++
		// Prune expired toasts so the overlay disappears on its own.
		if len(m.toasts) > 0 {
			now := time.Now()
			kept := m.toasts[:0]
			for _, t := range m.toasts {
				if t.ExpiresAt.After(now) {
					kept = append(kept, t)
				}
			}
			m.toasts = kept
		}
		// Tick-driven prompt-progress row update: while we're still in
		// prompt eval (no first token yet), rewrite the streaming row
		// every tick so the elapsed timer ticks smoothly even when the
		// proxy's poller is silent (e.g. between /slots probes, or when
		// /slots returns no token counters at all on this build).
		if !m.promptEvalStart.IsZero() && m.streamingLLM {
			elapsed := time.Since(m.promptEvalStart).Milliseconds()
			m.replaceLLMRow(formatPromptProgress(
				m.promptProcessed, m.promptTotal, m.promptPct, elapsed))
		}
		// Rescan files periodically so external changes (agent wrote
		// a file via /workspace, user added a file in another shell)
		// show up in the sidebar without a manual refresh. Dispatch
		// async so a slow disk doesn't stall the spinner.
		var refresh tea.Cmd
		if time.Since(m.lastFileScan) > 4*time.Second {
			m.lastFileScan = time.Now() // mark to debounce overlapping scans
			refresh = scanFilesCmd(m.workingDir)
		}
		return m, tea.Batch(tickEvery(100*time.Millisecond), refresh)

	case scanFilesMsg:
		// Result of an async scanFiles run. Apply only if newer than
		// what we have, so an old/slow scan doesn't overwrite a more
		// recent one.
		if msg.at.After(m.lastFileScan) || m.lastFileScan.IsZero() {
			m.fileEntries = msg.entries
			m.lastFileScan = msg.at
		}
		return m, nil
	}

	// Forward remaining keystrokes to the textarea (typing, arrows…).
	if !m.quitting {
		var taCmd tea.Cmd
		m.input, taCmd = m.input.Update(msg)
		cmds = append(cmds, taCmd)
		// Track input mode so the input-box border colors itself
		// (red=bash, purple=slash, default=cyan) and a completion
		// hint above the box can list matching commands.
		val := m.input.Value()
		switch {
		case strings.HasPrefix(val, "!"):
			m.inputMode = "bash"
		case strings.HasPrefix(val, "/"):
			m.inputMode = "slash"
		case strings.HasPrefix(val, "?"):
			m.inputMode = "help"
		default:
			m.inputMode = ""
		}
	}
	return m, tea.Batch(cmds...)
}

// appendChatEvent translates a /v1/agent SSE event into one or more
// chat history rows.
func (m *tuiModel) appendChatEvent(ev chatEvent) {
	switch ev.Type {
	case "turn_start":
		// Visual separator + turn counter. Compact one-liner so a long
		// task's chat doesn't drown in headers — but enough that the
		// user can see "where am I, what turn just started".
		var p struct {
			Turn     int  `json:"turn"`
			Messages int  `json:"messages"`
			Trimmed  bool `json:"trimmed"`
		}
		_ = json.Unmarshal(ev.Data, &p)
		body := fmt.Sprintf("turn %d  ·  ctx=%d msgs", p.Turn+1, p.Messages)
		if p.Trimmed {
			body += "  (trimmed)"
		}
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "turn", Body: body,
		})

	case "llm_call_start":
		// Marker: prompt is being encoded by llama-server. No tokens yet —
		// time-to-first-token reflects prompt eval duration. The body is
		// rewritten on llm_prompt_progress (live %), llm_first_token
		// (decoding starts), and llm_call_end (totals).
		var p struct {
			PromptTokens int `json:"prompt_tokens"`
		}
		_ = json.Unmarshal(ev.Data, &p)
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "llm",
			Body: "encoding prompt…",
		})
		m.streamingLLM = true
		m.streamingLLMText = ""
		m.promptProcessed = 0
		m.promptTotal = p.PromptTokens
		m.promptPct = 0
		m.promptEvalStart = time.Now()
		// Pre-fill the context gauge with the prompt-token estimate so
		// the user sees ctx fill up the moment the call starts, not
		// only on llm_call_end. Each llm_token below increments this
		// further; llm_call_end replaces with the authoritative count.
		if p.PromptTokens > 0 {
			m.lastTurnTokens = p.PromptTokens
		}

	case "llm_prompt_progress":
		// Live prompt-eval progress from the proxy's poller. ElapsedMS
		// is always set; processed/total/pct are present only when
		// llama-server's /slots endpoint exposes them. We render a bar
		// when we have %, a spinner+timer otherwise.
		var p struct {
			Processed int     `json:"processed"`
			Total     int     `json:"total"`
			Pct       float64 `json:"pct"`
			ElapsedMS int64   `json:"elapsed_ms"`
		}
		if json.Unmarshal(ev.Data, &p) == nil {
			m.promptProcessed = p.Processed
			m.promptTotal = p.Total
			m.promptPct = p.Pct
			// Live ctx gauge during prompt eval: if /slots gives us
			// processed-tokens, push that into lastTurnTokens so the
			// header context indicator fills as the prompt is encoded
			// (instead of jumping at llm_call_end). On builds where
			// /slots is silent this is a no-op — we still show the
			// chars/4 estimate from llm_call_start.
			if p.Processed > m.lastTurnTokens {
				m.lastTurnTokens = p.Processed
			}
			m.replaceLLMRow(formatPromptProgress(p.Processed, p.Total, p.Pct, p.ElapsedMS))
		}

	case "llm_first_token":
		// Prompt eval finished — decoding has started. Show the prompt
		// duration so the user can see "where the dead air went". The
		// body is rebuilt below as tokens stream in.
		var p struct {
			PromptMS int64 `json:"prompt_ms"`
		}
		_ = json.Unmarshal(ev.Data, &p)
		m.promptEvalStart = time.Time{} // stop tick-rewrite of the row
		secs := float64(p.PromptMS) / 1000.0
		header := fmt.Sprintf("decoding…  (prompt eval: %.1fs)", secs)
		m.streamingLLMHeader = header
		m.replaceLLMRow(header)

	case "llm_token":
		// One delta from the LLM stream. Append to the streaming buffer
		// and re-render the trailing llm row with header + tail of the
		// stream so the user sees the JSON come together token-by-token.
		// The rendered row is dim grey ("machine internals" style) —
		// the polished tool_call/text events below are the bright
		// "outputs from the machine".
		var p struct {
			Text string `json:"text"`
		}
		if json.Unmarshal(ev.Data, &p) == nil && p.Text != "" {
			m.streamingLLMText += p.Text
			body := m.streamingLLMHeader + "\n" +
				formatStreamingLLM(m.streamingLLMText)
			m.replaceLLMRow(body)
			// Live context-utilization update: each llm_token delta is
			// roughly 1 model token, so increment the gauge per event.
			// Authoritative count replaces this on llm_call_end.
			m.lastTurnTokens++
		}

	case "llm_call_end":
		// Replace the streaming row with totals so the scrollback shows
		// a compact "model replied · 8421 tok · 12.3s" instead of the
		// raw token tail. The actual tool_call / text output rows that
		// follow are the bright "outputs from the machine"; this row is
		// the dim "internals" summary.
		var p struct {
			Turn        int    `json:"turn"`
			Tokens      int    `json:"tokens"`
			TotalTokens int    `json:"total_tokens"`
			MS          int64  `json:"ms"`
			Chars       int    `json:"chars"`
			Error       string `json:"error"`
		}
		_ = json.Unmarshal(ev.Data, &p)
		secs := float64(p.MS) / 1000.0
		var body string
		switch {
		case p.Error != "" && (m.userCancelled || looksCancelled(p.Error)):
			body = fmt.Sprintf("model call cancelled after %.1fs", secs)
		case p.Error != "":
			body = fmt.Sprintf("model failed in %.1fs — %s", secs, p.Error)
		default:
			body = fmt.Sprintf("model replied · %d tok · %d chars · %.1fs",
				p.Tokens, p.Chars, secs)
		}
		m.replaceLLMRow(body)
		m.streamingLLM = false
		m.streamingLLMText = ""
		m.streamingLLMHeader = ""
		// Track tokens for the stats line. Qwen3.5's usage.total_tokens
		// is "prompt + completion of *this* call", which is the right
		// value for "context window utilization". The session-wide sum
		// comes from the proxy's running ctx.TotalTokens (==accumulated
		// per-call totals).
		m.lastTurnTokens = p.Tokens
		m.totalTokensSession = p.TotalTokens

	case "text":
		var p struct {
			Content string `json:"content"`
		}
		if json.Unmarshal(ev.Data, &p) == nil && p.Content != "" {
			m.chat = append(m.chat, chatMessage{
				Role: roleAssistant, Body: p.Content,
			})
		}

	case "tool_call":
		var p struct {
			Name string          `json:"name"`
			Args json.RawMessage `json:"args"`
			Turn int             `json:"turn"`
		}
		if json.Unmarshal(ev.Data, &p) == nil {
			m.chat = append(m.chat, chatMessage{
				Role: roleTool, Meta: "→ " + p.Name,
				Body: summarizeToolArgs(p.Name, p.Args),
			})
			// Highlight files touched by write/edit/delete in the
			// sidebar. The path is normalized to the same form
			// scanFiles produces (relative to workingDir) so the map
			// lookup hits in renderFilesPane. The actual rescan
			// happens on the next tick — fast enough that the new
			// file appears within a few hundred ms, but doesn't block
			// the event handler.
			switch p.Name {
			case "write_file", "edit_file", "delete_file":
				if path := extractWritePath(p.Args); path != "" {
					if m.modifiedFiles == nil {
						m.modifiedFiles = map[string]bool{}
					}
					m.modifiedFiles[path] = true
					// Force-expire the debounce so the next tick scans.
					m.lastFileScan = time.Time{}
				}
			}
		}

	case "tool_result":
		var p struct {
			Tool    string          `json:"tool"`
			Success bool            `json:"success"`
			Data    json.RawMessage `json:"data"`
			Error   string          `json:"error"`
			Elapsed string          `json:"elapsed"`
		}
		if json.Unmarshal(ev.Data, &p) == nil {
			body := p.Error
			if p.Success {
				body = summarizeToolResult(p.Tool, p.Data)
			}
			if p.Elapsed != "" {
				if body == "" {
					body = p.Elapsed
				} else {
					body = fmt.Sprintf("%s  ·  %s", body, p.Elapsed)
				}
			}
			m.chat = append(m.chat, chatMessage{
				Role: roleTool, Meta: "← " + p.Tool,
				Success: p.Success, Body: body,
			})
		}

	case "permission_request":
		var p struct {
			ToolName string `json:"tool_name"`
		}
		_ = json.Unmarshal(ev.Data, &p)
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "permission",
			Body: fmt.Sprintf("permission requested for %s (auto-allow in default mode for read tools)", p.ToolName),
		})

	case "permission_denied":
		var p struct {
			Tool string `json:"tool"`
		}
		_ = json.Unmarshal(ev.Data, &p)
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "denied",
			Body: fmt.Sprintf("permission denied for %s", p.Tool),
		})

	case "error":
		var p struct {
			Error string `json:"error"`
		}
		_ = json.Unmarshal(ev.Data, &p)
		// Suppress error rows that are really just cancellation echoes
		// (proxy still emits them when ctx.Ctx is cancelled). The user
		// already saw the "turn cancelled" row when they hit Ctrl+C.
		if m.userCancelled || looksCancelled(p.Error) {
			return
		}
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "error", Body: p.Error,
		})

	case "done":
		var p struct {
			Summary string `json:"summary"`
		}
		_ = json.Unmarshal(ev.Data, &p)
		if p.Summary != "" {
			m.chat = append(m.chat, chatMessage{
				Role: roleSystem, Meta: "done", Body: p.Summary,
			})
		}

	case "v3_llm_start":
		// V3 is starting an LLM call. Insert a dim "v3-llm" row that
		// the v3_token handler will fill in. Mirrors the agent's
		// llm_call_start row, but with a "V3" tag so the user can
		// tell V3-internal calls from agent-loop calls at a glance.
		var p struct {
			Detail string `json:"detail"`
		}
		_ = json.Unmarshal(ev.Data, &p)
		body := "calling model…"
		if p.Detail != "" {
			body = p.Detail + " · calling model…"
		}
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "v3-llm", Body: body,
		})
		m.streamingV3 = true
		m.streamingV3Text = ""

	case "v3_token":
		// Per-token delta from V3's streaming LLM call. Append to the
		// active v3-llm row (updated in place so we don't spawn
		// thousands of chat rows during a long candidate generation).
		var p struct {
			Text string `json:"text"`
		}
		if json.Unmarshal(ev.Data, &p) == nil && p.Text != "" {
			m.streamingV3Text += p.Text
			body := "decoding…\n" + formatStreamingLLM(m.streamingV3Text)
			m.replaceV3LLMRow(body)
		}

	case "v3_llm_end":
		// V3's LLM call finished. Replace the streaming row with the
		// summary detail ("1234 tok · 12345ms") so scrollback shows a
		// compact line, not the raw token tail.
		var p struct {
			Detail string `json:"detail"`
		}
		_ = json.Unmarshal(ev.Data, &p)
		body := "model replied"
		if p.Detail != "" {
			body = "model replied · " + p.Detail
		}
		m.replaceV3LLMRow(body)
		m.streamingV3 = false
		m.streamingV3Text = ""

	case "v3_progress":
		// V3 pipeline narration emitted by proxy/tools.go via
		// ctx.StreamFn("v3_progress", {message: "..."}). One row per
		// stage (e.g. "[probe] Generating probe candidate..."). These
		// were silently dropped in the first cut — without this case
		// the user sees a frozen chat pane during a 1-2 minute V3 run.
		var p struct {
			Message string `json:"message"`
		}
		if json.Unmarshal(ev.Data, &p) == nil && p.Message != "" {
			// Trim the leading box-drawing prefix the proxy adds for
			// legacy pretty-print; the TUI styles its own rows.
			msg := strings.TrimLeft(p.Message, " │└├")
			msg = strings.TrimSpace(msg)
			m.chat = append(m.chat, chatMessage{
				Role: roleSystem, Meta: "V3", Body: msg,
			})
		}

	// V3 typed observability events. Each carries a structured `data`
	// payload from the V3 service (counts, indices, timings, strategy
	// names) on top of the human-readable `detail` string. We render
	// each as a dedicated row in the chat with the stage tag bolded.
	// The pipeline pane reads the same events to drive its progress
	// rows. Added 2026-05.
	case "v3_phase", "v3_plansearch", "v3_divsampling", "v3_sandbox",
		"v3_select", "v3_repair", "v3_probe", "v3_self_test":
		body := formatV3StageEvent(ev.Type, ev.Data)
		if body != "" {
			m.chat = append(m.chat, chatMessage{
				Role: roleSystem, Meta: "V3", Body: body,
			})
		}
	}
}

// formatV3StageEvent renders a structured V3 stage event as a single
// chat-row body. We extract the most useful 1–3 fields and append them
// to the human-readable detail. Keeps the line short — the pipeline
// pane is the place to show timelines and counters in detail.
func formatV3StageEvent(eventType string, data json.RawMessage) string {
	var p struct {
		Stage     string  `json:"stage"`
		Detail    string  `json:"detail"`
		Index     int     `json:"index"`
		ElapsedMS int     `json:"elapsed_ms"`
		Energy    float64 `json:"energy"`
		Passed    int     `json:"passed"`
		Total     int     `json:"total"`
		K         int     `json:"k"`
		Plans     int     `json:"plans"`
		Slots     int     `json:"slots"`
		Tier      string  `json:"tier"`
		Strategy  string  `json:"strategy"`
		Iterations int    `json:"iterations"`
		Tokens    int     `json:"tokens"`
		Failing   int     `json:"failing"`
	}
	_ = json.Unmarshal(data, &p)
	if p.Detail == "" && p.Stage == "" {
		return ""
	}
	tag := strings.TrimPrefix(eventType, "v3_")
	body := tag
	if p.Stage != "" && p.Stage != tag {
		body += "·" + p.Stage
	}
	body += " — " + p.Detail
	// Append the most informative structured field for this event.
	switch p.Stage {
	case "sandbox_pass", "sandbox_fail":
		if p.ElapsedMS > 0 {
			body += fmt.Sprintf(" · %dms", p.ElapsedMS)
		}
	case "sandbox_done":
		if p.Total > 0 {
			body += fmt.Sprintf(" · %d/%d", p.Passed, p.Total)
		}
	case "phase2_allocated":
		if p.K > 0 {
			body += fmt.Sprintf(" · k=%d tier=%s", p.K, p.Tier)
		}
	case "plansearch_done":
		if p.Tokens > 0 {
			body += fmt.Sprintf(" · %d tok", p.Tokens)
		}
	case "refinement_pass":
		if p.Iterations > 0 {
			body += fmt.Sprintf(" · %d iter · %d tok", p.Iterations, p.Tokens)
		}
	case "s_star_winner", "selected":
		if p.Energy > 0 {
			body += fmt.Sprintf(" · E=%.2f", p.Energy)
		}
	}
	return body
}

// extractPaneSelection returns the plain text of `paneName`'s lines
// covered by a drag from (startX, startY) to (endX, endY) in screen
// coordinates. Looks up the named pane in paneSnaps (populated by
// the most recent View()).
//
// Behavior:
//   - Vertical selection is line-granular; column clipping is applied
//     to the first and last lines so a left-to-right drag in one row
//     copies the right substring.
//   - ANSI escape codes are stripped so the clipboard is readable.
//   - Out-of-bounds Y values are clamped to the pane's visible window.
func extractPaneSelection(paneName string, startY, endY, startX, endX int) string {
	pane := findPaneByName(paneName)
	if pane == nil || len(pane.lines) == 0 {
		return ""
	}
	if startY > endY {
		startY, endY = endY, startY
		startX, endX = endX, startX
	}
	if startY < pane.topY {
		startY = pane.topY
	}
	if endY > pane.bottomY {
		endY = pane.bottomY
	}
	if endY < pane.topY || startY > pane.bottomY {
		return ""
	}
	// Account for top-padding rows that windowLines/renderChatPane add
	// when there's less content than the pane height. The rendered pane
	// has `padTop` blank rows BEFORE the real content, but `pane.lines`
	// holds only the real content. So a click at screen Y maps to flat
	// index `viewStart + (Y - paneTopY) - padTop`. Without the padTop
	// subtraction, copies were offset by the number of pad rows — which
	// is why the user saw "wrong text" copied for short panes.
	paneH := pane.bottomY - pane.topY + 1
	visible := len(pane.lines) - pane.viewStart
	if visible > paneH {
		visible = paneH
	}
	if visible < 0 {
		visible = 0
	}
	padTop := paneH - visible
	rowStart := (startY - pane.topY) - padTop
	rowEnd := (endY - pane.topY) - padTop
	if rowStart < 0 {
		rowStart = 0
	}
	if rowEnd < 0 {
		// Both clicks landed in padding — nothing to copy.
		return ""
	}
	startLine := pane.viewStart + rowStart
	endLine := pane.viewStart + rowEnd
	if startLine < 0 {
		startLine = 0
	}
	if endLine >= len(pane.lines) {
		endLine = len(pane.lines) - 1
	}
	if startLine > endLine {
		return ""
	}
	out := make([]string, 0, endLine-startLine+1)
	for i := startLine; i <= endLine; i++ {
		raw := stripANSI(pane.lines[i])
		if i == startLine && i == endLine {
			lo, hi := startX, endX
			if lo > hi {
				lo, hi = hi, lo
			}
			raw = clipColumns(raw, lo-pane.leftX, hi-pane.leftX)
		} else if i == startLine {
			raw = clipColumns(raw, startX-pane.leftX, len(raw))
		} else if i == endLine {
			raw = clipColumns(raw, 0, endX-pane.leftX)
		}
		out = append(out, raw)
	}
	return strings.TrimRight(strings.Join(out, "\n"), "\n ")
}

// scrollPaneAt scrolls whichever pane is under (x, y) by `delta` rows.
// Wheel-up sends positive delta (toward older content); wheel-down
// negative (toward newest). Falls back to chat if no pane matches —
// the user wheeled in the gap between panes; chat is the most useful
// default.
func (m *tuiModel) scrollPaneAt(x, y, delta int) {
	pane := findPane(x, y)
	if pane == nil {
		m.scrollChat(delta)
		return
	}
	switch pane.name {
	case "chat":
		m.scrollChat(delta)
	case "events":
		m.eventsScroll += delta
		if m.eventsScroll < 0 {
			m.eventsScroll = 0
		}
		// windowLines clamps high end to total-height anyway, but
		// also cap here so consecutive wheel-ups don't grow the
		// counter unboundedly.
		if max := len(pane.lines); m.eventsScroll > max {
			m.eventsScroll = max
		}
	case "pipeline":
		m.pipelineScroll += delta
		if m.pipelineScroll < 0 {
			m.pipelineScroll = 0
		}
		if max := len(pane.lines); m.pipelineScroll > max {
			m.pipelineScroll = max
		}
	case "files":
		m.fileScanScroll += delta
		if m.fileScanScroll < 0 {
			m.fileScanScroll = 0
		}
		if max := len(pane.lines); m.fileScanScroll > max {
			m.fileScanScroll = max
		}
	}
}

// stripANSI removes ANSI CSI / OSC sequences from s. Bubbletea/lipgloss
// embed lots of styling in chat lines; the clipboard only wants the
// human-readable characters.
func stripANSI(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	i := 0
	for i < len(s) {
		c := s[i]
		if c == 0x1b && i+1 < len(s) {
			next := s[i+1]
			switch next {
			case '[':
				// CSI: ESC [ ... <final byte 0x40-0x7e>
				j := i + 2
				for j < len(s) {
					if s[j] >= 0x40 && s[j] <= 0x7e {
						j++
						break
					}
					j++
				}
				i = j
				continue
			case ']':
				// OSC: ESC ] ... BEL or ESC \
				j := i + 2
				for j < len(s) && s[j] != 0x07 {
					if s[j] == 0x1b && j+1 < len(s) && s[j+1] == '\\' {
						j += 2
						break
					}
					j++
				}
				if j < len(s) && s[j] == 0x07 {
					j++
				}
				i = j
				continue
			default:
				i += 2
				continue
			}
		}
		b.WriteByte(c)
		i++
	}
	return b.String()
}

// clipColumns returns s[lo:hi] in rune positions, clamped to the
// string's actual length. Used to apply the column-precision clip on
// the first and last lines of a multi-line drag.
func clipColumns(s string, lo, hi int) string {
	r := []rune(s)
	if lo < 0 {
		lo = 0
	}
	if hi < lo {
		hi = lo
	}
	if hi > len(r) {
		hi = len(r)
	}
	if lo > len(r) {
		return ""
	}
	return string(r[lo:hi])
}

// envelopeLooksCancelled returns true if a /events envelope is just
// the cancellation echo we should hide from the events / pipeline pane
// while m.userCancelled is set. Error envelopes always qualify; stage
// _end with success=false qualifies because the only reason a stage
// would mark itself failed during a user-cancelled turn is the
// context-cancelled propagation.
func envelopeLooksCancelled(ev Envelope) bool {
	if ev.Type == EvtError {
		return true
	}
	if ev.Type == EvtStageEnd {
		if ok, _ := ev.Payload["success"].(bool); !ok {
			return true
		}
	}
	return false
}

// looksCancelled returns true if an error string looks like the
// user-initiated context cancellation rather than a real failure.
// The proxy/Go runtime surface this as "context canceled" /
// "context deadline exceeded" / "client disconnected"; the chat-stream
// scanner adds its own "context canceled" wrapping. None of these are
// useful for the user to see — they already pressed Ctrl+C.
func looksCancelled(err string) bool {
	if err == "" {
		return false
	}
	low := strings.ToLower(err)
	for _, sig := range []string{"context canceled", "context cancelled",
		"client disconnected", "request canceled", "operation was canceled",
		"use of closed network connection"} {
		if strings.Contains(low, sig) {
			return true
		}
	}
	return false
}

// formatPromptProgress renders the encoding-prompt progress row. When
// llama.cpp's /slots exposes prompt-eval token counts (some builds do,
// others don't) we render a 24-cell bar plus the running counters.
// When only elapsed time is known, we render a spinner + timer + the
// chars/4 estimate so the user sees motion and rough magnitude. The
// proxy emits one of these every 250ms while llama-server is grinding
// through prompt eval (30–90s on long histories).
func formatPromptProgress(processed, total int, pct float64, elapsedMS int64) string {
	secs := float64(elapsedMS) / 1000.0
	if pct > 0 && total > 0 {
		const barWidth = 24
		if pct > 1 {
			pct = 1
		}
		filled := int(pct*float64(barWidth) + 0.5)
		if filled > barWidth {
			filled = barWidth
		}
		bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)
		return fmt.Sprintf("encoding prompt  [%s] %d/%d (%.0f%%)  · %.1fs",
			bar, processed, total, pct*100, secs)
	}
	// No token counters — show a 10-frame braille spinner indexed by
	// 100ms elapsed so the spinner advances every tick, and surface the
	// chars/4 prompt estimate (`total`) so the user knows how big the
	// prompt is even when llama.cpp doesn't report live progress.
	frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	frame := frames[(elapsedMS/100)%int64(len(frames))]
	// %.1fs is one-decimal seconds (e.g. "5.4s") — that's what the user
	// asked for. The row redraws every 100ms via the spinner ticker so
	// the timer increments every tick, not every 250ms poll.
	if total > 0 {
		return fmt.Sprintf("encoding prompt  %s  ~%d tok · %.1fs",
			frame, total, secs)
	}
	return fmt.Sprintf("encoding prompt  %s  %.1fs", frame, secs)
}

// formatStreamingLLM renders the partial JSON the model is mid-emitting.
// For write_file calls, the bulk of tokens land inside `"content":"..."`
// as JSON-escaped source code (\n, \", \t…). Showing those raw makes
// the streaming view unreadable. We split at the content boundary and
// unescape the suffix in-place so the user sees code as code.
//
// The escape order matters: replace `\\` last via a placeholder so it
// doesn't double-substitute through \n / \". Truncated trailing escapes
// (e.g. a stray `\` at the buffer tail) are left alone — they'll resolve
// on the next token.
func formatStreamingLLM(s string) string {
	s = strings.TrimLeft(s, " \n\r\t")
	var cut int
	for _, marker := range []string{`"content":"`, `"content": "`} {
		if i := strings.Index(s, marker); i >= 0 {
			cut = i + len(marker)
			break
		}
	}
	if cut == 0 {
		return s
	}
	prefix := s[:cut]
	suffix := s[cut:]

	// Order matters: protect literal backslashes via a placeholder so
	// they don't double-substitute through the \n / \" rules.
	const placeholder = "\x00BS\x00"
	suffix = strings.ReplaceAll(suffix, `\\`, placeholder)
	suffix = strings.ReplaceAll(suffix, `\"`, `"`)
	suffix = strings.ReplaceAll(suffix, `\n`, "\n")
	suffix = strings.ReplaceAll(suffix, `\r`, "")
	suffix = strings.ReplaceAll(suffix, `\t`, "    ")
	suffix = strings.ReplaceAll(suffix, placeholder, `\`)

	// Cap to last N lines. The streaming buffer grows unbounded as the
	// model decodes (a 30k-token write_file is many KB) and we re-wrap
	// it on EVERY tick + token + resize event. Without a cap, drag-
	// resizing the terminal fires dozens of WindowSizeMsg in quick
	// succession; each one runs wrapPlain across the entire buffer,
	// which on a big content payload looks like a freeze. The cap
	// shows a tail view during streaming; the full buffer isn't lost
	// — it's still there in m.streamingLLMText, just truncated for
	// display until llm_call_end replaces the row with stats.
	const streamTailLines = 80
	lines := strings.Split(suffix, "\n")
	if len(lines) > streamTailLines {
		omitted := len(lines) - streamTailLines
		head := fmt.Sprintf("… (%d earlier lines)", omitted)
		suffix = head + "\n" + strings.Join(lines[len(lines)-streamTailLines:], "\n")
	}

	return prefix + "\n" + suffix
}

func summarizeToolArgs(name string, args json.RawMessage) string {
	var generic map[string]interface{}
	if err := json.Unmarshal(args, &generic); err != nil {
		return truncate(string(args), 80)
	}
	switch name {
	case "read_file", "write_file":
		return fmt.Sprintf("path=%v", generic["path"])
	case "edit_file":
		return fmt.Sprintf("path=%v  old=%q",
			generic["path"], truncateAny(generic["old_str"], 40))
	case "run_command":
		return truncateAny(generic["command"], 80)
	}
	parts := []string{}
	for k, v := range generic {
		parts = append(parts, fmt.Sprintf("%s=%s", k, truncateAny(v, 40)))
	}
	return truncate(strings.Join(parts, "  "), 100)
}

func summarizeToolResult(name string, data json.RawMessage) string {
	var generic map[string]interface{}
	if err := json.Unmarshal(data, &generic); err != nil || generic == nil {
		return truncate(string(data), 80)
	}
	for _, k := range []string{"summary", "stdout", "content", "message"} {
		if v, ok := generic[k]; ok {
			return truncateAny(v, 100)
		}
	}
	return ""
}

func (m tuiModel) View() string {
	if m.quitting {
		return ""
	}
	// Render with safe defaults if WindowSizeMsg hasn't arrived yet.
	// Some terminals / multiplexers don't reliably emit the initial
	// resize on alt-screen startup — without these defaults the user
	// stares at a blank "starting…" forever. The real size will swap
	// in as soon as the first WindowSizeMsg fires (or on first SIGWINCH).
	width, height := m.width, m.height
	if width <= 0 {
		width = 100
	}
	if height <= 0 {
		height = 30
	}
	header := renderHeader(m.proxyURL, m.workingDir, m.mode, m.turnActive,
		m.spinnerFrame, width)
	sel := selectionState{}
	if m.selecting {
		sel = selectionState{
			pane:   m.selPane,
			startY: m.selStartY, endY: m.selEndY,
			startX: m.selStartX, endX: m.selEndX,
		}
	}
	out, totalChatLines := layoutFullScreen(&m.state, m.envelope, m.chat,
		m.input.View(), m.input.Value(), m.inputMode,
		m.chatRenderer, header, m.turnActive, m.spinnerFrame,
		m.chatScroll, m.eventsScroll, m.pipelineScroll,
		m.fileEntries, m.modifiedFiles, m.fileScanScroll, m.workingDir,
		m.lastTurnTokens, m.totalTokensSession, m.maxContextTokens,
		m.hideFiles, m.hidePipeline, m.hideEvents,
		sel,
		width, height)
	// View is supposed to be pure, but we need to know the rendered
	// line count to clamp PgUp / mouse-wheel-up. Stashing it on the
	// model via a field write inside View is technically a side-effect
	// — Bubbletea calls View after every Update, so the value is fresh
	// by the next keystroke. The model value passes through Bubbletea's
	// runtime by value but we use a pointer-like trick via the receiver.
	// Update the model in-place is illegal in Go's value-receiver world,
	// so we use a stashed sync.Once-like idiom: write through a package
	// var. Avoiding that here — instead, scrollChat tolerates a stale
	// max (only matters for one keystroke). Capture happens via the
	// View → Update path: we write totalChatLines to a package-level
	// variable that Update reads on the next event.
	lastChatTotalRendered = totalChatLines
	if len(m.toasts) > 0 {
		out = overlayToast(out, m.toasts[len(m.toasts)-1].Body, width)
	}
	return out
}

// toastStyle renders the floating overlay banner. Reverse video instead
// of named bg/fg colors because lipgloss color rendering is profile-
// dependent (256-color, truecolor, none) and can silently strip styles
// in environments where TERM advertises poorly. Reverse(true) is the
// most universally-honored ANSI attribute — it pops against any
// underlying styling.
var toastStyle = lipgloss.NewStyle().
	Reverse(true).
	Bold(true).
	Padding(0, 1)

// overlayToast splices the toast text into the right side of the
// rendered header (top row). Auto-dismisses via tickMsg pruning. We
// overlay onto the header rather than the bottom because the bottom is
// the input box (lipgloss border characters at fixed positions) —
// overwriting those breaks the box rendering. The header is a single
// contiguous styled string we can safely truncate at the right edge.
func overlayToast(rendered, body string, width int) string {
	if body == "" || width < 30 {
		return rendered
	}
	idx := strings.IndexByte(rendered, '\n')
	if idx < 0 {
		return rendered
	}
	head := rendered[:idx]
	rest := rendered[idx:]
	styled := toastStyle.Render(body)
	tw := lipgloss.Width(styled)
	if tw > width-4 {
		max := width - 6
		if max < 8 {
			return rendered
		}
		styled = toastStyle.Render(truncate(body, max))
		tw = lipgloss.Width(styled)
	}
	headW := lipgloss.Width(head)
	if headW <= tw {
		return styled + rest
	}
	// Strip ANSI and re-anchor: keep leftmost (headW - tw) visible cols
	// then append the styled toast. Header's uniform style means losing
	// its trailing ANSI codes at the right edge is harmless.
	plain := stripANSI(head)
	plainRunes := []rune(plain)
	keep := len(plainRunes) - tw
	if keep < 0 {
		keep = 0
	}
	return string(plainRunes[:keep]) + styled + rest
}

// lastChatTotalRendered is updated by View() (which receives a value
// receiver) and read by Update() to clamp scroll on the next keystroke.
// Package-level so the side-effect is visible across Bubbletea's
// value-semantics dance with the model. Single TUI process per session,
// so no concurrency concern.
var lastChatTotalRendered int

// paneSnapshot records a pane's screen bounds and full pre-window
// content so the mouse handler can map a screen-cell click to the
// right pane and the right line index. layoutFullScreen rebuilds
// the list from scratch on every render.
//
//	name      — "chat" | "events" | "pipeline" | "files"
//	topY/bottomY — INCLUSIVE screen Y range of the pane's content
//	               rows (just inside the box's top/bottom border).
//	leftX/rightX — INCLUSIVE screen X range of the pane's content
//	               columns (just inside the box's L/R border).
//	viewStart — index of the first VISIBLE line in `lines`. A mouse
//	            at screen Y maps to lines[viewStart + (Y - topY)].
//	lines     — the full flattened pane content, pre-window. Already
//	            ANSI-styled; consumers strip ANSI before clipboard.
type paneSnapshot struct {
	name                       string
	topY, bottomY, leftX, rightX int
	viewStart                  int
	lines                      []string
}

// paneSnaps holds the most recent layout's pane bounds. Single TUI
// process, so no concurrency concern between View() (writer) and
// Update() (reader).
var paneSnaps []paneSnapshot

// findPane returns the snapshot whose bounds contain (x, y), or nil.
func findPane(x, y int) *paneSnapshot {
	for i := range paneSnaps {
		p := &paneSnaps[i]
		if y >= p.topY && y <= p.bottomY &&
			x >= p.leftX && x <= p.rightX {
			return p
		}
	}
	return nil
}

// findPaneByName returns the most recently rendered snapshot for the
// given name, or nil. Used by selection rendering to locate the
// active pane to overlay highlights on.
func findPaneByName(name string) *paneSnapshot {
	for i := range paneSnaps {
		if paneSnaps[i].name == name {
			return &paneSnaps[i]
		}
	}
	return nil
}

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

func renderHeader(proxyURL, workingDir, mode string, busy bool,
	spinnerFrame, width int) string {
	status := "idle"
	if busy {
		status = spinnerFrames[spinnerFrame%len(spinnerFrames)] + " busy"
	}
	left := lipgloss.NewStyle().
		Bold(true).
		Background(lipgloss.Color("63")).
		Foreground(lipgloss.Color("231")).
		Padding(0, 1).
		Render(fmt.Sprintf("ATLAS TUI"))
	right := lipgloss.NewStyle().
		Background(lipgloss.Color("236")).
		Foreground(lipgloss.Color("251")).
		Padding(0, 1).
		Render(fmt.Sprintf("%s · cwd:%s · %s · %s",
			proxyURL, truncate(workingDir, 30), mode, status))
	gap := width - lipgloss.Width(left) - lipgloss.Width(right)
	if gap < 1 {
		gap = 1
	}
	return left + strings.Repeat(" ", gap) + right
}

func formatEventLine(ev Envelope, width int) string {
	ts := time.Unix(0, int64(ev.Timestamp*1e9)).Format("15:04:05")
	color := typeColor(ev.Type)
	typeCell := lipgloss.NewStyle().Foreground(color).Width(13).Render(ev.Type)
	stageCell := lipgloss.NewStyle().Foreground(lipgloss.Color("251")).
		Width(14).Render(truncate(ev.Stage, 14))
	detail := summarizePayload(ev)

	line := fmt.Sprintf("%s  %s %s %s", ts, typeCell, stageCell, detail)
	line = strings.ReplaceAll(line, "\n", " ")
	if lipgloss.Width(line) > width {
		line = line[:width]
	}
	return line
}

func typeColor(t string) lipgloss.Color {
	switch t {
	case EvtStageStart:
		return lipgloss.Color("33")
	case EvtStageEnd:
		return lipgloss.Color("42")
	case EvtToolCall:
		return lipgloss.Color("214")
	case EvtToolResult:
		return lipgloss.Color("70")
	case EvtMetric:
		return lipgloss.Color("99")
	case EvtError:
		return lipgloss.Color("196")
	case EvtDone:
		return lipgloss.Color("226")
	}
	return lipgloss.Color("245")
}

func summarizePayload(ev Envelope) string {
	switch ev.Type {
	case EvtToolCall:
		return fmt.Sprintf("%v  %v",
			ev.Payload["name"], truncateAny(ev.Payload["args_summary"], 60))
	case EvtToolResult:
		ok := ev.Payload["success"] == true
		mark := "✓"
		if !ok {
			mark = "✗"
		}
		dur := ""
		if ev.DurationMS > 0 {
			dur = fmt.Sprintf(" %dms", ev.DurationMS)
		}
		return fmt.Sprintf("%s  %v%s",
			mark, ev.Payload["name"], dur)
	case EvtMetric:
		return fmt.Sprintf("%v = %v",
			ev.Payload["name"], ev.Payload["value"])
	case EvtError:
		return truncateAny(ev.Payload["message"], 80)
	case EvtStageEnd:
		ok := ev.Payload["success"] == true
		mark := "✓"
		if !ok {
			mark = "✗"
		}
		dur := ""
		if ev.DurationMS > 0 {
			dur = fmt.Sprintf(" %dms", ev.DurationMS)
		}
		return mark + dur
	case EvtDone:
		ok := ev.Payload["success"] == true
		mark := "✓"
		if !ok {
			mark = "✗"
		}
		return fmt.Sprintf("%s  total %vms",
			mark, ev.Payload["total_duration_ms"])
	}
	if d, ok := ev.Payload["detail"].(string); ok {
		return truncate(d, 80)
	}
	return ""
}

func truncate(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "…"
}

func truncateAny(v interface{}, n int) string {
	s, ok := v.(string)
	if !ok {
		return fmt.Sprintf("%v", v)
	}
	return truncate(s, n)
}
