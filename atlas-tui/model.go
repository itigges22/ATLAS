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

	// Files added via /add — appended as a hint to each /v1/agent message.
	contextFiles map[string]bool

	// Working dir + permission mode for /v1/agent payloads.
	workingDir string
	mode       string

	// Polish state — spinner phase, last-sent message for Ctrl+R.
	spinnerFrame int
	lastUserMsg  string

	// Lifecycle
	quitting bool
}

func newTUIModel(proxyURL string) tuiModel {
	ta := textarea.New()
	ta.Placeholder = "Send a message…  Enter=send  Shift+Enter=newline  Ctrl+L=clear  Ctrl+T=mode  Ctrl+R=resend  Ctrl+C=cancel/quit"
	ta.Prompt = "> "
	ta.CharLimit = 8000
	ta.SetWidth(80)
	ta.SetHeight(3)
	ta.Focus()

	wd, _ := os.Getwd()

	// Glamour renderer for assistant markdown. Auto-style picks dark
	// or light based on the terminal background. Width is set per
	// render so we can adapt to resizes.
	renderer, _ := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(78),
	)

	return tuiModel{
		proxyURL:     proxyURL,
		events:       make(chan Envelope, 256),
		state:        newPipelineState(),
		maxLines:     1000,
		input:        ta,
		chatEvents:   make(chan chatEvent, 64),
		chatRenderer: renderer,
		workingDir:   wd,
		mode:         "default",
	}
}

func (m tuiModel) Init() tea.Cmd {
	return tea.Batch(
		waitForEnvelope(m.events),
		waitForChatEvent(m.chatEvents),
		tickEvery(time.Second),
		textarea.Blink,
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

// sendChatCmd kicks off a /v1/agent turn. Runs sendChat in a goroutine
// because Bubbletea Cmds should be quick — the goroutine pumps events
// onto m.chatEvents which the model drains via waitForChatEvent.
func (m *tuiModel) sendChatCmd(message string) tea.Cmd {
	ctx, cancel := context.WithCancel(context.Background())
	sessionID := newSessionID()
	m.turnCancel = cancel
	m.turnSessionID = sessionID
	m.turnActive = true

	proxyURL := m.proxyURL
	workingDir := m.workingDir
	mode := m.mode
	out := m.chatEvents

	return func() tea.Msg {
		go func() {
			err := sendChat(ctx, proxyURL, message, workingDir, mode, sessionID, out)
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
		case "enter":
			// Enter sends; Shift+Enter (or Alt+Enter) inserts newline.
			// textarea handles Shift+Enter as KeyShiftEnter ("shift+enter").
			if !m.turnActive {
				text := strings.TrimSpace(m.input.Value())
				if text == "" {
					return m, nil
				}
				m.input.Reset()
				// Slash commands intercepted before agent send.
				if consumed, slashCmd, quit := m.handleSlash(text); consumed {
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
				cmds = append(cmds, m.sendChatCmd(text+m.contextSuffix()))
				return m, tea.Batch(cmds...)
			}
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		// Textarea width = full screen minus border (2).
		m.input.SetWidth(max(20, msg.Width-2))
		// Re-build glamour renderer at the new width.
		if r, err := glamour.NewTermRenderer(
			glamour.WithAutoStyle(),
			glamour.WithWordWrap(max(20, msg.Width-4)),
		); err == nil {
			m.chatRenderer = r
		}

	case envelopeMsg:
		m.state.apply(msg.ev)
		m.envelope = append(m.envelope, msg.ev)
		if len(m.envelope) > m.maxLines {
			m.envelope = m.envelope[len(m.envelope)-m.maxLines:]
		}
		return m, waitForEnvelope(m.events)

	case chatStreamMsg:
		if msg.ev.Type == "__turn_done__" {
			m.turnActive = false
			var p struct {
				Err string `json:"err"`
			}
			_ = json.Unmarshal(msg.ev.Data, &p)
			if p.Err != "" {
				m.chat = append(m.chat, chatMessage{
					Role: roleSystem, Meta: "error",
					Body: p.Err,
				})
			}
		} else {
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
		return m, nil

	case tickMsg:
		m.spinnerFrame++
		return m, tickEvery(time.Second)
	}

	// Forward remaining keystrokes to the textarea (typing, arrows…).
	if !m.quitting {
		var taCmd tea.Cmd
		m.input, taCmd = m.input.Update(msg)
		cmds = append(cmds, taCmd)
	}
	return m, tea.Batch(cmds...)
}

// appendChatEvent translates a /v1/agent SSE event into one or more
// chat history rows.
func (m *tuiModel) appendChatEvent(ev chatEvent) {
	switch ev.Type {
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
				Role: roleTool, Meta: p.Name,
				Body: summarizeToolArgs(p.Name, p.Args),
			})
		}

	case "tool_result":
		var p struct {
			Tool    string          `json:"tool"`
			Success bool            `json:"success"`
			Data    json.RawMessage `json:"data"`
			Error   string          `json:"error"`
		}
		if json.Unmarshal(ev.Data, &p) == nil {
			body := p.Error
			if p.Success {
				body = summarizeToolResult(p.Tool, p.Data)
			}
			m.chat = append(m.chat, chatMessage{
				Role: roleTool, Meta: p.Tool,
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
	}
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
	if m.width == 0 {
		return "atlas-tui: starting…"
	}

	header := renderHeader(m.proxyURL, m.workingDir, m.mode, m.turnActive,
		m.spinnerFrame, m.width)
	return layoutFullScreen(&m.state, m.envelope, m.chat, m.input.View(),
		m.chatRenderer, header, m.width, m.height)
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
