// Demo mode — split-pane recording subprogram. Two concurrent
// /v1/agent sessions against the same atlas-proxy: left pane runs with
// bypass_v3=true (raw 9B, no orchestration), right pane runs V3
// normally. Same model, two llama-server slots, side-by-side.
//
// Implementation note: each pane uses a real `tuiModel` as its chat
// state holder. We forward every chatEvent into that model's
// appendChatEvent, then call renderChatPane to draw. That way the
// V3 pane formats EXACTLY like a normal atlas-tui session — every
// tool call, V3 stage, token stream, lens score row — with no
// reimplementation. The raw pane gets the same machinery so its
// tool-call and reasoning-token rendering look polished too.
//
// Reliability without scripting: prompt comes from a curated bank in
// docs/demo/demo_prompts.json — each entry is hand-validated to expose
// the V3 difference. Inference itself is real.

package main

import (
	"bufio"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

//go:embed demo_prompts_fallback.json
var demoPromptsFallback embed.FS

// demoPrompt is one entry from the curated bank. The expected_* fields
// document what the raw side is supposed to trip on and what V3 is
// supposed to repair — they're not enforced at runtime, but they let
// future contributors know what each prompt is *for*. Difficulty is the
// recording-budget bucket: "short" = single-file, fast (good for the
// 30-60s cut); "medium" = 2 files, ~3-5 min; "long" = multi-file
// Flask/FastAPI, can take 30+ min. /demo <length> filters the bank
// before random pick so the cut doesn't blow its budget.
type demoPrompt struct {
	Prompt             string `json:"prompt"`
	Difficulty         string `json:"difficulty,omitempty"` // "short" | "medium" | "long"
	ExpectedRawFailure string `json:"expected_raw_failure"`
	ExpectedV3Repair   string `json:"expected_v3_repair"`
	Notes              string `json:"notes,omitempty"`
}

// demoStreamMsg delivers a chatEvent from one of the two streaming
// goroutines into the Bubbletea loop. Side picks which child to mutate.
type demoStreamMsg struct {
	side string // "raw" | "v3"
	evt  chatEvent
}

// demoBatchMsg carries multiple events drained in one shot. Heavy token
// streams produce dozens of events/sec; without batching, the
// one-event-per-Update + two-pane glamour render becomes a backpressure
// bottleneck (~10 events/sec ceiling) and the TUI visibly stutters.
// Batching lets View() run once per drain instead of once per event.
type demoBatchMsg struct {
	stream []demoStreamMsg
	dones  []demoStreamDone
}

// demoStreamDone signals one of the two streams has finished (or failed).
type demoStreamDone struct {
	side string
	err  error
}

// demoTickMsg drives the prompt type-out animation and the post-finish
// hold timer.
type demoTickMsg time.Time

type demoModel struct {
	proxyURL   string
	workingDir string
	length     string // "short" (30s) | "medium" (60s) | "long" (3-5m)
	prompt     demoPrompt

	// Per-side sandbox subdirectories under workingDir, created at
	// startup and passed to the proxy as `sandbox_subdir`. Each side
	// writes only inside its own sandbox so AST-edits, write_file, and
	// V3 candidate selection on the V3 side don't collide with the raw
	// side's output (and vice-versa). Kept after exit per the "review
	// what each side produced" workflow.
	rawSandbox string // e.g. ".demo-raw-1715562342"
	v3Sandbox  string

	width, height int

	// Per-pane state. Each child tuiModel is constructed via newTUIModel
	// so its chatRenderer (glamour) and pipeline-state struct are wired,
	// but we never call its Init() — no goroutines, no /events SSE, no
	// keyboard input. We're just using it as a state container for chat
	// rendering. All event ingestion flows through child.appendChatEvent
	// and rendering through renderChatPane.
	rawChild *tuiModel
	v3Child  *tuiModel

	rawDone, v3Done bool
	rawErr, v3Err   error

	// Sticky "ever generated a token" flag per side. tuiModel's
	// streamingLLM/streamingV3 flags clear on llm_call_end (turn
	// boundary), so a multi-turn agent loop flips through
	// processing-prompt → streaming → processing-prompt-again. Tracking
	// it here at the demo level instead keeps the status header at
	// "streaming…" once the model starts producing tokens.
	rawEverStreamed, v3EverStreamed bool

	// Prompt type-out: chars revealed left of the user-input line at the
	// top of the screen. While typing, streams haven't fired yet — the
	// real /v1/agent requests go out the moment typing completes, not
	// on Init, so the viewer sees prompt → reaction in the right order.
	promptShown   int
	streamsFired  bool

	// Spinner frame index. Ticks on demoTickMsg; consumed in
	// streamStatus so each pane shows a moving glyph while in flight.
	// Without this the V3 side's long PlanSearch / repair windows look
	// dead to a viewer who can't see the proxy logs.
	spinnerFrame int

	// Output review mode: when both sides finish, each pane switches
	// from streaming chat to a file-tree + selected-file-contents view
	// of its sandbox. Tab/space cycles files within a pane; 1-9 jumps.
	// outputMode is set once both rawDone and v3Done are true.
	outputMode     bool
	rawFiles       []string // relative paths inside rawSandbox, sorted
	v3Files        []string
	rawSelectedIdx int
	v3SelectedIdx  int
	activePane     string // "raw" | "v3" — which side's file cycler responds to keys

	events chan demoEvent

	ctx    context.Context
	cancel context.CancelFunc

	startedAt  time.Time
	finishedAt time.Time
}

// demoEvent is the channel payload from goroutines to the Bubbletea
// loop. Exactly one of {stream, done} is meaningful.
type demoEvent struct {
	stream demoStreamMsg
	done   *demoStreamDone
}

// pickPrompt loads docs/demo/demo_prompts.json (looked up under the
// session's working dir, then the process cwd, then the embedded
// fallback) and returns
// a random prompt whose difficulty fits the requested length. Bucket
// rule: `short` picks only difficulty=short; `medium` picks short or
// medium (so the bank stays useful for 60s cuts even with no `medium`
// entries); `long` picks anything. If no prompt matches, falls back to
// the full bank so /demo always returns SOMETHING rather than
// erroring on a misconfigured length.
func pickPrompt(workingDir, length string) (demoPrompt, error) {
	candidates := []string{
		filepath.Join(workingDir, "docs", "demo", "demo_prompts.json"),
		"docs/demo/demo_prompts.json",
	}
	var raw []byte
	for _, p := range candidates {
		if b, err := os.ReadFile(p); err == nil {
			raw = b
			break
		}
	}
	if raw == nil {
		b, err := demoPromptsFallback.ReadFile("demo_prompts_fallback.json")
		if err != nil {
			return demoPrompt{}, fmt.Errorf("demo prompts not found: %w", err)
		}
		raw = b
	}
	var bank []demoPrompt
	if err := json.Unmarshal(raw, &bank); err != nil {
		return demoPrompt{}, fmt.Errorf("parse demo prompts: %w", err)
	}
	if len(bank) == 0 {
		return demoPrompt{}, fmt.Errorf("demo prompt bank empty")
	}
	pool := filterByDifficulty(bank, length)
	if len(pool) == 0 {
		pool = bank
	}
	return pool[rand.Intn(len(pool))], nil
}

// filterByDifficulty returns the subset of prompts that fit the
// requested length. Untagged prompts (Difficulty == "") are treated as
// "medium" so old banks keep working without back-fill.
func filterByDifficulty(bank []demoPrompt, length string) []demoPrompt {
	var out []demoPrompt
	for _, p := range bank {
		d := p.Difficulty
		if d == "" {
			d = "medium"
		}
		switch length {
		case "short":
			if d == "short" {
				out = append(out, p)
			}
		case "medium":
			if d == "short" || d == "medium" {
				out = append(out, p)
			}
		default: // "long" and anything unknown
			out = append(out, p)
		}
	}
	return out
}

func newDemoModel(proxyURL, workingDir, length string) (*demoModel, error) {
	p, err := pickPrompt(workingDir, length)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())

	rawChild := newTUIModel(proxyURL)
	v3Child := newTUIModel(proxyURL)

	// Per-run sandbox subdirs. Timestamps make recordings reproducible
	// (you know which dir came from which take) and stop collisions
	// between rapid re-runs. Created here so the demo can show "files
	// written:" view from the host side after both sides finish.
	ts := time.Now().Unix()
	rawSandbox := fmt.Sprintf(".demo-raw-%d", ts)
	v3Sandbox := fmt.Sprintf(".demo-v3-%d", ts)
	for _, sub := range []string{rawSandbox, v3Sandbox} {
		if err := os.MkdirAll(filepath.Join(workingDir, sub), 0o755); err != nil {
			cancel()
			return nil, fmt.Errorf("create sandbox %s: %w", sub, err)
		}
	}

	return &demoModel{
		proxyURL:   proxyURL,
		workingDir: workingDir,
		length:     length,
		prompt:     p,
		events:     make(chan demoEvent, 1024),
		ctx:        ctx,
		cancel:     cancel,
		rawChild:   &rawChild,
		v3Child:    &v3Child,
		rawSandbox: rawSandbox,
		v3Sandbox:  v3Sandbox,
		activePane: "v3",
	}, nil
}

func (m *demoModel) Init() tea.Cmd {
	m.startedAt = time.Now()
	// Streams are NOT fired here — they fire from the tick handler the
	// instant the prompt animation completes. This keeps the visual
	// timeline honest: viewer sees the prompt being typed, *then* both
	// sides react. Without this, V3's "candidate 1/3" stage label
	// renders before the prompt is fully visible.
	return tea.Batch(
		m.drainEvents(),
		demoTick(80*time.Millisecond),
	)
}

// startStreams fires both /v1/agent POSTs in parallel. They share the
// same proxy (same llama-server, two slots) but differ on bypass_v3.
// disable_fresh_slot=true on both so PC-045 doesn't wipe the prefix
// cache between the model's keep-warm pings and the demo run.
func (m *demoModel) startStreams() tea.Cmd {
	return func() tea.Msg {
		go m.runStream("raw", true)
		go m.runStream("v3", false)
		return nil
	}
}

func (m *demoModel) runStream(side string, bypassV3 bool) {
	out := make(chan chatEvent, 128)
	go func() {
		for evt := range out {
			m.events <- demoEvent{stream: demoStreamMsg{side: side, evt: evt}}
		}
	}()
	sid := fmt.Sprintf("demo-%s-%d", side, time.Now().UnixNano())
	sandbox := m.rawSandbox
	if side == "v3" {
		sandbox = m.v3Sandbox
	}
	err := sendChatOpts(m.ctx, m.proxyURL, m.prompt.Prompt, m.workingDir,
		"yolo", sid, nil, demoOpts{
			bypassV3:         bypassV3,
			disableFreshSlot: true,
			sandboxSubdir:    sandbox,
		}, out)
	close(out)
	m.events <- demoEvent{done: &demoStreamDone{side: side, err: err}}
}

// drainEvents pulls events from the shared channel and batches every
// ready event into a single demoBatchMsg. Blocks on the first event so
// idle ticks don't spin, then non-blockingly drains up to maxBatch more
// before returning. View() runs once per batch instead of once per
// event — the key win during heavy token streams.
const maxDemoBatch = 128

func (m *demoModel) drainEvents() tea.Cmd {
	return func() tea.Msg {
		var batch demoBatchMsg
		// First event: block until something arrives or context dies.
		select {
		case ev, ok := <-m.events:
			if !ok {
				return nil
			}
			absorb(&batch, ev)
		case <-m.ctx.Done():
			return nil
		}
		// Drain the rest of what's already queued, non-blocking.
		for i := 1; i < maxDemoBatch; i++ {
			select {
			case ev, ok := <-m.events:
				if !ok {
					return batch
				}
				absorb(&batch, ev)
			default:
				return batch
			}
		}
		return batch
	}
}

func absorb(b *demoBatchMsg, ev demoEvent) {
	if ev.done != nil {
		b.dones = append(b.dones, *ev.done)
		return
	}
	b.stream = append(b.stream, ev.stream)
}

func demoTick(d time.Duration) tea.Cmd {
	return tea.Tick(d, func(t time.Time) tea.Msg { return demoTickMsg(t) })
}

func (m *demoModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		// Forward a half-width sizing to each child so glamour wraps to
		// the per-pane column. The chat-pane wrapper takes ~4 columns
		// for borders + padding.
		colW := (msg.Width - 4) / 2
		m.rawChild.width = colW
		m.rawChild.height = msg.Height - 4
		m.v3Child.width = colW
		m.v3Child.height = msg.Height - 4
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q", "esc":
			m.cancel()
			return m, tea.Quit
		}
		if m.outputMode {
			m.handleOutputKey(msg.String())
			return m, nil
		}

	case demoBatchMsg:
		// Apply every event in this batch before returning so View only
		// renders once. Stream events first, then done flags.
		for _, s := range msg.stream {
			// Sticky generation flag: any token-bearing event flips
			// this side's "ever streamed" bit so the header doesn't
			// flop back to "processing prompt" on subsequent turns.
			switch s.evt.Type {
			case "llm_token", "reasoning_token", "v3_token", "v3_reasoning_token":
				if s.side == "raw" {
					m.rawEverStreamed = true
				} else {
					m.v3EverStreamed = true
				}
			}
			if s.side == "raw" {
				m.rawChild.appendChatEvent(s.evt)
			} else {
				m.v3Child.appendChatEvent(s.evt)
			}
		}
		for _, d := range msg.dones {
			switch d.side {
			case "raw":
				m.rawDone = true
				m.rawErr = d.err
			case "v3":
				m.v3Done = true
				m.v3Err = d.err
			}
		}
		if m.rawDone && m.v3Done && m.finishedAt.IsZero() {
			m.finishedAt = time.Now()
			m.enterOutputMode()
		}
		return m, m.drainEvents()

	case demoTickMsg:
		m.spinnerFrame++
		if !m.outputMode && m.promptShown < len(m.prompt.Prompt) {
			m.promptShown++
		}
		// Fire the streams exactly once, the tick after the prompt
		// animation finishes. The trailing tick on len() == promptShown
		// gives one frame of "complete prompt with no caret" before the
		// status flips to processing-prompt — small thing but reads
		// noticeably cleaner on camera.
		if !m.streamsFired && !m.outputMode && m.promptShown >= len(m.prompt.Prompt) {
			m.streamsFired = true
			return m, tea.Batch(
				m.startStreams(),
				demoTick(80*time.Millisecond),
			)
		}
		// In output mode the user explores at their own pace — no auto-quit.
		// Pre-output mode the tick is just driving the prompt-typing animation.
		return m, demoTick(80 * time.Millisecond)
	}

	return m, nil
}

var (
	demoRawTitleStyle = lipgloss.NewStyle().
				Bold(true).
				Padding(0, 1).
				Background(lipgloss.Color("88")). // muted red
				Foreground(lipgloss.Color("231"))

	demoV3TitleStyle = lipgloss.NewStyle().
				Bold(true).
				Padding(0, 1).
				Background(lipgloss.Color("28")). // muted green
				Foreground(lipgloss.Color("231"))

	demoStatusStyle = lipgloss.NewStyle().
			Faint(true).
			Padding(0, 1)

	demoPaneStyle = lipgloss.NewStyle().
			Border(lipgloss.NormalBorder()).
			BorderForeground(lipgloss.Color("240")).
			Padding(0, 1)

	demoPromptStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("11"))

	demoSelectedFileStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("11"))
)

func (m *demoModel) View() string {
	if m.width == 0 || m.height == 0 {
		return "loading demo…"
	}

	// Top row: typed prompt animation. Caret blinks via promptShown.
	// In output mode the prompt header becomes a compact reminder of
	// what was asked so the file diff has context.
	prompt := m.prompt.Prompt
	if !m.outputMode && m.promptShown < len(prompt) {
		prompt = prompt[:m.promptShown] + "▌"
	}
	// Wrap to terminal width — the real-world prompts in the bank are
	// 200+ chars and would otherwise overflow off the right edge.
	// Subtract 2 for the "> " prefix on the first visible line.
	headerLines := wrapPlain(prompt, m.width-2)
	headerStyled := make([]string, len(headerLines))
	for i, line := range headerLines {
		if i == 0 {
			headerStyled[i] = demoPromptStyle.Render("> " + line)
		} else {
			// Continuation lines get a 2-space indent so the eye tracks
			// them as a continuation of the prompt rather than separate text.
			headerStyled[i] = demoPromptStyle.Render("  " + line)
		}
	}
	header := strings.Join(headerStyled, "\n")

	// Column width accounting:
	//   m.width = total terminal width
	//   subtract 4 for the two outer borders + padding columns
	//   divide by 2 for two equal panes
	//   per-pane inner content width = colW - 4 (border + padding inside pane)
	colW := (m.width - 4) / 2
	// Body height shrinks by the wrapped header's line count. Without
	// this, a long prompt steals rows from the chat panes and chat
	// content gets clipped at the bottom.
	bodyH := m.height - len(headerLines) - 3 // header + footer + breathing

	var row, footer string
	if m.outputMode {
		rawPane := m.renderOutputPane("raw", colW, bodyH)
		v3Pane := m.renderOutputPane("v3", colW, bodyH)
		row = lipgloss.JoinHorizontal(lipgloss.Top, rawPane, v3Pane)
		footer = demoStatusStyle.Render(
			"output review  ·  tab: switch side  ·  n/p (or ←/→): cycle file  ·  1-9: jump  ·  q: quit  ·  active: " + m.activePane)
	} else {
		rawTitle := demoRawTitleStyle.Render("RAW 9B  ·  no V3 orchestration") + "  " +
			demoStatusStyle.Render(streamStatus(m.rawChild, m.rawDone, m.rawEverStreamed, m.rawErr))
		v3Title := demoV3TitleStyle.Render("ATLAS V3  ·  plan → sample → verify → repair") + "  " +
			demoStatusStyle.Render(streamStatus(m.v3Child, m.v3Done, m.v3EverStreamed, m.v3Err))

		// Reserve one row at the bottom of each pane for the thinking
		// spinner (matches the main ATLAS TUI's pattern). chat content
		// loses one row in flight; final frame on done has no spinner
		// so the chat reclaims that row.
		rawThink := thinkingRow(m.rawDone, m.spinnerFrame)
		v3Think := thinkingRow(m.v3Done, m.spinnerFrame+5) // phase offset

		rawChatH := bodyH - 2
		v3ChatH := bodyH - 2
		if rawThink != "" {
			rawChatH--
		}
		if v3Think != "" {
			v3ChatH--
		}

		rawChat, _, _, _, _, _ := renderChatPane(m.rawChild.chat, m.rawChild.chatRenderer,
			rawChatH, colW-4, 0)
		v3Chat, _, _, _, _, _ := renderChatPane(m.v3Child.chat, m.v3Child.chatRenderer,
			v3ChatH, colW-4, 0)

		rawBody := rawTitle + "\n\n" + rawChat
		if rawThink != "" {
			rawBody += "\n" + rawThink
		}
		v3Body := v3Title + "\n\n" + v3Chat
		if v3Think != "" {
			v3Body += "\n" + v3Think
		}
		rawPane := demoPaneStyle.Width(colW).Height(bodyH).Render(rawBody)
		v3Pane := demoPaneStyle.Width(colW).Height(bodyH).Render(v3Body)

		row = lipgloss.JoinHorizontal(lipgloss.Top, rawPane, v3Pane)
		footer = demoStatusStyle.Render("recording demo  ·  ctrl+c to abort")
	}

	return header + "\n" + row + "\n" + footer
}

// streamStatus produces the title-bar status text WITHOUT a spinner —
// the moving spinner lives at the bottom of each chat pane now (see
// thinkingRow), mirroring the main TUI's pattern. The title bar just
// states "processing prompt N%" or "streaming…" or "✓ done".
//
// `everStreamed` is the demo's sticky bit — once this side has produced
// any generation token, the header stays at "streaming…" until done.
// Without it, multi-turn agent loops flip back to "processing prompt"
// on every fresh LLM call, which reads as a glitch on camera.
func streamStatus(child *tuiModel, done, everStreamed bool, err error) string {
	if done {
		if err != nil {
			return "✗ error"
		}
		return "✓ done"
	}
	if everStreamed {
		return "streaming…"
	}
	if child.promptTotal > 0 && child.promptPct < 100 {
		return fmt.Sprintf("processing prompt %.0f%%", child.promptPct)
	}
	return "waiting…"
}

// thinkingRow is the bottom-of-pane orange spinner row, matching the
// main TUI's pattern (panes.go:752-759). Returns empty string when the
// side is done — last frame freezes on the chat content rather than a
// dangling spinner.
func thinkingRow(done bool, spinnerFrame int) string {
	if done {
		return ""
	}
	mark := spinnerFrames[spinnerFrame%len(spinnerFrames)]
	verb := thinkingVerbs[(spinnerFrame/20)%len(thinkingVerbs)]
	return runStyle.Render(fmt.Sprintf("  %s %s…", mark, verb))
}

// enterOutputMode scans both sandbox directories for files and flips
// the demo into review mode so the recorder can pan across the two
// outputs after generation finishes. We walk the tree once (caching the
// list) and read file contents lazily on selection so a large sandbox
// doesn't stall the transition.
func (m *demoModel) enterOutputMode() {
	m.outputMode = true
	m.rawFiles = scanSandbox(filepath.Join(m.workingDir, m.rawSandbox))
	m.v3Files = scanSandbox(filepath.Join(m.workingDir, m.v3Sandbox))
}

// scanSandbox walks a sandbox dir and returns relative paths to every
// regular file, sorted. Empty or unreadable trees return nil — the
// caller renders a "(no files written)" placeholder.
func scanSandbox(root string) []string {
	var files []string
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}
		files = append(files, rel)
		return nil
	})
	sort.Strings(files)
	return files
}

// handleOutputKey routes navigation in output-review mode. Tab cycles
// the active pane between sides; n/p (or arrow keys) cycles files
// within the active pane; 1-9 jumps.
func (m *demoModel) handleOutputKey(key string) {
	switch key {
	case "tab", "shift+tab":
		if m.activePane == "raw" {
			m.activePane = "v3"
		} else {
			m.activePane = "raw"
		}
	case "right", "l", "n", " ":
		m.cycleActiveFile(+1)
	case "left", "h", "p":
		m.cycleActiveFile(-1)
	default:
		if len(key) == 1 && key[0] >= '1' && key[0] <= '9' {
			idx := int(key[0] - '1')
			files := m.activeFiles()
			if idx < len(files) {
				m.setActiveIdx(idx)
			}
		}
	}
}

func (m *demoModel) activeFiles() []string {
	if m.activePane == "raw" {
		return m.rawFiles
	}
	return m.v3Files
}

func (m *demoModel) cycleActiveFile(delta int) {
	files := m.activeFiles()
	if len(files) == 0 {
		return
	}
	var idx *int
	if m.activePane == "raw" {
		idx = &m.rawSelectedIdx
	} else {
		idx = &m.v3SelectedIdx
	}
	*idx = (*idx + delta + len(files)) % len(files)
}

func (m *demoModel) setActiveIdx(i int) {
	if m.activePane == "raw" {
		m.rawSelectedIdx = i
	} else {
		m.v3SelectedIdx = i
	}
}

// renderOutputPane builds the post-generation file-tree + selected-file
// contents view for one side. The active pane gets a brighter border
// so the viewer knows which side keys apply to.
func (m *demoModel) renderOutputPane(side string, w, h int) string {
	var (
		sandbox   string
		files     []string
		selected  int
		title     string
		titleStyle lipgloss.Style
	)
	if side == "raw" {
		sandbox = m.rawSandbox
		files = m.rawFiles
		selected = m.rawSelectedIdx
		title = "RAW 9B  ·  " + sandbox
		titleStyle = demoRawTitleStyle
	} else {
		sandbox = m.v3Sandbox
		files = m.v3Files
		selected = m.v3SelectedIdx
		title = "ATLAS V3  ·  " + sandbox
		titleStyle = demoV3TitleStyle
	}

	// File list. Trim if too tall — the body needs space too.
	treeHeight := h / 3
	if treeHeight < 3 {
		treeHeight = 3
	}
	tree := []string{titleStyle.Render(title), ""}
	if len(files) == 0 {
		tree = append(tree, demoStatusStyle.Render("(no files written)"))
	} else {
		for i, f := range files {
			marker := "  "
			if i == selected {
				marker = "▸ "
			}
			line := marker + f
			if i == selected {
				line = demoSelectedFileStyle.Render(line)
			}
			tree = append(tree, line)
			if len(tree) >= treeHeight {
				tree = append(tree, demoStatusStyle.Render(fmt.Sprintf("  … +%d more", len(files)-i-1)))
				break
			}
		}
	}

	// File contents. Read on demand to keep the transition cheap.
	bodyHeight := h - len(tree) - 2
	if bodyHeight < 1 {
		bodyHeight = 1
	}
	body := ""
	if len(files) > 0 && selected < len(files) {
		fpath := filepath.Join(m.workingDir, sandbox, files[selected])
		body = readFileForDisplay(fpath, bodyHeight, w-4)
	}

	border := lipgloss.Color("240")
	if m.activePane == side {
		border = lipgloss.Color("11") // bright yellow on the focused side
	}
	pane := demoPaneStyle.
		BorderForeground(border).
		Width(w).
		Height(h).
		Render(strings.Join(tree, "\n") + "\n\n" + body)
	return pane
}

// readFileForDisplay reads up to maxLines lines and trims them to maxCols.
// Big files get a tail truncation note. Binary files are flagged.
func readFileForDisplay(path string, maxLines, maxCols int) string {
	const sniffBytes = 512
	f, err := os.Open(path)
	if err != nil {
		return demoStatusStyle.Render(fmt.Sprintf("(cannot read: %v)", err))
	}
	defer f.Close()
	sniff := make([]byte, sniffBytes)
	n, _ := f.Read(sniff)
	for _, b := range sniff[:n] {
		if b == 0 {
			return demoStatusStyle.Render("(binary file)")
		}
	}
	_, _ = f.Seek(0, 0)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		if maxCols > 0 && len(line) > maxCols {
			line = line[:maxCols-1] + "…"
		}
		lines = append(lines, line)
		if len(lines) >= maxLines {
			lines = append(lines, demoStatusStyle.Render(fmt.Sprintf("… (truncated to %d lines)", maxLines)))
			break
		}
	}
	return strings.Join(lines, "\n")
}

// runDemo launches the demo subprogram in the same terminal session.
// Called from main.go after the primary TUI exits with launchDemoMode
// set, or from a `--demo` flag on cold start.
func runDemo(proxyURL, workingDir, length string) error {
	model, err := newDemoModel(proxyURL, workingDir, length)
	if err != nil {
		return err
	}
	prog := tea.NewProgram(model, tea.WithAltScreen())
	_, err = prog.Run()
	return err
}
