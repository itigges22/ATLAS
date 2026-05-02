// PC-062: slash command handling.
//
// User input starting with "/" is intercepted before /v1/agent send.
// Three categories:
//
//   local      — /help, /quit, /add, /drop  (mutate TUI state, no I/O)
//   git wrappers — /commit, /diff, /undo    (shell out to git, capture output)
//   shell      — /run <cmd>                 (shell out, capture output)
//
// Shell-out commands return their output as a slashResultMsg which the
// model appends to chat as a tool-style row.

package main

import (
	"context"
	"fmt"
	"os/exec"
	"sort"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// slashResultMsg carries the output of a shelled-out slash command back
// to the model. err is non-nil when the command failed (non-zero exit
// or process error); output is still set so the user sees stderr.
type slashResultMsg struct {
	command string
	output  string
	err     error
}

// slashCommandHelp is the static text emitted by /help. Single source
// of truth — keep in lockstep with handleSlash's switch.
const slashCommandHelp = `Slash commands
  /help  (or ?)           Show this help.
  /add <path>             Add file to the agent's working context.
  /drop <path>            Remove file from the working context.
  /context                List files currently in context.
  /diff [path]            Show git diff.
  /commit [msg]           Stage all changes and create a commit.
  /undo                   Revert the last commit (keep changes in tree).
  /run <cmd>              Run a shell command in the working dir.
  /clear                  Clear the chat history (keeps session tokens).
  /compact                Ask the agent to compact conversation history.
  /hide <pane>            Hide a pane: files, pipeline, events, or all.
  /show <pane>            Show a pane (or all).
  /mouse on|off           Toggle mouse capture (off lets you copy text).
  /quit                   Exit.

Copying text
  When mouse capture is on, selecting text is blocked by the TUI so
  the wheel can scroll. To copy: either hold Shift (Linux/Windows) or
  Option (macOS) while dragging — most terminals honor the override —
  or run /mouse off to disable capture for the rest of the session.
  To launch with capture pre-disabled: ATLAS_TUI_MOUSE=off atlas tui
  (or pass --mouse off). The /mouse setting does not persist across
  TUI restarts — each launch re-reads the flag/env var.

Input modes
  message text            Send to agent (Enter).
  ! <cmd>                 Run as bash (no agent call). Same as /run.
  / <cmd>                 Slash command (this list).

Keys
  Enter                   Send message.
  Shift+Enter             Newline in input.
  Ctrl+C                  Cancel turn (or quit when idle).
  Ctrl+L                  Clear chat.
  Ctrl+T                  Cycle permission mode.
  Ctrl+R                  Resend last message.
  PgUp / PgDn             Scroll chat by ~10 rows.
  Mouse wheel             Scroll chat by ~3 rows.
  Ctrl+End / Ctrl+Home    Jump to bottom / top of chat.`

// handleSlash interprets a slash-prefixed input. Returns:
//
//	consumed = true  → the slash was a recognized command (handled here)
//	consumed = false → not a slash command; pass to /v1/agent as usual
//	cmd              → optional tea.Cmd to run async work (shell out)
//	quit             → true if the model should tea.Quit immediately
func (m *tuiModel) handleSlash(input string) (consumed bool, cmd tea.Cmd, quit bool) {
	if !strings.HasPrefix(input, "/") {
		return false, nil, false
	}

	// Echo the input as a "you" row so the chat reflects what was sent.
	m.chat = append(m.chat, chatMessage{Role: roleUser, Body: input})

	parts := strings.Fields(input)
	cmdName := parts[0]
	args := parts[1:]

	switch cmdName {
	case "/help", "/?":
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "help", Body: slashCommandHelp,
		})
		return true, nil, false

	case "/quit", "/exit":
		return true, tea.Quit, true

	case "/add":
		return true, m.cmdAddContext(args), false

	case "/drop":
		return true, m.cmdDropContext(args), false

	case "/context":
		return true, m.cmdListContext(), false

	case "/diff":
		return true, runShellCmd(m.workingDir, "/diff",
			append([]string{"git", "diff", "--color=never"}, args...)), false

	case "/commit":
		msg := strings.Join(args, " ")
		if msg == "" {
			msg = "atlas-tui: checkpoint"
		}
		return true, runShellCmd(m.workingDir, "/commit",
			[]string{"git", "commit", "-am", msg}), false

	case "/undo":
		return true, runShellCmd(m.workingDir, "/undo",
			[]string{"git", "reset", "--soft", "HEAD~1"}), false

	case "/run":
		if len(args) == 0 {
			m.chat = append(m.chat, chatMessage{
				Role: roleSystem, Meta: "error",
				Body: "/run requires a command (e.g. /run pytest -k snake)",
			})
			return true, nil, false
		}
		// Pass the rest as a single shell string so quoting/pipes work.
		return true, runShellCmd(m.workingDir, "/run",
			[]string{"bash", "-lc", strings.Join(args, " ")}), false

	case "/clear":
		m.chat = nil
		m.chatScroll = 0
		return true, nil, false

	case "/compact":
		// Ask the agent to compact via a synthetic user message. The
		// proxy's history-trimming kicks in at 12 messages; this lets
		// the user trigger an explicit summarization.
		compactMsg := "Summarize the conversation so far in 3-4 sentences and respond with only that summary, no tool calls."
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "compact",
			Body: "Asking agent to compact conversation…",
		})
		return true, m.sendChatCmd(compactMsg), false

	case "/hide":
		if len(args) == 0 {
			m.chat = append(m.chat, chatMessage{
				Role: roleSystem, Meta: "error",
				Body: "/hide files | pipeline | events | all",
			})
			return true, nil, false
		}
		m.applyPaneVisibility(args[0], true)
		return true, nil, false

	case "/show":
		if len(args) == 0 {
			m.chat = append(m.chat, chatMessage{
				Role: roleSystem, Meta: "error",
				Body: "/show files | pipeline | events | all",
			})
			return true, nil, false
		}
		m.applyPaneVisibility(args[0], false)
		return true, nil, false

	case "/mouse":
		// Toggle mouse capture so users can copy text without holding
		// Shift/Option. Defaults to "on" (capture); /mouse off disables
		// capture for the session, /mouse on re-enables it.
		want := ""
		if len(args) > 0 {
			want = strings.ToLower(args[0])
		}
		if want != "on" && want != "off" {
			m.chat = append(m.chat, chatMessage{
				Role: roleSystem, Meta: "error",
				Body: "/mouse on | off",
			})
			return true, nil, false
		}
		if want == "off" {
			m.chat = append(m.chat, chatMessage{
				Role: roleSystem, Meta: "mouse",
				Body: "mouse capture disabled — wheel-scroll won't work, but you can select/copy text normally. Run /mouse on to restore.",
			})
			return true, tea.DisableMouse, false
		}
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "mouse",
			Body: "mouse capture enabled — wheel scrolls chat. Hold Shift/Option to select text without disabling.",
		})
		return true, tea.EnableMouseCellMotion, false
	}

	// Unknown slash command — show help instead of sending to the
	// agent (a typo'd /diff shouldn't trigger an LLM call).
	m.chat = append(m.chat, chatMessage{
		Role: roleSystem, Meta: "unknown",
		Body: fmt.Sprintf("unknown command %q. Type /help for the list.", cmdName),
	})
	return true, nil, false
}

// cmdAddContext adds files to the in-context set. The set is sent
// alongside each /v1/agent call so the agent knows which files the
// user considers in-scope. Returns nil cmd — purely state mutation.
func (m *tuiModel) cmdAddContext(paths []string) tea.Cmd {
	if len(paths) == 0 {
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "error",
			Body: "/add requires at least one path",
		})
		return nil
	}
	if m.contextFiles == nil {
		m.contextFiles = map[string]bool{}
	}
	added := []string{}
	for _, p := range paths {
		if !m.contextFiles[p] {
			m.contextFiles[p] = true
			added = append(added, p)
		}
	}
	if len(added) == 0 {
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "context",
			Body: "no new files added (all already in context)",
		})
		return nil
	}
	m.chat = append(m.chat, chatMessage{
		Role: roleSystem, Meta: "context",
		Body: fmt.Sprintf("added to context: %s", strings.Join(added, ", ")),
	})
	return nil
}

func (m *tuiModel) cmdDropContext(paths []string) tea.Cmd {
	if len(paths) == 0 {
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "error",
			Body: "/drop requires at least one path",
		})
		return nil
	}
	dropped := []string{}
	for _, p := range paths {
		if m.contextFiles[p] {
			delete(m.contextFiles, p)
			dropped = append(dropped, p)
		}
	}
	if len(dropped) == 0 {
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "context",
			Body: "nothing dropped (none of those paths were in context)",
		})
		return nil
	}
	m.chat = append(m.chat, chatMessage{
		Role: roleSystem, Meta: "context",
		Body: fmt.Sprintf("dropped from context: %s", strings.Join(dropped, ", ")),
	})
	return nil
}

func (m *tuiModel) cmdListContext() tea.Cmd {
	if len(m.contextFiles) == 0 {
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "context",
			Body: "no files in context",
		})
		return nil
	}
	paths := make([]string, 0, len(m.contextFiles))
	for p := range m.contextFiles {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	m.chat = append(m.chat, chatMessage{
		Role: roleSystem, Meta: "context",
		Body: "files in context:\n  " + strings.Join(paths, "\n  "),
	})
	return nil
}

// applyPaneVisibility flips the hide flags for the named pane.
// Accepts: files, pipeline, events, all. Unknown names produce an
// error row in chat so the user can correct.
func (m *tuiModel) applyPaneVisibility(name string, hide bool) {
	verb := "shown"
	if hide {
		verb = "hidden"
	}
	switch strings.ToLower(name) {
	case "files":
		m.hideFiles = hide
	case "pipeline":
		m.hidePipeline = hide
	case "events":
		m.hideEvents = hide
	case "all":
		m.hideFiles = hide
		m.hidePipeline = hide
		m.hideEvents = hide
	default:
		m.chat = append(m.chat, chatMessage{
			Role: roleSystem, Meta: "error",
			Body: fmt.Sprintf("unknown pane %q. Use: files, pipeline, events, all.", name),
		})
		return
	}
	m.chat = append(m.chat, chatMessage{
		Role: roleSystem, Meta: "panes",
		Body: fmt.Sprintf("%s %s", name, verb),
	})
}

// runShellCmd shells out and returns a tea.Cmd that delivers the
// captured combined stdout/stderr as a slashResultMsg. Honors a
// 60-second deadline so a runaway command can't wedge the TUI.
func runShellCmd(workingDir, label string, argv []string) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
		cmd.Dir = workingDir
		out, err := cmd.CombinedOutput()
		return slashResultMsg{
			command: label,
			output:  strings.TrimRight(string(out), "\n"),
			err:     err,
		}
	}
}

// contextSuffix returns a string to append to the user's message so
// the agent sees the in-context file list. Empty if no files added.
//
// Format kept lightweight — just a single line listing the paths.
// The agent can then choose to read_file each one as needed.
func (m *tuiModel) contextSuffix() string {
	if len(m.contextFiles) == 0 {
		return ""
	}
	paths := make([]string, 0, len(m.contextFiles))
	for p := range m.contextFiles {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return "\n\n[atlas-tui context: " + strings.Join(paths, ", ") + "]"
}
