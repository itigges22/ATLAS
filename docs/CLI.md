# ATLAS CLI Guide

The ATLAS CLI launches the inference stack and drops you into an interactive
coding session. The canonical chat client is the native Bubbletea TUI
(`atlas tui`, introduced in PC-062). Plain `atlas` in an interactive
terminal launches the same TUI; pipe mode falls through to the built-in
`/solve` REPL.

<p align="center">
  <img src="images/ATLAS_CLI.png" alt="ATLAS CLI" width="600"/>
</p>

---

## Launching

```bash
cd /path/to/your/project
atlas              # interactive: launches the TUI
atlas tui          # explicit form
echo "fix bug" | atlas   # pipe mode: routes through /solve
```

`atlas` does the right thing automatically:

1. **Locates the `atlas-tui` binary** on `$PATH` or in `~/.local/bin`.
2. **Builds from source** in `tui/` if the binary is missing and Go
   1.24+ is available. (~10 s on first run.)
3. **Ensures atlas-proxy is running** via `_ensure_proxy()`. If the
   proxy's `/workspace` bind-mount doesn't already cover your current
   directory, the wrapper force-recreates the proxy container with the
   correct mount (~5 s) so tool calls can read and write your files.
4. **Execs the TUI** with `--proxy http://localhost:8090` and a debug
   log path under `~/.cache/atlas-tui/debug.log`.

```bash
atlas tui                                # default proxy at localhost:8090
atlas tui --proxy http://other-host:8090 # remote proxy
atlas tui --log /tmp/atlas-tui.log       # custom debug-log path
ATLAS_TUI_LOG=off atlas tui              # disable debug logging
```

If the binary is missing **and** Go is unavailable, the launcher prints
install instructions and exits.

---

## Layout

```
┌─ Header ─────────────────────────────────────────────────────────────────┐
│ ATLAS TUI   ⠹ Pondering · cwd:~/projects/snake · default                 │
├─ Pipeline ───────────────────────────────────┬──────── Files ────────────┤
│ ⚙  llm           RUN   2.3s  turn=1          │ ● snake                   │
│ ⚙  v3:probe      RUN   1.1s  generating cand │ ▸ tui/                    │
│ ✓  tool          OK    12ms  write_file      │ ▸ docs/                   │
├─ Chat ───────────────────────────────────────┤   ● index.html            │
│ you                                          │   README.md               │
│   build me a SaaS landing-page mockup        │                           │
│ ── turn 1 · ctx=2 msgs ──                    │                           │
│ · llm · model replied · 8421 tok · 26.0s     │                           │
│ → tool · write_file                          │                           │
│   path=index.html                            │                           │
│ ✓ tool · ← write_file  · 0.8ms               │                           │
│ · v3 · model replied · 1024 tok · 9.3s       │                           │
│   ⠹ Brewing…  (Ctrl+C to cancel)             │                           │
├─ Events ─────────────────────────────────────┤                           │
│ 14:32:01  stage_start  llm    turn=1         │                           │
│ 14:32:08  stage_end    llm    1188 tokens    │                           │
│ 14:32:08  tool_call    tool   write_file     │                           │
├─ ● llm  turn:1  ctx:8.5k/32k (26%)  session:9.5k  tools:1✓/0✗  events:42─┤
┌─ Message ────────────────────────────────────────────────────────────────┐
│   Type a message · ! for bash · / for command · ? for help               │
└──────────────────────────────────────────────────────────────────────────┘
```

The **Files** sidebar appears on the right when the terminal is ≥90 cols
wide. The pipeline, events, and files panes can each be hidden via slash
commands.

---

## Input modes

The message box has three modes, distinguished by border color and the
hint row above it:

| First char | Mode | Border | Behavior |
|---|---|---|---|
| _(none)_ | chat | cyan | Sent to `/v1/agent` as a normal message |
| `!` | bash | red | Run as `bash -lc <cmd>` in the working dir; output appears as a system row |
| `/` | command | purple | Slash command; dropdown row above input shows matching commands |

Switching modes is just typing the trigger character — the border flips
and the hint row appears immediately. Backspace past the trigger char
to return to chat mode.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` | Send message / run bash command / fire slash command |
| `Shift+Enter` | Insert a newline (multi-line input) |
| `Ctrl+L` | Clear chat history |
| `Ctrl+T` | Cycle permission mode (default → accept-edits → yolo) |
| `Ctrl+R` | Re-send the last message |
| `Ctrl+C` | First press cancels the in-flight turn; second press exits |
| `Ctrl+D` | Exit immediately |
| `PgUp` / `PgDn` | Scroll chat by 10 rows |
| `Mouse wheel` | Scroll chat by 3 rows |
| `Ctrl+Home` | Jump to top of chat |
| `Ctrl+End` | Jump to bottom (resume auto-follow) |

Bracketed paste is enabled by default — pasted code arrives as a single
input event, so newlines in pasted text don't trigger a premature send.

### Copying text from the TUI

Mouse capture is on by default. Drag-highlight inside any pane (chat,
events, pipeline, files); on release, the highlighted text is auto-copied
to your clipboard and a transient toast (`✓ copied N chars from <pane>`)
appears in the header for ~2.5s. OSC52 fallback covers SSH sessions. No
chat row gets pushed for the copy — it's pure overlay UX.

If your terminal handles selection itself, you can also:

1. **Hold Shift (Linux/Windows) or Option (macOS)** while dragging.
2. **`/mouse off`** to disable capture for the rest of the session;
   wheel-scroll stops working but native terminal select returns.
   `/mouse on` re-enables.

For programmatic copy of recent chat output use `/copy [N]` (defaults to
the last message; pass an integer for the last N messages).

---

## Slash commands

| Command | Description |
|---|---|
| `/help` | Show in-TUI help with the full keymap and command list |
| `/add <path>` | Add a file to the agent's working context (path-only — agent reads on demand) |
| `/drop <path>` | Remove a file from the working context |
| `/context` | List files currently in context |
| `/diff [path]` | Show `git diff` (optionally for a specific path) |
| `/commit [msg]` | Stage all changes and create a commit (default msg if omitted) |
| `/undo` | `git reset --soft HEAD~1` — revert the last commit, keep changes |
| `/run <cmd>` | Run a shell command in the working dir; output appears in chat |
| `/clear` | Clear chat history (session token counter is preserved) |
| `/compact` | Ask the agent to summarize the conversation in 3-4 sentences |
| `/hide <pane>` | Hide a pane: `files`, `pipeline`, `events`, or `all` |
| `/show <pane>` | Show a pane (or `all`) |
| `/mouse on\|off` | Toggle mouse capture (off lets you copy text) |
| `/copy [N]` | Copy the last N chat messages (default 1) to clipboard via OSC52 |
| `/yank [N]` | Alias for `/copy` |
| `/quit` | Exit (same as `Ctrl+D`) |

The `/add /drop /context` set is TUI-side state — file paths are
appended to outgoing messages as a hint
(`[atlas-tui context: foo.go, bar.go]`) so the agent can `read_file`
them on demand. No file content is sent eagerly.

---

## Panes

### Files

Workspace tree to depth 2. Skips noisy directories (`.git`,
`node_modules`, `__pycache__`, `.venv`, `venv`, `dist`, `build`,
`target`, `.next`, `.idea`, `.vscode`, `.cache`, `.pytest_cache`,
`.mypy_cache`, `.ruff_cache`). Capped at 500 entries — overflow renders
as `(+N more)`. Files modified by the agent during the session are
highlighted bold orange with a `●` prefix; folders are bold cyan with
`▸`. Re-scans every 4 s and immediately after every
`write_file`/`edit_file`/`delete_file` tool result.

### Pipeline

Live stage table fed by atlas-proxy's `/events` typed-envelope stream.
Each stage row shows an icon (⚙ running, ✓ done, ✗ failed), name,
status, duration, and a one-line detail. Stage names are emitted by the
proxy:

- `agent` — the whole `/v1/agent` turn
- `llm` — each LLM call (per turn)
- `tool` — each tool invocation
- `v3` — overall V3 pipeline (only when V3 fires for a write/edit)
- `v3:<phase>` — V3 sub-phases (`probe`, `plansearch`, `divsampling`,
  `sandbox_test`, `s_star`, etc.)

### Chat

User and agent messages, tool calls and results, and live LLM token
streaming. Visual hierarchy:

- **Bright** (outputs the user cares about): user messages (`you`),
  finished assistant text (`agent`), executed tool calls (`→ tool`)
  and their results (`✓ tool` / `✗ tool` with elapsed time).
- **Dim grey italic** (machine internals): turn separators
  (`── turn N · ctx=K msgs ──`), LLM-call rows (`· llm · …`), V3
  internal LLM rows (`· v3 · …`, violet tint), and other system
  metadata (mode changes, errors, V3 stage progress).

During an LLM call the dim row fills in token-by-token. For
`write_file` calls, partial JSON is unescaped on the fly so you see
actual indented HTML/code being generated. Display caps at the last 80
lines so very long generations don't churn the renderer.

A "thinking…" spinner with rotating verbs (Pondering, Cogitating,
Brewing, Conjuring, Synthesizing, Mulling, …) sits at the bottom of the
chat box during a turn. Word changes every ~3 s.

### Events

Compact log of the raw `/events` envelope stream — one line per event
with timestamp, type, stage, and a short summary. Useful for debugging
the proxy↔TUI protocol.

### Stats

One-line strip below the events pane:

- **Active stage** (`● llm`, `● v3:probe`)
- **Turn counter** (`turn:1`)
- **Context utilization** (`ctx:8.5k/32k (26%)`) — color-coded ≥50% (orange),
  ≥80% (red). Updates live during decode.
- **Session-wide token count** (`session:9.5k`)
- **Tool counters** (`tools:3✓/0✗`)
- **Event counter** (`events:42`)

---

## Permission modes

Cycle with `Ctrl+T`:

| Mode | Behavior |
|---|---|
| `default` | Read tools auto-allow; write/edit/delete and `run_command` require user approval |
| `accept-edits` | Auto-allow read + write_file + edit_file; still confirm `run_command` and `delete_file` |
| `yolo` | Auto-allow everything |

The current mode shows in the header. Approval prompts appear in chat
as `permission_request` rows.

---

## Cancelling a turn

Each `/v1/agent` POST is tagged with a `session_id`. On `Ctrl+C` the TUI
cancels the local `context.Context` (closing the TCP connection) **and**
POSTs `/cancel` with the same `session_id` as defense-in-depth, in case
a reverse proxy buffers the disconnect. The proxy's agent loop watches
`ctx.Done()` and exits at the next turn boundary. The cancel propagates
through to llama-server (PC-036).

---

## Debug log

The TUI mirrors every event it receives to an append-only log so you
can review what happened after the fact (alt-screen makes copying out
of the live view impractical).

```bash
tail -f ~/.cache/atlas-tui/debug.log
```

Each line is a JSON-tagged record:
`HH:MM:SS.mmm category:subject {fields}`. Categories are `session`,
`user` (input events), `turn` (turn lifecycle), `chat` (every
chatStreamMsg type except `llm_token` to keep the file readable), `event`
(every typed envelope), and `slash` (slash command dispatch + result).

Override the path via `--log <path>` or `$ATLAS_TUI_LOG`. Set
`ATLAS_TUI_LOG=off` to disable.

---

## Workspace alignment

The proxy executes file operations against `/workspace` inside its
container, which is bind-mounted to a directory on host disk (set in
`docker-compose.yml`). For tool calls to land in your project, that
mount has to point at the directory you're working in.

`atlas tui` aligns this automatically:

1. On startup, `_ensure_proxy()` checks whether the proxy's existing
   `/workspace` mount covers `os.getcwd()`.
2. If not, it force-recreates the `atlas-proxy` container with
   `ATLAS_PROJECT_DIR=$(pwd)` so the bind mount points at your cwd.
   This takes ~5 s.
3. The proxy itself overrides any `working_dir` field in `/v1/agent`
   requests with the container-internal `/workspace` path, so the
   agent's `read_file`/`write_file` calls always resolve correctly.

If you write code from one shell and `atlas tui` is running in another
that's pointing at a different directory, restart the TUI in the right
cwd to re-align.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ATLAS_PROXY_URL` | `http://localhost:8090` | Default `--proxy` value |
| `ATLAS_TUI_LOG` | `~/.cache/atlas-tui/debug.log` | TUI debug log path; set `off` to disable |
| `ATLAS_TUI_STARTUP_NOTE` | _(unset)_ | Initial system message inserted at startup (used by the Python wrapper to surface workspace warnings) |
| `GLAMOUR_STYLE` | `dark` | Markdown rendering style for assistant text |
| `ATLAS_AUTO_WORKSPACE` | `1` | Set `0` to disable auto-realign of the proxy's bind mount |

See [CONFIGURATION.md](CONFIGURATION.md) for the full set of variables
that affect the proxy and inference stack.

---

## Stack overview

The TUI is one of several services. See [ARCHITECTURE.md](ARCHITECTURE.md)
for the full picture; the short version:

| Service | Port | Role |
|---|---|---|
| llama-server | 8080 | Local LLM inference (llama.cpp / Qwen3.5-9B-Q6_K) |
| atlas-proxy | 8090 | Agent loop, tool execution, V3 routing, SSE event broker |
| v3-service | 8070 | V3 pipeline (PlanSearch, DivSampling, build verification, repair) |
| geometric-lens | 8099 | C(x)/G(x) energy scoring |
| sandbox | 30820 | Isolated code execution for V3 verification |

`atlas tui` only needs `atlas-proxy` reachable; the proxy fans out to
the other services internally.

---

## Troubleshooting

### TUI renders, but the file pane is empty

You're probably running from a directory the proxy's `/workspace` mount
doesn't cover. Check:

```bash
docker inspect atlas-atlas-proxy-1 --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}'
```

The output should match your `pwd`. If not, exit and restart `atlas tui`
from the right directory; the wrapper auto-realigns on launch.

### "atlas-tui binary not found and Go is not available"

Install Go 1.24+ from [https://go.dev/dl/](https://go.dev/dl/), or
build manually:

```bash
cd atlas-tui
go build -o ~/.local/bin/atlas-tui .
```

### Wheel scroll doesn't work in tmux

tmux intercepts mouse events. Either enable mouse passthrough in tmux
(`set -g mouse on`) or use `PgUp`/`PgDn` instead.

### V3 doesn't fire on small files

By design: V3 only fires for files ≥150 lines (HTML/JSX/TSX/Vue/Svelte)
or ≥50-line files with code-logic indicators. Short config/data files
go through the direct write path. See `classifyFileTier` in
`proxy/tools.go`.

### "encoding prompt…" lingers for >30 s

Llama.cpp doesn't flush HTTP response headers until the first decoded
token, so "header time" = "prompt eval time". Long conversation
histories (8K+ tokens) can take ~60 s of prompt eval before the first
token arrives. The proxy's `ResponseHeaderTimeout` is 10 minutes; if
you hit that, the prompt is genuinely too big — `/compact` to summarize.

---

## Building a non-TUI client

`atlas-proxy`'s `/v1/agent`, `/events`, and `/cancel` endpoints are the
public client contract. Anything that speaks SSE can be a chat client.
See [API.md](API.md#building-a-client) for the protocol and a minimal
Python example. PC-063 tracks the full spec writeup.
