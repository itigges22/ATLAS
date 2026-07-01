# ATLAS CLI Guide

The ATLAS CLI launches the inference stack and drops you into an interactive
coding session. The canonical chat client is the native Bubbletea TUI
(`atlas tui`, introduced in PC-062). Plain `atlas` in an interactive
terminal launches the same TUI; pipe mode falls through to the built-in
`/solve` REPL.

---

## Launching

```bash
cd /path/to/your/project
atlas              # interactive: launches the TUI
atlas tui          # explicit form
echo "fix bug" | atlas   # pipe mode: routes through /solve
```

The top-level `atlas` binary also dispatches to non-TUI subcommands:

| Subcommand | Purpose |
|---|---|
| `atlas init` | First-run wizard: probes hardware, picks a model, writes `.env` + `secrets/api-keys.json`. |
| `atlas tier` | Hardware probe + tier classification (NVIDIA / AMD / Apple Silicon detection). `atlas tier list` shows the full tier table; `atlas tier fit` sizes the runtime (context / KV type / ubatch) for the configured model + GPU (see below). |
| `atlas doctor` | Install diagnostic. GPU runtime, container health, endpoint reachability. Prints each result as it completes; `--json` buffers into one machine-readable document. |
| `atlas model list \| recommend \| install \| install-artifacts \| verify \| remove` | Model registry operations. `recommend` names the best registry model for this hardware; `install --url <hf>` fetches an **unregistered** model (drop-in / BYO); `install-artifacts <name>` fetches a registered model's published lens + ASA artifacts. |
| `atlas onboard` | Guided drop-in for a new model: arch check, rebuild gate, lens-retrain guidance (see below). |
| `atlas bench` | Generate + self-label candidates for the loaded model (baseline benchmark). Feeds `atlas lens build --from-results` (see below). |
| `atlas lens check \| build \| publish` | Geometric Lens compat probe + per-model training (PC-057 / PC-058 — see below). |
| `atlas asa check \| build \| publish` | ASA control-vector compat probe + per-model training + publish (PC-061 — see below). |
| `atlas publish` | One-step publish: lens artifacts + ASA vector to HF, one registry PR covering both (PC-215). `--lens-only` / `--asa-only` delegate to the per-component flows. |
| `atlas compose <args...>` | `docker compose` passthrough with ATLAS's compose file set (base file + the backend overlay resolved from `ATLAS_BACKEND`). E.g. `atlas compose ps`, `atlas compose logs -f atlas-proxy`. |

`atlas --help` (or `-h`) prints the subcommand list. An unknown subcommand prints the same usage to stderr and exits 2.

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
┌──────────────────────────────────────────────────────────────────┐
│ Header                                                           │
│   ATLAS TUI · status · cwd · permission mode                     │
├──────────────────────────────────────┬───────────────────────────┤
│ Pipeline                             │                           │
│   live stage table from /events      │  Files                    │
├──────────────────────────────────────┤   workspace tree (depth 2)│
│ Chat                                 │   modified files marked   │
│   user + agent messages              │                           │
│   tool calls and results             │                           │
│   live LLM token stream              │                           │
├──────────────────────────────────────┤                           │
│ Events                               │                           │
│   raw typed-envelope log             │                           │
├──────────────────────────────────────┴───────────────────────────┤
│ Stats   stage · turn · ctx % · session · tools · events          │
├──────────────────────────────────────────────────────────────────┤
│ Message   chat (default) · ! bash · / command · ? help           │
└──────────────────────────────────────────────────────────────────┘
```

The **Files** sidebar appears when the terminal is ≥90 columns wide;
below that, the remaining panes stack vertically. **Pipeline**,
**Events**, and **Files** can each be hidden with `/hide <pane>`. See
[Panes](#panes) for what each region renders in detail.

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
| `/good` | 👍 the last completed pass — bank its writes as positive lens-training samples |
| `/bad` | 👎 the last completed pass — bank its writes as negative lens-training samples |
| `/review` | List the files the last pass wrote, with any per-file verdicts |
| `/deny <path> [reason]` | Mark one file from the last pass bad (a confident negative); submitted on the next `/good`/`/bad` |
| `/accept <path>` | Undo a `/deny` |
| `/redo <path> [reason]` | Ask the agent to regenerate a rejected file (reuses the `/deny` reason) |
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

`/good` and `/bad` rate the most recently completed pass. The proxy turns
that pass's writes into labeled, weighted lens-training samples (collected
under `ATLAS_LENS_DATA_DIR`). For finer control, `/review` lists the pass's
files and `/deny <path>` marks individual ones bad — so a thumbs-up pass with
one denied file banks the good files as positives and the denied one as a
confident negative (the per-file verdict overrides the pass thumbs). `/redo`
asks the agent to regenerate a rejected file. As samples accumulate, the TUI
shows a one-time **"🧠 Lens retrain available"** banner with the command to run
(`atlas lens retrain`), which retrains the lens on your own workloads. See
[CONFIGURATION.md](CONFIGURATION.md) (lens onboarding) for the full loop.

---

## Panes

### Files

Workspace tree to depth 2. Skips noisy directories (`.git`,
`node_modules`, `__pycache__`, `.venv`, `venv`, `dist`, `build`,
`target`, `.next`, `.nuxt`, `.idea`, `.vscode`, `.cache`,
`.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `__MACOSX`). Capped at
500 entries — overflow renders as `(+N more)`. Files modified by the
agent during the session are highlighted bold orange with a `●` prefix;
folders are bold cyan with `▸`. Re-scans every 4 s and immediately
after every `write_file`/`edit_file`/`delete_file` tool result.

### Pipeline

Live stage table fed by atlas-proxy's `/events` typed-envelope stream.
Each stage row shows an icon (⚙ running, ✓ done, ✗ failed), name,
status, duration, and a one-line detail. Stage names are emitted by the
proxy:

- `agent` — the whole `/v1/agent` turn
- `llm` — each LLM call (per turn)
- `tool` — each tool invocation
- `v3` — overall V3 pipeline (only when V3 fires for a write/edit)
- `v3:<phase>` — V3 sub-phases. `v3:plan` fires once per turn before
  the agent loop (see plan-mode rows in [Chat](#chat) below).
  Write/edit-triggered V3 adds `probe`, `plansearch`, `divsampling`,
  `sandbox_test`, `s_star`, etc.

### Chat

User and agent messages, tool calls and results, and live LLM token
streaming. Visual hierarchy:

- **Bright** (outputs the user cares about): user messages (`you`),
  finished assistant text (`agent`), executed tool calls (`→ tool`)
  and their results (`✓ tool` / `✗ tool` with elapsed time).
- **Dim grey italic** (machine internals): turn separators
  (`── turn N · ctx=K msgs ──`), LLM-call rows (`· llm · …`), V3
  internal LLM rows (`· v3 · …`, violet tint), planner rows
  (`plan` meta — see below), and other system metadata (mode
  changes, errors, V3 stage progress).

Plan-mode rows (when the planner ran for this turn — see
[ARCHITECTURE.md § Plan Mode](ARCHITECTURE.md#plan-mode-per-turn-pre-flight)
for mechanics):

- `plan` rows from `v3_plan` events — planner progress
  (`generating 3 candidate plans`, `candidate 1/3 (temp=0.3)`,
  `candidate 1 score=0.80`, `plan 1 won (score=0.80)`).
- Multi-line `plan_loaded` row — the full step list with glyphs
  (☐ unsatisfied, ✓ satisfied, ⚐ verify-step). A revision appends a
  new row tagged `plan rev N` and replaces the internal plan state,
  so subsequent `plan_adherence` rows count against the revised steps.
- `plan` adherence one-liners — `✓ s2 satisfied · edit_file (1/3)`
  fires when a tool call matches an unsatisfied step. Off-plan
  calls are silent (they only update internal state).
- `Plan revising (rev 1): <reason>` — the agent went off-plan past
  the threshold; the next `plan_loaded` replaces the plan.

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
| `default` | Read tools and surgical edits (`edit_file`, `ast_edit`) auto-allow; `write_file`, `delete_file`, `run_command`, and `stop_background` require user approval |
| `accept-edits` | As above + `write_file` auto-allow; `delete_file`, `run_command`, and `stop_background` still confirm |
| `yolo` | Auto-allow everything |

The exact gate is `Destructive: true` on the tool definition in
`proxy/tools.go`; `accept-edits` additionally auto-approves
`write_file` and `edit_file` (the latter is already non-destructive).

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
| `ATLAS_TUI_MOUSE` | `on` | Mouse capture at startup; `off` skips `WithMouseCellMotion` so native terminal select works without modifiers. Mid-session toggle via `/mouse on\|off`. Also exposed as the `--mouse` flag. |
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
| llama-server | 8080 | Local GGUF inference through llama.cpp |
| atlas-proxy | 8090 | Agent loop, tool execution, V3 routing, SSE event broker |
| v3-service | 8070 | V3 pipeline (PlanSearch, DivSampling, build verification, repair) |
| geometric-lens | 8099 | C(x)/G(x) energy scoring |
| sandbox | 30820 | Isolated code execution for V3 verification |

`atlas tui` only needs `atlas-proxy` reachable; the proxy fans out to
the other services internally.

---

## atlas model

Registry-aware model management.

```bash
atlas model list                        # known models + install/lens/asa status
atlas model recommend                   # best registry model for this hardware (tier probe + registry)
atlas model install <name>              # download a registry model (SHA-256 verified)
atlas model install --url <hf-gguf-url> # download an unregistered GGUF (drop-in / BYO)
atlas model install-artifacts <name>    # fetch a registered model's published lens + ASA artifacts
atlas model verify <name>               # re-hash an installed model against the registry
atlas model remove <name>               # delete a model file from the models dir
```

Notes:

- **Models dir resolution:** `--models-dir` flag → `ATLAS_MODELS_DIR` shell env → `ATLAS_MODELS_DIR` in the checkout's Docker `.env` → `<atlas_root>/models`. Relative values resolve against the ATLAS root (the compose deployment's frame of reference), not your cwd.
- **Interrupted downloads resume.** A partial `.part` file continues from where it stopped (HTTP Range). If the server reports the range is already complete (HTTP 416), the `.part` is verified (SHA-256 when registered, size otherwise) and promoted in place; a `.part` that fails verification is deleted with retry guidance.
- **`install-artifacts` exit codes:** `0` success, `1` error, `3` when the registry has no artifacts registered for direct download for that model — the command prints where they are published (HF repo) or how to train them locally (`atlas lens build` / `atlas asa build`).

---

## atlas tier fit (PC-208)

Sizes the llama-server runtime for a specific model on *your* GPU. Reads the
GGUF header (layer count, per-layer KV-head geometry, sliding-window layout)
and the GPU's VRAM, then solves for the largest context that keeps inference
**fully on-GPU**:

```bash
atlas tier fit                          # fit the model configured in .env
atlas tier fit models/other.gguf        # fit a specific GGUF
atlas tier fit --write                  # apply the result to .env
atlas tier fit --slots 2                # size for 2 parallel slots instead of 4
atlas tier fit --json                   # machine-readable ({model, meta, fit, env})
```

Example:

```
atlas tier fit — selected-model-Q4_K_M.gguf
  arch modelarch | 48 layers | 3840-dim | head_dim 512 | window 1024, per-layer mask (40/48 sliding)
  GPU: NVIDIA GeForce RTX 5060 Ti (15.9 GiB)
  budget: weights 6.77 + KV 3.88 + compute 2.05 + reserve 1.9 GiB of 15.93 GiB
  fit: ctx 131072 (32768/slot × 4), KV f16, ubatch 2048
```

The VRAM budget it solves under:

```
weights (file size × 1.02)
+ KV: global-attention layers × total ctx              (per-layer KV-head dims
      + sliding-window layers × (slots × window         and per-group head
                                 + ubatch)              widths from the GGUF)
+ compute buffer (~ubatch × n_embd × 280 bytes)  ← the term that OOMs first
+ 1.9 GiB reserve (CUDA context, graphs, fragmentation)
≤ VRAM
```

It prefers f16 KV and a large micro-batch, trading down (q8_0 KV, smaller
ubatch) only when that buys context, and caps at 32k per slot. With `--write`
it updates `ATLAS_CTX_SIZE`, `ATLAS_PARALLEL_SLOTS`, `ATLAS_KV_TYPE_K/V`,
`ATLAS_UBATCH`, and `ATLAS_BATCH` in `.env`; apply with
`docker compose up -d llama-server --no-deps --force-recreate`.

`--write` always targets the **ATLAS install's** `.env` (the path is printed
as `wrote …`), regardless of your current directory — same root resolution as
`atlas doctor` and `atlas bench`. If you run it from inside a *different*
compose project, it warns that the cwd's `.env` was not the one written.

If the model can't fit at even the minimum acceptable context (8k per slot,
q8_0), it says so, names the largest quant file size of this model's geometry
that *would* fit (at the current slot count and at `--slots 1`), and exits 1.
Pre-download sizing guidance lives in
[TROUBLESHOOTING.md § What fits on my GPU?](TROUBLESHOOTING.md#what-fits-on-my-gpu).
The server itself runs with `--fit off`, so an oversized config **refuses to
start** instead of silently demoting layers to CPU (the 5×-slower failure
mode this command exists to prevent).

Run it whenever `ATLAS_MODEL_FILE` or the GPU changes. `atlas onboard` prints
the fit recommendation automatically and flags a stale `.env`.

---

## atlas onboard

Guided drop-in for a new (often unregistered) model. Automates the *safe* parts
of bringing up a model and stops at the one step only the operator can do — the
inference-image rebuild — because a careless rebuild can drop ATLAS's custom
llama.cpp patches.

```bash
atlas onboard                       # onboard the model already pointed at in .env
atlas onboard --url <hf-gguf-url>   # download an unregistered model first
atlas onboard --url <url> --apply   # ...and write ATLAS_MODEL_FILE/NAME into .env without prompting
atlas onboard --no-start            # inspect current state; don't (re)start llama-server
```

What it does:

1. **Resolve** the model from `.env` (`ATLAS_MODEL_FILE`). With `--url`, fetches
   it first via `atlas model install --url`, then offers to write
   `ATLAS_MODEL_FILE` + `ATLAS_MODEL_NAME` into `.env` (interactive prompt;
   `--apply` writes without asking) — re-run `atlas onboard` afterwards to
   continue with the arch and lens checks. Also
   prints the runtime-fit recommendation for this model + GPU (`atlas tier fit`)
   and flags when `.env` sizing differs.
2. **Arch gate** — reads the GGUF architecture and confirms llama-server actually
   loaded it (starts it if needed). If the bundled llama.cpp doesn't know the
   architecture, it prints the rebuild command **and stops** — it never rebuilds
   for you. The message links to the TROUBLESHOOTING.md procedure that ensures
   you don't strip the `expose-hidden-states` (PC-202) patch when rebuilding.
3. **Lens check** — reports the model's embedding dim and whether `C(x)` needs
   retraining.
4. **Next steps** — prints the operator-driven `bench → retrain → asa build`
   sequence (bench is hours on a large model, so onboard guides rather than runs it).

| Exit | Meaning |
|---|---|
| 0 | Engine ready (model loads). Lens retrain is the remaining manual step. |
| 1 | Model file missing / not configured — wire `.env` or use `--url`. |
| 2 | **Rebuild required** — the image can't load this architecture; rebuild yourself, then re-run. |

Full walkthrough: [CONFIGURATION.md § Adding your own model](CONFIGURATION.md#adding-your-own-model-drop-in--unregistered).

---

## atlas bench

Runs the baseline benchmark against whatever model llama-server has loaded:
generates a candidate per task, executes it, and writes per-task results with
`code` + `passed` labels. This is the candidate-build step of model onboarding —
its output feeds `atlas lens build --from-results`. Connectivity (llama/lens
URLs) resolves from the deployment's config (`.env` on Docker, `atlas.conf` on
K3s); explicit `LLAMA_URL`/`RAG_API_URL` env vars override.

```bash
atlas bench --tasks 15                       # quick sanity subset
atlas bench --run-id mymodel_lens --tasks 200   # named run for the lens retrain
atlas bench                                  # full dataset (hours on a local model)
```

| Flag | Default | Meaning |
|---|---|---|
| `--tasks N` | `0` (all) | how many tasks to run |
| `--run-id NAME` | `bench_livecodebench_<ts>` | names the run; results land in `benchmark/results/<run-id>/` |
| `--strategy` | `random` | candidate selection (`lens`/`random`/`logprob`/`oracle`) |

On completion it prints the matching `atlas lens build --force --from-results …` command.
(Also available as `/bench` inside the TUI.)

**Interrupted runs resume.** Each task's result is written atomically as it
completes; re-running the same `atlas bench` command skips finished tasks
(`Resuming: N/total complete, M remaining`). Nothing is lost to an OOM kill,
reboot, or closed session. If the run reports fewer tasks than `--tasks`
requested, the dataset cache is a partial download — see
[TROUBLESHOOTING.md § Benchmark Issues](TROUBLESHOOTING.md#benchmark-issues).

---

## atlas lens (PC-057 / PC-058)

Geometric Lens compat probe + per-model training. Lets you bring a non-default GGUF and either verify it'll score with the existing C(x) artifacts or train fresh ones for it.

### `atlas lens check`

Cheap pre-flight against the running llama-server. No training, no model download — just probes `/embedding` and `/props` to confirm the model is Lens-compatible.

```bash
atlas lens check                       # probe whatever llama-server has loaded
atlas lens check <registry-name>        # probe a registry entry by name
atlas lens check /path/to/model.gguf   # probe an arbitrary file
atlas lens check --json                # machine-readable for scripts / CI
```

Verdict + exit code:

| Verdict | Exit | Meaning |
|---|---|---|
| `compat` | 0 | Artifacts exist and accept this model's embedding dim. Ready to score. |
| `needs-build` | 1 | Model loads but no cost_field.pt at the right dim. Run `atlas lens build`. |
| `incompatible` | 2 | Can't probe — llama-server unreachable, `/embedding` silent, etc. |

Reports the model's embedding dim, layer count, PC-202 hidden-states-patch status, the artifact dir it checked, and the artifact's own input dim. JSON mode produces a stable shape (`verdict`, `reason`, `probe.*`, `artifact_dir`, `artifact_dim`, `matched_model`, `exit_code`).

### `atlas lens build`

Trains fresh lens artifacts — **both halves** — for whichever model llama-server has loaded:

1. `cost_field.pt` — C(x), contrastive ranking loss (`train_cost_field`). Test AUC is evaluated every epoch; the best checkpoint is kept and training stops early once it plateaus.
2. `gx_xgboost.json` + `gx_weights.json` — G(x), a PCA(→128) + XGBoost correctness classifier (`train_gx`), fit on the same embeddings with stratified-CV AUC reporting. G(x)'s PCA is dimension-coupled to the model just like C(x), so both retrain together.

```bash
atlas lens build --from-results benchmark/results/<run-id>/v3_lcb/per_task
                                                   # train on THIS model's own candidates
                                                   # (the per-task output of `atlas bench`)
atlas lens build --samples path/to/labeled.json    # or: a hand-labeled training file
atlas lens build --samples ... --epochs 400        # tune training
atlas lens build --samples ... --force             # retrain even if compat artifact exists
atlas lens build --samples ... --dry-run           # extract embeddings, skip training + save
```

**`--from-results`** points at a benchmark results directory (the `per_task/` dir written by `atlas bench`); each task's `code` + `passed` becomes a training sample. This is the recommended path when onboarding a new model — C(x) learns the model's *own* pass/fail geometry. Tasks without generated code are skipped.

When the run directory also holds `telemetry/embeddings.emb`, the build
merges it in automatically: v3_runner banks every sandbox-tested
candidate's embedding + PASS/FAIL label as the bench runs, so a V3-mode
run contributes several labeled samples per task (probe + PlanSearch
fan-out + repair iterations) — the cheapest way to grow the training set.
Banked copies of the already-extracted samples are deduped numerically;
`--no-telemetry` trains on the results dir alone. (A baseline-mode bench
banks one candidate per task, so the merge adds little there.)

**`--samples` format** — JSON array (or JSONL) of `{"text": "...", "label": 0|1}` where `label=1` means the snippet represents *passing* / correct code and `label=0` means *failing*. Pull the canonical training set (V3 ablation traces with pass/fail labels) from `huggingface.co/datasets/itigges22/ATLAS`.

Minimums: at least 50 samples with both classes present (build refuses below this — a too-small C(x) actively mis-ranks). Test AUC below 0.70 emits a warning suggesting more samples or epochs.

Training runs host-side and needs PyTorch plus XGBoost/scikit-learn
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`,
`pip install xgboost scikit-learn` — CPU builds are enough). Samples
longer than the server's micro-batch (e.g. runaway candidates that hit the
generation cap) are embedded in line-boundary chunks and mean-pooled rather
than dropped; the build log notes each chunked sample.

Extracted embeddings are cached (keyed by text hash and dim), so re-running
a build — after growing the results dir, or to finish G(x) after installing
a missing dependency — only embeds the new samples. For `--from-results
<run>/v3_lcb/per_task` the cache lives at `<run>/v3_lcb/embeddings_cache.jsonl`
(beside the `per_task/` dir); for `--samples file.json` it's
`file.json.embcache.jsonl`. A model switch changes the embedding dim and
invalidates the cache automatically.

After a successful build:
1. `cost_field.pt`, `gx_xgboost.json`, `gx_weights.json`, both calibration files, and `model_identity.json` land in the artifact dir (default `geometric-lens/geometric_lens/models/`, override with `--artifact-dir`). The identity file prevents a same-width artifact from being reused with a different model.
2. Restart the lens service so it loads them: `docker compose restart geometric-lens`.
3. Re-run `atlas lens check` — should now report `compat`.
4. Run `atlas lens publish` (PC-059, below) to upload to HuggingFace + open a registry PR. Or, for private/manual flows, hand-edit `atlas/cli/commands/model_registry.py` to set `lens_status="supported"`.

### `atlas lens publish`

Uploads trained artifacts to a HuggingFace repo and generates a maintainer-reviewable PR body that adds the model to the ATLAS registry (PC-059, GH #101).

> **New to publishing?** See [docs/PUBLISHING.md](PUBLISHING.md) for the end-to-end walkthrough — HF account setup, token generation, what happens after submission, and troubleshooting. The reference below assumes you've already got `HF_TOKEN` set.

```bash
atlas lens publish <registry-name> --repo alice/atlas-lens-my-model
atlas lens publish <model> --repo <user>/<repo> --license mit
atlas lens publish <model> --dry-run            # hash + render PR body, don't upload
atlas lens publish <model> --skip-pr            # upload to HF, print PR body for manual paste
```

**Pipeline:**
1. SHA-256 + size of `cost_field.pt` for the PR's verification checklist.
2. `huggingface_hub` `create_repo` (idempotent) + uploads `cost_field.pt` + `metric_tensor.pt` if present + an auto-generated `README.md` model card.
3. Renders a registry-PR markdown body with a verification checklist + suggested Python diff for `atlas/cli/commands/model_registry.py`.
4. Tries `gh pr create --repo itigges22/ATLAS` if `gh` is installed + authenticated; otherwise prints the body for manual paste.

**Requirements:**
- `HF_TOKEN` env var (write-scope) — get one at https://huggingface.co/settings/tokens.
- `pip install huggingface_hub` on the host (already bundled in the lens container).
- License must be permissive for redistribution (apache-2.0 default; mit / bsd-3-clause also fine).

**`--dry-run` is the no-upload mode** — runs the SHA + PR-body rendering pipeline without touching HF or `gh`. Useful for previewing the PR body before committing to a public upload, or for private deployments that don't want to share artifacts.

---

## atlas asa (PC-061)

ASA control-vector compat probe + per-model training + publish. Same shape as `atlas lens`, different mechanics. Wraps `geometric-lens/asa_calibration/build_steering_vector.py` end-to-end so swap-in models can be calibrated without learning the underlying scripts.

### `atlas asa check`

Probes the running llama-server + the configured `ATLAS_CONTROL_VECTOR` to verify the vector's residual dim matches the model's embedding dim.

```bash
atlas asa check                 # probe whatever's loaded
atlas asa check --json          # machine-readable for CI / monitoring
```

Verdict + exit code:

| Verdict | Exit | Meaning |
|---|---|---|
| `compat` | 0 | Vector present + dim matches model. Ready for `--control-vector-scaled`. |
| `needs-build` | 1 | No vector, or dim mismatched (vector was trained for a different model). |
| `incompatible` | 2 | llama-server unreachable. |

Reports the vector's dim, layer count (from GGUF metadata), and the `model_hint` baked in by `build_steering_vector.py`. Resolves container-relative paths (`/models/x.gguf` on llama-server) to host-visible paths by trying `<atlas_root>/models/` and `$ATLAS_MODELS_DIR` in turn — running `atlas asa check` on the host needs no manual path translation.

Requires the `gguf` Python pkg on the host (`pip install gguf`) for the dim probe. Without it the verdict falls back to `compat: unverified` rather than failing — llama-server will refuse to load an incompatible vector at boot anyway, so the worst case is a clear error in container logs.

### `atlas asa build`

Trains a fresh ASA vector by running `build_steering_vector.py` inside the lens container (which has the PC-202 hidden-states client + numpy + the gguf writer). The script + contrast pairs are docker-cp'd in, the run executes there, and the output `.gguf` is copied back to the host.

```bash
atlas asa build                                  # train w/ bundled contrast_pairs.jsonl
atlas asa build --pairs custom.jsonl             # custom pairs (same {prompt, label} schema)
atlas asa build                                  # layer defaults to 75% of loaded model depth
atlas asa build --layer <index>                  # explicit extraction-layer override
atlas asa build --limit 50                       # smoke test (50 pairs, ~1 min)
atlas asa build --container atlas-geometric-lens-1   # override container name
atlas asa build --dry-run                        # stage but don't run
```

Full 1000-pair training run takes ~25 min on the canonical RTX 5060 Ti. Smoke-test (`--limit 50`) is the fast path for validating the build pipeline works end-to-end before committing to the full run.

After build:
1. The `.gguf` lands at `<artifact-dir>/ast_edit_steering.gguf` (default: dirname of `ATLAS_CONTROL_VECTOR`).
2. Restart llama-server so it picks up the new vector: `docker compose restart llama-server` (the vector lives on the bind-mounted models dir — no image rebuild involved).
3. Verify with `atlas asa check`.

### `atlas asa publish`

Same shape as `atlas lens publish` — uploads the trained `.gguf` to a HuggingFace repo and generates a maintainer-reviewable registry PR. Full contributor walkthrough lives in [docs/PUBLISHING.md](PUBLISHING.md).

```bash
atlas asa publish <registry-name> --repo alice/atlas-asa-my-model
atlas asa publish --dry-run                      # render PR body, no upload
atlas asa publish --vector path/to/v.gguf        # custom vector path
```

Required: `HF_TOKEN` env var (same as `atlas lens publish`). PR body documents the residual dim, layer the vector was trained at, GGUF model hint, and the suggested registry diff for adding `asa_status="supported"` to the model entry.

### TUI calibration badge

When you launch `atlas` (the TUI), the Pipeline pane title gets a compact Lens/ASA badge fetched from the proxy's `/v1/calibration/status`:

```
┌ Pipeline   Lens ✓   ASA ⚠ ─────────────────────────────┐
```

`✓` = supported, `⚠` = no-artifacts / dim-mismatch / missing vector, `✗` = unreachable / incompatible, `?` = unknown verdict. If the proxy is reachable but the lens hint asks you to run `atlas lens check` or `atlas asa check`, the badge gives you a one-glance prompt — the full diagnostic stays in those CLI commands' output.

### Prereqs

Both subcommands require a running `llama-server`. `atlas lens check` reuses the same URL resolution as the lens service (`ATLAS_LLAMA_URL` → `LLAMA_EMBED_URL` → `LLAMA_URL` → `http://localhost:8080`). The PC-202 hidden-states patch (baked into `inference/Dockerfile.v31` and `Dockerfile.rocm`) is required for G(x) metric-tensor training but not for C(x) — `check` reports its presence as informational.

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
cd tui
go build -o ~/.local/bin/atlas-tui .
```

### Wheel scroll doesn't work in tmux

tmux intercepts mouse events. Either enable mouse passthrough in tmux
(`set -g mouse on`) or use `PgUp`/`PgDn` instead.

### V3 doesn't fire on small files

By design: V3 only fires for files that look like meaningful code. The
trigger rule (see `classifyFileTier` in `proxy/tools.go`):

- Config files by name (`package.json`, `tsconfig.json`, `Dockerfile`, …)
  → always T1 (direct write).
- Data extensions (`.json`, `.yaml`, `.toml`, `.csv`, `.xml`, `.env`),
  style files (`.css`, `.scss`, `.less`), prose (`.md`, `.txt`, `.rst`),
  and shell scripts (`.sh`, `.bash`) → always T1.
- Anything under 10 lines → T1 (nothing for V3 to meaningfully
  diversify on).
- ≥10 lines and either (a) `hasLogicIndicators` returns true (2+ matches
  across 9 pattern families — function/method, control flow, error
  handling, Flask/FastAPI, Express/Node, React state, validation,
  database, JSX) or (b) the extension is in the code/markup set
  (`.py`, `.go`, `.rs`, `.ts`, `.tsx`, `.js`, `.jsx`, `.c`, `.cpp`,
  `.cc`, `.h`, `.hpp`, `.java`, `.kt`, `.swift`, `.rb`, `.php`,
  `.vue`, `.svelte`, `.html`, `.htm`) → T2 (V3 pipeline).
- Unknown extensions → T1.

### "encoding prompt…" lingers for >30 s

Llama.cpp doesn't flush HTTP response headers until the first decoded
token, so "header time" = "prompt eval time". Long conversation
histories (8K+ tokens) can take ~60 s of prompt eval before the first
token arrives. As of May 2026 the proxy runs with no
`ResponseHeaderTimeout` so long V3 chains (which can spend several
minutes before the first SSE frame) complete instead of being killed
mid-flight. If "encoding prompt…" sits for many minutes, the prompt is
genuinely too big — `/compact` to summarize.

---

## Building a non-TUI client

`atlas-proxy`'s `/v1/agent`, `/events`, and `/cancel` endpoints are the
public client contract. Anything that speaks SSE can be a chat client.
See [API.md § Building a non-TUI client](API.md#building-a-non-tui-client) for the protocol and a minimal Python example. PC-063 tracks the full spec writeup.
