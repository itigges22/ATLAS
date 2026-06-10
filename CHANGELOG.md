# Changelog

> This changelog is maintained as a best-effort summary; for line-level detail and any gaps, see the commit history (`git log`) or the GitHub PR list.

## Unreleased

### V3.2 — RPG-style architecture-first planning (#120, experimental, opt-in)
- New `ATLAS_RPG_PLANNING` flag (default **off**) enables repository-level,
  plan-then-fill planning ahead of the existing problem-level PlanSearch:
  - **Wavelet substrate** (`v3-service/wavelet/`) — a faithful, dependency-free
    Python port of [wavescope-mcp](https://github.com/yogthos/wavescope-mcp)
    (Ricker CWT, structural signal, multi-resolution bands, project decomposition,
    peak-diff). Numeric parity with upstream is golden-tested.
  - **Repository Planning Graph** (`v3-service/rpg.py`, [arXiv:2509.16198](https://arxiv.org/abs/2509.16198)) —
    two-stage construction (proposal capability tree → implementation files +
    signatures + data-flow edges), graph validation/scoring, and a topological
    projection to the existing flat `Plan` so the agent loop is unchanged. The
    proposal stage is seeded with the wavelet coarse band on existing repos.
  - **Graph-guided generation** — each node's planned interface (signatures,
    edges) threads into its `/v3/generate` call (`proxy/rpg.go`), so the existing
    PlanSearch ([arXiv:2409.03733](https://arxiv.org/abs/2409.03733)) fills a node
    whose architecture is already pinned.
  - **Structural verification + drift** — the candidate veto now rejects code
    that doesn't realize its planned signatures; post-write drift detection
    surfaces the affected downstream subgraph for re-planning.
  - **Offline metrics** — `v3-service/rpg_eval.py` scores RPG artifacts for CI /
    benchmark summaries.
  - Strictly additive: with the flag off, planning and generation are unchanged.
  - Design + phased status: `docs/reports/RPG_WAVELET_PLANNING_V3_2.md`. Credit
    idea + framing to Dmitri Sotnikov (@yogthos), author of wavescope-mcp.

### Removed
- Removed dead `ATLAS_USE_FOX` code paths in benchmark runner (#22)

### Aider removed
- `proxy/aider_format.go` (whole-file format translator), `handleChatCompletions` + `handleStreamingChat`, and the OpenAI-compat agent-loop wrapping are all deleted (~2000 lines). `/v1/chat/completions` on the proxy is now a transparent passthrough to llama-server via the catch-all handler.
- `.aider.model.settings.yml`, `.aider.model.metadata.json`, the `.aider*` `.gitignore` exceptions, and the `_find_aider`/`launch_aider` paths in `atlas/cli/repl.py` are all gone. Bare `atlas` (interactive tty) now launches the TUI by default; pipe mode falls through to the built-in `/solve` REPL.
- Proxy launcher (`atlas/cli/repl.py`) now reaps any pre-existing `atlas-proxy-v2` process before spawning a fresh one and redirects proxy stdout/stderr to `~/.cache/atlas/proxy.log` instead of `/dev/null`. Closes the "old binary in memory after rebuild" foot-gun.

### Bubbletea TUI (PC-062)
- New `atlas tui` subcommand launches a native Bubbletea terminal UI as the canonical chat client (and is now the default for plain `atlas`)
- Five-pane layout: header (proxy/cwd/mode/spinner) + pipeline (live V3 stage table from `/events`) + chat (glamour-rendered markdown + inline tool calls) + events log + stats strip + textarea input
- Hotkeys: Enter send, Shift+Enter newline, Ctrl+L clear, Ctrl+T cycle permission mode, Ctrl+R resend last, Ctrl+C cancel turn / quit, Ctrl+D quit
- Slash commands inside the TUI: `/add /drop /context /diff /commit /undo /run /help /quit`
- New atlas-proxy `POST /cancel` endpoint indexed by `session_id` — TUI cancels the in-flight `/v1/agent` turn on Ctrl+C as defense-in-depth alongside TCP disconnect
- 43 atlas-tui Go tests + 4 atlas-proxy `/cancel` tests, all green under `go test -race`
- `tui/` is a standalone Go module (`github.com/itigges22/atlas-tui`) — depends on bubbletea, lipgloss, bubbles, glamour

### Documentation
- Added multilingual documentation: Simplified Chinese (zh-CN), Japanese (ja), Korean (ko) for README, SETUP, and TROUBLESHOOTING
- Added language selector badges to README
- Added star history chart to Latest News section
- Rewrote README contributing section to encourage issue reports and community feedback
- Fixed V3_1_STATUS.md false claims about speed optimizations that were never applied to code
- Documented RDNA4 (RX 9070 / 9070 XT, gfx1200/gfx1201) ROCm 7.x setup in SETUP.md and TROUBLESHOOTING.md — requires `ATLAS_ROCM_TAG=7.2.3-complete`; `ATLAS_HSA_OVERRIDE_GFX_VERSION` must stay unset (#119, thanks @Kaihui-AMD)
- Corrected stale Metal/macOS docs: the macOS hybrid Metal path (#32) is now documented as shipping across README, SETUP.md, CONFIGURATION.md, and ARCHITECTURE.md (was mislabeled "V3.1.2 planned"); rewrote ARCHITECTURE.md §8.4 to describe the actual hybrid (native llama-server + Docker) rather than the never-shipped pure-native install
- Restructured the README roadmap into V3.1.1 (hardware reach, landed), V3.1.2 (BYO-model + ROCm-on-K8s), and V3.2 (planning phase #120, structural+wavelet reasoning #39, reasoning-with-sampling #9), with a help-wanted backlog — all sourced from open issues
- De-staled user-facing CLI strings: `atlas init` and `atlas tier` no longer print "Metal — V3.1.2 planned"; they report Metal as the supported macOS hybrid path (#32) — strings/comments only, no logic change
- Synced zh-CN / ja / ko translations (README + SETUP.md) to the corrected English: Metal/macOS shown as shipping, multi-vendor GPU support table, V3.1.1/V3.1.2/V3.2 roadmap, and fixed NVIDIA-only requirements rows and SETUP_MACOS.md link paths

### Code Accuracy Audit
- Audited and corrected comments across 72 files for V3.0.1 accuracy
- Updated model references: Qwen3-14B to Qwen3.5-9B, embedding dimensions 5120 to 4096
- Renamed service references: rag-api to geometric-lens, Fox to llama-server
- Corrected G(x) XGBoost status: deployed and active (was incorrectly described as removed)
- Fixed normalization comments from "Fox 9B" to "Qwen3.5-9B C(x)"
- Marked legacy Fox code paths as unused in benchmark runner and geo_learning

### Test Fixes
- Fixed embedding dimensions in test fixtures (5120 to 4096)
- Fixed geometric-lens port in test conftest (8001 to 8099)
- Updated DivSampling test assertions to match actual 4+4+4 perturbation counts
- Corrected G(x) cost field parameter count: ~2.16M / 8.3MB (was ~2.7M / 10MB)
- Finished the 3.0.1 api-portal cleanup: removed `tests/integration/test_e2e_flow.py` and `tests/integration/test_e2e_training.py` (616 lines). These depended on the `test_api_key` fixture which calls the deleted api-portal service, so every test in them errored on session setup. The 3.0.1 changelog claimed this cleanup was done but these two files survived it.
- `test_empty_messages_handled` (`tests/infrastructure/test_llm.py`) now accepts 200/400/422/500. Current llama.cpp returns 500 for empty messages array; the test was hard-coded to 200 and broke against newer llama.cpp builds.
- PC-061 step B: implemented `_emit_event`, `_classify_stage`, `_logical_stage` in `v3-service/main.py`. The test file (`tests/v3-service/test_event_emission.py`) was committed in c5216be ("Install observability") but the implementation never landed, leaving the test red on dev. The contract is now satisfied: legacy `{stage, detail}` frame always emitted, typed envelope opt-in, suffix-based stage classification (`_pass`/`_skip`/`_done` → stage_end success=true, `_failed` → stage_end success=false, `_error` → error event, `_retry` → fresh stage_start), and stage_start→stage_end pairing via logical-name parent_id + duration_ms.

### Repo restructure
- Renamed `atlas-tui` → `tui` and `atlas-proxy` → `proxy` at the repo level; moved ablation data under `docs/reports`. 362 reference updates across the tree.

### Phase 0: first-run installer + model wizard
- New `atlas init` command (`atlas/cli/commands/init.py`): interactive first-run wizard that probes hardware, picks the right tier (T0/T1/T2/T3), recommends a model, writes `~/.atlas/config.yaml`.
- New `atlas model` command (`atlas/cli/commands/model.py`) with `list` / `verify` / `add` / `remove` subcommands; backed by `model_registry.py` (`add`/`get`/`list` with SHA verification) and `model_recommendations.py` (per-tier defaults, split out from `tier.py` in PC-055.2).
- `atlas/cli/events.py` (PC-061 step A): typed-event SSE protocol — `Event` dataclass, `parse_envelope`, `iter_events`, suffix-based stage classification. Schema documented in `docs/PROTOCOL.md`. Producer-side helpers in v3-service landed as PC-061 step B (see Test Fixes above).
- `atlas doctor` extended for the same hardware probe used by the wizard.

### Install + bootstrap hardening
- Hardened fresh-VM install path against partial failures across RHEL 9, Ubuntu, Rocky; `curl … | bash` and `curl … | sudo bash` both work.
- Auto-install NVIDIA driver libraries on RHEL 9 and put the Python CLI on `$PATH`.
- Bootstrap now installs Go and pre-builds `atlas-tui` so first-run latency is download-bound, not compile-bound.

### CI: lint + security + cross-distro
- Added ruff (Python lint) and CodeQL (security scan) as GitHub workflows.
- New PR-time test job that runs the full Python suite against a cross-distro install matrix (Ubuntu 22.04 / 24.04 / Rocky 9).
- Fixed pip PEP 660 friction, Rocky curl conflict, and a CLI-wizard GPU-mock path that was breaking the matrix.

### PC-159: surgical-edit gate (proxy)
- New gate in `proxy/agent.go` that refuses an `edit_file` when the proposed change would rewrite more than a configured fraction of the target file. Forces the model to pick the right tool (`write_file` for new files, `ast_edit` for structural rewrites, `edit_file` only for actual surgical patches).

### Chat history threading (proxy + tui)
- `/v1/agent` now accepts full prior chat history from the TUI, replacing the per-call stateless wrapper. Assistant turns are re-wrapped in a JSON envelope so the proxy can tell user messages from prior model turns when rebuilding context.

### Plan mode (May 5)
- New `/v3/plan` endpoint on v3-service generates a structured plan (steps + verify step + adherence score) before the agent loop begins; Qwen3 reasoning extraction fixed in the same commit.
- `proxy/agent.go` consumes the plan via a plan bridge, an agent-loop hook that pins the current step into each request, and an adherence gate that flags reasoning that drifts from the active step.
- TUI renders `plan_loaded` / `plan_adherence` / `plan_revise` events live (`tui/commands.go`, `tui/model.go`).
- New docs: `docs/PLAN_MODE.md`, `docs/PROTOCOL.md`.

### Proxy reliability (May 5)
- Output sanitiser strips reasoning preambles and dangling JSON fragments from model responses before parsing.
- Shell-op gate refuses dangerous `rm -rf /` style commands and the `bash -c` bypass route.
- System prompt hardened: clearer tool-use rules, fewer hallucinated fields.
- Verification gate added before `type=done` (foundation that tonight's done-without-action gate composes with).
- Host paths in tool-call arguments translated to container paths so the sandbox sees the right file when the model thinks in host-fs terms.
- Fixed a conversation-history drop bug where the post-V3 trim was eating the user's prompt; V3 pipeline now fires on more edit shapes (not just write_file).
- Lens-call timeout in v3-service bumped from 5s to 30s with structured fallback logging on miss.

### Sandbox + execution stack
- **PC-188**: every `run_command` now executes inside the sandbox container, not on the host. Closes the "model writes `rm` and the host runs it" risk.
- **PC-189**: workspace-drift fix and a false-positive in the truncating-redirect detector (was rejecting legit `> file.txt` writes).
- **PC-190**: sandbox verify stack pre-bakes common dev deps (pytest, ruff, etc.), uses tmpfs for the working tree, prints a "create a venv" hint when the model tries to install into the system Python.
- **PC-191/192/193**: sandbox is language-agnostic — works on a working codebase (not just a single-file scratchpad). Detects Python, Node, Go, Rust, Java, C/C++ project layouts and uses the appropriate runner.

### Anti-laziness gates
- **PC-194/195**: `write_file` rejects empty content, single-line stubs, "TODO"-only files, files with `pass`-only bodies, and other lazy outputs.
- **PC-196**: explicit `run_background` tool for long-running processes (e.g. `python app.py`); shell `&` backgrounding through `run_command` is detected and routed to `run_background`.
- **PC-197**: completion-claim verification — when the model declares `done`, the gate checks the workspace state matches the claim (structural check, foundation that tonight's claim-check gate extends).
- **PC-198**: trims boilerplate from the system prompt and strips host `/workspace/` prefixes from model-emitted paths.
- **PC-199/200**: detects "stops at the easy fix" pattern (one tweak then `done`); raises tier-aware turn caps so the model has runway to complete a real task.
- **PC-201**: `write_file` is allowed to overwrite an existing file when that file is corrupted (e.g. truncated mid-write from a prior crashed turn) instead of failing with the usual "file exists" gate.

### PC-202: per-layer residual hidden states from llama-server
- Patched llama-server's `/embedding` endpoint to accept a `layers: [int]` parameter and return the residual-stream hidden state at each requested layer. Foundation for both PC-207 (per-token lens scoring) and tonight's ASA steering vector build.

### PC-206 + PC-207: lens-as-PRM (per-step process reward)
- **PC-206**: thinking-mode plumbing in `v3-service/main.py` `LLMAdapter` — `thinking` keyword resolves per-call against an instance default.
- **PC-207**: lens computes per-token C(x) + G(x) scores during candidate generation; `/internal/lens/score-per-step` exposes aggregates (gx_min, gx_mean, off_rails_idx, cx_norm_max) the proxy and v3-service consume for early-exit and ranking. Wired into v3-service candidate generation, the agent loop (foundation for tonight's reasoning-repeat + path-aware detectors), with structured per-step logging across all three services.
- Severe-score short-circuit: gx_min below 0.05 fires a corrective immediately without waiting for a second sample (calibrated against the May 7 dashboard.html stub-loop session).
- V3↔lens alignment: lens now vetoes a sandbox-passing candidate when its gx_min indicates a stub or placeholder collapse — closes the "sandbox approves a stub V3 generated" loophole.

### Agent loop reliability sweep (May 7)
- Empty-response fallback: when the model returns nothing parseable, the loop emits a corrective hint instead of retrying the same prompt verbatim.
- Plan-threshold guard: refuses to enter the agent loop on a plan with adherence score below threshold.
- Tool-repeat detector: precursor to tonight's reasoning-repetition detector — catches verbatim tool-call repeats within a window.

### GH #39: AST-aware surgical edits + tier-aware V3 routing (May 8)
- **v1 (5e44ffb)**: new `ast_edit` tool — friendly-selector AST node replacement using tree-sitter. Supports `function:NAME`, `class:NAME`, and `<tag>` selectors. The selector vocabulary is intentionally small in v1; nested selectors (e.g. `<style>` inside `<head>`) are NOT supported and produce a "0 nodes matched" error.
- **Point 1 (468a555)**: structural verification veto for V3 candidates — rejects candidates that pass sandbox but fail structural shape checks (e.g. removed a required import, lost the class definition).
- **Point 2 (b95f741)**: cyclomatic-complexity enrichment in tier classification — `tier.py` now considers logic density, not just line count, when assigning T0/T1/T2/T3.
- **Point 3 (2629652)**: Phase 3 repair receives call-chain context (callers + callees of the file being repaired) so the repair model can reason about cross-file effects.
- **Point 4 (bd0b02b)**: auto-injection of a reachability slice from the user's message — the lens picks the most relevant file regions and inlines them into the system context before the loop starts.
- Plan generation made aware of `ast_edit` so plan steps suggest it when the target is a structural edit.
- `edit_file` "string not found" error now suggests `ast_edit` as the recovery; `write_file` rejection on existing files also points to `ast_edit`.
- Three follow-up fixes: encoding (HTML entities in selector args), trim-resilience (large `content` fields surviving the post-V3 trim), and parse-failure categorization in logs.
- Jinja crash fix when `symbol_index` injects snippets: the snippet role was being set to `system`, which Jinja resolved as a template literal; changed to `user` role.

### BiasBusters tool-selection mitigations
- Tool descriptions rewritten to push the model toward the right tool for the task: `edit_file` framed as the surgical default, `ast_edit` marked REQUIRED for HTML/Python structural edits, `write_file` restricted to new-file creation only.
- Conditional GBNF grammar built per turn: when the loop has already entered a step the model has just claimed done, the grammar bans re-emitting the same tool name token-side so the model can't loop on the same failed tool call.
- Per-step tool-list filter (`buildToolDescriptionsExcluding`): the system prompt strips tools the loop has explicitly excluded for this step, so the model never sees them as options.
- ASA (Activation Steering for Aast_edit) wired into the inference entrypoint: `inference/entrypoint-v3.1-9b.sh` auto-detects `/models/ast_edit_steering.gguf` and applies it always-on via llama.cpp `--control-vector`. Default scale 0.5, default layer range full-model, both overridable via env. PC-202's per-layer-residual `/embedding` patch is the upstream that makes this possible.

### ASA steering vector
- New `geometric-lens/asa_calibration/` directory: 1000 contrast-pair prompts (50+ base templates × variation pools) cover function selectors (54%), HTML tags (27%), and CSS classes (19%). `generate_pairs.py` produces `contrast_pairs.jsonl`; `build_steering_vector.py` extracts residuals via the lens `extract_per_layer_per_token` endpoint at layer 27 (of 36 in Qwen3.5-9B), means across tokens/prompts/sign, and writes a llama.cpp-format GGUF control vector. Final vector: 16736 bytes, ‖v_global‖ = 8.6444 after 730s on 2000 prompts.

### Agent loop hardening
- **Plan-progress reminder** (`proxy/plan_reminder.go`): ephemeral system note injected into every step request rendering `plan progress N/M — currently on step "sX": <action> <target>` plus done/remaining sub-step IDs. Lazy-initializes `ctx.PlanStepsSatisfied`. Not persisted to `ctx.Messages`, so it survives the post-V3 conversation trim cycle.
- **Reasoning-repetition detector** (`proxy/reasoning_repeat.go`): tracks the model's reasoning-stream opening; on 3 consecutive identical normalized openings (case-folded, whitespace-collapsed, 80-char snippet) the loop queues a corrective system message. Successfully broke a session-2 stuck loop in live testing.
- **Path-aware error breaker** (`extractFailurePath` in `proxy/lens_score.go`, breaker logic in `proxy/agent.go`): tracks `ctx.RecentFailurePaths` per tool failure. Known limitation: the v1 implementation resets on intervening successes, so it can miss long stuck-loop sequences with sporadic productive turns in between.
- **Done-without-action gate** (`proxy/guardrails.go`): refuses `type=done` when the user prompt is fix-intent and no successful verification command has run this loop. Action-intent words (`rewrite`, `create`, `add`, `update`, `redesign`) also trigger a productive-change check parallel to the existing verify check. Caught 4 false-success done attempts in live testing.
- **Truncation recovery shims** (`proxy/agent.go`): `recoverTruncatedAstEdit` + `recoverTruncatedEditFile` + `recoverTruncatedToolCall` rescue malformed tool emissions from the model and re-pack into a well-formed shape. Each shim is targeted at a specific failure mode observed in production logs.
- **Conversation history error surfacing** (`proxy/agent.go`): `extractModelResponse` now exposes the actual `Unmarshal` error path (directErr vs balancedErr) so debug logs distinguish parse-shape failures from content failures.
- **Removed `ResponseHeaderTimeout`** from `proxy/v3_bridge.go` and removed all client-level timeouts on the V3 HTTP path. Long V3 chains (10+ minute passes) were getting bounced by the 10-minute response-header window even when the pipeline was making progress.
- **Removed `absoluteMaxTurns` ceiling** from `proxy/types.go`. Turn caps now come solely from `TierMaxTurns` (T0:5, T1/T2/T3:0 = uncapped) with no override clamp. Reasoning: 8 detectors armed in the loop make a hard cap redundant — let the detectors decide when to break.

### Surgical-edit hardening (V3 routing)
- `proxy/tools.go` ast_edit executor: tier classification now uses `max(oldTier, newTier)` and the previous V3-tier floor for HTML was dropped (it was over-triggering V3 on the smallest CSS tweaks). Doctype dedup (`leadingDoctypeRe` + `stripLeadingDoctype` in `proxy/guardrails.go`) prevents the model's "<!DOCTYPE html>" prefix from being inserted twice when ast_edit replaces the `<body>`.
- Suspiciously-shrunk-edit guard (`validateNotSuspiciouslyShrunk` in `proxy/guardrails.go`): rejects an edit that shrinks an >100-byte file to <64 bytes. Final threshold tuned after a legitimate 80-byte one-liner refactor was false-rejected at 128. Triggered on a destructive 32-byte stub in pre-release testing.
- Working-directory phantom-dir guard (`validateWorkingDirReference` + `workspaceRefRe`): catches model emissions that try to `cd templates/workspace` or similar nested-workspace references; legitimate `cd /workspace` at the sandbox root is allowed.
- Action-intent gate (`actionIntentWords` + `isActionIntentMessage` + `actionWithoutProductiveChangeMessage`): companion to the verification gate, catches `done` declarations on `rewrite`/`create`/`add`/`redesign`-style prompts that don't include a productive edit this loop.

### TUI reasoning stream visibility
- `tui/model.go` adds a `streamingReasoningText` buffer and a `reasoning_token` event handler that renders with a `‹thinking›` prefix so the user sees the model's reasoning stream live alongside its content. Both buffers reset on `llm_call_start` / `llm_call_end`.
- `tui/commands.go` extended to forward the `delta.ReasoningContent` field from the SSE stream as `reasoning_token` events.
- `proxy/agent.go` plumbs reasoning content through the agent loop: stashes `ctx.LastTurnReasoning`, captures `pendingReasoningCorrective` via `recordReasoning`, and re-emits reasoning deltas to the client mid-turn (with a `sync.Mutex` around the `http.ResponseWriter` to fix the SSE race that produced the "chunked line ends with bare LF" errors).

### Tests
- New Go tests: `proxy/path_aware_test.go`, `proxy/reasoning_repeat_test.go`, `proxy/recover_truncated_test.go`, `proxy/step_restriction_test.go`. Extended `proxy/guardrails_test.go`, `proxy/plan_hook_test.go`.
- All `go test ./...` on both `proxy/` and `tui/` modules pass.
- Full Python suite: 1055 passed / 4 skipped / 0 failed / 0 errors locally.

## [3.0.1] - 2026-04-05

### Tool-Call Agent Loop Architecture
- Replaced Aider format-translation proxy with structured JSON tool-call agent loop
- Grammar-constrained output via llama-server `response_format:json_object` — 100% valid JSON
- 8 tool definitions: `read_file`, `write_file`, `edit_file`, `delete_file`, `run_command`, `search_files`, `list_directory`, `plan_tasks`
- Per-file tier classification: T1 (config/data) writes directly, T2 (logic/features) routes through V3 pipeline
- 3400+ lines new Go code across 12 files in `proxy/`

### V3 Pipeline Integration
- All 14 V3 steps wired into `write_file`/`edit_file` executors for T2/T3 files
- PlanSearch → DivSampling → Budget Forcing → Build Verification → C(x)/G(x) Scoring → Best-of-K → S*/Blend-ASC → Failure Analysis → PR-CoT Repair → Refinement Loop → Derivation Chains → Metacognitive → Final Write
- Per-file-type build verification: tsc, py_compile, gcc, go build, cargo check, bash -n
- V3 service SSE streaming: pipeline progress visible in real-time

### CLI Experience
- `atlas` command: starts all services and launches Aider
- Streaming progress: `[Turn N/M]` with tool call details, V3 pipeline steps, completion summary
- Exploration budget: 4 consecutive read-only calls triggers nudge, prevents model from over-exploring
- Pre-injected project context: model sees project file list in system prompt
- File deletion via fast-path before tier classification
- Truncation prevention: 32K context, reject write_file for existing files >100 lines, detect truncated args before execution

### Deployment
- Docker Compose (`docker-compose.yml`) for full stack orchestration
- Podman compatible with host networking
- `.env.example` with all configurable parameters
- `atlas` script auto-detects Docker vs bare-metal and routes accordingly

### Renames (362 total reference updates)
- `rag-api/` → `geometric-lens/` (directory + all references)
- `ATLAS_RAG_URL` → `ATLAS_LENS_URL`
- `ATLAS_FOX_URL` → `ATLAS_INFERENCE_URL`
- `foxURL` → `inferenceURL` (Go code)
- `ralph-loop` → `verify-repair loop`
- `rag.py` → `pipeline.py` (geometric-lens orchestration)

### Reliability
- 8-level test × 3 iterations: 95.8% (23/24)
- 5-language integration: 100% (Shell, Python, Rust, C, Go)
- L6 (add feature to existing project): 67% — marked as future improvement

### Documentation Overhaul
- **ARCHITECTURE.md**: Complete rewrite — 13 Mermaid diagrams (service topology, agent loop flow, V3 pipeline, module map, sequence diagrams), every component verified against source code
- **API.md**: Complete rewrite — every endpoint across all 5 services verified against source, request/response formats, SSE stages
- **CLI.md**: Complete rewrite — startup flow diagram, streaming format, workflow examples, troubleshooting, env vars, Aider config reference
- **CONFIGURATION.md**: Complete rewrite — every env var across all services verified, internal constants, Docker Compose vs K3s differences
- **MAP.md**: Complete rewrite — every file in repo with clickable tree, 150 file links, 18 description tables
- **SETUP.md**: Complete rewrite — verified build steps, first-run guide, bare metal, K3s, hardware sizing, Lens training guide
- **TROUBLESHOOTING.md**: Complete rewrite — quick diagnostics, 20+ issue scenarios with verified fixes
- **README.md**: Honest 7-step setup with actual download command, prerequisites, model clarity (Qwen3-14B vs Qwen3.5-9B)
- Reorganized historical docs into `docs/reports/` (ablation studies, status tracking, migration guides)

### Bug Fixes
- **geometric-lens Dockerfile port mismatch**: Container was listening on 8001 but docker-compose expected 8099 — fresh Docker Compose deploys had broken Lens service. Fixed Dockerfile to use port 8099.
- **Python CLI default RAG port**: `atlas/cli/client.py` defaulted to port 31144 (K3s NodePort) instead of 8099 (Docker Compose). Fixed default to match Docker Compose.
- **Missing Aider config files**: `.aider.model.settings.yml` and `.aider.model.metadata.json` were not in the repo — the `atlas` launcher would fail without them. Restored both files and added `.gitignore` exceptions.
- GitHub Issue #6: `hostname -I` → portable fallback chain (`ip addr` → `hostname -I` → `hostname -i`) for Arch Linux compatibility
- GitHub Issue #10: `rag-api/` → `geometric-lens/` restructuring resolved missing models directory
- GitHub Issue #11: Added Geometric Lens training documentation to SETUP.md with HuggingFace dataset link
- GitHub Issue #12 / PR #13: `docker image exists` → `docker image inspect` in build script

### Cleanup
- Removed 62 stale test directories, old v1 proxy binary, dead G(x) metric tensor training scripts
- Removed stale tests for deleted services (api-portal, dashboard, embedding-service, task-worker)
- Removed root-level development artifacts (bubble_sort.py, snake_game.py, etc.)
- All hardcoded `/home/isaac/` paths replaced with `$HOME` or `ATLAS_DIR` env vars

## [3.0] - 2026-03-05

### V3.0 Benchmark Release
- **74.6% LCB pass@1** (447/599) on frozen Qwen3-14B
- Full ablation study: conditions A–D with per-task results
- Phase 1 (PlanSearch/DivSampling): +12.4pp
- Phase 3 (PR-CoT/Refinement/Derivation): +7.3pp
- Self-verified Phase 3 using model-generated test cases

## [2.5.1] - 2026-02-23

### Confirmation Ablation: Embedding Source Hypothesis — STRONG CONFIRMATION
- **H1: Self-embeddings restore C(x) discrimination: CONFIRMED (+39.5pp)**
  - C(x) selects passing candidate 87.8% on mixed-result tasks vs 48.3% random (p < 0.000001)
  - V2.5 result (+0.6pp under nomic 768-dim) was an embedding source limitation, not architecture failure
  - Reverse energy selects only 4.3%, proving strong directional signal
  - Val AUC: 0.9934, energy separation: 21.75 (7.2x wider than V2.5)
- **H2: G(x) adds value beyond C(x): NEUTRAL (0.0pp)**
  - G(x) contributes zero at optimal alpha (0.001); monotonically degrades at higher alpha
  - Zero corrections, zero breakages across all mixed-result tasks
- **Outcome B**: Ship C(x)-only with self-embeddings, remove or redesign G(x)
- **Difficulty routing validated**: Q1 (low energy) = 100% oracle, Q4 (high energy) = 0.3%
- **C(x) confirmed as both verifier (87.8% selection) and router (perfect difficulty stratification)**
- Runtime: 24h 42m on LiveCodeBench v5 (599 tasks, K=3, 4 epochs)
- Infrastructure: Qwen3-14B with `--embeddings` (no spec decode, ~45 tok/s)
- Risk R6 (Lens non-discriminating) RESOLVED; Risk R11 (no verifier) substantially mitigated

## [2.5.0] - 2026-02-21

### Ablation Study
- Systematic ablation of Geometric Lens, router, and infrastructure components
- Finding: C(x) energy scoring ≈ random for candidate selection under nomic embeddings (37.7% vs 37.1%, within 3.4pp seed variance) — **V2.5.1 confirmed this was an embedding source limitation** (87.8% accuracy restored with self-embeddings)
- Finding: C(x) energy strongly correlates with task difficulty (58.5% vs 18.9% pass rate across tiers)
- Finding: G(x) metric tensor confirmed dormant (5.2M params, zero impact)
- Finding: Pattern cache bypassed entirely by benchmark runner

### Architecture Change
- Discovered `--embeddings` flag breaks speculative decoding (forces n_batch=512)
- Migrated to two-server sidecar architecture: generation + spec decode on Server A, embeddings via nomic-embed-text-v1.5 on Server B
- Recovered ~2.6x generation throughput (~38 tok/s → ~100 tok/s)
- Net VRAM delta: approximately -230 MiB (sidecar cheaper than --embeddings overhead)

## [2.0.0] - 2026-02-18

### Architecture Changes
- Replaced Qdrant vector DB + embedding service with PageIndex tree-based RAG
- Added Geometric Lens (Cost Field + Metric Tensor) for candidate quality prediction
- Added Confidence Router with difficulty-based adaptive-k selection
- Added Pattern Cache (Redis + Ebbinghaus memory decay)
- Added Best-of-K pipeline with parallel candidate generation
- Added sandboxed code execution for benchmark evaluation
- Added speculative decoding with Qwen3-0.6B draft model
- Added KV cache quantization (q4_0)

### Benchmark Results (Run ID: v2_run_20260217_125310)
- LiveCodeBench: 36-41% pass@1 (across Lens training epochs, k=3)
- GPQA Diamond: 47.0% (k=5)
- SciCode: 14.7% sub-problems (341 tasks, k=1)
- Geometric Lens: 0.968 Val AUC, ~80% first-pick accuracy (151/188)
- Throughput: 109 tasks/hr on RTX 5060 Ti 16GB

### Removed
- Qdrant vector database
- MiniLM-L6-v2 embedding service
- LoRA nightly training pipeline (moved to v1_archived/, CronJob suspended)
- V1 benchmark suite (HumanEval, MBPP, Custom)

### Fixed Post-Release
- mlock allocation failure — added LimitMEMLOCK=infinity systemd override for K3s
- Speculative decode slot 1 failure — quantized draft KV cache to q4_0 (-ctkd/-ctvd)
- Dashboard crash-loop — fixed missing Jinja2 default filters

### Notes
- IFBench evaluation incomplete (excluded from results)
- All results from single benchmark run (variance unknown)

## [1.0.0] - 2026-02-04

Initial release. See benchmark/v1_benchmark_report.md for V1 results.
