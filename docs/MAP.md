# ATLAS Repository Map

Every file in the repository. Click any directory in the tree to jump to its description table.

---

## File Tree

- [`.env.example`](#root-config) — Docker Compose environment template
- [`.gitignore`](#root-config) — Git ignore rules
- [`atlas.conf.example`](#root-config) — K3s deployment configuration template
- [`docker-compose.yml`](#root-config) — 6-service Docker Compose stack (incl. redis)
- [`docker-compose.rocm.yml`](#root-config) — AMD ROCm overlay
- [`docker-compose.vulkan.yml`](#root-config) — Vulkan overlay (universal GPU fallback)
- [`docker-compose.cpu.yml`](#root-config) — CPU-only overlay (no `/dev/dri`; lavapipe)
- [`docker-compose.macos.yml`](#root-config) — macOS hybrid overlay (native llama-server + Docker for the rest)
- [`docker-compose.override.yml.example`](#root-config) — local override template
- [`pyproject.toml`](#root-config) — Python package definition (atlas CLI entry point)
- [`LICENSE`](#root-docs) — GNU Affero General Public License v3.0 (AGPL-3.0)
- [`README.md`](#root-docs) — Project overview, benchmarks, setup
- [`CHANGELOG.md`](#root-docs) — Release history
- [`CODE_OF_CONDUCT.md`](#root-docs) — Community guidelines
- [`CONTRIBUTING.md`](#root-docs) — Contributor guide
- [`proxy/`](#atlas-proxy) — Go proxy: agent loop, grammar, tool calls (~25 source files + tests)
  - [`main.go`](#atlas-proxy) — HTTP server, routes, health/readiness, passthrough
  - [`agent.go`](#atlas-proxy) — Agent loop, LLM dispatch, gates, `/v1/agent` + `/cancel`
  - [`tools.go`](#atlas-proxy) — 15 tool definitions + executors, tier classifier, V3 routing
  - [`types.go`](#atlas-proxy) — Shared types: ToolCall, AgentContext, tiers
  - [`grammar.go`](#atlas-proxy) — JSON schema + GBNF grammar generation
  - [`guardrails.go`](#atlas-proxy) — Shell-mutation gate, content sanitizers, shrinkage guard
  - [`workspace.go`](#atlas-proxy) — Workspace containment for all path-taking tool args
  - [`security.go`](#atlas-proxy) — Log-field encoding for untrusted text
  - [`permissions.go`](#atlas-proxy) — Permission rules and hard deny patterns
  - [`permission_gate.go`](#atlas-proxy) — Interactive approve/deny round-trip: `permission_request` + `/v1/permission`
  - [`events.go`](#atlas-proxy) — `/events` typed-envelope SSE broker (PC-061)
  - [`lens_samples.go`](#atlas-proxy) — `/feedback` + `/v1/lens/training-status`: verdicts → lens training samples
  - [`lens_score.go`](#atlas-proxy) — PC-207 per-write lens scoring + regression intervention
  - [`calibration_status.go`](#atlas-proxy) — `/v1/calibration/status` lens + ASA verdicts
  - [`plan_adherence.go`](#atlas-proxy) — Plan-step matching, off-streak counter, auto-revise
  - [`plan_reminder.go`](#atlas-proxy) — Plan-progress reminder injection
  - [`claim_check.go`](#atlas-proxy) — Completion-claim verification gate (PC-197)
  - [`symbol_index.go`](#atlas-proxy) — Per-session project symbol scan + snippet injection
  - [`tool_repeat.go`](#atlas-proxy) — Tool-call repetition detector
  - [`reasoning_repeat.go`](#atlas-proxy) — Reasoning-repetition detector
  - [`traceback.go`](#atlas-proxy) — Crash-traceback → directed-edit steering
  - [`v3_bridge.go`](#atlas-proxy) — Go-to-Python V3 service SSE bridge
  - [`v3_adapter.go`](#atlas-proxy) — File requests to V3 pipeline format
  - [`build_verify.go`](#atlas-proxy) — Per-language build verification commands
  - [`project.go`](#atlas-proxy) — Language/framework detection
  - [`parallel.go`](#atlas-proxy) — plan_tasks executor with dependency graph
  - [`*_test.go`](#atlas-proxy) — unit + integration tests (one per subsystem)
  - [`go.mod`](#atlas-proxy), [`Dockerfile`](#atlas-proxy), [`README.md`](#atlas-proxy)
- [`tui/`](#atlas-tui) — Bubbletea TUI client (Go) — PC-062
  - [`main.go`](#atlas-tui) — Entry point + Bubbletea program setup
  - [`model.go`](#atlas-tui) — Bubbletea model: events, chat, textarea, hotkeys
  - [`panes.go`](#atlas-tui) — Pure pane renderers (pipeline / chat / events / files / stats / input)
  - [`state.go`](#atlas-tui) — Pipeline state machine (Envelope → derived UI state)
  - [`consumer.go`](#atlas-tui) — `/events` SSE consumer (typed Envelope stream)
  - [`chat.go`](#atlas-tui) — `/v1/agent` POST + SSE chat client; `/cancel`, `/feedback`, `/v1/permission` POST
  - [`session.go`](#atlas-tui) — Session persistence + `--continue`/`--resume` (`~/.cache/atlas-tui/sessions/`)
  - [`commands.go`](#atlas-tui) — Slash command dispatch (/add, /diff, /good, /bad, /review, …)
  - [`calibration.go`](#atlas-tui) — Lens/ASA header badge from `/v1/calibration/status`
  - [`files.go`](#atlas-tui) — Files sidebar (workspace tree scan)
  - [`plan.go`](#atlas-tui) — Plan-mode chat rows (plan_loaded / adherence / revise)
  - [`demo.go`](#atlas-tui) — `/demo` split-pane raw-vs-ATLAS comparison
  - [`debug.go`](#atlas-tui) — Append-only debug log
  - [`*_test.go`](#atlas-tui) — 110+ unit + integration tests
  - [`go.mod`](#atlas-tui), [`go.sum`](#atlas-tui) — Go module (github.com/itigges22/atlas-tui)
- [`atlas/`](#atlas-cli) — Python CLI package
  - [`__init__.py`](#atlas-cli)
  - [`cli/`](#atlas-cli)
    - [`repl.py`](#atlas-cli) — Entry point: subcommand dispatch, TUI launch, pipe-mode REPL
    - [`client.py`](#atlas-cli) — HTTP client for llama-server, Lens, sandbox
    - [`compose.py`](#atlas-cli) — Compose config: `.env` reader, service URL/port resolution, overlay selection
    - [`events.py`](#atlas-cli) — Typed event protocol (canonical Python envelope definition)
    - [`runtime_artifacts.py`](#atlas-cli) — Freshness checks for Go binaries built from the checkout
    - [`display.py`](#atlas-cli) — Terminal output formatting and colors
    - [`__init__.py`](#atlas-cli), [`__main__.py`](#atlas-cli)
    - [`commands/`](#atlas-cli) — 15 command modules
      - [`init.py`](#atlas-cli) — `atlas init` first-run wizard
      - [`doctor.py`](#atlas-cli) — `atlas doctor` install + service diagnostic
      - [`tier.py`](#atlas-cli) — `atlas tier` hardware probe + tier classification
      - [`fit.py`](#atlas-cli) — `atlas tier fit` model-aware runtime sizing (PC-208)
      - [`model.py`](#atlas-cli) — `atlas model` registry install/verify/remove/recommend
      - [`model_registry.py`](#atlas-cli) — Built-in model registry (schema + entries)
      - [`model_recommendations.py`](#atlas-cli) — Back-compat shim over the registry
      - [`onboard.py`](#atlas-cli) — `atlas onboard` guided model drop-in
      - [`lens.py`](#atlas-cli) — `atlas lens` check/build/retrain/publish
      - [`asa.py`](#atlas-cli) — `atlas asa` check/build/publish
      - [`publish.py`](#atlas-cli) — `atlas publish` combined lens+ASA publish (PC-215)
      - [`bench.py`](#atlas-cli) — `atlas bench` / `/bench` benchmark launcher
      - [`solve.py`](#atlas-cli) — `/solve`: generate + score + test
      - [`status.py`](#atlas-cli) — `/status` service health checks
      - [`tui.py`](#atlas-cli) — `atlas tui` locate/build/exec the TUI binary
      - [`__init__.py`](#atlas-cli)
- [`benchmark/`](#benchmark) — Benchmark runner and datasets
  - [`runner.py`](#benchmark-core) — Code execution, LLM API calls, prompt formatting
  - [`llm_client.py`](#benchmark-core) — Shared llama-server client for the runners
  - [`v2_runner.py`](#benchmark-core) — V2 benchmark runner (phases 0-6, telemetry)
  - [`v3_runner.py`](#benchmark-core) — V3 benchmark runner entry point
  - [`v2_report.py`](#benchmark-core) — Markdown report generator
  - [`cli.py`](#benchmark-core) — CLI entry point (atlas benchmark)
  - [`config.py`](#benchmark-core) — BenchmarkConfig from atlas.conf
  - [`models.py`](#benchmark-core) — BenchmarkTask, AttemptResult, TaskResult dataclasses
  - [`best_of_k.py`](#benchmark-core) — Best-of-K candidate evaluation
  - [`geo_learning.py`](#benchmark-core) — Geometric learning integration
  - [`run_v2_benchmark.sh`](#benchmark-core), [`measure_bok_latency.sh`](#benchmark-core), [`README.md`](#benchmark-core)
  - [`datasets/`](#benchmark-datasets) — HumanEval, MBPP, EvalPlus, LiveCodeBench, GPQA, IFBench, SciCode loaders
  - [`analysis/`](#benchmark-analysis) — cost analysis, hardware info, pass@k
  - [`custom/`](#benchmark-custom) — 100 custom tasks + validation
  - [`v3/`](#benchmark-v3) — V3 pipeline modules (19 files)
- [`geometric-lens/`](#geometric-lens) — Scoring, RAG, routing, pattern cache
  - [`main.py`](#geometric-lens-core) — FastAPI server (29 endpoints, port 8099)
  - [`pipeline.py`](#geometric-lens-core) — RAG pipeline orchestrator
  - [`config.py`](#geometric-lens-core) — Server/Redis/API configuration
  - [`storage.py`](#geometric-lens-core) — Project metadata CRUD
  - [`verify_loop.py`](#geometric-lens-core) — Verify-repair loop logic
  - [`sandbox_client.py`](#geometric-lens-core) / [`sandbox_analysis.py`](#geometric-lens-core) — Sandbox HTTP client + result analysis
  - [`requirements.txt`](#geometric-lens-core), [`Dockerfile`](#geometric-lens-core), [`.dockerignore`](#geometric-lens-core)
  - [`geometric_lens/`](#geometric-lens-models) — Scoring models
    - [`cost_field.py`](#geometric-lens-models) — C(x): model-hidden-dim→512→128→1 MLP
    - [`metric_tensor.py`](#geometric-lens-models) — G(x) metric-tensor fallback (XGBoost preferred)
    - [`service.py`](#geometric-lens-models) — Service layer: loading, identity checks, scoring API, hot-reload
    - [`training.py`](#geometric-lens-models) — C(x) + G(x) training pipeline
    - [`calibration.py`](#geometric-lens-models) — Per-model C(x) sigmoid calibration (`cx_normalization.json`)
    - [`thresholds.py`](#geometric-lens-models) — Per-model G(x) operating thresholds (`gx_thresholds.json`)
    - [`identity.py`](#geometric-lens-models) — Artifact↔model identity metadata (`model_identity.json`)
    - [`embedding_extractor.py`](#geometric-lens-models) — llama-server embedding client
    - [`ewc.py`](#geometric-lens-models) — Elastic Weight Consolidation
    - [`correction.py`](#geometric-lens-models) — Natural gradient correction engine
    - [`replay_buffer.py`](#geometric-lens-models) — Domain-stratified experience replay
    - [`models/`](#geometric-lens-models) — Artifact directory (per-model weights + calibration; large files gitignored)
  - [`asa_calibration/`](#geometric-lens-asa) — ASA control-vector build pipeline
  - [`indexer/`](#geometric-lens-indexer) — RAG indexing (tree-sitter AST, BM25, summaries, persistence)
  - [`retriever/`](#geometric-lens-retriever) — RAG retrieval (BM25 / tree / hybrid)
  - [`router/`](#geometric-lens-router) — Confidence routing (Thompson Sampling)
  - [`cache/`](#geometric-lens-cache) — Pattern cache (STM/LTM, decay, co-occurrence)
  - [`models/`](#geometric-lens-core) — Pydantic data models (pattern, route, tree_node)
  - [`data/sample/`](#geometric-lens-core) — Sample embeddings
  - [`tests/`](#geometric-lens-core) — Lens-local unit tests (identity, thresholds, calibration)
- [`v3-service/`](#v3-service) — V3 pipeline HTTP wrapper
  - [`main.py`](#v3-service) — HTTP server, pipeline orchestrator, LLM/Lens/Sandbox adapters
  - [`graph/`](#v3-service) — Structural call-graph engine (#39): extract, resolve, analyses, context, cache, datalog
  - [`Dockerfile`](#v3-service) — Container build (CPU PyTorch, port 8070)
- [`sandbox/`](#sandbox) — Isolated code execution
  - [`executor_server.py`](#sandbox) — FastAPI server: 8 language executors, `/shell`, background jobs, linting
  - [`Dockerfile`](#sandbox), [`requirements-runtime.txt`](#sandbox)
- [`inference/`](#inference) — llama-server configuration
  - [`Dockerfile.v31`](#inference) — CUDA build used by docker-compose
  - [`Dockerfile.rocm`](#inference) — AMD ROCm build
  - [`Dockerfile.vulkan`](#inference) — Vulkan build (universal fallback incl. lavapipe CPU)
  - [`Dockerfile`](#inference) / [`Dockerfile.mtp`](#inference) — base + MTP experimental builds
  - [`entrypoint-v3.1.sh`](#inference) — shared model-neutral Docker/K3s entrypoint
  - [`entrypoint-v3.1-9b.sh`](#inference), [`entrypoint-v3-specdec.sh`](#inference), [`entrypoint.sh`](#inference), [`entrypoint-embed.sh`](#inference), [`entrypoint-mtp.sh`](#inference)
  - [`patches/`](#inference) — `expose-hidden-states.patch` (PC-202), `fix-embeddings-spec-decode.patch`
  - [`templates/`](#inference) — bundled Jinja chat templates
- [`scripts/`](#scripts) — Install, deploy, and training automation
  - [`atlas-bootstrap.sh`](#scripts) — one-shot `curl | bash` installer (PC-051)
  - [`atlas-setup-macos.sh`](#scripts) / [`atlas-llama-macos.sh`](#scripts) — macOS hybrid setup + native launcher (#32)
  - [`install.sh`](#scripts) / [`uninstall.sh`](#scripts) — K3s install + teardown
  - [`build-containers.sh`](#scripts), [`deploy-9b.sh`](#scripts), [`generate-manifests.sh`](#scripts)
  - [`download-models.sh`](#scripts), [`verify-install.sh`](#scripts), [`smoke-test-9b.sh`](#scripts)
  - [`production-readiness.py`](#scripts) — repo-level release gates (compose validation, tests)
  - [`run_full_benchmarks.sh`](#scripts), [`run_v31_ablation.sh`](#scripts), [`validate_benchmarks.py`](#scripts), [`derive_ablation.py`](#scripts)
  - [`retrain_cx.py`](#scripts), [`retrain_cx_phase0.py`](#scripts), [`retrain_lens_from_results.py`](#scripts), [`collect_lens_training_data.py`](#scripts), [`prepare_lens_training.py`](#scripts)
  - [`lib/config.sh`](#scripts) — shared bash config loader
- [`templates/`](#templates) — K3s manifest templates (rendered via envsubst)
- [`tests/`](#tests) — Test suite
  - [`cli/`](#tests) — CLI tests (compose, doctor, init, model, onboard, tier, fit, lens, asa, events, …)
  - [`infrastructure/`](#tests) — llama/sandbox connectivity, compose configuration, CPU images
  - [`v3/`](#tests) — V3 module unit tests
  - [`v3-service/`](#tests) — v3-service tests (graph engine, ast_edit, plan scoring, winner selection, …)
  - [`validate_tests.py`](#tests), [`conftest.py`](#tests)
- [`.github/workflows/`](#ci) — CI: tests, image builds, installer test, CodeQL
- [`docs/`](#docs) — Documentation
  - [`ARCHITECTURE.md`](#docs) — Two-layer architecture, component diagrams, data flow
  - [`API.md`](#docs) — HTTP API reference for all services
  - [`CAPABILITIES.md`](#docs) — What ATLAS can and can't do
  - [`CLI.md`](#docs) — CLI + TUI usage
  - [`CONFIGURATION.md`](#docs) — All environment variables and settings
  - [`DEVELOPMENT.md`](#docs) — Dev workflow (rebuilds, local proxy, tests)
  - [`MAP.md`](#docs) — This file
  - [`PLAN_MODE.md`](#docs) — Plan mode mechanics + constants
  - [`PRODUCTION_READINESS.md`](#docs) — Release gate definitions
  - [`PROTOCOL.md`](#docs) — Typed event envelope contract
  - [`PUBLISHING.md`](#docs) — Publishing Lens + ASA artifacts back to ATLAS
  - [`SETUP.md`](#docs) — Installation guide (bootstrap, Docker, K3s)
  - [`SETUP_MACOS.md`](#docs) — macOS hybrid setup (#32)
  - [`SOURCES.md`](#docs) — Research papers by status bucket
  - [`STORY.md`](#docs) — Project background
  - [`TROUBLESHOOTING.md`](#docs) — Common issues and solutions
  - [`lang/`](#docs) — Translated documentation (zh-CN, ja, ko)
  - [`demo/`](#docs) — Demo prompt set for the TUI `/demo` mode
  - [`images/`](#docs) — README banner + demo GIF
  - [`reports/`](#docs-reports) — Ablation studies, status tracking, migration guides
  - [`reports/ablation/`](#v3-ablation-results) — Published ablation data (599-task conditions)

---

## Description Tables

<a id="root-config"></a>
### Root — Configuration

| File | Description |
|------|-------------|
| [`.env.example`](../.env.example) | Docker Compose env template: model selection, ports (8080/8099/8070/30820/8090), runtime sizing, runtime-tuning knobs |
| [`atlas.conf.example`](../atlas.conf.example) | K3s deployment config: model, parallel slots, NodePorts, namespace, storage paths |
| [`docker-compose.yml`](../docker-compose.yml) | 6-service stack: redis, llama-server, geometric-lens, v3-service, sandbox, atlas-proxy (all `restart: unless-stopped`) |
| [`docker-compose.rocm.yml`](../docker-compose.rocm.yml) | AMD overlay: ROCm image, `/dev/kfd` + `/dev/dri` passthrough |
| [`docker-compose.vulkan.yml`](../docker-compose.vulkan.yml) | Vulkan overlay: vulkan image, `/dev/dri` passthrough, clears the NVIDIA device reservation |
| [`docker-compose.cpu.yml`](../docker-compose.cpu.yml) | CPU-only overlay layered on the Vulkan overlay: strips the `/dev/dri` requirement so GPU-less hosts boot via the lavapipe CPU ICD |
| [`docker-compose.macos.yml`](../docker-compose.macos.yml) | macOS hybrid overlay: containers talk to the native host-side llama-server |
| [`docker-compose.override.yml.example`](../docker-compose.override.yml.example) | Template for local compose overrides |
| [`pyproject.toml`](../pyproject.toml) | Python package: `atlas` CLI entry point (`atlas.cli.repl:run`), requires Python >= 3.9 |
| [`.gitignore`](../.gitignore) | Ignores: model weights, __pycache__, logs, .env, build artifacts |

<a id="root-docs"></a>
### Root — Documentation

| File | Description |
|------|-------------|
| [`README.md`](../README.md) | Project overview, benchmark results, quickstart, hardware requirements |
| [`CHANGELOG.md`](../CHANGELOG.md) | Release history: V3.1.2 (2026-06-17), V3.1.0, V3.0.1, V3.0, V2.x, V1 |
| [`LICENSE`](../LICENSE) | GNU Affero General Public License v3.0 (AGPL-3.0) |
| [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | Contributor Covenant Code of Conduct |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | How to contribute: fork, branch, test, PR workflow |

<a id="atlas-proxy"></a>
### proxy/ — Agent Loop (Go)

The core of the ATLAS CLI. Hosts `/v1/agent` (the structured agent endpoint the TUI drives), runs a grammar-constrained agent loop with 15 tools, and routes complex files through the V3 pipeline. `/v1/chat/completions` is a transparent passthrough to llama-server for OpenAI-compat clients.

| File | Lines | Description |
|------|-------|-------------|
| [`main.go`](../proxy/main.go) | ~410 | HTTP server, route registration, `/health` + `/ready` (gates on inference, lens, sandbox, v3-service), passthrough handler |
| [`agent.go`](../proxy/agent.go) | ~3800 | Agent loop, system prompt, LLM calls with grammar constraint, exploration budget, done-gates (verification / action / claim-check), truncation recovery, `/v1/agent` + `/cancel` handlers |
| [`tools.go`](../proxy/tools.go) | ~3150 | 15 tool definitions + executors (read/outline/write/edit/ast_edit/delete/move file, search, find, list dir, run_command, plan_tasks, run/tail/stop_background), per-file tier classifier, V3 routing, sandbox bridge |
| [`types.go`](../proxy/types.go) | ~780 | AgentContext (with locked read-cache accessors), ToolDef, ToolResult, tier definitions, permission types |
| [`grammar.go`](../proxy/grammar.go) | ~300 | JSON schema (oneOf: tool_call/text/done) and GBNF grammar for constrained output |
| [`guardrails.go`](../proxy/guardrails.go) | ~760 | `validateShellCommand` catastrophic-command gate, whole-file fence sanitizer, suspicious-shrinkage guard |
| [`workspace.go`](../proxy/workspace.go) | ~150 | Workspace containment: every path-taking tool arg is resolved and verified inside the workspace root (symlink-safe) |
| [`security.go`](../proxy/security.go) | ~15 | `safeLogField` — control-byte-safe log encoding for untrusted text |
| [`permissions.go`](../proxy/permissions.go) | ~150 | Allow/deny rules, `DefaultDenyPatterns` (mode-independent), mode-based access |
| [`events.go`](../proxy/events.go) | ~195 | `/events` global typed-envelope SSE broker (PC-061) |
| [`lens_samples.go`](../proxy/lens_samples.go) | ~315 | `/feedback` (accept/deny/thumbs verdicts → weighted lens samples) and `/v1/lens/training-status` (counts + retrain-available flag) |
| [`lens_score.go`](../proxy/lens_score.go) | ~240 | PC-207: per-write lens scoring via `/internal/lens/score-per-step` + regression intervention |
| [`calibration_status.go`](../proxy/calibration_status.go) | ~250 | `/v1/calibration/status` — lens verdicts (supported / no-artifacts / incomplete-artifacts / uncalibrated / dim-mismatch / unreachable) + ASA verdicts (supported / missing / unverified / incompatible) |
| [`plan_adherence.go`](../proxy/plan_adherence.go) | ~360 | Plan-step matching, off-streak counter, auto-revise at `planAutoReviseThreshold=5` |
| [`plan_reminder.go`](../proxy/plan_reminder.go) | ~85 | Plan-progress reminder injection for long multi-file tasks |
| [`claim_check.go`](../proxy/claim_check.go) | ~275 | PC-197: structural verification of universal done-summary claims |
| [`symbol_index.go`](../proxy/symbol_index.go) | ~320 | Per-session project symbol scan; snippet injection via v3-service `/internal/symbol_index` |
| [`tool_repeat.go`](../proxy/tool_repeat.go) | ~90 | Same-(tool,args) repetition detector |
| [`reasoning_repeat.go`](../proxy/reasoning_repeat.go) | ~145 | Repeated-reasoning-prefix detector |
| [`traceback.go`](../proxy/traceback.go) | ~295 | run_command crash → deepest in-project frame → directed `edit_file` steer |
| [`v3_bridge.go`](../proxy/v3_bridge.go) | ~230 | HTTP bridge to the Python V3 service with SSE progress streaming; `ATLAS_V3_TIMEOUT` cap |
| [`v3_adapter.go`](../proxy/v3_adapter.go) | ~175 | Translates file write requests into V3GenerateRequest with project context + constraints |
| [`build_verify.go`](../proxy/build_verify.go) | ~155 | Per-file-type verification: tsc, py_compile, go build, cargo check, gcc, bash -n |
| [`project.go`](../proxy/project.go) | ~285 | Detects language (Node/Python/Rust/Go/C/Shell), framework, build/dev/test commands |
| [`parallel.go`](../proxy/parallel.go) | ~210 | plan_tasks executor: topological sort, concurrent sub-task execution |
| [`go.mod`](../proxy/go.mod) | — | Go module definition |
| [`Dockerfile`](../proxy/Dockerfile) | — | Multi-stage Go build for containerized deployment |

<a id="atlas-tui"></a>
### tui/ — Bubbletea TUI Client (Go)

Native terminal UI that consumes both atlas-proxy SSE streams (`/events` for typed envelopes, `/v1/agent` for chat). The canonical chat front-end. PC-062.

| File | Description |
|------|-------------|
| [`main.go`](../tui/main.go) | Entry point. Parses `--proxy`, spawns SSE consumer goroutine, runs Bubbletea program in alt-screen mode. |
| [`model.go`](../tui/model.go) | Bubbletea model — Envelope channel, chat history, textarea input, hotkeys, feedback verdict staging, retrain banner. |
| [`panes.go`](../tui/panes.go) | Pure pane renderers: pipeline (stage table), chat (markdown via glamour), events (log), files, stats, input. |
| [`state.go`](../tui/state.go) | Pipeline state machine — pure function from Envelope sequence to derived UI state. |
| [`consumer.go`](../tui/consumer.go) | `/events` SSE consumer. Reconnect with exponential backoff (backoff resets after a healthy connection). |
| [`chat.go`](../tui/chat.go) | `/v1/agent` POST + chat-protocol SSE parser; `/cancel` and `/feedback` POSTs. Bearer auth from `secrets/api-keys.json` (all three file shapes, incl. the `atlas init` token-keyed one). |
| [`commands.go`](../tui/commands.go) | Slash-command dispatch: `/add /drop /context /diff /commit /undo /run /good /bad /review /deny /accept /redo /copy /mouse /hide /show /help /quit`. |
| [`calibration.go`](../tui/calibration.go) | Lens/ASA badge fetched from `/v1/calibration/status` (PC-059 / PC-061). |
| [`files.go`](../tui/files.go) | Files sidebar: workspace tree scan, modified-file highlighting. |
| [`plan.go`](../tui/plan.go) | Plan-mode chat rows: step list, adherence one-liners, revisions. |
| [`demo.go`](../tui/demo.go) | `/demo` split-pane: raw lane is a direct model completion (no tools/files); V3 lane runs the full agent with its own sandbox subdir + file review. |
| [`debug.go`](../tui/debug.go) | Append-only JSON-tagged debug log (`~/.cache/atlas-tui/debug.log`). |
| [`*_test.go`](../tui/) | 110+ tests covering the state machine, slash commands, feedback flow, demo mode, chat client + bearer loader. |
| [`go.mod`](../tui/go.mod) | Go module definition (github.com/itigges22/atlas-tui). Deps: bubbletea, lipgloss, bubbles, glamour. |

<a id="atlas-cli"></a>
### atlas/ — Python CLI

Subcommand dispatcher + standalone REPL. `atlas <subcommand>` runs the install/onboarding tooling; plain `atlas` launches the TUI (pipe mode falls through to the `/solve` REPL). Service URLs resolve from shell env, then the Docker `.env` port keys, then defaults.

| File | Description |
|------|-------------|
| [`cli/repl.py`](../atlas/cli/repl.py) | Main entry point (`atlas` command). Subcommand dispatch (`init doctor tier model onboard lens asa publish bench compose tui`), `--help` usage, unknown-subcommand exit 2, TUI launch with workspace alignment, pipe-mode REPL. |
| [`cli/client.py`](../atlas/cli/client.py) | HTTP client for llama-server, Geometric Lens, sandbox. Health checks, chat + raw generation (batch/streaming, `reasoning_content` bridged to `<think>` tags), scoring, sandbox execution. |
| [`cli/compose.py`](../atlas/cli/compose.py) | Compose configuration: `.env` parsing, service URL/port resolution, backend→overlay mapping, `docker compose` command construction, container-id lookup via `compose ps -q`. |
| [`cli/events.py`](../atlas/cli/events.py) | Canonical Python definition of the typed event envelope (PC-061); `iter_events()` consumer. |
| [`cli/runtime_artifacts.py`](../atlas/cli/runtime_artifacts.py) | Checks whether built Go binaries (proxy/TUI) are current relative to the checkout. |
| [`cli/display.py`](../atlas/cli/display.py) | Terminal formatting: banner, colors, status blocks, progress bars |
| [`cli/commands/init.py`](../atlas/cli/commands/init.py) | First-run wizard: hardware probe, model pick, `.env` + `secrets/api-keys.json`; reports failure when api-keys.json can't be written |
| [`cli/commands/doctor.py`](../atlas/cli/commands/doctor.py) | 23-check install diagnostic (docker/compose/arch/gpu/containers/health/artifacts/e2e); prints results incrementally, `--json` buffers |
| [`cli/commands/tier.py`](../atlas/cli/commands/tier.py) | `atlas tier classify \| list \| fit` — hardware probe, tier table, fit dispatch |
| [`cli/commands/fit.py`](../atlas/cli/commands/fit.py) | GGUF-header + VRAM solver for ctx/KV/ubatch (`atlas tier fit`, PC-208) |
| [`cli/commands/model.py`](../atlas/cli/commands/model.py) | Registry install (SHA-256, resumable with HTTP-416 finalization), `--url` BYO download, `recommend`, `install-artifacts` (exit 3 when nothing is registered for direct download), verify, remove; models dir resolves from flag → env → `.env` → `<root>/models` |
| [`cli/commands/model_registry.py`](../atlas/cli/commands/model_registry.py) | Built-in model registry (lens/ASA status, artifact URLs, licenses) |
| [`cli/commands/model_recommendations.py`](../atlas/cli/commands/model_recommendations.py) | Back-compat shim over the registry (PC-055.2 → PC-056) |
| [`cli/commands/onboard.py`](../atlas/cli/commands/onboard.py) | Guided drop-in: `--url` download with `--apply`/interactive `.env` update, arch gate, lens check, next steps |
| [`cli/commands/lens.py`](../atlas/cli/commands/lens.py) | Lens check/build/retrain/publish (PC-057 / PC-058 / PC-059) |
| [`cli/commands/asa.py`](../atlas/cli/commands/asa.py) | ASA check/build/publish (PC-061); build runs inside the lens container via compose-resolved container id |
| [`cli/commands/publish.py`](../atlas/cli/commands/publish.py) | Combined lens+ASA publish with joint pre-flight (PC-215) |
| [`cli/commands/bench.py`](../atlas/cli/commands/bench.py) | Benchmark launcher with live progress (dataset size parsed from runner output) |
| [`cli/commands/solve.py`](../atlas/cli/commands/solve.py) | `/solve`: chat-completions generation (the GGUF's own template via `--jinja`), extract, score via Lens, test via sandbox |
| [`cli/commands/status.py`](../atlas/cli/commands/status.py) | `/status`: health of llama-server, Lens, sandbox |
| [`cli/commands/tui.py`](../atlas/cli/commands/tui.py) | Locate/build/exec the Bubbletea TUI binary |

<a id="benchmark"></a>
<a id="benchmark-core"></a>
### benchmark/ — Benchmark Infrastructure

Runner infrastructure for evaluating LLM code generation across multiple datasets.

| File | Description |
|------|-------------|
| [`runner.py`](../benchmark/runner.py) | Core execution: function mode + stdio mode, LLM API calls, code extraction |
| [`llm_client.py`](../benchmark/llm_client.py) | Shared llama-server client used by the runners |
| [`v2_runner.py`](../benchmark/v2_runner.py) | V2 benchmark runner: phases 0-6, telemetry, Mode A/B, crash recovery |
| [`v3_runner.py`](../benchmark/v3_runner.py) | V3 benchmark runner: full pipeline with ablation conditions A-F; atomic per-task results (interrupted runs resume) |
| [`v2_report.py`](../benchmark/v2_report.py) | Markdown report generator from benchmark results |
| [`cli.py`](../benchmark/cli.py) | CLI entry point: `atlas benchmark --humaneval --dry-run` etc. |
| [`config.py`](../benchmark/config.py) | BenchmarkConfig loaded from atlas.conf |
| [`models.py`](../benchmark/models.py) | Data models: BenchmarkTask, AttemptResult, TaskResult, BenchmarkRun |
| [`best_of_k.py`](../benchmark/best_of_k.py) | Best-of-K candidate evaluation with scoring |
| [`geo_learning.py`](../benchmark/geo_learning.py) | Geometric learning integration for benchmarks |

<a id="benchmark-datasets"></a>
### benchmark/datasets/ — Dataset Loaders

Each loader downloads from HuggingFace (JSON rows API, no pyarrow) and normalizes to BenchmarkTask format.

| File | Tasks | Eval Mode | Description |
|------|-------|-----------|-------------|
| [`base.py`](../benchmark/datasets/base.py) | — | — | Abstract BaseDataset class with download, parse, validate |
| [`humaneval.py`](../benchmark/datasets/humaneval.py) | 164 | function | HumanEval function completion |
| [`mbpp.py`](../benchmark/datasets/mbpp.py) | 500 | function | MBPP with 3-shot [BEGIN]/[DONE] format |
| [`evalplus_humaneval.py`](../benchmark/datasets/evalplus_humaneval.py) | 164 | function | HumanEval+ (EvalPlus augmented tests) |
| [`evalplus_mbpp.py`](../benchmark/datasets/evalplus_mbpp.py) | 500 | function | MBPP+ (EvalPlus augmented tests) |
| [`livecodebench.py`](../benchmark/datasets/livecodebench.py) | 599 | stdio | LiveCodeBench v5 from bzantium mirror |
| [`gpqa.py`](../benchmark/datasets/gpqa.py) | 198 | mcq | GPQA Diamond from OpenAI blob CSV |
| [`ifbench.py`](../benchmark/datasets/ifbench.py) | 300 | ifbench | IFBench instruction-following with loose eval |
| [`scicode.py`](../benchmark/datasets/scicode.py) | ~80 | function | SciCode cross-domain scientific coding |

<a id="benchmark-analysis"></a>
### benchmark/analysis/ — Analysis Utilities

| File | Description |
|------|-------------|
| [`cost_analysis.py`](../benchmark/analysis/cost_analysis.py) | Token cost and electricity cost analysis |
| [`hardware_info.py`](../benchmark/analysis/hardware_info.py) | GPU/CPU detection and reporting |
| [`pass_at_k.py`](../benchmark/analysis/pass_at_k.py) | pass@k metric calculation |

<a id="benchmark-custom"></a>
### benchmark/custom/ — Custom Tasks

| File | Description |
|------|-------------|
| [`tasks.json`](../benchmark/custom/tasks.json) | 100 custom benchmark tasks |
| [`validate.py`](../benchmark/custom/validate.py) | Validates custom task format |

<a id="benchmark-v3"></a>
### benchmark/v3/ — V3 Pipeline Modules

19 Python modules implementing the V3 code generation pipeline. Each module follows a Config + Event + Controller pattern.

| Module | Phase | Description |
|--------|-------|-------------|
| [`plan_search.py`](../benchmark/v3/plan_search.py) | 1A | 3-step pipeline: extract constraints -> construct plans -> generate code. 3 plans default, max 7. |
| [`div_sampling.py`](../benchmark/v3/div_sampling.py) | 1B | 12 perturbations: 4 roles + 4 instructions + 4 styles. Modular selection by candidate index. |
| [`budget_forcing.py`](../benchmark/v3/budget_forcing.py) | 1C | 5 tiers (nothink/light/standard/hard/extreme). Wait injection on premature thinking termination. Energy-to-tier sigmoid mapping. |
| [`blend_asc.py`](../benchmark/v3/blend_asc.py) | 2A | Adaptive K from C(x) energy: 4 bands mapping energy to k=1-12 and budget tier. |
| [`reasc.py`](../benchmark/v3/reasc.py) | 2B | Early stopping: energy < 0.10 AND bottom-10% logprob confidence > -0.5. |
| [`s_star.py`](../benchmark/v3/s_star.py) | 2C | Tiebreaking: generate edge-case inputs where candidates differ, sandbox both, majority wins. |
| [`candidate_selection.py`](../benchmark/v3/candidate_selection.py) | — | 4 strategies: lens (min energy), random, logprob (max mean), oracle (first pass). |
| [`failure_analysis.py`](../benchmark/v3/failure_analysis.py) | 3A | Categorize failures: wrong_algorithm, implementation_bug, edge_case_miss, time_limit, format_error, partial_correct. |
| [`constraint_refinement.py`](../benchmark/v3/constraint_refinement.py) | 3B | Generate refined hypotheses from failure analysis. Cosine distance >= 0.15 prevents repetition. |
| [`pr_cot.py`](../benchmark/v3/pr_cot.py) | 3C | 4 perspectives (logical_consistency, information_completeness, biases, alternative_solutions) x (analysis + repair) = 8 LLM calls. |
| [`derivation_chains.py`](../benchmark/v3/derivation_chains.py) | 3D | Decompose into <= 5 sub-problems, sandbox-verify each, compose final. 7+ LLM calls. |
| [`refinement_loop.py`](../benchmark/v3/refinement_loop.py) | 3E | Orchestrator: FailureAnalysis -> ConstraintRefiner -> CodeGen -> Test -> Learn. 2 iters, 120s budget. |
| [`metacognitive.py`](../benchmark/v3/metacognitive.py) | 3F | Model failure pattern library with frequency tracking, compensation injection, effectiveness monitoring. |
| [`ace_pipeline.py`](../benchmark/v3/ace_pipeline.py) | 3G | Evolving playbooks: Generator-Reflector-Curator pipeline with confidence decay. |
| [`self_test_gen.py`](../benchmark/v3/self_test_gen.py) | util | Generate test cases from problem description. Multiple parsing fallbacks. 50% majority threshold. |
| [`lens_feedback.py`](../benchmark/v3/lens_feedback.py) | util | Online Lens recalibration: collect pass/fail embeddings, trigger retrain at 50-sample intervals; keeps its buffer when the service refuses (read-only models dir). |
| [`embedding_store.py`](../benchmark/v3/embedding_store.py) | util | Binary append-only embedding storage: task_id + candidate_index + label + model-dim float32 vector. |
| [`ablation_analysis.py`](../benchmark/v3/ablation_analysis.py) | util | Bootstrap significance tests, pass rate computation across ablation conditions. |

<a id="geometric-lens"></a>
<a id="geometric-lens-core"></a>
### geometric-lens/ — Core Service

| File | Description |
|------|-------------|
| [`main.py`](../geometric-lens/main.py) | FastAPI server: 29 endpoints for scoring, indexing, routing, caching, retrain/reload (retrain refuses with 503 when the models dir is mounted read-only) |
| [`pipeline.py`](../geometric-lens/pipeline.py) | RAG orchestrator: retrieve chunks + patterns -> collect signals -> estimate difficulty -> route -> generate -> verify |
| [`config.py`](../geometric-lens/config.py) | ServerConfig (port 8099), Redis URL, API keys, YAML config loading |
| [`storage.py`](../geometric-lens/storage.py) | ProjectMetadata CRUD for indexed projects |
| [`verify_loop.py`](../geometric-lens/verify_loop.py) | Verify-repair loop with retry and escalation |
| [`sandbox_client.py`](../geometric-lens/sandbox_client.py) | HTTP client for sandbox code execution |
| [`sandbox_analysis.py`](../geometric-lens/sandbox_analysis.py) | Classify sandbox execution results |
| [`models/`](../geometric-lens/models/) | Pydantic data models: pattern, route, tree_node |
| [`tests/`](../geometric-lens/tests/) | Lens-local unit tests: model identity, threshold loading, score calibration, G(x) weights, ASA prompt templates |
| [`requirements.txt`](../geometric-lens/requirements.txt) | Dependencies: FastAPI, uvicorn, torch (CPU), pydantic, redis, tree-sitter, gguf |
| [`Dockerfile`](../geometric-lens/Dockerfile) | Python 3.11-slim, CPU PyTorch, port 8099 |

<a id="geometric-lens-models"></a>
### geometric-lens/geometric_lens/ — Scoring Models

| File | Description |
|------|-------------|
| [`cost_field.py`](../geometric-lens/geometric_lens/cost_field.py) | C(x): model-hidden-dim→512→128→1 MLP (SiLU + Softplus), contrastive ranking loss. Input dim comes from the loaded model's embedding width. |
| [`metric_tensor.py`](../geometric-lens/geometric_lens/metric_tensor.py) | G(x) metric-tensor fallback: PCA + diagonal metric tensor. Used only when the XGBoost G(x) is unavailable. |
| [`service.py`](../geometric-lens/geometric_lens/service.py) | Service layer: lazy loading, served-model identity verification (probes llama-server `/v1/models`), embedding-dim checks, evaluate_combined(), hot-reload |
| [`training.py`](../geometric-lens/geometric_lens/training.py) | train_cost_field() / retrain_cost_field_bce() for C(x); train_gx() (PCA + XGBoost, thresholds derived from out-of-fold CV scores) |
| [`calibration.py`](../geometric-lens/geometric_lens/calibration.py) | Per-model C(x) sigmoid calibration: derive/save/load `cx_normalization.json` |
| [`thresholds.py`](../geometric-lens/geometric_lens/thresholds.py) | Per-model G(x) operating thresholds: `gx_thresholds.json` (`off_rails` / `low` / `severe`) |
| [`identity.py`](../geometric-lens/geometric_lens/identity.py) | `model_identity.json` read/write — binds artifacts to a model name + embedding dim |
| [`embedding_extractor.py`](../geometric-lens/geometric_lens/embedding_extractor.py) | llama-server embedding client, handles pooled and per-token responses, mean pooling |
| [`ewc.py`](../geometric-lens/geometric_lens/ewc.py) | Elastic Weight Consolidation: Fisher Information Matrix penalty against catastrophic forgetting |
| [`correction.py`](../geometric-lens/geometric_lens/correction.py) | Natural gradient correction: -alpha * G_inv * grad_C. PCA projection/unprojection. Correctability score. |
| [`replay_buffer.py`](../geometric-lens/geometric_lens/replay_buffer.py) | Domain-stratified reservoir sampling. 30% old / 70% new training mix. JSON persistence. |
| [`models/`](../geometric-lens/geometric_lens/models/) | Artifact directory: `cost_field.pt`, `gx_xgboost.json`, `gx_weights.json`, calibration + identity files (weights gitignored; stats JSONs tracked) |

<a id="geometric-lens-asa"></a>
### geometric-lens/asa_calibration/ — ASA Control Vector

| File | Description |
|------|-------------|
| [`build_steering_vector.py`](../geometric-lens/asa_calibration/build_steering_vector.py) | Extracts contrast activations via the PC-202 hidden-states endpoint and writes the control vector in llama.cpp GGUF format |
| [`generate_pairs.py`](../geometric-lens/asa_calibration/generate_pairs.py) | Generates the contrast-pair corpus |
| [`build_cvector_prompts.py`](../geometric-lens/asa_calibration/build_cvector_prompts.py) | Renders pairs with the loaded model's own chat template |
| [`README.md`](../geometric-lens/asa_calibration/README.md) | Manual build walkthrough (the `atlas asa build` CLI wraps this) |

<a id="geometric-lens-indexer"></a>
### geometric-lens/indexer/ — RAG Indexing

| File | Description |
|------|-------------|
| [`ast_parser.py`](../geometric-lens/indexer/ast_parser.py) | tree-sitter Python AST parsing: classes, functions, imports, top-level blocks. Fallback regex parser. |
| [`tree_builder.py`](../geometric-lens/indexer/tree_builder.py) | Build hierarchical TreeIndex from parsed files. Supports incremental updates. |
| [`bm25_index.py`](../geometric-lens/indexer/bm25_index.py) | Inverted index with BM25 scoring (k1=1.5, b=0.75). CamelCase/snake_case tokenization. |
| [`summarizer.py`](../geometric-lens/indexer/summarizer.py) | LLM-generated summaries for tree nodes. |
| [`persistence.py`](../geometric-lens/indexer/persistence.py) | Save/load TreeIndex + BM25Index as JSON to disk. |

<a id="geometric-lens-retriever"></a>
### geometric-lens/retriever/ — RAG Retrieval

| File | Description |
|------|-------------|
| [`bm25_search.py`](../geometric-lens/retriever/bm25_search.py) | BM25 keyword search: min_score=0.1, top_k=20. Strong match detection (threshold=3.0). |
| [`tree_search.py`](../geometric-lens/retriever/tree_search.py) | LLM-guided tree traversal: max_depth=6, max_reasoning_calls=40. Scores children 0-10. |
| [`hybrid.py`](../geometric-lens/retriever/hybrid.py) | Routes between bm25_first, tree_only, and both strategies. Deduplication + score normalization. |

<a id="geometric-lens-router"></a>
### geometric-lens/router/ — Confidence Router

| File | Description |
|------|-------------|
| [`route_selector.py`](../geometric-lens/router/route_selector.py) | Thompson Sampling with Beta(alpha,beta) posteriors. 4 routes: CACHE_HIT(1) -> FAST_PATH(50) -> STANDARD(300) -> HARD_PATH(1500). |
| [`difficulty_estimator.py`](../geometric-lens/router/difficulty_estimator.py) | Weighted fusion of 4 signals -> D(x). Adjusts weights when Geometric Lens is available. |
| [`signal_collector.py`](../geometric-lens/router/signal_collector.py) | Collects: pattern_cache_score, retrieval_confidence, query_complexity, geometric_energy, gx_score. |
| [`feedback_recorder.py`](../geometric-lens/router/feedback_recorder.py) | Records route outcomes to Redis for Thompson Sampling posterior updates. |
| [`fallback_chain.py`](../geometric-lens/router/fallback_chain.py) | Retry escalation: CACHE_HIT -> FAST_PATH -> STANDARD -> HARD_PATH -> terminal. |

<a id="geometric-lens-cache"></a>
### geometric-lens/cache/ — Pattern Cache

| File | Description |
|------|-------------|
| [`pattern_store.py`](../geometric-lens/cache/pattern_store.py) | Redis-backed storage: STM (100 max), LTM, PERSISTENT tiers. Sorted set management. |
| [`pattern_matcher.py`](../geometric-lens/cache/pattern_matcher.py) | BM25 index over pattern summaries. Normalized [0,1] similarity scores. |
| [`pattern_extractor.py`](../geometric-lens/cache/pattern_extractor.py) | LLM-driven extraction of reusable patterns from successful task solutions. |
| [`pattern_scorer.py`](../geometric-lens/cache/pattern_scorer.py) | Ebbinghaus decay: recency-weighted composite score for STM/LTM promotion. |
| [`co_occurrence.py`](../geometric-lens/cache/co_occurrence.py) | Tracks patterns used together. Graph traversal for linked pattern retrieval. |
| [`consolidator.py`](../geometric-lens/cache/consolidator.py) | Category surprise tracking for pattern novelty assessment. |
| [`seed_patterns.py`](../geometric-lens/cache/seed_patterns.py) | Bootstrap patterns for initial cache population. |

<a id="v3-service"></a>
### v3-service/ — V3 Pipeline HTTP Wrapper

| File | Description |
|------|-------------|
| [`main.py`](../v3-service/main.py) | Threaded HTTP server (port 8070). Pipeline orchestrator: Phase 0 (probe) -> Phase 2 (allocate K) -> Phase 1 (generate) -> Selection (lens + structural vetoes; winners matched by original candidate index) -> Phase 3 (repair). LLMAdapter, EmbedAdapter, SandboxAdapter; client-disconnect aborts at phase boundaries. Serves `/v3/generate`, `/v3/run`, `/v3/plan`, `/internal/ast_edit`, `/internal/symbol_index`, `/internal/cyclomatic_complexity`. |
| [`graph/`](../v3-service/graph/) | Structural call-graph engine (#39, port of chiasmus): `extract.py` (tree-sitter → CodeGraph), `resolve.py` / `resolve_calls.py` (import + call resolution), `analyses.py` (O(V+E) graph queries), `context.py` (repair-context slices), `cache.py` (thread-safe per-file LRU), `datalog.py` (optional solver layer), `flags.py` (`ATLAS_CALL_GRAPH` gate, default off), `types.py`, `facts.py`. |
| [`Dockerfile`](../v3-service/Dockerfile) | Python 3.11, CPU PyTorch, copies benchmark/ for V3 module access. Port 8070. |

<a id="sandbox"></a>
### sandbox/ — Isolated Code Execution

| File | Description |
|------|-------------|
| [`executor_server.py`](../sandbox/executor_server.py) | FastAPI server (port 8020). 8 language executors (process-group cleanup on timeout, optional stdin), `/shell` with workspace-snapshot overlays, background jobs (`/jobs/*`, abandoned-job reaping), O_NOFOLLOW path containment on all write paths, linting, error classification. |
| [`Dockerfile`](../sandbox/Dockerfile) | Python 3.11-slim + Node.js 20 + Go 1.22 + Rust stable + gcc/g++. tmpfs workspace, read-only root. |
| [`requirements-runtime.txt`](../sandbox/requirements-runtime.txt) | Python runtime dependencies for the executor |

<a id="inference"></a>
### inference/ — llama-server Configuration

| File | Description |
|------|-------------|
| [`Dockerfile.v31`](../inference/Dockerfile.v31) | CUDA build used by docker-compose. Builds llama.cpp from a pinned revision with the PC-202 hidden-states patch. EXPOSE 8080. |
| [`Dockerfile.rocm`](../inference/Dockerfile.rocm) | AMD ROCm build (V3.1.1). Installs curl for the compose healthcheck. |
| [`Dockerfile.vulkan`](../inference/Dockerfile.vulkan) | Vulkan build (PC-114) — universal fallback; the lavapipe CPU ICD covers GPU-less hosts. |
| [`Dockerfile`](../inference/Dockerfile) | Base llama.cpp build with CUDA support. |
| [`Dockerfile.mtp`](../inference/Dockerfile.mtp) | Multi-Token Prediction experimental build. |
| [`entrypoint-v3.1.sh`](../inference/entrypoint-v3.1.sh) | Shared model-neutral Docker/K3s entrypoint: flash-attn, mlock, `--fit off`, embeddings, `--jinja`; context/KV/batch sizing from env (`atlas tier fit --write`); ASA control-vector gate; `ATLAS_GPU_INDEX` device selection. |
| [`entrypoint-v3.1-9b.sh`](../inference/entrypoint-v3.1-9b.sh) | Compatibility wrapper for the former model-specific filename. |
| [`entrypoint-v3-specdec.sh`](../inference/entrypoint-v3-specdec.sh) | K3s spec-decode entrypoint (main + draft model, embeddings patch). |
| [`entrypoint.sh`](../inference/entrypoint.sh) | Default entrypoint: basic llama-server launch with configurable flags. |
| [`entrypoint-embed.sh`](../inference/entrypoint-embed.sh) | Dedicated embedding-server entrypoint. |
| [`entrypoint-mtp.sh`](../inference/entrypoint-mtp.sh) | MTP experimental entrypoint. |
| [`patches/expose-hidden-states.patch`](../inference/patches/expose-hidden-states.patch) | PC-202: per-layer residual `hidden_states` extension on `/embedding` (the Lens and ASA depend on it). |
| [`patches/fix-embeddings-spec-decode.patch`](../inference/patches/fix-embeddings-spec-decode.patch) | One-line patch: prevents embedding=true from poisoning draft model context in spec decode. |
| [`templates/`](../inference/templates/) | Bundled Jinja chat templates (normally unused — the GGUF's own template renders via `--jinja`). |

<a id="scripts"></a>
### scripts/ — Automation

| File | Description |
|------|-------------|
| [`atlas-bootstrap.sh`](../scripts/atlas-bootstrap.sh) | One-shot `curl \| bash` installer (PC-051): distro detection, Docker + GPU runtime install, `.env` creation with a default model selection when none is set, model + lens artifact download, overlay selection (ROCm / Vulkan / +CPU when no `/dev/dri`), compose up + health wait, ASA build, `atlas doctor` |
| [`atlas-setup-macos.sh`](../scripts/atlas-setup-macos.sh) | macOS hybrid setup (#32): native llama.cpp build + Docker stack |
| [`atlas-llama-macos.sh`](../scripts/atlas-llama-macos.sh) | Native macOS llama-server launcher (mirrors the Docker entrypoint defaults) |
| [`install.sh`](../scripts/install.sh) | Full K3s installation: prerequisites (incl. bc), GPU Operator, namespace, image build, manifest deployment |
| [`uninstall.sh`](../scripts/uninstall.sh) | K3s teardown and cleanup |
| [`build-containers.sh`](../scripts/build-containers.sh) | Build all container images and import to K3s |
| [`deploy-9b.sh`](../scripts/deploy-9b.sh) | Deploy the default 9B configuration to K3s |
| [`generate-manifests.sh`](../scripts/generate-manifests.sh) | Generate K3s manifests from atlas.conf via envsubst |
| [`download-models.sh`](../scripts/download-models.sh) | Download the selected model weights via curl (relative `default.gguf` symlink) |
| [`verify-install.sh`](../scripts/verify-install.sh) | Post-install health verification |
| [`smoke-test-9b.sh`](../scripts/smoke-test-9b.sh) | Quick deployment smoke test |
| [`production-readiness.py`](../scripts/production-readiness.py) | Release gates: compose validation for every shipped overlay combination (base / rocm / vulkan / vulkan+cpu / macos), tests, lint |
| [`run_full_benchmarks.sh`](../scripts/run_full_benchmarks.sh) | Run all benchmark suites sequentially |
| [`run_v31_ablation.sh`](../scripts/run_v31_ablation.sh) | V3.1 ablation study launcher with conditions A-F |
| [`validate_benchmarks.py`](../scripts/validate_benchmarks.py) | Validate benchmark results for completeness |
| [`derive_ablation.py`](../scripts/derive_ablation.py) | Derive ablation conditions from raw benchmark runs |
| [`retrain_cx.py`](../scripts/retrain_cx.py) | Retrain C(x) cost field from collected embeddings (resolves ports from the Docker `.env`, K3s NodePort fallback) |
| [`retrain_cx_phase0.py`](../scripts/retrain_cx_phase0.py) | Phase 0 C(x) initial training |
| [`retrain_lens_from_results.py`](../scripts/retrain_lens_from_results.py) | Retrain Lens models from benchmark result embeddings (mean-pools per-token responses; hot-reloads the service) |
| [`collect_lens_training_data.py`](../scripts/collect_lens_training_data.py) | Collect pass/fail embeddings from benchmark runs |
| [`prepare_lens_training.py`](../scripts/prepare_lens_training.py) | Prepare and validate training data format |
| [`lib/config.sh`](../scripts/lib/config.sh) | Shared bash config: loads atlas.conf, validates paths, sets defaults (incl. `ATLAS_LENS_TRAINING_DIR`) |

<a id="templates"></a>
### templates/ — K3s Manifest Templates

Rendered from `atlas.conf` via envsubst by `scripts/generate-manifests.sh`. Container-side ports are pinned (8080/8099/8070/8020); the `ATLAS_*_PORT` / `ATLAS_*_NODEPORT` vars move only the Service ports.

| File | Description |
|------|-------------|
| [`llama-deployment.yaml.tmpl`](../templates/llama-deployment.yaml.tmpl) | llama-server Deployment + Service (GPU request, models hostPath) |
| [`atlas-proxy-deployment.yaml.tmpl`](../templates/atlas-proxy-deployment.yaml.tmpl) | atlas-proxy Deployment + Service (workspace, read-only models mount, lens-training hostPath, ctx/slots env) |
| [`geometric-lens-deployment.yaml.tmpl`](../templates/geometric-lens-deployment.yaml.tmpl) | geometric-lens Deployment + Service |
| [`v3-service-deployment.yaml.tmpl`](../templates/v3-service-deployment.yaml.tmpl) | v3-service Deployment + Service |
| [`sandbox-deployment.yaml.tmpl`](../templates/sandbox-deployment.yaml.tmpl) | sandbox Deployment + Service |
| [`redis-deployment.yaml.tmpl`](../templates/redis-deployment.yaml.tmpl) | Redis Deployment + Service + PVC |

<a id="tests"></a>
### tests/ — Test Suite

| Path | Description |
|------|-------------|
| [`validate_tests.py`](../tests/validate_tests.py) | Test runner entry point |
| [`conftest.py`](../tests/conftest.py) | Pytest shared fixtures |
| [`cli/`](../tests/cli/) | CLI tests: compose config + `atlas compose`, doctor, init, model (registry, downloads, artifacts), onboard, tier, fit, lens, asa, events, client, macOS launcher, runtime artifacts, REPL dispatch |
| [`infrastructure/`](../tests/infrastructure/) | llama-server connectivity, sandbox connectivity, compose configuration (env passthrough, restart policy), CPU image checks |
| [`v3/`](../tests/v3/) | V3 module unit tests — one per `benchmark/v3/` module plus runner extraction and phase-4 validation |
| [`v3-service/`](../tests/v3-service/) | v3-service tests: graph engine (extract/resolve/analyses/context/solver/multilang), ast_edit HTML, event emission, language verification, lens calibration, plan scoring, sandbox syntax, winner selection |

<a id="ci"></a>
### .github/workflows/ — CI

| File | Description |
|------|-------------|
| [`test.yml`](../.github/workflows/test.yml) | Go + Python test suites, compose validation for every shipped overlay combination |
| [`build-images.yml`](../.github/workflows/build-images.yml) | Container image builds + GHCR publish |
| [`install-test.yml`](../.github/workflows/install-test.yml) | Bootstrap installer end-to-end test (asserts a non-empty model selection lands in `.env`) |
| [`codeql.yml`](../.github/workflows/codeql.yml) | CodeQL static analysis |

<a id="docs"></a>
### docs/ — Documentation

| File | Description |
|------|-------------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Two-layer architecture with Mermaid diagrams, component breakdowns, sequence diagrams |
| [`API.md`](API.md) | HTTP API reference: all endpoints for all services, request/response formats |
| [`CAPABILITIES.md`](CAPABILITIES.md) | What ATLAS can and can't do |
| [`CLI.md`](CLI.md) | CLI + TUI usage, subcommands, workflow examples, troubleshooting |
| [`CONFIGURATION.md`](CONFIGURATION.md) | Every environment variable across all services, internal constants, K3s config |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | Dev workflow: targeted rebuilds, host-side proxy, test suites |
| [`MAP.md`](MAP.md) | This file — repository file map |
| [`PLAN_MODE.md`](PLAN_MODE.md) | Plan mode: per-turn pre-flight planning + adherence constants |
| [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) | Release gate definitions (`scripts/production-readiness.py`) |
| [`PROTOCOL.md`](PROTOCOL.md) | Typed event envelope contract shared by proxy, v3-service, and clients |
| [`PUBLISHING.md`](PUBLISHING.md) | Contributor walkthrough: HF + GitHub publish flow for Lens / ASA artifacts (PC-059, PC-061) |
| [`SETUP.md`](SETUP.md) | Installation: one-shot bootstrap, Docker Compose, K3s |
| [`SETUP_MACOS.md`](SETUP_MACOS.md) | macOS hybrid install (#32): native Metal llama-server + Docker stack |
| [`SOURCES.md`](SOURCES.md) | Research papers bucketed by status relative to the current release |
| [`STORY.md`](STORY.md) | Project background |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Common issues and solutions |
| [`lang/`](lang/) | Translated documentation (zh-CN, ja, ko) |
| [`demo/demo_prompts.json`](demo/demo_prompts.json) | Prompt set for the TUI `/demo` split-pane mode |
| [`images/`](images/) | README banner + demo GIF |

<a id="docs-reports"></a>
### docs/reports/ — Studies, Status, Migration

| File | Description |
|------|-------------|
| [`V3_ABLATION_STUDY.md`](reports/V3_ABLATION_STUDY.md) | V3 ablation methodology: conditions A-D, 599 tasks, statistical analysis |
| [`CALL_GRAPH_REASONING_V3.md`](reports/CALL_GRAPH_REASONING_V3.md) | Structural call-graph reasoning design (#39) |
| [`V2_5_ABLATION_STUDY.md`](reports/V2_5_ABLATION_STUDY.md) | Historical: V2.5 Geometric Lens ablation study |
| [`V2_TO_V2_5_MIGRATION.md`](reports/V2_TO_V2_5_MIGRATION.md) | Historical: V2 to V2.5 migration guide |
| [`V3_STATUS.md`](reports/V3_STATUS.md) | Historical: V3 implementation tracking |
| [`V3_1_STATUS.md`](reports/V3_1_STATUS.md) | V3.1 implementation status |

<a id="v3-ablation-results"></a>
### docs/reports/ablation/ — Published Evidence

Per-task pass/fail data for all V3 ablation conditions. 2,396 task results across 4 conditions. See [README](reports/ablation/README.md) for data format.

| Condition | Directory | Pass@1 | Tasks |
|-----------|-----------|--------|-------|
| A (baseline) | `condition_a_baseline/` | 54.9% | 599 |
| B (+Phase 1) | `condition_b_phase1/` | 67.3% | 599 |
| C (+Phase 1+2) | `condition_c_phase1_2/` | 67.3% | 599 |
| D (+Phase 1+3) | `condition_d_phase1_3/` | 74.6% | 599 |

Each condition contains `summary.json`, `v3_lcb/results.json`, and 599 per-task JSON files in `v3_lcb/per_task/`.
