# ATLAS Repository Map

Every file in the repository. Click any directory in the tree to jump to its description table.

---

## File Tree

- [`.aider.model.metadata.json`](#root-config) — Aider model token limits and cost
- [`.aider.model.settings.yml`](#root-config) — Aider model behavior settings
- [`.env.example`](#root-config) — Docker Compose environment template
- [`.gitignore`](#root-config) — Git ignore rules
- [`atlas.conf.example`](#root-config) — K3s deployment configuration template
- [`docker-compose.yml`](#root-config) — 5-service Docker Compose stack
- [`pyproject.toml`](#root-config) — Python package definition (atlas CLI entry point)
- [`LICENSE`](#root-docs) — ATLAS Source Available License v1.0
- [`README.md`](#root-docs) — Project overview, benchmarks, setup
- [`CHANGELOG.md`](#root-docs) — Release history
- [`CODE_OF_CONDUCT.md`](#root-docs) — Community guidelines
- [`CONTRIBUTING.md`](#root-docs) — Contributor guide
- [`atlas-proxy/`](#atlas-proxy) — Go proxy: agent loop, grammar, tool calls
  - [`main.go`](#atlas-proxy) — HTTP server, chat handler, verify-repair, tier classification
  - [`agent.go`](#atlas-proxy) — Agent loop, LLM dispatch, exploration budget, error recovery
  - [`tools.go`](#atlas-proxy) — 8 tool definitions + executors, tier classifier
  - [`aider_format.go`](#atlas-proxy) — Agent results to Aider whole-file format
  - [`grammar.go`](#atlas-proxy) — JSON schema + GBNF grammar generation
  - [`types.go`](#atlas-proxy) — Shared types: ToolCall, AgentContext, tiers
  - [`v3_bridge.go`](#atlas-proxy) — Go-to-Python V3 service SSE bridge
  - [`v3_adapter.go`](#atlas-proxy) — File requests to V3 pipeline format
  - [`build_verify.go`](#atlas-proxy) — Per-language build verification commands
  - [`project.go`](#atlas-proxy) — Language/framework detection
  - [`permissions.go`](#atlas-proxy) — Permission rules and deny patterns
  - [`parallel.go`](#atlas-proxy) — plan_tasks executor with dependency graph
  - [`go.mod`](#atlas-proxy) — Go module definition
  - [`Dockerfile`](#atlas-proxy) — Multi-stage Go build
  - [`README.md`](#atlas-proxy) — Proxy documentation
  - [`atlas-proxy`](#atlas-proxy) — Compiled Go binary (gitignored in production)
- [`atlas/`](#atlas-cli) — Python CLI package
  - [`__init__.py`](#atlas-cli)
  - [`cli/`](#atlas-cli)
    - [`repl.py`](#atlas-cli) — Interactive REPL entry point
    - [`client.py`](#atlas-cli) — HTTP client for llama-server, Lens, sandbox
    - [`display.py`](#atlas-cli) — Terminal output formatting and colors
    - [`__init__.py`](#atlas-cli), [`__main__.py`](#atlas-cli)
    - [`commands/`](#atlas-cli)
      - [`solve.py`](#atlas-cli) — /solve command: generate + score + test
      - [`bench.py`](#atlas-cli) — /bench command: run V3 benchmarks
      - [`status.py`](#atlas-cli) — /status command: service health checks
      - [`__init__.py`](#atlas-cli)
- [`benchmark/`](#benchmark) — Benchmark runner and datasets
  - [`runner.py`](#benchmark-core) — Code execution, LLM API calls, ChatML formatting
  - [`v2_runner.py`](#benchmark-core) — V2 benchmark runner (phases 0-6, telemetry)
  - [`v3_runner.py`](#benchmark-core) — V3 benchmark runner entry point
  - [`v2_report.py`](#benchmark-core) — Markdown report generator
  - [`cli.py`](#benchmark-core) — CLI entry point (atlas benchmark)
  - [`config.py`](#benchmark-core) — BenchmarkConfig from atlas.conf
  - [`models.py`](#benchmark-core) — BenchmarkTask, AttemptResult, TaskResult dataclasses
  - [`best_of_k.py`](#benchmark-core) — Best-of-K candidate evaluation
  - [`geo_learning.py`](#benchmark-core) — Geometric learning integration
  - [`run_v2_benchmark.sh`](#benchmark-core) — V2 benchmark launch script
  - [`measure_bok_latency.sh`](#benchmark-core) — Best-of-K latency measurement
  - [`README.md`](#benchmark-core) — Benchmark documentation
  - [`datasets/`](#benchmark-datasets)
    - [`base.py`](#benchmark-datasets) — Abstract BaseDataset class
    - [`humaneval.py`](#benchmark-datasets) — HumanEval (164 tasks, function completion)
    - [`mbpp.py`](#benchmark-datasets) — MBPP (500 tasks, 3-shot format)
    - [`evalplus_humaneval.py`](#benchmark-datasets) — HumanEval+ (EvalPlus augmented)
    - [`evalplus_mbpp.py`](#benchmark-datasets) — MBPP+ (EvalPlus augmented)
    - [`livecodebench.py`](#benchmark-datasets) — LiveCodeBench v5 (599 tasks, stdio)
    - [`gpqa.py`](#benchmark-datasets) — GPQA Diamond (198 MCQ)
    - [`ifbench.py`](#benchmark-datasets) — IFBench (300 instruction-following)
    - [`scicode.py`](#benchmark-datasets) — SciCode (~80 scientific coding)
    - [`__init__.py`](#benchmark-datasets)
  - [`analysis/`](#benchmark-analysis)
    - [`cost_analysis.py`](#benchmark-analysis) — Cost/token analysis
    - [`hardware_info.py`](#benchmark-analysis) — GPU/CPU detection
    - [`pass_at_k.py`](#benchmark-analysis) — pass@k metric calculation
    - [`__init__.py`](#benchmark-analysis)
  - [`custom/`](#benchmark-custom)
    - [`tasks.json`](#benchmark-custom) — 100 custom benchmark tasks
    - [`tasks.json.lock`](#benchmark-custom) — Task lock file
    - [`validate.py`](#benchmark-custom) — Custom task validation
    - [`__init__.py`](#benchmark-custom)
  - [`v3/`](#benchmark-v3) — V3 pipeline modules (19 files)
    - [`plan_search.py`](#benchmark-v3) — PlanSearch (1A): 3 constraint-based plans
    - [`div_sampling.py`](#benchmark-v3) — DivSampling (1B): 12 perturbations
    - [`budget_forcing.py`](#benchmark-v3) — BudgetForcing (1C): 5 tiers, Wait injection
    - [`blend_asc.py`](#benchmark-v3) — BlendASC (2A): adaptive K allocation
    - [`reasc.py`](#benchmark-v3) — ReASC (2B): early stopping
    - [`s_star.py`](#benchmark-v3) — S* (2C): differential tiebreaking
    - [`candidate_selection.py`](#benchmark-v3) — 4 selection strategies
    - [`failure_analysis.py`](#benchmark-v3) — FailureAnalysis (3A): 6 failure categories
    - [`constraint_refinement.py`](#benchmark-v3) — ConstraintRefiner (3B): cosine filtering
    - [`pr_cot.py`](#benchmark-v3) — PR-CoT (3C): 4-perspective repair
    - [`derivation_chains.py`](#benchmark-v3) — DerivationChains (3D): sub-problem decomposition
    - [`refinement_loop.py`](#benchmark-v3) — RefinementLoop (3E): orchestrator
    - [`metacognitive.py`](#benchmark-v3) — Metacognitive (3F): failure pattern library
    - [`ace_pipeline.py`](#benchmark-v3) — ACE (3G): playbook learning
    - [`self_test_gen.py`](#benchmark-v3) — Model-generated test cases
    - [`lens_feedback.py`](#benchmark-v3) — Online Lens recalibration
    - [`embedding_store.py`](#benchmark-v3) — Binary embedding persistence
    - [`ablation_analysis.py`](#benchmark-v3) — Statistical ablation analysis
    - [`__init__.py`](#benchmark-v3)
- [`geometric-lens/`](#geometric-lens) — Scoring, RAG, routing, pattern cache
  - [`main.py`](#geometric-lens-core) — FastAPI server (26 endpoints)
  - [`pipeline.py`](#geometric-lens-core) — RAG pipeline orchestrator
  - [`config.py`](#geometric-lens-core) — Server/Redis/API configuration
  - [`storage.py`](#geometric-lens-core) — Project metadata CRUD
  - [`verify_loop.py`](#geometric-lens-core) — Verify-repair loop logic
  - [`sandbox_client.py`](#geometric-lens-core) — Sandbox HTTP client
  - [`sandbox_analysis.py`](#geometric-lens-core) — Sandbox result analysis
  - [`requirements.txt`](#geometric-lens-core) — Python dependencies (CPU PyTorch)
  - [`Dockerfile`](#geometric-lens-core) — Container build (port 8099)
  - [`.dockerignore`](#geometric-lens-core)
  - [`geometric_lens/`](#geometric-lens-models) — Scoring models
    - [`cost_field.py`](#geometric-lens-models) — C(x): 4096->512->128->1 MLP
    - [`metric_tensor.py`](#geometric-lens-models) — G(x): diagonal metric tensor + PCA
    - [`service.py`](#geometric-lens-models) — Service layer: evaluate_combined(), scoring API
    - [`training.py`](#geometric-lens-models) — Training pipeline: contrastive loss, retraining
    - [`embedding_extractor.py`](#geometric-lens-models) — llama-server /v1/embeddings client
    - [`ewc.py`](#geometric-lens-models) — Elastic Weight Consolidation
    - [`correction.py`](#geometric-lens-models) — Natural gradient correction engine
    - [`replay_buffer.py`](#geometric-lens-models) — Domain-stratified experience replay
    - [`__init__.py`](#geometric-lens-models)
  - [`indexer/`](#geometric-lens-indexer) — RAG indexing
    - [`ast_parser.py`](#geometric-lens-indexer) — tree-sitter AST parsing
    - [`tree_builder.py`](#geometric-lens-indexer) — Hierarchical code index
    - [`bm25_index.py`](#geometric-lens-indexer) — BM25 inverted index
    - [`summarizer.py`](#geometric-lens-indexer) — LLM-generated node summaries
    - [`persistence.py`](#geometric-lens-indexer) — JSON index persistence
    - [`__init__.py`](#geometric-lens-indexer)
  - [`retriever/`](#geometric-lens-retriever) — RAG retrieval
    - [`bm25_search.py`](#geometric-lens-retriever) — BM25 keyword search
    - [`tree_search.py`](#geometric-lens-retriever) — LLM-guided tree traversal
    - [`hybrid.py`](#geometric-lens-retriever) — Hybrid retriever (routes bm25/tree/both)
    - [`__init__.py`](#geometric-lens-retriever)
  - [`router/`](#geometric-lens-router) — Confidence routing
    - [`route_selector.py`](#geometric-lens-router) — Thompson Sampling route selection
    - [`difficulty_estimator.py`](#geometric-lens-router) — 4-signal difficulty fusion
    - [`signal_collector.py`](#geometric-lens-router) — Signal collection
    - [`feedback_recorder.py`](#geometric-lens-router) — Redis-backed outcome recording
    - [`fallback_chain.py`](#geometric-lens-router) — Route escalation chain
    - [`__init__.py`](#geometric-lens-router)
  - [`cache/`](#geometric-lens-cache) — Pattern cache
    - [`pattern_store.py`](#geometric-lens-cache) — Redis STM/LTM/PERSISTENT storage
    - [`pattern_matcher.py`](#geometric-lens-cache) — BM25 pattern matching
    - [`pattern_extractor.py`](#geometric-lens-cache) — LLM-driven pattern extraction
    - [`pattern_scorer.py`](#geometric-lens-cache) — Ebbinghaus decay scoring
    - [`co_occurrence.py`](#geometric-lens-cache) — Co-occurrence graph
    - [`consolidator.py`](#geometric-lens-cache) — Category surprise tracking
    - [`seed_patterns.py`](#geometric-lens-cache) — Bootstrap seed patterns
    - [`__init__.py`](#geometric-lens-cache)
- [`v3-service/`](#v3-service) — V3 pipeline HTTP wrapper
  - [`main.py`](#v3-service) — HTTP server, pipeline orchestrator, LLM/Lens/Sandbox adapters
  - [`Dockerfile`](#v3-service) — Container build (CPU PyTorch, port 8070)
- [`sandbox/`](#sandbox) — Isolated code execution
  - [`executor_server.py`](#sandbox) — FastAPI server, 8 language executors, linting, error classification
  - [`Dockerfile`](#sandbox) — Container build (Python, Node, Go, Rust, gcc)
- [`inference/`](#inference) — llama-server configuration
  - [`Dockerfile.v31`](#inference) — V3.1 9B model build (used by docker-compose)
  - [`Dockerfile`](#inference) — Base llama.cpp build
  - [`Dockerfile.mtp`](#inference) — Multi-Token Prediction experimental build
  - [`entrypoint-v3.1-9b.sh`](#inference) — K3s 9B entrypoint (flash-attn, mlock, 4 slots)
  - [`entrypoint-v3-specdec.sh`](#inference) — K3s 14B + spec decode entrypoint
  - [`entrypoint.sh`](#inference) — Default entrypoint
  - [`entrypoint-embed.sh`](#inference) — Dedicated embedding server entrypoint
  - [`entrypoint-mtp.sh`](#inference) — MTP experimental entrypoint
  - [`patches/fix-embeddings-spec-decode.patch`](#inference) — Fix for embeddings + spec decode conflict
  - [`templates/Qwen3-custom.jinja`](#inference) — Custom Qwen3 chat template
  - [`templates/Qwen3-no-think.jinja`](#inference) — Qwen3 template with thinking suppressed
- [`scripts/`](#scripts) — Build, deploy, and training automation
  - [`install.sh`](#scripts) — K3s + GPU Operator installation
  - [`uninstall.sh`](#scripts) — K3s teardown
  - [`build-containers.sh`](#scripts) — Build all container images
  - [`deploy-9b.sh`](#scripts) — Deploy 9B model to K3s
  - [`generate-manifests.sh`](#scripts) — K3s manifests from atlas.conf
  - [`download-models.sh`](#scripts) — Download model weights
  - [`verify-install.sh`](#scripts) — Post-install verification
  - [`smoke-test-9b.sh`](#scripts) — Quick 9B deployment smoke test
  - [`run_full_benchmarks.sh`](#scripts) — Run all benchmark suites
  - [`run_v31_ablation.sh`](#scripts) — V3.1 ablation study launcher
  - [`validate_benchmarks.py`](#scripts) — Benchmark result validation
  - [`derive_ablation.py`](#scripts) — Derive ablation conditions from runs
  - [`retrain_cx.py`](#scripts) — Retrain C(x) cost field
  - [`retrain_cx_phase0.py`](#scripts) — Phase 0 C(x) training
  - [`retrain_lens_from_results.py`](#scripts) — Retrain Lens from benchmark results
  - [`collect_lens_training_data.py`](#scripts) — Collect embeddings for training
  - [`prepare_lens_training.py`](#scripts) — Prepare training data
  - [`lib/config.sh`](#scripts) — Shared bash config loader
- [`tests/`](#tests) — Test suite
  - [`validate_tests.py`](#tests) — Test runner entry point
  - [`conftest.py`](#tests) — Pytest shared fixtures
  - [`infrastructure/`](#tests)
    - [`test_llm.py`](#tests) — llama-server connectivity tests
    - [`test_sandbox.py`](#tests) — Sandbox connectivity tests
  - [`integration/`](#tests)
    - [`test_e2e_flow.py`](#tests) — End-to-end pipeline flow test
    - [`test_e2e_training.py`](#tests) — End-to-end training test
  - [`v3/`](#tests) — V3 module unit tests (22 files)
    - [`test_plan_search.py`](#tests), [`test_div_sampling.py`](#tests), [`test_budget_forcing.py`](#tests), [`test_blend_asc.py`](#tests), [`test_reasc.py`](#tests), [`test_s_star.py`](#tests), [`test_candidate_selection.py`](#tests), [`test_failure_analysis.py`](#tests), [`test_constraint_refinement.py`](#tests), [`test_pr_cot.py`](#tests), [`test_derivation_chains.py`](#tests), [`test_refinement_loop.py`](#tests), [`test_metacognitive.py`](#tests), [`test_ace_pipeline.py`](#tests), [`test_self_test_gen.py`](#tests), [`test_lens_feedback.py`](#tests), [`test_embedding_store.py`](#tests), [`test_ablation_analysis.py`](#tests), [`test_ewc.py`](#tests), [`test_replay_buffer.py`](#tests), [`test_enhanced_retrain.py`](#tests), [`test_phase4_validation.py`](#tests), [`test_sandbox_adapter.py`](#tests)
- [`docs/`](#docs) — Documentation
  - [`ARCHITECTURE.md`](#docs) — Two-layer architecture, component diagrams, data flow
  - [`API.md`](#docs) — HTTP API reference for all 5 services
  - [`CLI.md`](#docs) — CLI usage, streaming output, troubleshooting
  - [`CONFIGURATION.md`](#docs) — All environment variables and settings
  - [`MAP.md`](#docs) — This file
  - [`SETUP.md`](#docs) — Installation guide (Docker, bare-metal, K3s)
  - [`TROUBLESHOOTING.md`](#docs) — Common issues and solutions
  - [`reports/`](#docs-reports) — Ablation studies, status tracking, migration guides
    - [`V3_ABLATION_STUDY.md`](#docs-reports) — V3 ablation methodology and results
    - [`V2_5_ABLATION_STUDY.md`](#docs-reports) — V2.5 Geometric Lens ablation (historical)
    - [`V2_TO_V2_5_MIGRATION.md`](#docs-reports) — V2 to V2.5 migration guide (historical)
    - [`V3_STATUS.md`](#docs-reports) — V3 implementation status (historical)
    - [`V3_1_STATUS.md`](#docs-reports) — V3.1 implementation status
  - [`images/banner.png`](#docs) — README banner image
  - [`images/ATLAS_CLI.png`](#docs) — CLI screenshot
- [`v3_ablation_results/`](#v3-ablation-results) — Published ablation data
  - [`README.md`](#v3-ablation-results) — Data format documentation
  - [`config.json`](#v3-ablation-results) — Ablation run configuration
  - [`preflight.json`](#v3-ablation-results) — Pre-run system checks
  - `condition_a_baseline/` — Baseline (54.9%, 599 tasks)
  - `condition_b_phase1/` — +Phase 1 (67.3%, 599 tasks)
  - `condition_c_phase1_2/` — +Phase 1+2 (67.3%, 599 tasks)
  - `condition_d_phase1_3/` — +Phase 1+3 (74.6%, 599 tasks)
  - Each condition contains `summary.json`, `v3_lcb/results.json`, and `v3_lcb/per_task/` (599 per-task JSON files)

---

## Description Tables

<a id="root-config"></a>
### Root — Configuration

| File | Description |
|------|-------------|
| [`.aider.model.metadata.json`](../.aider.model.metadata.json) | Aider model metadata: token limits (32K), cost ($0 — local), provider (openai) |
| [`.aider.model.settings.yml`](../.aider.model.settings.yml) | Aider behavior: whole-file edit format, repo map enabled, streaming on, temperature 0.3 |
| [`.env.example`](../.env.example) | Docker Compose env template: model path, ports (8080/8099/8070/30820/8090), context size |
| [`atlas.conf.example`](../atlas.conf.example) | K3s deployment config: model, GPU layers, parallel slots, NodePorts, namespace |
| [`docker-compose.yml`](../docker-compose.yml) | 5-service stack: llama-server, geometric-lens, v3-service, sandbox, atlas-proxy |
| [`pyproject.toml`](../pyproject.toml) | Python package: `atlas` CLI entry point (`atlas.cli.repl:run`), requires Python >= 3.9 |
| [`.gitignore`](../.gitignore) | Ignores: model weights, __pycache__, .aider* (except config files), logs, .env |

<a id="root-docs"></a>
### Root — Documentation

| File | Description |
|------|-------------|
| [`README.md`](../README.md) | Project overview, 74.6% LCB benchmark, setup instructions, hardware requirements |
| [`CHANGELOG.md`](../CHANGELOG.md) | Release history: V3.0.1 (2026-04-05), V3.0, V2.5, V2 |
| [`LICENSE`](../LICENSE) | ATLAS Source Available License v1.0 — free for personal/research, restricted commercial |
| [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | Contributor Covenant Code of Conduct |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | How to contribute: fork, branch, test, PR workflow |

<a id="atlas-proxy"></a>
### atlas-proxy/ — Agent Loop (Go)

The core of the V3.0.1 CLI. Receives OpenAI-compatible requests from Aider, runs a grammar-constrained agent loop with 8 tools, and routes complex files through the V3 pipeline.

| File | Lines | Description |
|------|-------|-------------|
| [`main.go`](../atlas-proxy/main.go) | 2890 | HTTP server, `/v1/chat/completions` handler, verify-repair pipeline, best-of-K, format normalization, error analysis, Lens scoring, sandbox testing |
| [`agent.go`](../atlas-proxy/agent.go) | 740 | Agent loop iteration, JSON schema generation, system prompt building, LLM calls with grammar constraint, exploration budget, truncation recovery |
| [`tools.go`](../atlas-proxy/tools.go) | 905 | 8 tool definitions (read/write/edit/delete file, run command, search, list dir, plan tasks), per-file tier classifier, V3 routing |
| [`aider_format.go`](../atlas-proxy/aider_format.go) | 697 | Converts agent results to Aider whole-file blocks, streams real-time status with icons, project directory detection, delete fast-path |
| [`grammar.go`](../atlas-proxy/grammar.go) | 192 | JSON schema (oneOf: tool_call/text/done) and GBNF grammar for constrained output, tool documentation generation |
| [`types.go`](../atlas-proxy/types.go) | 390 | AgentContext, ToolDef, ToolResult, tier definitions (T0-T3), max turns per tier, permission types |
| [`v3_bridge.go`](../atlas-proxy/v3_bridge.go) | 120 | HTTP bridge to Python V3 service with SSE progress streaming, Lens scoring bridge |
| [`v3_adapter.go`](../atlas-proxy/v3_adapter.go) | 177 | Translates file write requests into V3GenerateRequest with project context, framework detection, constraint extraction |
| [`build_verify.go`](../atlas-proxy/build_verify.go) | 157 | Per-file-type verification: tsc, py_compile, go build, cargo check, gcc, bash -n. Framework-specific overrides |
| [`project.go`](../atlas-proxy/project.go) | 226 | Detects language (Node/Python/Rust/Go/C/Shell), framework (Next.js/Flask/Express), build/dev/test commands |
| [`permissions.go`](../atlas-proxy/permissions.go) | 150 | Allow/deny rules, dangerous pattern detection (rm -rf, .env, credentials), mode-based access |
| [`parallel.go`](../atlas-proxy/parallel.go) | 213 | plan_tasks executor: topological sort, concurrent sub-task execution (15-turn budget each) |
| [`go.mod`](../atlas-proxy/go.mod) | — | Go module definition |
| [`Dockerfile`](../atlas-proxy/Dockerfile) | — | Multi-stage Go build for containerized deployment |

<a id="atlas-cli"></a>
### atlas/ — Python CLI

Standalone REPL for direct interaction with ATLAS services (without Aider).

| File | Description |
|------|-------------|
| [`cli/repl.py`](../atlas/cli/repl.py) | Main entry point (`atlas` command). Interactive REPL with /solve, /bench, /status, /help. Pipe mode support. |
| [`cli/client.py`](../atlas/cli/client.py) | HTTP client for llama-server, Geometric Lens, sandbox. Health checks, generation (batch + streaming), scoring, sandbox execution. |
| [`cli/display.py`](../atlas/cli/display.py) | Terminal formatting: banner, colors, status blocks, prompts, separators |
| [`cli/commands/solve.py`](../atlas/cli/commands/solve.py) | /solve: generate code from LLM, extract from think blocks, score via Lens, test via sandbox |
| [`cli/commands/bench.py`](../atlas/cli/commands/bench.py) | /bench: delegates to benchmark.v3_runner with dataset/strategy/task-count args |
| [`cli/commands/status.py`](../atlas/cli/commands/status.py) | /status: check health of llama-server, Lens, sandbox |

<a id="benchmark"></a>
<a id="benchmark-core"></a>
### benchmark/ — Benchmark Infrastructure

Runner infrastructure for evaluating LLM code generation across multiple datasets.

| File | Description |
|------|-------------|
| [`runner.py`](../benchmark/runner.py) | Core execution: function mode + stdio mode, LLM API calls, ChatML formatting, code extraction |
| [`v2_runner.py`](../benchmark/v2_runner.py) | V2 benchmark runner: phases 0-6, telemetry, Mode A/B, crash recovery |
| [`v3_runner.py`](../benchmark/v3_runner.py) | V3 benchmark runner: full pipeline with ablation conditions A-F |
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
| [`lens_feedback.py`](../benchmark/v3/lens_feedback.py) | util | Online Lens recalibration: collect pass/fail embeddings, trigger retrain at 50-sample intervals. |
| [`embedding_store.py`](../benchmark/v3/embedding_store.py) | util | Binary append-only embedding storage: task_id + candidate_index + label + 4096-dim float32 vector. |
| [`ablation_analysis.py`](../benchmark/v3/ablation_analysis.py) | util | Bootstrap significance tests, pass rate computation across ablation conditions. |

<a id="geometric-lens"></a>
<a id="geometric-lens-core"></a>
### geometric-lens/ — Core Service

| File | Description |
|------|-------------|
| [`main.py`](../geometric-lens/main.py) | FastAPI server: 26 endpoints for scoring, indexing, routing, caching, pattern management |
| [`pipeline.py`](../geometric-lens/pipeline.py) | RAG orchestrator: retrieve chunks + patterns -> collect signals -> estimate difficulty -> route -> generate -> verify |
| [`config.py`](../geometric-lens/config.py) | ServerConfig (port 8001), Redis URL, API keys, YAML config loading |
| [`storage.py`](../geometric-lens/storage.py) | ProjectMetadata CRUD for indexed projects |
| [`verify_loop.py`](../geometric-lens/verify_loop.py) | Verify-repair loop with retry and escalation |
| [`sandbox_client.py`](../geometric-lens/sandbox_client.py) | HTTP client for sandbox code execution |
| [`sandbox_analysis.py`](../geometric-lens/sandbox_analysis.py) | Classify sandbox execution results |
| [`requirements.txt`](../geometric-lens/requirements.txt) | Dependencies: FastAPI, uvicorn, torch (CPU), pydantic, redis, tree-sitter |
| [`Dockerfile`](../geometric-lens/Dockerfile) | Python 3.11-slim, CPU PyTorch, port 8099 |

<a id="geometric-lens-models"></a>
### geometric-lens/geometric_lens/ — Scoring Models

| File | Description |
|------|-------------|
| [`cost_field.py`](../geometric-lens/geometric_lens/cost_field.py) | C(x): 4096->512->128->1 MLP (SiLU + Softplus). 2.16M params. Contrastive ranking loss. |
| [`metric_tensor.py`](../geometric-lens/geometric_lens/metric_tensor.py) | G(x): PCA(4096->128) + diagonal metric tensor + input-dependent modulation. Code exists, not deployed. |
| [`service.py`](../geometric-lens/geometric_lens/service.py) | Service layer: lazy model loading, evaluate_combined() (single embedding for C(x)+G(x)), verdict thresholds, hot-reload |
| [`training.py`](../geometric-lens/geometric_lens/training.py) | train_cost_field() (200 epochs), retrain_cost_field_bce() (production retrain with class weights, early stopping) |
| [`embedding_extractor.py`](../geometric-lens/geometric_lens/embedding_extractor.py) | Calls llama-server POST /v1/embeddings, handles pooled and per-token responses, mean pooling |
| [`ewc.py`](../geometric-lens/geometric_lens/ewc.py) | Elastic Weight Consolidation: Fisher Information Matrix, penalty term, prevents catastrophic forgetting |
| [`correction.py`](../geometric-lens/geometric_lens/correction.py) | Natural gradient correction: -alpha * G_inv * grad_C. PCA projection/unprojection. Correctability score. |
| [`replay_buffer.py`](../geometric-lens/geometric_lens/replay_buffer.py) | Domain-stratified reservoir sampling. 30% old / 70% new training mix. JSON persistence. |

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
| [`main.py`](../v3-service/main.py) | HTTP server (port 8070). Pipeline orchestrator: Phase 0 (probe) -> Phase 2 (allocate K) -> Phase 1 (generate) -> Selection -> Phase 3 (repair). LLMAdapter, EmbedAdapter, SandboxAdapter, BuildVerifier. Imports all 19 V3 modules. |
| [`Dockerfile`](../v3-service/Dockerfile) | Python 3.11, CPU PyTorch, copies benchmark/ for V3 module access. Port 8070. |

<a id="sandbox"></a>
### sandbox/ — Isolated Code Execution

| File | Description |
|------|-------------|
| [`executor_server.py`](../sandbox/executor_server.py) | FastAPI server (port 8020). 8 language executors with compilation, pytest/pylint for Python, syntax checking, error classification (15 types), output truncation. |
| [`Dockerfile`](../sandbox/Dockerfile) | Python 3.11-slim + Node.js 20 + Go 1.22 + Rust stable + gcc/g++. tmpfs workspace, read-only root. |

<a id="inference"></a>
### inference/ — llama-server Configuration

| File | Description |
|------|-------------|
| [`Dockerfile.v31`](../inference/Dockerfile.v31) | V3.1 9B model Docker build. Used by docker-compose. Builds llama.cpp from source with CUDA. |
| [`Dockerfile`](../inference/Dockerfile) | Base llama.cpp build with CUDA support. |
| [`Dockerfile.mtp`](../inference/Dockerfile.mtp) | Multi-Token Prediction experimental build. |
| [`entrypoint-v3.1-9b.sh`](../inference/entrypoint-v3.1-9b.sh) | K3s 9B production entrypoint: flash-attn, mlock, --parallel 4, KV quant (q8_0/q4_0), embeddings, 160K context. |
| [`entrypoint-v3-specdec.sh`](../inference/entrypoint-v3-specdec.sh) | K3s 14B + spec decode entrypoint: Qwen3-14B main + Qwen3-0.6B draft, embeddings patch. |
| [`entrypoint.sh`](../inference/entrypoint.sh) | Default entrypoint: basic llama-server launch with configurable flags. |
| [`entrypoint-embed.sh`](../inference/entrypoint-embed.sh) | Dedicated embedding server entrypoint (nomic-embed-text-v1.5). |
| [`entrypoint-mtp.sh`](../inference/entrypoint-mtp.sh) | MTP experimental entrypoint. |
| [`patches/fix-embeddings-spec-decode.patch`](../inference/patches/fix-embeddings-spec-decode.patch) | One-line patch: prevents embedding=true from poisoning draft model context in spec decode. |
| [`templates/Qwen3-custom.jinja`](../inference/templates/Qwen3-custom.jinja) | Custom Qwen3 Jinja2 chat template. |
| [`templates/Qwen3-no-think.jinja`](../inference/templates/Qwen3-no-think.jinja) | Qwen3 template that suppresses `<think>` blocks. |

<a id="scripts"></a>
### scripts/ — Automation

| File | Description |
|------|-------------|
| [`install.sh`](../scripts/install.sh) | Full K3s installation: prerequisites, GPU Operator, namespace, image build, manifest deployment |
| [`uninstall.sh`](../scripts/uninstall.sh) | K3s teardown and cleanup |
| [`build-containers.sh`](../scripts/build-containers.sh) | Build all container images and import to K3s |
| [`deploy-9b.sh`](../scripts/deploy-9b.sh) | Deploy Qwen3.5-9B to K3s cluster |
| [`generate-manifests.sh`](../scripts/generate-manifests.sh) | Generate K3s manifests from atlas.conf via envsubst |
| [`download-models.sh`](../scripts/download-models.sh) | Download model weights from HuggingFace |
| [`verify-install.sh`](../scripts/verify-install.sh) | Post-install health verification |
| [`smoke-test-9b.sh`](../scripts/smoke-test-9b.sh) | Quick smoke test for 9B deployment |
| [`run_full_benchmarks.sh`](../scripts/run_full_benchmarks.sh) | Run all benchmark suites sequentially |
| [`run_v31_ablation.sh`](../scripts/run_v31_ablation.sh) | V3.1 ablation study launcher with conditions A-F |
| [`validate_benchmarks.py`](../scripts/validate_benchmarks.py) | Validate benchmark results for completeness |
| [`derive_ablation.py`](../scripts/derive_ablation.py) | Derive ablation conditions from raw benchmark runs |
| [`retrain_cx.py`](../scripts/retrain_cx.py) | Retrain C(x) cost field from collected embeddings |
| [`retrain_cx_phase0.py`](../scripts/retrain_cx_phase0.py) | Phase 0 C(x) initial training (597 embeddings) |
| [`retrain_lens_from_results.py`](../scripts/retrain_lens_from_results.py) | Retrain Lens models from benchmark result embeddings |
| [`collect_lens_training_data.py`](../scripts/collect_lens_training_data.py) | Collect pass/fail embeddings from benchmark runs |
| [`prepare_lens_training.py`](../scripts/prepare_lens_training.py) | Prepare and validate training data format |
| [`lib/config.sh`](../scripts/lib/config.sh) | Shared bash config: loads atlas.conf, validates paths, sets defaults |

<a id="tests"></a>
### tests/ — Test Suite

| File | Description |
|------|-------------|
| [`validate_tests.py`](../tests/validate_tests.py) | Test runner entry point |
| [`conftest.py`](../tests/conftest.py) | Pytest shared fixtures |
| **infrastructure/** | |
| [`test_llm.py`](../tests/infrastructure/test_llm.py) | llama-server health and generation tests |
| [`test_sandbox.py`](../tests/infrastructure/test_sandbox.py) | Sandbox execution tests |
| **integration/** | |
| [`test_e2e_flow.py`](../tests/integration/test_e2e_flow.py) | End-to-end pipeline flow test |
| [`test_e2e_training.py`](../tests/integration/test_e2e_training.py) | End-to-end Lens training test |
| **v3/** — 22 unit tests, one per V3 module | |
| `test_plan_search.py` `test_div_sampling.py` `test_budget_forcing.py` `test_blend_asc.py` `test_reasc.py` `test_s_star.py` `test_candidate_selection.py` `test_failure_analysis.py` `test_constraint_refinement.py` `test_pr_cot.py` `test_derivation_chains.py` `test_refinement_loop.py` `test_metacognitive.py` `test_ace_pipeline.py` `test_self_test_gen.py` `test_lens_feedback.py` `test_embedding_store.py` `test_ablation_analysis.py` `test_ewc.py` `test_replay_buffer.py` `test_enhanced_retrain.py` `test_phase4_validation.py` `test_sandbox_adapter.py` | |

<a id="docs"></a>
### docs/ — Documentation

| File | Description |
|------|-------------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Two-layer architecture with 13 Mermaid diagrams, component breakdowns, sequence diagrams |
| [`API.md`](API.md) | HTTP API reference: all endpoints for all 5 services, request/response formats |
| [`CLI.md`](CLI.md) | CLI usage, streaming output format, workflow examples, troubleshooting |
| [`CONFIGURATION.md`](CONFIGURATION.md) | Every environment variable across all services, internal constants, Aider config |
| [`MAP.md`](MAP.md) | This file — repository file map |
| [`SETUP.md`](SETUP.md) | Installation: Docker Compose, bare-metal, K3s |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Common issues and solutions |
<a id="docs-reports"></a>
### docs/reports/ — Studies, Status, Migration

| File | Description |
|------|-------------|
| [`V3_ABLATION_STUDY.md`](reports/V3_ABLATION_STUDY.md) | V3 ablation methodology: conditions A-D, 599 tasks, statistical analysis |
| [`V2_5_ABLATION_STUDY.md`](reports/V2_5_ABLATION_STUDY.md) | Historical: V2.5 Geometric Lens ablation study |
| [`V2_TO_V2_5_MIGRATION.md`](reports/V2_TO_V2_5_MIGRATION.md) | Historical: V2 to V2.5 migration guide |
| [`V3_STATUS.md`](reports/V3_STATUS.md) | Historical: V3 implementation tracking |
| [`V3_1_STATUS.md`](reports/V3_1_STATUS.md) | V3.1 implementation status and roadmap |

<a id="v3-ablation-results"></a>
### v3_ablation_results/ — Published Evidence

Per-task pass/fail data for all V3 ablation conditions. 2,396 task results across 4 conditions. See [README](../v3_ablation_results/README.md) for data format.

| Condition | Directory | Pass@1 | Tasks |
|-----------|-----------|--------|-------|
| A (baseline) | `condition_a_baseline/` | 54.9% | 599 |
| B (+Phase 1) | `condition_b_phase1/` | 67.3% | 599 |
| C (+Phase 1+2) | `condition_c_phase1_2/` | 67.3% | 599 |
| D (+Phase 1+3) | `condition_d_phase1_3/` | 74.6% | 599 |

Each condition contains `summary.json`, `v3_lcb/results.json`, and 599 per-task JSON files in `v3_lcb/per_task/`.
