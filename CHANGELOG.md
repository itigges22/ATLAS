# Changelog

## Unreleased

### Documentation
- Added multilingual documentation: Simplified Chinese (zh-CN), Japanese (ja), Korean (ko) for README, SETUP, and TROUBLESHOOTING
- Added language selector badges to README
- Added star history chart to Latest News section
- Rewrote README contributing section to encourage issue reports and community feedback
- Fixed V3_1_STATUS.md false claims about speed optimizations that were never applied to code

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

## [3.0.1] - 2026-04-05

### Tool-Call Agent Loop Architecture
- Replaced Aider format-translation proxy with structured JSON tool-call agent loop
- Grammar-constrained output via llama-server `response_format:json_object` — 100% valid JSON
- 8 tool definitions: `read_file`, `write_file`, `edit_file`, `delete_file`, `run_command`, `search_files`, `list_directory`, `plan_tasks`
- Per-file tier classification: T1 (config/data) writes directly, T2 (logic/features) routes through V3 pipeline
- 3400+ lines new Go code across 12 files in `atlas-proxy/`

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
