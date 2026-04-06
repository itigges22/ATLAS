# ATLAS Repository Map

Visual guide to every directory and key file in the ATLAS repository.

## Top-Level Structure

| Directory | Purpose |
|-----------|---------|
| `atlas-proxy/` | Go proxy server — agent loop, grammar-constrained tool calls, Aider format translation |
| `geometric-lens/` | Geometric Lens service — C(x)/G(x) scoring, embedding extraction, training infrastructure |
| `benchmark/` | Benchmark runner and V3 pipeline modules for LiveCodeBench evaluation |
| `v3-service/` | V3 pipeline HTTP service wrapper |
| `inference/` | llama-server Dockerfiles and entrypoint scripts |
| `sandbox/` | Isolated code execution service for build verification |
| `scripts/` | Build, deploy, train, and validate scripts |
| `tests/` | Infrastructure, integration, and V3 component tests |
| `v3_ablation_results/` | Published ablation data (conditions A–D, 599 tasks each) |
| `docs/` | All documentation |
| `atlas/` | Python CLI commands (legacy, minimal) |

## atlas-proxy/ — The Agent Loop

The core of the V3.0.1 CLI architecture. Written in Go.

| File | Purpose |
|------|---------|
| `agent.go` | Agent loop: iterates tool calls until done, LLM dispatch, JSON extraction, truncation recovery, exploration budget |
| `tools.go` | 8 tool definitions + executors, per-file tier classifier (T1/T2), V3 routing |
| `aider_format.go` | Translates agent results → Aider whole-file format, streaming status, file tracking, delete fast-path, project dir detection |
| `grammar.go` | JSON schema + GBNF grammar generation for constrained output |
| `main.go` | HTTP server, `/v1/chat/completions` handler, tier classification, V3/agent loop routing |
| `v3_bridge.go` | Go ↔ Python V3 service HTTP bridge with SSE streaming |
| `v3_adapter.go` | Translates file requests into V3 pipeline format with project context |
| `build_verify.go` | Per-file-type build verification commands (tsc, py_compile, gcc, etc.) |
| `project.go` | Language/framework detection from config files |
| `permissions.go` | Permission rules, deny patterns, mode-based access control |
| `parallel.go` | `plan_tasks` tool executor with dependency graph |
| `types.go` | Shared types: ToolCall, ToolResult, AgentContext, tier definitions |
| `Dockerfile` | Multi-stage Go build for containerized deployment |

## geometric-lens/ — C(x)/G(x) Scoring

Energy-based code quality scoring without execution.

| Path | Purpose |
|------|---------|
| `geometric_lens/cost_field.py` | C(x) model: 4096→512→128→1 MLP mapping embeddings to correctness energy |
| `geometric_lens/metric_tensor.py` | G(x) model: 4096→512→4096 geometric correction field |
| `geometric_lens/service.py` | HTTP service for scoring code via C(x)/G(x) |
| `geometric_lens/training.py` | Training loop with contrastive loss, warmup, gradient clipping |
| `geometric_lens/ewc.py` | Elastic Weight Consolidation for continual learning |
| `geometric_lens/replay_buffer.py` | Experience replay buffer for training stability |
| `geometric_lens/models/` | Trained model weights (gitignored, download from HuggingFace) |
| `data/sample/` | Small training data sample (10 embeddings, 1.1MB) |
| `main.py` | FastAPI application — scoring, indexing, routing endpoints |
| `pipeline.py` | Orchestration pipeline for retrieval and scoring |

## benchmark/v3/ — V3 Pipeline Modules

20 Python modules implementing the V3 benchmark pipeline.

| Module | Phase | Purpose |
|--------|-------|---------|
| `plan_search.py` | 1 | Generate diverse implementation plans from constraints |
| `div_sampling.py` | 1 | Perturbation diversity: role/instruction/style variations |
| `budget_forcing.py` | 1 | Control thinking token budget (nothink/light/standard/deep) |
| `blend_asc.py` | 2 | Adaptive compute allocation based on probe energy |
| `reasc.py` | 2 | Early stopping for low-confidence tasks |
| `s_star.py` | 2 | Tiebreaking via edge-case input generation |
| `failure_analysis.py` | 3 | Categorize why candidates failed |
| `constraint_refinement.py` | 3 | Refine constraints from failure patterns |
| `pr_cot.py` | 3 | Progressive repair: root cause → fix → verify |
| `refinement_loop.py` | 3 | Iterative analyze→refine→generate→test cycle |
| `derivation_chains.py` | 3 | Problem decomposition into verified sub-steps |
| `metacognitive.py` | 3 | Model failure pattern awareness and compensation |
| `ace_pipeline.py` | 3 | Evolving playbook with confidence decay |
| `self_test_gen.py` | 3 | Generate test assertions from problem description |
| `candidate_selection.py` | — | Strategy-based candidate selection (lens, random, energy) |
| `lens_feedback.py` | — | C(x)/G(x) feedback integration |
| `embedding_store.py` | — | Per-task embedding storage |
| `ablation_analysis.py` | — | Ablation condition reporting |

## v3_ablation_results/ — Published Evidence

Per-task pass/fail data for all V3 ablation conditions. See [v3_ablation_results/README.md](../v3_ablation_results/README.md) for format details.

| Condition | Directory | Pass@1 |
|-----------|-----------|--------|
| A (baseline) | `condition_a_baseline/` | 54.9% |
| B (+Phase 1) | `condition_b_phase1/` | 67.3% |
| C (+Phase 1+2) | `condition_c_phase1_2/` | 67.3% |
| D (+Phase 1+3) | `condition_d_phase1_3/` | 74.6% |
