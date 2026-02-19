# ATLAS Architecture

Detailed architecture companion to the README. Covers the V2 verified system as deployed and benchmarked on 2026-02-17.

---

## 1. System Overview

ATLAS is a self-hosted benchmark infrastructure for evaluating LLM code generation. It runs a local Qwen3-14B model on consumer hardware (single 16GB GPU) under K3s, combining retrieval-augmented generation, energy-based verification, and adaptive routing to maximize pass rates within a fixed VRAM budget.

### Data Flow

```
                         K3s Cluster
  +-------------------------------------------------------------+
  |                                                             |
  |  problem                                                    |
  |    |                                                        |
  |    v                                                        |
  |  +-------------+    prompt    +---------------+             |
  |  |   rag-api   | ----------> | llama-server  |             |
  |  | (router,    | <---------- | (Qwen3-14B)   |             |
  |  |  PageIndex, |  completion | :8000          |             |
  |  |  lens)      |             | NodePort:32735 |             |
  |  | :8001       |             +---------------+             |
  |  | NodePort:   |                                            |
  |  |  31144      |                                            |
  |  +------+------+                                            |
  |         |                                                   |
  |         | code candidate                                    |
  |         v                                                   |
  |  +-------------+         +----------+                       |
  |  |   sandbox   |         |  redis   |                       |
  |  | (exec+test) |         | (cache,  |                       |
  |  | :8020       |         |  state)  |                       |
  |  +------+------+         +----------+                       |
  |         |                                                   |
  |         v                                                   |
  |       result (pass/fail)                                    |
  |                                                             |
  |  +-------------+                                            |
  |  |  dashboard  |  (scaled to 0)                             |
  |  +-------------+                                            |
  +-------------------------------------------------------------+
```

### Control Flow (Best-of-K Pipeline)

```
  router                         generation           lens scoring        selection
    |                                |                      |                 |
    v                                v                      v                 v
  4 signals --> difficulty D(x) --> k candidates --> score C(x) each --> sort by energy
  (cache, retrieval,                (parallel,        (cost field MLP)    --> sandbox lowest
   complexity, energy)               staggered)                            --> early exit on pass
```

### K3s Pods

| Pod | Purpose |
|-----|---------|
| llama-server | GPU inference (Qwen3-14B), embeddings |
| rag-api | Orchestration, routing, PageIndex, Geometric Lens |
| redis | Pattern Cache, Thompson Sampling state, AOF persistence |
| sandbox | Isolated code execution and test validation |
| api-portal | User authentication, API key management, JWT issuance |
| llm-proxy | Rate-limited OpenAI-compatible proxy in front of llama-server |
| task-worker | Async task execution with Ralph Loop retry logic |
| dashboard | Monitoring UI (scaled to 0 by default) |
| atlas-nightly-training | CronJob for nightly LoRA fine-tuning (runs if ATLAS_ENABLE_TRAINING=true) |

### NodePorts

| Service | NodePort |
|---------|----------|
| llama-server | 32735 |
| rag-api | 31144 |
| api-portal | 30000 |
| llm-proxy | 30080 |
| sandbox | 30820 |
| dashboard | 30001 |

---

## 2. Components

### 2.1 llama-server (Inference Engine)

GPU-accelerated LLM inference via llama.cpp, serving OpenAI-compatible API endpoints.

**Models**:

| Model | Format | Size |
|-------|--------|------|
| Qwen3-14B-Q4_K_M | GGUF (4.87 BPW) | 8.38 GiB |
| Qwen3-0.6B-Q8_0 (draft) | GGUF | 610 MiB |

The speculative decode draft model is partially broken in current deployment. It is loaded but acceptance rates are low.

**Verified Flag List**:

```
--ctx-size 40960          # 20480 per slot
--parallel 2
--cont-batching
--flash-attn
--no-mmap
--jinja
--cache-type-k q4_0
--cache-type-v q4_0
--reasoning-format deepseek
--embeddings
--draft-max 16
--draft-min 5
--draft-p-min 0.9
--ngl 99
--mlock                   # FAILED (insufficient privileges)
```

**VRAM Budget**: 12,235 / 16,311 MiB (75.0%). See Section 4 for breakdown.

**Throughput**: 109 tasks/hr (V2 benchmark aggregate across all datasets).

**Deployment Notes**:
- Strategy: Recreate (single GPU, cannot run 2 pods simultaneously).
- Entrypoint managed via ConfigMap `llama-entrypoint` to avoid full image rebuilds (llama.cpp compiles from source, pins all CPU/RAM during build).
- Model files mounted from the host path configured in `manifests/llama-deployment.yaml` (default: `/opt/atlas/models/`).
- First request after model load is slow (cold KV cache); warm up before benchmarking.
- Qwen3-14B thinking mode: at temperature=0, model uses `<think>` tags consuming 8000+ tokens. Must set max_tokens >= 16384 or use `/nothink`.

### 2.2 Geometric Lens (Energy-Based Verifier)

An energy-based model that scores code generation candidates, enabling the best-of-K selection pipeline to pick the most likely correct candidate before running expensive sandbox tests.

**Theoretical Foundation**: ARM-EBM duality (Blondel et al., 2025). The cost field C(x) maps embedding-space points to scalar energy values, where low energy correlates with correct solutions and high energy with incorrect ones.

**Architecture**:

C(x) Cost Field (~2.7M params):
```
Linear(5120 -> 512) + SiLU
Linear(512 -> 128)  + SiLU
Linear(128 -> 1)    + Softplus
```

G(x) Metric Tensor (~5.2M params):
```
Linear(5120 -> 512) + SiLU
Linear(512 -> 5120) + Softplus
```

**Core Equation**:
```
delta_x = -alpha * G(x)^{-1} * grad_C(x)
```

The correction vector delta_x indicates the direction in embedding space that would reduce energy, providing a geometric interpretation of what the model "should have generated."

**Training**: Contrastive ranking loss on real benchmark pass/fail data. Self-supervised -- no human labels required beyond the sandbox's binary pass/fail signal.

**Verified Performance** (V2 benchmark run):

| Metric | Value |
|--------|-------|
| Validation AUC | 0.968 (Epoch 3) |
| Selection efficiency | 100% (188/188) |
| First-try accuracy | 80% |
| PASS energy (mean) | 5.00 |
| FAIL energy (mean) | 14.04 |
| Energy separation | 9.04 |
| Correction magnitude | \|\|delta_x\|\| / \|\|x\|\| = 0.0006 |
| Latency | ~75ms per evaluation |

Energy values are raw model outputs (not normalized). Training targets: PASS=2.0, FAIL=25.0. Measured: PASS mean=5.00, FAIL mean=14.04.

**Weights**: `rag-api/geometric_lens/models/cost_field.pt`, `metric_tensor.pt`. Baked into the container image. PyTorch CPU only (torch 2.10.0+cpu).

**Embedding Source**: llama-server `--embeddings` flag enables the `/embedding` endpoint, which returns per-token 5120-dimensional vectors, mean-pooled to a single vector per input.

**Environment**: `GEOMETRIC_LENS_ENABLED` env var. Models loaded lazily on first use.

### 2.3 Confidence Router

Adaptive per-task routing that decides how many candidates (k) to generate based on estimated problem difficulty. Uses Thompson Sampling (Bayesian bandit) with Redis-backed Beta(alpha, beta) posteriors, updated with real outcomes.

**4 Input Signals**:

| Signal | Source | Range | Description |
|--------|--------|-------|-------------|
| pattern_cache_score (s_p) | Pattern Cache | 0.0--1.0 | Cache hit confidence |
| retrieval_confidence (r_c) | PageIndex | 0.0--1.0 | Retrieval quality score |
| query_complexity (q_c) | Query analysis | 0.0--1.0 | Estimated difficulty |
| geometric_energy (g_e) | Geometric Lens | 0.0--1.0 (normalized) | Cost field energy |

**Difficulty Fusion**:
```
D(x) = 0.30 * (1 - s_p) + 0.25 * (1 - r_c) + 0.20 * q_c + 0.25 * g_e
```

**Thompson Sampling**: Beta(alpha, beta) parameters maintained per (difficulty_bin, route) pair in Redis. On each task, the router samples from each route's Beta distribution and selects the route with the highest sampled value. Outcomes (pass/fail) update the winning route's parameters.

**Routes**:

| Route | k (candidates) | When Selected |
|-------|----------------|---------------|
| CACHE_HIT | 0 | High cache confidence, known pattern |
| FAST_PATH | 1 | Low difficulty, single attempt sufficient |
| STANDARD | 5 | Medium difficulty, moderate exploration |
| HARD_PATH | 20 | High difficulty, extensive exploration |

**Environment**: `ROUTING_ENABLED` env var. When disabled (Mode B), all tasks use STANDARD route (k=5).

**Location**: `rag-api/router/` -- signal_collector, difficulty_estimator, route_selector, feedback_recorder, fallback_chain.

### 2.4 Best-of-K Selection Pipeline

The selection pipeline generates k candidates in parallel, scores them with the Geometric Lens, and tests them in the sandbox in energy-sorted order with early exit on first pass.

**Generation**:
- ThreadPoolExecutor with max_workers=2.
- 100ms stagger between candidate submissions to avoid llama-server slot contention.
- cache_prompt=True for KV cache reuse across candidates sharing the same prefix.

**Seed Determinism**: Each candidate i uses seed = index * 42 + 1, ensuring reproducibility.

**Temperature Schedule**:

| Condition | Temperature |
|-----------|-------------|
| k <= 1 (single attempt) | 0.0 |
| MCQ or IFBench | 0.3 |
| k <= 5 (code tasks) | 0.6 |
| k > 5 (code tasks) | 0.8 |

**Selection Process**:
1. Generate k candidates via llama-server `/v1/chat/completions`.
2. Extract code from each response.
3. Score each candidate via `/internal/lens/score-text` (Geometric Lens energy).
4. Sort candidates by energy (ascending -- lowest energy first).
5. Run sandbox tests in sorted order.
6. Early exit: return the first candidate that passes all tests.

### 2.5 PageIndex RAG

Replaced V1's Qdrant vector database with AST-aware retrieval.

**Indexing Pipeline**:
1. **AST Parsing** (`indexer/ast_parser.py`): tree-sitter + tree-sitter-python parses source into syntax trees.
2. **Tree Building** (`indexer/tree_builder.py`): Hierarchical TreeIndex -- modules, classes, functions.
3. **BM25 Indexing** (`indexer/bm25_index.py`): Term-frequency index over node content.
4. **Summarization** (`indexer/summarizer.py`): LLM-generated summaries for tree nodes.
5. **Persistence** (`indexer/persistence.py`): JSON to `/data/projects/{project_id}/tree_index.json` + `bm25_index.json`.

**Retrieval Strategies**:

| Strategy | Module | Description |
|----------|--------|-------------|
| Tree Search | `retriever/tree_search.py` | LLM-guided traversal of AST tree |
| BM25 Search | `retriever/bm25_search.py` | Token-based keyword matching |
| Hybrid | `retriever/hybrid_router.py` | Combines tree + BM25, deduplicates |

**Caching**: Lazy in-memory cache in `rag.py` (`_pageindex_cache`), invalidated on project sync.

**Provides**: `retrieval_confidence` signal to the Confidence Router.

### 2.6 Pattern Cache (Redis)

Stores problem-to-strategy patterns for cache routing, implementing Ebbinghaus memory decay.

**Tiers**:

| Tier | TTL | Purpose |
|------|-----|---------|
| STM (Short-Term Memory) | 1 hour | Recent query-result pairs |
| LTM (Long-Term Memory) | 7 days | Frequently reinforced patterns |

Patterns accessed frequently are promoted from STM to LTM. Unused patterns decay naturally via TTL expiration.

Also serves as the backend for Thompson Sampling router state (Beta distribution parameters per route).

### 2.7 Sandbox

Isolated code execution environment providing ground-truth pass/fail signals. This is the objective verification layer -- the model cannot corrupt this signal.

**Execution Modes**:

| Mode | Field | Description |
|------|-------|-------------|
| function | test_code | Python test assertions run against generated function |
| stdio | test_inputs / test_outputs | Input piped to stdin, output compared to expected |
| mcq | N/A | Multiple-choice answer extraction and comparison |
| ifbench | N/A | Instruction-following constraint evaluation |

---

## 3. V2 Benchmark Results

Run ID: `v2_run_20260217_125310`
Hardware: RTX 5060 Ti 16GB VRAM
Throughput: 109 tasks/hr aggregate

All results from a single benchmark run. Not averaged across multiple runs; variance unknown.

### Benchmark Scores

| Benchmark | Tasks | pass@1 | Conditions |
|-----------|-------|--------|------------|
| LiveCodeBench v5 | 599 | 36--41% | k=3, Geometric Lens selection |
| GPQA Diamond | 198 | 47.0% | k=5, MCQ format |
| IFBench | 300 | excluded | k=5, instruction following (see note) |
| Custom | 100 | 53--55% | k=1, real-world tasks |
| SciCode | ~80 | 14.7% sub / 5.0% main | k=1, scientific computing |

### Geometric Lens Verification

| Metric | Value |
|--------|-------|
| Validation AUC | 0.968 (Epoch 3) |
| Selection efficiency | 100% (188/188 correct selections) |
| PASS energy (mean) | 5.00 |
| FAIL energy (mean) | 14.04 |
| Energy separation | 9.04 |

### Dataset Notes

- **LiveCodeBench**: Source mirror bzantium/livecodebench (primary livecodebench/code_generation_lite returns 404). 599 tasks, stdio evaluation, AA prompt format.
- **GPQA Diamond**: Gated on HuggingFace; sourced from openaipublic.blob.core.windows.net CSV. 198 multiple-choice questions.
- **IFBench**: IFBench evaluation is incomplete -- evaluate_ifbench_loose() defaults to True for ~11/15 instruction categories. The 100% score in benchmark results reflects broken evaluation logic, not model capability. IFBench is excluded from headline results pending proper implementation.
- **Custom**: 100 tasks from benchmark/custom/tasks.json.
- **SciCode**: ~80 problems. Known parsing issue (list-in-join TypeError in _build_step_test_code). Stretch goal; execution environment lacks numpy.

---

## 4. VRAM Budget

Total available: 16,311 MiB (RTX 5060 Ti 16GB).

| Component | Size |
|-----------|------|
| Model weights (Q4_K_M) | ~8,380 MiB |
| KV cache (2 slots x 20480 ctx x q4_0) | ~1,800 MiB |
| Draft model (Qwen3-0.6B-Q8_0) | ~610 MiB |
| CUDA overhead | ~1,445 MiB |
| **Total** | **~12,235 MiB (75.0%)** |

KV cache size depends on quantization method and context window. With q4_0 at 40960 context (2 slots x 20480): ~1,800 MiB.

Headroom: ~4,076 MiB. Sufficient for inference but does not permit increasing slot count or context size significantly without evicting the draft model.

---

## 5. MaaS Layer (Optional)

ATLAS includes an optional Model-as-a-Service layer for external API access to the model. These services are not required for benchmarking or internal use.

### api-portal (NodePort 30000)

User-facing gateway providing authentication, API key management, and model discovery.

- **Auth**: JWT-based login/register + `sk-llm-` prefixed API keys
- **Key endpoints**: `/api/auth/register`, `/api/auth/login`, `/api/keys` (CRUD), `/api/validate-key` (internal)
- **Model discovery**: Proxies to llama-server `/v1/models` and `/props`
- **Database**: SQLite (`portal.db`) for users, keys, usage logs

### llm-proxy (NodePort 30080)

Rate-limited OpenAI-compatible proxy. All requests require a valid API key (validated against api-portal).

- **Key endpoints**: `/v1/chat/completions`, `/v1/models`, catch-all proxy to llama-server
- **Rate limiting**: Per-key sliding window (60s) via Redis
- **Streaming**: Full SSE support for streaming completions
- **Metrics**: Logged to Redis (`metrics:daily:{date}`)

### task-worker (ClusterIP)

Async task processor implementing the Ralph Loop (retry-until-success code generation). Polls Redis task queues (`tasks:p0`, `tasks:p1`, `tasks:p2`) and executes code generation with sandbox validation.

- **Ralph Loop**: Up to 5 attempts with escalating temperature (0.3→1.0), error feedback between attempts
- **Services used**: llama-server (direct, no auth), rag-api (optional context), sandbox (execution)
- **No HTTP API** — communicates only via Redis queues

### atlas-nightly-training (CronJob — Suspended)

LoRA fine-tuning cronjob. Currently suspended (`suspend: true`). V2 uses a frozen base model; the only model adaptation is Geometric Lens retraining during benchmark runs.

---

## 6. Removed in V2

The following V1 components were removed and replaced:

| V1 Component | V2 Replacement | Reason |
|-------------|----------------|--------|
| Qdrant vector database | PageIndex (AST tree + BM25) | Structural code understanding, lower resource usage |
| Dedicated embedding service | llama-server `--embeddings` | Single model serves both inference and embeddings |
| Chunking pipeline (`rag-api/chunker.py`) | AST-based tree indexing | Chunk boundaries are semantic (functions, classes) rather than arbitrary token windows |

Removed manifests: `embedding-deployment.yaml`, `qdrant-deployment.yaml`.
Removed modules: `rag-api/chunker.py`, `rag-api/vector_store.py`.
