# ATLAS Configuration Reference

Complete reference for all environment variables, command-line flags, and configuration files across every ATLAS service. All settings have sensible defaults — most users only need to edit `.env`.

---

## Quick Start

```bash
cp .env.example .env
# Edit .env only if you need to change model path or ports
docker compose up -d
```

The defaults work if your model is at `./models/Qwen3.5-9B-Q6_K.gguf`.

---

## 1. Docker Compose (.env)

These variables are read by `docker-compose.yml` and control host-side port mappings and model paths. Copy `.env.example` to `.env` to configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_MODELS_DIR` | `./models` | Host path to directory containing GGUF model weights |
| `ATLAS_MODEL_FILE` | `Qwen3.5-9B-Q6_K.gguf` | Model filename (must exist in ATLAS_MODELS_DIR) |
| `ATLAS_MODEL_NAME` | `Qwen3.5-9B-Q6_K` | Model identifier used in API responses |
| `ATLAS_CTX_SIZE` | `32768` | Context window size in tokens |
| `ATLAS_LLAMA_PORT` | `8080` | llama-server host port |
| `ATLAS_LENS_PORT` | `8099` | Geometric Lens host port |
| `ATLAS_V3_PORT` | `8070` | V3 Pipeline service host port |
| `ATLAS_SANDBOX_PORT` | `30820` | Sandbox host port (container listens on 8020) |
| `ATLAS_PROXY_PORT` | `8090` | atlas-proxy host port (Aider connects here) |

Docker Compose also sets inter-service URLs using Docker networking (e.g., `http://llama-server:8080`). These are hardcoded in `docker-compose.yml` and do not need to be configured by users.

---

## 2. atlas-proxy

The Go proxy that runs the agent loop, routes tool calls, and translates between Aider and the ATLAS stack.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_PROXY_PORT` | `8090` | Port to listen on |
| `ATLAS_INFERENCE_URL` | `http://localhost:8080` | llama-server endpoint for generation |
| `ATLAS_LLAMA_URL` | (falls back to ATLAS_INFERENCE_URL) | llama-server endpoint for grammar-constrained calls |
| `ATLAS_LENS_URL` | `http://localhost:8099` | Geometric Lens scoring endpoint |
| `ATLAS_SANDBOX_URL` | `http://localhost:30820` | Sandbox code execution endpoint |
| `ATLAS_V3_URL` | `http://localhost:8070` | V3 Pipeline service endpoint |
| `ATLAS_MODEL_NAME` | `Qwen3.5-9B-Q6_K` | Model name for API responses |
| `ATLAS_AGENT_LOOP` | (unset) | Set to `1` to enable tool-call agent loop. When unset or any other value, proxy forwards to llama-server directly. |
| `ATLAS_V3_CLI` | (unset) | Set to `1` to enable V3 CLI mode (routes all generation through V3 service) |

### Internal Settings (not configurable via env)

| Setting | Value | Description |
|---------|-------|-------------|
| Max turns (T0 Conversational) | 5 | Text-only chat responses |
| Max turns (T1 Simple) | 30 | Config files, short files, data files |
| Max turns (T2 Feature) | 30 | Feature files routed through V3 pipeline |
| Max turns (T3 Hard) | 60 | Complex multi-file tasks |
| Exploration budget warning | 4 consecutive reads | Injects "write your changes now" |
| Exploration budget skip | 5+ consecutive reads | Skips the read, returns warning |
| Error loop breaker | 3 consecutive failures | Stops agent loop |
| T2 threshold | 50 lines + 3 logic indicators | Minimum for V3 activation |
| write_file rejection | Existing files > 100 lines | Forces use of edit_file |
| Conversation trim | 12 messages max | Keeps system + first user + last 8 |
| Command stdout limit | 8,000 chars | Prevents context flooding |
| Command stderr limit | 4,000 chars | Prevents context flooding |
| Search results limit | 200 matches | Prevents context flooding |
| File search skip | Files > 1 MB | Performance |
| max_tokens | 32,768 | Sent to llama-server |
| temperature | 0.3 | Sent to llama-server |

---

## 3. V3 Pipeline Service

Python HTTP service that orchestrates the V3 code generation pipeline (PlanSearch, DivSampling, Budget Forcing, PR-CoT, etc.).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_INFERENCE_URL` | `http://localhost:8080` | llama-server endpoint for generation and embeddings |
| `ATLAS_LENS_URL` | `http://localhost:8099` | Geometric Lens endpoint for C(x)/G(x) scoring |
| `ATLAS_SANDBOX_URL` | `http://localhost:30820` | Sandbox endpoint for code execution |
| `ATLAS_V3_PORT` | `8070` | Port to listen on |
| `ATLAS_MODEL_NAME` | `Qwen3.5-9B-Q6_K` | Model name for API calls |

### Internal Constants

| Setting | Value | Description |
|---------|-------|-------------|
| BASE_TEMPERATURE | 0.6 | Default generation temperature |
| DIVERSITY_TEMPERATURE | 0.8 | Temperature for diverse candidate sampling |
| MAX_TOKENS | 8,192 | Max output tokens per generation call |
| PlanSearch plans | 3 (max 7) | Number of structural plans generated |
| DivSampling perturbations | 12 | 4 roles + 4 instructions + 4 styles |
| Budget Forcing tiers | 5 | nothink (0), light (1024), standard (2048), hard (4096), extreme (8192) |
| PR-CoT perspectives | 4 | logical_consistency, information_completeness, biases, alternative_solutions |
| PR-CoT max rounds | 3 | Maximum repair attempts |
| Refinement max iterations | 2 | Maximum refinement cycles |
| Refinement time budget | 120s | Maximum time for refinement loop |
| Derivation max sub-problems | 5 | Maximum problem decomposition depth |
| Derivation max attempts/step | 3 | Retries per sub-problem |
| Constraint min cosine distance | 0.15 | Prevents hypothesis repetition |

---

## 4. Geometric Lens

Python FastAPI service for C(x)/G(x) scoring, RAG/project indexing, confidence routing, and pattern caching.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEOMETRIC_LENS_ENABLED` | `false` | Enable C(x)/G(x) scoring. Docker Compose sets this to `true`. |
| `LLAMA_URL` | `http://llama-service:8000` | llama-server endpoint. Docker Compose overrides to `http://llama-server:8080`. |
| `LLAMA_EMBED_URL` | (falls back to LLAMA_URL) | Embedding endpoint. Set separately if using a dedicated embedding server. |
| `PROJECT_DATA_DIR` | `/data/projects` | Directory for project index storage |
| `REDIS_URL` | `redis://redis:6379` | Redis connection for confidence router and pattern cache. Features using Redis degrade gracefully if unavailable. |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:8080` | Allowed CORS origins (comma-separated) |
| `CONFIG_PATH` | `/app/config/config.yaml` | Path to YAML config file (optional, defaults used if missing) |
| `API_KEYS_PATH` | `/app/secrets/api-keys.json` | Path to API keys JSON (optional) |
| `API_PORTAL_URL` | `http://api-portal:3000` | API portal URL (K3s deployment only) |

### Scoring Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| C(x) sigmoid midpoint | 19.0 | Energy value mapping to 0.5 normalized score |
| C(x) sigmoid steepness | 2.0 | Controls normalization curve sharpness |
| G(x) "likely_correct" threshold | >= 0.7 | Verdict threshold |
| G(x) "uncertain" threshold | >= 0.3 | Verdict threshold |
| G(x) "likely_incorrect" threshold | < 0.3 | Verdict threshold |

### Confidence Router

| Parameter | Value | Description |
|-----------|-------|-------------|
| CACHE_HIT route cost | 1 | Cheapest route (k=0 retrieval) |
| FAST_PATH route cost | 50 | Quick route (k=1) |
| STANDARD route cost | 300 | Default route (k=5) |
| HARD_PATH route cost | 1,500 | Expensive route (k=20) |
| BM25 k1 | 1.5 | BM25 term frequency saturation |
| BM25 b | 0.75 | BM25 document length normalization |
| Tree search max depth | 6 | LLM-guided traversal depth |
| Tree search max calls | 40 | Maximum LLM scoring calls |
| Pattern cache STM capacity | 100 | Short-term memory max entries |

---

## 5. Sandbox

Python FastAPI service for isolated code execution with compilation, linting, and testing.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_EXECUTION_TIME` | `60` | Maximum execution time in seconds |
| `MAX_MEMORY_MB` | `512` | Maximum memory per execution in MB |
| `WORKSPACE_BASE` | `/tmp/sandbox` | Base directory for execution workspaces |

### Internal Limits

| Setting | Value | Description |
|---------|-------|-------------|
| Default timeout per request | 30s | Can be overridden per request up to MAX_EXECUTION_TIME |
| stdout truncation | 4,000 chars | Last N chars kept |
| stderr truncation | 2,000 chars | Last N chars kept |
| error_message truncation | 500 chars | First N chars kept |
| Supported languages | 8 | python, javascript, typescript, go, rust, c, cpp, bash |

---

## 6. llama-server

C++ inference server (llama.cpp) with CUDA GPU acceleration and grammar-constrained JSON output.

### Docker Compose Flags

Used when running via `docker compose up`:

| Flag | Value | Description |
|------|-------|-------------|
| `--model` | `/models/${ATLAS_MODEL_FILE}` | Path to GGUF model (inside container) |
| `--host` | `0.0.0.0` | Listen on all interfaces |
| `--port` | `8080` | Listen port |
| `--ctx-size` | `${ATLAS_CTX_SIZE:-32768}` | Context window in tokens |
| `--n-gpu-layers` | `99` | Offload all layers to GPU |
| `--no-mmap` | — | Disable mmap for stability |

### K3s Entrypoint Flags (Additional)

The K3s deployment (`inference/entrypoint-v3.1-9b.sh`) uses additional flags for production performance:

| Flag | Value | Description |
|------|-------|-------------|
| `--parallel` | `4` | Parallel request slots |
| `--cont-batching` | — | Enable continuous batching |
| `--flash-attn` | `on` | Enable flash attention |
| `--mlock` | — | Lock model in RAM (prevents swapping) |
| `-b` | `4096` | Batch size |
| `-ub` | `4096` | Micro-batch size |
| `-ctk` | `q8_0` | KV cache key quantization |
| `-ctv` | `q4_0` | KV cache value quantization |
| `--embeddings` | — | Enable self-embedding endpoint |
| `--no-cache-prompt` | — | Disable prompt caching |
| `--ctx-checkpoints` | `0` | Disable context checkpoints |
| `--jinja` | — | Enable Jinja template support |
| `--port` | `8000` | K3s uses port 8000 (not 8080) |

> **Note:** Docker Compose uses a simpler configuration optimized for single-user local use. K3s uses the full production configuration with flash attention, KV cache quantization, and multi-slot parallelism.

---

## 7. Python CLI

The standalone Python REPL (`pip install -e . && atlas`) reads these variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_INFERENCE_URL` | `http://localhost:8080` | llama-server endpoint |
| `ATLAS_RAG_URL` | `http://localhost:8099` | Geometric Lens endpoint |
| `ATLAS_SANDBOX_URL` | `http://localhost:30820` | Sandbox endpoint |
| `ATLAS_MODEL_NAME` | `Qwen3.5-9B-Q6_K` | Model name for API calls |

### Generation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_tokens | 8,192 | Max output tokens |
| temperature | 0.6 | Generation temperature |
| top_k | 20 | Top-K sampling |
| top_p | 0.95 | Nucleus sampling |
| stop | `["<\|im_end\|>"]` | Stop sequence |

---

## 8. Aider Configuration

Two files in the project root control how Aider interacts with the ATLAS proxy:

### `.aider.model.settings.yml`

| Field | Value | Purpose |
|-------|-------|---------|
| `name` | `openai/atlas` | Model identifier Aider uses to find settings |
| `edit_format` | `whole` | Aider sends full file content (not diffs) |
| `weak_model_name` | `openai/atlas` | Same model for commit messages and summaries |
| `use_repo_map` | `true` | Include repository file tree in context |
| `send_undo_reply` | `true` | Notify model when user undoes a change |
| `examples_as_sys_msg` | `true` | Put few-shot examples in system prompt |
| `max_tokens` | `32768` | Must match llama-server context window |
| `temperature` | `0.3` | Low temperature for deterministic tool calls |
| `streaming` | `true` | Enable SSE streaming for real-time output |

### `.aider.model.metadata.json`

| Field | Value | Purpose |
|-------|-------|---------|
| `max_tokens` | `32768` | Max output tokens |
| `max_input_tokens` | `32768` | Max input context |
| `max_output_tokens` | `32768` | Max generation length |
| `input_cost_per_token` | `0` | Free (local inference) |
| `output_cost_per_token` | `0` | Free (local inference) |
| `litellm_provider` | `openai` | OpenAI-compatible API protocol |
| `mode` | `chat` | Chat completion mode |

---

## 9. K3s Configuration (atlas.conf)

For K3s deployment only. Copy `atlas.conf.example` to `atlas.conf` and edit:

```bash
# Model
ATLAS_MODELS_DIR="$HOME/models"
ATLAS_MAIN_MODEL="Qwen3.5-9B-Q6_K.gguf"

# Inference
ATLAS_CONTEXT_LENGTH=40960        # Per-slot context (× PARALLEL_SLOTS)
ATLAS_GPU_LAYERS=99
ATLAS_PARALLEL_SLOTS=4
ATLAS_FLASH_ATTENTION=true

# Service NodePorts
ATLAS_LLAMA_NODEPORT=32735
ATLAS_LENS_NODEPORT=31144
ATLAS_SANDBOX_NODEPORT=30820

# Geometric Lens
ATLAS_ENABLE_LENS=true
ATLAS_LENS_CONTEXT_BUDGET=6000

# K3s
ATLAS_NAMESPACE=atlas
```

See `atlas.conf.example` for the full documented template with all available options. K3s manifests are generated from these values by `scripts/generate-manifests.sh`.

> **Note:** `atlas.conf` is only used by K3s deployment scripts. Docker Compose uses `.env` instead. The two files configure different deployment targets and should not be mixed.
