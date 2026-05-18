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
| `ATLAS_CTX_SIZE` | `32768` | Context window size in tokens (mapped to `CONTEXT_LENGTH` inside the llama container) |
| `ATLAS_PROJECT_DIR` | (cwd at `compose up`) | Host directory bind-mounted to `/workspace` inside the atlas-proxy container. Switch projects by re-creating the proxy container with this var set. |
| `ATLAS_GHCR_OWNER` | `itigges22` | GHCR namespace to pull images from. Set to your own GitHub username if you've published forked images. |
| `ATLAS_IMAGE_TAG` | `latest` | Image tag to pull (`latest` for main, `dev` for the dev branch, `vX.Y.Z` or `sha-...` for pinned releases). |
| `ATLAS_LLAMA_PORT` | `8080` | llama-server host port |
| `ATLAS_LENS_PORT` | `8099` | Geometric Lens host port |
| `ATLAS_V3_PORT` | `8070` | V3 Pipeline service host port |
| `ATLAS_SANDBOX_PORT` | `30820` | Sandbox host port (container listens on 8020) |
| `ATLAS_PROXY_PORT` | `8090` | atlas-proxy host port (TUI and OpenAI-compat clients connect here) |
| `ATLAS_BACKEND` | `cuda` | Inference backend. `cuda` (NVIDIA, V3.1.0+), `rocm` (AMD, V3.1.1), `metal` (Apple Silicon, V3.1.2 planned — native install only), `sycl` (Intel Arc, roadmap). Set by `atlas init`; the entrypoint scripts read this to pick per-vendor env vars. ROCm also requires bringing up the stack with `-f docker-compose.rocm.yml` (the wizard prints the right command). |
| `ATLAS_GPU_VENDOR` | (auto-detected) | Vendor of the GPU ATLAS should use: `nvidia`, `amd`, `apple`, `intel`. Only meaningful on multi-vendor hosts; auto-detect picks the largest-VRAM GPU. |
| `ATLAS_GPU_INDEX` | `0` | Vendor-local index of the GPU ATLAS should use. The entrypoint sets `CUDA_VISIBLE_DEVICES` (NVIDIA) or `HIP_VISIBLE_DEVICES` + `ROCR_VISIBLE_DEVICES` (AMD) from this value. Multi-GPU hosts pick a specific card with this. |
| `ATLAS_GFX_TARGET` | `gfx1100;gfx1101;gfx1102;gfx1030;gfx90a` | **ROCm only.** AMD compute target(s), semicolon-separated. Forwarded to `Dockerfile.rocm` as `AMDGPU_TARGETS` at build time. Trim to your GPU for a smaller image — see [SETUP.md § AMD GPU Targets](SETUP.md#amd-gpu-targets-dockerfilerocm-v311). |
| `ATLAS_ROCM_TAG` | `6.2-complete` | **ROCm only.** Base image tag for `rocm/dev-ubuntu-22.04`. Bump when you want to test a newer ROCm release. |
| `ATLAS_HSA_OVERRIDE_GFX_VERSION` | (unset) | **ROCm only.** Force a specific HSA gfx version at runtime — workaround for "officially unsupported" GPUs (e.g., older Vega) that still work with a compatible target. Example: `10.3.0` makes RDNA1 cards masquerade as RDNA2 for HIP kernel selection. |

Docker Compose also sets inter-service URLs using Docker networking (e.g., `http://llama-server:8080`). These are hardcoded in `docker-compose.yml` and do not need to be configured by users.

#### Backend-vs-Compose-override matrix

| `ATLAS_BACKEND` | Required compose invocation |
|---|---|
| `cuda` (default) | `docker compose up -d` |
| `rocm` | `docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d` |
| `metal` / `sycl` | Not supported via Docker — see [SETUP.md § Method 1](SETUP.md) for native paths (V3.1.2+) |

`atlas init` prints the right invocation as part of its "Next steps" summary. `atlas-bootstrap.sh` picks it automatically based on `tier.detect_gpu()`.

---

## 2. atlas-proxy

The Go proxy that runs the agent loop, routes tool calls, and orchestrates the ATLAS pipeline (llama-server + Lens + V3 + sandbox).

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
| `ATLAS_KEEP_LLAMA_WARM` | `1` | Set to `0` to disable the keep-warm goroutine that pings llama-server every 45s with a 1-token completion. Keeping warm avoids the cold-start path that fires after 1-2 min idle (see ISSUES.md PC-035). Disable for CPU-only or tightly power-budgeted setups. |
| `ATLAS_FRESH_SLOT_PER_SESSION` | `1` | Set to `0` to disable per-session llama.cpp KV-slot erase. With it enabled (default), the proxy POSTs `/slots/0?action=erase` at the start of each agent loop invocation, giving each turn a clean cache. Adds ~1-2s to the first turn but prevents cross-session token-state leakage (e.g. filenames hallucinated from prior sessions). See ISSUES.md PC-045. |
| `ATLAS_MAX_TURNS` | (unset) | Operator override for the agent-loop turn cap. Any positive int caps all tiers; unset / `0` / invalid falls through to tier defaults (T0=5, T1/T2/T3=uncapped). See `proxy/types.go:envOverrideMaxTurns`. |
| `ATLAS_CONTROL_VECTOR` | `/models/ast_edit_steering.gguf` | Path to the ASA control-vector GGUF for ast_edit steering. Auto-loaded if the file exists; ignored otherwise. |
| `ATLAS_CONTROL_VECTOR_SCALE` | `0.5` | Strength multiplier applied to the control vector via `--control-vector-scaled`. |
| `ATLAS_CONTROL_VECTOR_LAYER_RANGE` | (unset) | Restrict the control vector to a layer band. Format is two space-separated integers (e.g. `"24 30"`) — passed straight to llama-server's `--control-vector-layer-range start end`. Unset applies it to all layers. |
| `ATLAS_WORKSPACE_DIR` | (proxy's container `/workspace`) | Working-dir override that the proxy substitutes for the TUI-supplied `working_dir` field. Set inside the container so file tools always resolve under `/workspace` regardless of what the client sends. |
| `ATLAS_VERIFY_IN` | `sandbox` | Where `run_command` and the V3 verify path execute: `sandbox` (default) routes through the sandbox container; `host` runs commands directly on the proxy host (only safe when the proxy itself is local, not containerized). Per-project override: `[execution] target = "host"` in `.atlas/config.toml`. (PC-192) |

### Internal Settings (not configurable via env)

| Setting | Value | Description |
|---------|-------|-------------|
| Max turns (T0 Conversational) | 5 | Text-only chat — shape constraint, not runaway protection |
| Max turns (T1 / T2 / T3) | `0` (uncapped) | Removed May 10 2026. The 8 stuck-pattern detectors (parse-error, tool-repeat, reasoning-repeat, lens-regression, exploration-budget, path-aware error-loop, action-gate, verification-gate) are the safety net. Operator can re-cap any tier with `ATLAS_MAX_TURNS=<n>`. |
| Exploration budget warning | 4 consecutive reads | Injects "write your changes now" |
| Exploration budget skip | 5+ consecutive reads | Skips the read, returns warning |
| Error loop breaker | 3 consecutive failures on the **same path** | Path-aware — same `(tool, path)` 3× breaks the loop; rotating failure paths do not trip it (see `proxy/agent.go:838-877`) |
| T2 trigger (V3 activation) | `lines ≥ 10` AND (`hasLogicIndicators` ≥ 2 family matches OR known code/markup extension) | `classifyFileTier` in `proxy/tools.go`. Config files / data exts / styles / prose / shell scripts always T1; under 10 lines always T1; recognized code/markup extensions auto-T2 even without logic-indicator matches. |
| write_file rejection | Existing files > 5 lines | Forces `ast_edit` (whole node, .py/.html/.htm) or `edit_file` (surgical). Skipped when the existing file looks corrupted on disk (PC-201 self-heal). |
| Suspicious-shrinkage guard | `oldSize ≥ 100B` AND `newSize < 64B` | Rejects writes that replace a non-trivial file with a stub (doctype-only / mid-output cut). See `validateNotSuspiciouslyShrunk`. |
| Per-step grammar gate | Trigger: write_file rejection on existing .py/.html/.htm > 5 lines | Bans `edit_file` and `write_file` from GBNF tool-name production for next decision (BiasBusters #2/#3) |
| ASA control vector | Auto-loaded from `/models/ast_edit_steering.gguf` if present (always-on by default) | Activates llama-server `--control-vector-scaled` whenever the file exists at the standard path. Override path with `ATLAS_CONTROL_VECTOR`; tune with `ATLAS_CONTROL_VECTOR_SCALE` (default 0.5) and `ATLAS_CONTROL_VECTOR_LAYER_RANGE` (default all layers). Workflow: `geometric-lens/asa_calibration/README.md` |
| Conversation trim | Trigger: `> 12` messages | Keeps `system + most-recent user message (pinned) + last 8` (`trimMessages` in `proxy/agent.go`). The pin is the most-recent `role=="user"` message so long tool-call chains don't push the user's task off the tail. |
| Command stdout limit | 8,000 chars | Prevents context flooding |
| Command stderr limit | 4,000 chars | Prevents context flooding |
| Search results limit | 200 matches | Prevents context flooding |
| File search skip | Files > 1 MB | Performance |
| max_tokens | 32,768 | Sent to llama-server |
| temperature | 0.3 default; 0.7 on retry after a stuck-loop nudge | Sent to llama-server |

### Stuck-pattern detectors

These are the 8 safety detectors that replaced the per-tier turn cap on 2026-05-10. Each fires independently — first match breaks the loop or injects a corrective.

| Detector | Threshold | Source |
|----------|-----------|--------|
| Tool-call repetition | Same `(tool, args)` signature `3×` within the last `8` calls | `proxy/tool_repeat.go` (`toolRepeatThreshold=3`, `toolRepeatWindow=8`) |
| Reasoning repetition | Same reasoning snippet `2×` consecutive turns | `proxy/reasoning_repeat.go` (`reasoningRepeatThreshold=2`) |
| Lens regression | `gx_score_min` runs `2` consecutive turns below `0.15`, OR a single turn below `0.05` (severe) | `proxy/lens_score.go` (`lensRegressionRunLength=2`, `lensLowScoreThreshold=0.15`, `lensSevereThreshold=0.05`) |
| Exploration-budget | 4 consecutive read-only calls → nudge; 5+ → skip | `proxy/agent.go:953` |
| Path-aware error-loop | 3 consecutive failures on the **same** path (rotating paths don't trip) | `proxy/agent.go:838-877` (`consecutiveErrors >= 3` + path match) |
| Action gate | Turn emits `done` but the user prompt has action-intent and no successful write/edit/ast_edit fired this loop | `proxy/agent.go:404` (action_gate) |
| Verification gate | Turn emits `done` after a fix-intent prompt with no successful verification command this loop | `proxy/agent.go:375` (verification_gate) |
| Claim-check gate | `done` summary makes universal claims (`works perfectly`, `tested all routes`) without backing evidence, OR the prompt asks for multi-issue work | `proxy/agent.go:441` + `proxy/claim_check.go` |

### Plan-mode auto-revision (PC-205 / plan_adherence)

| Setting | Value | Description |
|---------|-------|-------------|
| `planAutoReviseThreshold` | `5` | Consecutive off-plan tool calls before the proxy regenerates the plan |
| `planMaxRevisions` | `2` | Hard cap on auto-revisions per loop (prevents revision oscillation) |

### Hard-blocked patterns (`DefaultDenyPatterns`)

These pattern matches in `proxy/permissions.go:shouldDenyToolCall` are checked BEFORE the per-tool permission gate, so `yolo` mode does not bypass them.

| Tool | Pattern | Behavior |
|------|---------|----------|
| `run_command` | `rm -rf /`, `rm -rf /*`, `mkfs*`, `dd if=*of=/dev/*` | Reject with "tool call refused" — host-destroying commands |
| `write_file` | `.env`, `*.pem`, `*.key`, `*credentials*` | Reject — model can't accidentally clobber secret files |

These are pattern-`Contains` matches against the command / file path, so the `.env` pattern catches `app/.env`, `.env.local`, etc.

### Shell-mutation gate

`run_command` is rejected on the leading verb when it would mutate user files — destructive ops must use the dedicated `write_file` / `edit_file` / `delete_file` tools (which go through V3 + the surgical-edit gate + audit logging). Source: `proxy/guardrails.go:validateShellCommand`.

| Pattern | Reaction |
|---------|----------|
| Leading verb in `{rm, mv, cp, rmdir, chmod, chown, truncate}` | Reject |
| `find … -delete` or `find … -exec rm` | Reject |
| `bash -c "…"` / `sh -c "…"` / `zsh -c "…"` / `dash -c "…"` / `eval …` | Reject (wrappers hide arbitrary commands from the per-segment check) |
| Truncating `> /path` redirect | Reject — except `/dev/null`, `/dev/stderr`, `.log`, `.out` (intentional log capture) |

These checks are enforced regardless of permission mode — `yolo` does NOT bypass them.

### Symbol indexing (per-session startup)

`proxy/symbol_index.go` scans the project once per `/v1/agent` session to seed a symbol → file map so the planner can resolve names like `dashboard` to `app/dashboard.py` without re-reading the tree on every turn.

| Setting | Value | Description |
|---------|-------|-------------|
| `projectScanMaxFiles` | `50` | Max `.py` files read during the scan |
| `projectScanMaxBytes` | `500 KB` | Total source-byte budget across all scanned files |
| `projectScanTimeout` | `5 s` | Round-trip cap on the v3-service call; falls back to "no injection" on timeout |
| `symbolMaxCandidates` | `10` | Max symbols extracted from a user message before regex-order truncation |

### Event broker (`/events` SSE)

| Setting | Value | Description |
|---------|-------|-------------|
| `subscriberBuffer` | `256` | Per-subscriber channel buffer. Slow consumers start dropping events once the buffer fills. |

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
| `ATLAS_PLAN_THINKING` | `0` | Enable Qwen3.5 thinking mode during V3 plan generation. `0` keeps planner `max_tokens=2048` (fast); `1` raises it to `8192` so the model can emit reasoning before the plan JSON. Off by default — thinking pushes planner latency from ~5-30 s to >4 min per candidate on tight hardware (PC-206). |

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
| `LLAMA_URL` | `http://llama-server:8080` | llama-server endpoint. Read by `config.py:LlamaConfig` and also by `embedding_extractor.py` as the embedding source. |
| `LLAMA_EMBED_URL` | (falls back to `LLAMA_URL`) | Dedicated embedding endpoint. Use this if you have a separate embedding server; otherwise embeddings reuse the LLAMA_URL host. |
| `ROUTING_ENABLED` | `true` | Master switch for the confidence-router pipeline. Setting `false` short-circuits routing and uses STANDARD for every query. |
| `PROJECT_DATA_DIR` | `/data/projects` | Directory for project index storage |
| `REDIS_URL` | `redis://redis:6379` | Redis connection for confidence router and pattern cache. Features using Redis degrade gracefully if unavailable. |
| `SANDBOX_URL` | `http://sandbox:8020` | Sandbox endpoint used by the lens's own `sandbox_client.py` (separate from `ATLAS_SANDBOX_URL` read by atlas-proxy). |
| `SANDBOX_TIMEOUT` | `30` | Per-request timeout (seconds) when the lens itself calls the sandbox. |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:8080` | Allowed CORS origins (comma-separated) |
| `CONFIG_PATH` | `/app/config/config.yaml` | Path to YAML config file (optional, defaults used if missing) |
| `API_KEYS_PATH` | `/app/secrets/api-keys.json` | Path to API keys JSON. The lens's `/v1/*` endpoints return 401 until a key file is mounted. |
| `EMBEDDING_DIM` | `768` | Fallback embedding dimension used by `training.py` when not inferrable from saved weights. (Runtime models use the dim baked into the GGUF; this only matters when training from scratch.) |

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

### Project Indexing Limits (YAML-overridable via `CONFIG_PATH`)

These come from `geometric-lens/config.py:LimitsConfig` and `RetrievalConfig`. They are NOT env-var-configurable on their own — override by mounting a YAML file at `CONFIG_PATH` with matching nested keys.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `limits.max_files` | 10,000 | Per-project file cap during indexing |
| `limits.max_loc` | 500,000 | Per-project lines-of-code cap |
| `limits.max_size_mb` | 100 | Per-file size cap (MB) — larger files are skipped |
| `limits.project_ttl_hours` | 24 | Project index TTL before re-index |
| `retrieval.top_k` | 20 | Default top-K returned per retrieval call |
| `retrieval.context_budget_tokens` | 8,000 | Max tokens of retrieved context returned to the caller |

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
| Default timeout per request | 30s | Can be overridden per request up to `MAX_EXECUTION_TIME` |
| `/shell` stdout truncation | 4,000 chars | Last N chars kept |
| `/shell` stderr truncation | 2,000 chars | Last N chars kept |
| `error_message` truncation | 500 chars | First N chars kept on `/execute` failures |
| Timeout error preview | 50 chars | Tail of stdout/stderr shown when `/execute` times out |
| Supported languages | 8 | python, javascript, typescript, go, rust, c, cpp, bash |

### Background process tools (PC-196)

`run_background` / `tail_background` / `stop_background` are sandbox-only — they require `ATLAS_VERIFY_IN=sandbox` (the default).

| Setting | Value | Description |
|---------|-------|-------------|
| `BG_MAX_LINES` | `500` | Ring-buffer size per stream (stdout / stderr) per job |
| `BG_MAX_JOBS` | `32` | Hard cap on concurrent background jobs |
| `BG_RETENTION_SEC` | `600` | How long finished jobs stay queryable via `tail_background` before reaping (10 min) |

### Workspace paths

| Mount | Path | Source | Purpose |
|-------|------|--------|---------|
| Workspace bind-mount | `/workspace` | Host `${ATLAS_PROJECT_DIR}` (or `${ATLAS_PROJECTS_DIR}` under K3s) | Persistent, user-visible. `run_background`, `/shell`, and project-context file lookups all see this. |
| Execute tmpfs | `WORKSPACE_BASE` (default `/tmp/sandbox`) | Per-request `tempfile.mkdtemp` | Ephemeral, per-`/execute` call. PC-191's universal tmpfs sandboxes language toolchains run here. |

---

## 6. llama-server

C++ inference server (llama.cpp) with CUDA GPU acceleration and grammar-constrained JSON output.

Both Docker Compose and K3s use the same image with the same entrypoint (`inference/entrypoint-v3.1-9b.sh`), so the flag set is identical. The only differences are which env vars feed the entrypoint and what default value each deployment passes for them.

### Entrypoint env vars (read by `entrypoint-v3.1-9b.sh`)

| Env var | Docker default | K3s default | Description |
|---------|----------------|-------------|-------------|
| `MODEL_PATH` | `/models/${ATLAS_MODEL_FILE}` | `/models/${ATLAS_MAIN_MODEL}` | GGUF path inside the container |
| `PORT` | `8080` | `${ATLAS_LLAMA_PORT}` (defaults to `8080`) | Listen port |
| `CONTEXT_LENGTH` | `${ATLAS_CTX_SIZE:-32768}` | `${ATLAS_CONTEXT_LENGTH}` (atlas.conf default `16384`) | Context window in tokens |
| `PARALLEL_SLOTS` | `${ATLAS_PARALLEL_SLOTS:-4}` (compose default `4`) | `${ATLAS_PARALLEL_SLOTS}` (atlas.conf default `1`) | Concurrent request slots. Compose defaults to `4` because the `/demo` split-pane runs V3 (which fans out into 3 parallel PlanSearch candidates) alongside a raw-9B session — 4 total concurrent inferences. DeltaNet KV is ~144 MB per slot, well within budget on a 16 GB card. |
| `KV_CACHE_TYPE_K` | `q8_0` | `q8_0` | KV cache key quantization |
| `KV_CACHE_TYPE_V` | `q4_0` | `q4_0` | KV cache value quantization |
| `SLOT_SAVE_PATH` | `/tmp/slots` | `/tmp/slots` | Slot-save directory used by `/slots/0?action=save` |
| `ATLAS_CONTROL_VECTOR` | `/models/ast_edit_steering.gguf` | same | ASA control-vector path (auto-loaded if present) |
| `ATLAS_CONTROL_VECTOR_SCALE` | `0.5` | same | Scale applied to the control vector |
| `ATLAS_CONTROL_VECTOR_LAYER_RANGE` | (unset → all layers) | same | Optional layer restriction — two space-separated integers, e.g. `"24 30"` |

### Effective llama-server flags

The entrypoint always launches with this flag set (regardless of deployment mode):

| Flag | Value | Description |
|------|-------|-------------|
| `-m` | `$MODEL_PATH` | Model path |
| `-c` | `$CONTEXT_LENGTH` | Context window |
| `-ctk` / `-ctv` | `$KV_CACHE_TYPE_K` / `_V` | KV cache quantization (default `q8_0` / `q4_0`) |
| `--parallel` | `$PARALLEL_SLOTS` | Concurrent request slots |
| `--cont-batching` | — | Continuous batching |
| `-ngl` | `99` | Offload all GPU layers |
| `--host` | `0.0.0.0` | Listen on all interfaces |
| `--port` | `$PORT` | Listen port |
| `--flash-attn` | `on` | Flash attention |
| `--mlock` | — | Lock model in RAM (prevents swapping) |
| `-b` / `-ub` | `4096` / `4096` | Batch / micro-batch size |
| `--slot-save-path` | `$SLOT_SAVE_PATH` | Where llama-server persists slot state |
| `--ctx-checkpoints` | `0` | Disable context checkpoints |
| `--no-cache-prompt` | — | Disable prompt caching (PC-045: prevents cross-session leakage) |
| `--embeddings` | — | Enable self-embedding endpoint (lens C(x)/G(x) needs this) |
| `--jinja` | — | Jinja chat-template support |
| `--control-vector-scaled` | `$ATLAS_CONTROL_VECTOR:$ATLAS_CONTROL_VECTOR_SCALE` | Added only when the control-vector file exists |

> **Note:** The Docker entrypoint and the K3s entrypoint are the same script. The only practical knobs that diverge are `CONTEXT_LENGTH` (Docker defaults `32768` via `ATLAS_CTX_SIZE`; K3s defaults `16384` via `ATLAS_CONTEXT_LENGTH`) and `PARALLEL_SLOTS` (Docker compose defaults `4` via `ATLAS_PARALLEL_SLOTS` to support `/demo` split-pane plus V3 plan-search fanout; K3s defaults `1` via `atlas.conf`).

---

## 7. Python CLI

The standalone Python REPL (`pip install -e . && atlas`) reads these variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_INFERENCE_URL` | `http://localhost:8080` | llama-server endpoint |
| `ATLAS_LENS_URL` | `http://localhost:8099` | Geometric Lens endpoint (used by `doctor`, `repl`, and the TUI launcher) |
| `ATLAS_RAG_URL` | `http://localhost:8099` | Legacy alias for the lens URL; still read by `atlas/cli/client.py`. New code should use `ATLAS_LENS_URL`. |
| `ATLAS_SANDBOX_URL` | `http://localhost:30820` | Sandbox endpoint |
| `ATLAS_V3_URL` | `http://localhost:8070` | V3 pipeline endpoint (used by `atlas doctor` for reachability checks) |
| `ATLAS_MODELS_DIR` | `./models` | Host directory holding GGUF model files (used by `atlas doctor` and `atlas model`). |
| `ATLAS_MODEL_FILE` | `Qwen3.5-9B-Q6_K.gguf` | Expected model filename inside `ATLAS_MODELS_DIR`. |
| `ATLAS_LENS_MODELS` | `./geometric-lens/geometric_lens/models` | Host path that maps to the lens's weight directory. Used by `atlas doctor` so it checks the same directory Docker bind-mounts into the lens container. |
| `ATLAS_MODEL_NAME` | `Qwen3.5-9B-Q6_K` | Model name for API calls |
| `HF_TOKEN` | (unset) | HuggingFace write token used by `atlas lens publish` / `atlas asa publish` for artifact upload. Get one at https://huggingface.co/settings/tokens (scope: write). `HUGGINGFACE_HUB_TOKEN` and `HUGGING_FACE_HUB_TOKEN` are also honored. Full walkthrough: [PUBLISHING.md](PUBLISHING.md). |
| `ATLAS_PUBLISH_BRANCH` | (unset → gh infers from checkout) | Override the `--head` branch passed to `gh pr create` during publish. Useful when you're working on a long-lived feature branch but want the registry PR to target main from a different ref. When unset, `gh` auto-detects the current branch (preferred for most contributors). |

### Generation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_tokens | 8,192 | Max output tokens |
| temperature | 0.6 | Generation temperature |
| top_k | 20 | Top-K sampling |
| top_p | 0.95 | Nucleus sampling |
| stop | `["<\|im_end\|>"]` | Stop sequence |

---

## 8. K3s Configuration (atlas.conf)

For K3s deployment only. Copy `atlas.conf.example` to `atlas.conf` and edit. The install pipeline reads this file, renders `templates/*.yaml.tmpl` via `envsubst`, and applies the resulting manifests in `manifests/*.yaml`.

> **Note:** `atlas.conf` is only used by K3s deployment scripts. Docker Compose uses `.env` instead. The two files configure different deployment targets and should not be mixed (ISSUES.md PC-021).

> **May 2026 cleanup.** `atlas.conf.example` was trimmed from 114 variables to 55. The removed entries were scaffolding for features that were planned, removed, or never wired up (RAG knob injection, Ralph training loop, LoRA model retraining, cache manager daemon, JWT/admin/rate-limit auth scheme, V3 phase-component toggles, log-level/external-URL placeholders, etc.). Every variable below is consumed by at least one of: the install/uninstall scripts, the K3s manifest templates, or the benchmark/v3 ablation runner. If you're upgrading from an older `atlas.conf` that sets removed vars, those settings are now silently ignored — see §8.12 below for the migration list.

### 8.1 Cluster & Network

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_NAMESPACE` | `atlas` | Kubernetes namespace for every ATLAS pod / service / PVC |
| `ATLAS_NODE_IP` | `auto` | Node IP for NodePort URL output. `auto` runs `ip` then `hostname -I` then `hostname -i`. |
| `ATLAS_KUBECONFIG` | `/etc/rancher/k3s/k3s.yaml` | Path to kubeconfig the install scripts use. Leave `auto` to inherit from environment. |
| `ATLAS_PROXY_NODEPORT` | `30080` | atlas-proxy external port (renamed from `ATLAS_LLM_PROXY_NODEPORT` on May 2 2026) |
| `ATLAS_LENS_NODEPORT` | `31144` | geometric-lens external port |
| `ATLAS_LLAMA_NODEPORT` | `32735` | llama-server external port |
| `ATLAS_SANDBOX_NODEPORT` | `30820` | sandbox external port |
| `ATLAS_V3_NODEPORT` | `30070` | v3-service external port (cluster-internal-only by default; set a NodePort if you want to hit `/v3/*` from outside) |
| `ATLAS_LLAMA_PORT` | `8080` | llama-server internal port (matches Dockerfile EXPOSE) |
| `ATLAS_LENS_PORT` | `8099` | geometric-lens internal port |
| `ATLAS_PROXY_PORT` | `8090` | atlas-proxy internal port |
| `ATLAS_V3_PORT` | `8070` | v3-service internal port |
| `ATLAS_SANDBOX_PORT` | `8020` | sandbox internal port |
| `ATLAS_REDIS_PORT` | `6379` | Redis internal port |

### 8.2 Storage paths

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_MODELS_DIR` | `/opt/atlas/models` | GGUF model files. Mounted into llama-server at `/models` (read-only) via `hostPath` in `templates/llama-deployment.yaml.tmpl`. |
| `ATLAS_PROJECTS_DIR` | `/opt/atlas/data/projects` | User project workspace. Bind-mounted at `/workspace` in BOTH atlas-proxy and sandbox pods (`hostPath` with `DirectoryOrCreate`) so the agent sees the same files in both. |
| `ATLAS_DATA_DIR` | `/opt/atlas/data` | Housekeeping path. Printed at install time; `uninstall.sh` does `rm -rf "$ATLAS_DATA_DIR"` when `--remove-data` is set. Not mounted as a volume. |
| `ATLAS_TRAINING_DIR` | `/opt/atlas/data/training` | Housekeeping path. Referenced by `uninstall.sh` cleanup; not mounted by any deployment template. |
| `ATLAS_LORA_DIR` | `/opt/atlas/models/lora` | Housekeeping path. Created by `install.sh` and `download-models.sh`; populated by the training pipeline; not currently mounted into any pod. |

### 8.3 Persistent Volume sizes

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_PVC_REDIS_SIZE` | `5Gi` | Redis persistence PVC |
| `ATLAS_PVC_PROJECTS_SIZE` | `20Gi` | `lens-projects` PVC used by the geometric-lens pod for its project index storage |

### 8.4 Model & Inference

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_MAIN_MODEL` | `Qwen3.5-9B-Q6_K.gguf` | Main GGUF filename. Becomes `MODEL_PATH=/models/<name>` inside the container. |
| `ATLAS_DRAFT_MODEL` | `Qwen3-0.6B-Q8_0.gguf` | Draft model filename for speculative decoding. Gated by `ATLAS_ENABLE_SPECULATIVE`; note that Qwen3.5-9B speculative is disabled at the entrypoint level today regardless of this setting. |
| `ATLAS_CONTEXT_LENGTH` | `16384` | Per-slot context tokens. V3's `--parallel 1` budget is sized around 16K; raise if you have GPU headroom and want longer turns. |
| `ATLAS_PARALLEL_SLOTS` | `1` | Concurrent KV slots. V3 self-embeddings push VRAM tight on 16 GB cards, so `1` is the safe default. |

### 8.5 Resource limits (Kubernetes pod spec)

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_LLAMA_CPU_REQUEST` | `2` | CPU request for llama-server |
| `ATLAS_LLAMA_CPU_LIMIT` | `4` | CPU limit for llama-server |
| `ATLAS_LLAMA_MEMORY_REQUEST` | `8Gi` | Memory request for llama-server |
| `ATLAS_LLAMA_MEMORY_LIMIT` | `16Gi` | Memory limit for llama-server |
| `ATLAS_SERVICE_CPU_REQUEST` | `0.5` | CPU request for non-llama services (proxy, lens, v3-service, sandbox) |
| `ATLAS_SERVICE_CPU_LIMIT` | `2` | CPU limit for non-llama services |
| `ATLAS_SERVICE_MEMORY_REQUEST` | `512Mi` | Memory request for non-llama services |
| `ATLAS_SERVICE_MEMORY_LIMIT` | `2Gi` | Memory limit for non-llama services |

> GPU is requested as a count (`nvidia.com/gpu: 1`), not a memory budget — there is no `ATLAS_LLAMA_GPU_MEMORY` knob.

### 8.6 Auth bootstrap

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_JWT_SECRET` | `auto` | When `auto`, `scripts/lib/config.sh` generates a random 32-byte hex secret on first install and caches it in `.jwt_secret`. No service currently consumes the secret — this is forward-compatible scaffolding for the eventual auth layer. |

### 8.7 Feature flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_ENABLE_SPECULATIVE` | `true` | Gates draft-model download in `scripts/download-models.sh`. Inference-time behavior is currently fixed by the entrypoint (llama.cpp can't speculate hybrid DeltaNet+Attention models yet — see comment in `inference/entrypoint-v3.1-9b.sh`), so this flag only affects whether the draft GGUF is downloaded. |
| `ATLAS_ENABLE_TRAINING` | `true` | When `true`, `install.sh` applies `templates/training-cronjob.yaml.tmpl` (the nightly cost-field retrain). When `false`, the cronjob is skipped. |

### 8.8 Timeouts (seconds)

| Variable | Default | Used by |
|----------|---------|---------|
| `ATLAS_LLM_TIMEOUT` | `120` | `scripts/verify-install.sh` for the smoke-test `curl` against llama-server |
| `ATLAS_HEALTH_CHECK_TIMEOUT` | `10` | `scripts/verify-install.sh` `--max-time` for `curl` against each `/health` endpoint during post-install verification. (The healthchecks defined inside the K3s templates use hardcoded timeouts, not this var.) |

### 8.9 Training cronjob (only when `ATLAS_ENABLE_TRAINING=true`)

Consumed by `templates/training-cronjob.yaml.tmpl`. The job hits `geometric-lens /internal/lens/retrain`, which retrains the C(x) cost-field MLP — not the model itself.

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_TRAINING_SCHEDULE` | `"0 2 * * *"` | Cron expression for the nightly retrain |
| `ATLAS_TRAINING_MIN_RATING` | `4` | Minimum user-feedback rating to include in the retrain set |
| `ATLAS_TRAINING_VALIDATION_THRESHOLD` | `66` | Percentage of held-out validation that must pass for the new C(x) weights to be promoted |

### 8.10 V3 ablation knobs (benchmark-only)

Consumed by `benchmark/v3_runner.py:_load_v3_config` for ablation studies. The production `v3-service` reads its own constants from `benchmark/v3/*.py` config dataclasses and does NOT pick these up at runtime.

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_V3_BUDGET_FORCING_DEFAULT_TIER` | `"standard"` | Default Budget Forcing tier when difficulty estimation is unavailable |
| `ATLAS_V3_BUDGET_FORCING_MAX_WAIT_INJECTIONS` | `3` | Max "Wait, let me reconsider…" injections per generation |
| `ATLAS_V3_PLAN_SEARCH_NUM_PLANS` | `3` | Plans generated per problem (overrides `PlanSearchConfig.num_plans`) |
| `ATLAS_V3_BLEND_ASC_DEFAULT_K` | `3` | Default K candidates when adaptive routing is unavailable |
| `ATLAS_V3_REASC_CONFIDENCE_THRESHOLD` | `-0.5` | Logprob threshold for ReASC early-stop |
| `ATLAS_V3_REASC_ENERGY_THRESHOLD` | `0.10` | C(x) threshold for ReASC early-stop |
| `ATLAS_V3_S_STAR_ENERGY_DELTA` | `1.0` | S* tiebreak fires when candidate energies are within this delta |
| `ATLAS_V3_EWC_LAMBDA` | `1000.0` | EWC regularization strength (Phase 4A-EWC) |
| `ATLAS_V3_REPLAY_BUFFER_MAX_SIZE` | `5000` | Replay buffer capacity (Phase 4A-CL) |
| `ATLAS_V3_REPLAY_BUFFER_REPLAY_RATIO` | `0.30` | Fraction of new training mixed with replayed examples |
| `ATLAS_V3_LENS_FEEDBACK_ENABLED` | `false` | Toggle online lens recalibration during benchmark runs |
| `ATLAS_V3_LENS_FEEDBACK_RETRAIN_INTERVAL` | `50` | Retrain every N benchmark problems |

### 8.11 Advanced

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_REGISTRY` | `localhost` | Container registry prefix for locally-built images. Used by `scripts/build-containers.sh` to tag (e.g. `localhost/atlas-proxy:latest`). The K3s manifests themselves pull from `ghcr.io/${ATLAS_GHCR_OWNER}/...`, so this matters only when you're building images locally and side-loading them into k3s. |
| `ATLAS_IMAGE_TAG` | `latest` | Image tag for both the local-build path and the GHCR pull path |

The install scripts also honor two runtime-only env vars (not in `atlas.conf` itself):

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_CONFIG_FILE` | (auto) | Path override for `atlas.conf` itself. `scripts/lib/config.sh` looks at this before falling back to `$K8S_DIR/atlas.conf`. |
| `ATLAS_AUTO_CONFIRM` | `false` | Set to `true` in the environment to skip the interactive install prompts in `scripts/install.sh` |

### 8.12 Migrating from a pre-May-2026 atlas.conf

If you're carrying forward an `atlas.conf` from before the trim, these variables are now silently ignored. Leaving them in place won't break config-load (Bash just sets them as shell variables that nothing reads), but they have no effect and you can delete them. Group them by reason:

| Group | Variables | Why removed |
|-------|-----------|-------------|
| Cache manager daemon | `ATLAS_CACHE_MANAGER_ENABLED`, `_SOFT_THRESHOLD_MB`, `_HARD_THRESHOLD_MB`, `_CHECK_INTERVAL_SEC`, `_ERASE_COOLDOWN_SEC`, `_RESTART_COOLDOWN_SEC`, `_WARMUP_ENABLED` | Scripted daemon was never built (`scripts/llama-cache-manager.py` doesn't exist) |
| Ralph training loop | `ATLAS_RALPH_MAX_RETRIES`, `_BASE_TEMP`, `_TEMP_INCREMENT`, `_MAX_TEMP` | Training code paths don't read these |
| RAG knobs | `ATLAS_RAG_CONTEXT_BUDGET`, `_TOP_K`, `_MAX_FILES` | The lens has its own `RetrievalConfig` (YAML at `CONFIG_PATH`); these atlas.conf names aren't injected |
| LoRA params | `ATLAS_LORA_RANK`, `ATLAS_LORA_ALPHA` | Nightly retrain hits the C(x) cost field, not the model — no LoRA training in this path |
| Auth scaffolding | `ATLAS_ADMIN_EMAIL`, `ATLAS_DEFAULT_RATE_LIMIT`, `ATLAS_JWT_EXPIRY_HOURS`, `ATLAS_KEY_HASH_ALGORITHM` | No service consumes them; only `ATLAS_JWT_SECRET` is touched (for future use) |
| Feature flags with no consumer | `ATLAS_ENABLE_RAG`, `ATLAS_ENABLE_PROVENANCE`, `ATLAS_ENABLE_DASHBOARD` | Listed for future use; no code reads them. `ATLAS_ENABLE_DASHBOARD` specifically refers to the V1 atlas-dashboard service that was removed. |
| Logging placeholders | `ATLAS_LOG_LEVEL`, `ATLAS_LOG_REQUESTS` | Services use their own logger defaults |
| External-URL placeholders | `ATLAS_EXTERNAL_URL`, `ATLAS_API_EXTERNAL_URL` | Placeholders for ingress / reverse-proxy URLs; not consumed |
| Timeouts with no consumer | `ATLAS_SANDBOX_TIMEOUT`, `ATLAS_TASK_TIMEOUT` | Sandbox uses its own `MAX_EXECUTION_TIME` env (see §5); `_TASK_TIMEOUT` was never wired |
| Inference flags overridden by entrypoint | `ATLAS_GPU_LAYERS`, `ATLAS_FLASH_ATTENTION`, `ATLAS_LLAMA_GPU_MEMORY` | Entrypoint hardcodes `-ngl 99` and `--flash-attn on`; GPU is requested by count not memory budget |
| V3 phase toggles (constructor-driven) | `ATLAS_V3_PHASE1_ENABLED`/`_PHASE2_ENABLED`/`_PHASE3_ENABLED`, plus all 16 per-component `_ENABLED` flags (`_BUDGET_FORCING_ENABLED`, `_PLAN_SEARCH_ENABLED`, `_DIV_SAMPLING_ENABLED`, `_BLEND_ASC_ENABLED`, `_REASC_ENABLED`, `_S_STAR_ENABLED`, `_FAILURE_ANALYSIS_ENABLED`, `_CONSTRAINT_REFINEMENT_ENABLED`, `_PR_COT_ENABLED`, `_DERIVATION_CHAINS_ENABLED`, `_REFINEMENT_LOOP_ENABLED`, `_METACOGNITIVE_ENABLED`, `_ACE_ENABLED`, `_SELF_TEST_ENABLED`, `_REPLAY_BUFFER_ENABLED`, `_EWC_ENABLED`) | Phase + component enables come from `V3Runner(enable_phase1=…, …)` constructor args, not env vars |
| V3 numeric vars never wired | `ATLAS_V3_PR_COT_MAX_ROUNDS`, `_REFINEMENT_LOOP_MAX_ITERATIONS`, `_REFINEMENT_LOOP_TIME_BUDGET_SEC`, `_SELF_TEST_NUM_CASES`, `_SELF_TEST_MAJORITY_THRESHOLD`, `_LENS_FEEDBACK_DOMAIN`, `_SELECTION_STRATEGY`, `_ENABLE_FEEDBACK` | Listed in the example file but `_load_v3_config` doesn't read them — the in-code dataclass defaults are used instead |
