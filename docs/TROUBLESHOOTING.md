# ATLAS Troubleshooting Guide

Common issues and solutions for ATLAS V3.0.1, organized by service.

---

## Quick Diagnostics

Run these first to identify where the problem is:

```bash
# Docker Compose — check all services at once
docker compose ps

# Individual health checks
curl -s http://localhost:8000/health | python3 -m json.tool   # vLLM
curl -s http://localhost:31144/health | python3 -m json.tool   # geometric-lens
curl -s http://localhost:8070/health | python3 -m json.tool   # v3-service
curl -s http://localhost:30820/health | python3 -m json.tool  # sandbox
curl -s http://localhost:8090/health | python3 -m json.tool   # atlas-proxy (shows all service statuses)

# GPU status
nvidia-smi

# Docker Compose logs (last 50 lines per service)
docker compose logs --tail 50
```

The atlas-proxy health endpoint reports the status of all upstream services:
```json
{
  "status": "ok",
  "inference": true,
  "lens": true,
  "sandbox": true,
  "port": "8090",
  "stats": { "requests": 0, "repairs": 0, "sandbox_passes": 0, "sandbox_fails": 0 }
}
```

If any field is `false`, that service is the problem.

---

## Docker / Podman Issues

### GPU Not Detected in Container

**Symptom:** vLLM container starts but model loads on CPU (very slow, ~2 tok/s). `nvidia-smi` shows the GPU from the host but the container can't see it.

**Fix:** Install NVIDIA Container Toolkit:

```bash
# RHEL/Fedora
sudo dnf install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=podman
sudo systemctl restart podman

# Ubuntu/Debian
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is visible inside containers:
```bash
# Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Podman
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.0-base nvidia-smi
```

### First Build Fails (CUDA Not Found)

**Symptom:** `docker compose build` fails with CUDA-related errors during vLLM compilation.

**Fix:** The vLLM Dockerfile builds vLLM inside a `nvidia/cuda:12.8.0-devel` base image, so CUDA headers are available during build without host GPU access. Common causes of build failure:
1. Insufficient disk space (~5GB needed for build artifacts)
2. Network issues downloading the CUDA base image or cloning vLLM
3. Podman rootless builds may fail with permission issues — try `podman-compose build` with `--podman-build-args="--format docker"`

### SELinux Blocking Container Access (Fedora/RHEL)

**Symptom:** Containers can't read mounted volumes, permission denied on model files.

**Fix:**
```bash
# Allow container access to model directory
chcon -Rt svirt_sandbox_file_t ~/models/

# Or add :Z flag to volume mounts (Docker Compose handles this)
```

### Sandbox Unreachable

**Symptom:** Proxy health shows `"sandbox": false`. V3 build verification fails.

**Fix:** Ensure all services are on the same Docker network. Docker Compose creates the `atlas` network automatically. If running containers manually:
```bash
docker network create atlas
# Start all containers with --network atlas
```

### Port Conflicts

**Symptom:** `docker compose up` fails with "address already in use" on a port.

**Fix:** Check what's using the port and either stop it or change ATLAS ports in `.env`:
```bash
# Find what's using port 8080
lsof -i :8080

# Change port in .env
ATLAS_GEN_PORT=8081    # Different port for vLLM
```

All ports are configurable via `.env`. See [CONFIGURATION.md](CONFIGURATION.md).

---

## vLLM Issues

### Model Loading on CPU Instead of GPU

**Symptom:** Generation is glacial; `nvidia-smi` doesn't show vLLM using the GPU.

**Fix:** vLLM auto-detects the GPU via PyTorch CUDA. If it's running on CPU something deeper is wrong — usually the NVIDIA container runtime is missing. Verify:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-runtime-ubuntu22.04 nvidia-smi
```
If that fails, install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). For bare metal, ensure `pip install vllm` was run with CUDA-enabled PyTorch (the wheels.vllm.ai/nightly index handles this automatically).

### Model Directory Not Found

**Symptom:** vLLM exits immediately with `OSError: ... is not a valid model identifier` or similar.

**Fix:** vLLM loads from a directory of `.safetensors` shards (NOT a single GGUF file). Check the AWQ directory exists:
```bash
# Docker Compose — model must be in ATLAS_MODELS_DIR (default: ./models/)
ls -la models/Qwen3.5-9B-AWQ/  # should contain config.json + *.safetensors

# Bare metal — check ATLAS_MODEL_PATH
ls -la "$ATLAS_MODEL_PATH"
```

Pull the weights with `make model` or:
```bash
huggingface-cli download QuantTrio/Qwen3.5-9B-AWQ --local-dir models/Qwen3.5-9B-AWQ
```

For gated repos, set `HF_TOKEN` first.

### Out of VRAM

**Symptom:** vLLM crashes or gets OOMKilled. `nvidia-smi` shows VRAM near 100%, or vLLM logs `torch.cuda.OutOfMemoryError`.

**Fix:** AWQ Q4 of Qwen3.5-9B needs ~5 GB for weights, plus KV cache. The two-instance setup (gen 0.55 + embed 0.20) targets ~80 GB cards. On smaller GPUs:

```bash
# 16 GB consumer card — gen-only, no Lens embeddings
GEOMETRIC_LENS_ENABLED=false \
ATLAS_GEN_MAX_NUM_SEQS=4 \
ATLAS_GEN_CTX_SIZE=16384 \
ATLAS_GEN_GPU_MEM=0.85 \
docker compose up vllm-gen geometric-lens v3-service sandbox atlas-proxy
```

Disable the embed instance entirely with `docker compose up --scale vllm-embed=0` if you don't need Lens scoring during testing.

### Thinking Tokens in Output

**Symptom:** Responses contain `<think>...</think>` blocks or `reasoning_content` is filled but `content` is empty.

**Fix:** vLLM with `--reasoning-parser qwen3` separates thinking from the answer. Toggle thinking via `chat_template_kwargs`:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role":"user","content":"Say hi"}],
    "max_tokens": 50,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Qwen3.5 dropped the soft `/nothink` and `/think` commands — passing them in the prompt does nothing. Use the `chat_template_kwargs` body field instead.

### G(x) Lens Scoring Disabled

**Symptom:** Lens calls return `gx_available: false`. Phase 2 candidate selection only uses C(x).

**Fix:** xgboost is missing from the Lens container. Check the Lens log:
```bash
docker compose logs geometric-lens | grep -i xgboost
# expect: "G(x) XGBoost loaded (AUC=0.8840, PCA 4096→128)"
# bug:    "xgboost package not installed"
```

If the bug message appears, `xgboost` and `scikit-learn` are missing from `geometric-lens/requirements.txt`. The Dockerfile installs from that file. Rebuild:
```bash
docker compose build --no-cache geometric-lens
docker compose up -d geometric-lens
```

### vLLM 0.18.0+ Engine Crash on Qwen3.5

**Symptom:** vLLM logs `EngineCore encountered an issue` during startup,
the container restarts repeatedly, model loading and `torch.compile`
appear to succeed before the engine dies. Affects Qwen3.5 specifically;
other Qwen variants (Qwen3-embedding, Qwen3-reranker) keep working.

**Fix:** Pin `vllm==0.17.1` until upstream resolves the regression. The
ATLAS Dockerfile already pins this, but if you're installing vLLM
separately:
```bash
pip install -U "vllm==0.17.1"
```

See [vllm-project/vllm#37749](https://github.com/vllm-project/vllm/issues/37749).

### KV Cache Memory Overestimation (Hybrid Mamba)

**Symptom:** vLLM reports much higher KV cache utilization than `nvidia-smi`
shows the GPU actually using. Concurrency caps lower than expected for the
configured `--gpu-memory-utilization`.

**Fix:** This is a known vLLM-side overestimation for hybrid Mamba/attention
models like Qwen3.5 — the profiler treats Mamba's constant-size state as if
it scaled like attention's KV cache (~7x overestimation). Workaround: bump
`--gpu-memory-utilization` higher than you'd otherwise need (the real KV
footprint is much smaller than vLLM thinks).

See [vllm-project/vllm#37121](https://github.com/vllm-project/vllm/issues/37121).

### vLLM Crashes on Startup with `block_size > max_num_batched_tokens`

**Symptom:** vLLM exits during config validation with an error like:
```
Validation Error: block_size (2096) > max_num_batched_tokens (2048)
```

**Fix:** Qwen3.5's Gated DeltaNet layers force vLLM to align the block_size
to 2096 tokens for prefix caching. The default `max_num_batched_tokens`
(2048) is one token too small. Both ATLAS docker-compose and the H200
entrypoint already pass `--max-num-batched-tokens 8192` (gen) / `4096`
(embed) to dodge this. If you're running `vllm serve` directly, add:

```bash
vllm serve ... --max-num-batched-tokens 8192
```

This is a known [vLLM issue](https://github.com/vllm-project/vllm/issues/36697)
with hybrid Mamba models.

### Embed Instance Returns 0-dim Vectors

**Symptom:** `/v1/embeddings` returns `{"data": [{"embedding": []}]}` or wrong dimensionality.

**Fix:** The embed instance must use `--runner pooling --convert embed` (the current API). The deprecated `--task embed` may produce different shapes on newer vLLM versions:
```bash
docker compose exec vllm-embed bash -c 'ps aux | grep vllm | grep -- --runner'
# expect: "--runner pooling --convert embed"
```

If you see `--task embed`, your docker-compose is out of date. Pull the latest. Stage 19 of the vLLM cutover migrated to the new API.

---

## Proxy Issues

### Agent Loop Not Activating

**Symptom:** Requests go directly to vLLM. No tool calls, no streaming status icons, no V3 pipeline.

**Fix:** Set `ATLAS_AGENT_LOOP=1`. The `atlas` launcher does this automatically. If running the proxy manually:
```bash
ATLAS_AGENT_LOOP=1 atlas-proxy-v2
```

In Docker Compose, this is set in `docker-compose.yml` and doesn't need manual configuration.

### V3 Pipeline Not Firing on Feature Files

**Symptom:** All `write_file` calls are T1 (direct write). No V3 pipeline stages in output.

V3 only fires when **all three conditions** are met:
1. File has **50+ lines** of content
2. File has **3+ logic indicators** (function defs, control flow, API patterns)
3. V3 service is reachable at `ATLAS_V3_URL`

**Diagnose:**
```bash
# Check V3 service health
curl -s http://localhost:8070/health

# Check proxy logs for tier classification
docker compose logs atlas-proxy | grep "write_file"
# Look for: T1 (direct) vs T2 (V3 pipeline)
```

If V3 is unreachable, the proxy falls back to direct write silently.

### Truncation Errors (write_file Fails Repeatedly)

**Symptom:** Repeated errors like "Your output was truncated — the content is too long for a single tool call."

**Cause:** The model is trying to write too much content in one call. The proxy detects truncated JSON and rejects the tool call.

**What happens automatically:**
- For existing files > 100 lines: proxy rejects `write_file` and tells the model to use `edit_file` instead
- After 3 consecutive failures: error loop breaker stops the agent and returns a summary

**What you can do:** Rephrase your request to ask for targeted changes rather than full file rewrites. For example, "Add input validation to the login function" instead of "Rewrite auth.py".

### File Not Read Before Editing

**Symptom:** `edit_file` fails with "file not read yet — use read_file first before editing."

**Cause:** The proxy tracks which files the agent has read. If the model tries to edit a file it hasn't read in this session, the edit is rejected as a staleness protection.

**Fix:** This is normal behavior — the model should read the file first. If it keeps failing, the model may be confused about which files it has seen. Try `/clear` in Aider and rephrase.

### File Modified Externally

**Symptom:** `edit_file` fails with "file modified since last read — read it again before editing."

**Cause:** The file was changed on disk (by you or another process) after the model read it. The proxy compares modification timestamps.

**Fix:** The model needs to re-read the file. This usually resolves automatically on the next turn.

### Exploration Budget Warning

**Symptom:** Output shows "You have full project context in the system prompt. Do not read more files." or reads are being skipped.

**Cause:** The model has made 4+ consecutive read-only calls (read_file, search_files, list_directory) without writing anything. After 4 reads, the proxy warns. After 5+, it skips reads entirely and tells the model to write.

**Fix:** This is protective behavior. If the model is genuinely stuck exploring, try being more specific about what you want changed.

---

## Geometric Lens Issues

### Lens Not Loaded / Unavailable

**Symptom:** Proxy health shows `"lens": false`. Or startup shows "Lens unavailable — verification disabled."

**Impact:** ATLAS still works but without C(x)/G(x) scoring. V3 candidate selection falls back to sandbox-only verification.

**Fix:** Check Lens health and logs:
```bash
curl -s http://localhost:31144/health
docker compose logs geometric-lens
```

Common causes:
- Lens can't connect to vLLM (check `LLAMA_URL` env var)
- Model weight files missing (service degrades gracefully — this is expected if you haven't trained custom models)

### All Scores Near 0.5

**Symptom:** Every candidate gets `cx_energy: 0.0` and `gx_score: 0.5` regardless of code quality.

**Cause:** Model weights are not loaded. The service returns neutral defaults when models are absent.

**Verify:**
```bash
curl -s http://localhost:31144/internal/lens/gx-score \
  -H "Content-Type: application/json" \
  -d '{"text": "print(1)"}' | python3 -m json.tool
```

If `enabled: false` or `cx_energy: 0.0`, the models aren't loaded. This is expected for a fresh install — model weights are not included in the repository and must be trained or downloaded from [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS).

### Embedding Extraction Fails

**Symptom:** Lens logs show errors like "embedding extraction failed" or timeouts.

**Cause:** Lens calls vLLM's `/v1/embeddings` endpoint. If vLLM is overloaded or the endpoint isn't enabled, this fails.

**Fix:**
```bash
# Test embedding endpoint directly
curl -s http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test"}' | python3 -m json.tool
```

The `/v1/embeddings` endpoint is available in vLLM without special flags for self-embeddings from generation models. In K3s, the `--embeddings` flag is set explicitly in the entrypoint for full embedding support.

---

## Sandbox Issues

### Sandbox Unreachable

**Symptom:** Code is never tested. Proxy health shows `"sandbox": false`.

**Fix:** Check sandbox health:
```bash
# Docker Compose (host port 30820 maps to container port 8020)
curl -s http://localhost:30820/health

# Bare metal (direct port 8020)
curl -s http://localhost:8020/health
```

If the sandbox container is running but unhealthy, check logs:
```bash
docker compose logs sandbox
```

### Code Execution Timeout

**Symptom:** Sandbox returns `"error_type": "Timeout"`. Code takes too long to execute.

**Default timeout:** 30 seconds per request, max 60 seconds (configurable via `MAX_EXECUTION_TIME` env var).

**Fix:** If your code legitimately needs more time, set a higher timeout in the request. If the code has an infinite loop, this is expected behavior.

### Language Not Supported

**Symptom:** Sandbox returns an error for a specific language.

**Supported languages:** Python, JavaScript, TypeScript, Go, Rust, C, C++, Bash.

Check available runtimes:
```bash
curl -s http://localhost:30820/languages | python3 -m json.tool
```

---

## Aider Issues

### `atlas` Shows REPL Instead of Aider (No File Read/Write)

**Symptom:** Running `atlas` shows the built-in REPL with a `Model`, `Speed`, `Lens`, `Sandbox` status block and a `◆` prompt. Typing requests works but no files are created or modified. `--message` flag is ignored.

**Cause:** The `atlas` command auto-detects the proxy and Aider. If either is missing, it falls back to the built-in REPL which supports `/solve` and `/bench` but not file operations.

**Fix:**
1. Ensure the proxy is running: `curl -s http://localhost:8090/health`
2. Ensure Aider is installed: `pip install aider-chat`
3. Ensure services are up: `docker compose ps` (all should show "healthy")

If the proxy is healthy and Aider is installed, `atlas` will automatically launch Aider with the full agent loop (tool calls, file read/write, V3 pipeline).

If Go 1.24+ is installed, `atlas` can also build and launch the proxy automatically — you don't need to start it manually.

### Proxy Lists Wrong Directory or `/tmp`

**Symptom:** The model lists files from `/tmp` or the ATLAS repo instead of your project. `write_file` creates files in the wrong location.

**Cause:** The Docker Compose proxy runs inside a container and can only see the directory mounted at startup. If you're working in a different directory, the proxy can't see it.

**Fix (recommended):** Install Go 1.24+ ([https://go.dev/dl/](https://go.dev/dl/)). The `atlas` CLI will automatically build and launch the proxy locally in your current directory with full file access. No Docker mount needed.

**Fix (without Go):** Set `ATLAS_PROJECT_DIR` in your `.env` to your project path, then restart the proxy:
```bash
# In .env:
ATLAS_PROJECT_DIR=/path/to/your/project

# Restart proxy to pick up new mount:
docker compose up -d atlas-proxy
```

You must update this and restart each time you switch project directories. This is a limitation of running the proxy inside Docker.

### `.env.example` Missing After Clone

**Symptom:** `cp .env.example .env` fails with "No such file or directory".

**Fix:** This was fixed in V3.0.1. If you cloned before the fix, pull the latest:
```bash
git pull
cp .env.example .env
```

### Aider Disconnects on Long Tasks

**Symptom:** Aider times out or disconnects before the agent loop completes, especially during V3 pipeline phases.

**Fix:** Aider's HTTP request timeout needs to be long enough for V3 pipeline execution (which can take minutes). The `.aider.model.settings.yml` in the repo configures streaming mode which keeps the connection alive. If you're still seeing timeouts:

1. Ensure you're using the repo's config files (`.aider.model.settings.yml` and `.aider.model.metadata.json`)
2. Check that `streaming: true` is set in the settings file

### Empty Response

**Symptom:** Aider shows the completion summary but no file content was produced.

**Cause:** The model emitted a `done` signal without making any file changes. This can happen with:
- Very short conversational prompts ("hi", "thanks")
- Ambiguous requests where the model doesn't know what file to create

**Fix:** Be more specific. Tell the model exactly what file to create or edit.

### Wrong Working Directory

**Symptom:** Files created in the wrong location. `list_directory` shows unexpected contents.

**Cause:** The proxy detects the project directory by finding the most recently modified `.aider.chat.history.md` file. If you have multiple Aider sessions open, the newest one wins.

**Fix:** Close other Aider sessions, or `cd` into the correct project directory before running `atlas`.

### "Model not found" Error

**Symptom:** Aider fails to start with a model-related error.

**Fix:** Ensure both Aider config files exist in the ATLAS root:
```bash
ls -la .aider.model.settings.yml .aider.model.metadata.json
```

These are included in the repository. If missing, re-clone or restore from backup. They tell Aider to use the `openai/atlas` model pointing at the proxy.

---

## Performance

### Slow Generation (~2 tok/s)

The model is running on CPU instead of GPU. Check:
1. `nvidia-smi` — is vLLM listed as a GPU process?
2. `--n-gpu-layers 99` — are all layers offloaded?
3. NVIDIA Container Toolkit — is the container runtime configured for GPU access?

**Expected performance:** ~51 tok/s on RTX 5060 Ti 16GB with grammar enforcement.

### V3 Pipeline Takes Several Minutes

This is normal for T2 files. The V3 pipeline makes multiple LLM calls:
- **Probe only (best case):** ~10-15 seconds (1 generation + 1 score + 1 test)
- **Phase 1 generation:** ~1-2 minutes (PlanSearch + DivSampling + scoring)
- **Phase 3 repair:** ~2-5 minutes (PR-CoT + Refinement + Derivation, if needed)

To get faster (but lower quality) results:
- Keep files under 50 lines (stays T1, no V3)
- Reduce logic complexity (fewer functions, control flow)
- V3 only fires when truly needed — simple files are written instantly

### High RAM Usage

**Symptom:** System becomes sluggish or services get OOMKilled.

**Expected RAM usage:**
- vLLM: ~8 GB (model in VRAM, minimal RAM)
- geometric-lens: ~200 MB (PyTorch runtime + models)
- v3-service: ~150 MB (PyTorch runtime)
- sandbox: ~100 MB (base, spikes during compilation)
- atlas-proxy: ~30 MB (Go binary)

**Total:** ~500 MB RAM + 8.2 GB VRAM. If you have less than 14 GB system RAM, other services may compete for memory.

---

## Getting Help

If your issue isn't listed here:
1. Check service logs: `docker compose logs <service-name>`
2. Check the proxy health endpoint: `curl http://localhost:8090/health`
3. See [CONFIGURATION.md](CONFIGURATION.md) for all environment variables
4. Open an issue on [GitHub](https://github.com/itigges22/ATLAS/issues)
