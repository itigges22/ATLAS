# ATLAS Troubleshooting Guide

Common issues and solutions for ATLAS V3.0.1, organized by service.

---

## Quick Diagnostics

Run these first to identify where the problem is:

```bash
# Docker Compose — check all services at once
docker compose ps

# Individual health checks
curl -s http://localhost:8080/health | python3 -m json.tool   # llama-server
curl -s http://localhost:8099/health | python3 -m json.tool   # geometric-lens
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

**Symptom:** llama-server container starts but model loads on CPU (very slow, ~2 tok/s). `nvidia-smi` shows the GPU from the host but the container can't see it.

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

**Symptom:** `docker compose build` fails with CUDA-related errors during llama-server compilation.

**Fix:** The llama-server Dockerfile builds llama.cpp inside a `nvidia/cuda:12.8.0-devel` base image, so CUDA headers are available during build without host GPU access. Common causes of build failure:
1. Insufficient disk space (~5GB needed for build artifacts)
2. Network issues downloading the CUDA base image or cloning llama.cpp
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
ATLAS_LLAMA_PORT=8081    # Different port for llama-server
```

All ports are configurable via `.env`. See [CONFIGURATION.md](CONFIGURATION.md).

---

## llama-server Issues

### Model Loading on CPU Instead of GPU

**Symptom:** Generation at ~2 tok/s instead of ~50 tok/s. `nvidia-smi` doesn't show llama-server using the GPU.

**Fix:** Ensure `--n-gpu-layers 99` is set (offloads all layers to GPU). In Docker Compose this is the default. For bare metal, check the command:
```bash
ps aux | grep llama-server | grep 'n-gpu-layers'
```

If using Docker, ensure the NVIDIA container runtime is configured (see GPU section above).

### Model File Not Found

**Symptom:** llama-server exits immediately with "failed to load model" or similar.

**Fix:** Check the model path:
```bash
# Docker Compose — model must be in ATLAS_MODELS_DIR (default: ./models/)
ls -la models/Qwen3.5-9B-Q6_K.gguf

# Bare metal — check ATLAS_MODEL_PATH
ls -la ~/models/Qwen3.5-9B-Q6_K.gguf
```

The filename must match `ATLAS_MODEL_FILE` in `.env` (default: `Qwen3.5-9B-Q6_K.gguf`).

### Out of VRAM

**Symptom:** llama-server crashes or gets OOMKilled shortly after starting. `nvidia-smi` shows VRAM near 100%.

**Fix:** The 9B Q6_K model needs ~8.2 GB VRAM (model + KV cache). Ensure:
1. No other GPU processes are running (`nvidia-smi` — check for other CUDA processes)
2. You have 16GB+ VRAM
3. Context size isn't set too high (default 32K is fine, don't increase without checking VRAM)

```bash
# Kill other GPU processes if needed
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill {}
```

### Grammar Not Enforced (Model Outputs Thinking Blocks)

**Symptom:** Model outputs `<think>` tags or raw text instead of JSON tool calls.

**Fix:** The proxy sets `response_format: {"type": "json_object"}` automatically when `ATLAS_AGENT_LOOP=1`. If using llama-server directly, include it in your request:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q6_K",
    "messages": [{"role":"user","content":"Say hi"}],
    "max_tokens": 50,
    "response_format": {"type": "json_object"}
  }'
```

If this returns raw text instead of JSON, your llama.cpp build doesn't support `response_format`. Rebuild from the latest source.

### Context Window Too Small

**Symptom:** Tool call arguments get truncated. `write_file` fails with "unexpected end of JSON" or proxy logs show "truncation detected".

**Fix:** Context size should be 32768 (default in Docker Compose). Check:
```bash
# Docker Compose
grep CTX_SIZE .env

# Bare metal
ps aux | grep llama-server | grep ctx-size
```

---

## Proxy Issues

### Agent Loop Not Activating

**Symptom:** Requests go directly to llama-server. No tool calls, no streaming status icons, no V3 pipeline.

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
curl -s http://localhost:8099/health
docker compose logs geometric-lens
```

Common causes:
- Lens can't connect to llama-server (check `LLAMA_URL` env var)
- Model weight files missing (service degrades gracefully — this is expected if you haven't trained custom models)

### All Scores Near 0.5

**Symptom:** Every candidate gets `cx_energy: 0.0` and `gx_score: 0.5` regardless of code quality.

**Cause:** Model weights are not loaded. The service returns neutral defaults when models are absent.

**Verify:**
```bash
curl -s http://localhost:8099/internal/lens/gx-score \
  -H "Content-Type: application/json" \
  -d '{"text": "print(1)"}' | python3 -m json.tool
```

If `enabled: false` or `cx_energy: 0.0`, the models aren't loaded. This is expected for a fresh install — model weights are not included in the repository and must be trained or downloaded from [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS).

### Embedding Extraction Fails

**Symptom:** Lens logs show errors like "embedding extraction failed" or timeouts.

**Cause:** Lens calls llama-server's `/v1/embeddings` endpoint. If llama-server is overloaded or the endpoint isn't enabled, this fails.

**Fix:**
```bash
# Test embedding endpoint directly
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test"}' | python3 -m json.tool
```

The `/v1/embeddings` endpoint is available in llama.cpp without special flags for self-embeddings from generation models. In K3s, the `--embeddings` flag is set explicitly in the entrypoint for full embedding support.

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
1. `nvidia-smi` — is llama-server listed as a GPU process?
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
- llama-server: ~8 GB (model in VRAM, minimal RAM)
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
