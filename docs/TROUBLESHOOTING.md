# ATLAS Troubleshooting Guide

Common issues and solutions for ATLAS V3.0.1.

---

## Docker / Podman Issues

### GPU Not Detected in Container

**Symptom**: llama-server container starts but model loads on CPU (very slow, ~2 tok/s).

**Fix**: Install NVIDIA Container Toolkit:
```bash
# RHEL/Fedora
sudo dnf install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=podman
sudo systemctl restart podman

# Ubuntu
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify: `podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.0-base nvidia-smi`

### Sandbox Unreachable in Docker

**Symptom**: Proxy health shows `sandbox:false`, V3 build verification fails.

**Fix**: Ensure all services are on the same Docker network:
```bash
# Check network
podman network ls
podman inspect atlas-proxy-container --format '{{.NetworkSettings.Networks}}'
```

For Podman rootless: use `--network=host` for all containers, or create a shared network:
```bash
podman network create atlas
podman-compose up -d  # compose handles networking automatically
```

### SELinux Blocking Container Access (Fedora/RHEL)

**Symptom**: Containers can't read mounted volumes, permission denied errors.

**Fix**:
```bash
# Allow container access to model directory
chcon -Rt svirt_sandbox_file_t ~/models/
# Or use :Z flag in volume mounts
podman run -v ~/models:/models:Z ...
```

---

## llama-server Issues

### Model Loading on CPU Instead of GPU

**Symptom**: Generation at ~2 tok/s instead of ~50 tok/s.

**Fix**: Ensure `--n-gpu-layers 99` is set. Check with:
```bash
nvidia-smi  # Should show llama-server process using GPU
```

If using Docker, ensure the NVIDIA runtime is configured (see GPU section above).

### Grammar Not Enforced (Model Outputs Thinking Blocks)

**Symptom**: Model outputs `<think>` tags or raw text instead of JSON tool calls.

**Fix**: Ensure `response_format: {"type": "json_object"}` is in the request. The proxy sets this automatically when `ATLAS_AGENT_LOOP=1`.

If using llama-server directly, verify your build supports `response_format`:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-9B-Q6_K","messages":[{"role":"user","content":"Say hi in JSON"}],"max_tokens":20,"response_format":{"type":"json_object"}}'
```

### Context Window Too Small

**Symptom**: Tool calls get truncated, write_file fails with "unexpected end of JSON".

**Fix**: Increase `--ctx-size` to 32768 (default in Docker Compose). Check:
```bash
ps aux | grep llama-server | grep ctx-size
```

---

## Proxy Issues

### Agent Loop Not Activating

**Symptom**: Requests go directly to llama-server without tool calls.

**Fix**: Set `ATLAS_AGENT_LOOP=1` environment variable. The `atlas` command sets this automatically.

### V3 Pipeline Not Firing on Feature Files

**Symptom**: All write_file calls go direct (T1), no V3 pipeline steps in logs.

**Fix**: V3 fires on files with 50+ lines AND 3+ logic indicators. Check the proxy log:
```bash
tail -f logs/atlas-proxy.log | grep "write_file"
# Should show: [write_file] app.py → T2:medium (V3 pipeline activating)
```

If V3 service is unreachable: `curl http://localhost:8070/health`

### Truncation Errors (write_file fails repeatedly)

**Symptom**: `Content was truncated — the file is too large for write_file`.

**Fix**: The model is trying to rewrite an entire large file. The proxy automatically redirects to `edit_file` for existing files >100 lines. If the error persists:
- Ask for smaller, targeted changes instead of full rewrites
- The error loop breaker stops after 3 consecutive failures

### File Deletion Not Working

**Symptom**: Model calls delete_file but file reappears.

**Fix**: File deletion uses a fast-path before the tier classifier. It requires the proxy to detect the real project directory. If the wrong directory is detected, the file is deleted from the wrong location.

Check proxy log for: `delete fast-path:` and `deleted:` lines.

---

## Geometric Lens Issues

### Lens Model Not Loaded

**Symptom**: `Lens model: NOT LOADED` in service startup.

**Fix**: Model weights (`cost_field.pt`, `metric_tensor.pt`) must be in `geometric-lens/geometric_lens/models/`. Download from HuggingFace:
Download from [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS).

### Low C(x) Scores (All Near 0.5)

**Symptom**: All candidates score ~0.5 regardless of quality.

**Fix**: The Lens model may not be loaded. Check:
```bash
curl http://localhost:8099/internal/lens/gx-score \
  -H "Content-Type: application/json" \
  -d '{"text": "print(1)"}'
```

If `cx_energy: 0.0` and `gx_score: 0.5`, the model is not loaded.

---

## Aider Issues

### Timeout on Long Tasks

**Symptom**: Aider disconnects before the agent loop completes.

**Fix**: The Aider timeout has been increased to 24 hours. If you see timeouts, check:
```python
# In aider/models.py
request_timeout = 86400  # Should be 86400, not 600
```

### Empty Response (No Text Displayed)

**Symptom**: `[done: 0 files, 0 tool calls]` but no text shown.

**Fix**: For conversational responses, the agent loop returns text via `formatForAider`. If the model only emits a `done` signal without text, the response appears empty. This is a model behavior issue with very short prompts.

### Wrong Working Directory

**Symptom**: Files created in wrong location, `list_directory` shows stale test dirs.

**Fix**: The proxy detects the project directory from Aider's `.aider.chat.history.md` timestamps. If you have many Aider sessions open, the most recently modified one wins. Close other Aider sessions or specify the directory explicitly.

---

## Performance

### Slow Generation (~2 tok/s)

The model is running on CPU. Check:
1. `nvidia-smi` — is llama-server using GPU?
2. `--n-gpu-layers 99` — is GPU offloading enabled?
3. Container runtime — is NVIDIA runtime configured?

Expected: ~51 tok/s on RTX 5060 Ti with grammar enforcement.

### V3 Pipeline Takes 5+ Minutes Per File

Normal for T2 files. The pipeline makes 10+ LLM calls (PlanSearch, DivSampling, scoring, repair). Each call takes ~10s at 51 tok/s. Total: 2-5 minutes per T2 file.

To skip V3 for faster (lower quality) results, make files shorter (<50 lines) or simpler (fewer logic indicators).
