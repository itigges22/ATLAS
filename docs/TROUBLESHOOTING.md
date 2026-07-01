# ATLAS Troubleshooting Guide

Common issues and solutions, organized by service.

---

## Quick Diagnostics

Run these first to identify where the problem is:

```bash
# Docker Compose — check all services at once
docker compose ps

# GPU status
nvidia-smi

# Docker Compose logs (last 50 lines per service)
docker compose logs --tail 50
```

For the per-service health-check curls, see [SETUP.md § Verify Installation](SETUP.md#verify-installation). The atlas-proxy health endpoint is the most useful for triage — it reports the status of all upstream services:
```json
{
  "status": "ok",
  "inference": true,
  "lens": true,
  "lens_ready": true,
  "sandbox": true,
  "port": "8090",
  "stats": { "requests": 0, "repairs": 0, "sandbox_passes": 0, "sandbox_fails": 0 }
}
```

If any field is `false`, that service is the problem. `status` flips to `"degraded"` whenever any of `inference`, `lens`, `lens_ready`, or `sandbox` is false. The split between `lens` and `lens_ready` lets you tell "Lens process is up but its `/ready` gate is failing — usually missing weights or embedding-dim mismatch" apart from "Lens HTTP is unreachable."

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

### `libnvidia-ml.so.1: cannot open shared object file`

**Symptom:** During `docker compose up`, llama-server fails with:

```
nvidia-container-cli: initialization error: load library failed:
libnvidia-ml.so.1: cannot open shared object file: no such file or directory
```

**What it means:** the host has the NVIDIA *kernel module* (so `nvidia-smi` works) but the *userspace driver libraries* aren't where the container toolkit expects. On RHEL/Rocky/Alma minimal installs the `nvidia-driver-cuda-libs` package isn't pulled in by default; on Debian/Ubuntu the issue is usually a stale `ldconfig` cache after a driver upgrade.

**Fix sequence** — try in order, stop when `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` works:

1. **Refresh ldconfig + restart docker:**
   ```bash
   sudo ldconfig
   sudo systemctl restart docker
   ```

2. **RHEL 9 — add CUDA repo + install open-dkms module** (verified working on RHEL 9.7 with RTX 5060 Ti):
   ```bash
   # Add NVIDIA's CUDA repo
   sudo dnf config-manager --add-repo \
     https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

   # Enable CodeReady Builder (provides dkms / kernel-devel)
   sudo subscription-manager repos --enable=codeready-builder-for-rhel-9-x86_64-rpms

   # Make sure EPEL is present
   sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm

   # Install the open driver module (REQUIRED for Blackwell — RTX 50xx)
   sudo dnf module install -y nvidia-driver:open-dkms

   sudo ldconfig && sudo systemctl restart docker
   ```

   **Rocky/Alma/CentOS Stream 9** — same as above, but replace the `subscription-manager` line with:
   ```bash
   sudo dnf config-manager --set-enabled crb
   ```

   > Note: the `nvidia-driver-cuda-libs` package only exists once the NVIDIA CUDA repo is added. RHEL 9's stock `BaseOS`/`AppStream` repos do not ship NVIDIA packages. The `nvidia-driver:open-dkms` module is **required** for Blackwell GPUs (RTX 5060/70/80/90); older GPUs accept either open or proprietary.

3. **Ubuntu/Debian — install matching userspace libs:**
   ```bash
   DRV_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
   sudo apt install -y libnvidia-compute-${DRV_MAJOR}
   sudo ldconfig && sudo systemctl restart docker
   ```

4. **Generate a CDI spec:**
   ```bash
   sudo mkdir -p /etc/cdi
   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
   docker run --rm --device=nvidia.com/gpu=all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

The `atlas-bootstrap.sh` script now runs steps 1, 2 (auto-detects RHEL/Rocky/Alma vs subscription path), and 4 automatically. Step 3 is auto-handled on Debian/Ubuntu via `libnvidia-compute-NN` matched to the running driver version.

### AMD GPU not detected (ROCm)

**Symptom:** `atlas tier` says "no GPU detected" on a host that clearly has an AMD GPU, OR `docker compose up` fails with `/dev/kfd: no such file or directory`.

**What it means:** the `amdgpu` kernel driver isn't loaded with compute support (the `kfd` — Kernel Fusion Driver — submodule). Display-only loads of `amdgpu` don't expose `/dev/kfd`.

**Fix sequence:**

1. **Verify the driver is loaded and `/dev/kfd` exists:**
   ```bash
   lsmod | grep amdgpu       # should print amdgpu + amdkfd
   ls -l /dev/kfd            # should print a character-device entry
   ls -l /dev/dri/render*    # should print one or more render nodes
   ```

2. **Install ROCm + kernel driver (if /dev/kfd is missing):**
   - **RHEL 9 / Rocky / Alma:**
     ```bash
     sudo dnf install -y https://repo.radeon.com/amdgpu-install/6.2/rhel/9.4/amdgpu-install-6.2.60200-1.el9.noarch.rpm
     sudo amdgpu-install --usecase=dkms,rocm
     sudo reboot   # required — the kernel module needs a fresh boot
     ```
   - **Ubuntu/Debian:** follow [the official AMD install guide](https://rocm.docs.amd.com/projects/install-on-linux/) for your distro. The typical sequence is `amdgpu-install --usecase=dkms,rocm` after adding the AMDGPU repo.

3. **After reboot, confirm `rocm-smi` sees the GPU:**
   ```bash
   rocm-smi --showproductname --showmeminfo vram
   ```

### AMD GPU detected but Docker can't reach it

**Symptom:** `atlas doctor` reports "AMD GPU detected but Docker can't reach `/dev/kfd`" or the ROCm container fails with `Permission denied` on `/dev/kfd`.

**What it means:** the user running Docker isn't in the `render` and/or `video` groups. ROCm uses those groups to gate access to `/dev/kfd` and `/dev/dri/render*`.

**Fix:**

```bash
# 1. Confirm which groups you're currently in
id -nG | tr ' ' '\n' | grep -E '^(render|video)$'
# Expect both. If either is missing:

# 2. Create the groups if they don't exist (rare; default on most distros)
sudo groupadd -f render
sudo groupadd -f video

# 3. Add your user to both
sudo usermod -aG video,render $USER

# 4. Re-login (or use newgrp for the current shell)
newgrp render
newgrp video

# 5. Re-verify, then re-run `atlas doctor`
id -nG | grep -E 'render.*video|video.*render'
atlas doctor
```

### AMD GPU is "unsupported" by ROCm but you want to try anyway

**Symptom:** `rocm-smi` reports your GPU, but `rocminfo` doesn't, or HIP kernels fail with "no kernel image is available for execution on the device."

**What it means:** llama.cpp's HIP kernels were compiled for `gfx` targets that don't include your GPU. ROCm has a long-standing pattern of dropping older consumer GPUs from official support while still letting them work with the right override.

**Fix:** force a compatible gfx version at runtime via `ATLAS_HSA_OVERRIDE_GFX_VERSION`. Common overrides (for the canonical card→gfx table, see [SETUP.md § AMD GPU Targets](SETUP.md#amd-gpu-targets-dockerfilerocm)):

| Your GPU | Set `ATLAS_HSA_OVERRIDE_GFX_VERSION=` |
|---|---|
| RDNA1 (RX 5700 XT / 5500 XT) | `10.3.0` (makes it look like RDNA2 / gfx1030) |
| Vega 56/64 (gfx900) | `9.0.0` (usually already supported, override rarely needed) |
| Polaris (RX 580/590, gfx803) | `8.0.3` (deep override; mileage varies) |

Set the var in `.env` so it propagates through the compose override into the container env:

```bash
echo "ATLAS_HSA_OVERRIDE_GFX_VERSION=10.3.0" >> .env
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d --force-recreate llama-server
```

If this works for you on a previously-unsupported card, please leave a note on [GH #26](https://github.com/itigges22/ATLAS/issues/26) — community-tested overrides feed into the next release's docs.

### RDNA4 (RX 9070 / 9070 XT, gfx1200 / gfx1201) — ROCm 7.x required

**Symptom:** Build fails during `docker compose ... build llama-server` with errors like `error: AMDGPU target 'gfx1201' is not supported`, or the container starts but immediately exits with a HIP initialization error.

**What it means:** The default ROCm base image (`rocm/dev-ubuntu-22.04:6.2-complete`) predates RDNA4. The gfx1200 and gfx1201 compiler targets were added in ROCm 7.0 — see the [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) for the full supported hardware list.

**Fix:** Set `ATLAS_ROCM_TAG` to a ROCm 7.x tag before building:

```env
# Add to your .env
ATLAS_ROCM_TAG=7.2.3-complete
ATLAS_GFX_TARGET=gfx1201   # gfx1200 for RX 9070, gfx1201 for RX 9070 XT
```

Then rebuild and bring up the stack:

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml build llama-server
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d
```

**Important: do NOT set `ATLAS_HSA_OVERRIDE_GFX_VERSION` for gfx1200/gfx1201.** ROCm 7.0+ supports these targets natively; overriding the GFX version inside Docker causes a mismatch between the compiled kernels and the runtime target, which results in crashes. Leave `ATLAS_HSA_OVERRIDE_GFX_VERSION` unset (the default).

> Tested on AMD Radeon AI PRO R9700 (gfx1201) with ROCm 7.2, `ATLAS_ROCM_TAG=7.2.3-complete`. The hidden-states patch applies cleanly to the pinned llama.cpp SHA. Inference runs correctly across text generation and embedding generation without any additional flags.

### ROCm container can't pull `rocm/rocm-terminal`

**Symptom:** `atlas doctor` ROCm check times out at the image pull, or `docker compose -f ... -f docker-compose.rocm.yml pull` fails on the `llama-server` build.

**What it means:** ROCm images are large (~2 GB) and Docker Hub rate-limits anonymous pulls.

**Fix:** authenticate (free Docker Hub account allows higher rate limits), or pull during off-peak hours, or pin to a specific tag in `.env`:

```bash
docker login
ATLAS_ROCM_TAG=6.2-complete docker compose -f docker-compose.yml -f docker-compose.rocm.yml pull
```

### First Build Fails (CUDA Not Found)

**Symptom:** `docker compose build` fails with CUDA-related errors during llama-server compilation.

**Fix:** The llama-server Dockerfile builds llama.cpp inside a `nvidia/cuda:12.8.0-devel` base image, so CUDA headers are available during build without host GPU access. Common causes of build failure:
1. Insufficient disk space (~5GB needed for build artifacts)
2. Network issues downloading the CUDA base image or cloning llama.cpp
3. Podman rootless builds may fail with permission issues — try `podman-compose build` with `--podman-build-args="--format docker"`

### llama.cpp Clone Times Out

**Symptom:** Build hangs in the `llama-server builder 3/3` stage and eventually fails with:

```
error: RPC failed; curl 56 OpenSSL SSL_read: Connection timed out, errno 110
fatal: early EOF
fatal: fetch-pack: invalid index-pack output
```

**Cause:** The full llama.cpp git history is large (~1 GB) and the clone is sensitive to flaky/slow connections. A momentary stall causes the SSL read to time out and the whole transfer to abort.

**Fix:** `inference/Dockerfile.v31` uses `git clone --depth 1 --single-branch` with `http.postBuffer=524288000` and `http.lowSpeedLimit/Time` to fail-fast on dead connections. If you have an older Dockerfile or the issue recurs:

1. Retry the build — transient network blips happen, especially on residential connections.
2. If retries keep failing, pre-pull the repo on the host and bind-mount it into the build context. Quick recipe:
   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
   # then edit Dockerfile.v31 to COPY from /tmp/llama.cpp instead of cloning
   ```
3. Prebuilt llama-server images on GHCR skip this step entirely — pull instead of building.

### Rebuilding llama.cpp (new model architecture, or patch drift)

Developer-maintenance task. Two triggers land here:

- **A dropped-in model fails to load** with `error loading model: unknown (model) architecture 'gemma4'` — the pinned llama.cpp predates that architecture.
- **A build fails** with `error: patch failed: tools/server/server-context.cpp:NN` / `patch does not apply` — upstream drifted past the pinned SHA.

The `atlas-llama` image pins llama.cpp via `LLAMA_CPP_REV` in all four Dockerfiles (`Dockerfile`, `Dockerfile.v31`, `Dockerfile.rocm`, `Dockerfile.vulkan`) and re-applies `inference/patches/expose-hidden-states.patch` (the per-layer `hidden_states` extension the Geometric Lens depends on) during the build. To learn a new architecture, move the pin to a llama.cpp SHA that includes it. Prebuilt GHCR images skip the local build; only rebuild when you need an architecture newer than the published image.

**Preserve the hidden-states patch — rebase it, don't delete it.** Removing the `git apply` step builds a server that has silently lost the lens plumbing (`/embedding` ignores the `layers:` parameter). Bump runbook:

1. **Verify the patch against the target SHA** (fast, no Docker):
   ```bash
   mkdir -p /tmp/llama-check && cd /tmp/llama-check
   git init -q llama.cpp && cd llama.cpp
   git remote add origin https://github.com/ggml-org/llama.cpp
   git fetch --depth 1 origin <NEW_SHA> && git checkout -q FETCH_HEAD
   git apply --check $REPO/inference/patches/expose-hidden-states.patch
   ```
   (Only this patch is `git apply`-ed. The spec-decode embeddings fix is a `sed` in the Dockerfiles, a no-op when its target line is absent.)
2. **If it applies cleanly:** bump `LLAMA_CPP_REV` in all four Dockerfiles to the new SHA. The CI smoke test verifies they agree.
3. **If it fails:** `git apply --reject …` to land the clean hunks, re-insert each `*.rej` hunk at its moved anchor (watch for upstream renames in surrounding code, e.g. `model` → `model_tgt`, and update the patch's added lines), then `git diff > $REPO/inference/patches/expose-hidden-states.patch`. Re-run step 1. Compile just the touched file CPU-only to catch member/type errors before the long CUDA build: `cmake -B build-cpu -DGGML_CUDA=OFF && make -C build-cpu server-context`.
4. Rebuild and bring up:
   ```bash
   docker compose build --build-arg LLAMA_CPP_REV=<sha> llama-server
   docker compose up -d llama-server --no-deps
   ```

Prefer regenerating the patch over pinning to an older SHA — pinning backward means missing upstream fixes.

After the rebuild loads the model, the Geometric Lens still needs retraining for the new model — see [CONFIGURATION.md § Adding your own model](CONFIGURATION.md#adding-your-own-model-drop-in--unregistered).

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

### Model + KV cache don't fit on the GPU (startup fails, or generation is 5× slow)

**Symptom (current entrypoint):** llama-server exits at startup with a CUDA
allocation error right after "fitting params to device memory".

**Symptom (older entrypoints without `--fit off`):** the server *starts* and
`nvidia-smi` shows the model loaded, but generation runs at a fraction of the
expected speed, the llama-server process burns several CPU cores
(`top` shows 400–800%), and its host RSS holds gigabytes of model weights —
llama.cpp's memory auto-fitter silently moved layers to the CPU.

**Cause:** the model's weights plus the KV cache (`ATLAS_CTX_SIZE` ×
`PARALLEL` slots × per-layer KV dims) plus the compute buffer
(~`ATLAS_UBATCH` × hidden-dim × 280 bytes) exceed VRAM. These budgets are
per-model — a config tuned for one model can overflow on another with
different KV geometry.

**Fix:** size the runtime for this model + GPU and recreate the container:
```bash
atlas tier fit --write
docker compose up -d llama-server --no-deps --force-recreate
```
`atlas tier fit` reads the GGUF header and your GPU's VRAM and solves for the
largest fully-on-GPU configuration (see [CLI.md § atlas tier fit](CLI.md#atlas-tier-fit)).
ATLAS runs llama-server with `--fit off` so a config that doesn't fit fails
loudly at startup instead of silently running partly on the CPU.

If `atlas tier fit` reports **DOES NOT FIT**, the model itself is too large
for the card — the output names the largest quant file size that *would* fit.
In order of preference:

1. **Use a smaller quant of the same model** (e.g. Q4_K_M instead of Q6_K —
   usually the best quality-per-GiB trade below 16 GB VRAM).
2. **Reduce parallel slots**: `atlas tier fit --slots 1 --write` frees the
   per-slot KV minimum (drops `/demo` split-pane and V3 parallel candidates,
   single-stream use still works).
3. **Pick a smaller model.** See the sizing table below.

### What fits on my GPU?

Approximate rule before you download anything: on the default 4 slots, a GGUF
fits comfortably when

```
file size  ≤  VRAM − ~4.5 GiB
```

(the ~4.5 GiB covers the minimum KV cache at 4 × 8k context, compute buffers,
and the ~1.9 GiB fixed CUDA overhead). With `--slots 1` the margin shrinks to
roughly `VRAM − 3 GiB`. Sliding-window models (Gemma-style) need less than
this; the rule is sized for full-attention models.

| VRAM | GGUF file size (4 slots) | GGUF file size (1 slot) | Typical models |
|------|--------------------------|--------------------------|----------------|
| 8 GB | ≤ ~3 GiB | ≤ ~4.5 GiB | 3–4B Q4–Q6, 7–8B Q2–Q3 |
| 12 GB | ≤ ~7 GiB | ≤ ~8.5 GiB | 7–9B Q4–Q6, 12B Q3–Q4 |
| 16 GB | ≤ ~11 GiB | ≤ ~12.5 GiB | 9B Q6–Q8, 12–14B Q4–Q6 |
| 24 GB | ≤ ~19 GiB | ≤ ~20.5 GiB | 14B Q8, 27–32B Q4 |

HuggingFace model pages list the file size per quant — check it against this
table before downloading. The table is a pre-download estimate only; once the
file is on disk, `atlas tier fit /path/to/model.gguf` is authoritative (it
reads the model's real KV geometry, which can swing the budget by gigabytes
in either direction), and `atlas onboard` prints the same fit automatically.

### Model File Not Found

**Symptom:** llama-server exits immediately with "failed to load model" or similar.

**Fix:** Check the model path:
```bash
# Docker Compose — model must be in ATLAS_MODELS_DIR (default: ./models/)
ls -la "models/$ATLAS_MODEL_FILE"

# Bare metal — check ATLAS_MODEL_PATH
ls -la "$ATLAS_MODELS_DIR/$ATLAS_MODEL_FILE"
```

The filename must match the required `ATLAS_MODEL_FILE` selection in `.env`.

### Out of VRAM

**Symptom:** llama-server crashes or gets OOMKilled shortly after starting. `nvidia-smi` shows VRAM near 100%.

**Fix:** Ensure:
1. No other GPU processes are running (`nvidia-smi` — check for other CUDA processes)
2. You have 16GB+ VRAM
3. The runtime is sized for your model + GPU: `atlas tier fit --write` (don't raise `ATLAS_CTX_SIZE` past what it recommends)

```bash
# Kill other GPU processes if needed
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill {}
```

### Grammar Not Enforced (Model Outputs Thinking Blocks)

**Symptom:** Model outputs `<think>` tags or raw text instead of JSON tool calls.

**Fix:** The proxy sets `response_format: {"type": "json_object"}` automatically inside the `/v1/agent` agent-loop handler — this is unconditional (no env-var toggle). If you're hitting llama-server directly via `/v1/chat/completions` or `/v1/completions`, you have to include the parameter yourself:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role":"user","content":"Say hi"}],
    "max_tokens": 50,
    "response_format": {"type": "json_object"}
  }'
```

If this returns raw text instead of JSON, your llama.cpp build doesn't support `response_format`. Rebuild from the latest source.

### Context Window Too Small

**Symptom:** Tool call arguments get truncated. `write_file` fails with "unexpected end of JSON" or proxy logs show "truncation detected".

**Fix:** Per-slot context (`ATLAS_CTX_SIZE` ÷ `ATLAS_PARALLEL_SLOTS`; compose default 131072 ÷ 4 = 32k per slot) may be too small for the task. `atlas tier fit` shows the largest budget your GPU supports. Check:
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

**Cause:** You're hitting the wrong endpoint. The agent loop only runs on `POST /v1/agent`. `POST /v1/chat/completions` (and anything else under `/v1/`) is a transparent passthrough to llama-server — no tools, no V3, no streaming chat events.

**Fix:** Point your client at `POST http://localhost:8090/v1/agent`. The Bubbletea TUI (`atlas` / `atlas tui`) and the built-in `/solve` REPL both do this automatically. If you're writing a third-party client, see [docs/API.md](API.md) for the `/v1/agent` SSE event protocol. There is no longer an `ATLAS_AGENT_LOOP` env-var toggle — the split is endpoint-based, not config-based.

### V3 Pipeline Not Firing on Feature Files

**Symptom:** All `write_file` *or* `edit_file` calls are T1 (direct write). No V3 pipeline stages in output.

V3 fires when **all conditions** are met:
1. File has **50+ lines** of content
2. File has **3+ logic indicators** (function defs, control flow, API patterns)
3. V3 service is reachable at `ATLAS_V3_URL`
4. **Request tier ≥ T2** (classifier output, after any agent override) **AND** the file's own tier ≥ T2

Both `write_file` and `edit_file` route through V3.

**Diagnose:**
```bash
# Check V3 service health
curl -s http://localhost:8070/health

# Check proxy logs for tier classification + V3 activation
docker compose logs atlas-proxy | grep -E "write_file|edit_file|tier="
# Look for:
#   "tier=T2:medium" or higher in classifier output
#   "[edit_file] V3 pipeline activating for X (req_tier=2, file_tier=2)"
#   "[write_file] V3 pipeline activating for X"
# T1 means direct write — no V3.
```

If V3 is unreachable, the proxy logs `V3 failed: ...` and falls back to direct write without breaking the edit.

### Truncation Errors (write_file Fails Repeatedly)

**Symptom:** Repeated errors like "Your output was truncated — the content is too long for a single tool call."

**Cause:** The model is trying to write too much content in one call. The proxy detects truncated JSON and rejects the tool call.

The proxy rejects `write_file` on existing files over 100 lines and tells the model to use `edit_file` instead; after 3 consecutive failures the error loop breaker stops the agent and returns a summary.

**Fix:** Rephrase your request to ask for targeted changes rather than full file rewrites — "Add input validation to the login function" instead of "Rewrite auth.py".

The proxy distinguishes real truncation (args payload over 200 bytes) from a tool call sent with empty or missing `args` — the latter gets a per-tool hint like `read_file: no arguments provided. Call with {"path":"<file>"}` instead of the truncation remap. It also normalizes OpenAI-style (`arguments`), Anthropic-style (`parameters`), and top-level-inlined argument shapes into the canonical `args` envelope. If a tool call still arrives empty after normalization, the proxy logs `[agent] turn=N EMPTY ARGS — raw model output: "..."` so you can see the exact shape and rephrase.

### Long Pause Between Tool Result and Next Action

**Symptom:** A tool succeeds, then the agent loop sits idle for ~30 seconds before the next turn fires. No errors, no output — eventually the next tool call appears.

**What's happening:** Under a constrained JSON grammar, some local models emit EOS as their first token after a tool result, returning empty content that the parse-error retry path has to recover from — that's the lost ~30 seconds.

**What to do:** The proxy catches the empty turn inside `callLLMConstrained` and retries inline once with `temperature=0.7` and a continuation nudge. If it recurs consistently, restart the proxy to clear llama.cpp's slot cache:
```bash
docker compose restart atlas-proxy llama-server
```
Check `docker compose logs atlas-proxy | grep -E "empty LLM|raw_len=0"` — `raw_len=0` on both the initial call and the retry means the model is in a worse state than the retry handles.

### Model Keeps Editing After V3 Already Confirmed the Fix

**Symptom:** The agent makes a successful V3-verified edit (the TUI shows V3 progress events ending in `Probe passed`), then re-reads the same file and starts editing unrelated functions. Each follow-on edit triggers another full V3 cycle (~110s).

**What's happening:** Compact local models can have trouble self-assessing "is the user's original problem solved?" and keep planning more work after a verified edit.

**What to do:** The agent loop appends a strong user-role nudge after a V3-verified write toward emitting `{"type":"done"}`. If the model ignores it, be more explicit in your prompt that the single change is all you want. Harder stops (per-file edit cap, auto-done) are tracked as follow-up options.

### Model Hallucinates Filenames From Previous Sessions

**Symptom:** Brand-new session, fresh prompt, and the model's first tool call is a `read_file` on a filename that doesn't exist in this workspace — usually one that exists somewhere else you've worked recently.

**What's happening:** llama.cpp's KV slot persists between chat completions to keep the cache warm. Across sessions, residual attention bias from the previous session's tokens can leak into low-entropy outputs like fabricated filenames.

**What to do:** Every user turn starts by erasing llama slot 0 so the next completion re-encodes the system prompt fresh (~1-2s on warm GPU). To disable the per-session erase if you'd rather keep the cache fully warm:
```bash
# .env
ATLAS_FRESH_SLOT_PER_SESSION=0
```
Restart the proxy after changing. If you see hallucinations with the erase disabled, restart `llama-server` to clear all slots.

### Multi-File Project: Sandbox `ModuleNotFoundError`

**Symptom:** Edit on a file that imports another module in the same project. V3 reports verification failure with `ModuleNotFoundError: No module named 'utils'` even though the import works on your machine.

**What's happening:** V3's `SandboxAdapter` ships every file the agent has read into the sandbox workspace alongside `solution.py`. A file that isn't in the read set (`ctx.FilesRead`) won't be there, so its imports fail.

**What to do:** Read the missing file via `read_file` so it lands in the project context. If you're calling the sandbox `/execute` API directly, pass supporting files in the request body:
```bash
curl -X POST http://localhost:30820/execute -d '{
  "code": "from utils import greet\nprint(greet(\"x\"))",
  "language": "python",
  "files": {"utils.py": "def greet(n): return f\"hi {n}\""}
}'
```

### Curses Bottom-Row `addwstr() returned ERR`

**Symptom:** Your curses program crashes at runtime with `_curses.error: addwstr() returned ERR`, but ATLAS reported the edit passed V3 verification.

**What's happening:** Writing to the last cell of a curses window (row=LINES-1 or column=COLS-1) returns ERR by documented curses behavior. `interactive_lint` rejects candidates that write there without a `try/except curses.error` wrap, so V3 has to find a wrapped variant before certifying. The idiomatic fix:
```python
try:
    stdscr.addstr(curses.LINES - 1, 0, border)
except curses.error:
    pass  # writing the bottom-right cell errors; benign
```

**What to do:** If V3 can't synthesize the wrap on its own, tell the model explicitly: *"wrap the addstr call at line N in `try: ... except curses.error: pass`."* Check `docker compose logs v3-service | grep interactive_lint` to confirm the lint gate fired.

### V3 Hangs for Several Minutes on Non-Python Files

**Symptom:** Asking ATLAS to write an HTML/CSS/JSON file causes a ~5-minute pause with PR-CoT repair attempts and LLM timeouts. The file eventually lands via the direct-write fallback.

**What's happening:** The V3 smoke check is language-aware — it derives language from the target file's extension and routes to the right checker (`.py` → Python compile, `.js` → `node --check`, `.ts` → `tsc --noEmit`, `.go` → `gofmt -e`, `.rs` → `rustc`, `.c`/`.cpp` → `-fsyntax-only`, `.sh` → `bash -n`, `.html` → `html.parser`, `.xml` → `ElementTree`, `.json` → `json.loads`, `.yaml` → `yaml.safe_load`). An unrecognized extension falls back to Python and fails, which cascades into repair.

If `/v3/generate` receives an approved project build command, V3 emits a `build_verify` event after syntax/self-test verification. The command runs in an ephemeral sandbox workspace with the candidate overlaid onto the project, so failed build evidence blocks `passed=true` without writing the candidate into the real checkout. Overlay snapshots skip dependency caches, secrets, model/data artifacts, symlinks, and large files, and enforce file-count and byte limits. If a project needs heavyweight dependencies to build, install them inside the sandbox workspace as part of the explicit verification workflow.

**What to do:** For an unrecognized extension, add it to `_ext_to_lang` in `v3-service/main.py` and rebuild the `v3-service` image. The proxy falls back to a direct write when V3 errors out, so the file lands regardless — just slowly. Check `docker compose logs v3-service | grep smoke_check` to confirm the right language was routed.

### V3 Pipeline Doesn't Fire on "Fix It Again" Prompts

**Symptom:** First request creates a file and V3 runs. Follow-up "still doesn't work, try again"-style prompts complete in microseconds with no V3 events — the model just edits and exits.

**What's happening:** The agent-loop tier classifier covers natural fix language (`doesn't`, `is not`, `failed`, `wrong`, …) plus continuation markers (`still`, `again`, `retry`, `another`) that substitute for an explicit file name. A prompt that misses this vocabulary classifies as T1, so V3 never fires.

**What to do:** Mention the file by name (`app.py` is enough) — the explicit-file gate promotes it to T2. Check `docker compose logs atlas-proxy | grep "agent tier override"`: `T2:medium` means it promoted correctly, `T1:simple` on a clearly-iterative prompt means the vocabulary missed it.

### File Not Read Before Editing

**Symptom:** `edit_file` fails with "file not read yet — use read_file first before editing."

**Cause:** The proxy tracks which files the agent has read. If the model tries to edit a file it hasn't read in this session, the edit is rejected as a staleness protection.

**Fix:** The model should read the file first. If it keeps failing, type `/clear` in the TUI to reset chat history and rephrase.

### File Modified Externally

**Symptom:** `edit_file` fails with "file modified since last read — read it again before editing."

**Cause:** The file was changed on disk (by you or another process) after the model read it. The proxy compares modification timestamps.

**Fix:** The model needs to re-read the file. This usually resolves automatically on the next turn.

### Exploration Budget Warning

**Symptom:** Output shows "You have full project context in the system prompt. Do not read more files." or reads are being skipped.

**Cause:** The model has made 4+ consecutive read-only calls (read_file, search_files, list_directory) without writing anything. After 4 reads, the proxy warns. After 5+, it skips reads entirely and tells the model to write.

**Fix:** If the model is genuinely stuck exploring, be more specific about what you want changed.

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

### `/internal/lens/retrain` Returns 503 "models directory is mounted read-only"

**Symptom:** POSTing `/internal/lens/retrain` on the lens service returns HTTP 503 with `"reason": "models directory is mounted read-only; run host-side retrain via atlas lens retrain"`.

**Cause:** The standard Compose deployment mounts the lens models directory into the container read-only (`:ro`), so the in-service retrain endpoint cannot write new weights. The endpoint probes writability before training and refuses up front rather than burning a training run.

**Fix:** Run the retrain host-side — `atlas lens retrain` (feedback corpus) or `atlas lens build` (bench candidates) write the artifacts on the host, then the service reloads them via `/internal/lens/reload` (or `docker compose restart geometric-lens`). Benchmark-driven online recalibration (`lens_feedback`) logs the refusal and keeps its sample buffer, so nothing is lost.

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

**Default timeout:** 30 seconds per request, capped at `MAX_EXECUTION_TIME`. The Compose stack sets that cap to 300 seconds (via `ATLAS_SANDBOX_MAX_EXECUTION_TIME` in `.env`), matching the proxy's `run_command` cap so long builds and test suites complete; outside compose the executor's in-code cap is 60 seconds.

**Fix:** If your code legitimately needs more time, set a higher timeout in the request (up to the cap), or raise `ATLAS_SANDBOX_MAX_EXECUTION_TIME`. If the code has an infinite loop, this is expected behavior. On timeout the whole process group is killed, so child processes the command spawned don't linger.

### Language Not Supported

**Symptom:** Sandbox returns an error for a specific language.

**Supported languages:** Python, JavaScript, TypeScript, Go, Rust, C, C++, Bash.

Check available runtimes:
```bash
curl -s http://localhost:30820/languages | python3 -m json.tool
```

---

## Benchmark Issues

### Bench runs fewer tasks than requested (`LIMITED MODE: running N tasks` with N below `--tasks`)

**Symptom:** `atlas bench --tasks 200` reports `LIMITED MODE: running 100
tasks` (or any count below what you asked for), or a resumed run prints
`Resuming: N/N complete, 0 remaining` and exits immediately.

**Cause:** the LiveCodeBench dataset cache
(`benchmark/datasets/.cache/livecodebench_v5.jsonl`) holds a partial
download. The HuggingFace rows API can fail mid-pagination; older versions
cached whatever they had and trusted the file forever. The full release_v5
set is ~880 tasks.

**Fix:** flag the cache as partial and re-run — the loader retries the full
fetch (falling back to the existing copy only if every source fails):
```bash
touch benchmark/datasets/.cache/livecodebench_v5.jsonl.partial
atlas bench --run-id <your-run-id> --tasks 200
```
Completed tasks are never lost: results live one-JSON-per-task under
`benchmark/results/<run-id>/v3_lcb/per_task/` and the runner resumes by
skipping any task whose result file exists. A run interrupted for any
reason (OOM, reboot, closed session) resumes the same way — just re-run
the identical `atlas bench` command.

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
