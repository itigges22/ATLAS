# ATLAS Troubleshooting Guide

Common issues and solutions for ATLAS V3.1.0, organized by service.

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
  "lens_ready": true,
  "sandbox": true,
  "port": "8090",
  "stats": { "requests": 0, "repairs": 0, "sandbox_passes": 0, "sandbox_fails": 0 }
}
```

If any field is `false`, that service is the problem. `status` flips to `"degraded"` whenever any of `inference`, `lens`, `lens_ready`, or `sandbox` is false. The split between `lens` and `lens_ready` (PC-019) lets you tell "Lens process is up but its `/ready` gate is failing — usually missing weights or embedding-dim mismatch" apart from "Lens HTTP is unreachable."

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

4. **Generate a CDI spec (newer toolkit replaces "legacy" mode):**
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

**Fix:** force a compatible gfx version at runtime via `ATLAS_HSA_OVERRIDE_GFX_VERSION`. Common overrides:

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

> Tested on AMD Radeon AI PRO R9700 (gfx1201) with ROCm 7.2, `ATLAS_ROCM_TAG=7.2.3-complete`. ATLAS PC-202 patch applies cleanly to the pinned llama.cpp SHA. Inference runs correctly across text generation and embedding generation without any additional flags.

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

**Fix (already applied in `inference/Dockerfile.v31`):** the Dockerfile uses `git clone --depth 1 --single-branch` with `http.postBuffer=524288000` and `http.lowSpeedLimit/Time` to fail-fast on dead connections instead of hanging for 11 minutes. If you have an older Dockerfile or the issue recurs:

1. Retry the build — transient network blips happen, especially on residential connections.
2. If retries keep failing, pre-pull the repo on the host and bind-mount it into the build context. Quick recipe:
   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
   # then edit Dockerfile.v31 to COPY from /tmp/llama.cpp instead of cloning
   ```
3. Long term: prebuilt llama-server images on GHCR will skip this step entirely (Phase 0 roadmap item).

### Rebuilding llama.cpp for a new model architecture

**Symptom:** A freshly dropped-in model fails to load with:

```
error loading model: unknown (model) architecture 'gemma4'
```

**Cause:** The `atlas-llama` image bundles a llama.cpp built at a fixed point in
time. A model whose architecture was added to llama.cpp *after* your image was
built won't load until you rebuild the inference image against newer llama.cpp.

**Fix:** rebuild just the llama-server image (this is the one legitimate reason
to do a long rebuild — `atlas onboard` will tell you when it's needed; it will
**not** rebuild for you):

```bash
# The image pins llama.cpp via LLAMA_CPP_REV (see Dockerfile.v31) so a
# plain rebuild reuses the SAME revision and won't learn new architectures.
# Point the pin at a llama.cpp commit that includes your model's arch:
docker compose build --build-arg LLAMA_CPP_REV=<sha> llama-server   # ~70 min on CUDA
docker compose up -d llama-server --no-deps
```

> ⚠️ **Preserve ATLAS's custom llama.cpp patches — do not strip them.** The build
> re-applies `inference/patches/expose-hidden-states.patch` (PC-202: the
> per-layer `hidden_states` extension the Geometric Lens depends on for SAE /
> lens-as-PRM) to the freshly-cloned source, via the `git apply` step in
> `inference/Dockerfile.v31` (shared by `Dockerfile.rocm` / `Dockerfile.vulkan`).
> When upstream has drifted, that step fails and the build aborts:
>
> ```
> error: patch failed: tools/server/server-context.cpp:NN
> error: tools/server/server-context.cpp: patch does not apply
> ```
>
> **The fix is to rebase the patch, NOT to delete it or remove the `git apply`
> line.** Removing it builds a working server that has silently lost the lens
> plumbing (`/embedding` will ignore the `layers:` parameter). To rebase:
>
> 1. Shallow-fetch the SHA you're bumping to (see the bump runbook in
>    "llama.cpp patch drift" below for the fetch recipe).
> 2. `cd /tmp/llcpp && git apply --reject inference/patches/expose-hidden-states.patch`
>    — clean hunks apply; failures land in `*.rej`.
> 3. Re-insert each rejected hunk at its (moved) anchor. These are additive, so
>    watch for upstream **renames** in the surrounding code — e.g. the server
>    context member `model` → `model_tgt` — and update the patch's added lines to
>    match.
> 4. Regenerate: `git diff > expose-hidden-states.patch`, validate with
>    `git apply --check` on a clean checkout, and (recommended) compile just the
>    touched file CPU-only to catch member/type errors before the 70-min CUDA
>    build: `cmake -B build-cpu -DGGML_CUDA=OFF && make -C build-cpu server-context`.
> 5. Replace `inference/patches/expose-hidden-states.patch` and rebuild.

After the rebuild loads the model, the Geometric Lens still needs retraining for
the new model — see [CONFIGURATION.md § Adding your own model](CONFIGURATION.md#adding-your-own-model-drop-in--unregistered).

### llama.cpp patch drift (when the publish workflow fails at "patch does not apply")

**Symptom:** The `Build & publish container images` workflow fails in the `llama` job with:

```
error: patch failed: tools/server/server-context.cpp:36
error: tools/server/server-context.cpp: patch does not apply
```

**Cause:** The PC-202 hidden-states patch (`inference/patches/expose-hidden-states.patch`) is pinned against a specific llama.cpp SHA via the `LLAMA_CPP_REV` build arg in all four Dockerfiles. When upstream llama.cpp shifts context around the patch target (e.g. a blank line removed, an include reordered), the patch's expected line numbers stop matching even though the SHA pin is still valid. This usually means someone bumped `LLAMA_CPP_REV` without regenerating the patch against the new SHA.

The CI smoke test (`tests` workflow, `llama.cpp patches apply to pinned SHA` job) catches this in ~30 seconds before the 30+ minute publish workflow burns runner time. If you see this fail locally instead of in CI, follow the bump runbook below.

**Bump runbook** — when you need to move `LLAMA_CPP_REV` forward (new llama.cpp feature, security fix, or an old SHA you no longer want to pin):

1. **Find a candidate SHA.** Browse [ggml-org/llama.cpp commits](https://github.com/ggml-org/llama.cpp/commits/master) — pick something recent that includes the feature/fix you want.

2. **Verify the existing patch still applies.** Fast check, no Docker needed:
   ```bash
   mkdir -p /tmp/llama-check && cd /tmp/llama-check
   git init -q llama.cpp && cd llama.cpp
   git remote add origin https://github.com/ggml-org/llama.cpp
   git fetch --depth 1 origin <NEW_SHA>
   git checkout -q FETCH_HEAD
   git apply --check $REPO/inference/patches/expose-hidden-states.patch
   ```
   (Only this patch is `git apply`-ed by the build. The spec-decode
   embeddings fix is applied as a `sed` inside the Dockerfiles and is a
   no-op when its target line is absent — don't `git apply` it.)

3. **If it applies cleanly:** great, just bump `LLAMA_CPP_REV` in all four Dockerfiles (`Dockerfile`, `Dockerfile.v31`, `Dockerfile.rocm`, `Dockerfile.vulkan`) to the new SHA. The CI smoke test will verify all four agree.

4. **If a patch fails:** regenerate it against the new SHA.
   ```bash
   cd /tmp/llama-check/llama.cpp
   # Apply the OLD patch's intent manually (look at the patch body to see
   # what hunks should land), then:
   git diff > $REPO/inference/patches/expose-hidden-states.patch
   ```
   Re-run step 2 to verify, then bump the four Dockerfiles.

5. **Walk forward, not backward.** If you can't find a recent SHA where the patch applies, prefer regenerating the patch over pinning to an older SHA — pinning further into the past means missing upstream fixes.

**Why no automatic patch-against-master CI?** That would notify us of upstream drift as soon as it happens, but it would also notify us constantly (llama.cpp moves fast) and there's nothing actionable until we want to bump. The pinned SHA + smoke test pattern gates on intent: drift becomes a problem only when someone tries to move forward.

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
largest fully-on-GPU configuration (see [CLI.md § atlas tier fit](CLI.md#atlas-tier-fit-pc-208)).
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
4. **Request tier ≥ T2** (classifier output, after any agent override) **AND** the file's own tier ≥ T2 (PC-042)

**Both** `write_file` and `edit_file` route through V3 since PC-042. Before that, only `write_file` did — and since the system prompt steers the model toward `edit_file` for all changes to existing files, V3 effectively never ran on real edits. If you're on a build that predates PC-042, that's why.

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

**What happens automatically:**
- For existing files > 100 lines: proxy rejects `write_file` and tells the model to use `edit_file` instead
- After 3 consecutive failures: error loop breaker stops the agent and returns a summary

**What you can do:** Rephrase your request to ask for targeted changes rather than full file rewrites. For example, "Add input validation to the login function" instead of "Rewrite auth.py".

**False positives, pre-PC-040.** Before PC-040, *any*
`unexpected end of JSON` from a tool's input parser was
relabeled "tool call truncated." The most common trigger
was the model emitting a tool call with **no `args` field
at all** — e.g. `{"type":"tool_call","name":"read_file"}`
— which is malformed input, not truncated output. The old
remap then steered the model toward `edit_file` of a file
it had never read, looping until the 3-error breaker
fired. PC-040 fixes this in two ways:

1. Empty/missing `args` is caught **before** the tool's
   `Execute` runs, and the proxy returns a per-tool hint
   like `read_file: no arguments provided. Call with
   {"path":"<file>"}. Use list_directory {"path":"."}
   first if you need to discover what files exist.`
2. The "truncated" remap now only fires when the args
   payload is over 200 bytes (real truncation territory).
   Short or empty args fall through to the actual parser
   error.

If you still see "tool call truncated" after PC-040 ships,
it's a real truncation — the model was actually trying to
write a payload too long for the context window. The
`edit_file` advice still applies in that case.

**PC-041 alt-shape lifting.** Some models emit tool calls
in OpenAI-style (`arguments` instead of `args`),
Anthropic-style (`parameters`), or with arguments inlined
at the top level (`{"name":"read_file","path":"x.py"}`).
The proxy now normalizes all three shapes into the
canonical `args` envelope automatically. If a tool call
still arrives with empty args after normalization, the
proxy logs `[agent] turn=N EMPTY ARGS — raw model output:
"..."` so you can see the exact shape it sent and either
add it to the lift list or rephrase the prompt.

### Long Pause Between Tool Result and Next Action

**Symptom:** A tool succeeds, then the agent loop sits
idle for ~30 seconds before the next turn fires. No
errors, no output — eventually the next tool call appears.

**Cause (PC-043).** Some local models under a constrained JSON
grammar occasionally emit zero
tokens after a tool result. The grammar requires the
response to start with `{`, but the model's natural
continuation after a tool result is a brief whitespace /
acknowledgment, which the grammar rejects. The model
emits EOS as its first token, returning empty content,
which the parse-error retry path then has to recover
from with a fresh user message — that's the lost ~30
seconds.

PC-043 catches this inside `callLLMConstrained` and
retries inline once with `temperature=0.7` and a
transient continuation nudge appended to the messages.
The agent loop never sees the empty turn.

**Diagnose:**
```bash
docker compose logs atlas-proxy | grep -E "PC-043|empty LLM|raw_len=0"
```
- `[agent] empty LLM response (PC-043), retrying with
  temp=0.7 + continuation nudge` — the retry fired; if
  the next log line is a normal `turn=N type=tool_call ...`
  the recovery worked.
- `parse error: ... raw_len=0 | raw: ""` — both the
  initial call AND the PC-043 retry returned empty. The
  outer parse-error retry will handle it, but you'll see
  the long pause. If this happens consistently, model is
  in a worse state than PC-043 anticipates — file a
  follow-up ticket with the full proxy log.

**Workaround if PC-043 isn't enough:** Restart the proxy
to clear llama.cpp's slot cache:
```bash
docker compose restart atlas-proxy llama-server
```

### Model Keeps Editing After V3 Already Confirmed the Fix

**Symptom:** The agent makes a successful V3-verified
edit (the TUI shows V3 progress events ending in
`Probe passed`), then re-reads the same file and starts
editing other unrelated functions. Each follow-on edit
triggers another full V3 cycle (~110s), and the new edits
sometimes touch code that has nothing to do with the
original bug.

**Cause (PC-044).** Compact local models can have trouble
self-assessing "is the user's original problem solved?"
After a tool result with `v3_used=true,
phase_solved=probe`, it has no strong signal to stop, so
it just continues planning more work.

**What PC-044 does.** Immediately after a V3-verified
write_file or edit_file, the agent loop appends a strong
user-role nudge: *"V3 verified this edit passed its
{phase} pipeline. The fix is on disk and build-checked.
If this resolves the user's original request, respond
NOW with {"type":"done","summary":"..."}. Only continue
if you have a specific, concrete additional change to
make — do not re-read the file to double-check, and do
not edit unrelated code."*

**Diagnose:**
```bash
docker compose logs atlas-proxy | grep "PC-044"
```
- `[agent] PC-044: V3-verified edit_file on ... — nudging
  toward done` — the nudge fired. The next agent turn
  should be `type=done`. If it isn't, the model ignored
  the nudge — file a follow-up ticket noting the
  prompt and the next-turn tool call.

**If the model still won't stop after PC-044:** The
follow-up options (hard-stop after re-read, per-file
edit cap, or auto-done from the proxy) are listed in
ISSUES.md PC-044 under "Caveat — promote to a harder
option if the soft nudge doesn't stick."

### Model Hallucinates Filenames From Previous Sessions

**Symptom:** Brand-new session, fresh prompt about a file
in the current directory, and the model's first tool call
is a `read_file` on a filename that doesn't exist
anywhere in this workspace — usually a filename that
*does* exist somewhere else you've worked recently.

**Cause (PC-045).** llama.cpp's KV slot persists between
chat completions to keep the cache warm (PC-035). Across
*sessions*, that means residual attention bias from the
previous session's tokens leaks into the new session.
Most prompts dominate this bias, but model-fabricated
filenames and other low-entropy outputs can pick it up.

**What PC-045 does.** Every `runAgentLoop` invocation
(one per user turn) starts by POSTing
`/slots/0?action=erase` to llama-server. The KV cache is
reset; the next chat completion re-encodes the system
prompt from scratch (~1-2s on warm GPU). Within the
session, subsequent turns share the now-fresh cache as
normal.

**Diagnose:**
```bash
docker compose logs atlas-proxy | grep "PC-045"
```
- `[PC-045] erased llama slot 0 — fresh KV cache for
  this session` on every user turn — working as
  intended.
- `[PC-045] erase slot: ...` followed by an error — the
  HTTP call to llama-server failed. Slot may still hold
  stale state, but next chat completion will partially
  overwrite it. Worst case: pre-PC-045 behavior.

**Disable** if you measure the per-message ~1-2s blip
and decide it's worse than occasional cross-session
leakage:
```bash
# .env
ATLAS_FRESH_SLOT_PER_SESSION=0
```
Restart the proxy after changing.

**Workaround if PC-045 is somehow disabled and you see
hallucinations:** Restart `llama-server` to fully clear
all slots:
```bash
docker compose restart llama-server
```

### Multi-File Project: Sandbox `ModuleNotFoundError`

**Symptom:** Edit on a file that imports another module
in the same project. V3 reports verification failure
with `ModuleNotFoundError: No module named 'utils'` (or
similar) even though the import works fine on your
machine.

**Cause (PC-046).** Pre-PC-046 the sandbox wrote *only*
the candidate file as `solution.py` to its workspace.
Any `from utils import …` failed because `utils.py`
didn't exist in the sandbox's tmpdir.

**What PC-046 does.** Sandbox `/execute` accepts a
`files: Dict[str, str]` map; V3's `SandboxAdapter`
ships every file the agent has read (the same
`ProjectContext` dict V3 already feeds to the LLM
prompt) into the sandbox workspace alongside
`solution.py`. Multi-file imports resolve.

**Diagnose:** if you still see `ModuleNotFoundError`
in V3 progress events, the file is probably not in
`ctx.FilesRead` (the proxy's read-tracking set). Read
the missing file via `read_file` so it lands in the
project context that V3 ships to the sandbox.

**If you're using the sandbox `/execute` API directly**
(scripts, tests), pass the supporting files in the
request body:
```bash
curl -X POST http://localhost:30820/execute -d '{
  "code": "from utils import greet\nprint(greet(\"x\"))",
  "language": "python",
  "files": {"utils.py": "def greet(n): return f\"hi {n}\""}
}'
```

### Curses Bottom-Row `addwstr() returned ERR`

**Symptom:** Your curses program (snake game, TUI menu,
status bar, etc.) crashes at runtime with:
```
_curses.error: addwstr() returned ERR
```
…but ATLAS reported the edit passed V3 verification.

**Cause.** Writing to the last cell of a curses window
(any row=LINES-1, or column=COLS-1) is documented as
returning ERR. This is decades-old curses behavior. The
idiomatic fix:
```python
try:
    stdscr.addstr(curses.LINES - 1, 0, border)
except curses.error:
    pass  # writing the bottom-right cell errors; benign
```

**What PC-047 does.** `interactive_lint` now AST-walks
for `addstr/addnstr/addch(curses.LINES - N, ...)` (and
the bare `LINES - N` form after `from curses import LINES`)
that aren't inside a `try/except curses.error` block.
Such candidates are rejected at the lint gate — V3 has
to find a wrapped variant before certifying.

**Diagnose:**
```bash
docker compose logs v3-service | grep "interactive_lint"
```
- `[interactive_lint] OK` — candidate passed all checks.
- `[interactive_lint] FAIL: curses bottom-row write
  without try/except curses.error wrap — line N: ...` —
  PC-047 fired. V3 will either find a wrapped variant
  or surface the failure to the model so it can produce
  one.

**If V3 can't find a wrapped variant**, the model is in
the structural-reasoning gap (Issue B): it knows the
file uses `curses.LINES - 1` but can't reliably
synthesize the try/except wrap. Workaround: tell the
model explicitly in your prompt: *"wrap the
addstr call at line N in `try: ... except curses.error:
pass`."*

### V3 Hangs for Several Minutes on Non-Python Files

**Symptom:** Asking ATLAS to write an HTML/CSS/JSON file
causes a long pause (~5 minutes) with progress events
showing PR-CoT repair attempts and LLM timeouts. The
file eventually gets written via the direct-write
fallback, but the V3 cycle was wasted.

**Cause (PC-048).** Pre-PC-048 the V3 smoke check
hardcoded `compile(_src, '<smoke>', 'exec')` (Python AST
parse) for **every** interactive-task candidate — HTML,
CSS, JSON, anything. Any non-Python file failed the
smoke check with `SYNTAX_ERROR`, which kicked V3 into
PR-CoT repair, which made LLM calls that timed out, then
fell back to direct write.

**What PC-048 does.** `smoke_compile_check` is now
language-aware. The V3 pipeline derives language from
the target file's extension (`pipeline.run(file_path=…)`)
and routes:
- `.py` → Python parse/compile
- `.js` / `.mjs` / `.cjs` → `node --check`
- `.ts` / `.tsx` → `tsc --noEmit --strict`
- `.go` → `gofmt -e`
- `.rs` → `rustc` syntax check
- `.c` / `.cpp` → compiler `-fsyntax-only`
- `.sh` / `.bash` → `bash -n`
- `.html` / `.htm` → `html.parser`
- `.xml` → `xml.etree.ElementTree`
- `.json` → `json.loads`
- `.yaml` / `.yml` → `yaml.safe_load`

Unknown formats fail with an explicit "syntax verification
unavailable" error instead of passing without evidence.

If `/v3/generate` receives an approved project build command, V3 emits a
`build_verify` event after syntax/self-test verification. The command runs in
an ephemeral sandbox workspace with the candidate overlaid onto the project, so
failed build evidence blocks `passed=true` without writing the candidate into
the real checkout. Overlay snapshots intentionally skip dependency caches,
secrets, model/data artifacts, symlinks, and large files, and enforce file-count
and byte limits. If a project needs heavyweight dependencies to build, install
them inside the sandbox workspace as part of the explicit verification workflow
or use release/container qualification for that project.

**Diagnose:**
```bash
docker compose logs v3-service | grep "smoke_check"
```
- `[smoke_check] compile=OK (html)` — PC-048 routed
  correctly.
- `[smoke_check] compile=OK (python)` on a `.html` file —
  the proxy didn't pass `file_path` through. Check
  `proxy/v3_bridge.go` and the
  `V3GenerateRequest`.
- `[smoke_check] compile=FAIL` followed by
  `[phase3] All candidates failed — entering repair
  phase` followed by `[LLM] Attempt N failed: timed
  out` — the cascade PC-048 was supposed to prevent
  is happening anyway. File a follow-up ticket with
  the failing file extension.

**If you're hitting this on a file extension PC-048
doesn't recognize**, the smoke check defaults to Python
and you get the same cascade. Workaround: add the
extension to `_ext_to_lang` in `v3-service/main.py`
(see the existing dispatch table around the `_ext_to_lang`
constant) and rebuild the `v3-service` image. As an
immediate escape valve, the proxy falls back to a direct
write when V3 errors out — so the file does eventually
land on disk, just slowly.

### V3 Pipeline Doesn't Fire on "Fix It Again" Prompts

**Symptom:** First request creates a file, V3 pipeline
runs (you see V3 progress events). Follow-up "still
doesn't work, try again"-style prompts complete in
microseconds with no V3 events visible. The model just
edits and exits without verification.

**Cause (PC-049).** Pre-PC-049 the agent-loop tier
classifier checked a narrow vocabulary (`fix`,
`broken`, `doesn't work`, `bug`, …) and required at
least one explicit file extension in the prompt. Real
iterative-fix prompts use natural phrases ("still does
not", "isn't working", "try again") with no `.py` in
sight, so the classifier returned T1, V3 never fires.

**What PC-049 does.** Vocabulary expanded to cover
natural fix language (`doesn't`, `is not`, `aren't`,
`failed`, `wrong`, etc.), plus a separate
"continuation marker" detector (`still`, `again`,
`retry`, `another`). Continuation markers substitute
for explicit file names — if you say "still doesn't
work" we now know you mean "the existing file isn't
working" even if you don't name it.

**Diagnose:**
```bash
docker compose logs atlas-proxy | grep "agent tier override"
```
- `agent tier override: T2:medium` — PC-049 promoted
  correctly. V3 should fire on the next edit_file.
- `agent tier override: T1:simple` on a clearly-iterative
  prompt — PC-049's vocabulary missed it. File a
  follow-up ticket with the exact prompt; the
  vocabulary is finite.

**Workaround if classifier still misses your prompt:**
Mention the file by name in the prompt — `app.py` is
enough. The original `fileIndicators >= 1` gate still
works for explicit file mentions.

### File Not Read Before Editing

**Symptom:** `edit_file` fails with "file not read yet — use read_file first before editing."

**Cause:** The proxy tracks which files the agent has read. If the model tries to edit a file it hasn't read in this session, the edit is rejected as a staleness protection.

**Fix:** This is normal behavior — the model should read the file first. If it keeps failing, the model may be confused about which files it has seen. Type `/clear` in the TUI to reset chat history and rephrase.

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
