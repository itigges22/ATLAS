# ATLAS Setup Guide

Three deployment methods: Docker Compose (recommended and tested), bare-metal, and K3s.

---

## Prerequisites (All Methods)

| Requirement | Details |
|-------------|---------|
| **NVIDIA GPU** | 16GB+ VRAM (tested on RTX 5060 Ti 16GB) |
| **NVIDIA drivers** | Proprietary drivers installed (`nvidia-smi` should show your GPU) |
| **Python 3.9+** | With pip |
| **HuggingFace CLI** (or wget) | For downloading model weights |
| **Model weights** | QuantTrio/Qwen3.5-9B-AWQ (~12GB AWQ-Q4 directory of safetensors shards) |

### Verify GPU

```bash
nvidia-smi
# Should show your GPU with driver version and VRAM
# If this fails, install NVIDIA proprietary drivers first
```

---

## Method 1: Docker Compose (Recommended)

This is the tested deployment method for V3.0.1.

### Additional Prerequisites

- **Docker** with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), **or Podman**
- ~20GB disk space (model weights + container images)

### Setup

```bash
# 1. Clone
git clone https://github.com/itigges22/ATLAS.git
cd ATLAS

# 2. Download model weights (~12 GiB AWQ-Q4 directory of safetensors shards).
#    vLLM doesn't load GGUF natively — it consumes the AWQ build directly.
#    `make model` runs the same huggingface-cli pull under the hood.
make model
# Or directly:
#   pip install -q huggingface_hub
#   huggingface-cli download QuantTrio/Qwen3.5-9B-AWQ \
#       --local-dir models/Qwen3.5-9B-AWQ --local-dir-use-symlinks False

# 3. Install the ATLAS CLI + Aider
pip install -e . aider-chat

# 4. (Recommended) Install Go 1.24+ for full file access from any directory
#    https://go.dev/dl/ — the proxy builds automatically on first run
#    Without Go, the proxy runs in Docker with file access limited to ATLAS_PROJECT_DIR

# 5. Configure environment
cp .env.example .env
# Defaults work if your model is in ./models/ — edit .env only if you changed the path

# 6. Start all services (first run builds container images — this takes several minutes)
docker compose up -d         # or: podman-compose up -d

# 7. Verify everything is healthy (wait for all services to show "healthy")
docker compose ps

# 8. Start coding (from your project directory)
cd /path/to/your/project
atlas
```

### What Happens on First Run

1. Docker builds the local images:
   - **vllm-gen** + **vllm-embed** — pulled prebuilt from `vllm/vllm-openai:latest` (no compile step)
   - **geometric-lens** — installs PyTorch CPU + FastAPI
   - **v3-service** — installs PyTorch CPU + benchmark modules
   - **sandbox** — installs Node.js, Go, Rust, gcc
   - **atlas-proxy** — compiles Go binary
2. vLLM gen + embed each load the AWQ weights into GPU VRAM (~1-2 min each)
3. All services start health checks
4. Once vllm-gen, vllm-embed, geometric-lens, v3-service, sandbox, and atlas-proxy all report healthy, `atlas` connects and launches Aider

Subsequent `docker compose up -d` starts are fast (seconds) since images are cached.

### Verify Installation

```bash
# Check each service individually
curl -s http://localhost:8000/health | python3 -m json.tool   # vLLM
curl -s http://localhost:31144/health | python3 -m json.tool   # geometric-lens
curl -s http://localhost:8070/health | python3 -m json.tool   # v3-service
curl -s http://localhost:30820/health | python3 -m json.tool  # sandbox
curl -s http://localhost:8090/health | python3 -m json.tool   # atlas-proxy

# Quick functional test (requires aider: pip install aider-chat)
atlas --message "Create hello.py that prints hello world"
```

All health endpoints should return `{"status": "ok"}` or `{"status": "healthy"}`.

> **Note:** The `atlas` command auto-detects the proxy and launches Aider for the full agent loop (tool calls, V3 pipeline, file read/write). If Aider is not installed, it falls back to the built-in REPL which supports `/solve` and `/bench` but not file operations. Install Aider for the full experience: `pip install aider-chat`

### Stopping

```bash
docker compose down          # Stop all services (preserves images)
docker compose down --rmi all  # Stop and remove images (next start rebuilds)
```

### Viewing Logs

```bash
docker compose logs -f vllm-gen        # Follow vLLM gen logs
docker compose logs -f vllm-embed      # Follow vLLM embed logs
docker compose logs -f geometric-lens  # Follow Lens logs
docker compose logs -f v3-service      # Follow V3 pipeline logs
docker compose logs -f atlas-proxy     # Follow proxy logs
docker compose logs -f sandbox         # Follow sandbox logs
docker compose logs --tail 50          # Last 50 lines from all services
```

### Updating

```bash
git pull
docker compose down
docker compose build         # Rebuild changed images
docker compose up -d
```

---

## Method 2: Bare Metal

Run all services as local processes without containers. Useful for development or systems where Docker isn't available.

### Additional Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python 3.10+** | vLLM 0.17 requires Python ≥ 3.10 (the project itself is 3.9+, but bare-metal vLLM is the binding constraint) |
| **Go 1.24+** | For building atlas-proxy |
| **vLLM** | `pip install -U --pre vllm` (>=0.17 for Qwen3.5 DeltaNet support; CUDA 12.8) |
| **Aider** | `pip install aider-chat` |
| **Node.js 20+** | Required by sandbox for JavaScript/TypeScript execution |
| **Rust** | Required by sandbox for Rust execution |

### Build

```bash
# 1. Clone and install Python CLI
git clone https://github.com/itigges22/ATLAS.git
cd ATLAS
pip install -e .

# 2. Download AWQ-quantized model weights (~12GB, vLLM-compatible)
#    Use the HuggingFace CLI rather than wget — model is sharded.
pip install huggingface_hub
huggingface-cli download QuantTrio/Qwen3.5-9B-AWQ --local-dir models/Qwen3.5-9B-AWQ

# 3. Install vLLM (nightly required for Qwen3.5 DeltaNet kernels)
pip install -U --pre --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/nightly "vllm==0.17.1"  # 0.18.0 has a Qwen3.5-specific regression — see vllm-project/vllm#37749

# 4. Build atlas-proxy
cd atlas-proxy
go build -o ~/.local/bin/atlas-proxy-v2 .
cd ..

# 5. Install geometric-lens Python dependencies
pip install -r geometric-lens/requirements.txt

# 6. Install sandbox dependencies
pip install fastapi uvicorn pylint pytest pydantic
```

### Start Services

vLLM serves only one task per process, so the inference layer runs as **two**
vLLM instances side by side: one for chat completions, one for embeddings.

```bash
# Terminal 1: vLLM gen (chat completions on port 8000)
vllm serve models/Qwen3.5-9B-AWQ \
  --served-model-name qwen3.5-9b \
  --host 0.0.0.0 --port 8000 \
  --max-num-seqs 32 --max-model-len 32768 \
  --gpu-memory-utilization 0.55 \
  --tensor-parallel-size 1 \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --trust-remote-code

# Terminal 2: vLLM embed (4096-dim hidden states for the Lens, on port 8001)
# --runner pooling + --convert embed: the current API for serving a generation
# model via /v1/embeddings (the older --task embed still works but is
# deprecated in vLLM 0.17+).
vllm serve models/Qwen3.5-9B-AWQ \
  --served-model-name qwen3.5-9b-embed \
  --runner pooling \
  --convert embed \
  --host 0.0.0.0 --port 8001 \
  --max-num-seqs 8 --max-model-len 4096 \
  --gpu-memory-utilization 0.20 \
  --tensor-parallel-size 1 \
  --trust-remote-code

# Terminal 3: Geometric Lens
cd geometric-lens
LLAMA_GEN_URL=http://localhost:8000 \
LLAMA_EMBED_URL=http://localhost:8001 \
GEOMETRIC_LENS_ENABLED=true \
PROJECT_DATA_DIR=/tmp/atlas-projects \
python -m uvicorn main:app --host 0.0.0.0 --port 31144

# Terminal 4: V3 Pipeline
cd v3-service
LLAMA_GEN_URL=http://localhost:8000 \
LLAMA_EMBED_URL=http://localhost:8001 \
ATLAS_LENS_URL=http://localhost:31144 \
ATLAS_SANDBOX_URL=http://localhost:8020 \
python main.py

# Terminal 5: Sandbox
cd sandbox
python executor_server.py

# Terminal 6: atlas-proxy
ATLAS_PROXY_PORT=8090 \
LLAMA_GEN_URL=http://localhost:8000 \
LLAMA_EMBED_URL=http://localhost:8001 \
ATLAS_LENS_URL=http://localhost:31144 \
ATLAS_SANDBOX_URL=http://localhost:8020 \
ATLAS_V3_URL=http://localhost:8070 \
ATLAS_AGENT_LOOP=1 \
LLAMA_GEN_MODEL=qwen3.5-9b \
LLAMA_EMBED_MODEL=qwen3.5-9b-embed \
atlas-proxy-v2
```

> **VRAM**: 16 GB consumer cards fit only the gen instance at concurrency 1-4.
> Embed needs another ~5-8 GB; for full Lens-on benchmarks you need 24 GB+
> (e.g. 4090, A40) or run gen-only with `GEOMETRIC_LENS_ENABLED=false`.

> **Note:** The sandbox listens on port **8020** in bare-metal mode (no Docker port remapping). The proxy's `ATLAS_SANDBOX_URL` must use port 8020, not 30820.

### Start with the Launcher Script

Alternatively, copy the launcher script to your PATH:

```bash
cp /path/to/atlas-launcher ~/.local/bin/atlas
chmod +x ~/.local/bin/atlas
atlas    # Starts all missing services and launches Aider
```

The launcher auto-detects which services are already running and starts only what's missing. If it detects a Docker Compose stack, it connects to that instead.

---

## Method 3: K3s

For production Kubernetes deployment with GPU scheduling, health probes, and resource limits.

### Additional Prerequisites

| Requirement | Details |
|-------------|---------|
| **K3s** | Single-node or multi-node cluster |
| **NVIDIA GPU Operator** or **device plugin** | GPU must be visible as `nvidia.com/gpu` resource |
| **Helm** | For GPU Operator installation |
| **Podman or Docker** | For building container images |

### Automated Install

The install script handles the complete setup — K3s installation, GPU Operator, container builds, and deployment:

```bash
# 1. Configure
cp atlas.conf.example atlas.conf
# Edit atlas.conf: model paths, GPU layers, context size, NodePorts

# 2. Run the installer (requires root)
sudo scripts/install.sh
```

The installer will:
1. Check prerequisites (NVIDIA drivers, GPU VRAM, system RAM)
2. Install K3s if not already running
3. Install NVIDIA GPU Operator via Helm (if GPU not visible to cluster)
4. Build container images and import to K3s containerd
5. Generate manifests from `atlas.conf` via envsubst
6. Deploy to the `atlas` namespace
7. Wait for all services to be healthy

### Manual Deploy

If K3s is already running with GPU support:

```bash
# 1. Configure
cp atlas.conf.example atlas.conf
# Edit atlas.conf

# 2. Build and import images
scripts/build-containers.sh

# 3. Generate manifests from atlas.conf
scripts/generate-manifests.sh

# 4. Deploy
kubectl apply -n atlas -f manifests/

# 5. Verify
scripts/verify-install.sh
```

### K3s-Specific Configuration

K3s uses `atlas.conf` (not `.env`) for configuration. Key differences from Docker Compose:

| Setting | Docker Compose | K3s |
|---------|---------------|-----|
| Config file | `.env` | `atlas.conf` |
| Context size | 32K | 40K per slot (× 4 slots = 160K total) |
| Parallel slots | 1 (implicit) | 4 |
| Flash attention | Off | On |
| KV cache quantization | None | q8_0 (keys) + q4_0 (values) |
| Memory locking | No | mlock enabled |
| Embeddings endpoint | Not exposed | `--embeddings` flag |
| Service exposure | Host ports | NodePorts |

See [CONFIGURATION.md](CONFIGURATION.md) for the full `atlas.conf` reference.

### Verify K3s Deployment

```bash
# Check pods
kubectl get pods -n atlas

# Check GPU allocation
kubectl describe nodes | grep nvidia.com/gpu

# Run verification suite
scripts/verify-install.sh
```

> **Note:** Docker Compose is the verified deployment method for V3.0.1. K3s manifests are generated from templates at deploy time. The K3s deployment was used for V3.0 benchmarks on Qwen3-14B and is production-tested, but the template files may need adjustment for your cluster configuration.

---

## Hardware Sizing

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| GPU VRAM | 16 GB | 16 GB | Model (~7GB) + KV cache (~1.3GB) + overhead |
| System RAM | 14 GB | 16 GB+ | PyTorch runtime + container overhead |
| Disk | 15 GB | 25 GB | Model (7GB) + container images (5-8GB) + working space |
| CPU | 4 cores | 8+ cores | V3 pipeline is CPU-intensive during repair phases |

### Supported GPUs

Any NVIDIA GPU with 16GB+ VRAM and CUDA support. Tested on:
- RTX 5060 Ti 16GB (primary development GPU)

AMD and Intel GPUs are not yet tested. vLLM supports ROCm and other backends — ROCm support is a V3.1 priority.

#### CUDA Compute Capability (Dockerfile.v31)

`benchmarks/h200/Dockerfile` compiles vLLM for a specific CUDA compute capability. The default is `120;121` (Blackwell, RTX 50xx). If you see build failures like `nvcc fatal: unsupported gpu architecture` or runtime errors like `no kernel image available for execution`, your GPU needs a different arch.

Override at build time with `--build-arg CUDA_ARCH=<value>`:

```bash
# Single arch — RTX 4060/4070/4080/4090 (Ada Lovelace)
podman build --build-arg CUDA_ARCH=89 -f benchmarks/h200/Dockerfile -t vLLM:local inference/

# Multiple archs (semicolon-separated) — build a fat binary for Ampere + Ada + Hopper
podman build --build-arg CUDA_ARCH="86;89;90" -f benchmarks/h200/Dockerfile -t vLLM:local inference/
```

Common values:

| Arch | Architecture | Cards |
|------|--------------|-------|
| `60`, `61` | Pascal | GTX 10xx, Tesla P4/P40 |
| `70` | Volta | V100 |
| `75` | Turing | RTX 20xx, T4 |
| `80`, `86` | Ampere | A100, RTX 30xx |
| `89` | Ada Lovelace | RTX 40xx, L4 |
| `90` | Hopper | H100 |
| `100`, `120`, `121` | Blackwell | B100, RTX 50xx |

Your GPU's compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv` (drop the dot — `8.9` → `89`).

---

## Geometric Lens Weights (Optional)

ATLAS works without Geometric Lens weights — the service degrades gracefully, returning neutral scores. The V3 pipeline falls back to sandbox-only verification.

To enable C(x)/G(x) scoring, you need trained model weights. Pre-trained weights and training data are available on HuggingFace:

**[ATLAS Dataset on HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS)** — includes embeddings, training data, and weight files.

Place weight files in `geometric-lens/geometric_lens/models/` (or mount via `ATLAS_LENS_MODELS` in Docker Compose). The service loads them automatically on startup.

Training scripts are provided in `scripts/` if you want to train on your own benchmark data:
- `scripts/retrain_cx_phase0.py` — Initial C(x) training from collected embeddings
- `scripts/retrain_cx.py` — Production C(x) retraining with class weights
- `scripts/collect_lens_training_data.py` — Collect pass/fail embeddings from benchmark runs
- `scripts/prepare_lens_training.py` — Prepare and validate training data format

---

## Next Steps

- [CLI.md](CLI.md) — How to use ATLAS once it's running
- [CONFIGURATION.md](CONFIGURATION.md) — All environment variables and tuning options
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues and solutions
- [ARCHITECTURE.md](ARCHITECTURE.md) — How the system works internally
