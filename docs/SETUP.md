# ATLAS V2 Setup Guide

This guide covers installation and configuration of ATLAS V2. The V2 architecture removes the Qdrant vector database and standalone embedding service in favor of PageIndex (AST-based tree indexing with BM25) and llama-server's built-in embedding endpoint.

---

## Prerequisites

- **Python:** 3.10+ (tested on 3.11)
- **Operating system:** RHEL 9 or Ubuntu 24.04 (tested on RHEL 9 in a Proxmox VM with GPU passthrough)
- **GPU:** NVIDIA GPU with 16GB+ VRAM and CUDA support (tested: RTX 5060 Ti, compute capability 12.0)
- **Cluster:** K3s or K8s with the NVIDIA device plugin installed
- **Container runtime:** containerd (bundled with K3s) or Docker
- **System RAM:** 14GB minimum
- **Storage:** NVMe SSD recommended; models and data require roughly 20GB

The NVIDIA driver must be installed on the host before proceeding. Run `nvidia-smi` to confirm.

---

## Model Downloads

Two GGUF model files are required. Place both in the same directory (the path configured as `ATLAS_MODELS_DIR` in `atlas.conf`).

| Model | File | Size | Purpose |
|-------|------|------|---------|
| Qwen3-14B (Q4_K_M) | `Qwen3-14B-Q4_K_M.gguf` | ~8.38 GiB | Main inference model |
| Qwen3-0.6B (Q8_0) | `Qwen3-0.6B-Q8_0.gguf` | ~610 MiB | Speculative decoding draft model |

Download with the provided script or manually:

```bash
mkdir -p ~/models

# Option A: use the download script
./scripts/download-models.sh

# Option B: manual download with huggingface-cli
huggingface-cli download Qwen/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf --local-dir ~/models
huggingface-cli download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir ~/models
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/itigges22/atlas.git
cd atlas
```

### Step 2: Configure

```bash
cp atlas.conf.example atlas.conf
vim atlas.conf
```

At minimum, set the following:

```bash
# Directory containing the two GGUF files
ATLAS_MODELS_DIR="/home/yourusername/models"

# Directory for persistent data (indexes, training data, projects)
ATLAS_DATA_DIR="/opt/atlas/data"

# Main model filename
ATLAS_MAIN_MODEL="Qwen3-14B-Q4_K_M.gguf"

# Draft model for speculative decoding
ATLAS_DRAFT_MODEL="Qwen3-0.6B-Q8_0.gguf"

# Context window (V2 default)
ATLAS_CONTEXT_LENGTH=40960

# Parallel inference slots
ATLAS_PARALLEL_SLOTS=2

# GPU layers (99 = full offload)
ATLAS_GPU_LAYERS=99
```

See [Configuration Reference](CONFIGURATION.md) for all available options.

### Step 3: Run the Installer

```bash
sudo ./scripts/install.sh
```

The installer will:

1. Check prerequisites (NVIDIA driver, GPU VRAM, system RAM).
2. Install K3s if not already present.
3. Install the NVIDIA GPU Operator via Helm.
4. Create namespace and secrets.
5. Build container images for all services.
6. Process manifest templates (substitute config values).
7. Deploy services to K3s in the `atlas` namespace.
8. Wait for all pods to reach Running status.

The first build takes 1-2 hours because it downloads CUDA base images (~8GB) and compiles llama.cpp from source. Subsequent builds use cached layers and complete much faster.

### Step 4: Verify Installation

```bash
./scripts/verify-install.sh

# Or check manually
kubectl get pods -n atlas
```

---

## Verification

### Expected Pods

The V2 deployment runs these pods:

```
NAME                           READY   STATUS    RESTARTS   AGE
llama-server-xxx               1/1     Running   0          5m
rag-api-xxx                    1/1     Running   0          5m
redis-xxx                      1/1     Running   0          5m
sandbox-xxx                    1/1     Running   0          5m
task-worker-xxx                1/1     Running   0          5m
api-portal-xxx                 1/1     Running   0          5m    (MaaS — optional)
llm-proxy-xxx                  1/1     Running   0          5m    (MaaS — optional)
```

The MaaS services (api-portal, llm-proxy) provide multi-user API access and are optional for single-user or benchmark-only deployments. The atlas-dashboard deployment exists but is currently scaled to 0 replicas (non-essential for benchmarks).

### Health Checks

```bash
# llama-server health
curl http://localhost:32735/health

# rag-api health
curl http://localhost:31144/health

# api-portal health
curl http://localhost:30000/health

# llm-proxy health
curl http://localhost:30080/health

# sandbox health
curl http://localhost:30820/health
```

### Run the V2 Benchmark

```bash
./benchmark/run_v2_benchmark.sh
```

Expected results on RTX 5060 Ti with Qwen3-14B-Q4_K_M:

| Metric | Expected Range |
|--------|---------------|
| LiveCodeBench pass@1 | 36-41% |
| Throughput | ~109 tasks/hr |

---

## Feature Flags

The rag-api deployment supports these runtime feature flags, set as environment variables in `manifests/rag-api-deployment.yaml`:

| Variable | Default (code) | Default (manifest) | Description |
|----------|----------------|---------------------|-------------|
| `ROUTING_ENABLED` | `"true"` | `"true"` | Enables the Confidence Router (Thompson Sampling route selection) |
| `GEOMETRIC_LENS_ENABLED` | `"false"` | `"true"` | Enables Geometric Lens energy scoring for candidate selection |

To disable a feature, edit the manifest and re-apply:

```bash
kubectl apply -n atlas -f manifests/rag-api-deployment.yaml
```

---

## Troubleshooting

See [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues including mlock failures, speculative decoding slot errors, and GPU memory pressure.

---

## Next Steps

- [Configuration Reference](CONFIGURATION.md) -- all configuration options and environment variables
- [Architecture](ARCHITECTURE.md) -- system design and component interactions
- [Troubleshooting](TROUBLESHOOTING.md) -- diagnosing and resolving common problems
