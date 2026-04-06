# ATLAS Setup Guide

Three deployment methods: Docker Compose (recommended), bare-metal, and K3s.

---

## Prerequisites (All Methods)

- **NVIDIA GPU** with 16GB+ VRAM (tested on RTX 5060 Ti)
- **NVIDIA drivers** installed and working (`nvidia-smi` shows the GPU)
- **Model weights**: Download `Qwen3.5-9B-Q6_K.gguf` (~6.9GB)

---

## Method 1: Docker Compose (Recommended)

### Prerequisites
- Docker or Podman with `podman-compose`
- NVIDIA Container Toolkit (`nvidia-container-toolkit` package)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/itigges22/ATLAS.git
cd ATLAS

# 2. Download model weights
mkdir -p models
# Download Qwen3.5-9B-Q6_K.gguf into models/
# Source: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF

# 3. Configure environment
cp .env.example .env
# Edit .env — set ATLAS_MODELS_DIR to absolute path of your models directory

# 4. Build and start
podman-compose up -d    # or: docker compose up -d

# 5. Verify all services are healthy
podman-compose ps       # All should show "healthy"

# 6. Launch ATLAS
atlas
```

### Verify Installation

```bash
# Check service health
curl http://localhost:8080/health   # llama-server
curl http://localhost:8099/health   # geometric-lens
curl http://localhost:8070/health   # v3-service
curl http://localhost:8090/health   # atlas-proxy

# Quick test
atlas --message "Create hello.py that prints hello world"
```

### Stop

```bash
podman-compose down     # or: docker compose down
```

---

## Method 2: Bare Metal

### Prerequisites
- **Go 1.24+** (for building atlas-proxy)
- **Python 3.10+** with pip
- **llama.cpp** built with CUDA from [llama-cpp-mtp](https://github.com/ggml-org/llama.cpp)
- **Aider** (`pip install aider-chat`)

### Build

```bash
# Build atlas-proxy
cd atlas-proxy
go build -o ~/.local/bin/atlas-proxy-v2 .

# Install Python dependencies for geometric-lens
cd ../geometric-lens
pip install -r requirements.txt

# Install Python dependencies for V3 service
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Start Services Manually

```bash
# 1. Start llama-server
llama-server \
  --model ~/models/Qwen3.5-9B-Q6_K.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 32768 --n-gpu-layers 99 --no-mmap &

# 2. Start geometric-lens
cd geometric-lens
LLAMA_URL=http://localhost:8080 \
GEOMETRIC_LENS_ENABLED=true \
python -m uvicorn main:app --host 0.0.0.0 --port 8099 &

# 3. Start V3 service
cd ../v3-service
ATLAS_INFERENCE_URL=http://localhost:8080 \
ATLAS_LENS_URL=http://localhost:8099 \
ATLAS_SANDBOX_URL=http://localhost:8020 \
python main.py &

# 4. Start sandbox
cd ../sandbox
python executor_server.py &

# 5. Start proxy
ATLAS_PROXY_PORT=8090 \
ATLAS_INFERENCE_URL=http://localhost:8080 \
ATLAS_LLAMA_URL=http://localhost:8080 \
ATLAS_LENS_URL=http://localhost:8099 \
ATLAS_SANDBOX_URL=http://localhost:8020 \
ATLAS_V3_URL=http://localhost:8070 \
ATLAS_AGENT_LOOP=1 \
atlas-proxy-v2 &

# 6. Launch ATLAS
atlas
```

Or use the `atlas` command which starts all services automatically:

```bash
atlas    # Detects and starts any missing services
```

---

## Method 3: K3s

K3s manifests are generated at runtime from `atlas.conf`.

### Prerequisites
- K3s cluster with NVIDIA device plugin
- `kubectl` configured
- Podman for building container images

### Deploy

```bash
# 1. Configure
cp atlas.conf.example atlas.conf
# Edit atlas.conf with your model paths, ports, etc.

# 2. Build container images
scripts/build-containers.sh

# 3. Generate manifests
scripts/generate-manifests.sh

# 4. Deploy
kubectl apply -f manifests/ -n atlas

# 5. Verify
scripts/verify-install.sh
```

> **Note**: Docker Compose is the verified deployment method for V3.0.1. K3s manifests are provided but may need adjustment for your cluster.

---

## Configuration

See [CONFIGURATION.md](CONFIGURATION.md) for all environment variables and config options.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.
