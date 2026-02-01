# ATLAS Setup Guide

This guide covers installation and configuration of ATLAS on a fresh system.

## Prerequisites

### Operating System

- RHEL 9, Rocky Linux 9, or compatible
- Ubuntu 22.04+ also supported
- 64-bit x86_64 architecture

### NVIDIA Drivers

ATLAS requires NVIDIA GPU drivers for inference acceleration.

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# If not installed (RHEL/Rocky):
sudo dnf install -y nvidia-driver nvidia-driver-cuda

# Reboot after installation
sudo reboot
```

### Container Runtime

ATLAS uses Podman for building containers and K3s for orchestration.

```bash
# Install Podman (RHEL/Rocky)
sudo dnf install -y podman

# Verify installation
podman --version
```

### K3s

```bash
# Install K3s with NVIDIA GPU support
curl -sfL https://get.k3s.io | sh -

# Configure kubectl
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
chmod 600 ~/.kube/config

# Verify
kubectl get nodes
```

### NVIDIA Device Plugin

```bash
# Install NVIDIA device plugin for Kubernetes
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU is available
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
```

## Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU | 8GB VRAM | 16GB+ VRAM | NVIDIA only |
| RAM | 16GB | 32GB | For K3s + services |
| Storage | 50GB SSD | 200GB+ SSD | Models + vector DB |
| CPU | 4 cores | 8+ cores | Embedding service CPU-bound |

### Tested Configurations

- **Development**: RTX 4060 Ti (16GB), 32GB RAM, 256GB NVMe
- **Production**: RTX 5060 Ti (16GB), 64GB RAM, 1TB NVMe

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/atlas.git
cd atlas
```

### Step 2: Configure Installation

```bash
# Copy example configuration
cp atlas.conf.example atlas.conf

# Edit configuration
vim atlas.conf
```

Required settings:
```ini
# Path to your GGUF model file
MODEL_PATH=/path/to/your/model.gguf

# GPU layers to offload (use 99 for full GPU)
GPU_LAYERS=99

# Context size (match your model's training context)
CONTEXT_SIZE=16384
```

### Step 3: Download Model

ATLAS works with any GGUF-format model. Recommended models:

```bash
# Create models directory
mkdir -p ~/models

# Example: Download Qwen model (requires huggingface-cli)
huggingface-cli download Qwen/Qwen3-14B-GGUF qwen3-14b-q4_k_m.gguf --local-dir ~/models
```

Update `MODEL_PATH` in atlas.conf:
```ini
MODEL_PATH=/home/yourusername/models/qwen3-14b-q4_k_m.gguf
```

### Step 4: Run Installer

```bash
./scripts/install.sh
```

The installer will:
1. Build container images for all services
2. Generate Kubernetes manifests from your configuration
3. Deploy services to K3s
4. Wait for all pods to be ready

### Step 5: Verify Installation

```bash
# Check all pods are running
kubectl get pods

# Expected output:
# NAME                                READY   STATUS    RESTARTS   AGE
# llama-server-xxx                    1/1     Running   0          2m
# qdrant-xxx                          1/1     Running   0          2m
# embedding-service-xxx               1/1     Running   0          2m
# rag-api-xxx                         1/1     Running   0          2m
# api-portal-xxx                      1/1     Running   0          2m
# redis-xxx                           1/1     Running   0          2m
# task-worker-xxx                     1/1     Running   0          2m
# sandbox-xxx                         1/1     Running   0          2m
# dashboard-xxx                       1/1     Running   0          2m
```

Test the API:
```bash
# Test LLM health
curl http://localhost:8000/health

# Test embedding service
curl http://localhost:8080/health

# Test RAG API
curl http://localhost:8001/health
```

## Post-Installation

### Access Web UI

The API Portal is accessible at:
```
http://your-server-ip:3000
```

1. Create an account (first user is auto-promoted to admin)
2. Generate an API key
3. Use the connection string shown in dashboard

### Configure Client

Connection string format:
```
http://your-server-ip:8001|sk-your-api-key-here
```

### Enable HTTPS (Recommended)

For production use, configure HTTPS via Cloudflare Tunnel or reverse proxy.

## Updating

```bash
# Pull latest changes
cd atlas
git pull

# Re-run installer
./scripts/install.sh
```

## Uninstalling

```bash
# Delete all ATLAS deployments
kubectl delete -f manifests/
kubectl delete -f atlas/manifests/

# Remove container images
podman rmi localhost/llama-server:latest
podman rmi localhost/rag-api:latest
podman rmi localhost/api-portal:latest
podman rmi localhost/task-worker:latest
podman rmi localhost/sandbox:latest
podman rmi localhost/dashboard:latest

# Remove data (optional - destructive!)
rm -rf /data/projects /data/qdrant
```

## Next Steps

- [Architecture](ARCHITECTURE.md) - Understand system design
- [Configuration](CONFIGURATION.md) - All configuration options
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
