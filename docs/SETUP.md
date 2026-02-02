# ATLAS Setup Guide

This guide covers installation and configuration of ATLAS on a fresh system.

## Prerequisites

### Operating System

- RHEL 9, Rocky Linux 9, or compatible
- Ubuntu 22.04+ also supported
- 64-bit x86_64 architecture

### Hardware Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| GPU | 8GB VRAM | NVIDIA only, 16GB+ for larger models |
| RAM | 12GB | 16GB+ for comfortable headroom |
| Storage | 50GB SSD | Models are 8-15GB each |
| CPU | 4 cores | Embedding service is CPU-bound |

#### Tested Configurations

- RYZEN 5 2600 (4 vCPU), RTX 5060 Ti (16GB), 12GB DDR4 RAM, 150GB SSD

### NVIDIA Drivers (Required)

ATLAS requires NVIDIA GPU drivers. This is the only prerequisite you **must** install manually.

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# If not installed (RHEL/Rocky):
sudo dnf install -y nvidia-driver nvidia-driver-cuda

# Reboot after installation
sudo reboot
```

Once `nvidia-smi` shows your GPU, you're ready to proceed.

---

## Installation Options

After installing NVIDIA drivers, you have two options:

| Option | Best For | What It Does |
|--------|----------|--------------|
| **Quick Install** | Most users | Automated script handles everything |
| **Manual Install** | Advanced users | Full control over each component |

---

## Option 1: Quick Install (Recommended)

The install script automates K3s, GPU Operator, and all ATLAS services.

> **Note:** The install script is safe to run even if you already have K3s or GPU configured manually. It detects existing setups and skips components that are already working.

### Step 1: Clone Repository

```bash
git clone https://github.com/itigges22/atlas.git
cd atlas
```

### Step 2: Configure

```bash
# Copy example configuration
cp atlas.conf.example atlas.conf

# Edit configuration
vim atlas.conf
```

Key settings to review:

```bash
# Where your models are stored (directory path)
ATLAS_MODELS_DIR="/home/yourusername/models"

# Model filename (must exist in ATLAS_MODELS_DIR)
ATLAS_MAIN_MODEL="Qwen3-14B-Q4_K_M.gguf"

# GPU layers to offload (99 = all layers to GPU)
ATLAS_GPU_LAYERS=99

# Context window size (tokens)
ATLAS_CONTEXT_LENGTH=16384
```

See [Configuration Reference](#configuration-reference) for all options.

### Step 3: Download Models

```bash
# Create models directory
mkdir -p ~/models

# Option A: Use the download script (auto-selects best quantization for your GPU)
./scripts/download-models.sh

# Option B: Manual download with huggingface-cli
huggingface-cli download Qwen/Qwen3-14B-GGUF qwen3-14b-q4_k_m.gguf --local-dir ~/models
```

### Step 4: Run Installer

```bash
sudo ./scripts/install.sh
```

The installer will:
1. Check prerequisites (NVIDIA driver, GPU memory, system memory)
2. Install K3s (if not already installed)
3. Install NVIDIA GPU Operator via Helm
4. Build container images for all services
5. Deploy services to K3s
6. Wait for all pods to be ready

> **First Build Time:** The initial build takes **1-2 hours** because it downloads large CUDA images (~8GB) and compiles llama.cpp from source. Subsequent builds are much faster since layers are cached.

### Step 5: Verify Installation

```bash
./scripts/verify-install.sh

# Or manually check pods
kubectl get pods -n atlas
```

Expected output:
```
NAME                                READY   STATUS    RESTARTS   AGE
llama-server-xxx                    1/1     Running   0          2m
qdrant-xxx                          1/1     Running   0          2m
embedding-service-xxx               1/1     Running   0          2m
rag-api-xxx                         1/1     Running   0          2m
api-portal-xxx                      1/1     Running   0          2m
redis-xxx                           1/1     Running   0          2m
task-worker-xxx                     1/1     Running   0          2m
sandbox-xxx                         1/1     Running   0          2m
dashboard-xxx                       1/1     Running   0          2m
```

---

## Option 2: Manual Install

Use this if you want full control over K3s and GPU configuration, or if the automated installer doesn't work for your environment.

### Step 1: Install Container Runtime

```bash
# Install Podman (RHEL/Rocky)
sudo dnf install -y podman

# Verify installation
podman --version
```

### Step 2: Install NVIDIA Container Toolkit

```bash
# Add NVIDIA container toolkit repo (RHEL/Rocky)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install the toolkit
sudo dnf install -y nvidia-container-toolkit

# Verify installation
nvidia-ctk --version
ls -la /usr/bin/nvidia-container-runtime
```

### Step 3: Install K3s

```bash
# Install K3s
curl -sfL https://get.k3s.io | sh -

# Configure kubectl
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
chmod 600 ~/.kube/config

# Set the env var (add to ~/.bashrc for persistence)
export KUBECONFIG=~/.kube/config

# Verify
kubectl get nodes
```

### Step 4: Configure K3s Containerd for NVIDIA GPU

**This is critical.** K3s uses its own bundled containerd, and the NVIDIA device plugin cannot access the GPU without proper containerd configuration. The `nvidia-ctk runtime configure` command does NOT work with K3s's containerd - you must create a custom template.

K3s uses containerd config **version 3** format. Create the template file:

```bash
sudo tee /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl << 'EOF'
version = 3

[plugins.'io.containerd.cri.v1.images']
  snapshotter = "{{ .NodeConfig.AgentConfig.Snapshotter }}"

[plugins.'io.containerd.cri.v1.runtime']
  device_ownership_from_security_context = true
{{ if .NodeConfig.AgentConfig.Systemd }}
  systemd_cgroup = true
{{ end }}
{{ if .PrivateRegistryConfig }}
{{ if .PrivateRegistryConfig.Mirrors }}
[plugins.'io.containerd.cri.v1.images'.registry]
{{ range $k, $v := .PrivateRegistryConfig.Mirrors }}
[plugins.'io.containerd.cri.v1.images'.registry.mirrors."{{ $k }}"]
  endpoint = [{{ range $i, $j := $v.Endpoints }}{{ if $i }}, {{ end }}{{ printf "%q" . }}{{ end }}]
{{ if $v.Rewrites }}
[plugins.'io.containerd.cri.v1.images'.registry.mirrors."{{ $k }}".rewrite]
{{ range $pattern, $replace := $v.Rewrites }}
  "{{ $pattern }}" = "{{ $replace }}"
{{ end }}
{{ end }}
{{ end }}
{{ end }}
{{ range $k, $v := .PrivateRegistryConfig.Configs }}
{{ if $v.Auth }}
[plugins.'io.containerd.cri.v1.images'.registry.configs."{{ $k }}".auth]
  {{ if $v.Auth.Username }}username = {{ printf "%q" $v.Auth.Username }}{{ end }}
  {{ if $v.Auth.Password }}password = {{ printf "%q" $v.Auth.Password }}{{ end }}
  {{ if $v.Auth.Auth }}auth = {{ printf "%q" $v.Auth.Auth }}{{ end }}
  {{ if $v.Auth.IdentityToken }}identity_token = {{ printf "%q" $v.Auth.IdentityToken }}{{ end }}
{{ end }}
{{ if $v.TLS }}
[plugins.'io.containerd.cri.v1.images'.registry.configs."{{ $k }}".tls]
  {{ if $v.TLS.CAFile }}ca_file = "{{ $v.TLS.CAFile }}"{{ end }}
  {{ if $v.TLS.CertFile }}cert_file = "{{ $v.TLS.CertFile }}"{{ end }}
  {{ if $v.TLS.KeyFile }}key_file = "{{ $v.TLS.KeyFile }}"{{ end }}
  {{ if $v.TLS.InsecureSkipVerify }}insecure_skip_verify = true{{ end }}
{{ end }}
{{ end }}
{{ end }}

[plugins.'io.containerd.cri.v1.runtime'.cni]
  bin_dir = "{{ .NodeConfig.AgentConfig.CNIBinDir }}"
  conf_dir = "{{ .NodeConfig.AgentConfig.CNIConfDir }}"

[plugins.'io.containerd.cri.v1.runtime'.containerd]
  default_runtime_name = "nvidia"

[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'runc']
  runtime_type = "io.containerd.runc.v2"

[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'runc'.options]
  SystemdCgroup = {{ .NodeConfig.AgentConfig.Systemd }}

[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia']
  runtime_type = "io.containerd.runc.v2"

[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia'.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
  SystemdCgroup = {{ .NodeConfig.AgentConfig.Systemd }}
EOF
```

**Important notes:**
- The template uses Go templating to include K3s's required variables (CNI paths, etc.)
- The `default_runtime_name = "nvidia"` sets NVIDIA as the default container runtime
- Without the CNI configuration, the node will go NotReady
- The template path must be `/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl`

Restart K3s to apply the configuration:

```bash
sudo systemctl restart k3s

# Wait for node to be Ready
kubectl get nodes -w
```

### Step 5: Install NVIDIA Device Plugin

Once containerd is configured, deploy the device plugin:

```bash
# Install NVIDIA device plugin for Kubernetes
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml

# Wait for the plugin pod to be ready
kubectl -n kube-system wait --for=condition=Ready pod -l name=nvidia-device-plugin-ds --timeout=120s

# Verify plugin can see the GPU (should show "found NVML library")
kubectl logs -n kube-system -l name=nvidia-device-plugin-ds | grep -i nvml

# Verify GPU is allocatable (should return "1" or more)
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
```

### Step 6: Verify GPU Access from Containers

Before proceeding with ATLAS installation, verify end-to-end GPU access:

```bash
# Create a test pod
cat <<'TESTEOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: Never
  containers:
  - name: gpu-test
    image: nvcr.io/nvidia/cuda:12.6.0-base-ubi9
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
TESTEOF

# Wait for completion and check output
sleep 30
kubectl logs gpu-test

# You should see nvidia-smi output showing your GPU
# Clean up
kubectl delete pod gpu-test
```

If `nvidia-smi` runs successfully inside the container, your GPU setup is complete.

### Step 7: Clone and Configure ATLAS

```bash
git clone https://github.com/yourusername/atlas.git
cd atlas

# Copy and edit configuration
cp atlas.conf.example atlas.conf
vim atlas.conf
```

### Step 8: Download Models

```bash
# Create models directory (should match ATLAS_MODELS_DIR in atlas.conf)
mkdir -p ~/models

# Download models (auto-selects best quantization for your GPU)
./scripts/download-models.sh
```

### Step 9: Build and Deploy

```bash
# Build container images (takes 1-2 hours on first run)
./scripts/build-containers.sh

# Create namespace
kubectl create namespace atlas

# Process manifest templates with your config values
# This substitutes paths from atlas.conf into the manifests
source scripts/lib/config.sh
export ${!ATLAS_@}
for tmpl in templates/*.yaml.tmpl; do
    envsubst < "$tmpl" > "manifests/$(basename "$tmpl" .tmpl)"
done

# Deploy all services to atlas namespace
kubectl apply -n atlas -f manifests/

# Verify pods are starting
kubectl get pods -n atlas -w
```

> **First Build Time:** The initial build takes **1-2 hours** because it downloads large CUDA images (~8GB) and compiles llama.cpp from source. Subsequent builds are much faster since layers are cached.

Wait for all pods to show `Running` status (this may take a few minutes on first deploy as images are pulled).

### Step 10: Verify Installation

```bash
./scripts/verify-install.sh
```

---

## Configuration Reference

All configuration is in `atlas.conf`. Key settings:

### Storage Paths

```bash
# Where models are stored (directory)
ATLAS_MODELS_DIR="/home/yourusername/models"

# Where persistent data is stored
ATLAS_DATA_DIR="/home/yourusername/data"

# Training data directory
ATLAS_TRAINING_DIR="/home/yourusername/data/training"

# LoRA adapters directory
ATLAS_LORA_DIR="/home/yourusername/models/lora"
```

### Model Configuration

```bash
# Main model filename (must exist in ATLAS_MODELS_DIR)
ATLAS_MAIN_MODEL="Qwen3-14B-Q4_K_M.gguf"

# Draft model for speculative decoding (leave empty to disable)
ATLAS_DRAFT_MODEL="Qwen3-0.6B-Q8_0.gguf"

# Context window size (tokens)
ATLAS_CONTEXT_LENGTH=16384

# GPU layers to offload (99 = all layers)
ATLAS_GPU_LAYERS=99

# Parallel inference slots (increase if you have more VRAM)
ATLAS_PARALLEL_SLOTS=1

# Flash attention (recommended for better performance)
ATLAS_FLASH_ATTENTION=true
```

### Network Configuration

```bash
# Kubernetes namespace for all ATLAS services
ATLAS_NAMESPACE="atlas"

# External NodePorts (how you access services from outside the cluster)
ATLAS_API_PORTAL_NODEPORT=30000
ATLAS_LLM_PROXY_NODEPORT=30080
ATLAS_RAG_API_NODEPORT=31144
ATLAS_DASHBOARD_NODEPORT=30001
```

See `atlas.conf` for all available options.

---

## Available Scripts

| Script | Purpose |
|--------|---------|
| `scripts/install.sh` | Full automated installation (K3s + GPU Operator + ATLAS) |
| `scripts/download-models.sh` | Download recommended models for your GPU |
| `scripts/build-containers.sh` | Build all container images |
| `scripts/generate-manifests.sh` | Generate K8s manifests from atlas.conf |
| `scripts/verify-install.sh` | Verify installation health |
| `scripts/uninstall.sh` | Remove ATLAS from cluster |

---

## Post-Installation

### Access Web UI

The API Portal is accessible at:
```
http://your-server-ip:30000
```

1. Create an account (first user is auto-promoted to admin)
2. Generate an API key
3. Use the connection string shown in dashboard

### Configure Client

Connection string format:
```
http://your-server-ip:31144|sk-your-api-key-here
```

### Enable HTTPS (Recommended)

For production use, configure HTTPS via Cloudflare Tunnel or reverse proxy.

---

## Updating

```bash
cd atlas
git pull

# Re-run installer (will rebuild and redeploy)
sudo ./scripts/install.sh
```

---

## Uninstalling

```bash
# Use the uninstall script
./scripts/uninstall.sh

# Or manually:
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

---

## Next Steps

- [Architecture](ARCHITECTURE.md) - Understand system design
- [Configuration](CONFIGURATION.md) - All configuration options
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
