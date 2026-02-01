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

### NVIDIA Container Toolkit

The container toolkit enables GPU access from containers:

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

### K3s

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

### Configure K3s Containerd for NVIDIA GPU

**This is critical.** K3s uses its own bundled containerd, and the NVIDIA device plugin
cannot access the GPU without proper containerd configuration. The `nvidia-ctk runtime configure`
command does NOT work with K3s's containerd - you must create a custom template.

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

### NVIDIA Device Plugin

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

### Verify GPU Access from Containers

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
