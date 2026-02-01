# ATLAS Configuration Reference

This document describes all configuration options in `atlas.conf`.

## Quick Start

```bash
# Copy example configuration
cp atlas.conf.example atlas.conf

# Edit configuration
vim atlas.conf

# Apply changes
./scripts/install.sh
```

## Configuration Categories

- [Network Configuration](#network-configuration)
- [Storage Paths](#storage-paths)
- [Persistent Volume Sizes](#persistent-volume-sizes)
- [Model Configuration](#model-configuration)
- [Resource Limits](#resource-limits)
- [Authentication & Security](#authentication--security)
- [Feature Flags](#feature-flags)
- [Timeouts](#timeouts)
- [RAG Configuration](#rag-configuration)
- [Training Configuration](#training-configuration)
- [Ralph Loop Configuration](#ralph-loop-configuration)
- [Logging](#logging)
- [Advanced](#advanced)

---

## Network Configuration

### ATLAS_NODE_IP
Node IP address for external access.

| Default | Type | Required |
|---------|------|----------|
| `auto` | string | No |

Set to `auto` to auto-detect, or specify an IP address manually.

### ATLAS_NAMESPACE
Kubernetes namespace for all ATLAS services.

| Default | Type | Required |
|---------|------|----------|
| `atlas` | string | No |

### External NodePorts

How services are accessed from outside the cluster.

| Variable | Default | Service |
|----------|---------|---------|
| `ATLAS_API_PORTAL_NODEPORT` | 30000 | API Portal (Web UI) |
| `ATLAS_LLM_PROXY_NODEPORT` | 30080 | LLM Proxy |
| `ATLAS_RAG_API_NODEPORT` | 31144 | RAG API |
| `ATLAS_DASHBOARD_NODEPORT` | 30001 | Task Dashboard |
| `ATLAS_LLAMA_NODEPORT` | 32735 | llama-server |
| `ATLAS_QDRANT_NODEPORT` | 30633 | Qdrant HTTP |
| `ATLAS_QDRANT_GRPC_NODEPORT` | 30634 | Qdrant gRPC |
| `ATLAS_EMBEDDING_NODEPORT` | 30808 | Embedding Service |
| `ATLAS_SANDBOX_NODEPORT` | 30820 | Sandbox |

### Internal Service Ports

Ports used for inter-service communication. Usually don't need to change.

| Variable | Default | Service |
|----------|---------|---------|
| `ATLAS_REDIS_PORT` | 6379 | Redis |
| `ATLAS_QDRANT_PORT` | 6333 | Qdrant HTTP |
| `ATLAS_QDRANT_GRPC_PORT` | 6334 | Qdrant gRPC |
| `ATLAS_EMBEDDING_PORT` | 8080 | Embedding |
| `ATLAS_LLAMA_PORT` | 8000 | llama-server |
| `ATLAS_RAG_API_PORT` | 8001 | RAG API |
| `ATLAS_API_PORTAL_PORT` | 3000 | API Portal |
| `ATLAS_SANDBOX_PORT` | 8020 | Sandbox |
| `ATLAS_DASHBOARD_PORT` | 3001 | Dashboard |
| `ATLAS_LLM_PROXY_PORT` | 8000 | LLM Proxy |

---

## Storage Paths

### ATLAS_MODELS_DIR
Directory containing GGUF model files.

| Default | Type | Required |
|---------|------|----------|
| `/root/models` | path | Yes |

### ATLAS_DATA_DIR
Base directory for persistent data.

| Default | Type | Required |
|---------|------|----------|
| `/root/data` | path | No |

### ATLAS_TRAINING_DIR
Directory for training data exports.

| Default | Type | Required |
|---------|------|----------|
| `/root/data/training` | path | No |

### ATLAS_LORA_DIR
Directory for LoRA adapters.

| Default | Type | Required |
|---------|------|----------|
| `/root/models/lora` | path | No |

### ATLAS_PROJECTS_DIR
Directory for RAG project storage.

| Default | Type | Required |
|---------|------|----------|
| `/root/data/projects` | path | No |

---

## Persistent Volume Sizes

| Variable | Default | Purpose |
|----------|---------|---------|
| `ATLAS_PVC_QDRANT_SIZE` | 50Gi | Vector database storage |
| `ATLAS_PVC_REDIS_SIZE` | 5Gi | Task queue and metrics |
| `ATLAS_PVC_PROJECTS_SIZE` | 20Gi | RAG project files |
| `ATLAS_PVC_API_PORTAL_SIZE` | 5Gi | User database |

---

## Model Configuration

### ATLAS_MAIN_MODEL
Filename of the main GGUF model.

| Default | Type | Required |
|---------|------|----------|
| `Qwen3-14B-Q4_K_M.gguf` | string | Yes |

Must exist in `ATLAS_MODELS_DIR`.

### ATLAS_DRAFT_MODEL
Filename for speculative decoding draft model.

| Default | Type | Required |
|---------|------|----------|
| `Qwen3-1.5B-Q4_K_M.gguf` | string | No |

Leave empty to disable speculative decoding.

### ATLAS_CONTEXT_LENGTH
Maximum context window size in tokens.

| Default | Type | Range |
|---------|------|-------|
| 16384 | integer | 512-131072 |

### ATLAS_GPU_LAYERS
Number of model layers to offload to GPU.

| Default | Type | Range |
|---------|------|-------|
| 99 | integer | 0-999 |

Use 99 to offload all layers.

### ATLAS_PARALLEL_SLOTS
Number of parallel inference slots.

| Default | Type | Range |
|---------|------|-------|
| 1 | integer | 1-8 |

Increase if you have extra VRAM.

### ATLAS_FLASH_ATTENTION
Enable flash attention for better performance.

| Default | Type |
|---------|------|
| true | boolean |

---

## Resource Limits

### llama-server Resources

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_LLAMA_GPU_MEMORY` | 14Gi | GPU memory limit |
| `ATLAS_LLAMA_CPU_LIMIT` | 4 | CPU cores limit |
| `ATLAS_LLAMA_CPU_REQUEST` | 2 | CPU cores request |
| `ATLAS_LLAMA_MEMORY_LIMIT` | 16Gi | RAM limit |
| `ATLAS_LLAMA_MEMORY_REQUEST` | 8Gi | RAM request |

### Other Services

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_SERVICE_CPU_LIMIT` | 2 | CPU cores limit |
| `ATLAS_SERVICE_CPU_REQUEST` | 0.5 | CPU cores request |
| `ATLAS_SERVICE_MEMORY_LIMIT` | 2Gi | RAM limit |
| `ATLAS_SERVICE_MEMORY_REQUEST` | 512Mi | RAM request |

---

## Authentication & Security

### ATLAS_JWT_SECRET
Secret key for JWT signing.

| Default | Type | Required |
|---------|------|----------|
| `auto` | string | No |

Set to `auto` to generate a random secret at install time.

**Important**: Change this in production!

### ATLAS_JWT_EXPIRY_HOURS
JWT token expiration time.

| Default | Type | Range |
|---------|------|-------|
| 24 | integer | 1-168 |

### ATLAS_ADMIN_EMAIL
Email of the first admin user.

| Default | Type | Required |
|---------|------|----------|
| (empty) | string | No |

Leave empty to auto-promote the first registered user to admin.

### ATLAS_DEFAULT_RATE_LIMIT
Default rate limit for new API keys (requests per minute).

| Default | Type | Range |
|---------|------|-------|
| 1000 | integer | 1-10000 |

### ATLAS_KEY_HASH_ALGORITHM
Algorithm for hashing API keys.

| Default | Type | Options |
|---------|------|---------|
| sha256 | string | sha256, sha512 |

---

## Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_ENABLE_SPECULATIVE` | true | Enable speculative decoding |
| `ATLAS_ENABLE_TRAINING` | true | Enable nightly LoRA training |
| `ATLAS_ENABLE_RAG` | true | Enable RAG codebase context |
| `ATLAS_ENABLE_PROVENANCE` | true | Enable provenance tracking |
| `ATLAS_ENABLE_DASHBOARD` | true | Enable real-time dashboard |

---

## Timeouts

All values in seconds.

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_LLM_TIMEOUT` | 120 | LLM inference timeout |
| `ATLAS_SANDBOX_TIMEOUT` | 60 | Sandbox execution timeout |
| `ATLAS_TASK_TIMEOUT` | 300 | Full task timeout |
| `ATLAS_EMBEDDING_TIMEOUT` | 30 | Embedding generation timeout |
| `ATLAS_HEALTH_CHECK_TIMEOUT` | 10 | Health check timeout |

---

## RAG Configuration

### ATLAS_RAG_CONTEXT_BUDGET
Maximum tokens for RAG context.

| Default | Type | Range |
|---------|------|-------|
| 8000 | integer | 1000-32000 |

### ATLAS_RAG_TOP_K
Number of chunks to retrieve per query.

| Default | Type | Range |
|---------|------|-------|
| 20 | integer | 5-100 |

### ATLAS_RAG_CHUNK_SIZE
Lines per code chunk.

| Default | Type | Range |
|---------|------|-------|
| 100 | integer | 20-500 |

### ATLAS_RAG_CHUNK_OVERLAP
Overlap lines between chunks.

| Default | Type | Range |
|---------|------|-------|
| 20 | integer | 0-50 |

### ATLAS_RAG_MAX_FILES
Maximum files per project.

| Default | Type | Range |
|---------|------|-------|
| 10000 | integer | 100-100000 |

---

## Training Configuration

### ATLAS_TRAINING_MIN_RATING
Minimum task rating to include in training data.

| Default | Type | Range |
|---------|------|-------|
| 4 | integer | 1-5 |

### ATLAS_TRAINING_VALIDATION_THRESHOLD
Minimum validation pass rate (percentage).

| Default | Type | Range |
|---------|------|-------|
| 66 | integer | 0-100 |

### ATLAS_LORA_RANK
LoRA adapter rank.

| Default | Type | Range |
|---------|------|-------|
| 8 | integer | 1-64 |

### ATLAS_LORA_ALPHA
LoRA alpha scaling factor.

| Default | Type | Range |
|---------|------|-------|
| 16 | integer | 1-128 |

### ATLAS_TRAINING_SCHEDULE
Cron schedule for nightly training.

| Default | Type |
|---------|------|
| `0 2 * * *` | cron |

Default runs at 2:00 AM daily.

---

## Ralph Loop Configuration

### ATLAS_RALPH_MAX_RETRIES
Maximum retry attempts.

| Default | Type | Range |
|---------|------|-------|
| 5 | integer | 1-10 |

### ATLAS_RALPH_BASE_TEMP
Initial temperature.

| Default | Type | Range |
|---------|------|-------|
| 0.7 | float | 0.0-2.0 |

### ATLAS_RALPH_TEMP_INCREMENT
Temperature increase per retry.

| Default | Type | Range |
|---------|------|-------|
| 0.1 | float | 0.0-0.5 |

### ATLAS_RALPH_MAX_TEMP
Maximum temperature.

| Default | Type | Range |
|---------|------|-------|
| 1.2 | float | 0.5-2.0 |

---

## Logging

### ATLAS_LOG_LEVEL
Log verbosity level.

| Default | Type | Options |
|---------|------|---------|
| INFO | string | DEBUG, INFO, WARNING, ERROR |

### ATLAS_LOG_REQUESTS
Enable request logging.

| Default | Type |
|---------|------|
| true | boolean |

---

## Advanced

### ATLAS_EXTERNAL_URL
External URL for reverse proxy/ingress.

| Default | Type | Required |
|---------|------|----------|
| (empty) | string | No |

### ATLAS_API_EXTERNAL_URL
External URL for API endpoint.

| Default | Type | Required |
|---------|------|----------|
| (empty) | string | No |

### ATLAS_REGISTRY
Container registry for pre-built images.

| Default | Type |
|---------|------|
| localhost | string |

### ATLAS_IMAGE_TAG
Image tag for containers.

| Default | Type |
|---------|------|
| latest | string |

### ATLAS_KUBECONFIG
Path to kubeconfig file.

| Default | Type |
|---------|------|
| /etc/rancher/k3s/k3s.yaml | path |

---

## Example Configurations

### Minimal Configuration

```bash
ATLAS_MODELS_DIR="/home/user/models"
ATLAS_MAIN_MODEL="llama-7b.gguf"
```

### Production Configuration

```bash
# Storage
ATLAS_MODELS_DIR="/data/models"
ATLAS_DATA_DIR="/data/atlas"

# Model
ATLAS_MAIN_MODEL="Qwen3-14B-Q4_K_M.gguf"
ATLAS_DRAFT_MODEL="Qwen3-1.5B-Q4_K_M.gguf"
ATLAS_CONTEXT_LENGTH=32768
ATLAS_GPU_LAYERS=99

# Security
ATLAS_JWT_SECRET="your-secure-secret-here"
ATLAS_JWT_EXPIRY_HOURS=8
ATLAS_DEFAULT_RATE_LIMIT=100

# Performance
ATLAS_PARALLEL_SLOTS=2
ATLAS_FLASH_ATTENTION=true

# RAG
ATLAS_RAG_CONTEXT_BUDGET=16000
ATLAS_RAG_TOP_K=30

# Logging
ATLAS_LOG_LEVEL="WARNING"
```

### Low-VRAM Configuration (8GB)

```bash
ATLAS_MAIN_MODEL="llama-7b-q4.gguf"
ATLAS_DRAFT_MODEL=""  # Disable speculative
ATLAS_ENABLE_SPECULATIVE=false
ATLAS_CONTEXT_LENGTH=8192
ATLAS_GPU_LAYERS=40  # Partial offload
ATLAS_PARALLEL_SLOTS=1
```
