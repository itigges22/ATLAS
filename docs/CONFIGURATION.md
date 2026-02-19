# ATLAS V2 Configuration Reference

All cluster-level configuration lives in `atlas.conf` at the repository root. Service-specific behavior is controlled by environment variables in the Kubernetes deployment manifests.

```bash
cp atlas.conf.example atlas.conf
vim atlas.conf
./scripts/install.sh
```

---

## Table of Contents

- [Network Configuration](#network-configuration)
- [Storage Paths](#storage-paths)
- [Persistent Volume Sizes](#persistent-volume-sizes)
- [Model Configuration](#model-configuration)
- [Resource Limits](#resource-limits)
- [Authentication and Security](#authentication-and-security)
- [Feature Flags](#feature-flags)
- [Timeouts](#timeouts)
- [RAG Configuration](#rag-configuration)
- [Training Configuration](#training-configuration)
- [Ralph Loop Configuration](#ralph-loop-configuration)
- [Logging](#logging)
- [Advanced](#advanced)
- [Deployment Environment Variables](#deployment-environment-variables)
- [Example Configurations](#example-configurations)

---

## Network Configuration

### ATLAS_NODE_IP

Node IP address for external access.

| Default | Type | Required |
|---------|------|----------|
| `auto` | string | No |

Set to `auto` to auto-detect from the default route, or specify a static IP.

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
| `ATLAS_SANDBOX_NODEPORT` | 30820 | Sandbox |

### Internal Service Ports

Ports used for inter-service communication inside the cluster. Normally no reason to change these.

| Variable | Default | Service |
|----------|---------|---------|
| `ATLAS_REDIS_PORT` | 6379 | Redis |
| `ATLAS_LLAMA_PORT` | 8000 | llama-server |
| `ATLAS_RAG_API_PORT` | 8001 | RAG API |
| `ATLAS_API_PORTAL_PORT` | 3000 | API Portal |
| `ATLAS_SANDBOX_PORT` | 8020 | Sandbox |
| `ATLAS_DASHBOARD_PORT` | 3001 | Dashboard |
| `ATLAS_LLM_PROXY_PORT` | 8000 | LLM Proxy |

---

## Storage Paths

### ATLAS_MODELS_DIR

Directory containing GGUF model files. Both the main model and the draft model must be in this directory.

| Default | Type | Required |
|---------|------|----------|
| `/root/models` | path | Yes |

### ATLAS_DATA_DIR

Base directory for persistent data (Redis snapshots, training exports).

| Default | Type | Required |
|---------|------|----------|
| `/root/data` | path | No |

### ATLAS_TRAINING_DIR

Directory for training data exports used by the nightly LoRA training pipeline.

| Default | Type | Required |
|---------|------|----------|
| `/root/data/training` | path | No |

### ATLAS_LORA_DIR

Directory for LoRA adapter files. llama-server mounts this at `/models/lora`.

| Default | Type | Required |
|---------|------|----------|
| `/root/models/lora` | path | No |

### ATLAS_PROJECTS_DIR

Directory for RAG project storage. PageIndex tree indexes, BM25 indexes, and project source files are persisted here.

| Default | Type | Required |
|---------|------|----------|
| `/root/data/projects` | path | No |

---

## Persistent Volume Sizes

| Variable | Default | Purpose |
|----------|---------|---------|
| `ATLAS_PVC_REDIS_SIZE` | 5Gi | Redis data (task queue, Thompson Sampling posteriors) |
| `ATLAS_PVC_PROJECTS_SIZE` | 20Gi | RAG project files and indexes |
| `ATLAS_PVC_API_PORTAL_SIZE` | 5Gi | User database and API keys |

---

## Model Configuration

### ATLAS_MAIN_MODEL

Filename of the main GGUF model. Must exist in `ATLAS_MODELS_DIR`.

| Default | Type | Required |
|---------|------|----------|
| `Qwen3-14B-Q4_K_M.gguf` | string | Yes |

### ATLAS_DRAFT_MODEL

Filename of the speculative decoding draft model. Must exist in `ATLAS_MODELS_DIR`. Leave empty to disable speculative decoding.

| Default | Type | Required |
|---------|------|----------|
| `Qwen3-0.6B-Q8_0.gguf` | string | No |

In V2, this was changed from Qwen3-1.5B-Q4_K_M to Qwen3-0.6B-Q8_0 for lower VRAM overhead.

### ATLAS_CONTEXT_LENGTH

Maximum context window size in tokens. With `--parallel 2`, each slot gets half this value (e.g., 40960 total = 20480 per slot).

| Default | Type | Range |
|---------|------|-------|
| 40960 | integer | 512-131072 |

Changed from 16384 in V1 to 40960 in V2.

### ATLAS_GPU_LAYERS

Number of model layers to offload to the GPU. Set to 99 to offload all layers. Reduce if running out of VRAM.

| Default | Type | Range |
|---------|------|-------|
| 99 | integer | 0-999 |

### ATLAS_PARALLEL_SLOTS

Number of parallel inference slots. Each slot gets `ATLAS_CONTEXT_LENGTH / ATLAS_PARALLEL_SLOTS` tokens of context and requires its own KV cache allocation.

| Default | Type | Range |
|---------|------|-------|
| 2 | integer | 1-8 |

Changed from 1 in V1 to 2 in V2 (for best-of-k pipelining).

### ATLAS_FLASH_ATTENTION

Enable flash attention for reduced memory usage and faster inference.

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
| `ATLAS_LLAMA_MEMORY_LIMIT` | 16Gi | System RAM limit |
| `ATLAS_LLAMA_MEMORY_REQUEST` | 8Gi | System RAM request |

### Other Services

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_SERVICE_CPU_LIMIT` | 2 | CPU cores limit |
| `ATLAS_SERVICE_CPU_REQUEST` | 0.5 | CPU cores request |
| `ATLAS_SERVICE_MEMORY_LIMIT` | 2Gi | RAM limit |
| `ATLAS_SERVICE_MEMORY_REQUEST` | 512Mi | RAM request |

---

## Authentication and Security

### ATLAS_JWT_SECRET

Secret key for JWT token signing. Set to `auto` to generate a random secret at install time.

| Default | Type | Required |
|---------|------|----------|
| `auto` | string | No |

Change this to a fixed value in production.

### ATLAS_JWT_EXPIRY_HOURS

JWT token expiration time in hours.

| Default | Type | Range |
|---------|------|-------|
| 24 | integer | 1-168 |

### ATLAS_ADMIN_EMAIL

Email of the first admin user. Leave empty to auto-promote the first registered user.

| Default | Type | Required |
|---------|------|----------|
| (empty) | string | No |

### ATLAS_DEFAULT_RATE_LIMIT

Default rate limit for new API keys, in requests per minute.

| Default | Type | Range |
|---------|------|-------|
| 1000 | integer | 1-10000 |

### ATLAS_KEY_HASH_ALGORITHM

Algorithm for hashing API keys at rest.

| Default | Type | Options |
|---------|------|---------|
| sha256 | string | sha256, sha512 |

---

## Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_ENABLE_SPECULATIVE` | true | Enable speculative decoding (requires draft model) |
| `ATLAS_ENABLE_TRAINING` | true | Enable nightly LoRA training pipeline |
| `ATLAS_ENABLE_RAG` | true | Enable RAG codebase context |
| `ATLAS_ENABLE_PROVENANCE` | true | Enable provenance tracking (source attribution in responses) |
| `ATLAS_ENABLE_DASHBOARD` | true | Enable real-time task dashboard |

---

## Timeouts

All values in seconds.

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_LLM_TIMEOUT` | 120 | LLM inference request timeout |
| `ATLAS_SANDBOX_TIMEOUT` | 60 | Code sandbox execution timeout |
| `ATLAS_TASK_TIMEOUT` | 300 | Full task processing timeout |
| `ATLAS_HEALTH_CHECK_TIMEOUT` | 10 | Health check probe timeout |

---

## RAG Configuration

### ATLAS_RAG_CONTEXT_BUDGET

Maximum number of tokens allocated for retrieved context in a prompt.

| Default | Type | Range |
|---------|------|-------|
| 8000 | integer | 1000-32000 |

### ATLAS_RAG_TOP_K

Number of code snippets to retrieve per query. In V2 with PageIndex, this controls how many tree nodes are returned by the hybrid retriever (BM25 + tree search).

| Default | Type | Range |
|---------|------|-------|
| 20 | integer | 5-100 |

### ATLAS_RAG_MAX_FILES

Maximum number of files that can be indexed per project.

| Default | Type | Range |
|---------|------|-------|
| 10000 | integer | 100-100000 |

---

## Training Configuration

### ATLAS_TRAINING_MIN_RATING

Minimum task quality rating to include in training data exports.

| Default | Type | Range |
|---------|------|-------|
| 4 | integer | 1-5 |

### ATLAS_TRAINING_VALIDATION_THRESHOLD

Minimum validation pass rate (percentage) for a LoRA adapter to be accepted.

| Default | Type | Range |
|---------|------|-------|
| 66 | integer | 0-100 |

### ATLAS_LORA_RANK

LoRA adapter rank. Higher values increase expressiveness at the cost of more parameters.

| Default | Type | Range |
|---------|------|-------|
| 8 | integer | 1-64 |

### ATLAS_LORA_ALPHA

LoRA alpha scaling factor. Typically set to 2x the rank.

| Default | Type | Range |
|---------|------|-------|
| 16 | integer | 1-128 |

### ATLAS_TRAINING_SCHEDULE

Cron schedule for the nightly training job.

| Default | Type |
|---------|------|
| `0 2 * * *` | cron expression |

Runs at 2:00 AM daily by default.

---

## Ralph Loop Configuration

The Ralph loop controls retry behavior with escalating temperature.

### ATLAS_RALPH_MAX_RETRIES

Maximum number of retry attempts before giving up on a task.

| Default | Type | Range |
|---------|------|-------|
| 5 | integer | 1-10 |

### ATLAS_RALPH_BASE_TEMP

Initial sampling temperature for the first attempt.

| Default | Type | Range |
|---------|------|-------|
| 0.7 | float | 0.0-2.0 |

### ATLAS_RALPH_TEMP_INCREMENT

How much to increase temperature on each retry.

| Default | Type | Range |
|---------|------|-------|
| 0.1 | float | 0.0-0.5 |

### ATLAS_RALPH_MAX_TEMP

Upper bound on sampling temperature regardless of retry count.

| Default | Type | Range |
|---------|------|-------|
| 1.2 | float | 0.5-2.0 |

---

## Logging

### ATLAS_LOG_LEVEL

Log verbosity for all services.

| Default | Type | Options |
|---------|------|---------|
| INFO | string | DEBUG, INFO, WARNING, ERROR |

### ATLAS_LOG_REQUESTS

Log every incoming HTTP request.

| Default | Type |
|---------|------|
| true | boolean |

---

## Advanced

### ATLAS_EXTERNAL_URL

External URL if running behind a reverse proxy or ingress controller.

| Default | Type | Required |
|---------|------|----------|
| (empty) | string | No |

### ATLAS_API_EXTERNAL_URL

External URL specifically for the API endpoint.

| Default | Type | Required |
|---------|------|----------|
| (empty) | string | No |

### ATLAS_REGISTRY

Container registry prefix for image names. Use `localhost` for locally built images.

| Default | Type |
|---------|------|
| localhost | string |

### ATLAS_IMAGE_TAG

Tag applied to all container images.

| Default | Type |
|---------|------|
| latest | string |

### ATLAS_KUBECONFIG

Path to the kubeconfig file used by install and management scripts.

| Default | Type |
|---------|------|
| /etc/rancher/k3s/k3s.yaml | path |

---

## Deployment Environment Variables

These environment variables are set directly in the Kubernetes deployment manifests and control runtime behavior of individual services.

### llama-server

Set in `manifests/llama-deployment.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/Qwen3-14B-Q4_K_M.gguf` | Full path to the main model inside the container |
| `CONTEXT_LENGTH` | `40960` | Context window size in tokens |
| `GPU_LAYERS` | `99` | Number of layers offloaded to GPU |
| `PARALLEL_SLOTS` | `2` | Number of parallel inference slots |
| `DRAFT_MODEL` | `/models/Qwen3-0.6B-Q8_0.gguf` | Full path to draft model; unset to disable speculative decoding |
| `ENABLE_EMBEDDINGS` | `true` | Enable the `/embedding` endpoint for V2 geometric lens |
| `KV_CACHE_TYPE` | `q4_0` | KV cache quantization type (q4_0, q8_0, f16) |
| `CHAT_TEMPLATE` | `Qwen3-custom.jinja` | Jinja2 chat template filename |
| `GGML_CUDA_NO_PINNED` | `0` | Set to 0 to enable pinned host memory for PCIe transfers |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Reduce CUDA context switch overhead |
| `CUDA_MODULE_LOADING` | `LAZY` | Lazy CUDA module loading for faster startup |

### rag-api

Set in `manifests/rag-api-deployment.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_URL` | `http://llama-service:8000` | URL of the llama-server service |
| `REDIS_HOST` | `redis` | Redis hostname for caching and Thompson Sampling posteriors |
| `REDIS_PORT` | `6379` | Redis port |
| `ROUTING_ENABLED` | `true` | Enable the confidence router (Thompson Sampling route selection). When false, all queries use the STANDARD route |
| `GEOMETRIC_LENS_ENABLED` | `true` | Enable geometric lens correction (MLP-based energy field for embedding quality estimation). Requires `--embeddings` on llama-server |
| `CONTEXT_BUDGET` | `8000` | Maximum tokens of retrieved context per query |
| `TOP_K` | `20` | Number of code snippets returned by the hybrid retriever |
| `MAX_FILES` | `10000` | Maximum files per indexed project |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `ENABLE_PROVENANCE` | `true` | Include source file attribution in responses |

---

## Example Configurations

### Minimal Configuration

The smallest viable `atlas.conf` -- everything else uses defaults:

```bash
ATLAS_MODELS_DIR="/home/user/models"
ATLAS_MAIN_MODEL="Qwen3-14B-Q4_K_M.gguf"
```

### Production Configuration

Full V2 production setup with speculative decoding, routing, and geometric lens:

```bash
# Storage
ATLAS_MODELS_DIR="/data/models"
ATLAS_DATA_DIR="/data/atlas"
ATLAS_PROJECTS_DIR="/data/atlas/projects"

# Model
ATLAS_MAIN_MODEL="Qwen3-14B-Q4_K_M.gguf"
ATLAS_DRAFT_MODEL="Qwen3-0.6B-Q8_0.gguf"
ATLAS_CONTEXT_LENGTH=40960
ATLAS_GPU_LAYERS=99
ATLAS_PARALLEL_SLOTS=2
ATLAS_FLASH_ATTENTION=true

# Security
ATLAS_JWT_SECRET="your-secure-secret-here"
ATLAS_JWT_EXPIRY_HOURS=8
ATLAS_DEFAULT_RATE_LIMIT=100

# RAG
ATLAS_RAG_CONTEXT_BUDGET=8000
ATLAS_RAG_TOP_K=20

# Logging
ATLAS_LOG_LEVEL="WARNING"
```

Then in the rag-api deployment manifest, ensure:

```yaml
- name: ROUTING_ENABLED
  value: "true"
- name: GEOMETRIC_LENS_ENABLED
  value: "true"
```

### Low-VRAM Configuration (8GB GPU)

For GPUs with 8GB VRAM, use a smaller model and disable features that consume VRAM:

```bash
ATLAS_MAIN_MODEL="Qwen3-8B-Q4_K_M.gguf"
ATLAS_DRAFT_MODEL=""
ATLAS_ENABLE_SPECULATIVE=false
ATLAS_CONTEXT_LENGTH=16384
ATLAS_GPU_LAYERS=99
ATLAS_PARALLEL_SLOTS=1
ATLAS_FLASH_ATTENTION=true
```

In the rag-api deployment, disable geometric lens (it requires the embedding endpoint, which adds VRAM pressure):

```yaml
- name: GEOMETRIC_LENS_ENABLED
  value: "false"
```

Routing can remain enabled -- it runs on CPU and adds negligible overhead.
