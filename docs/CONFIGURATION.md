# ATLAS Configuration Reference

Configuration via environment variables and `atlas.conf`. All settings have sensible defaults.

---

## Docker Compose (.env)

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_MODELS_DIR` | `./models` | Path to GGUF model weights on host |
| `ATLAS_MODEL_FILE` | `Qwen3.5-9B-Q6_K.gguf` | Model filename |
| `ATLAS_MODEL_NAME` | `Qwen3.5-9B-Q6_K` | Model name for API responses |
| `ATLAS_CTX_SIZE` | `32768` | Context window size (tokens) |
| `ATLAS_LLAMA_PORT` | `8080` | llama-server port |
| `ATLAS_LENS_PORT` | `8099` | Geometric Lens port |
| `ATLAS_V3_PORT` | `8070` | V3 pipeline service port |
| `ATLAS_SANDBOX_PORT` | `30820` | Sandbox service port |
| `ATLAS_PROXY_PORT` | `8090` | Proxy port (Aider connects here) |

---

## Proxy (atlas-proxy)

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_PROXY_PORT` | `8090` | Port to listen on |
| `ATLAS_INFERENCE_URL` | `http://localhost:8080` | llama-server endpoint |
| `ATLAS_LLAMA_URL` | `http://localhost:8080` | llama-server for grammar-constrained calls |
| `ATLAS_LENS_URL` | `http://localhost:8099` | Geometric Lens scoring endpoint |
| `ATLAS_SANDBOX_URL` | `http://localhost:30820` | Sandbox code execution endpoint |
| `ATLAS_V3_URL` | `http://localhost:8070` | V3 pipeline service endpoint |
| `ATLAS_AGENT_LOOP` | `0` | Set to `1` to enable tool-call agent loop |
| `ATLAS_MODEL_NAME` | `Qwen3.5-9B-Q6_K` | Model name for API calls |

---

## Geometric Lens

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_URL` | `http://localhost:8080` | llama-server for embeddings |
| `LLAMA_EMBED_URL` | `http://localhost:8080` | Embedding endpoint |
| `GEOMETRIC_LENS_ENABLED` | `false` | Enable C(x)/G(x) scoring |
| `PROJECT_DATA_DIR` | `/data/projects` | Project index storage |

---

## V3 Pipeline Service

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_INFERENCE_URL` | `http://localhost:8080` | llama-server endpoint |
| `ATLAS_LENS_URL` | `http://localhost:8099` | Geometric Lens endpoint |
| `ATLAS_SANDBOX_URL` | `http://localhost:30820` | Sandbox endpoint |
| `ATLAS_MODEL_NAME` | `Qwen3.5-9B-Q6_K` | Model name |
| `ATLAS_V3_PORT` | `8070` | Port to listen on |

---

## llama-server

Configured via command-line flags (see `inference/entrypoint-v3.1-9b.sh`):

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | Path to GGUF model file |
| `--ctx-size` | `32768` | Context window (tokens) |
| `--n-gpu-layers` | `99` | GPU layers (99 = all) |
| `--host` | `0.0.0.0` | Listen address |
| `--port` | `8080` | Listen port |
| `--no-mmap` | — | Disable mmap (recommended for stability) |

---

## atlas.conf

Template at `atlas.conf.example`. Used by K3s deployment scripts. Key sections:

```bash
# Model configuration
ATLAS_MODELS_DIR="$HOME/models"
ATLAS_MODEL_NAME="Qwen3.5-9B-Q6_K"

# Service ports
ATLAS_LLAMA_NODEPORT=32735
ATLAS_LENS_NODEPORT=31144
ATLAS_SANDBOX_NODEPORT=30820

# Geometric Lens
ATLAS_ENABLE_LENS=true
ATLAS_LENS_CONTEXT_BUDGET=6000

# K3s namespace
ATLAS_NAMESPACE=atlas
```

See `atlas.conf.example` for the full documented template.

---

## Agent Loop Configuration

These are internal to the proxy, not configurable via env vars:

| Setting | Value | Description |
|---------|-------|-------------|
| Max turns (T0) | 5 | Conversational messages |
| Max turns (T1) | 30 | Simple coding tasks |
| Max turns (T2) | 30 | Feature files with V3 |
| Max turns (T3) | 60 | Complex multi-file projects |
| Exploration budget | 4 reads | Consecutive reads before nudge |
| Error loop limit | 3 | Consecutive failures before stop |
| T2 threshold | 50 lines + 3 indicators | Minimum for V3 activation |
