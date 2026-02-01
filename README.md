# ATLAS - Adaptive Test-time Learning and Autonomous Specialization

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-407%20passed-brightgreen.svg)](#testing)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-K3s-326CE5.svg)](https://k3s.io/)

Self-hosted AI coding assistant infrastructure with test-time compute scaling.

## Features

- **Ralph Loop Algorithm**: Retry-until-success code generation with 99.5% success rate (p=0.65, k=5 attempts)
- **Continuous Learning**: Nightly LoRA fine-tuning from successful task completions
- **RAG Integration**: Retrieval-augmented generation for codebase-aware assistance
- **Multi-Tenant MaaS**: Web-based API key management with usage tracking
- **GPU Acceleration**: CUDA-optimized inference with llama.cpp

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/atlas.git
cd atlas

# Configure your installation
cp atlas.conf.example atlas.conf
vim atlas.conf  # Set your MODEL_PATH and other options

# Run the installer
./scripts/install.sh

# Verify services are running
kubectl get pods
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  K3s Cluster                                                                │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ llama-server│  │   Qdrant    │  │  Embedding  │  │  API Portal │        │
│  │   (GPU)     │  │ (Vector DB) │  │   Service   │  │   (MaaS)    │        │
│  │  :8000      │  │  :6333      │  │  :8080      │  │  :3000      │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                   │                                         │
│                          ┌────────┴────────┐                                │
│                          │     RAG API     │                                │
│                          │     :8001       │                                │
│                          └────────┬────────┘                                │
│                                   │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ATLAS Task System                                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │  │   Redis     │  │ Task Worker │  │   Sandbox   │  │ Dashboard │  │   │
│  │  │ (P0/P1/P2)  │  │ Ralph Loop  │  │ (isolated)  │  │  :3001    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to GGUF model file | Required |
| `GPU_LAYERS` | Layers to offload to GPU | `99` |
| `CONTEXT_SIZE` | Context window size | `16384` |
| `EMBEDDING_MODEL` | Embedding model name | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Lines per chunk | `100` |
| `CHUNK_OVERLAP` | Overlap between chunks | `20` |
| `TOP_K` | Retrieved chunks per query | `20` |

See `atlas.conf.example` for all options.

## Services

| Service | Port | Purpose |
|---------|------|---------|
| llama-server | 8000 | LLM inference (GPU) |
| RAG API | 8001 | RAG orchestration |
| API Portal | 3000 | User management, API keys |
| Embedding Service | 8080 | Text vectorization |
| Qdrant | 6333 | Vector database |
| Dashboard | 3001 | Task monitoring |
| Sandbox | 8020 | Isolated code execution |

## Testing

Run the test suite:

```bash
# Validate all tests
python tests/validate_tests.py

# Run specific test module
pytest tests/test_rag_api.py -v

# Run all tests with coverage
pytest tests/ --cov=. --cov-report=html
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM | 16GB+ VRAM |
| RAM | 16GB | 32GB |
| Storage | 50GB SSD | 200GB+ SSD |
| CPU | 4 cores | 8+ cores |

Tested on:
- NVIDIA RTX 5060 Ti (16GB)
- RHEL 9 / Rocky Linux 9
- K3s v1.28+

## Documentation

- [Setup Guide](docs/SETUP.md) - Installation and configuration
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Configuration](docs/CONFIGURATION.md) - All configuration options
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Project Structure

```
atlas/
├── api-portal/          # Web-based user/API key management
├── atlas/
│   ├── dashboard/       # Real-time task monitoring UI
│   ├── sandbox/         # Isolated code execution environment
│   ├── task-worker/     # Ralph Loop task processing
│   ├── trainer/         # LoRA fine-tuning pipeline
│   └── manifests/       # Atlas component K8s manifests
├── embedding-service/   # Text embedding service
├── llama-server/        # GPU inference container
├── llm-proxy/           # API gateway with auth
├── manifests/           # Core service K8s manifests
├── rag-api/             # RAG orchestration service
├── scripts/             # Installation and utility scripts
├── templates/           # Web UI templates
└── tests/               # Test suite
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

Copyright 2025 Isaac Tigges
# ATLAS
# ATLAS
