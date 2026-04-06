# A.T.L.A.S.

**Adaptive Test-time Learning and Autonomous Specialization**

![Version](https://img.shields.io/badge/version-V3.0.1-blue)
![LCB](https://img.shields.io/badge/LiveCodeBench-74.6%25_pass%401-green)
![Model](https://img.shields.io/badge/model-Qwen3.5--9B-orange)
![GPU](https://img.shields.io/badge/GPU-RTX_5060_Ti_16GB-red)
![License](https://img.shields.io/badge/license-ATLAS_Source_Available-lightgrey)

ATLAS is a locally-hosted AI coding assistant that wraps a frozen 9B-parameter language model in intelligent infrastructure — structured tool calls, grammar-constrained output, diverse candidate generation, build verification, and energy-based scoring — to produce code quality that exceeds what the base model can achieve alone. It runs entirely on a single consumer GPU with no cloud API dependency.

## Why ATLAS Exists

I'm a business student at Virginia Tech. My background is in marketing, not computer science. I'm a hobbyist who got curious about what's possible when you stop assuming only the biggest players can build meaningful things.

My twin sister was born with Loeys-Dietz syndrome. When we were five, doctors told my parents she would never walk. A year later, she walked into that same doctor's office. She remembered looking back at him and seeing tears in his eyes. She passed away last year on March 29th. But that memory stayed with me. The people who tell you what's impossible are usually just describing the limits of their own experience. Sometimes all it takes is a single moment to realize the barrier was never technical — it was assumption.

ATLAS isn't the destination. It's proof of what we can build.

## Benchmark Results

| Benchmark | Result | Tasks | Model | Notes |
|-----------|--------|-------|-------|-------|
| LiveCodeBench v5 | **74.6% pass@1** | 599 | Qwen3-14B (V3.0) | Frozen model, no fine-tuning |
| V3 Ablation A (baseline) | 54.9% | 599 | Qwen3-14B | No V3 pipeline |
| V3 Ablation B (+Phase 1) | 67.3% | 599 | Qwen3-14B | +PlanSearch, DivSampling, Budget Forcing |
| V3 Ablation D (+Phase 1+3) | 74.6% | 599 | Qwen3-14B | +PR-CoT Repair, Refinement, Derivation Chains |
| CLI Reliability (8-level) | 95.8% | 24 | Qwen3.5-9B (V3.0.1) | Tool-call agent loop, 3 iterations |
| 5-Language Integration | 100% | 5 | Qwen3.5-9B | Python, Rust, Go, C, Shell |

Raw ablation data: [`v3_ablation_results/`](v3_ablation_results/) — per-task pass/fail for all conditions.

Full training data and benchmark traces: [ATLAS Geometric Lens Dataset](https://huggingface.co/datasets/itigges22/ATLAS)

## Architecture

ATLAS uses a two-layer architecture:

```
┌─────────────────────────────────────────────────────┐
│  User runs `atlas` in project directory              │
│                                                      │
│  Aider (TUI) ──► atlas-proxy (agent loop)            │
│                    │                                  │
│    OUTER LAYER:    │  Structured JSON tool calls      │
│    Agent Loop      │  Grammar-constrained output      │
│                    │  8 tools: read, write, edit,     │
│                    │  delete, run, search, list, plan │
│                    │                                  │
│    INNER LAYER:    │  V3 Pipeline (T2/T3 files)       │
│    V3 Pipeline     │  PlanSearch → DivSampling →      │
│                    │  Budget Forcing → Build Verify →  │
│                    │  C(x)/G(x) Score → Best-of-K →   │
│                    │  PR-CoT Repair → Select Winner   │
│                    │                                  │
│    ┌───────────────┼──────────────────────┐          │
│    │ llama-server  │  geometric-lens      │ sandbox  │
│    │ (CUDA, 32K)   │  C(x)/G(x) scoring   │ (code   │
│    │ ~51 tok/s     │  Val AUC 0.9467      │  exec)  │
│    └───────────────┴──────────────────────┘          │
└─────────────────────────────────────────────────────┘
```

**Outer layer** (agent loop): The model emits structured JSON tool calls — `write_file`, `edit_file`, `run_command`, etc. Grammar enforcement via `response_format:json_object` guarantees 100% valid JSON on every output. The proxy executes tools and returns results. The model iterates until done.

**Inner layer** (V3 pipeline): When `write_file` or `edit_file` targets a file with real application logic (50+ lines, T2 classification), the V3 pipeline activates. It generates diverse implementation candidates via PlanSearch and DivSampling, build-verifies each one, scores with C(x)/G(x) energy, selects the best candidate, and repairs failures with PR-CoT.

**Per-file tier classification**: Config files, data files, CSS, and boilerplate (T1) write directly — no pipeline overhead. Feature files with logic (T2) route through V3 for quality enhancement.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/itigges22/ATLAS.git && cd ATLAS

# 2. Download model weights
mkdir -p models
# Download Qwen3.5-9B-Q6_K.gguf into models/

# 3. Configure
cp .env.example .env
# Edit .env: set ATLAS_MODELS_DIR to your models path

# 4. Start the stack
podman-compose up -d   # or: docker compose up -d

# 5. Launch ATLAS
atlas
```

See [docs/SETUP.md](docs/SETUP.md) for detailed setup including bare-metal and K3s deployment.

## Project Structure

```
ATLAS/
├── atlas-proxy/          Go proxy — agent loop, grammar, 8 tools, Aider format
├── geometric-lens/       C(x)/G(x) scoring service (Geometric Lens)
│   ├── geometric_lens/   Cost field, metric tensor, training, EWC, replay buffer
│   └── data/sample/      Training data sample (full dataset on HuggingFace)
├── benchmark/            Benchmark infrastructure
│   └── v3/               V3 pipeline modules (20 components)
├── v3-service/           V3 pipeline HTTP service
├── inference/            llama-server Dockerfiles and entrypoints
├── sandbox/              Isolated code execution service
├── scripts/              Build, deploy, train, validate scripts
├── tests/                Infrastructure and V3 component tests
├── v3_ablation_results/  Published ablation data (conditions A–D)
├── docs/                 Architecture, API, setup, CLI, troubleshooting
├── docker-compose.yml    Full stack orchestration
└── atlas.conf.example    Configuration template
```

## Documentation

| Document | Description |
|----------|-------------|
| [SETUP.md](docs/SETUP.md) | Installation — Docker, bare-metal, K3s |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and design |
| [CLI.md](docs/CLI.md) | CLI usage, streaming output, tips |
| [API.md](docs/API.md) | HTTP API endpoints and formats |
| [MAP.md](docs/MAP.md) | Visual guide to the repo |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Environment variables and config |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and fixes |
| [V3_ABLATION_STUDY.md](docs/V3_ABLATION_STUDY.md) | Ablation methodology and results |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Roadmap

### V3.0 — Complete
Benchmark pipeline: 74.6% LCB pass@1 on frozen Qwen3-14B through iterative constraint generation with sandbox verification.

### V3.0.1 — Complete (Current Release)
CLI tool-call architecture, Docker Compose deployment, V3 pipeline integration, grammar-constrained structured output, per-file tier classification, 95.8% reliability.

### V3.1 — In Progress
- **Planned benchmarks** (not yet run): LiveCodeBench v5 on Qwen3.5-9B with CLI pipeline, GPQA Diamond, SciCode, AA-LCR, AA-Omniscience, Humanity's Last Exam, CritPt
- **Fox optimization**: C-side sampler chain for grammar speed (14→50 tok/s target)
- **Geometric Lens retraining**: Online C(x) recalibration
- **G(x) redesign**: Metric tensor architecture improvements
- **Target**: 80-90% LCB pass@1

## License

[ATLAS Source Available License v1.0](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
