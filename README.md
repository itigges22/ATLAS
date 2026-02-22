![License](https://img.shields.io/badge/license-Source%20Available-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![K8s](https://img.shields.io/badge/platform-K3s%20%7C%20K8s-blue)
![GPU](https://img.shields.io/badge/GPU-RTX%205060%20Ti%2016GB-green)
![Status](https://img.shields.io/badge/status-v2.5-orange)

# ATLAS -- Adaptive Test-time Learning and Autonomous Specialization

ATLAS achieves 36-41% LiveCodeBench pass@1 with a frozen 14B model on a single consumer GPU through intelligent test-time compute allocation.

---

## V2 Results (Verified)

| Benchmark | Score | Tasks | Notes |
|-----------|-------|-------|-------|
| LiveCodeBench v5 | 36-41% pass@1 | 599 evals / ~600 problems | k=3, Geometric Lens selection, 4 epochs (100+200+200+99) |
| GPQA Diamond | 47.0% | 198 | k=5, multiple-choice knowledge reasoning |
| SciCode (sub-problems) | 14.7% | 341 | Cross-domain scientific coding |

**Lens learning curve (LiveCodeBench, k=3):**

| Epoch | Tasks | Pass Rate | First-Pick Accuracy | Energy Gap (pass vs fail) |
|-------|-------|-----------|---------------------|---------------------------|
| 0 (baseline, no Lens) | 100 | 36.0% | n/a | n/a |
| 1 (1st retrain) | 200 | 38.0% | 82.9% | 5.3 |
| 2 (2nd retrain) | 200 | 35.5% | 78.9% | 11.5 |
| 3 (3rd retrain) | 99 | 41.4% | 78.0% | 11.3 |

First-pick accuracy = how often the Lens's lowest-energy candidate actually passes. The energy gap between pass and fail candidates doubled after retraining (5.3 to 11.3), showing the Lens learned to separate passing from failing code. Val AUC reached 0.968 at epoch 3.

**Note**: The V2.5 ablation study found that while C(x) learns real energy separation, this does not translate to statistically significant candidate selection improvement (37.7% vs 37.1% random, within seed variance). Most tasks are all-pass or all-fail across k=3 candidates, so ordering has limited effect. The pass rate improvement across epochs is primarily driven by Best-of-K diversity, not Lens ranking. See [V2.5 Ablation Study](#v25-ablation-study).

**Hardware:** RTX 5060 Ti 16GB VRAM. Total cost: ~$500 GPU.
**Runtime:** 109 tasks/hr aggregate throughput on V2 benchmark.
**Run ID:** `v2_run_20260217_125310`.

All results from a single benchmark run. Not averaged across multiple runs. Variance unknown. LCB "36-41%" reflects epoch 0 to epoch 3 of Geometric Lens retraining on 100-200 task batches, not a confidence interval.

## V2.5 Ablation Study

A systematic ablation (2026-02-21) tested whether the Geometric Lens C(x) energy scoring provides real candidate selection value beyond diversity. **Result: Lens scoring is statistically indistinguishable from random selection** -- energy-sorted candidates achieve 37.7% pass@1 vs 37.1% for random ordering (0.6pp gap within the 3.4pp seed-to-seed variance, mean 36.0% +/- 1.7% across 3 seeds). The Best-of-K diversity benefit (generating 3 candidates at temp=0.6) accounts for nearly all improvement.

The study also discovered that llama.cpp's `--embeddings` flag was silently breaking speculative decoding (forcing n_batch=512, causing 0% draft token acceptance). This led to a two-server sidecar architecture: generation with spec decode (~100 tok/s) on the main server, embeddings via a lightweight nomic-embed-text-v1.5 sidecar (~300 MiB VRAM). C(x) energy does correlate with task difficulty (58.5% vs 18.9% pass rate across energy tiers) and will be repurposed for difficulty-adaptive routing in V3.

Full results: [docs/V2_5_ABLATION_STUDY.md](docs/V2_5_ABLATION_STUDY.md) | Architecture change: [docs/V2_TO_V2_5_MIGRATION.md](docs/V2_TO_V2_5_MIGRATION.md)

## Architecture Overview

```mermaid
flowchart TB
  subgraph Input
    Problem[Coding Problem]
  end

  subgraph Routing["Confidence Router"]
    DE[Difficulty Estimator<br/>Weights: 0.30 / 0.25 / 0.20 / 0.25]
    AK[Adaptive-k Selection<br/>CACHE_HIT k=0 / FAST k=1<br/>STANDARD k=5 / HARD k=20]
  end

  subgraph Generation["Best-of-K Pipeline"]
    LS[Server A: llama-server<br/>Qwen3-14B-Q4_K_M<br/>+ Qwen3-0.6B Draft<br/>Spec decode ON]
    EM[Server B: Embeddings<br/>nomic-embed-text-v1.5<br/>768-dim]
    PC[Pattern Cache<br/>Redis + Ebbinghaus Decay]
  end

  subgraph Evaluation["Candidate Selection"]
    GL[Geometric Lens<br/>C x Cost Field ~0.5M params<br/>G x Metric Tensor ~0.8M params DORMANT]
    SB[Sandbox<br/>Code Execution + Testing]
  end

  subgraph Knowledge["Context Retrieval"]
    PI[PageIndex RAG<br/>Tree Index + LLM Reasoning]
  end

  Problem --> DE
  DE --> AK
  AK --> LS
  PC -.->|strategy hints| LS
  PI -.->|relevant context| LS
  LS -->|k candidates| GL
  GL -->|extract embedding| EM
  GL -->|sorted by energy| SB
  SB -->|result + feedback| PC

  style GL fill:#2d5016,color:#fff
  style LS fill:#1a3a5c,color:#fff
  style EM fill:#1a3a5c,color:#fff
  style DE fill:#5c3a1a,color:#fff
```

Full architecture details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### API Portal

The system includes an API Portal (`api-portal` service, port 3000, NodePort 30000) for multi-user access. It provides user registration and login with JWT authentication, API key management (`sk-llm-*` keys), and an OpenAI-compatible `/v1/models` endpoint. The `rag-api` validates API keys against the portal on every `/v1/*` request. A web UI is included for key management.

## The Geometric Lens

The Lens implements an ARM-EBM (Adaptive Riemannian Metric / Energy-Based Model) duality. A cost field C(x) maps code embeddings to a scalar energy: passing code concentrates near energy 5.00, failing code near 14.04. A metric tensor G(x) defines a Riemannian geometry over embedding space, enabling gradient-based correction via dx = -alpha * G^{-1} * grad(C). The implementation uses standard PyTorch autograd.

**What the Lens learns**: C(x) achieves strong energy separation between passing and failing code (Val AUC 0.968, energy gap doubling from 5.3 to 11.3 over 3 retraining epochs). This is real learned structure, not an artifact.

**What it doesn't do**: The V2.5 ablation found that energy-sorted candidate selection is statistically indistinguishable from random ordering (37.7% vs 37.1%, within 3.4pp seed variance). The reason: 92% of tasks are either all-pass or all-fail across k=3 candidates, so ordering doesn't matter. The pass rate improvement in V2 comes from Best-of-K diversity (generating 3 candidates at temp=0.6), not from Lens ranking.

**What it's good for**: C(x) energy correlates strongly with task difficulty (58.5% vs 18.9% pass rate across energy tiers). V3 repurposes it as a difficulty predictor for adaptive compute routing rather than candidate selection. G(x) is currently dormant (loaded but unused by the benchmark pipeline).

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/itigges22/atlas.git && cd atlas
   ```

2. **Configure**
   ```bash
   cp atlas.conf.example atlas.conf
   # Edit atlas.conf: set MODEL_PATH, DATA_DIR, GPU device
   ```

3. **Install dependencies**
   ```bash
   sudo ./scripts/install.sh
   ```

4. **Verify installation**
   ```bash
   ./scripts/verify-install.sh
   ```

5. **Run the V2 benchmark**
   ```bash
   benchmark/run_v2_benchmark.sh
   ```

See [docs/SETUP.md](docs/SETUP.md) for full installation instructions.

## Hardware Requirements

| Resource | Minimum | Tested |
|----------|---------|--------|
| Python | 3.10+ | 3.11 |
| GPU VRAM | 16 GB | RTX 5060 Ti 16 GB |
| System RAM | 14 GB | 16 GB |
| Storage | ~20 GB | 150 GB SSD |
| OS | RHEL 9 / Ubuntu 24 | RHEL 9 (Proxmox VM) |

## Project Structure

```
api-portal/      -- API key management portal (JWT auth, web UI)
benchmark/       -- V2 benchmark suite (LCB, GPQA, SciCode, Custom, IFBench)
docs/            -- Architecture, setup, configuration, troubleshooting
manifests/       -- K3s deployment manifests
rag-api/         -- Core API: Geometric Lens, router, RAG, cache
llama-server/    -- llama.cpp server container
atlas/sandbox/   -- Isolated code execution environment
scripts/         -- Installation and management scripts
tests/           -- Test suite
```

## V3 Roadmap

V3 targets 70%+ LiveCodeBench through diversity-driven generation, adaptive compute allocation, and novel inference-time theory formation. The core thesis: a frozen model with the right selection and routing infrastructure can match models 10x its size.

## License

Licensed under the ATLAS Source Available License v1.0 -- see [LICENSE](LICENSE).
