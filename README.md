<p align="center">
  <img src="docs/images/banner.png" alt="ATLAS Banner"/>
</p>

<h1 align="center">A.T.L.A.S.</h1>
<p align="center"><b>Adaptive Test-time Learning and Autonomous Specialization</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/version-V3.0.1-blue" alt="Version"/>
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License"/>
  <img src="https://img.shields.io/badge/model-Qwen3.5--9B-green" alt="Model"/>
  <img src="https://img.shields.io/badge/GPU-RTX_5060_Ti_16GB-red" alt="GPU"/>
</p>

<p align="center">
  <a href="docs/lang/zh-CN/README.md"><img src="https://img.shields.io/badge/文档-简体中文-orange" alt="简体中文"/></a>
  <a href="docs/lang/ja/README.md"><img src="https://img.shields.io/badge/ドキュメント-日本語-orange" alt="日本語"/></a>
  <a href="docs/lang/ko/README.md"><img src="https://img.shields.io/badge/문서-한국어-orange" alt="한국어"/></a>
</p>


## 🌎 What is ATLAS?
ATLAS is a self-hosted coding assistant built on intelligent inference infrastructure. You point it at an open-weight model running locally, and it turns that model into something that competes with frontier systems, with no fine-tuning, no API calls, and no cloud in between.

Instead of training a larger model or routing to a hosted one, ATLAS wraps a frozen local model in a pipeline that plans before generating, verifies its own output against constraints it extracts from the problem, scores candidates with an energy-based lens, and repairs failures through self-generated test feedback. The weights never change. The intelligence lives in the scaffolding around them.

The result is a serious coding assistant that runs on a single consumer GPU for fractions of a cent per task. Nothing leaves your machine, no vendor can pull the model out from under you, and the entire stack is open source. One model, one GPU, no one else's infrastructure in the loop.

---

## 🔥 Latest News

- **2026-04-13** - ["How to Run an AI Coding Assistant on a $500 GPU and Beat Claude Sonnet"](https://devtrends.ru/python/itigges22-atlas) - devtrends.ru
- **2026-04-05** - **[V3.0.1 released](CHANGELOG.md)** - interactive CLI, Docker Compose deployment, 95.8% reliability
- **2026-04-03** - ["$500 GPU Beats Claude: Local AI Revolution for Web Devs"](https://ownet.it/blog/500-gpu-beats-claude-local-ai-revolution-for-web-devs) - ownet.it
- **2026-03-29** - ["A $500 GPU Just Outscored Claude Sonnet on Coding Benchmarks"](https://aivy.com.au/news/atlas-500-gpu-outperforms-claude-sonnet-coding/) - Aivy
- **2026-03-28** - ["Why a $500 GPU Can Beat Claude Sonnet on Coding Benchmarks"](https://medium.com/data-science-collective/why-a-500-gpu-can-beat-claude-sonnet-on-coding-benchmarks-6c8169ffe4fe) - Data Science Collective
- **2026-03-27** - ["ATLAS: A $500 GPU Outperforms Claude Sonnet"](https://clauday.com/article/b92c5551-b490-4d76-ae3d-d8dedf10d88b) - Clauday
- **2026-03-26** - [Hacker News front page](https://news.ycombinator.com/item?id=47533297) - 489 points, 285 comments
- **2026-03-05** - **[V3.0 released](docs/reports/V3_ABLATION_STUDY.md)** - 74.6% LiveCodeBench pass@1-v(k=3) on frozen Qwen3-14B
- **2026-02-18** - **[V2.0 released](CHANGELOG.md)** - benchmark infrastructure, HumanEval/MBPP/LiveCodeBench/GPQA/SciCode evaluation suite

<a href="https://star-history.com/#itigges22/ATLAS&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=itigges22/ATLAS&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=itigges22/ATLAS&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=itigges22/ATLAS&type=Date" width="100%" />
  </picture>
</a>

---

## 🧱 What ATLAS Does

1. **[atlas-proxy](docs/ARCHITECTURE.md#3-atlas-proxy-outer-layer)** - Go-based agent loop that orchestrates the entire system.
  - a. [Tool-call routing](docs/ARCHITECTURE.md#tools) - classifies file operations by complexity tier
  - b. [Grammar enforcement](docs/ARCHITECTURE.md#grammar-enforcement) - GBNF schemas guarantee 100% valid JSON output
  - c. [Safety limits](docs/ARCHITECTURE.md#safety-limits) - turn caps, token budgets, timeout enforcement

2. **[V3 Pipeline](docs/ARCHITECTURE.md#4-v3-pipeline-inner-layer)** - multi-phase code generation that turns a single prompt into verified, high-quality output.
  - a. [PlanSearch](docs/reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - constraint-driven structured planning
  - b. [DivSampling](docs/reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - diverse candidate generation across temperature and strategy
  - c. [Budget Forcing](docs/reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - controls thinking token allocation per phase
  - d. [PR-CoT Repair](docs/reports/V3_ABLATION_STUDY.md#pr-cot-repair-36-rescues) - self-generated test cases for iterative fix cycles
  - e. [Refinement Loops](docs/reports/V3_ABLATION_STUDY.md#refinement-loop-6-rescues) - repeated sandbox verification and correction
  - f. [Derivation Chains](docs/reports/V3_ABLATION_STUDY.md#derivation-chains-0-rescues) - multi-step reasoning for complex problems

3. **[Geometric Lens](docs/ARCHITECTURE.md#5-geometric-lens)** - energy-based scoring and retrieval without external oracles. ([What is a "Geometric Lens"?](docs/ARCHITECTURE.md#why-geometric-lens))
  - a. [C(x) Cost Field](docs/ARCHITECTURE.md#scoring-models) - MLP that scores candidate quality from embeddings
  - b. [G(x) Quality Prediction](docs/ARCHITECTURE.md#scoring-models) - XGBoost model for selection decisions
  - c. [RAG / PageIndex V2](docs/ARCHITECTURE.md#rag--pageindex-v2) - AST-aware code retrieval and project indexing
  - d. [Confidence Router](docs/ARCHITECTURE.md#confidence-router--pattern-cache) - Thompson Sampling routes compute where it matters

4. **[Sandbox](docs/ARCHITECTURE.md#6-sandbox)** - isolated execution environment for build verification.
  - a. Multi-language execution - Python, Rust, Go, C, Shell, and more
  - b. Compilation and linting - syntax verification before scoring
  - c. Test running - executes generated and existing test suites

5. **[llama-server](docs/CONFIGURATION.md#6-llama-server)** - local LLM inference on a single consumer GPU.
  - a. CUDA acceleration - quantized model inference (Q6_K / Q4_K_M)
  - b. Grammar-constrained decoding - structured output at the token level
  - c. Self-embeddings - embedding extraction without a separate model

6. **[Interactive CLI](docs/CLI.md)** - type `atlas` in any project directory and start building.
  - a. [Tool-call agent loop](docs/CLI.md#streaming-output) - read, write, edit, delete, run commands
  - b. [Streaming output](docs/CLI.md#how-streaming-works) - real-time response via SSE
  - c. [Project-aware context](docs/CLI.md#proxy-file-access) - automatic file discovery and injection

Full documentation - setup guides, architecture, configuration, troubleshooting, and benchmark reports - lives in the [docs/](docs/) directory.

---

## 🚀 Get Started

ATLAS requires a GPU with 16GB+ VRAM, Docker (with nvidia-container-toolkit) or Podman, and Python 3.9+. Currently tested on NVIDIA GPUs - ATLAS is not NVIDIA-specific, and ROCm support for AMD GPUs is on the roadmap. See **[SETUP.md](docs/SETUP.md)** for full installation instructions covering Docker Compose, bare-metal, and K3s deployment. Once running, type `atlas` in any project directory and start building.

---

## ⚠️ Known Limitations

- **Tested on NVIDIA only** - ATLAS uses llama.cpp for inference, which supports multiple accelerator backends. ROCm support is a V3.1 priority.
- **9B model not formally benchmarked** - the CLI ships Qwen3.5-9B with the full V3 pipeline, but formal LiveCodeBench scores are from the 14B model. 9B benchmarks are V3.1 work. For the V3 (14B) benchmark results, methodology, and ablation analysis, see [`docs/reports/V3_ABLATION_STUDY.md`](docs/reports/V3_ABLATION_STUDY.md); raw benchmark traces are published on [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS).
- **Complex feature additions can fail** - adding features to existing projects succeeds ~67% of the time. The model sometimes over-explores instead of writing code.
- **Grammar-constrained inference speed** - ~51 tok/s on llama-server. Faster grammar integration is planned for V3.1.

---

## 🗺️ Roadmap

**V3.0.1** - Current release. Interactive CLI, Docker Compose deployment, V3 pipeline integration.

**V3.1** - In progress.
- ROCm support - AMD GPU inference via llama.cpp ROCm backend.
- Formal 9B benchmarks - LiveCodeBench, GPQA Diamond, SciCode on Qwen3.5-9B.
- CLI reliability - expanded testing, targeting L6 ≥ 90%.
- Grammar speed - C-side sampler chain for faster constrained decoding.
- Structural code reasoning - tree-sitter + solver-backed (Prolog or Z3) call graph for reachability queries, scoped context injection, and real cyclomatic complexity. **Bottleneck it solves:** L6 tasks (modifying existing codebases) currently burn agent turns and tokens as the model manually explores unfamiliar projects through the MCP loop. A call graph with canned reachability queries ("who calls X?", "is this path reachable from main?") lets the pipeline skip the exploration and hand the model a scoped context window instead. Also upgrades tier classification from pattern matching to true complexity and lets candidate verification catch structural bugs the sandbox can't. Language-dependent — ships with Python and JS/TS adapters, pluggable rules for others. Inspired by [Dmitri Sotnikov's chiasmus](https://github.com/yogthos/chiasmus/tree/main).

**V3.2** - Exploratory.
- [Reasoning with Sampling](https://arxiv.org/abs/2510.14901) - MCMC over logits during decoding so bad token trajectories are pruned before completion. **Bottleneck it solves:** today every PlanSearch candidate runs to the full 8K-token budget before the Geometric Lens can score and reject it, which wastes compute on trajectories that were already going off the rails at token 500. In-generation backtracking kills bad candidates early; the saved compute buys more attempts at harder problems. Complementary to G(x)'s post-hoc ranking — token-block level pruning vs whole-candidate level selection. Requires DeltaNet state checkpoint/restore in our patched llama.cpp (hybrid architecture has a non-trivially reversible recurrent state, so it's inference-only but more invasive than a sampler tweak).

---

## 🤝 Contributing

We're building ATLAS in the open and we're actively looking for contributors and core maintainers. Whether you're fixing a bug, adding accelerator support, or rethinking a whole subsystem - there's a place for you here. If you believe open models deserve better infrastructure, come build with us.

Found a bug or hit a wall? **[Open an issue](https://github.com/itigges22/ATLAS/issues)** - you don't need to submit a fix. Bug reports and feedback help just as much as code.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines.

---

## 📄 License

Licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).
