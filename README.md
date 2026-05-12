<p align="center">
  <img src="docs/images/herodemo.gif" alt="ATLAS TUI in action"/><br/>
  <sub><i>The ATLAS TUI live, 10× sped up, running the V3 pipeline on a file creation.</i></sub>
</p>

<h1 align="center">A.T.L.A.S.</h1>
<p align="center"><b>Adaptive Test-time Learning and Autonomous Specialization</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/version-V3.1.0-blue" alt="Version"/>
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License"/>
  <img src="https://img.shields.io/badge/model-Qwen3.5--9B-green" alt="Model"/>
</p>

<p align="center">
  <a href="docs/lang/zh-CN/README.md"><img src="https://img.shields.io/badge/文档-简体中文-orange" alt="简体中文"/></a>
  <a href="docs/lang/ja/README.md"><img src="https://img.shields.io/badge/ドキュメント-日本語-orange" alt="日本語"/></a>
  <a href="docs/lang/ko/README.md"><img src="https://img.shields.io/badge/문서-한국어-orange" alt="한국어"/></a>
</p>


## 🌎 What is ATLAS?

ATLAS is a coding assistant that runs on your own GPU. Point it at a project and it does the kind of work you'd ask Claude or Copilot to do: read a codebase, write a feature, fix a bug. The model never leaves your machine.

Every hosted AI tool is a subscription, a privacy tradeoff, and a vendor you trust to stick around. ATLAS isn't any of those. Your code stays on your hardware. You don't pay per token. If the project disappears tomorrow, your install keeps working.

Open models historically can't keep up with hosted ones. ATLAS gets there anyway with a layer of inference scaffolding: planning before generation, verifying answers against self-generated tests, repairing failures. The 14B reference build scored 74.6% on LiveCodeBench. ATLAS canonically runs a 9B that fits on a $500 GPU, but isn't tied to any one model.

---

## 🔥 Latest News

- **2026-04-13** - ["How to Run an AI Coding Assistant on a $500 GPU and Beat Claude Sonnet"](https://devtrends.ru/python/itigges22-atlas) - devtrends.ru
- **2026-04-05** - **[V3.0.1 released](CHANGELOG.md)** - interactive CLI, Docker Compose deployment, 95.8% reliability
- **2026-04-03** - ["$500 GPU Beats Claude: Local AI Revolution for Web Devs"](https://ownet.it/blog/500-gpu-beats-claude-local-ai-revolution-for-web-devs) - ownet.it
- **2026-03-29** - ["A $500 GPU Just Outscored Claude Sonnet on Coding Benchmarks"](https://aivy.com.au/news/atlas-500-gpu-outperforms-claude-sonnet-coding/) - Aivy
- **2026-03-28** - ["Why a $500 GPU Can Beat Claude Sonnet on Coding Benchmarks"](https://medium.com/data-science-collective/why-a-500-gpu-can-beat-claude-sonnet-on-coding-benchmarks-6c8169ffe4fe) - Data Science Collective
- **2026-03-27** - ["ATLAS: A $500 GPU Outperforms Claude Sonnet"](https://clauday.com/article/b92c5551-b490-4d76-ae3d-d8dedf10d88b) - Clauday
- **2026-03-27** - ["ATLAS – lokal AI-koding på 5000kr GPU slår Claude på benchmark"](https://www.jansverre.net/atlas-lokal-ai-koding-pa-500-gpu-slar-claude-pa-benchmark/) - jansverre.net (Norwegian)
- **2026-03-26** - ["Local LLM Coding: $500 GPU Beats Claude: Not the Story"](https://novaknown.com/2026/03/26/local-llm-coding/) - Sarah Fraser, novaknown.com
- **2026-03-26** - ["ATLAS: How a $500 GPU Achieves 74.6% LiveCodeBench Performance Through Intelligent Infrastructure"](https://techplanet.today/post/atlas-how-a-500-gpu-achieves-746-livecodebench-performance-through-intelligent-infrastructure) - TechPlanet
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

1. **[atlas-tui](docs/CLI.md)** - native Bubbletea terminal UI; the canonical chat client (PC-062). Type `atlas` in any project directory to launch it.
  - a. [Live pipeline view](docs/CLI.md#panes) - watch V3 stages stream in a side pane
  - b. [Slash commands](docs/CLI.md#slash-commands) - `/add`, `/diff`, `/commit`, `/run` for local file context and shell-out
  - c. [Input modes](docs/CLI.md#input-modes) - chat, `!bash`, and `/slash` with a hint dropdown

2. **[atlas-proxy](docs/ARCHITECTURE.md#3-atlas-proxy-outer-layer)** - Go agent loop that orchestrates the system.
  - a. [Tool-call routing](docs/ARCHITECTURE.md#tools) - classifies file operations by complexity tier
  - b. [Grammar enforcement](docs/ARCHITECTURE.md#grammar-enforcement) - GBNF schemas keep JSON output valid
  - c. [BiasBusters](docs/ARCHITECTURE.md#tool-selection-bias-mitigations-may-2026-biasbusters-synthesis) - four composed mitigations (descriptions, grammar bans, system notes, ASA steering) that push the model toward `ast_edit` for structural code edits
  - d. [Safety limits](docs/ARCHITECTURE.md#safety-limits) - turn caps, token budgets, timeouts

3. **[V3 Pipeline](docs/ARCHITECTURE.md#4-v3-pipeline-inner-layer)** - multi-phase code generation; turns a single prompt into a verified candidate.
  - a. [PlanSearch](docs/reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - constraint-driven structured planning
  - b. [DivSampling](docs/reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - diverse candidates across temperature and strategy
  - c. [Budget Forcing](docs/reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - per-phase thinking-token allocation
  - d. [PR-CoT Repair](docs/reports/V3_ABLATION_STUDY.md#pr-cot-repair-36-rescues) - self-generated test cases for iterative fixes
  - e. [Refinement Loops](docs/reports/V3_ABLATION_STUDY.md#refinement-loop-6-rescues) - sandbox verify and correct, then repeat
  - f. [Derivation Chains](docs/reports/V3_ABLATION_STUDY.md#derivation-chains-0-rescues) - multi-step reasoning for harder problems

4. **[Geometric Lens](docs/ARCHITECTURE.md#5-geometric-lens)** - energy-based scoring over the model's own embeddings, no external oracle. ([What is a "Geometric Lens"?](docs/ARCHITECTURE.md#why-geometric-lens))
  - a. [C(x) Cost Field](docs/ARCHITECTURE.md#scoring-models) - 4096→512→128→1 MLP that scores candidate quality
  - b. [G(x) Quality Prediction](docs/ARCHITECTURE.md#scoring-models) - XGBoost ensemble used for selection
  - c. [RAG / PageIndex V2](docs/ARCHITECTURE.md#rag--pageindex-v2) - AST-aware code retrieval and project indexing
  - d. [Confidence Router](docs/ARCHITECTURE.md#confidence-router--pattern-cache) - Thompson Sampling routes compute to the candidates that need it

5. **[Sandbox](docs/ARCHITECTURE.md#6-sandbox)** - isolated execution for build verification.
  - a. Multi-language execution: Python, Rust, Go, C, Shell, others
  - b. Compilation and linting before scoring
  - c. Runs both generated and existing test suites

6. **[llama-server](docs/CONFIGURATION.md#6-llama-server)** - local LLM inference on one consumer GPU.
  - a. CUDA-accelerated quantized inference (Q6_K / Q4_K_M)
  - b. Grammar-constrained decoding at the token level
  - c. Self-embeddings, so the lens doesn't need a second model

Full documentation (setup, architecture, configuration, troubleshooting, benchmark reports, and the [research behind each component](docs/SOURCES.md)) lives in the [docs/](docs/) directory.

---

## 🚀 Get Started

One-shot install:
```bash
curl -fsSL https://raw.githubusercontent.com/itigges22/ATLAS/main/scripts/atlas-bootstrap.sh | bash
```
The script detects your distro (Ubuntu, Debian, RHEL, Fedora, Rocky, Alma), installs Docker and nvidia-container-toolkit, downloads the model weights, builds the ASA steering vector, and starts the stack. Expect 10-30 minutes; the model download is the bottleneck.

Then in any project directory, run `atlas`.

**Requirements**

| | |
|---|---|
| GPU | NVIDIA, 16GB+ VRAM (tested on RTX 5060 Ti 16GB) |
| Runtime | Docker with nvidia-container-toolkit, or Podman |
| Python | 3.9+ |
| Disk | ~20GB (model weights + container images) |

Tested on NVIDIA only. macOS, Windows, and AMD ROCm are on the V3.1.1 roadmap. For the manual install path (Docker Compose, bare-metal, K3s) and the full set of bootstrap flags, see **[SETUP.md](docs/SETUP.md)**.

---

## ⚠️ Known Limitations

- **NVIDIA-only.** Tested on NVIDIA GPUs. AMD ROCm and Apple Metal are on the V3.1.1 roadmap.
- **9B model is not formally benchmarked yet.** V3.1.0 ships Qwen3.5-9B with the full V3 pipeline, but the canonical 74.6% LiveCodeBench score is from the 14B reference build. Formal 9B numbers land with V3.1.1. The 14B methodology and ablations live in [`docs/reports/V3_ABLATION_STUDY.md`](docs/reports/V3_ABLATION_STUDY.md); raw traces are on [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS).
- **Complex feature additions can be inconsistent.** The model sometimes spends agent turns exploring an unfamiliar codebase before writing code. Reliability has improved on the 9B build since the V3.0 measurement; a fresh number lands with the V3.1.1 benchmark pass.
- **Grammar-constrained decoding is slow.** Around 51 tok/s on llama-server.

---

## 🗺️ Roadmap

**V3.1.0** - Current release. Bubbletea TUI as the canonical chat client (PC-062), `atlas init` first-run wizard (PC-054), `atlas doctor` install diagnostic (PC-053), `atlas tier` hardware-aware presets (PC-055), K3s deployment templates restored, ASA steering vectors auto-built during install (BiasBusters #4).

**V3.1.1** - Next release.
- OS support - macOS and Windows installers.
- Expanded accelerator support - AMD ROCm via llama.cpp; Apple Metal once macOS lands.
- Formal 9B benchmarks - LiveCodeBench, GPQA Diamond, SciCode on Qwen3.5-9B.

---

## 💖 Support ATLAS

ATLAS is built by a single college student in his free time on a single consumer GPU. If the project has been useful to you and you want to help keep it sustainable, please consider **[sponsoring on GitHub](https://github.com/sponsors/itigges22)**.

Sponsorship directly funds:

- **Compute & hardware** — more GPUs for faster benchmark iteration, access to architectures the maintainer can't afford (AMD ROCm, higher VRAM cards, cloud rentals for larger-model experiments).
- **Contributor bounties** — meaningful compensation for external contributors who put real time into substantive PRs, so ATLAS can grow faster than a single-person pace allows.
- **Research** — continued academic engagement around the architecture, from future workshop and conference submissions to paper writing and collaborations that validate and extend the approach.
- **Community** — continued support for the community and platforms ATLAS runs on, including documentation, user-facing channels, and educational content that help ATLAS reach more developers and better serve the ones already using it.

Every sponsor is credited in the release notes of the version they helped fund.

---

## 🤝 Contributing

We're building ATLAS in the open and we're actively looking for contributors and core maintainers. Whether you're fixing a bug, adding accelerator support, or rethinking a whole subsystem - there's a place for you here. If you believe open models deserve better infrastructure, come build with us.

Found a bug or hit a wall? **[Open an issue](https://github.com/itigges22/ATLAS/issues)** - you don't need to submit a fix. Bug reports and feedback help just as much as code.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines.

---

## 📄 License

Licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).
