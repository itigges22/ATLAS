<p align="center">
  <img src="docs/images/herodemo.gif" alt="ATLAS TUI in action"/><br/>
  <sub><i>The ATLAS TUI live, 10× sped up, running the V3 pipeline on a file creation.</i></sub>
</p>

<h1 align="center">A.T.L.A.S.</h1>
<p align="center"><b>Adaptive Test-time Learning and Autonomous Specialization</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/version-V3.1.3-blue" alt="Version"/>
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License"/>
  <img src="https://img.shields.io/badge/model-agnostic-green" alt="Model-agnostic"/>
</p>

<p align="center">
  <a href="https://github.com/itigges22/ATLAS/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/itigges22/ATLAS/test.yml?branch=main&label=tests" alt="Tests"/></a>
  <a href="https://github.com/itigges22/ATLAS/actions/workflows/install-test.yml"><img src="https://img.shields.io/github/actions/workflow/status/itigges22/ATLAS/install-test.yml?branch=main&label=install%20matrix" alt="Install matrix"/></a>
  <a href="https://github.com/itigges22/ATLAS/actions/workflows/codeql.yml"><img src="https://img.shields.io/github/actions/workflow/status/itigges22/ATLAS/codeql.yml?branch=main&label=codeql" alt="CodeQL"/></a>
  <a href="https://github.com/itigges22/ATLAS/actions/workflows/container-scan.yml"><img src="https://img.shields.io/github/actions/workflow/status/itigges22/ATLAS/container-scan.yml?label=container%20scan" alt="Container scan"/></a>
  <a href="https://github.com/itigges22/ATLAS/actions/workflows/verify-tags.yml"><img src="https://img.shields.io/github/actions/workflow/status/itigges22/ATLAS/verify-tags.yml?label=release%20signature" alt="Release signature"/></a>
</p>

<p align="center">
  <a href="docs/lang/zh-CN/README.md"><img src="https://img.shields.io/badge/文档-简体中文-orange" alt="简体中文"/></a>
  <a href="docs/lang/ja/README.md"><img src="https://img.shields.io/badge/ドキュメント-日本語-orange" alt="日本語"/></a>
  <a href="docs/lang/ko/README.md"><img src="https://img.shields.io/badge/문서-한국어-orange" alt="한국어"/></a>
</p>


## 🌎 What is ATLAS?

**ATLAS is a local coding agent that brings frontier-style reasoning and verification to compact open models.** It puts more intelligence in the system around the model (planning, candidate generation, quality scoring, sandboxed testing, and repair) so smaller models can tackle real software work entirely on your own hardware, without a hosted API or per-token fees.

## 💡 Why ATLAS?

* **Get more from smaller models.** ATLAS adds planning, candidate selection, verification, and repair around the model instead of depending on a single generation.
* **Verify before accepting.** Generated code can be compiled, tested, and corrected inside an isolated execution environment.
* **Spend compute where it matters.** Straightforward edits take a shorter path, while harder tasks receive more candidates, reasoning, and validation.
* **Run your own model.** Use a compatible GGUF model on NVIDIA, AMD, Apple Silicon, Vulkan, or CPU-supported hardware.
* **Keep everything local.** Your repository, inference, test execution, and learned artifacts stay on your machine.
* **Own the full stack.** ATLAS is open source, self-hosted, and has no required hosted API or per-token fee.

---

## 📰 Latest News

- **2026-07-06** - **[V3.1.3 "Maia" released](https://github.com/itigges22/ATLAS/releases/tag/v3.1.3)** - production-platform pass: staged upgrade/rollback with auto-restore, SQLite state store (no more Redis), signed artifact manifests, structured logs + correlation IDs, interactive permissions, session resume, and two adversarial bug-fix sweeps
- **2026-06-17** - **[V3.1.2 "Maia" released](https://github.com/itigges22/ATLAS/releases/tag/v3.1.2)** - broader hardware reach (ROCm / Metal / Vulkan), bring-your-own-model Lens + ASA training, in-the-loop lens retraining from your own workloads, and an agent-reliability pass
- **2026-05-12** - **[V3.1.0 "Maia" released](https://github.com/itigges22/ATLAS/releases/tag/v3.1.0)** - native Bubbletea TUI, one-command bootstrap, streaming Lens + ASA activation steering, AST-aware surgical edits
- **2026-03-26** - [Hacker News front page](https://news.ycombinator.com/item?id=47533297) - 489 points, 285 comments
- **2026-03-05** - **[V3.0 released](docs/reports/V3_ABLATION_STUDY.md)** - 74.6% LiveCodeBench pass@1-v(k=3) on frozen Qwen3-14B (pass@1 with k=3 generated candidates, Lens selection, and repair - not single-generation pass@1; [methodology](docs/reports/V3_ABLATION_STUDY.md))
- **2026-02-18** - **[V2.0 released](CHANGELOG.md)** - benchmark infrastructure, HumanEval/MBPP/LiveCodeBench/GPQA/SciCode evaluation suite

## ⭐ Star History

<!-- Self-hosted chart: rendered weekly by .github/workflows/star-chart.yml
     onto the `star-history` asset branch (scripts/star-history-chart.py).
     Replaces the star-history.com embed, whose shared token pool
     rate-limits unpredictably. -->
<a href="https://github.com/itigges22/ATLAS/stargazers">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/itigges22/ATLAS/star-history/star-history-dark.svg" />
   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/itigges22/ATLAS/star-history/star-history-light.svg" />
   <img alt="Star history chart" src="https://raw.githubusercontent.com/itigges22/ATLAS/star-history/star-history-light.svg" width="100%" />
 </picture>
</a>

---

## 🧱 What ATLAS Does

1. **[atlas-tui](docs/CLI.md)** - native Bubbletea terminal UI; the canonical chat client. Type `atlas` in any project directory to launch it.
   - [Live pipeline view](docs/CLI.md#panes) - watch V3 stages stream in a side pane
   - [Slash commands](docs/CLI.md#slash-commands) - `/add`, `/diff`, `/commit`, `/run` for local file context and shell-out
   - [Input modes](docs/CLI.md#input-modes) - chat, `!bash`, and `/slash` with a hint dropdown

2. **[atlas-proxy](docs/ARCHITECTURE.md#3-atlas-proxy-outer-layer)** - Go agent loop that orchestrates the system.
   - [Tool-call routing](docs/ARCHITECTURE.md#tools) - classifies file operations by complexity tier
   - [Grammar enforcement](docs/ARCHITECTURE.md#grammar-enforcement) - GBNF schemas keep JSON output valid
   - [BiasBusters](docs/ARCHITECTURE.md#tool-selection-bias-mitigations) - four composed mitigations (descriptions, grammar bans, system notes, ASA steering) that push the model toward `ast_edit` for structural code edits
   - [Safety limits](docs/ARCHITECTURE.md#safety-limits) - turn caps, token budgets, timeouts

3. **[V3 Pipeline](docs/ARCHITECTURE.md#4-v3-pipeline-inner-layer)** - multi-phase code generation; turns a single prompt into a verified candidate.
   - [PlanSearch](docs/reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - constraint-driven structured planning
   - [DivSampling](docs/reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - diverse candidates across temperature and strategy
   - [Budget Forcing](docs/reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - per-phase thinking-token allocation
   - [PR-CoT Repair](docs/reports/V3_ABLATION_STUDY.md#pr-cot-repair-36-rescues) - self-generated test cases for iterative fixes
   - [Refinement Loops](docs/reports/V3_ABLATION_STUDY.md#refinement-loop-6-rescues) - sandbox verify and correct, then repeat
   - [Derivation Chains](docs/reports/V3_ABLATION_STUDY.md#derivation-chains-0-rescues) - multi-step reasoning for harder problems

4. **[Geometric Lens](docs/ARCHITECTURE.md#5-geometric-lens)** - energy-based scoring over the model's own embeddings, no external oracle. ([What is a "Geometric Lens"?](docs/ARCHITECTURE.md#why-geometric-lens))
   - [C(x) Cost Field](docs/ARCHITECTURE.md#scoring-models) - model-hidden-dim→512→128→1 MLP that scores candidate quality
   - [G(x) Quality Prediction](docs/ARCHITECTURE.md#scoring-models) - XGBoost ensemble used for selection
   - [RAG / PageIndex V2](docs/ARCHITECTURE.md#rag--pageindex-v2) - AST-aware code retrieval and project indexing
   - [Confidence Router](docs/ARCHITECTURE.md#confidence-router--pattern-cache) - Thompson Sampling routes compute to the candidates that need it

5. **[Sandbox](docs/ARCHITECTURE.md#6-sandbox)** - isolated execution for build verification.
   - Multi-language execution: Python, Rust, Go, C, Shell, others
   - Compilation and linting before scoring
   - Runs both generated and existing test suites

6. **[llama-server](docs/CONFIGURATION.md#6-llama-server)** - local LLM inference on one consumer GPU.
   - GPU-accelerated quantized inference (Q6_K / Q4_K_M) - NVIDIA CUDA, AMD ROCm, Apple Metal (macOS hybrid), and Vulkan; Intel SYCL on the roadmap
   - Grammar-constrained decoding at the token level
   - Self-embeddings, so the lens doesn't need a second model

Full documentation (setup, architecture, configuration, troubleshooting, benchmark reports, and the [research behind each component](docs/SOURCES.md)) lives in the [docs/](docs/) directory.

---

## 🚀 Get Started

One-shot install:
```bash
curl -fsSL https://raw.githubusercontent.com/itigges22/ATLAS/main/scripts/atlas-bootstrap.sh | bash
```

Prefer not to pipe a moving script into bash? Same installer, two more careful ways to run it:
```bash
# Pinned to a release: script, checkout, and images all at the signed tag
curl -fsSL https://raw.githubusercontent.com/itigges22/ATLAS/v3.1.3/scripts/atlas-bootstrap.sh \
  | ATLAS_BOOTSTRAP_REF=v3.1.3 bash

# Review before running
curl -fsSL -o atlas-bootstrap.sh https://raw.githubusercontent.com/itigges22/ATLAS/main/scripts/atlas-bootstrap.sh
less atlas-bootstrap.sh
bash atlas-bootstrap.sh
```

The script detects your distro (Ubuntu, Debian, RHEL, Fedora, Rocky, Alma) and your GPU vendor (NVIDIA → nvidia-container-toolkit; AMD → ROCm device passthrough), installs the appropriate runtime, downloads the model weights, builds the ASA steering vector, and starts the stack. Expect 10-30 minutes; the model download is the bottleneck.

Then in any project directory, run `atlas`.

**Requirements**

| | |
|---|---|
| GPU | 16 GB+ VRAM. NVIDIA (CUDA), AMD (ROCm), or Apple Silicon (Metal, macOS hybrid); Vulkan covers most other GPUs. See [SETUP.md § Supported GPUs](docs/SETUP.md#supported-gpus). To size a specific model to your card, see [What fits on my GPU?](docs/TROUBLESHOOTING.md#what-fits-on-my-gpu). |
| Runtime | Docker (NVIDIA: + nvidia-container-toolkit; AMD: standalone Docker is enough) or Podman |
| Python | 3.9+ |
| Disk | ~20 GB CUDA / ~22 GB ROCm (model weights + container images) |

Apple Silicon runs natively through the macOS hybrid Metal path (native llama-server + Docker for the rest - see **[SETUP_MACOS.md](docs/SETUP_MACOS.md)**); Intel Arc (SYCL) is on the roadmap. For the manual install path (Docker Compose, bare-metal, K3s) and the full set of bootstrap flags, see **[SETUP.md](docs/SETUP.md)**.

---

## ⚠️ Known Limitations

- **Linux Docker stack, plus a native macOS path.** NVIDIA, AMD ROCm, and Vulkan Docker paths ship today; Apple Silicon runs via the native macOS hybrid Metal path ([#32](https://github.com/itigges22/ATLAS/issues/32)). Intel Arc / SYCL is on the roadmap.
- **Current registry models are not formally benchmarked yet.** The canonical 74.6% LiveCodeBench score is from the frozen 14B reference build. New model-specific numbers are tracked in [#28](https://github.com/itigges22/ATLAS/issues/28). The reference methodology and ablations live in [`docs/reports/V3_ABLATION_STUDY.md`](docs/reports/V3_ABLATION_STUDY.md); raw traces are on [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS).
- **Complex feature additions can be inconsistent.** Compact models sometimes spend agent turns exploring an unfamiliar codebase before writing code. Reliability has improved through the V3.1.2 agent-reliability pass; fresh model-specific numbers are tracked in [#28](https://github.com/itigges22/ATLAS/issues/28).
- **Grammar-constrained decoding is slow.** Around 51 tok/s on llama-server.

---

## 🗺️ Roadmap

**V3.1.3 "Maia"** - Current release. Production-platform pass on top of V3.1.2: staged `atlas upgrade`/`rollback` with automatic restore, SQLite state store replacing Redis ([ADR 0007](docs/adr/0007-sqlite-state-store.md)), signed artifact manifests, structured JSON logs with cross-service correlation IDs, interactive permission prompts, session resume, typed config validation/migration, and two adversarial bug-fix sweeps (33 confirmed fixes).

**V3.1.2 "Maia"** - Broader hardware reach, bring-your-own-model training, and an agent-reliability pass on top of the V3.1.0 base (TUI, one-command install, streaming Lens + ASA).
- Hardware reach: AMD ROCm via llama.cpp incl. RDNA4 / RX 9070 (gfx1200/gfx1201) ([#26](https://github.com/itigges22/ATLAS/issues/26)); Apple Silicon native macOS hybrid Metal path ([#32](https://github.com/itigges22/ATLAS/issues/32), see [SETUP_MACOS.md](docs/SETUP_MACOS.md)); Vulkan universal fallback covering AMD / Intel / Snapdragon / Apple-via-MoltenVK / CPU ([#114](https://github.com/itigges22/ATLAS/issues/114)).
- Bring-your-own-model: local Lens training pipeline (`atlas lens build` / `retrain`, [#100](https://github.com/itigges22/ATLAS/issues/100)) and ASA per-model calibration parity (`atlas asa check/build/publish`, [#113](https://github.com/itigges22/ATLAS/issues/113)) - train Lens + ASA artifacts for additional GGUFs, with per-model operating thresholds that ship with the lens.
- In-the-loop lens training: rate passes in the TUI (`/good` · `/bad` · `/review` · `/deny`) → collected, weighted samples → `atlas lens retrain` on your own workloads.
- Agent reliability: tool-result visibility fix, read-dedup, traceback → directed-edit, `move_file`, pip-install / case-mismatch steers, sandbox shell policy + host-sized cgroup limits.
- Structural call-graph reasoning ([#39](https://github.com/itigges22/ATLAS/issues/39) / [#125](https://github.com/itigges22/ATLAS/pull/125), thanks [@yogthos](https://github.com/yogthos)); ARCHITECTURE.md translated to zh-CN / ja / ko ([#25](https://github.com/itigges22/ATLAS/issues/25)).

**V3.2** - Next milestone: deeper code reasoning and planning.
- Architecture-first planning phase - RPG-style plan-then-fill: plan at module scope, then implement at function scope ([#120](https://github.com/itigges22/ATLAS/issues/120), PR [#124](https://github.com/itigges22/ATLAS/pull/124)).
- Structural code reasoning (tail) - solver-backed reachability + syntax-agnostic wavelet decomposition for multi-resolution "which files matter" retrieval ([#39](https://github.com/itigges22/ATLAS/issues/39)).
- Reasoning with sampling - efficiency and quality gains ([#9](https://github.com/itigges22/ATLAS/issues/9)).
- Deferred infra: automated HuggingFace submission pipeline ([#102](https://github.com/itigges22/ATLAS/issues/102)); ROCm on K3s / Kubernetes; formal registry-model benchmarks - LiveCodeBench, GPQA Diamond, SciCode ([#28](https://github.com/itigges22/ATLAS/issues/28)).

**Backlog / help wanted**
- Hardware: ARM64 multi-arch builds ([#115](https://github.com/itigges22/ATLAS/issues/115)), multi-GPU for larger models ([#34](https://github.com/itigges22/ATLAS/issues/34)), Intel oneAPI / SYCL ([#27](https://github.com/itigges22/ATLAS/issues/27)).
- Tooling: VS Code / JetBrains extension ([#35](https://github.com/itigges22/ATLAS/issues/35)).
- Sandbox languages: Java / Kotlin ([#29](https://github.com/itigges22/ATLAS/issues/29)), Ruby / PHP ([#30](https://github.com/itigges22/ATLAS/issues/30)).
- Architecture: model-agnostic platform ([#66](https://github.com/itigges22/ATLAS/issues/66)).

---

## ❤️ Support ATLAS

ATLAS is built by a single college student in his free time on a single consumer GPU ([the story behind ATLAS](docs/STORY.md)). If the project has been useful to you and you want to help keep it sustainable, please consider **[sponsoring on GitHub](https://github.com/sponsors/itigges22)**.

Sponsorship directly funds:

- **Compute & hardware** - more GPUs for faster benchmark iteration, access to architectures the maintainer can't afford (AMD ROCm, higher VRAM cards, cloud rentals for larger-model experiments).
- **Contributor bounties** - meaningful compensation for external contributors who put real time into substantive PRs, so ATLAS can grow faster than a single-person pace allows.
- **Research** - continued academic engagement around the architecture, from future workshop and conference submissions to paper writing and collaborations that validate and extend the approach.
- **Community** - continued support for the community and platforms ATLAS runs on, including documentation, user-facing channels, and educational content that help ATLAS reach more developers and better serve the ones already using it.

Every sponsor is credited in the release notes of the version they helped fund.

---

## 🤝 Contributing

ATLAS is developed in the open and welcomes contributors and core maintainers. Bug fixes, accelerator support, and larger subsystem work are all welcome.

Found a bug or hit a wall? **[Open an issue](https://github.com/itigges22/ATLAS/issues)** - you don't need to submit a fix. Bug reports and feedback help just as much as code.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines and the [repository map](docs/MAP.md) for an overview of the codebase layout.

---

## 📄 License

Licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).
