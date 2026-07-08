# ATLAS Repository Map

Orientation map for the repository: what each top-level directory owns and where
the documentation lives. For component internals see
[ARCHITECTURE.md](ARCHITECTURE.md); for the full API surface see
[API.md](API.md).

---

## Top-level directories

| Directory | Owns | More |
|---|---|---|
| `proxy/` | Go agent loop: `/v1/agent`, grammar-constrained tool calls, gates, V3 routing, OpenAI passthrough | [proxy/README.md](../proxy/README.md) |
| `tui/` | Bubbletea terminal client — the canonical chat front-end, consumes proxy SSE streams | — |
| `atlas/` | Python CLI package: `atlas` subcommand dispatch (init, doctor, tier, model, onboard, lens, asa, publish, bench, compose, tui, upgrade, rollback, diagnostics, artifact, config) and pipe-mode REPL | — |
| `v3-service/` | Python HTTP wrapper for the V3 generation pipeline and the structural call-graph engine | — |
| `geometric-lens/` | Scoring (C(x)/G(x)), RAG indexing and retrieval, confidence routing, pattern cache, ASA control-vector build | [asa_calibration/README.md](../geometric-lens/asa_calibration/README.md) |
| `sandbox/` | Isolated multi-language code execution and shell, with workspace containment | — |
| `inference/` | llama-server Docker builds (CUDA / ROCm / Vulkan) and model-neutral entrypoints | — |
| `benchmark/` | Benchmark runners, dataset loaders, and the V3 pipeline modules | [benchmark/README.md](../benchmark/README.md) |
| `scripts/` | Install, deploy, K3s, and training/retraining automation | — |
| `templates/` | K3s manifest templates rendered from `atlas.conf` via envsubst | — |
| `tests/` | Cross-cutting test suite: CLI, infrastructure, V3, and v3-service | — |

Root also holds the Docker Compose stack (`docker-compose.yml` plus the rocm /
vulkan / cpu / macos overlays), `atlas.conf.example` (K3s), `pyproject.toml`
(the `atlas` CLI entry point), and the standard `.env.example`.

---

## Documentation

| Doc | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Two-layer architecture, component breakdowns, data-flow and sequence diagrams |
| [API.md](API.md) | HTTP API reference for every service |
| [CLI.md](CLI.md) | CLI + TUI usage, subcommands, and workflow examples |
| [CONFIGURATION.md](CONFIGURATION.md) | Environment variables, internal constants, and K3s config |
| [SETUP.md](SETUP.md) | Installation: one-shot bootstrap, Docker Compose, K3s |
| [SETUP_MACOS.md](SETUP_MACOS.md) | macOS hybrid install: native Metal llama-server + Docker stack |
| [OPERATIONS.md](OPERATIONS.md) | Day-2 operations: health, logs, runbooks, upgrades, rollback, backup |
| [CONTAINER_PACKAGING.md](CONTAINER_PACKAGING.md) | How the service images are built: runtime accounts, writable paths, dependency pinning |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Dev workflow: targeted rebuilds, host-side proxy, tests |
| [PUBLISHING.md](PUBLISHING.md) | HuggingFace + GitHub publish flow for Lens / ASA artifacts |
| [RELEASE.md](RELEASE.md) | Release contract: capability status, service contracts, and verification levels |
| [PLAN_MODE.md](PLAN_MODE.md) | Plan mode: per-turn pre-flight planning and adherence constants |
| [PROTOCOL.md](PROTOCOL.md) | Typed event envelope contract shared by proxy, v3-service, and clients |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |
| [SOURCES.md](SOURCES.md) | Research papers bucketed by status relative to the current release |
| [STORY.md](STORY.md) | Project background |
| [MAP.md](MAP.md) | This file |
| [adr/](adr/) | Architecture Decision Records (trust model, per-model bundles, SQLite state store, …) |
| [schemas/](schemas/) | Machine-readable contracts: proxy OpenAPI spec, SSE envelope and error-envelope JSON Schemas |
| [lang/](lang/) | Translated documentation (zh-CN, ja, ko) |
| [reports/](reports/) | Ablation studies and call-graph design notes; [reports/archive/](reports/archive/) holds historical status trackers |

Root docs: [README.md](../README.md), [CHANGELOG.md](../CHANGELOG.md),
[SUPPORT_MATRIX.md](../SUPPORT_MATRIX.md), [SECURITY.md](../SECURITY.md),
[GOVERNANCE.md](../GOVERNANCE.md), [MAINTAINERS.md](../MAINTAINERS.md),
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md),
[CONTRIBUTING.md](../CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md), and [LICENSE](../LICENSE).
