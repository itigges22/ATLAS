# ATLAS Support Matrix

Applies to: **V3.1.2 and the current `dev` branch.** This document is
versioned with the repo — the matrix for a release is the file at that
release's tag.

Support levels used throughout:

| Level | Meaning |
|---|---|
| **Supported** | Validated on real hardware or in CI; regressions are release blockers; bugs are triaged first. |
| **Preview** | Wired and tested in CI, but real-hardware/quality evidence is incomplete; behavior may change. |
| **Experimental** | Complete and intentionally optional; off by default; enable-at-your-own-risk. |
| **Community-tested** | Works per a community report we link to; not on maintainer hardware. |
| **Research-only** | Exists for the benchmark/ablation pipeline; not part of the product runtime. |
| **Unsupported** | Not wired, not tested, or explicitly rejected; failure is expected and should be clear. |

Nothing below is marked Supported solely because code exists — each
Supported row cites its validation.

## Operating systems

| OS | Arch | Level | Validation |
|---|---|---|---|
| Ubuntu 22.04 / 24.04 | amd64 | Supported | CI install matrix (bootstrap ×2 runs, artifact checks) + maintainer hardware |
| Debian 12 | amd64 | Supported | CI install matrix |
| Rocky Linux 9 (RHEL-like) | amd64 | Supported | CI install matrix; maintainer dev box is EL9 |
| Linux Mint / Pop!_OS (ID_LIKE ubuntu) | amd64 | Preview | Accepted by bootstrap with a warning; not separately tested |
| macOS 14+ (Apple Silicon) | arm64 | Supported | Maintainer-verified on M2 Pro 32GB (hybrid Metal deployment) |
| Linux | arm64 | Preview | llama-vulkan image is built multi-arch on releases; no end-to-end arm64 device validation yet (see § ARM64 scope) |
| Arch / openSUSE / Alpine / NixOS | any | Unsupported | Bootstrap refuses with a clear message |
| Windows (incl. WSL) | any | Unsupported | Untested; no claims made |

## Inference backends

| Backend | Level | Tested device | Validation |
|---|---|---|---|
| CUDA (NVIDIA) | Supported | RTX 5060 Ti 16GB | Primary dev hardware; release smoke tests |
| ROCm (AMD, x86_64) | Community-tested | RX 7900 XTX | [GH #26](https://github.com/itigges22/ATLAS/issues/26) |
| Metal (Apple Silicon hybrid) | Supported | M2 Pro 32GB | Maintainer-verified; native llama-server + Docker services |
| Vulkan (universal) | Preview | lavapipe (CPU ICD) boot path | Smoke-tested; the designated fallback for Intel/others |
| CPU (lavapipe via Vulkan image) | Preview | CI-adjacent smoke | Functional but slow by design |
| Intel SYCL | Unsupported (roadmap) | — | Vulkan is the supported Intel path today |
| Multi-GPU | Unsupported | — | `ATLAS_GPU_INDEX` selects ONE GPU; splitting across GPUs is untested (GH #34 is roadmap) |

## Models (registry)

`lens=supported` means published weights exist; **calibrated** requires
per-model `cx_normalization.json` + `gx_thresholds.json` (see
`atlas lens check`). Quality-validation status is per the acceptance
matrix in `docs/reports/COMPLETION_WIRING_2026-07.md`.

| Registry ID | Level | Lens | ASA | Notes |
|---|---|---|---|---|
| Qwen3.5-9B-Q6_K | Supported | supported (uncalibrated legacy bundle) | supported (A/B-validated May 2026) | Reference model; hash-pinned public download |
| gemma-4-12b-it-Q4_K_M | Preview | supported; calibration **derived + verified** on maintainer hardware (val AUC 0.73, 287 LCB samples) — live lens reports `cx_calibrated: true`. The published HF bundle is still the uncalibrated one; re-publishing the calibrated bundle is a maintainer decision (moderate AUC, shared artifact) | Preview — vector built, published, hash-pinned; **off by default** (no `.model` marker) pending an A/B measurement. Opt in with `atlas asa build`. Not Supported until §9.6 A/B + quality bounds exist | Manual GGUF download (Gemma ToU); artifacts hash-pinned |
| Qwen3.5-9B-Q4_K_M / Q8_0 | Preview | unverified (same-family artifacts, combo unvalidated) | unverified | Hash-pinned public downloads |
| Qwen3.5-7B / 14B / 32B | Preview | no-artifacts | no-artifacts | HF-gated upstream (HF_TOKEN required; no anonymous hash) |
| Bring-your-own GGUF | Preview | Requires `atlas lens build` (per-model bundle) | Requires `atlas asa build` | Direct agent mode works model-agnostically; V3 scoring/steering need the per-model bundle — see § Model contract |
| Frozen Qwen3-14B (74.6% LCB) | Research-only | frozen reference | — | Benchmark provenance only; not a runtime registry entry |

### Reference-model status dimensions

"The lens works" is ambiguous — it can mean the model is served, or raw
scoring is available, or per-model calibration is loaded, or automatic
interventions are firing. These are **separate** dimensions and every
status surface (the proxy `/v1/calibration/status` endpoint, the TUI
badge, `atlas doctor`, `atlas lens check`) reports the same seven so
they cannot disagree — the CLI/TUI/doctor all read the endpoint, which
computes them in one place (`proxy/calibration_status.go`).

| Dimension | Meaning | Statuses |
|---|---|---|
| `model_runtime` | Is the model served and reachable | supported / unreachable |
| `direct_agent` | The agent loop (tools, permissions, sandbox verify) | **supported** always — model-agnostic, independent of lens/ASA |
| `lens_identity` | Cost field matches the served model (identity + dimension) | supported / no-artifacts / dim-mismatch |
| `lens_scoring` | Raw C(x)+G(x) scoring available | supported / partial / disabled |
| `lens_calibration` | Per-model normalization + thresholds loaded | calibrated / uncalibrated / disabled |
| `lens_intervention` | Automatic corrective behavior | active *(only when calibrated)* / neutral / disabled |
| `asa` | Activation-steering vector | supported / unverified / missing |

**Automatic intervention stays neutral or disabled whenever calibration
is absent** — this is enforced in the runtime, not just displayed: the
agent applies thresholds only when `calibratedThresholds()` succeeds
(`proxy/agent.go`), so an uncalibrated or mismatched lens produces
telemetry but never steers with another model's cutoffs.

Reference model (Qwen3.5-9B-Q6_K), current: `model_runtime` supported,
`direct_agent` supported, `lens_identity` supported, `lens_scoring`
supported, `lens_calibration` **uncalibrated** (legacy bundle predates
the calibration files), `lens_intervention` **neutral**, `asa`
supported (A/B-validated). The gemma reference install additionally has
`lens_calibration` calibrated (derived + verified locally) with
`lens_intervention` active and `asa` unverified (marker withheld).

### Model contract

ATLAS is **direct-mode model-agnostic, per-model-bundle for Lens/ASA**:
any llama.cpp-loadable GGUF drives the direct agent loop (grammar
constraints, tools, sandbox verification) with no model-family
assumptions — behavior keys off GGUF metadata and stream shape, never
model names. The differentiating V3 scoring/steering stack requires the
model's own Lens bundle (identity-checked, dimension-checked at load;
mismatched bundles are rejected and the lens reports itself disabled)
and ASA vector (marker-gated at llama-server startup). "Any model, full
stack" is therefore not claimed: full-stack support = registry entry or
locally-built bundle.

## Deployment modes

| Mode | Level | Validation |
|---|---|---|
| Docker Compose (base + backend overlay) | Supported | CI compose validation on every overlay; releases smoke-tested; the deterministic E2E drives the control plane |
| macOS hybrid (native llama + compose) | Supported | Maintainer-verified (M2 Pro) |
| K3s (generated manifests) | Preview | Templates validated + rendered in CI; no automated live-cluster test |
| Bare metal | Preview | Documented (SETUP.md Method 2); manual validation only |
| Offline / air-gapped | Unsupported | Model + artifact downloads require network; no offline bundle exists |
| Rootless Docker / Docker Desktop | Unsupported | Untested; no claims made |

## Sandbox languages

Verification depth: **executed** = code runs via `/execute` with
timeouts/output caps; **syntax** = compile/parse check only.

| Language | Depth | Level |
|---|---|---|
| Python 3 | executed + self-tests | Supported |
| JavaScript / TypeScript (node, tsx) | executed | Supported |
| Go | executed | Supported |
| Rust | executed | Supported |
| C / C++ | executed (gcc/g++) | Supported |
| Bash / sh | executed | Supported |
| HTML / XML / JSON / YAML | syntax | Supported |
| Java / Kotlin / Ruby / PHP | — | Unsupported (roadmap; will ship as separate toolchain images, not in the default sandbox) |

## Feature paths

| Path | Level | Validation |
|---|---|---|
| Direct agent (tools, permissions, sandbox verify) | Supported | Deterministic E2E in CI + unit/contract suites |
| V3 pipeline (probe → candidates → selection) | Supported (control plane) / Preview (per-model quality) | Deterministic V3/Lens E2E in CI; real-model quality validated on the reference model only |
| Lens C(x)/G(x) scoring | Supported (contract) / per-model calibration required for interventions | Identity + dim checks enforced; calibration status surfaced everywhere |
| ASA steering | Supported on Qwen3.5-9B-Q6_K; Preview on gemma (off by default — opt in with `atlas asa build`) | A/B-validated (May 2026) on Qwen; gemma effect unmeasured, so steering is withheld by default rather than shipped unvalidated |
| Call-graph reasoning (#39) | Experimental | `ATLAS_CALL_GRAPH=1`; hermetic tests |
| Host verification (`ATLAS_VERIFY_IN=host`) | Experimental | Explicit opt-in; removes the container backstop |
| Benchmark/ablation stack (`ATLAS_V3_*`, ReASC, lens feedback, ACE) | Research-only | Never read by the product runtime (contract-tested) |
| IDE integration | Unsupported | No extension exists |

## Context lengths

Sized per model + hardware by `atlas tier fit` (KV-cache-aware). The
compose default is 131072 total across 4 slots on a 16 GB card;
macOS-native defaults are smaller (documented in SETUP_MACOS). Any
context a model + VRAM combination can hold is in scope; exceeding VRAM
fails fast at llama-server startup (fit is off).

## Installation methods

| Method | Level |
|---|---|
| `curl \| bash` bootstrap (`scripts/atlas-bootstrap.sh`) | Supported (CI-tested on 4 distros, idempotent, sudo/non-root paths) |
| Manual compose (`cp .env.example .env` + `atlas init`) | Supported |
| `pip install -e .` CLI from checkout | Supported (the only packaged distribution today; no PyPI release) |
| K3s `scripts/install.sh` | Preview |

## Version compatibility policy

- **Supported versions:** the latest release (N) fully; N−1 receives
  security fixes and critical-bug fixes for 90 days after N ships.
- **Registry / artifact bundle schemas:** additive changes only within
  a minor release; consumers ignore unknown fields; identity
  (`model_identity.json`) is mandatory in every bundle from V3.1.2 on.
- **HTTP/SSE protocol:** additive event types are non-breaking (clients
  drop unknown types — the TUI does); field removals or renames require
  a major version and one release of deprecation notice.
- **Python:** ≥3.9 (CLI), 3.11 (service containers), CI runs 3.12.
  **Go:** proxy 1.24+, TUI 1.26+ (GOTOOLCHAIN auto-fetch).
- **Docker:** Engine 24+ with Compose v2. **llama.cpp:** pinned by
  revision in all inference Dockerfiles; bumps go through the CI
  patch-apply gate and a hardware smoke before release.
- **Deprecations:** announced in the changelog one minor release before
  removal; removed config keys are listed in CONFIGURATION.md § removed
  variables and ignored (never fatal) when present in old configs.
