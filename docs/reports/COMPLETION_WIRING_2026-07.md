# Completion & Wiring Pass — July 2026

Repository-wide pass making the `dev` branch internally complete: every
feature fully wired, tested, and documented, or removed. Commits
`0c84cca..1729ec2` (plus the same-day CI/security hardening wave
`6a1f896..ea9bda3`).

## Completed Wiring

| Previously unwired | Completion |
|---|---|
| `v3_reasoning_token` emitted by the proxy, consumed by nothing | Rendered in the TUI's in-place V3 streaming row (`tui/model.go`) with the `‹thinking›` prefix |
| Session-less `/v1/agent` requests silently bypassed permission prompts (yolo-equivalent in `default` mode) | Fail closed: denied with an actionable log; API.md documents the contract and the non-TUI client guide covers `/v1/permission` |
| TUI permission modal survived proxy-side denies/turn end, gating input on a dead prompt | Cleared on `permission_denied` and `__turn_done__` |
| Lens retrains never wrote `model_identity.json`; the load path hard-requires it, so every retrained bundle disabled the whole lens on the next restart (the live deployment was in this state) | Both retrain flows stamp the served model's identity; the live gemma bundle is stamped and verified loading (C(x)+G(x), self-test pass) |
| Published lens bundles (HF) lacked identity files — downloadable but never loadable | Identity files published to both HF locations, pinned in `lens_artifact_files`/`lens_artifact_sha256`; verified end-to-end via live `atlas model install-artifacts` |
| gemma registry entry had no artifact URL bases | Lens + ASA url bases and per-file SHA-256 pins added; all six files download and hash-verify |
| Documented knobs unreachable in container deployments (`ATLAS_PLAN_THINKING`, `ATLAS_SHELL_SNAPSHOT_*`, `ATLAS_CONTROL_VECTOR_*`) | Compose passthroughs added |
| `scripts/build-containers.sh` built from a removed directory layout (1 of 5 images, names no manifest pulls) | Builds the five current services from their real contexts, tagged exactly as the K3s manifests reference |
| `scripts/uninstall.sh` aborted on an unset `$IMAGES` under `set -u` | Explicit image list, GHCR-style names |
| ASA marker check case-sensitive at boot but casefolded in `atlas asa check` | Case-insensitive in both launchers |
| `atlas publish --dry-run` without repos crashed on a `None` argv | Repos passed only when set |
| `train` extra missing xgboost/scikit-learn (lens build step 5 hard-requires them) | Added; ImportError guidance mentions `pip install 'atlas[train]'` |
| Five consumed keys absent from `.env.example`; four vars read in code documented nowhere | `.env.example` + CONFIGURATION.md aligned (`ATLAS_DEDUP_READS`, `ATLAS_MAX_TURNS`, `ATLAS_VERIFY_IN`, `ATLAS_LENS_HOST_DIR`, `ATLAS_LENS_MODELS`; `ATLAS_GPU_VENDOR` moved to the table where it's actually read; `ATLAS_UBUNTU_TAG`, `ATLAS_GO_VERSION`, `ATLAS_SECRETS_DIR` documented) |
| `model_recommendations` shim awaiting "a future cleanup" | Callers migrated to `model_registry`; shim deleted |

## Removed Incomplete Features

All verified caller-less before deletion (grep + build + full test suite):

- **`plan_tasks`** — model-visible tool that acknowledged tasks as
  "pending" without executing them; its parallel executor
  (`proxy/parallel.go`, 226 lines) was never wired in. Tool, types,
  grammar case, exclusion plumbing, docs rows (en/ja/ko/zh) removed;
  regression tests assert its absence.
- **`PermissionRule` rules engine** (`checkPermissionRules`,
  `matchPattern`, `PermissionConfig`) — nothing loaded or evaluated
  rules; the live machinery is `needsPermission` + `awaitPermission` +
  the built-in safety deny-list.
- **Metric-tensor G(x) path** — `evaluate_gx` (the path proxy /
  v3-service / benchmark call) is XGBoost-only; the metric tensor served
  only `/internal/lens/correctability`, an endpoint with no caller.
  Endpoint, `evaluate_correctability`, `metric_tensor.py`,
  `correction.py`, and the 67 MB `metric_tensor.pt` registry
  requirement removed (stale copies still purged by the bundle
  activator's superseded-formats list).
- **Dead Go code** — `build_verify.go`, `v3_adapter.go`,
  `buildToolCallSchemaJSONForTools`, `buildGBNFGrammar`,
  `buildToolDescriptions`, `EmitSimple`, `calibrationTooltip`,
  `firstNonEmptyLine`, `TierUsesV3`, builtin-shadowing `min`.
- **v3-service `_emit_event`** dual-emit envelope helper — never called
  by any handler; PROTOCOL.md / events.py now describe only the live
  protocol (including correcting the "done is always last" claim).
- **V3.0-era inference files** — `Dockerfile.mtp`, `entrypoint-mtp.sh`,
  `entrypoint-v3-specdec.sh`, `entrypoint-v3.1-9b.sh`,
  `entrypoint-embed.sh`, both custom jinja templates, and the malformed
  unused `fix-embeddings-spec-decode.patch` (the Dockerfile `sed` is the
  patch; CI verifies its target line).
- **Zero-reference scripts** — `run_full_benchmarks.sh`,
  `validate_benchmarks.py`, `smoke-test-9b.sh`, `deploy-9b.sh`,
  `measure_bok_latency.sh`; `router/fallback_chain.py`.
- **Stubs & no-ops** — the `/ablation` "coming soon" REPL command (and
  its help row), the benchmark `--runs` flag (parsed, passed, ignored),
  `ATLAS_ENABLE_TRAINING` (reserved, no reader), `ATLAS_REGISTRY`
  (superseded by manifest-aligned tagging).

## Canonical Runtime Paths

One per platform, all through `scripts/atlas-bootstrap.sh` (or
`atlas init` on an existing checkout) → `docker compose` with the
backend overlay → `atlas` (TUI):

| Platform | Overlay | Backend selection |
|---|---|---|
| Linux + NVIDIA CUDA | base `docker-compose.yml` | `ATLAS_BACKEND=cuda` (auto-detected) |
| Linux + AMD ROCm | `-f docker-compose.rocm.yml` | `ATLAS_BACKEND=rocm`, x86_64 only |
| Linux + Vulkan | `-f docker-compose.vulkan.yml` | universal fallback |
| CPU-only | vulkan + `-f docker-compose.cpu.yml` | lavapipe ICD |
| Apple Silicon | `-f docker-compose.macos.yml` + native `scripts/atlas-llama-macos.sh` | Metal hybrid |

Every step of the 22-point checklist (install → detect → select →
download+verify → configure → start → health → session → read → tool
call → permission → edit → sandbox verify → stream → complete) is
either exercised by CI (install matrix dry-runs the bootstrap on four
distros twice; the E2E acceptance test drives session→done through the
real protocol; compose overlays validate; artifact downloads
hash-verify) or documented in SETUP.md/SETUP_MACOS.md with tested
commands. Model/lens/ASA downloads are SHA-256-verified on both the
Python and shell paths.

## Test Coverage Added

- **`tests/e2e/test_acceptance.py`** (mandatory acceptance test): real
  proxy binary + real sandbox executor (host uvicorn) + scripted fake
  llama-server; one full agent turn over the production SSE protocol
  with ordered-stage assertions, exactly-one-permission-prompt, final
  file contents, and the sandbox side-effect. Plus the fail-closed
  session-less deny test.
- `proxy/permission_gate_test.go` — session-less deny contract
  (replaces the fail-open pin).
- `proxy/llm_failure_test.go`, `proxy/fuzzy_edit_test.go`,
  `proxy/v3_bridge_test.go` (generate path), `tests/cli/test_bench.py`,
  `proxy/permissions_rules_test.go` (needsPermission) — from the
  same-day hardening wave.
- Fixtures updated for the identity-bearing bundle contract.

## CI Gates Added

- `e2e-acceptance` job — builds the proxy, boots the control plane,
  hard-fails if any acceptance test skips. Covers the direct agent
  path (test_acceptance.py) AND the differentiated V3/Lens pipeline
  (test_v3_lens_acceptance.py: real v3-service, real proxy→V3 bridge,
  fake deterministic llama + lens, real sandbox — asserting V3 is not
  bypassed, lens scoring is called, winner selection picks the
  lens-preferred candidate, plus V3-unreachable/malformed/timeout and
  lens-outage failure modes).
- `tests/contracts/` drift gates — proxy↔TUI event producer/consumer
  parity, envelope-type parity across Go/Python, config keys vs
  readers vs docs, CLI subcommand implementations, and registry
  hash/consumption contracts.
- (Same-day wave) two-phase image publish gated on the tests workflow;
  SHA-pinned actions; `geometric-lens/tests` + static infrastructure
  tests + `test-integrity`/`python-compile` gates; shellcheck over all
  of `scripts/`; PR-time Dockerfile builds; Dependabot across
  actions/pip/gomod/docker.

## Remaining Experimental Features (complete, intentionally optional)

| Feature | Enable | Tested | Why not default |
|---|---|---|---|
| Call-graph reasoning (#39) | `ATLAS_CALL_GRAPH=1` | `tests/v3-service/test_graph_*` | Measured win is model/workload-dependent; adds indexing latency |
| Loose grammar mode | `ATLAS_GRAMMAR_MODE=loose` | grammar tests cover both modes | Strict schema-GBNF is the correct default; loose is the documented escape hatch (required for Gemma) |
| Host verification | `ATLAS_VERIFY_IN=host` | E2E covers sandbox path; host path unit-tested | Drops the container backstop — explicit opt-in |
| ASA scale/layer overrides | `ATLAS_CONTROL_VECTOR_*` | entrypoint gate logic exercised by install tests | Default 0.5 scale is the validated operating point |
| Benchmark ablation knobs | `ATLAS_V3_*` in `atlas.conf` | ablation runner only | Explicitly research-only; documented as not read by the product path |

## Known Limitations (external/product constraints)

- The 7B/14B/32B registry entries are HF-gated upstream: installable
  with `HF_TOKEN`, but anonymous SHA-256 capture is impossible, so their
  gguf hashes are absent by construction (documented in the registry).
- `cx_normalization.json` for the live gemma bundle requires a retrain
  to derive (it is training-statistics-derived, not constructible);
  until then C(x) normalized scores are neutral — explicitly surfaced
  as `cx_calibrated: false`.
- The sandbox has open egress and is memory-uncapped before `atlas
  init` writes `ATLAS_SANDBOX_MEM` — stated in SECURITY.md rather than
  silently claimed otherwise.
- Headline 9B/gemma benchmark numbers remain unpublished; README pins
  the 74.6% figure to the frozen 14B reference build.

## Verification Commands

```
python scripts/production-readiness.py            # 15/15 gates
python -m pytest tests geometric-lens/tests       # 1416 passed
cd proxy && go vet ./... && go test -race ./...   # ok
cd tui   && go vet ./... && go test -race ./...   # ok
cd proxy && go build -o /tmp/test-atlas-proxy .
pip install -r sandbox/requirements-runtime.txt
python -m pytest tests/e2e -v                     # 7 passed (~26s)
atlas model install-artifacts gemma-4-12b-it-Q4_K_M --models-dir <tmp>  # 6/6 sha256 verified
curl -s localhost:8099/health                     # lens self_test_pass: true
```

## Scope of Validation

**Verified in standard (GitHub-hosted, deterministic) CI:** the direct
agent control plane, the proxy→V3 bridge and the full V3 pipeline
orchestration (real v3-service), the Lens scoring contract and winner
selection (fake deterministic lens proving the calls and the choice),
permission flow, sandbox execution, TUI/CLI/proxy unit suites,
producer/consumer contract gates, configuration, compose, packaging,
and static validation.

**Verified outside standard CI (hardware-gated / manual):** CUDA, ROCm,
Metal, and Vulkan inference; hidden-state extraction from real models
(PC-202 patch application is CI-checked, behavior is not); ASA steering
effect on real models; real-model V3/Lens quality behavior; performance.
The cards and validation provenance are listed in SETUP.md's hardware
table; the hardware-gated pytest suites are `tests/infrastructure/`
(`integration` marker).

**Not yet verified:** — (empty for supported software paths; hardware
coverage is a support-matrix limitation, not unfinished implementation).

## Acceptance Matrix

| Area | Deterministic CI | Hardware-gated | Manual | Status |
|---|---:|---:|---:|---|
| Direct agent control plane | Yes | No | No | Pass |
| V3 bridge (proxy→v3-service) | Yes | No | No | Pass |
| V3 pipeline orchestration | Yes | No | No | Pass |
| Lens scoring contract | Yes | No | No | Pass |
| Winner selection | Yes | No | No | Pass |
| Permission flow | Yes | No | No | Pass |
| Sandbox execution | Yes | No | No | Pass |
| Event/config/CLI/registry contracts | Yes | No | No | Pass |
| Install bootstrap (4 distros) | Yes | No | No | Pass |
| Image builds + gated publish | Yes | No | No | Pass |
| CUDA inference | No | Yes | Yes | Pass (RTX 5060 Ti — primary dev box) |
| ROCm inference | No | Yes | Yes | Pass (RX 7900 XTX — community, GH #26) |
| Metal inference | No | Yes | Yes | Pass (M2 Pro — maintainer-verified) |
| Vulkan/CPU inference | No | Yes | Yes | Smoke-tested (lavapipe boot path) |
| Real hidden-state Lens | No | Yes | Yes | Pass on dev box (gemma bundle, self-test) |
| Real ASA steering | No | Yes | Yes | Validated on Qwen3.5-9B (A/B, May 2026); gemma vector built, marker pending |

## Final Statement

- All advertised features are wired: **yes** — every documented tool,
  endpoint, event, command, and config key has a producer, consumer,
  and reader, or has been removed from code and docs.
- All critical control-plane paths — including the differentiated
  V3/Lens pipeline — are covered by deterministic CI acceptance
  testing. Real llama.cpp inference, GPU backend behavior,
  hidden-state extraction, ASA steering, and model-dependent V3/Lens
  quality remain hardware-gated or manually validated (see the
  Acceptance Matrix).
- All placeholders removed: **yes** — no production code path remains
  whose purpose is "wired in later".
- All incomplete abstractions removed: **yes** (verified caller-less
  before each deletion).
- Both end-to-end acceptance tests pass, locally and as required CI
  jobs; contract drift gates run in the CI pytest matrix.
- Ready to merge: **yes**, with the scope above — the full suite,
  static checks, builds, compose validation, both acceptance tests,
  and the contract gates pass on `dev` (verified GitHub Actions runs
  recorded below).

## Verified CI Runs (dev code head `e93cfdd`)

| Workflow | Conclusion | Run |
|---|---|---|
| tests (incl. both e2e acceptance jobs + contract gates) | success | [28618554574](https://github.com/itigges22/ATLAS/actions/runs/28618554574) |
| install matrix (4 distros × 2 runs + sudo path) | success | [28618554576](https://github.com/itigges22/ATLAS/actions/runs/28618554576) |
| Build & publish container images (6 images + gated promote) | success | [28618554644](https://github.com/itigges22/ATLAS/actions/runs/28618554644) |
| codeql (python + go) | success | [28618554573](https://github.com/itigges22/ATLAS/actions/runs/28618554573) |

Within the tests run: `e2e acceptance (proxy + sandbox + fake llama)`
— success (covers both `test_acceptance.py` and
`test_v3_lens_acceptance.py`; the job hard-fails on any skip) — and
`pytest (tests/contracts)` — success. Skipped jobs: the `PR build
check` and `promote` steps skip by design on the event types that
don't use them. Allowed failures: none. Hardware-gated exclusions:
`tests/infrastructure/test_llm.py` + `test_sandbox.py` (`integration`
marker — need a live model stack; see Scope of Validation). Commits
after `e93cfdd` on this branch are documentation-only.
