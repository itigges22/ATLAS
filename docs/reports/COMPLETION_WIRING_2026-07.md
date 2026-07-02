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
  hard-fails if the acceptance test skips.
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
python -m pytest tests geometric-lens/tests       # 1396 passed
cd proxy && go vet ./... && go test -race ./...   # ok
cd tui   && go vet ./... && go test -race ./...   # ok
cd proxy && go build -o /tmp/test-atlas-proxy .
pip install -r sandbox/requirements-runtime.txt
python -m pytest tests/e2e -v                     # 2 passed (1.7s)
atlas model install-artifacts gemma-4-12b-it-Q4_K_M --models-dir <tmp>  # 6/6 sha256 verified
curl -s localhost:8099/health                     # lens self_test_pass: true
```

## Final Statement

- All advertised features are wired: **yes** — every documented tool,
  endpoint, event, command, and config key has a producer, consumer,
  and reader, or has been removed from code and docs.
- All production paths are tested: **yes** at the unit/contract level
  and via the E2E acceptance test; GPU-inference behavior itself is
  validated by the hardware-gated integration suite, not CI.
- All placeholders removed: **yes** — no production code path remains
  whose purpose is "wired in later".
- All incomplete abstractions removed: **yes** (verified caller-less
  before each deletion).
- The end-to-end test passes: **yes**, locally and as a required CI
  job.
- Ready to merge: **yes** — the full suite, static checks, builds,
  compose validation, and the acceptance test pass on `dev`.
