# Production Readiness Tracker

Tracked checklist for the production-grade platform directive.
Statuses: **Done** (evidence cited) · **Partial** (what exists / what's
missing) · **Not started** · **Maintainer** (requires the lead's
hardware, keys, accounts, or a second human) · **Deferred** (explicitly
skipped this pass, on maintainer instruction or as P1/P2).

ATLAS is **not** claimed to be mature production-grade. Per the
directive's own acceptance criteria, several P0 items are organizational
(a second release-capable maintainer) or hardware-bound (real-model
validation per supported entry) and remain open below. This file is the
honest ledger of the distance.

Last updated: 2026-07-02 (dev).

## §1 Define the supported product

| Item | Status | Evidence / gap |
|---|---|---|
| 1.1–1.2 Support classifications + versioned matrix | **Done** | `SUPPORT_MATRIX.md` — every OS/backend/model/language/feature path labeled with validation provenance |
| 1.3 Compatibility policy | **Done** | `SUPPORT_MATRIX.md` § compatibility (N/N-1, schemas, protocol, toolchains, llama.cpp rev policy, deprecation) |
| 1.4 Roadmap vs blockers separation | **Done** | Roadmap items labeled `roadmap` on the tracker; none in the runtime (verified by the completion passes) |

## §2 Repository contradictions

| Item | Status | Evidence / gap |
|---|---|---|
| 2.1 Issue/PR audit | **Done** | Full triage 2026-07-02: labels created + applied to all 14 open issues; #39 closed with evidence; status comments on #66/#115/#27; every open item classified |
| 2.2 Model-agnostic status (#66) | **Done** (contract) | Fresh audit found no name-keyed behavior, probed dims, per-model normalization; contract stated in `SUPPORT_MATRIX.md` + ADR 0003 and enforced at load (identity/dim gates). #66 kept open, narrowed to LLMBackend/MoE/spec-draft |
| 2.3 Redis (#57) | **Done** (decision + hardening) | ADR 0002: keep + harden. Digest-pinned, maxmemory/noeviction, mem_limit, healthcheck-gated, graceful queue 503s, outage behavior documented honestly. Backup guidance in `BACKUP_RESTORE.md` |
| 2.4 Block incomplete model PRs (#126) | **Done** (gate + review) | Registry contract tests (`tests/contracts/test_registry_contract.py`) gate entries in CI; #126 marked blocked with the exact conformance list |
| 2.5 Isolate experimental planning (#124) | **Done** (policy applied) | #124 marked blocked pending its own listed offline eval + measurements; Experimental bar cited from SUPPORT_MATRIX |

## §3 Security boundary hardening

| Item | Status | Evidence / gap |
|---|---|---|
| 3.1 Internal service auth | **Deferred** (maintainer instruction, this pass) | Full implementation map exists (audit: choke points for all ~30 client sites, llama `--api-key`, doctor checks). ADR 0001 records why localhost-only is the current model |
| 3.2 Bind safety | **Partial** | All compose publishes are 127.0.0.1-only (verified); no CI static check yet asserting it |
| 3.3 Sandbox hardening | **Done** (this pass) | Non-root (uid-mapped via init), cap_drop ALL, no-new-privileges, read-only root, sized tmpfs, cpu/mem/pids limits, tree-kill, no docker.sock/privileged/host-PID; K8s securityContext + seccomp RuntimeDefault. Verified on a local hardened-profile run. Remaining: TOCTOU/symlink-race test suite (executor unit tests cover overlay paths; container-level race tests absent) |
| 3.4 Egress modes | **Partial** | `none`-equivalent via `ATLAS_SANDBOX_NET_INTERNAL=true` (internal network; caveats documented) and `unrestricted` (default). `package-registries-only` not implemented (needs an egress proxy — P1). SSRF testing deferred with 3.1 |
| 3.5 Hostile-repo protections | **Partial** | Workspace containment, O_NOFOLLOW writes, snapshot caps, output bounds, binary/size limits exist and are tested; no explicit trust-mode prompt, prompt-injection quality cases only in the (unbuilt) adversarial suite |
| 3.6 Secret redaction | **Deferred** (maintainer instruction, this pass) | Existing: log field-encoding (proxy), `_safe_log`/`_safe_detail` (lens), no-secrets-by-default logging. A shared pattern-based redaction layer was not built |
| 3.7 Safe artifact formats | **Done** | `weights_only=True` torch loads; pickle opt-in only; JSON/GGUF/safetensors preferred; per-file SHA-256 verification; tamper tests (`tests/cli/test_model.py`) |
| 3.8 Fuzzing | **Not started** | (P1 scheduled-CI item) |
| 3.9 Security scans | **Partial** | CodeQL (both languages) enforced; Dependabot (actions/pip/gomod/docker). Missing: container/secret/license scanning jobs |
| 3.10 Security response | **Done** | `SECURITY.md`: severities, targets, embargo/CVE, backports, artifact revocation |

## §4 Supply chain

| Item | Status | Evidence / gap |
|---|---|---|
| 4.1 Dependency pinning | **Partial** | Actions SHA-pinned; Go modules via go.sum; service images pin requirements; llama.cpp rev-pinned + CI-checked; models/artifacts hash-pinned. Missing: full Python lock/constraints files for the CLI |
| 4.2 Base image digests | **Done** | All non-ARG bases digest-pinned with tag comments; ARG'd bases (rocm/vulkan) pinned by default value; Dependabot docker maintains |
| 4.3 SBOMs | **Done** (images) | BuildKit SPDX SBOM attestations on every pushed image. Missing: Python dist SBOM (no PyPI release exists) |
| 4.4 Signing + provenance | **Done** (images) | SLSA provenance (mode=max) + keyless cosign signature per pushed digest; verify command documented. Missing: signed git tags (maintainer GPG), installer-side verification |
| 4.5 Immutable references | **Done** | Immutable `sha-*` tags; test-gated promotion; rollback-by-pin documented (`ROLLBACK.md`) |
| 4.6 Release-asset verification | **Partial** | Images: hash+signature+provenance. No separate release-asset bundle exists to sign |

## §5 API/protocol maturity

| Item | Status | Evidence / gap |
|---|---|---|
| 5.1–5.2 Versioned contracts + machine-readable schemas | **Not started** | Contracts documented (API/PROTOCOL md) + drift-tested, but no version fields or OpenAPI/JSON-Schema files |
| 5.3 Replace regex discovery | **Partial** | Regex contract tests exist and pass (the "keep until replaced" state the directive accepts) |
| 5.4 Error taxonomy | **Not started** | Errors are structured but string-based |
| 5.5 Compatibility tests | **Not started** | (additive-event tolerance exists in clients; untested against N-1) |
| 5.6 Idempotency | **Partial** | Permission delivery keyed by session+call-id (replay-safe 404); artifact installs idempotent (skip-if-present + hash). File writes/turns are not idempotent by design (agent semantics) |
| 5.7 Session contract | **Partial** | Documented in CLI.md (creation/resume/cleanup, per-turn persistence); crash-recovery + concurrent-client semantics informal |

## §6 Reliability and recovery

| Item | Status | Evidence / gap |
|---|---|---|
| 6.1 Liveness/readiness | **Done** | All five services expose /health (+ /ready on proxy/lens with dependency checks); degraded states detailed (lens self-test) |
| 6.2 Startup/shutdown | **Partial** | Healthcheck-gated dependency ordering, tree-kill, port hygiene; no circuit breakers/jittered retries |
| 6.3 Degradation policy | **Done** | ADRs 0004/0005 + OPERATIONS.md: V3 fail-soft (visible), lens 3-state, redis split behavior, sandbox hard-fail; structured events for interventions |
| 6.4 Crash recovery | **Partial** | Atomic artifact/config writes (tmp+rename), resume-safe downloads (.part + hash), stateless containers; no injected-crash test suite |
| 6.5 Atomic updates | **Done** | model.py downloads, identity/calibration writers, bundle activation staging — all tmp+rename with cleanup |
| 6.6 State migrations | **Partial** | Removed-key tolerance + registry additive policy; no schema-version fields |
| 6.7 Backup/restore | **Done** (documented) | `BACKUP_RESTORE.md` — complete state inventory, procedures, honest "manual copies" statement |
| 6.8–6.10 Exhaustion/concurrency/soak | **Partial** | Sandbox limits enforce exhaustion bounds; race detector on all Go tests; concurrent-session E2E and soak/chaos suites absent (P1) |

## §7 Observability

| Item | Status | Evidence / gap |
|---|---|---|
| 7.1 Structured logging | **Partial** | Lens/sandbox structured-ish; proxy uses log.Printf with stable prefixes; no unified JSON schema |
| 7.2 Correlation | **Partial** | session_id + tool_call_id flow proxy↔TUI and into permission/cancel; not propagated into v3/lens/sandbox requests |
| 7.3–7.4 Metrics/tracing | **Not started** | /health counters only (P1) |
| 7.5 Audit log | **Partial** | The event stream + TUI session files record files/commands/permissions per turn; no dedicated redacted-export audit file |
| 7.6 Diagnostic bundle | **Not started** | `atlas doctor` covers live diagnosis; no `diagnostics collect` bundle command |
| 7.7 Actionable errors | **Done** | Verified across installer, doctor, lens/asa/model CLI, bootstrap (failure text names cause + recovery command) |

## §8 Performance and capacity

| Item | Status | Evidence / gap |
|---|---|---|
| 8.1–8.2 Budgets + regression tracking | **Not started** | (P0.16 — needs maintainer hardware baselines) |
| 8.3 Equal-compute benchmark rules | **Done** (policy) | `benchmark/README.md` product-vs-bench contract + ablation-study methodology; equal-budget rules stated there |
| 8.4 Capacity limits | **Done** | Sandbox mem/cpu/pids/output/snapshot caps; context/turn caps; documented in CONFIGURATION.md |
| 8.5 Model profiles | **Partial** | Registry has tier/size/VRAM guidance + `atlas tier fit`; no throughput ranges per entry |

## §9 Real-model validation

| Item | Status | Evidence / gap |
|---|---|---|
| 9.1 Per-model validation | **Maintainer** | Reference model (Qwen3.5-9B-Q6_K) validated on dev hardware; gemma Preview; harness exists (`atlas bench`, integration suite) — running it per entry is hardware work |
| 9.2 Quality regression suite | **Not started** | (P1.11-adjacent) |
| 9.3 Repo-level benchmark | **Not started** | (P1.11) |
| 9.4 Ablations | **Partial** | 14B 4-condition study published; 9B 6-condition scripted (`run_v31_ablation.sh`) awaiting the maintainer's run |
| 9.5 Lens provenance | **Partial** | Bundles carry identity/hashes/thresholds; full provenance manifest (dataset, seed, hyperparams, training commit) not embedded |
| 9.6 ASA validation | **Partial** | Qwen vector A/B-validated; gemma = Preview (vector published + hash-pinned; runtime marker unset — `atlas asa build` or marker write pending, maintainer's call). SUPPORT_MATRIX reflects this |
| 9.7 Claim governance | **Done** | README pins the 74.6% claim to the frozen build + methodology link; completion report records commit/hardware/run evidence for CI claims |

## §10–11 Packaging, deployment

| Item | Status | Evidence / gap |
|---|---|---|
| 10.1 Atomic installer | **Partial** | Preflight, staged+hash-verified downloads, health wait, idempotent re-run, clear failures (CI-tested ×4 distros ×2 runs). No transactional commit/rollback of a failed install |
| 10.2–10.3 upgrade/rollback commands | **Done** (procedures) / **Not started** (commands) | `UPGRADE.md`/`ROLLBACK.md` document the supported manual procedures incl. immutable-tag pinning; no `atlas upgrade/rollback` binaries |
| 10.4 Uninstall | **Done** | SETUP.md § Uninstalling (compose) + `uninstall.sh` (K3s, preserves-data flag, fixed image list) |
| 10.5 Offline | **Done** (explicit rejection) | SUPPORT_MATRIX: Unsupported, stated |
| 10.6 Real install matrix | **Partial** | CI dry-runs 4 distros (real bootstrap, skips docker/GPU); full clean-machine installs are maintainer/hardware work |
| 10.7 PyPI release | **N/A** | Not published; checkout install is the supported path (stated) |
| 11.1 Non-root containers | **Partial** | Sandbox non-root (this pass). proxy/v3/lens/llama still root — llama needs GPU device access; others are candidates (P1) |
| 11.2 Minimal images | **Partial** | Multi-stage proxy + llama; python-slim services; sandbox intentionally carries toolchains |
| 11.3 Resource limits | **Done** (sandbox+redis) / defaults elsewhere are deliberate |
| 11.4 Compose profiles | **Done** | Overlay-per-backend, all CI-validated |
| 11.5 Deployment failure tests | **Partial** | Compose validation + install idempotency; port-collision/stale-state tests absent |
| 11.6 Kubernetes | **Done** (decision) | K3s = Preview in SUPPORT_MATRIX (templates validated, no live-cluster CI); hardened securityContext this pass |

## §12 CI/CD

| Item | Status | Evidence / gap |
|---|---|---|
| 12.1 Branch protection | **Maintainer** | Needs repo-admin action; recommended config documented in GOVERNANCE.md flow (PRs to dev, ff promotion) |
| 12.2 CODEOWNERS | **Done** | `.github/CODEOWNERS` |
| 12.3 Required jobs | **Partial** | Have: go test/race/vet, py tests, lens tests, contracts, both E2Es, install matrix, compose, images, shellcheck, ruff, CodeQL, yamllint, patch-apply. Missing: staticcheck, type checking, scans (container/secret/license), SBOM validation, fuzz smoke, repro check, link checker |
| 12.4 Static typing | **Not started** | |
| 12.5 Coverage tracking | **Partial** | Race detector everywhere; measured coverage (proxy 42.8%/tui 48.5% at last audit) not gated |
| 12.6 Scheduled CI | **Partial** | CodeQL weekly; no scheduled scans/fuzz/soak |
| 12.7 Release channels | **Done** | dev/sha → semver+latest promotion of the same immutable digests |
| 12.8 Release checklist | **Done** | docs/RELEASE.md + SECURITY release procedure |
| 12.9 Reproducibility | **Not started** | (P1.5) |
| 12.10 Release recovery | **Partial** | Two-phase promote proved itself on a real failed build (llama templates break: sha-* published, no tag moved); revocation documented; failed-migration tests absent |

## §13 Registry/artifacts

| Item | Status | Evidence / gap |
|---|---|---|
| 13.1 Registry schema | **Partial** | Typed dataclass + contract tests + honest status enums; no schema-version field |
| 13.2 Transactional activation | **Done** | Staging download → size/hash verify → identity/dim validation → self-test → atomic replace (bundle activator + superseded-format purge). Missing: keep-previous-for-rollback (documented gap in ROLLBACK.md) |
| 13.3 Revocation | **Done** (mechanism) | Hash-pin poisoning + SECURITY.md procedure; doctor/`model verify` detect non-matching files |
| 13.4 Registry availability | **Done** | Registry is IN the package (no remote registry to be down); artifacts fail with clear errors + cached files keep working |
| 13.5 Community artifacts (#102) | **Done** (correctly deferred) | Labeled roadmap+security; prerequisites listed in the issue history |

## §14 Configuration

| Item | Status | Evidence / gap |
|---|---|---|
| 14.1 Precedence | **Done** | env > .env > defaults, documented + exercised (conftest hides .env; bench/doctor resolve chains tested) |
| 14.2 Config schema | **Partial** | Contract tests assert key↔reader↔doc parity; no typed validation of values |
| 14.3 Config migration | **Done** (policy) | Additive keys, ignored-removed-keys (§8.12), legacy fallbacks in compose |
| 14.4 Secret storage | **Done** | `secrets/api-keys.json` 0600/0700, refused when loose; HF tokens env-only |
| 14.5 Doctor as contract | **Done** (scope) | Verifies install integrity, artifacts+identity+hashes, calibration, ASA, backends, disk, services; auth/bind checks deferred with 3.1 |

## §15–17 Model/backend, sandbox languages, UX

| Item | Status |
|---|---|
| 15.1–15.2 Model contract + fail-closed | **Done** (ADR 0003; identity/dim/marker gates) |
| 15.3 Capability discovery | **Partial** (probes /props, /v1/models, embedding dim, PC-202 patch; grammar/spec-decode support assumed from pinned rev) |
| 15.4 Model switching | **Partial** (documented recreate flow + identity re-probe on reload; no quiesce/stress tests) |
| 15.5 Multi-GPU | **Done** (explicitly Unsupported; #34 labeled roadmap) |
| 15.6 ARM64 | **Done** (scope published in SUPPORT_MATRIX + #115 status comment) |
| 16.1–16.3 Language matrix / lean image / build safety | **Done** / **Done** (policy in SUPPORT_MATRIX: new languages = separate images) / **Partial** (verified flows for pip/npm/go/cargo; abuse tests deferred with 3.x) |
| 17.1 Diff review | **Partial** (TUI /review + per-file verdicts + diff previews on edits; no pre-completion consolidated diff gate) |
| 17.2 Git safety | **Done** (never commits without explicit /commit; works in non-git dirs) |
| 17.3 Permission modes | **Done** (documented per-tool in API.md; fail-closed default; timeout deny) |
| 17.4 Cancellation | **Done** (cancel aborts LLM/V3/sandbox trees + pending permission; E2E + unit pinned) |
| 17.5 Terminal accessibility | **Partial** (pipe-mode no-TUI path exists + tested; narrow/no-color matrices untested) |
| 17.6 Privacy | **Done** (zero telemetry, external calls enumerable: HF downloads + model fetches only) |

## §18–20 Docs, governance, licensing

| Item | Status |
|---|---|
| 18.1 Doc set | **Done** — all 23 listed docs exist (support matrix, upgrade, rollback, backup, operations, governance, maintainers added this pass) |
| 18.2 Executable docs | **Partial** (contract tests check config keys/commands; link checker absent) |
| 18.3 Runbooks | **Done** (OPERATIONS.md runbook table + TROUBLESHOOTING.md) |
| 18.4 ADRs | **Done** (docs/adr/ 0001–0006) |
| 19.1 Governance files | **Done** (GOVERNANCE, MAINTAINERS, CODEOWNERS) |
| 19.2 Bus factor | **Maintainer** — honestly documented as 1; open seats listed; cannot be closed by tooling |
| 19.3 PR requirements | **Done** (PR template) |
| 19.4 Issue hygiene | **Done** (labels created+applied; milestones still unused) |
| 19.5 Support policy | **Done** (SUPPORT_MATRIX + SECURITY targets + MAINTAINERS channel) |
| 20.1 Third-party notices | **Done** (`THIRD_PARTY_NOTICES.md`) |
| 20.2 License CI | **Not started** |
| 20.3 AGPL documentation | **Partial** (license stated; obligations explainer not written) |
| 20.4 Data rights | **Partial** (benchmark/dataset terms noted in notices; per-artifact dataset provenance embedded is §9.5's gap) |

## §21 Test matrix

Covered today: unit (1,400+ Python, 500+ Go w/ race), contract (events/
config/CLI/registry), integration (proxy↔v3↔lens↔sandbox via the V3/Lens
E2E; registry↔downloader via live install-artifacts tests), E2E (direct,
V3/Lens, permission approve/deny, session-less deny, install matrix),
security-adjacent (traversal, containment, tamper, timeout, tree-kill).
Absent: fuzzing, soak/chaos, performance, per-model real-model suites,
upgrade/rollback E2E, offline install (unsupported).

## P0 scoreboard

| # | P0 item | Status |
|---|---|---|
| 1 | Support/compat matrix | **Done** |
| 2 | Model-agnostic resolution | **Done** |
| 3 | Redis topology | **Done** |
| 4 | Block incomplete model PRs | **Done** |
| 5 | Internal-service auth | **Deferred** (maintainer instruction; design complete) |
| 6 | Sandbox hardening | **Done** (core set; race-test suite outstanding) |
| 7 | Secret redaction | **Deferred** (maintainer instruction) |
| 8 | Safe signed artifact manifests | **Partial** (hash-pinned + verified; no signature layer on HF artifacts) |
| 9 | Digest pinning | **Done** |
| 10 | SBOM/signing/provenance/scanning | **Done** (images) / scans partial |
| 11 | API/schema versioning | **Not started** |
| 12 | Upgrade/rollback/migration | **Partial** (procedures done; commands absent) |
| 13 | Structured logging/diagnostics | **Partial** |
| 14 | Real-model validation per model | **Maintainer** |
| 15 | ASA resolution | **Partial** (Qwen validated; gemma Preview pending marker) |
| 16 | Performance budgets | **Not started** |
| 17 | Concurrency/soak/chaos/recovery | **Partial** |
| 18 | Branch protection + CODEOWNERS | CODEOWNERS **Done**; protection **Maintainer** |
| 19 | Governance docs | **Done** |
| 20 | Signed immutable releases | **Done** (image pipeline) |
