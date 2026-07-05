# Production Readiness Tracker

Tracked checklist for the production-grade platform program. Each
repository item records owner, required changes, tests, CI gate,
documentation, and evidence. Status vocabulary:

- **Done** — implemented, tested, and evidenced.
- **In progress** — actively underway this program.
- **Repository-maintenance item** — remaining code/docs work that can be
  done entirely in-repo (not yet started or partial).
- **External blocker** — requires the maintainer's hardware, keys,
  accounts, or a second person; cannot be closed by code.
- **Roadmap** — deliberately deferred feature work (labeled `roadmap`).
- **Unsupported** — explicitly out of scope.

ATLAS is **not** claimed to be mature production-grade — that bar
includes real-model validation across the support matrix and a second
release-capable maintainer, both external. But every repository-
controlled P0 item (1–17) is now implemented, tested, and CI-gated; the
only open items are the genuinely external ones below. This file is the
honest ledger.

Last updated: 2026-07-05 (dev) — P0.7–16 landed; documented remainders closed (OpenAPI, precedence resolver, JSON logs + correlation IDs, license/mypy/container-scan).

---

## Completed this program (Done)

### P0.1 — Local service authentication
- **Owner:** repo. **Status:** Done.
- **Changes:** per-installation token (`secrets/service-token`, 0600,
  `atlas init` / `--rotate-token`); enforcement middleware in proxy, v3,
  lens (`/internal/*`), sandbox, and llama (`--api-key-file`); outbound
  injection at every client choke point; `/health`+`/ready` stay open.
- **Tests:** `proxy/auth_test.go`, `tests/cli/test_service_token.py`,
  `tests/e2e/test_service_auth.py`; the whole E2E suite runs with auth
  **enabled** session-wide.
- **CI gate:** go test + `tests/e2e` + `tests/cli` (tests workflow).
- **Docs:** SECURITY.md, ADR 0001, CONFIGURATION.md, API.md.
- **Evidence:** commit `8ed97aa`; token value never in `.env`/argv/logs/
  401 bodies (asserted by tests).

### P0.2 — Local-only binding validation
- **Owner:** repo. **Status:** Done.
- **Changes:** contract test asserting every compose publish binds
  loopback; K3s templates reject hostNetwork/LoadBalancer; macOS
  launcher defaults to 127.0.0.1.
- **Tests/CI:** `tests/contracts/test_binding_contract.py`.
- **Docs:** SECURITY.md. **Evidence:** commit `eadc455`.

### P0.3 — Private-value filtering + sensitive-file exclusion
- **Owner:** repo. **Status:** Done.
- **Changes:** shared filter masks credential-shaped values before log
  serialization in all four services; agent read tools refuse credential
  files by default with `ATLAS_ALLOW_CREDENTIAL_READS=1` override.
- **Tests/CI:** `proxy/private_values_test.go`,
  `proxy/credential_read_test.go`,
  `tests/contracts/test_private_value_filtering.py` (synthetic corpus;
  three Python copies asserted byte-identical).
- **Docs:** SECURITY.md, CONFIGURATION.md. **Evidence:** commit `eadc455`.

### P0.4 — Reproducible dependency builds
- **Owner:** repo. **Status:** Done.
- **Changes:** pinned all production Python deps (`geometric-lens/`,
  `v3-service/`, `sandbox/` requirements); pinned proxy apk; digest-
  pinned bases.
- **Tests/CI:** `tests/infrastructure/test_dependency_pinning.py` reports
  any unpinned production dependency.
- **Docs:** docs/CONTAINER_PACKAGING.md. **Evidence:** commit `bec692d`;
  images rebuilt + healthy.

### P0.5 — Proxy image simplification
- **Owner:** repo. **Status:** Done.
- **Changes:** removed the unused language toolchain; kept curl+bash;
  verification still routes to the sandbox `/shell` with no local
  fallback. Image 516MB → 40MB.
- **Tests/CI:** dependency-pinning + E2E (run_command routing unchanged).
- **Docs:** docs/CONTAINER_PACKAGING.md, Dockerfile comment.
- **Evidence:** commit `bec692d`; rebuilt image confirmed toolchain-free.

### P0.6 — Non-root proxy / v3 / lens
- **Owner:** repo. **Status:** Done.
- **Changes:** named in-image accounts (atlas/appuser/lens, uid 1001);
  writable dirs scoped; `PYTHONDONTWRITEBYTECODE`.
- **Tests/CI:** service start + health/readiness; uid confirmed live.
- **Docs:** docs/CONTAINER_PACKAGING.md (writable-dir table + lens-volume
  migration note). **Evidence:** commit `bec692d`; all four containers
  run non-root, healthy.

### P0.17 — Reference-model status clarity (seven dimensions)
- **Owner:** repo. **Status:** Done.
- **Changes:** canonical seven-dimension status computed once
  (`proxy/calibration_status.go`), rendered by TUI + `atlas doctor`
  (shared endpoint), documented in SUPPORT_MATRIX; intervention reported
  neutral/disabled unless calibrated (matches enforced runtime).
- **Tests/CI:** `proxy/status_dimensions_test.go`,
  `tests/contracts/test_status_dimensions_contract.py`.
- **Docs:** SUPPORT_MATRIX § Reference-model status dimensions.
- **Evidence:** commit `e7f3f54`; verified against the live endpoint.

### Production-platform foundations (earlier in this program)
- **Support/compatibility matrix** — Done (`SUPPORT_MATRIX.md`).
- **Supply chain** — Done: digest-pinned bases, SLSA provenance + SPDX
  SBOM attestations, keyless cosign signatures per pushed digest
  (verified against the live registry).
- **Sandbox hardening** — Done: non-root, cap_drop ALL, read-only root,
  cpu/mem/pids caps, optional egress cutoff; K3s securityContext.
- **Redis decision** — Done (ADR 0002): keep + harden (maxmemory/
  noeviction, mem_limit, graceful queue 503s).
- **Governance** — Done: GOVERNANCE, MAINTAINERS, CODEOWNERS, SECURITY
  response process, THIRD_PARTY_NOTICES, six ADRs, UPGRADE/ROLLBACK/
  BACKUP_RESTORE/OPERATIONS runbooks.
- **Model-agnostic contract (#66)** — Done: audited, stated in
  SUPPORT_MATRIX + ADR 0003, enforced at load.
- **Issue/PR hygiene** — Done: label vocabulary applied; #39 closed;
  #124/#126/#128 blocked with conformance lists.

---

## Repository-maintenance items (P0.7–16) — completed

All landed this program with tests + docs; see per-item commits.

| # | Item | Status | Evidence |
|---|---|---|---|
| P0.7 | Versioned API + schemas + error taxonomy | **Done** | `/version`, 12-code taxonomy, OpenAPI 3.1 + route-parity contract |
| P0.8 | Typed config + validation + migration | **Done** | typed schema, `atlas config validate/migrate` (+ `--dry-run` preview, `.env.bak`), precedence-aware resolver |
| P0.9 | Automated upgrade / rollback | **Done** | `atlas upgrade`/`rollback`, auto-restore on failure, cosign signature verification before apply |
| P0.10 | Signed artifact manifests + rollback | **Done** | sign/verify (signature + per-file hash), `atlas artifact verify/snapshot/rollback` |
| P0.11 | Structured logging + diagnostics | **Done** | `atlas diagnostics collect`, JSON logs + cross-service correlation IDs |
| P0.12 | Dependency/repo quality CI | **Done** | staticcheck, secret scan, license check, scoped mypy, container scan, SBOM+provenance attestation verify |
| P0.13 | Local boundary regression tests | **Done** | 23-case matrix (symlink/TOCTOU/limits) + trust modes + conservative sandbox defaults |
| P0.14 | Concurrency / recovery tests | **Done** | deterministic suite; caught + fixed an atomic-write race |
| P0.15 | Performance harness + budgets | **Done** | versioned result format + deterministic budget gate wired into CI |
| P0.16 | Lens provenance manifest | **Done** | auto-written on retrain, completeness-gates Supported |

---

## External blockers (cannot be closed by code)

Only these require the maintainer, dedicated hardware, or another person.

| Item | Why external | Current state |
|---|---|---|
| **Signed Git tags (Verified badge)** | Registering the key on the maintainer's GitHub account | SSH tag signing is IMPLEMENTED (scripts/release-tag.sh, verify-tags.yml, allowed_signers) and produces a locally-verified signed tag; only `gh ssh-key add --type signing` on the account remains for GitHub's Verified badge |
| **Real-model hardware validation runs** | Needs GPU hardware per registry entry | Reference model validated on the maintainer's box; `atlas bench` harness ready; per-entry runs are hardware work |
| **Second release-capable maintainer** | Organizational (bus factor 1) | Documented in MAINTAINERS.md with open seats; no tooling can close it |

### Formerly external — now resolved

The original directive listed three more as external; they have since
been done in-repo (with the maintainer's authorization) and are no
longer blockers:

- **Branch protection** — Done: applied to `main` + `dev` (21 required
  checks, no force-push/deletion; enforce_admins off to preserve the
  solo ff-flow). GOVERNANCE.md records the config.
- **Gemma ASA marker decision** — Done: decided **off by default**
  (marker withheld pending an A/B); opt in via `atlas asa build`. ADR
  0005 + SUPPORT_MATRIX.
- **Gemma Lens calibration** — Done: derived + verified on the
  maintainer's hardware (val AUC 0.73; live lens `cx_calibrated: true`);
  CALIBRATION_PROVENANCE.md. Re-publishing the calibrated bundle as the
  shared HF artifact remains a maintainer decision (moderate AUC).

---

## P0 scoreboard

| # | P0 item | Status |
|---|---|---|
| 1 | Local service authentication | **Done** |
| 2 | Local-only binding validation | **Done** |
| 3 | Private-value filtering + file exclusion | **Done** |
| 4 | Reproducible dependency builds | **Done** |
| 5 | Proxy image simplification | **Done** |
| 6 | Non-root proxy/v3/lens | **Done** |
| 7 | Versioned API + schemas | **Done** (version endpoint, error taxonomy, OpenAPI 3.1 + route-parity) |
| 8 | Typed config + migration | **Done** (typed schema, validate/migrate, precedence-aware resolver) |
| 9 | Upgrade/rollback commands | **Done** (auto-restore + cosign verify before apply) |
| 10 | Signed artifact manifests | **Done** (sign/verify + snapshot/rollback) |
| 11 | Structured logs + diagnostics | **Done** (diagnostics bundle + JSON logs + cross-service correlation IDs) |
| 12 | Dependency/repo quality checks | **Done** (staticcheck, secret scan, license check, scoped mypy, container scan, SBOM+provenance attestation verify) |
| 13 | Local boundary regression tests | **Done** (matrix + trust modes) |
| 14 | Concurrency/recovery tests | **Done** (deterministic suite) |
| 15 | Performance harness + budgets | **Done** (versioned format + gate) |
| 16 | Lens provenance manifest | **Done** (auto-write + completeness gate) |
| 17 | Reference-model status clarity | **Done** |

Repository-controlled P0 items complete: **1–17** (all). The remaining
repository items (7–16) are in-repo work with no external dependency.
Genuinely external: signed git tags, per-entry hardware validation,
second maintainer.
