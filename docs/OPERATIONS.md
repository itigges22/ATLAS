# Operations Guide

Day-2 operations for a running ATLAS install: health, logs, runbooks,
upgrades, rollback, and backup. Companion to TROUBLESHOOTING.md
(symptom→fix) — this file is procedure-oriented.

## Health

```bash
atlas doctor                         # the production contract: services,
                                     # artifacts, identity, calibration, disk
curl -s localhost:8090/health        # proxy liveness (+ upstream summary)
curl -s localhost:8090/ready         # readiness: 200 only when llama, lens,
                                     # sandbox, v3 are all healthy
docker compose ps                    # container states + restarts
```

Per-service health: llama `:8080/health`, lens `:8099/health` (rich
degraded-state JSON incl. lens self-test), v3 `:8070/health`, sandbox
`:30820/health`.

## Logs

```bash
docker compose logs -f atlas-proxy   # agent loop, tool calls, gates
docker compose logs -f v3-service    # pipeline phases
docker compose logs -f geometric-lens
docker compose logs --tail 100      # everything recent
```

TUI-side debugging: `atlas --log` writes a local TUI event log
(`ATLAS_TUI_LOG`).

## Runbooks

| Symptom | Procedure |
|---|---|
| Service won't start | `docker compose logs <svc>`; port collision → change the `ATLAS_*_PORT` in `.env`; bad override → `docker compose config` names the offending key |
| Model load failure | llama logs name the reason (VRAM, arch, quant); `atlas tier fit --write` re-sizes; `atlas model verify` checks file integrity against the pinned hash |
| Lens degraded (`self_test_error`) | `atlas lens check` prints the exact missing artifact + fix command (`atlas lens build` / `retrain`); identity mismatch means the bundle belongs to another model |
| ASA inactive | llama startup log prints why (missing vector / marker mismatch); `atlas asa check`, then `atlas asa build` |
| Sandbox failures | `docker compose logs sandbox`; egress-cut mode (`ATLAS_SANDBOX_NET_INTERNAL=true`) intentionally breaks dependency installs; resource kills show as 137/timeout in tool results |
| GPU OOM | reduce `ATLAS_CTX_SIZE`/slots via `atlas tier fit --write`; check nothing else holds VRAM (`nvidia-smi`) |
| Disk full | models dir is the usual consumer; `atlas model remove <name> --yes`; learned state is a single SQLite file on the `lens-state` volume (small — pattern/router state, not bulk data) |
| Corrupt state after crash | containers are stateless except the volumes; `docker compose down && up -d` rebuilds runtime state; artifacts re-verify by hash on doctor |
| Failed upgrade | § Rolling back (pin the previous tag, restore `.env.bak`) |
| Bad/revoked artifact | SECURITY.md § artifact revocation; `atlas model verify` + `--force-artifacts` reinstall pins |
| Full reset (keep models) | `docker compose down -v && docker compose up -d` — wipes the learned SQLite state (`lens-state` volume) + lens project index, keeps models/config |

## Resource tuning

All knobs in CONFIGURATION.md; the load-bearing ones: `ATLAS_CTX_SIZE`
+ `ATLAS_PARALLEL_SLOTS` (VRAM), `ATLAS_SANDBOX_MEM/CPUS/PIDS`
(runaway-build protection), `ATLAS_V3_TIMEOUT` (interactive cap).

---

# Upgrading

Applies to the Docker Compose deployment (the supported path).

## Standard upgrade

```bash
cd /opt/atlas            # your checkout
cp .env .env.bak         # config backup (one file holds all your settings)
git pull
docker compose pull      # fetch the target images
docker compose up -d     # recreate changed services
atlas doctor             # verify: services healthy, model/artifacts intact
```

Pin instead of tracking `latest` for production use: set
`ATLAS_IMAGE_TAG=3.1.3` (or an exact `sha-<commit>`) in `.env` before
`docker compose pull`. Every published digest is immutable and cosign-
signed; `sha-*` tags never move.

## What an upgrade can and cannot touch

- **Config:** `.env` is never rewritten by an upgrade. New keys are
  additive with safe defaults; removed keys are ignored (see
  CONFIGURATION.md § removed variables). Re-run `atlas init` only if
  you want re-detected hardware sizing.
- **Models and artifacts:** never modified by image upgrades. Lens/ASA
  bundles are per-model and identity-checked at load; an upgrade that
  changes bundle requirements surfaces as a doctor warning with the
  exact rebuild command, not a silent break.
- **Learned state:** the `lens-state` volume (patterns, router
  posteriors in `geometric_state.db`) and `lens-data` volume persist
  across upgrades.

## Version compatibility

N-1 configs are supported: a `.env` written by the previous release
boots the current one. Registry/artifact schema changes are additive
within a minor version (SUPPORT_MATRIX.md § compatibility policy).

## Automated upgrade (`atlas upgrade`)

```bash
atlas upgrade --to 3.1.3       # or --to latest (default); a leading v is accepted and stripped
```

This records a restore point (current tag + image digests + a `.env`
backup) before staging the target images, starts them, waits for
readiness, runs a quick-doctor smoke check, and finalizes. **If any step
fails — a bad pull, a service that never becomes ready, or a failed
smoke check — it automatically restores the previous release** (your
`.env`, including `ATLAS_IMAGE_TAG`, and brings the old images back up on
the locally cached layers — the restore never re-pulls, since a mutable
previous tag could have moved). `--skip-smoke` skips only the final
check; the restore-on-failure guarantee still holds for the earlier
steps.

Re-running with the tag already deployed: a release tag (`vX.Y.Z`) is a
no-op — those tags never move. A mutable tag (`latest`, `dev`) runs the
full staged flow anyway ("refresh"), because the registry may point the
same tag at newer images; the pull is cheap when nothing changed. Note
that a refresh replaces the locally cached images under the same tag, so
`atlas rollback` after a successful refresh cannot return to the
pre-refresh build — pin release tags for reversible upgrades. Images not
published in the registry for your backend (e.g. the locally-built ROCm
llama image) are skipped by signature verification, and a slow pull gets
up to `ATLAS_UPGRADE_PULL_TIMEOUT` (default 3600 s) to finish.

The manual sequence above remains valid and is what `atlas upgrade`
automates. To undo a *successful* upgrade later, `atlas rollback`
(§ Rolling back) returns to the recorded restore point.

`atlas upgrade` verifies each target image's keyless cosign signature
before applying (best-effort: if cosign isn't installed it logs and
continues; a signature that *fails* aborts the upgrade and the previous
release stays in place). Override with `ATLAS_UPGRADE_SKIP_VERIFY=1`.

`atlas upgrade --to <tag> --dry-run` previews the plan (current tag +
image digests → target, and the ordered steps) without changing
anything.

---

# Rolling back

## Automated (`atlas rollback`)

```bash
atlas rollback            # restore the last upgrade's previous release
atlas rollback --to 3.1.2   # or target a specific immutable tag
```

With no argument it reads the restore point written by `atlas upgrade`
(`.atlas-upgrade/restore-point.json`) and brings the previous release
back up. With `--to TAG`, a pull/start failure (e.g. a typo'd tag that
doesn't exist) restores `.env` to the tag that was deployed before the
attempt. The manual procedures below are the equivalent by hand and the
fallback when no restore point exists.

## Images

Every push publishes immutable `sha-<commit>` tags and releases publish
semver tags; none are ever repointed. To roll back:

```bash
# 1. Find the last-good tag (release tag, or a sha-* from
#    `docker compose images` / the GHCR package page)
# 2. Pin it:
sed -i 's/^ATLAS_IMAGE_TAG=.*/ATLAS_IMAGE_TAG=3.1.2/' .env   # or sha-abc1234 — registry semver tags carry no leading v
docker compose pull
docker compose up -d
atlas doctor
```

Signatures verify against any historical digest (`cosign verify`, see
build-images.yml for the identity flags).

## Configuration

`.env` is a single flat file — restore the `.env.bak` you took before
upgrading (§ Standard upgrade, step 1), then `docker compose up -d`.

## Code (checkout)

```bash
git log --oneline          # find the last-good commit
git checkout <tag-or-sha>  # or: git reset --hard <sha> on your branch
pip install -e . --no-deps # refresh the CLI entry point
```

## Lens/ASA artifacts

`atlas artifact snapshot` (run before activating a new bundle) keeps
one previous-bundle copy; `atlas artifact rollback` restores it and
`atlas artifact verify` checks signature + file hashes. Without a
snapshot, re-download the pinned published bundle
(`atlas model install-artifacts <model> --force-artifacts` — hashes are
pinned in the registry, so you get exactly the published bytes) or
restore your own backup of the lens models dir (§ Backup and restore).

## Learned state

The SQLite state store has no schema coupling to ATLAS versions; it
rolls back with the `lens-state` volume (or keeps working across
versions untouched).

---

# Backup and restore

What actually holds state, where it lives, and what losing it costs.

| State | Location | Loss impact | Backup |
|---|---|---|---|
| Configuration | `.env` (+ `atlas.conf` for K3s) | Re-run `atlas init` | copy the file |
| Models | `ATLAS_MODELS_DIR` (default `./models`) | Re-download (hash-verified) | optional — large, re-fetchable |
| Lens/ASA bundles | `geometric-lens/geometric_lens/models/` + `models/*.gguf(.model)` | Published bundles re-download; **locally-trained calibration does not** | copy the dir after any `atlas lens build`/`retrain` |
| Lens training corpus | `ATLAS_LENS_HOST_DIR` (default `./lens_training`) + `benchmark/results/` | Lose the ability to retrain calibration | copy before pruning |
| Learned state | `geometric_state.db` on the `lens-state` volume (patterns, co-occurrence graph, router posteriors — TTL-less) | Learning resets to seeds/uniform priors; nothing breaks | one file — see below |
| TUI sessions | `~/.cache/atlas-tui/sessions/` | Lose `--resume` history | copy the dir |
| Project files | your repo | — | your VCS |

## Restore

Config/models/bundles/corpus: copy back into place, `docker compose up
-d`, `atlas doctor` (it re-verifies artifact identity + hashes).

## Learned state (SQLite)

The entire learned state is one file: `geometric_state.db` at
`SQLITE_DB_PATH` (default `/data/state/geometric_state.db`) on the
`lens-state` volume. Two safe ways to copy it out:

```bash
# 1. Cold copy — stop the stack first (no writers, plain file copy).
#    Copy the WAL/SHM siblings too: recent commits can still live in
#    geometric_state.db-wal until SQLite checkpoints them.
docker compose stop
docker run --rm -v atlas_lens-state:/data/state -v "$PWD":/backup alpine \
  sh -c 'cp /data/state/geometric_state.db* /backup/'
docker compose start

# 2. Online copy — SQLite's backup API is consistent under WAL
#    (python stdlib; the lens image has no sqlite3 CLI)
docker compose exec geometric-lens python -c "import sqlite3; \
  src = sqlite3.connect('/data/state/geometric_state.db'); \
  dst = sqlite3.connect('/tmp/state-backup.db'); \
  src.backup(dst); dst.close(); src.close()"
docker compose cp geometric-lens:/tmp/state-backup.db ./geometric_state.db
```

Do NOT `cp` the live file while the stack is running — a plain copy of
a database mid-write can be torn; use one of the two forms above.

Restore: stop the stack, copy the file back into the volume (inverse of
the cold copy), start, check `/health` on the lens — the
`subsystems.sqlite` block should report the store available.

## Honest gaps

Backups are manual copies — there is no `atlas backup` command. The
table above is the complete state inventory; nothing else on the
machine is ATLAS state.
