# Upgrading ATLAS

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
`ATLAS_IMAGE_TAG=v3.1.2` (or an exact `sha-<commit>`) in `.env` before
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
- **Learned state:** the `redis-data` volume (patterns, router
  posteriors) and `lens-data` volume persist across upgrades.

## Version compatibility

N-1 configs are supported: a `.env` written by the previous release
boots the current one. Registry/artifact schema changes are additive
within a minor version (SUPPORT_MATRIX.md § compatibility policy).

## Automated upgrade (`atlas upgrade`)

```bash
atlas upgrade --to v1.2.0      # or --to latest (default)
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
(ROLLBACK.md) returns to the recorded restore point.

`atlas upgrade` verifies each target image's keyless cosign signature
before applying (best-effort: if cosign isn't installed it logs and
continues; a signature that *fails* aborts the upgrade and the previous
release stays in place). Override with `ATLAS_UPGRADE_SKIP_VERIFY=1`.

`atlas upgrade --to <tag> --dry-run` previews the plan (current tag +
image digests → target, and the ordered steps) without changing
anything.
