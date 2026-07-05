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
the cached layers). `--skip-smoke` skips only the final check; the
restore-on-failure guarantee still holds for the earlier steps.

The manual sequence above remains valid and is what `atlas upgrade`
automates. To undo a *successful* upgrade later, `atlas rollback`
(ROLLBACK.md) returns to the recorded restore point.

Still roadmap: a digest-diff preview before applying, and cryptographic
signature verification of the target images as an upgrade gate (image
digests are cosign-signed today; the upgrade command does not yet verify
them inline).
