# Rolling back ATLAS

## Automated (`atlas rollback`)

```bash
atlas rollback            # restore the last upgrade's previous release
atlas rollback --to v1.1.0  # or target a specific immutable tag
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
sed -i 's/^ATLAS_IMAGE_TAG=.*/ATLAS_IMAGE_TAG=v3.1.1/' .env   # or sha-abc1234
docker compose pull
docker compose up -d
atlas doctor
```

Signatures verify against any historical digest (`cosign verify`, see
build-images.yml for the identity flags).

## Configuration

`.env` is a single flat file — restore the `.env.bak` you took before
upgrading (UPGRADE.md step 1), then `docker compose up -d`.

## Code (checkout)

```bash
git log --oneline          # find the last-good commit
git checkout <tag-or-sha>  # or: git reset --hard <sha> on your branch
pip install -e . --no-deps # refresh the CLI entry point
```

## Lens/ASA artifacts

Bundle activation keeps no automatic previous-bundle copy yet (roadmap).
Rollback = re-download the pinned published bundle
(`atlas model install-artifacts <model> --force-artifacts` — hashes are
pinned in the registry, so you get exactly the published bytes) or
restore your own backup of the lens models dir (BACKUP_RESTORE.md).

## Learned state

Redis state has no schema coupling to ATLAS versions; it rolls back
with the volume (or keeps working across versions untouched).
