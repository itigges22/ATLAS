# Backup and Restore

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
