# Backup and Restore

What actually holds state, where it lives, and what losing it costs.

| State | Location | Loss impact | Backup |
|---|---|---|---|
| Configuration | `.env` (+ `atlas.conf` for K3s) | Re-run `atlas init` | copy the file |
| Models | `ATLAS_MODELS_DIR` (default `./models`) | Re-download (hash-verified) | optional — large, re-fetchable |
| Lens/ASA bundles | `geometric-lens/geometric_lens/models/` + `models/*.gguf(.model)` | Published bundles re-download; **locally-trained calibration does not** | copy the dir after any `atlas lens build`/`retrain` |
| Lens training corpus | `ATLAS_LENS_HOST_DIR` (default `./lens_training`) + `benchmark/results/` | Lose the ability to retrain calibration | copy before pruning |
| Learned Redis state | `redis-data` volume (patterns, co-occurrence graph, router posteriors — TTL-less) | Learning resets to seeds/uniform priors; nothing breaks | `docker run --rm -v atlas_redis-data:/data -v $PWD:/backup alpine tar czf /backup/redis-data.tgz /data` |
| TUI sessions | `~/.cache/atlas-tui/sessions/` | Lose `--resume` history | copy the dir |
| Project files | your repo | — | your VCS |

## Restore

Config/models/bundles/corpus: copy back into place, `docker compose up
-d`, `atlas doctor` (it re-verifies artifact identity + hashes).
Redis: stop the stack, untar into the volume with the inverse of the
backup command, start, check `/health` on the lens.

## Honest gaps

Backups are manual copies — there is no `atlas backup` command. The
table above is the complete state inventory; nothing else on the
machine is ATLAS state.
