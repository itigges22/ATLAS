# Operations Guide

Day-2 operations for a running ATLAS install. Companion to
TROUBLESHOOTING.md (symptom→fix) — this file is procedure-oriented.

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
| Failed upgrade | ROLLBACK.md (pin the previous tag, restore `.env.bak`) |
| Bad/revoked artifact | SECURITY.md § artifact revocation; `atlas model verify` + `--force-artifacts` reinstall pins |
| Full reset (keep models) | `docker compose down -v && docker compose up -d` — wipes the learned SQLite state (`lens-state` volume) + lens project index, keeps models/config |

## Resource tuning

All knobs in CONFIGURATION.md; the load-bearing ones: `ATLAS_CTX_SIZE`
+ `ATLAS_PARALLEL_SLOTS` (VRAM), `ATLAS_SANDBOX_MEM/CPUS/PIDS`
(runaway-build protection), `ATLAS_V3_TIMEOUT` (interactive cap).

## Upgrades / rollback / backup

See UPGRADE.md, ROLLBACK.md, BACKUP_RESTORE.md.
