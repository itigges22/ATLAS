# ADR 0002: Keep Redis as the lens-service state store

Status: superseded by 0007 (2026-07)

## Context
Redis backs the pattern cache, co-occurrence graph, Thompson-sampling
router state, task queue, and metrics — all inside geometric-lens; no
other service touches it. GH #57 proposed replacing it with SQLite; PR
#128 attempted it and delivered a half-migration that would break the
pattern cache (removes the redis dep while cache/pattern_store.py still
imports it).

## Decision
Keep Redis, hardened: digest-pinned image, AOF persistence,
`--maxmemory` with `noeviction` (learned TTL-less state must fail
visibly rather than be silently evicted), container mem_limit,
healthcheck-gated startup, internal-network only. Outage behavior is
documented honestly: cache/router degrade to neutral, task queue
returns 503.

## Consequences
One external dependency with a support policy instead of a temporary
one without. A future SQLite migration remains possible but must move
ALL consumers (pattern store, co-occurrence, router, queue, metrics,
compose, doctor, uninstall) in one change — the #128 shape is rejected.
Learned state resets if the redis-data volume is lost; backup guidance
lives in BACKUP_RESTORE.md.
