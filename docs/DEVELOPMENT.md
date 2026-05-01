# Development workflow

Guide for contributors actively editing ATLAS source. The default
`docker compose up -d` flow is geared at end users — it COPYs source
into images at build time, so every code change requires a rebuild.
This page documents the opt-in dev iteration mode that skips the
rebuild loop.

For first-time setup, see `docs/SETUP.md`. For configuration, see
`docs/CONFIGURATION.md`. For troubleshooting, see
`docs/TROUBLESHOOTING.md`.

---

## When to use dev mode

Use it when you are actively editing Python in `geometric-lens/` or
`v3-service/`, or Go in `atlas-proxy/`. Skip it for one-off
contributions where a single rebuild is acceptable.

Dev mode does not change image behavior in production-like ways —
it only short-circuits the COPY step in the Dockerfile by
bind-mounting your working tree into the container.

---

## Enabling dev mode (Python services)

Compose merges `docker-compose.override.yml` automatically if the
file exists. The repo ships `docker-compose.override.yml.example` as
a template:

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
docker compose up -d
```

To turn dev mode off, delete the override file (or rename it) and
re-run `docker compose up -d`. The override is git-ignored by
convention; commit changes to the `.example` file instead.

### geometric-lens

The override bind-mounts `./geometric-lens` over `/app` and switches
the command to `uvicorn ... --reload`. Editing any `.py` file under
`geometric-lens/` reloads the worker within ~1 second; you do not
need to restart the container.

Verify the reloader is active:

```bash
docker compose logs geometric-lens | grep -i reload
# expect: "Started reloader process ... using StatReload"
```

### v3-service

The override bind-mounts `./v3-service/main.py` and `./benchmark/`
into the container. The service runs on Python's stdlib
`http.server`, which has no built-in reloader, so after editing you
need to restart the container — but no rebuild:

```bash
docker compose restart v3-service
```

Restart is ~1 second vs ~30 seconds for a rebuild. If you find
yourself doing this constantly, the next step is to add `watchfiles`
or migrate to uvicorn — file as a follow-up if it becomes a real
pain point.

---

## atlas-proxy (Go)

The proxy is **not** in the compose override. Its runtime image is
alpine-based with no Go toolchain, so bind-mounting source would not
recompile inside the container. Two viable options for iteration:

### Option A — run on host against the compose stack (recommended)

Bring up the dependencies in compose, then stop only the proxy:

```bash
docker compose up -d
docker compose stop atlas-proxy
```

Then run the proxy on the host, pointing it at the compose-exposed
ports (the defaults work because each upstream service publishes its
port to localhost):

```bash
cd atlas-proxy
ATLAS_PROXY_PORT=8090 \
ATLAS_INFERENCE_URL=http://localhost:8080 \
ATLAS_LLAMA_URL=http://localhost:8080 \
ATLAS_LENS_URL=http://localhost:8099 \
ATLAS_SANDBOX_URL=http://localhost:30820 \
ATLAS_V3_URL=http://localhost:8070 \
ATLAS_AGENT_LOOP=1 \
ATLAS_MODEL_NAME=Qwen3.5-9B-Q6_K \
ATLAS_WORKSPACE_DIR=$(pwd)/.. \
go run .
```

Aider clients pointed at `http://localhost:8090` will hit the host
process. Edits become live with a `Ctrl-C` + re-run. For
auto-rebuild on save, install `air` (`go install
github.com/air-verse/air@latest`) and run `air` in the same
directory.

When done, restart the container version:

```bash
docker compose up -d atlas-proxy
```

### Option B — full container rebuild on each change

If you only have a handful of Go edits, the targeted rebuild is
fine:

```bash
docker compose up -d --build atlas-proxy
```

This is what users on `main` get, ~30-60s per cycle.

---

## Other services

`llama-server`, `redis`, and `sandbox` are not bind-mounted.

- llama-server is a third-party binary — changes there are
  Dockerfile / model-config changes, which need a rebuild anyway.
- redis is upstream redis with no project-side code.
- sandbox runs read-only on purpose; the override should not
  weaken that. If you're editing the sandbox harness itself, do a
  targeted rebuild: `docker compose up -d --build sandbox`.

---

## Targeted rebuilds (no override needed)

Even without dev mode, you can rebuild a single service rather than
the full stack:

```bash
docker compose up -d --build geometric-lens
docker compose up -d --build v3-service
docker compose up -d --build atlas-proxy
```

This is the fallback whenever dev mode is off or doesn't apply
(Dockerfile/dependency changes, requirements.txt edits, etc.).

---

## Cross-references

- `ISSUES.md` PC-030 — ticket that introduced this workflow.
- `docs/SETUP.md` — first-time install.
- `docs/CONFIGURATION.md` — env vars used by each service.
- `docs/TROUBLESHOOTING.md` — runtime symptoms and fixes.
