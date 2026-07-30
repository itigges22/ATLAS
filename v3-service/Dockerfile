# python:3.11-slim — digest-pinned for reproducible builds (Dependabot maintains)
FROM docker.io/python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying source so code edits do not invalidate
# the expensive Torch/tree-sitter layers during local image rebuilds.
# Versions are pinned in requirements.txt for reproducible builds. Torch
# comes from the CPU-only index first so pip cannot pick a CUDA/NCCL wheel;
# the remaining pinned deps (tree-sitter grammars) resolve from PyPI. The
# version here MUST match requirements.txt's torch pin (contract-tested):
# a mismatch makes the second pip "upgrade" torch from PyPI, silently
# dragging the ~8 GB nvidia/cu* dependency stack into a CPU-only image.
COPY v3-service/requirements.txt /tmp/v3-requirements.txt
RUN pip install --no-cache-dir torch==2.13.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/v3-requirements.txt

# V3 pipeline stages (candidate generation, selection, repair)
COPY v3-service/stages/ /app/stages/
COPY v3-service/main.py /app/main.py
COPY v3-service/adapters.py /app/adapters.py
COPY v3-service/scoring.py /app/scoring.py
COPY v3-service/symbols.py /app/symbols.py
COPY v3-service/planning.py /app/planning.py
COPY v3-service/pipeline.py /app/pipeline.py
COPY v3-service/private_values.py /app/private_values.py
COPY v3-service/structured_log.py /app/structured_log.py
# Structural call-graph reasoning (issue #39): tree-sitter-backed call graph.
# Pure stdlib + the tree-sitter grammars installed above; no extra deps.
COPY v3-service/graph/ /app/graph/

# Non-root application account created inside this image. The V3 service
# writes nothing to the container fs at runtime (telemetry disabled; it
# proxies to llama/lens/sandbox), so /app stays root-owned + world-
# readable and the app user needs no writable app directory. HOME is a
# writable home for any library cache; bytecode caching is disabled so
# Python never tries to write .pyc into the read-only /app.
ENV PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 1001 --user-group appuser
USER appuser
ENV HOME=/home/appuser

EXPOSE 8070

CMD ["python", "main.py"]
