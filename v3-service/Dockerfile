# python:3.11-slim — digest-pinned for reproducible builds (Dependabot maintains)
FROM docker.io/python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying source so code edits do not invalidate
# the expensive Torch/tree-sitter layers during local image rebuilds.
# Versions are pinned in requirements.txt for reproducible builds. Torch
# comes from the CPU-only index first so pip cannot pick a CUDA/NCCL wheel
# on linux/arm64; the remaining pinned deps (tree-sitter grammars) resolve
# from PyPI.
COPY v3-service/requirements.txt /tmp/v3-requirements.txt
RUN pip install --no-cache-dir torch==2.12.1 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/v3-requirements.txt

# Copy benchmark modules (V3 pipeline depends on them)
COPY benchmark/ /app/benchmark/
COPY v3-service/main.py /app/main.py
COPY v3-service/private_values.py /app/private_values.py
COPY v3-service/structured_log.py /app/structured_log.py
# Structural call-graph reasoning (issue #39): tree-sitter-backed call graph.
# Pure stdlib + the tree-sitter grammars installed above; no extra deps.
COPY v3-service/graph/ /app/graph/
# V3.2 RPG-style planning (issue #120): wavelet substrate + RPG construction.
# Pure stdlib — no extra pip installs needed.
COPY v3-service/wavelet/ /app/wavelet/
COPY v3-service/rpg.py /app/rpg.py

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
