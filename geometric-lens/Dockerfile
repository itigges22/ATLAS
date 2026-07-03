# python:3.11-slim — digest-pinned for reproducible builds (Dependabot maintains)
FROM docker.io/python:3.11-slim@sha256:b27df5841f3355e9473f9a516d38a6783b6c8dfeacaf2d14a240f443b368ddb6
WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
# Lens runs on CPU; all model inference stays in llama-server. Install torch
# from the CPU-only index first so pip cannot choose PyPI's CUDA/NCCL wheel on
# Linux arm64, then resolve the ordinary application dependencies from PyPI.
RUN pip install --no-cache-dir torch==2.10.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .
COPY indexer/ indexer/
COPY retriever/ retriever/
COPY models/ models/
COPY cache/ cache/
COPY router/ router/
COPY geometric_lens/ geometric_lens/

# Create directories for config and data. /data/projects is the only
# location the serving path writes at runtime (the project index, on the
# lens-data volume); config + secrets are read-only at runtime and the
# models dir is mounted read-only, so only /data/projects and the app
# user's HOME need to be writable.
RUN mkdir -p /app/config /app/secrets /data/projects

# Non-root application account created inside this image. /app files stay
# root-owned + world-readable (COPY default); /data/projects is chowned so
# a freshly-created lens-data volume inherits writable ownership. HOME is a
# writable home for library caches (numpy/torch config, tiktoken); bytecode
# caching is disabled so Python never tries to write .pyc into /app.
#
# NOTE: an EXISTING lens-data volume was created root-owned by a prior
# image; after upgrading, chown it once —
#   docker compose run --rm --user root geometric-lens chown -R lens:lens /data/projects
# — or recreate the volume. Fresh installs need nothing.
ENV PYTHONDONTWRITEBYTECODE=1
RUN groupadd --gid 1001 lens \
    && useradd --create-home --uid 1001 --gid 1001 lens \
    && chown -R lens:lens /data/projects
USER lens
ENV HOME=/home/lens

EXPOSE 8099

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8099"]
