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

# Create directories for config and data
RUN mkdir -p /app/config /app/secrets /data/projects

EXPOSE 8099

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8099"]
