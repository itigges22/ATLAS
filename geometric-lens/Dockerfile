FROM docker.io/python:3.11-slim

# Real-time logs in docker — without these, output stalls under pipe
# buffering and `docker compose logs` shows stale Lens activity.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

# Lens listens on 31144 by convention (matches docker-compose service port and
# the entrypoint scripts in benchmarks/h200/). Override via --port at runtime.
EXPOSE 31144

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "31144"]
