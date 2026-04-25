FROM docker.io/python:3.11-slim

# Unbuffered stdout/stderr so `docker logs` shows progress in real-time.
# Without this, Python block-buffers when stdout is a pipe and the user
# sees stale logs for seconds at a time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy benchmark modules (V3 pipeline depends on them)
COPY benchmark/ /app/benchmark/
COPY v3-service/main.py /app/main.py

# Install Python deps. v3-service is a thin HTTP front-end (stdlib http.server)
# in front of the V3 pipeline modules. None of those modules import torch,
# numpy, or any heavy ML lib — they only marshal text in/out of vLLM. httpx
# is the one optional perf upgrade for benchmark.runner; without it we fall
# back to urllib (functionally identical, ~5% slower).
#
# Earlier images installed the ~2 GiB torch CPU wheel here unnecessarily,
# which slowed `docker compose build` by minutes and ballooned the v3-service
# image. Confirmed unused: `python -c "import sys; sys.modules['torch']=...;
# import main"` succeeds.
RUN pip install --no-cache-dir httpx

EXPOSE 8070

CMD ["python", "main.py"]
