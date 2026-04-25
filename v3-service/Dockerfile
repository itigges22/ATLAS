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

# Install Python deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8070

CMD ["python", "main.py"]
