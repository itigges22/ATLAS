FROM docker.io/python:3.11-slim

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
