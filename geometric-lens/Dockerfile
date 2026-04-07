FROM docker.io/python:3.11-slim

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

EXPOSE 8099

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8099"]
