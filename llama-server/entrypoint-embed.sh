#!/bin/bash

# Server B: Embedding sidecar — nomic-embed-text-v1.5 on GPU
# Serves /embedding endpoint for geometric lens energy scoring
# Runs as sidecar in llama-server pod, shares GPU, port 8001

echo "=== Server B: Embeddings (nomic-embed-text-v1.5, GPU sidecar) ==="

exec /usr/local/bin/llama-server \
  -m /models/nomic-embed-text-v1.5.Q8_0.gguf \
  -c 2048 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8001 \
  --embeddings
