#!/bin/bash
export LLAMA_NO_MTP=1
MODEL="${MODEL_PATH:-/models/Qwen3.5-9B-Q6_K.gguf}"
PORT="${PORT:-8080}"
echo "=== Qwen3.5-9B Q6_K — No MTP, Parallel 1 ==="
exec /usr/local/bin/llama-server \
  -m "$MODEL" -c 32768 \
  -ctk q8_0 -ctv q4_0 \
  --parallel 1 --cont-batching -ngl 99 \
  --host 0.0.0.0 --port $PORT \
  --flash-attn on --mlock \
  -b 4096 -ub 4096 \
  --ctx-checkpoints 0 --no-cache-prompt \
  --embeddings --jinja --no-warmup 2>&1
