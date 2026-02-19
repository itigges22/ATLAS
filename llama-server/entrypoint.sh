#!/bin/bash

# Check if speculative decoding is enabled via env var
SPEC_FLAGS=""
if [ -n "$DRAFT_MODEL" ] && [ -f "$DRAFT_MODEL" ]; then
    echo "Speculative decoding enabled with draft model: $DRAFT_MODEL"
    SPEC_FLAGS="--model-draft $DRAFT_MODEL --draft-max 16 --draft-min 5 --draft-p-min 0.9"
else
    echo "Speculative decoding disabled (no DRAFT_MODEL set or file not found)"
fi

# Context length (default 65536)
CTX_LENGTH="${CONTEXT_LENGTH:-65536}"

# KV cache quantization (default q4_0)
KV_CACHE_TYPE="${KV_CACHE_TYPE:-q4_0}"
KV_FLAGS="-ctk $KV_CACHE_TYPE -ctv $KV_CACHE_TYPE"

# Template (default custom — supports both think and nothink modes)
TEMPLATE="${CHAT_TEMPLATE:-Qwen3-custom.jinja}"

# Parallel slots (default 2 for best-of-k pipelining)
PARALLEL="${PARALLEL_SLOTS:-2}"

# Embeddings support (default enabled)
EMBED_FLAGS=""
if [ "${ENABLE_EMBEDDINGS:-true}" = "true" ]; then
    echo "Embeddings endpoint enabled"
    EMBED_FLAGS="--embeddings"
fi

# CUDA/GGML environment for PCIe transfer optimization
export GGML_CUDA_NO_PINNED="${GGML_CUDA_NO_PINNED:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

echo "Configuration:"
echo "  Context length: $CTX_LENGTH"
echo "  KV cache type: $KV_CACHE_TYPE"
echo "  Template: $TEMPLATE"
echo "  Parallel slots: $PARALLEL"
echo "  Embeddings: ${ENABLE_EMBEDDINGS:-true}"
echo "  GGML_CUDA_NO_PINNED: $GGML_CUDA_NO_PINNED"
echo "  CUDA_DEVICE_MAX_CONNECTIONS: $CUDA_DEVICE_MAX_CONNECTIONS"
echo "  CUDA_MODULE_LOADING: $CUDA_MODULE_LOADING"

exec /usr/local/bin/llama-server \
  -m /models/Qwen3-14B-Q4_K_M.gguf \
  $SPEC_FLAGS \
  -c $CTX_LENGTH \
  $KV_FLAGS \
  --parallel $PARALLEL \
  --cont-batching \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8000 \
  --flash-attn on \
  --mlock \
  --no-mmap \
  --jinja \
  --reasoning-format deepseek \
  --chat-template-file /templates/$TEMPLATE \
  $EMBED_FLAGS
