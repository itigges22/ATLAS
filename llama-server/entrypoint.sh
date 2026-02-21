#!/bin/bash

# Server A: Generation with speculative decoding, NO embeddings
# Embeddings served by separate llama-embed deployment (CPU-only)

SPEC_FLAGS=""
if [ -n "$DRAFT_MODEL" ] && [ -f "$DRAFT_MODEL" ]; then
    echo "Speculative decoding enabled with draft model: $DRAFT_MODEL"
    SPEC_FLAGS="--model-draft $DRAFT_MODEL --draft-max 16 --draft-min 1"
else
    echo "Speculative decoding disabled (no DRAFT_MODEL set or file not found)"
fi

CTX_LENGTH="${CONTEXT_LENGTH:-65536}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-q4_0}"
KV_FLAGS="-ctk $KV_CACHE_TYPE -ctv $KV_CACHE_TYPE"
TEMPLATE="${CHAT_TEMPLATE:-Qwen3-custom.jinja}"
PARALLEL="${PARALLEL_SLOTS:-2}"

export GGML_CUDA_NO_PINNED="${GGML_CUDA_NO_PINNED:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

echo "=== Server A: Generation (spec decode ON, embeddings OFF) ==="
echo "  Context: $CTX_LENGTH | KV: $KV_CACHE_TYPE | Parallel: $PARALLEL"

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
  --no-mmap
