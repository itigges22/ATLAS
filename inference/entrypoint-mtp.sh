#!/bin/bash
# V3.1 MTP: Qwen3.5-9B with Multi-Token Prediction
#
# Uses inline MTP (not speculative framework) for ~1.5-2x speedup.
# MTP head predicts next token in parallel with main generation.

SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-/tmp/slots}"
mkdir -p "$SLOT_SAVE_PATH"

CTX_LENGTH="${CONTEXT_LENGTH:-40960}"
KV_CACHE_K="${KV_CACHE_TYPE_K:-q8_0}"
KV_CACHE_V="${KV_CACHE_TYPE_V:-q4_0}"
KV_FLAGS="-ctk $KV_CACHE_K -ctv $KV_CACHE_V"
PARALLEL="${PARALLEL_SLOTS:-4}"
MODEL_FILE="${MODEL_PATH:-/models/Qwen3.5-9B-MTP-Q4_K_M-F16mtp.gguf}"

export GGML_CUDA_NO_PINNED="${GGML_CUDA_NO_PINNED:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

echo "=== V3.1 MTP: Qwen3.5-9B — Generation + MTP + Self-Embeddings ==="
echo "  Model: $MODEL_FILE"
echo "  Context: $CTX_LENGTH | KV: K=$KV_CACHE_K V=$KV_CACHE_V | Parallel: $PARALLEL"
echo "  MTP: ENABLED (inline, 1 draft token per step)"
echo "  Embeddings: ENABLED (4096-dim Qwen3.5 self-embeddings)"

exec /usr/local/bin/llama-server \
  -m "$MODEL_FILE" \
  -c $CTX_LENGTH \
  $KV_FLAGS \
  --parallel $PARALLEL \
  --cont-batching \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8000 \
  --flash-attn on \
  --mlock \
  -b 4096 \
  -ub 2 \
  --slot-save-path "$SLOT_SAVE_PATH" \
  --ctx-checkpoints 0 \
  --no-cache-prompt \
  --embeddings \
  --jinja \
  --no-warmup
