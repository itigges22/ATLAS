#!/bin/bash
# V3.1: Qwen3.5-9B — Generation + Self-Embeddings (no spec decode)
#
# Qwen3.5-9B uses hybrid DeltaNet+Attention architecture.
# Speculative decoding is NOT supported for Qwen3.5 in llama.cpp yet
# (see: github.com/ggml-org/llama.cpp/issues/20039).
#
# Without draft model, VRAM budget is much more relaxed:
#   Main model Q6_K: ~7.5GB
#   KV caches: ~1.4GB (DeltaNet hybrid — minimal KV, mostly recurrent state)
#   Compute: ~4GB
#   Total: ~12GB / 16.3GB (headroom: ~3.7GB)
#
# DeltaNet KV cache is tiny (~144MB for 2 slots at 16K). This allows
# --parallel 4 with 40K context per slot while staying well within VRAM.
#
# Self-embeddings: 4096-dim (Qwen3.5 hidden_size), not 5120-dim.
# Lens C(x) must be retrained on 4096-dim embeddings.
#
# Expected throughput: ~40-60 tok/s (no spec decode, but smaller model)

SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-/tmp/slots}"
mkdir -p "$SLOT_SAVE_PATH"

CTX_LENGTH="${CONTEXT_LENGTH:-163840}"
KV_CACHE_K="${KV_CACHE_TYPE_K:-q8_0}"
KV_CACHE_V="${KV_CACHE_TYPE_V:-q4_0}"
KV_FLAGS="-ctk $KV_CACHE_K -ctv $KV_CACHE_V"
PARALLEL="${PARALLEL_SLOTS:-4}"
MODEL_FILE="${MODEL_PATH:-/models/Qwen3.5-9B-Q6_K.gguf}"
PORT="${PORT:-8080}"

export GGML_CUDA_NO_PINNED="${GGML_CUDA_NO_PINNED:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

echo "=== V3.1: Qwen3.5-9B — Generation + Self-Embeddings ==="
echo "  Model: $MODEL_FILE"
echo "  Context: $CTX_LENGTH | KV: K=$KV_CACHE_K V=$KV_CACHE_V | Parallel: $PARALLEL"
echo "  Embeddings: ENABLED (4096-dim Qwen3.5 self-embeddings)"
echo "  Speculative decoding: DISABLED (not supported for Qwen3.5)"
echo "  Slot save path: $SLOT_SAVE_PATH"

exec /usr/local/bin/llama-server \
  -m "$MODEL_FILE" \
  -c $CTX_LENGTH \
  $KV_FLAGS \
  --parallel $PARALLEL \
  --cont-batching \
  -ngl 99 \
  --host 0.0.0.0 \
  --port $PORT \
  --flash-attn on \
  --mlock \
  -b 4096 \
  -ub 4096 \
  --slot-save-path "$SLOT_SAVE_PATH" \
  --ctx-checkpoints 0 \
  --no-cache-prompt \
  --embeddings \
  --jinja
