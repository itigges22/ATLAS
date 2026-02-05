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

# Template (default no-think)
TEMPLATE="${CHAT_TEMPLATE:-Qwen3-no-think.jinja}"

echo "Configuration:"
echo "  Context length: $CTX_LENGTH"
echo "  KV cache type: $KV_CACHE_TYPE"
echo "  Template: $TEMPLATE"

exec /usr/local/bin/llama-server \
  -m /models/Qwen3-14B-Q4_K_M.gguf \
  $SPEC_FLAGS \
  -c $CTX_LENGTH \
  $KV_FLAGS \
  --parallel 1 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8000 \
  --flash-attn on \
  --jinja \
  --chat-template-file /templates/$TEMPLATE
