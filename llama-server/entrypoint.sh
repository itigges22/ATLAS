#!/bin/bash

# Check if speculative decoding is enabled via env var
SPEC_FLAGS=""
if [ -n "$DRAFT_MODEL" ] && [ -f "$DRAFT_MODEL" ]; then
    echo "Speculative decoding enabled with draft model: $DRAFT_MODEL"
    SPEC_FLAGS="--model-draft $DRAFT_MODEL --draft-max 16 --draft-min 5 --draft-p-min 0.9"
else
    echo "Speculative decoding disabled (no DRAFT_MODEL set or file not found)"
fi

exec /usr/local/bin/llama-server \
  -m /models/Qwen3-14B-Q4_K_M.gguf \
  $SPEC_FLAGS \
  -c 16384 \
  --parallel 1 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8000 \
  --flash-attn on \
  --jinja \
  --chat-template-file /templates/Qwen3-custom.jinja
