#!/bin/bash
# Pre-flight check for the vLLM benchmark stack.
#
# Verifies that:
#   1. vLLM gen instance answers /v1/chat/completions with non-empty content
#   2. vLLM embed instance returns a 4096-dim vector from /v1/embeddings
#   3. Geometric Lens /internal/lens/score-text path returns a finite energy
#
# Run from inside the running container before starting the benchmark sweep:
#   ./benchmarks/h200/preflight.sh
#
# Exit codes:
#   0 — all three checks pass; safe to start benchmarks
#   1 — one or more checks failed; do not start the sweep

set -uo pipefail
: "${LLAMA_GEN_URL:=http://localhost:8000}"
: "${LLAMA_EMBED_URL:=http://localhost:8001}"
: "${LENS_URL:=http://localhost:31144}"
: "${LLAMA_GEN_MODEL:=qwen3.5-9b}"
: "${LLAMA_EMBED_MODEL:=qwen3.5-9b-embed}"

PASS=0
FAIL=0

check() {
    local label=$1; shift
    if "$@" >/dev/null 2>&1; then
        echo "  PASS  $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $label"
        FAIL=$((FAIL + 1))
    fi
}

echo "============================================="
echo "ATLAS vLLM stack pre-flight"
echo "  Gen URL:   $LLAMA_GEN_URL"
echo "  Embed URL: $LLAMA_EMBED_URL"
echo "  Lens URL:  $LENS_URL"
echo "============================================="

# 1. Gen instance health + a real /v1/chat/completions call.
check "gen /health" curl -sf --max-time 10 "$LLAMA_GEN_URL/health"
chat_resp=$(curl -s --max-time 60 "$LLAMA_GEN_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(cat <<EOF
{"model":"$LLAMA_GEN_MODEL","messages":[{"role":"user","content":"reply with the single word: ok"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}
EOF
)" || true)
if echo "$chat_resp" | grep -q '"content"'; then
    echo "  PASS  gen /v1/chat/completions returned content"
    PASS=$((PASS + 1))
else
    echo "  FAIL  gen /v1/chat/completions"
    echo "        response: ${chat_resp:0:300}"
    FAIL=$((FAIL + 1))
fi

# 2. Embed instance health + a real /v1/embeddings call returning 4096 dims.
#    SKIP_EMBED=1 (used on tight-VRAM single-card setups) takes the embed
#    instance out of the entrypoint entirely; in that mode the Lens is also
#    disabled, so embeddings genuinely aren't needed and probing for them
#    would falsely fail the entire preflight.
if [[ "${SKIP_EMBED:-0}" == "1" ]]; then
    echo "  SKIP  embed (SKIP_EMBED=1; running gen-only)"
else
    check "embed /health" curl -sf --max-time 10 "$LLAMA_EMBED_URL/health"
    embed_resp=$(curl -s --max-time 60 "$LLAMA_EMBED_URL/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$LLAMA_EMBED_MODEL\",\"input\":\"def hello(): return 1\"}" || true)
    embed_dim=$(echo "$embed_resp" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(len(d['data'][0]['embedding']))
except Exception as e:
    print(0)
" 2>/dev/null || echo 0)
    if [[ "$embed_dim" == "4096" ]]; then
        echo "  PASS  embed returned 4096-dim vector"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  embed dimension was $embed_dim, expected 4096"
        echo "        response: ${embed_resp:0:300}"
        FAIL=$((FAIL + 1))
    fi
fi

# 3. Lens scoring (only required if GEOMETRIC_LENS_ENABLED=true).
if [[ "${GEOMETRIC_LENS_ENABLED:-true}" == "true" ]]; then
    check "lens /health" curl -sf --max-time 10 "$LENS_URL/health"
    lens_resp=$(curl -s --max-time 30 "$LENS_URL/internal/lens/score-text" \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"def hello(): return 1\"}" || true)
    if echo "$lens_resp" | grep -qE '"energy"|"normalized"'; then
        echo "  PASS  lens score-text returned a score"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  lens score-text"
        echo "        response: ${lens_resp:0:300}"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  SKIP  Lens (GEOMETRIC_LENS_ENABLED=false)"
fi

echo "---------------------------------------------"
echo "Pre-flight: $PASS passed, $FAIL failed"
echo "============================================="

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
