#!/bin/bash
# Best-of-K Latency Measurement Script
#
# Measures inference latency for k=1,3,5,10 at various temperatures.
# Run BEFORE and AFTER optimizations to quantify improvement.
#
# Usage:
#   ./benchmark/measure_bok_latency.sh [LLAMA_URL]
#
# Default LLAMA_URL: http://localhost:32735

set -euo pipefail

LLAMA_URL="${1:-http://localhost:32735}"
API_URL="$LLAMA_URL/v1/chat/completions"
RESULTS_DIR="benchmark/results/latency_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

PROMPT="Write a Python function that implements binary search on a sorted array and returns the index of the target element, or -1 if not found."
K_VALUES=(1 3 5 10)
TEMPS=(0.0 0.6 0.8)
N_PREDICT=200

echo "=== Best-of-K Latency Measurement ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Endpoint: $API_URL"
echo "Results: $RESULTS_DIR"
echo ""

# Check server health
echo "Checking llama-server health..."
if ! curl -sf "$LLAMA_URL/health" > /dev/null 2>&1; then
    echo "ERROR: llama-server not responding at $LLAMA_URL"
    exit 1
fi
echo "Server OK"
echo ""

# Prompt cache verification
echo "=== Prompt Cache Verification ==="
echo "Sending identical prompt twice to check cache_prompt behavior..."

FIRST=$(curl -sf "$API_URL" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"qwen3-14b\",
        \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
        \"max_tokens\": 50,
        \"temperature\": 0.0,
        \"stream\": false,
        \"cache_prompt\": true
    }" -w '\n%{time_total}' 2>/dev/null)

FIRST_TIME=$(echo "$FIRST" | tail -1)
echo "  First request (cold): ${FIRST_TIME}s"

SECOND=$(curl -sf "$API_URL" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"qwen3-14b\",
        \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
        \"max_tokens\": 50,
        \"temperature\": 0.6,
        \"seed\": 42,
        \"stream\": false,
        \"cache_prompt\": true
    }" -w '\n%{time_total}' 2>/dev/null)

SECOND_TIME=$(echo "$SECOND" | tail -1)
echo "  Second request (warm): ${SECOND_TIME}s"
echo ""

# Main benchmark loop
echo "=== Latency Measurements ==="
echo ""

SUMMARY_FILE="$RESULTS_DIR/summary.csv"
echo "k,temperature,total_ms,per_candidate_ms,candidate_hashes" > "$SUMMARY_FILE"

for k in "${K_VALUES[@]}"; do
    for temp in "${TEMPS[@]}"; do
        echo "--- k=$k, temperature=$temp ---"

        START_NS=$(date +%s%N)
        HASHES=""

        for i in $(seq 1 "$k"); do
            SEED=$((i * 42 + 1))

            RESP=$(curl -sf "$API_URL" \
                -H "Content-Type: application/json" \
                -d "{
                    \"model\": \"qwen3-14b\",
                    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
                    \"max_tokens\": $N_PREDICT,
                    \"temperature\": $temp,
                    \"seed\": $SEED,
                    \"stream\": false,
                    \"cache_prompt\": true
                }" 2>/dev/null)

            # Extract content and hash it for diversity check
            CONTENT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:200])" 2>/dev/null || echo "ERROR")
            HASH=$(echo "$CONTENT" | md5sum | cut -c1-8)
            HASHES="$HASHES $HASH"
        done

        END_NS=$(date +%s%N)
        ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))
        PER_CANDIDATE=$((ELAPSED_MS / k))

        # Count unique hashes
        N_UNIQUE=$(echo "$HASHES" | tr ' ' '\n' | sort -u | grep -c . || echo 0)

        echo "  Total: ${ELAPSED_MS}ms"
        echo "  Per candidate: ${PER_CANDIDATE}ms"
        echo "  Unique outputs: $N_UNIQUE / $k"
        echo ""

        echo "$k,$temp,$ELAPSED_MS,$PER_CANDIDATE,$N_UNIQUE/$k" >> "$SUMMARY_FILE"
    done
done

echo "=== Summary ==="
echo ""
column -t -s',' "$SUMMARY_FILE"
echo ""
echo "Results saved to $RESULTS_DIR/summary.csv"
