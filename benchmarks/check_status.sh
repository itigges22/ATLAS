#!/bin/bash
# Check status of all V3.0.1 benchmarks
cd "$(dirname "$0")/.."

echo "=== ATLAS V3.0.1 Benchmark Status ==="
echo "Date: $(date)"
echo ""

for section in benchmarks/section_*/; do
    section_name=$(basename "$section" | sed 's/section_//' | tr '_' ' ')
    echo "--- $section_name ---"
    for bench in "$section"*/; do
        name=$(basename "$bench")
        resp="$bench/responses.jsonl"
        results="$bench/results.json"

        if [ -f "$results" ]; then
            acc=$(python3 -c "import json; d=json.load(open('$results')); print(f'{d.get(\"accuracy\",0):.1f}%')" 2>/dev/null)
            echo "  $name: COMPLETE ($acc)"
        elif [ -f "$resp" ]; then
            count=$(wc -l < "$resp" 2>/dev/null || echo 0)
            echo "  $name: IN PROGRESS ($count responses)"
        else
            echo "  $name: NOT STARTED"
        fi
    done
    echo ""
done

# Check model status
echo "--- Infrastructure ---"
GEN_URL="${LLAMA_GEN_URL:-${LLAMA_URL:-http://localhost:8000}}"
EMBED_URL="${LLAMA_EMBED_URL:-http://localhost:8001}"
gen=$(curl -s "$GEN_URL/health" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo "DOWN")
echo "  vllm-gen: $gen"

embed=$(curl -s "$EMBED_URL/health" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo "DOWN")
echo "  vllm-embed: $embed"

sandbox=$(curl -s http://localhost:30820/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo "DOWN")
echo "  sandbox: $sandbox"
