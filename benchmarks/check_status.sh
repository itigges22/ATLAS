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

# Check model status. vLLM `/health` returns 200 with an EMPTY body
# (it's a readiness probe, not a structured status report) so don't
# try to json.load — that would JSONDecodeError on every healthy
# stack and falsely report DOWN. Just check HTTP success via curl -sf.
# Sandbox + Lens are FastAPI services that DO return JSON, but we
# only need readiness here, so the same approach is fine.
echo "--- Infrastructure ---"
GEN_URL="${LLAMA_GEN_URL:-${LLAMA_URL:-http://localhost:8000}}"
EMBED_URL="${LLAMA_EMBED_URL:-http://localhost:8001}"

probe() {
    if curl -sf --max-time 2 "$1/health" >/dev/null 2>&1; then
        echo "UP"
    else
        echo "DOWN"
    fi
}

echo "  vllm-gen: $(probe "$GEN_URL")"
echo "  vllm-embed: $(probe "$EMBED_URL")"
echo "  sandbox: $(probe http://localhost:30820)"
