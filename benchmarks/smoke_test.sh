#!/bin/bash
# Smoke test: 10 tasks per benchmark (baseline). Verify all work before full runs.
# Usage: ./benchmarks/smoke_test.sh
#
# Output: per-benchmark results + timing projections

set -u
cd "$(dirname "$0")/.."

BENCHMARKS=(ifbench ifeval gpqa_diamond mmlu_pro c_eval)
LIMIT=10

echo "=============================================="
echo "V3.0.1 Smoke Test - 10 tasks per benchmark"
echo "Started: $(date)"
echo "=============================================="

mkdir -p benchmarks/logs/smoke

for bench in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=== $bench ==="
    # Clear any existing data
    section_dir=""
    case $bench in
        ifbench) section_dir="benchmarks/section_b_instruction_following/ifbench" ;;
        ifeval) section_dir="benchmarks/section_b_instruction_following/ifeval" ;;
        gpqa_diamond) section_dir="benchmarks/section_a_knowledge_stem/gpqa_diamond" ;;
        mmlu_pro) section_dir="benchmarks/section_a_knowledge_stem/mmlu_pro" ;;
        c_eval) section_dir="benchmarks/section_a_knowledge_stem/c_eval" ;;
    esac

    # Back up existing data
    if [ -f "$section_dir/responses.jsonl" ]; then
        mv "$section_dir/responses.jsonl" "$section_dir/responses.jsonl.bak.$(date +%s)"
    fi
    rm -f "$section_dir/results.json"

    # Run 10 tasks with parallel=4 (matches server slots).
    # Thinking benchmarks need 7200s, MCQ fine with 1800s.
    log="benchmarks/logs/smoke/${bench}_$(date +%Y%m%d_%H%M%S).log"
    case $bench in
        ifbench|ifeval|gpqa_diamond) bench_timeout=7200 ;;
        *) bench_timeout=1800 ;;
    esac
    start_time=$(date +%s)
    BENCHMARK_PARALLEL=4 timeout $bench_timeout python -m benchmarks.v301_runner --benchmark $bench --limit $LIMIT > "$log" 2>&1
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    # Show results
    count=$(wc -l < "$section_dir/responses.jsonl" 2>/dev/null || echo 0)
    echo "  Completed: $count/$LIMIT tasks in ${elapsed}s"

    if [ -f "$section_dir/results.json" ]; then
        python3 -c "
import json
with open('$section_dir/results.json') as f: r = json.load(f)
print(f'  Score: {r.get(\"accuracy\", \"N/A\")}%')
print(f'  Baseline: {r.get(\"baseline_qwen\", \"N/A\")}%')
"
    fi

    # Project full run time
    if [ "$count" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
        case $bench in
            ifbench) total=300 ;;
            ifeval) total=541 ;;
            gpqa_diamond) total=198 ;;
            mmlu_pro) total=3000 ;;
            c_eval) total=1346 ;;
        esac
        projected_hours=$((elapsed * total / count / 3600))
        echo "  Projected full run: ${projected_hours}h"
    fi
done

echo ""
echo "=============================================="
echo "Smoke test complete: $(date)"
echo "=============================================="
