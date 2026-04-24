#!/bin/bash
# Full baseline run: all 5 benchmarks at parallel 4. No per-benchmark timeout.
# Clean slate (moves any existing responses aside first).
# Usage: nohup ./benchmarks/run_full_baseline.sh > benchmarks/logs/full_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -u
cd "$(dirname "$0")/.."

# C-Eval already complete (80.01%); skip to preserve result.
# All remaining benchmarks use Qwen published methodology:
# temp=1.0, top_k=20, top_p=0.95, presence_penalty=1.5, thinking mode enabled.
BENCHMARKS=(mmlu_pro ifeval gpqa_diamond ifbench)

echo "=============================================="
echo "V3.0.1 Full Baseline Run (parallel 4)"
echo "Started: $(date)"
echo "=============================================="

mkdir -p benchmarks/logs/full

for bench in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=== $bench ==="
    section_dir=""
    case $bench in
        ifbench) section_dir="benchmarks/section_b_instruction_following/ifbench" ;;
        ifeval) section_dir="benchmarks/section_b_instruction_following/ifeval" ;;
        gpqa_diamond) section_dir="benchmarks/section_a_knowledge_stem/gpqa_diamond" ;;
        mmlu_pro) section_dir="benchmarks/section_a_knowledge_stem/mmlu_pro" ;;
        c_eval) section_dir="benchmarks/section_a_knowledge_stem/c_eval" ;;
    esac

    if [ -f "$section_dir/responses.jsonl" ]; then
        mv "$section_dir/responses.jsonl" "$section_dir/responses.jsonl.bak.$(date +%s)"
    fi
    rm -f "$section_dir/results.json"

    log="benchmarks/logs/full/${bench}_$(date +%Y%m%d_%H%M%S).log"
    start_time=$(date +%s)
    echo "[$bench] Running full dataset (no task limit)..."
    BENCHMARK_PARALLEL=4 python -m benchmarks.v301_runner --benchmark $bench > "$log" 2>&1
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    count=$(wc -l < "$section_dir/responses.jsonl" 2>/dev/null || echo 0)
    echo "  Completed: $count tasks in ${elapsed}s ($(echo "scale=1; $elapsed/3600" | bc)h)"

    if [ -f "$section_dir/results.json" ]; then
        python3 -c "
import json
with open('$section_dir/results.json') as f: r = json.load(f)
print(f'  Score:    {r.get(\"accuracy\", r.get(\"loose_prompt\", \"N/A\"))}')
print(f'  Baseline: {r.get(\"baseline_qwen\", r.get(\"baseline\", \"N/A\"))}')
"
    fi
done

echo ""
echo "=============================================="
echo "Full baseline complete: $(date)"
echo "=============================================="
