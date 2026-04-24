#!/bin/bash
# Watches benchmark output. When all target results.json files are present,
# archives artifacts to /tmp/atlas_results.tar.gz and shuts the instance down.
#
# Run in a separate tmux pane or ssh session:
#   ./benchmarks/h200/watchdog.sh [--no-shutdown] [--targets c_eval,mmlu_pro,...]

set -euo pipefail

SHUTDOWN=1
TARGETS_DEFAULT="c_eval mmlu_pro ifeval gpqa_diamond ifbench"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-shutdown) SHUTDOWN=0; shift ;;
        --targets)     TARGETS_DEFAULT="${2//,/ }"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

TARGETS=($TARGETS_DEFAULT)

declare -A BENCH_DIR=(
    [c_eval]="benchmarks/section_a_knowledge_stem/c_eval"
    [mmlu_pro]="benchmarks/section_a_knowledge_stem/mmlu_pro"
    [ifeval]="benchmarks/section_b_instruction_following/ifeval"
    [gpqa_diamond]="benchmarks/section_a_knowledge_stem/gpqa_diamond"
    [ifbench]="benchmarks/section_b_instruction_following/ifbench"
)

echo "Watchdog targets: ${TARGETS[*]}"
echo "Shutdown on completion: $SHUTDOWN"
echo "Checking every 60s..."
echo ""

while true; do
    DONE=0
    MISSING=()
    for t in "${TARGETS[@]}"; do
        dir="${BENCH_DIR[$t]:-}"
        if [[ -z "$dir" ]]; then
            echo "unknown target: $t" >&2
            continue
        fi
        if [[ -f "$dir/results.json" ]]; then
            DONE=$((DONE+1))
        else
            MISSING+=("$t")
        fi
    done

    ts=$(date "+%H:%M:%S")
    echo "[$ts] done ${DONE}/${#TARGETS[@]}  missing: ${MISSING[*]:-none}"

    if [[ $DONE -eq ${#TARGETS[@]} ]]; then
        echo ""
        echo "All targets complete. Archiving..."
        tar czf /tmp/atlas_results.tar.gz \
            benchmarks/section_*/*/responses.jsonl \
            benchmarks/section_*/*/results.json \
            benchmarks/section_*/*/traces \
            benchmarks/logs/h200_* 2>/dev/null || true
        ls -lh /tmp/atlas_results.tar.gz

        echo ""
        echo "========================================"
        echo "Pull results locally with:"
        echo "  rsync -avz <user>@<this-host>:/tmp/atlas_results.tar.gz ./"
        echo "========================================"

        if [[ $SHUTDOWN -eq 1 ]]; then
            echo ""
            echo "Shutting down in 60 seconds. Ctrl+C to abort."
            sleep 60
            sudo shutdown -h now
        else
            echo "(--no-shutdown set; leaving instance running)"
        fi
        exit 0
    fi

    sleep 60
done
