#!/bin/bash
# V3.1 Full Ablation Study — optimized: 2 runs + derivation
#
# Instead of running 6 separate conditions (60+ hours), this runs:
#   1. Condition A (baseline, k=1): ~2 hours
#   2. Condition F (full pipeline, k=3): ~15 hours
#   3. Derive B/C/D/E from F's stored candidates: ~seconds
#
# Total: ~17 hours instead of ~60 hours
#
# Why this works: Conditions B-F all generate the same k=3 candidates
# (same model, seeds, PlanSearch). The only differences are selection
# strategy (random vs lens) and Phase 3 (on/off). Storing all candidate
# codes enables post-hoc replay with different strategies.
#
# Usage:
#   ./scripts/run_v31_ablation.sh           # Full study (A + F + derive)
#   ./scripts/run_v31_ablation.sh A         # Just baseline
#   ./scripts/run_v31_ablation.sh F         # Just full pipeline
#   ./scripts/run_v31_ablation.sh derive    # Derive B-E from existing F
#
# Environment:
#   ATLAS_PARALLEL_TASKS: concurrent tasks (default 4)
#   ATLAS_LLM_PARALLEL: enable lock-free LLM requests (default 1)

set -euo pipefail
cd "$(dirname "$0")/.."

# Parallel config
export ATLAS_LLM_PARALLEL="${ATLAS_LLM_PARALLEL:-1}"
export ATLAS_PARALLEL_TASKS="${ATLAS_PARALLEL_TASKS:-4}"

DATE=$(date +%Y%m%d)
RESULT_BASE="benchmark/results"
LOG_DIR="${RESULT_BASE}/v31_ablation_${DATE}"
mkdir -p "$LOG_DIR"

A_RUN_ID="v31_ablation_A_${DATE}"
F_RUN_ID="v31_ablation_F_${DATE}"
DERIVED_DIR="${RESULT_BASE}/v31_ablation_derived_${DATE}"

echo "============================================================"
echo "  ATLAS V3.1 Ablation Study (Optimized)"
echo "  Date: $(date)"
echo "  Parallel: ${ATLAS_PARALLEL_TASKS} tasks"
echo "  Strategy: Run A + F, derive B/C/D/E"
echo "  Results: ${RESULT_BASE}/"
echo "============================================================"

preflight_check() {
    if ! curl -sf http://localhost:32735/health > /dev/null 2>&1; then
        echo "  ERROR: llama-server not healthy. Waiting 60s..."
        sleep 60
        if ! curl -sf http://localhost:32735/health > /dev/null 2>&1; then
            echo "  FATAL: llama-server still not healthy. Aborting."
            return 1
        fi
    fi
    echo "  Pre-flight: OK"
}

run_A() {
    echo ""
    echo "============================================================"
    echo "  Condition A (Baseline): Starting at $(date)"
    echo "  Run ID: ${A_RUN_ID}"
    echo "============================================================"

    preflight_check || return 1

    python3 -m benchmark.v3_runner \
        --run-id "${A_RUN_ID}" \
        --baseline \
        2>&1 | tee "${LOG_DIR}/condition_A.log"

    echo "  Condition A: Completed at $(date)"
}

run_F() {
    echo ""
    echo "============================================================"
    echo "  Condition F (Full Pipeline): Starting at $(date)"
    echo "  Run ID: ${F_RUN_ID}"
    echo "  NOTE: This stores all candidate codes for B-E derivation"
    echo "============================================================"

    preflight_check || return 1

    python3 -m benchmark.v3_runner \
        --run-id "${F_RUN_ID}" \
        --selection-strategy lens \
        2>&1 | tee "${LOG_DIR}/condition_F.log"

    echo "  Condition F: Completed at $(date)"
}

derive_BCDE() {
    local F_DIR="${RESULT_BASE}/${F_RUN_ID}"

    if [ ! -d "${F_DIR}/v3_lcb/per_task" ]; then
        echo "ERROR: Condition F results not found at ${F_DIR}"
        echo "Run Condition F first: ./scripts/run_v31_ablation.sh F"
        return 1
    fi

    echo ""
    echo "============================================================"
    echo "  Deriving Conditions B/C/D/E from F"
    echo "  Source: ${F_DIR}"
    echo "  Output: ${DERIVED_DIR}"
    echo "============================================================"

    python3 scripts/derive_ablation.py "${F_DIR}" "${DERIVED_DIR}"

    echo ""
    echo "  Derivation complete at $(date)"
}

print_summary() {
    echo ""
    echo "============================================================"
    echo "  ABLATION STUDY SUMMARY"
    echo "============================================================"

    # Condition A
    local A_DIR="${RESULT_BASE}/${A_RUN_ID}/v3_lcb/per_task"
    if [ -d "$A_DIR" ]; then
        local A_TOTAL A_PASS
        A_TOTAL=$(ls "${A_DIR}"/*.json 2>/dev/null | wc -l)
        A_PASS=$(grep -l '"passed": true' "${A_DIR}"/*.json 2>/dev/null | wc -l || echo 0)
        echo "  A (Baseline):          ${A_PASS}/${A_TOTAL} ($(python3 -c "print(f'{${A_PASS}/${A_TOTAL}*100:.1f}%' if ${A_TOTAL} > 0 else 'N/A')"))"
    fi

    # Derived conditions
    for COND in B C D E; do
        local SUMMARY="${DERIVED_DIR}/condition_${COND}/summary.json"
        if [ -f "$SUMMARY" ]; then
            python3 -c "
import json
d = json.load(open('${SUMMARY}'))
desc = d['description']
p = d['passed']
t = d['total_tasks']
pct = d['pass_rate'] * 100
print(f'  ${COND} ({desc}): {p}/{t} ({pct:.1f}%)')
"
        fi
    done

    # Condition F
    local F_DIR_TASKS="${RESULT_BASE}/${F_RUN_ID}/v3_lcb/per_task"
    if [ -d "$F_DIR_TASKS" ]; then
        local F_TOTAL F_PASS
        F_TOTAL=$(ls "${F_DIR_TASKS}"/*.json 2>/dev/null | wc -l)
        F_PASS=$(grep -l '"passed": true' "${F_DIR_TASKS}"/*.json 2>/dev/null | wc -l || echo 0)
        echo "  F (Full pipeline):     ${F_PASS}/${F_TOTAL} ($(python3 -c "print(f'{${F_PASS}/${F_TOTAL}*100:.1f}%' if ${F_TOTAL} > 0 else 'N/A')"))"
    fi

    echo ""
    echo "  Finished at: $(date)"
    echo "============================================================"
}

# Run requested step(s)
if [ $# -eq 0 ]; then
    echo "Running optimized ablation: A → F → derive B/C/D/E..."
    run_A
    run_F
    derive_BCDE
    print_summary
else
    for STEP in "$@"; do
        STEP_UPPER=$(echo "$STEP" | tr '[:lower:]' '[:upper:]')
        case "$STEP_UPPER" in
            A) run_A ;;
            F) run_F ;;
            DERIVE) derive_BCDE ;;
            SUMMARY) print_summary ;;
            ALL)
                run_A
                run_F
                derive_BCDE
                print_summary
                ;;
            *) echo "Unknown step: $STEP (valid: A, F, derive, summary, all)"; exit 1 ;;
        esac
    done
fi
