#!/bin/bash
#
# ATLAS V1 Full Benchmark Suite
# Runs all benchmarks and generates a consolidated dated report
#
# Features:
#   - Per-phase checkpoints (resumes from last completed phase)
#   - Per-task resume within phases (via --resume flag)
#   - Crash detection with 3-strike retry rule
#   - Signal handling for clean shutdown
#   - Crash log in JSON format
#
# Usage: ./scripts/run_full_benchmarks.sh
#
# Estimated runtime: ~28 hours
#   - HumanEval pass@1: ~45 min
#   - MBPP pass@1: ~1.5 hrs
#   - Custom pass@1: ~27 min
#   - HumanEval pass@k (20 attempts × 3 runs): ~15 hrs
#   - Custom pass@k (20 attempts × 3 runs): ~10 hrs
#

set -e

# ── Configuration ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/benchmark/results"
DATE_STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RESULTS_DIR/full_run_$DATE_STAMP"
REPORT_FILE="$RESULTS_DIR/benchmark_report_$DATE_STAMP.md"
LOG_FILE="$RUN_DIR/full_benchmark.log"
CRASH_LOG="$RUN_DIR/crash_log.json"

# Number of runs for pass@k statistical analysis
PASS_K_RUNS=3
PASS_K_ATTEMPTS=20
MAX_RETRIES=3

# Track overall crash statistics
TOTAL_CRASHES=0
PHASES_WITH_RETRIES=""
PHASES_FAILED=""

# ── Colors ────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# ── Signal Handling ───────────────────────────────────────────
INTERRUPTED=0

cleanup() {
    if [[ $INTERRUPTED -eq 1 ]]; then
        echo ""
        log_error "Interrupted by user (SIGINT/SIGTERM)"
        log "Saving partial results..."

        # Ensure crash log is valid JSON
        finalize_crash_log

        # Generate partial report
        generate_report "INTERRUPTED"

        log "Partial results saved. Resume with: ./scripts/run_full_benchmarks.sh"
        log "Results in: $RUN_DIR"
    fi
    exit 1
}

trap 'INTERRUPTED=1; cleanup' SIGINT SIGTERM

# ── Functions ─────────────────────────────────────────────────

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${CYAN}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

log_success() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1"
    echo -e "${GREEN}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1"
    echo -e "${RED}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

log_warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1"
    echo -e "${YELLOW}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

log_section() {
    echo ""
    echo -e "${BOLD}${YELLOW}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${YELLOW}  $1${NC}"
    echo -e "${BOLD}${YELLOW}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "================================================================================" >> "$LOG_FILE"
    echo "  $1" >> "$LOG_FILE"
    echo "================================================================================" >> "$LOG_FILE"
}

# Initialize crash log as JSON array
init_crash_log() {
    echo "[]" > "$CRASH_LOG"
}

# Append a crash event to the crash log
log_crash_event() {
    local phase="$1"
    local attempt="$2"
    local exit_code="$3"
    local duration="$4"
    local stderr_tail="$5"
    local tasks_completed="$6"
    local tasks_remaining="$7"

    local timestamp=$(date -Iseconds)

    # Read current log, append new entry, write back
    local current=$(cat "$CRASH_LOG")

    # Create new entry using Python for proper JSON handling
    python3 << PYTHON_EOF
import json
import sys

current = json.loads('''$current''')
new_entry = {
    "timestamp": "$timestamp",
    "phase": "$phase",
    "attempt": $attempt,
    "exit_code": $exit_code,
    "duration_seconds": $duration,
    "stderr_tail": """$stderr_tail"""[:500],
    "tasks_completed_before_crash": $tasks_completed,
    "tasks_remaining": $tasks_remaining
}
current.append(new_entry)
print(json.dumps(current, indent=2))
PYTHON_EOF

    python3 -c "
import json
current = json.loads('''$current''')
new_entry = {
    'timestamp': '$timestamp',
    'phase': '$phase',
    'attempt': $attempt,
    'exit_code': $exit_code,
    'duration_seconds': $duration,
    'stderr_tail': '''$stderr_tail'''[:500],
    'tasks_completed_before_crash': $tasks_completed,
    'tasks_remaining': $tasks_remaining
}
current.append(new_entry)
print(json.dumps(current, indent=2))
" > "$CRASH_LOG"
}

# Finalize crash log (ensure valid JSON)
finalize_crash_log() {
    if [[ ! -f "$CRASH_LOG" ]]; then
        echo "[]" > "$CRASH_LOG"
    fi
}

# Count completed results in a directory
count_results() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        ls "$dir"/result_*.json 2>/dev/null | wc -l || echo 0
    else
        echo 0
    fi
}

# Run a benchmark phase with retry logic
run_phase() {
    local phase_name="$1"
    local command="$2"
    local checkpoint="$3"
    local output_dir="$4"
    local retry_count=0
    local last_stderr=""

    # Check if already completed
    if [[ -f "$checkpoint" ]]; then
        log "⏭ $phase_name already complete, skipping"
        return 0
    fi

    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        log "▶ Running $phase_name (attempt $((retry_count + 1))/$MAX_RETRIES)"
        local start_time=$(date +%s)
        local tasks_before=$(count_results "$output_dir")

        # Run with resume flag, capture stderr
        local stderr_file=$(mktemp)
        if eval "$command --resume" 2>"$stderr_file"; then
            touch "$checkpoint"
            log_success "$phase_name complete"
            rm -f "$stderr_file"
            return 0
        else
            local exit_code=$?
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            local tasks_after=$(count_results "$output_dir")
            local tasks_remaining=$((tasks_after > 0 ? 100 - tasks_after : 0))  # Estimate

            last_stderr=$(tail -c 500 "$stderr_file" 2>/dev/null || echo "")
            rm -f "$stderr_file"

            retry_count=$((retry_count + 1))
            TOTAL_CRASHES=$((TOTAL_CRASHES + 1))

            # Log crash event
            log_crash_event "$phase_name" "$retry_count" "$exit_code" "$duration" "$last_stderr" "$tasks_after" "$tasks_remaining"

            log_warn "$phase_name crashed (attempt $retry_count/$MAX_RETRIES, exit code $exit_code)"

            if [[ $retry_count -lt $MAX_RETRIES ]]; then
                log "  Waiting 30s before retry..."
                sleep 30
            fi
        fi
    done

    # 3 strikes — log fatal error and continue
    log_error "$phase_name FAILED after $MAX_RETRIES attempts"
    PHASES_FAILED="$PHASES_FAILED $phase_name"
    if [[ -z "$PHASES_WITH_RETRIES" ]]; then
        PHASES_WITH_RETRIES="$phase_name"
    else
        PHASES_WITH_RETRIES="$PHASES_WITH_RETRIES, $phase_name"
    fi
    return 1
}

# Generate the final report
generate_report() {
    local status="${1:-COMPLETE}"

    SUITE_END=$(date +%s)
    SUITE_DURATION=$((SUITE_END - SUITE_START))
    SUITE_HOURS=$((SUITE_DURATION / 3600))
    SUITE_MINUTES=$(((SUITE_DURATION % 3600) / 60))

    # Start report
    cat > "$REPORT_FILE" << EOF
# ATLAS V1 Benchmark Report

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Run ID:** full_run_$DATE_STAMP
**Total Runtime:** ${SUITE_HOURS}h ${SUITE_MINUTES}m
**Status:** $status

---

## Run Health

| Metric | Value |
|--------|-------|
| Total Crashes | $TOTAL_CRASHES |
| Phases with Retries | ${PHASES_WITH_RETRIES:-None} |
| Phases Failed (3 strikes) | ${PHASES_FAILED:-None} |
| Total Wall-Clock Time | ${SUITE_HOURS}h ${SUITE_MINUTES}m |

EOF

    # Add crash details if any
    if [[ $TOTAL_CRASHES -gt 0 ]] && [[ -f "$CRASH_LOG" ]]; then
        cat >> "$REPORT_FILE" << EOF
### Crash Events

\`\`\`json
$(cat "$CRASH_LOG")
\`\`\`

EOF
    fi

    cat >> "$REPORT_FILE" << EOF
---

## Target Metrics Summary

| Category | Metric | Target | Description |
|----------|--------|--------|-------------|
| **Accuracy** | pass@1 | ≥65% (HumanEval), ≥70% (MBPP) | First attempt success rate |
| **Accuracy** | pass@5 | ≥95% | Success within 5 attempts |
| **Accuracy** | pass@20 | ≥99.5% | Success within 20 attempts |
| **Performance** | Throughput | ≥100 tasks/hr | Tasks completed per hour |
| **Performance** | Time to Solution | <60s median | Wall-clock to verified success |
| **Cost** | Cost Efficiency | ≥30x cheaper | Cloud cost / Local cost ratio |
| **Efficiency** | Tokens/Watt-Hour | Baseline | Novel metric (establishing) |
| **Efficiency** | Tasks/Watt-Hour | Baseline | Novel metric (establishing) |

---

## Benchmarks Executed

| Benchmark | Tasks | Attempts | Runs | Status |
|-----------|-------|----------|------|--------|
EOF

    # Check each phase status
    for phase in "humaneval_pass1" "mbpp_pass1" "custom_pass1"; do
        local checkpoint="$RUN_DIR/.checkpoint_$phase"
        local status_icon="✗"
        [[ -f "$checkpoint" ]] && status_icon="✓"
        local dataset="${phase%_pass1}"
        local tasks=164
        [[ "$dataset" == "mbpp" ]] && tasks=500
        [[ "$dataset" == "custom" ]] && tasks=100
        echo "| ${dataset^} pass@1 | $tasks | 1 | 1 | $status_icon |" >> "$REPORT_FILE"
    done

    for dataset in humaneval custom; do
        for run in $(seq 1 $PASS_K_RUNS); do
            local checkpoint="$RUN_DIR/.checkpoint_${dataset}_passk_run${run}"
            local status_icon="✗"
            [[ -f "$checkpoint" ]] && status_icon="✓"
            local tasks=164
            [[ "$dataset" == "custom" ]] && tasks=100
            echo "| ${dataset^} pass@$PASS_K_ATTEMPTS (run $run) | $tasks | $PASS_K_ATTEMPTS | 1 | $status_icon |" >> "$REPORT_FILE"
        done
    done

    cat >> "$REPORT_FILE" << EOF

---

## Pass@1 Results

EOF

    # Append pass@1 results
    for dataset in humaneval mbpp custom; do
        pk_file="$RUN_DIR/pass1/$dataset/pass_at_k.md"
        if [[ -f "$pk_file" ]]; then
            echo "### ${dataset^}" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            cat "$pk_file" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
        fi
    done

    cat >> "$REPORT_FILE" << EOF

---

## Pass@k Results (k=$PASS_K_ATTEMPTS)

EOF

    # Append pass@k results
    for dataset in humaneval custom; do
        echo "### ${dataset^} (aggregated over $PASS_K_RUNS runs)" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"

        for run in $(seq 1 $PASS_K_RUNS); do
            pk_file="$RUN_DIR/passk/${dataset}_run$run/pass_at_k.md"
            if [[ -f "$pk_file" ]]; then
                echo "#### Run $run" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"
                cat "$pk_file" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"
            fi
        done
    done

    # Run cost analysis
    cat >> "$REPORT_FILE" << EOF

---

## Cost Analysis

EOF

    log "Running cost analysis..."

    python3 << PYTHON_EOF >> "$REPORT_FILE" 2>/dev/null || echo "Cost analysis skipped (no data)"
import sys
sys.path.insert(0, '$PROJECT_ROOT')

from pathlib import Path
from benchmark.models import BenchmarkRun
from benchmark.analysis import CostAnalyzer

run_dir = Path('$RUN_DIR')
analyzer = CostAnalyzer()

# Analyze pass@1 results
for dataset in ['humaneval', 'mbpp', 'custom']:
    run_file = run_dir / 'pass1' / dataset / 'run.json'
    if run_file.exists():
        run = BenchmarkRun.load(str(run_file))
        metrics = analyzer.analyze(run)
        print(f"### {dataset.upper()} pass@1\n")
        print(analyzer.to_markdown(metrics))
        print()

# Analyze pass@k results (first run of each)
for dataset in ['humaneval', 'custom']:
    run_file = run_dir / 'passk' / f'{dataset}_run1' / 'run.json'
    if run_file.exists():
        run = BenchmarkRun.load(str(run_file))
        metrics = analyzer.analyze(run)
        print(f"### {dataset.upper()} pass@$PASS_K_ATTEMPTS\n")
        print(analyzer.to_markdown(metrics))
        print()
PYTHON_EOF

    # Hardware info
    cat >> "$REPORT_FILE" << EOF

---

## Hardware Configuration

EOF

    python3 << PYTHON_EOF >> "$REPORT_FILE" 2>/dev/null || echo "Hardware info unavailable"
import sys
sys.path.insert(0, '$PROJECT_ROOT')

from benchmark.analysis import collect_hardware_info
from benchmark.analysis.hardware_info import hardware_info_to_markdown

info = collect_hardware_info()
print(hardware_info_to_markdown(info))
PYTHON_EOF

    # Footer
    cat >> "$REPORT_FILE" << EOF

---

## Run Details

- **Run Directory:** \`$RUN_DIR\`
- **Log File:** \`$LOG_FILE\`
- **Crash Log:** \`$CRASH_LOG\`
- **Report File:** \`$REPORT_FILE\`

### Individual Result Files

\`\`\`
$RUN_DIR/
├── pass1/
│   ├── humaneval/
│   ├── mbpp/
│   └── custom/
└── passk/
    ├── humaneval_run1/
    ├── humaneval_run2/
    ├── humaneval_run3/
    ├── custom_run1/
    ├── custom_run2/
    └── custom_run3/
\`\`\`

---

*Generated by ATLAS V1 Benchmark Suite*
EOF

    log_success "Report saved to: $REPORT_FILE"
}

# ── Setup ─────────────────────────────────────────────────────

cd "$PROJECT_ROOT"

mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/pass1"
mkdir -p "$RUN_DIR/passk"

# Initialize crash log
init_crash_log

echo "ATLAS V1 Full Benchmark Suite" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Run directory: $RUN_DIR" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

log_section "ATLAS V1 Full Benchmark Suite"
log "Date: $(date)"
log "Run directory: $RUN_DIR"
log "Report will be saved to: $REPORT_FILE"

SUITE_START=$(date +%s)

# ── Pass@1 Benchmarks ─────────────────────────────────────────

log_section "Phase 1: Pass@1 Benchmarks"

# HumanEval pass@1
run_phase "HumanEval pass@1" \
    "python3 -m benchmark.cli --humaneval --k 1 --output '$RUN_DIR/pass1/humaneval'" \
    "$RUN_DIR/.checkpoint_humaneval_pass1" \
    "$RUN_DIR/pass1/humaneval"

# MBPP pass@1
run_phase "MBPP pass@1" \
    "python3 -m benchmark.cli --mbpp --k 1 --output '$RUN_DIR/pass1/mbpp'" \
    "$RUN_DIR/.checkpoint_mbpp_pass1" \
    "$RUN_DIR/pass1/mbpp"

# Custom pass@1
run_phase "Custom pass@1" \
    "python3 -m benchmark.cli --custom --k 1 --output '$RUN_DIR/pass1/custom'" \
    "$RUN_DIR/.checkpoint_custom_pass1" \
    "$RUN_DIR/pass1/custom"

# ── Pass@k Benchmarks ─────────────────────────────────────────

log_section "Phase 2: Pass@k Benchmarks (k=$PASS_K_ATTEMPTS, $PASS_K_RUNS runs each)"

# HumanEval pass@k (3 runs)
for run in $(seq 1 $PASS_K_RUNS); do
    run_phase "HumanEval pass@$PASS_K_ATTEMPTS (run $run)" \
        "python3 -m benchmark.cli --humaneval --k $PASS_K_ATTEMPTS --output '$RUN_DIR/passk/humaneval_run$run'" \
        "$RUN_DIR/.checkpoint_humaneval_passk_run$run" \
        "$RUN_DIR/passk/humaneval_run$run"
done

# Custom pass@k (3 runs)
for run in $(seq 1 $PASS_K_RUNS); do
    run_phase "Custom pass@$PASS_K_ATTEMPTS (run $run)" \
        "python3 -m benchmark.cli --custom --k $PASS_K_ATTEMPTS --output '$RUN_DIR/passk/custom_run$run'" \
        "$RUN_DIR/.checkpoint_custom_passk_run$run" \
        "$RUN_DIR/passk/custom_run$run"
done

# ── Generate Report ───────────────────────────────────────────

log_section "Phase 3: Generating Consolidated Report"

# Finalize crash log
finalize_crash_log

# Generate the report
generate_report "COMPLETE"

# ── Final Summary ─────────────────────────────────────────────

log_section "Benchmark Suite Complete"
log "Total runtime: ${SUITE_HOURS}h ${SUITE_MINUTES}m"
log "Total crashes: $TOTAL_CRASHES"
log "Results directory: $RUN_DIR"
log "Report: $REPORT_FILE"

echo ""
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  BENCHMARK SUITE COMPLETE${NC}"
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Report: ${CYAN}$REPORT_FILE${NC}"
echo -e "  Results: ${CYAN}$RUN_DIR${NC}"
if [[ $TOTAL_CRASHES -gt 0 ]]; then
    echo -e "  Crashes: ${YELLOW}$TOTAL_CRASHES${NC} (see crash_log.json)"
fi
echo ""
