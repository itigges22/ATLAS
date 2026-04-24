#!/bin/bash
# ATLAS V3.0.1 Benchmark Suite — 15-Day Execution Plan
#
# Usage: nohup ./run_suite.sh > logs/suite_$(date +%Y%m%d).log 2>&1 &
#
# 7 benchmarks, ~15 days, sequential execution.
# Each benchmark has crash recovery — safe to restart at any point.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=========================================="
echo "ATLAS V3.0.1 Benchmark Suite (15-day plan)"
echo "Started: $(date)"
echo "=========================================="

# Phase 1: IFBench (Day 1, ~0.8 days)
echo ""
echo "=== Phase 1: IFBench (300 tasks, nothink) ==="
python -m benchmarks.v301_runner --benchmark ifbench
echo ""

# Phase 2: LiveCodeBench v6 (Days 1-4, ~2.6 days)
echo ""
echo "=== Phase 2: LiveCodeBench v6 (1055 tasks, V3 pipeline) ==="
benchmarks/run_lcb_v6.sh
echo ""

# Phase 3: GPQA Diamond (Days 4-6, ~2.0 days)
echo ""
echo "=== Phase 3: GPQA Diamond (198 tasks, thinking) ==="
python -m benchmarks.v301_runner --benchmark gpqa_diamond
echo ""

# Phase 4: IFEval (Days 6-7, ~1.3 days)
echo ""
echo "=== Phase 4: IFEval (541 tasks, thinking) ==="
python -m benchmarks.v301_runner --benchmark ifeval
echo ""

# Phase 5: MMLU-Pro sampled (Days 7-10, ~2.6 days)
echo ""
echo "=== Phase 5: MMLU-Pro sampled (3000 tasks, nothink) ==="
python -m benchmarks.v301_runner --benchmark mmlu_pro --limit 3000
echo ""

# Phase 6: C-Eval (Days 10-11, ~1.2 days)
echo ""
echo "=== Phase 6: C-Eval (1346 tasks, nothink) ==="
python -m benchmarks.v301_runner --benchmark c_eval
echo ""

# Phase 7: MMLU-Redux (Days 11-15, ~4.9 days)
echo ""
echo "=== Phase 7: MMLU-Redux (5600 tasks, nothink) ==="
python -m benchmarks.v301_runner --benchmark mmlu_redux
echo ""

echo "=========================================="
echo "Suite completed: $(date)"
echo "=========================================="
