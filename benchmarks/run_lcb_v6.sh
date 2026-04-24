#!/bin/bash
# Run LiveCodeBench v6 through the ATLAS V3 pipeline
#
# This uses the existing v3_runner.py (benchmark/v3_runner.py) which handles:
# - PlanSearch for constraint generation
# - DivSampling for candidate diversity
# - Budget Forcing for token control
# - Sandbox code execution and verification
# - Geometric Lens scoring
# - Phase 2/3 adaptive compute and refinement
#
# The only change from v5: dataset loader uses release_v6
#
# Usage:
#   nohup ./run_lcb_v6.sh > logs/lcb_v6.log 2>&1 &
#   ./run_lcb_v6.sh --smoke       # 10-task smoke test
#   ./run_lcb_v6.sh --max-tasks 50  # limited run

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=========================================="
echo "LiveCodeBench v6 — ATLAS V3 Pipeline"
echo "Started: $(date)"
echo "=========================================="

# Temporarily patch the v3_runner to load v6 dataset
# This is done via environment variable to avoid modifying production code
export ATLAS_LCB_VERSION=v6

# Enable parallel task dispatch (server runs --parallel 4)
export ATLAS_LLM_PARALLEL=1
: "${ATLAS_PARALLEL_TASKS:=4}"; export ATLAS_PARALLEL_TASKS

# Run through V3 pipeline
python -c "
import sys
sys.path.insert(0, '.')
from benchmark.v3_runner import run_v3_benchmark
from benchmark.datasets.livecodebench import LiveCodeBenchV6Dataset

# Monkey-patch the load function
import benchmark.v3_runner as runner
original_load = runner.load_lcb_tasks
def load_v6():
    ds = LiveCodeBenchV6Dataset()
    ds.load()
    return ds.tasks
runner.load_lcb_tasks = load_v6

# Parse args and run
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--max-tasks', type=int, default=None)
parser.add_argument('--run-id', default='lcb_v6_$(date +%Y%m%d_%H%M%S)')
args, _ = parser.parse_known_args()

run_v3_benchmark(
    run_id=args.run_id,
    smoke_only=args.smoke,
    max_tasks=args.max_tasks,
)
" "$@"

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
