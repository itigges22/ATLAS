#!/bin/bash
# Run LiveCodeBench v6 through the ATLAS V3 pipeline against vLLM.
#
# vLLM gen + embed instances are brought up by the container entrypoint
# (benchmarks/h200/entrypoint.sh) before this script is invoked, so we
# don't need to start or restart anything here — just point the V3
# runner at LLAMA_GEN_URL and let it dispatch.
#
# Usage:
#   ./run_lcb_v6.sh                    # full 1054-task LCB v6 sweep
#   ./run_lcb_v6.sh --smoke            # 10-task smoke
#   ./run_lcb_v6.sh --max-tasks 50     # limited

set -euo pipefail
cd "$(dirname "$0")/.."

: "${LLAMA_GEN_URL:=${LLAMA_URL:-http://localhost:8000}}"
: "${LLAMA_EMBED_URL:=http://localhost:8001}"
: "${ATLAS_PARALLEL_TASKS:=16}"
export LLAMA_GEN_URL LLAMA_EMBED_URL ATLAS_PARALLEL_TASKS
# Back-compat — runners that still read LLAMA_URL get the gen URL.
export LLAMA_URL="$LLAMA_GEN_URL"
export ATLAS_LLM_PARALLEL=1
export ATLAS_LCB_VERSION=v6

echo "=========================================="
echo "LiveCodeBench v6 — ATLAS V3 Pipeline (vLLM)"
echo "Started: $(date)"
echo "Gen URL:   $LLAMA_GEN_URL"
echo "Embed URL: $LLAMA_EMBED_URL"
echo "Parallel:  $ATLAS_PARALLEL_TASKS"
echo "=========================================="

# Sanity: gen + embed both reachable.
if ! curl -s --max-time 5 "$LLAMA_GEN_URL/health" | grep -q ok; then
    echo "ERROR: vLLM gen instance not healthy at $LLAMA_GEN_URL" >&2
    exit 1
fi
if ! curl -s --max-time 5 "$LLAMA_EMBED_URL/health" | grep -q ok; then
    echo "WARNING: vLLM embed instance not healthy at $LLAMA_EMBED_URL — Lens scoring will be unavailable." >&2
fi

python -c "
import sys, datetime
sys.path.insert(0, '.')
from benchmark.v3_runner import run_v3_benchmark
from benchmark.datasets.livecodebench import LiveCodeBenchV6Dataset

import benchmark.v3_runner as runner
def load_v6():
    ds = LiveCodeBenchV6Dataset()
    ds.load()
    return ds.tasks
runner.load_lcb_tasks = load_v6

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--max-tasks', type=int, default=None)
parser.add_argument('--run-id', default='lcb_v6_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
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
