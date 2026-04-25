#!/bin/bash
# ATLAS V2 Benchmark Launch Script
#
# Usage:
#   ./benchmark/run_v2_benchmark.sh                  # Full run (phases 0-6)
#   ./benchmark/run_v2_benchmark.sh --smoke-only      # Smoke test only
#   ./benchmark/run_v2_benchmark.sh --start-phase 3   # Resume from phase 3
#   ./benchmark/run_v2_benchmark.sh --run-id v2_run_xxx  # Resume specific run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "============================================"
echo "  ATLAS V2 Benchmark Suite"
echo "  $(date -Iseconds)"
echo "============================================"
echo ""

# Check vLLM gen instance is reachable.
GEN_URL="${LLAMA_GEN_URL:-${LLAMA_URL:-http://localhost:8000}}"
echo "Checking vLLM gen at $GEN_URL..."
if ! curl -s --max-time 5 "$GEN_URL/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM gen instance not reachable at $GEN_URL"
    echo "Bring up the inference stack with: docker compose up vllm-gen vllm-embed"
    exit 1
fi
echo "  vLLM gen: OK"

# Check Geometric Lens is reachable
echo "Checking Geometric Lens..."
if ! curl -s --max-time 5 http://localhost:31144/health > /dev/null 2>&1; then
    echo "WARNING: Geometric Lens not reachable at localhost:31144"
    echo "V2 telemetry will be incomplete"
fi
echo "  Geometric Lens: OK"
echo ""

# Run the benchmark
python3 -m benchmark.v2_runner "$@"
