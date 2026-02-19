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

# Check llama-server is reachable
echo "Checking llama-server..."
if ! curl -s --max-time 5 http://localhost:32735/health > /dev/null 2>&1; then
    echo "ERROR: llama-server not reachable at localhost:32735"
    echo "Is the server running?"
    exit 1
fi
echo "  llama-server: OK"

# Check RAG API is reachable
echo "Checking RAG API..."
if ! curl -s --max-time 5 http://localhost:31144/health > /dev/null 2>&1; then
    echo "WARNING: RAG API not reachable at localhost:31144"
    echo "V2 telemetry will be incomplete"
fi
echo "  RAG API: OK"
echo ""

# Run the benchmark
python3 -m benchmark.v2_runner "$@"
