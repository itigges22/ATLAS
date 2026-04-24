#!/bin/bash
# ATLAS V3.0.1 Benchmark Runner
# Usage: ./run_benchmark.sh <benchmark_name> [--limit N]
#
# Runs a benchmark in the background with logging.
# Output goes to benchmarks/logs/<benchmark>_<timestamp>.log

set -euo pipefail

BENCHMARK="${1:?Usage: $0 <benchmark_name> [--limit N]}"
shift

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$(dirname "$0")/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${BENCHMARK}_${TIMESTAMP}.log"

echo "Starting benchmark: $BENCHMARK"
echo "Log: $LOG_FILE"
echo "Args: $@"

cd "$(dirname "$0")/.."

nohup python -m benchmarks.v301_runner --benchmark "$BENCHMARK" "$@" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"
echo "$PID" > "$LOG_DIR/${BENCHMARK}.pid"

echo "Benchmark $BENCHMARK started in background (PID $PID)"
echo "Monitor: tail -f $LOG_FILE"
