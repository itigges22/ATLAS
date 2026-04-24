#!/bin/bash
# ATLAS benchmark runner entrypoint.
# Designed for RunPod / Lambda / any cloud NVIDIA pod with this image.
#
# Environment variables (with sensible defaults in the Dockerfile):
#   MODEL_PATH       — where the GGUF lives (default /workspace/models/...)
#   MODEL_URL        — where to download it from if DOWNLOAD_MODEL=1
#   DOWNLOAD_MODEL   — 1 to fetch model if not present, 0 to require it mounted
#   SERVER_PORT      — llama-server port (default 8000)
#   SERVER_PARALLEL  — slots (default 16 for H200; drop to 4 on consumer GPUs)
#   SERVER_CONTEXT   — total KV context (default 262144 = 16K per slot)
#   BENCHMARK_PARALLEL — runner thread count (match SERVER_PARALLEL)
#   ATLAS_PARALLEL_TASKS — V3 pipeline concurrent tasks
#   MODE             — atlas_only | baseline_only | all (default atlas_only)
#   RESULT_TAR       — where to write the final archive
#   SHUTDOWN_ON_COMPLETE — 1 = sudo shutdown after results, 0 = leave running

set -euo pipefail

echo "============================================="
echo "ATLAS Benchmark Runner — container entrypoint"
echo "============================================="
date
echo "Mode:              $MODE"
echo "Model path:        $MODEL_PATH"
echo "Server parallel:   $SERVER_PARALLEL"
echo "Server context:    $SERVER_CONTEXT"
echo "Benchmark parallel: $BENCHMARK_PARALLEL"
echo "---------------------------------------------"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || echo "no nvidia-smi"
echo "============================================="
echo ""

# 1. Ensure model is present
if [[ ! -f "$MODEL_PATH" ]]; then
    if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
        echo "--- Downloading model from $MODEL_URL (~7GB) ---"
        mkdir -p "$(dirname "$MODEL_PATH")"
        wget --progress=dot:giga -O "$MODEL_PATH" "$MODEL_URL"
    else
        echo "ERROR: Model not found at $MODEL_PATH and DOWNLOAD_MODEL=0." >&2
        echo "Mount a volume at /workspace/models or set DOWNLOAD_MODEL=1." >&2
        exit 1
    fi
fi

echo "Model present: $(ls -lh "$MODEL_PATH" | awk '{print $5}')"

# 2. Start llama-server in background
echo ""
echo "--- Starting llama-server ---"
LLAMA_LOG=/tmp/llama-server.log
nohup /usr/local/bin/llama-server \
    -m "$MODEL_PATH" \
    -c "$SERVER_CONTEXT" \
    -ctk q8_0 -ctv q4_0 \
    --parallel "$SERVER_PARALLEL" --cont-batching -ngl 99 \
    --host 0.0.0.0 --port "$SERVER_PORT" \
    --flash-attn on --mlock \
    -b 4096 -ub 4096 \
    --ctx-checkpoints 0 --no-cache-prompt \
    --embeddings --jinja --no-warmup \
    > "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
echo "llama-server PID: $LLAMA_PID (log: $LLAMA_LOG)"

# 3. Wait for healthy
echo ""
echo "--- Waiting for server to be healthy (timeout 600s) ---"
for i in $(seq 1 120); do
    if curl -s --max-time 2 "http://localhost:${SERVER_PORT}/health" 2>/dev/null | grep -q ok; then
        echo "Server healthy after ${i}×5s."
        break
    fi
    printf "."
    sleep 5
done
if ! curl -s "http://localhost:${SERVER_PORT}/health" 2>/dev/null | grep -q ok; then
    echo ""
    echo "ERROR: llama-server did not come up. Tail of log:" >&2
    tail -30 "$LLAMA_LOG" >&2
    exit 1
fi

export ATLAS_LLM_URL="http://localhost:${SERVER_PORT}"
export BENCHMARK_PARALLEL
export ATLAS_LLM_PARALLEL
export ATLAS_PARALLEL_TASKS

# 4. Optional smoke test (set SKIP_SMOKE=1 to skip)
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    echo ""
    echo "--- Smoke test: 3 C-Eval tasks ---"
    BENCHMARK_PARALLEL=4 timeout 300 python -m benchmarks.v301_runner \
        --benchmark c_eval --limit 3 --output-dir /tmp/smoke_out \
        > /tmp/smoke.log 2>&1 \
        && echo "Smoke OK" \
        || { echo "Smoke failed:"; tail -30 /tmp/smoke.log; exit 1; }
    rm -rf /tmp/smoke_out
fi

# 5. Run the benchmarks
LOG_DIR="/workspace/results/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

if [[ "$MODE" == "baseline_only" || "$MODE" == "all" ]]; then
    echo ""
    echo "--- Running baseline ---"
    ./benchmarks/run_full_baseline.sh 2>&1 | tee "$LOG_DIR/baseline.log" || true
fi

if [[ "$MODE" == "atlas_only" || "$MODE" == "all" ]]; then
    echo ""
    echo "--- Running ATLAS V3 pipeline (LCB v6) ---"
    ./benchmarks/run_lcb_v6.sh 2>&1 | tee "$LOG_DIR/atlas_lcb.log" || true
fi

# 6. Archive results
echo ""
echo "--- Archiving results ---"
mkdir -p "$(dirname "$RESULT_TAR")"
tar czf "$RESULT_TAR" \
    benchmarks/section_*/*/responses.jsonl \
    benchmarks/section_*/*/results.json \
    benchmarks/section_*/*/traces \
    benchmarks/logs \
    2>/dev/null || true
ls -lh "$RESULT_TAR" || true

# 7. Stop llama-server cleanly
kill "$LLAMA_PID" 2>/dev/null || true

echo ""
echo "============================================="
echo "Done. Results tarball: $RESULT_TAR"
echo "Pull it back via: runpodctl receive / ssh rsync / web download"
echo "============================================="

if [[ "$SHUTDOWN_ON_COMPLETE" == "1" ]]; then
    echo "SHUTDOWN_ON_COMPLETE=1 — stopping pod in 60s. Ctrl-C to abort."
    sleep 60
    # RunPod handles pod stop via API; sudo shutdown works on most providers.
    sudo shutdown -h now 2>/dev/null || shutdown -h now 2>/dev/null || true
else
    echo "SHUTDOWN_ON_COMPLETE=0 — container will stay alive for you to rsync results."
    # Keep container alive so RunPod doesn't terminate and lose results.
    # Tail the llama log to keep PID 1 attached. Ctrl-C or stop pod when done.
    exec tail -f "$LLAMA_LOG"
fi
