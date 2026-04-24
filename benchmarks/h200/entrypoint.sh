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
echo "--- Starting Geometric Lens service on port ${LENS_PORT:-31144} ---"
LENS_LOG=/tmp/lens-service.log
cd /workspace/ATLAS/geometric-lens
GEOMETRIC_LENS_ENABLED=true \
LLAMA_URL="http://localhost:${SERVER_PORT}" \
LLAMA_EMBED_URL="http://localhost:${SERVER_PORT}" \
PROJECT_DATA_DIR=/data/projects \
nohup python -m uvicorn main:app --host 0.0.0.0 --port "${LENS_PORT:-31144}" \
    > "$LENS_LOG" 2>&1 &
LENS_PID=$!
cd /workspace/ATLAS
echo "Lens service PID: $LENS_PID (log: $LENS_LOG)"

# 2a. Wait for Lens health (much faster to start than llama-server since no model load)
echo "--- Waiting for Lens service (timeout 120s) ---"
for i in $(seq 1 24); do
    if curl -s --max-time 2 "http://localhost:${LENS_PORT:-31144}/health" 2>/dev/null | grep -qE "ok|healthy"; then
        echo "Lens service healthy after ${i}×5s."
        break
    fi
    printf "."
    sleep 5
done
if ! curl -s "http://localhost:${LENS_PORT:-31144}/health" 2>/dev/null | grep -qE "ok|healthy"; then
    echo "WARNING: Lens service did not come up — ATLAS will run without Lens scoring."
    echo "Lens service log tail:"
    tail -30 "$LENS_LOG" >&2 || true
    # Don't exit; V3 pipeline degrades gracefully without Lens.
    export GEOMETRIC_LENS_ENABLED=false
fi

# 2b. Start llama-server
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

# 3. Wait for llama-server healthy
echo ""
echo "--- Waiting for llama-server to be healthy (timeout 600s) ---"
for i in $(seq 1 120); do
    if curl -s --max-time 2 "http://localhost:${SERVER_PORT}/health" 2>/dev/null | grep -q ok; then
        echo "llama-server healthy after ${i}×5s."
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

# Both runners read LLAMA_URL (not ATLAS_LLM_URL) — keep both for safety.
export LLAMA_URL="http://localhost:${SERVER_PORT}"
export ATLAS_LLM_URL="http://localhost:${SERVER_PORT}"
export RAG_API_URL="http://localhost:${LENS_PORT:-31144}"
export BENCHMARK_PARALLEL
export ATLAS_LLM_PARALLEL
export ATLAS_PARALLEL_TASKS
export GEOMETRIC_LENS_ENABLED
# The V3 pipeline uses the Lens when GEOMETRIC_LENS_ENABLED=true and degrades
# gracefully if the Lens service dies mid-run. The entrypoint sets GEOMETRIC_LENS_ENABLED=false
# above if the Lens failed to start so the pipeline doesn't waste time calling a dead service.
# Numbers when Lens is live = full ATLAS V3. Numbers when Lens is down = "V3 minus Lens" — acceptable
# for V3.1 since the Lens is trained on Q6_K embeddings, which matches our
# current Q6_K inference, but running it as a separate service is V3.2 work.
export GEOMETRIC_LENS_ENABLED

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
    echo "SHUTDOWN_ON_COMPLETE=1 — exiting entrypoint in 60s."
    echo "Note: to actually stop a RunPod pod and halt billing, use the RunPod"
    echo "web UI, or call runpodctl stop <pod-id> from outside. Container exit"
    echo "alone does not stop the pod."
    sleep 60
    exit 0
else
    echo "SHUTDOWN_ON_COMPLETE=0 — container will stay alive for you to rsync results."
    # Keep container alive so RunPod doesn't terminate and lose results.
    # Tail the llama log to keep PID 1 attached. Stop the pod from RunPod UI when done.
    exec tail -f "$LLAMA_LOG"
fi
