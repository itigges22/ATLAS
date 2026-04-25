#!/bin/bash
# ATLAS benchmark runner entrypoint (vLLM-based).
#
# Brings up three services in this order:
#   1. Geometric Lens (FastAPI on port 31144) — fast, no model load
#   2. vLLM EMBED instance on $EMBED_PORT (--runner pooling --convert embed,
#      4096-dim hidden states for the Lens)
#   3. vLLM GEN instance on $GEN_PORT (--reasoning-parser qwen3, prefix caching)
#
# Why this order: Lens starts instantly. EMBED instance is smaller and used by Lens
# for code-to-embedding extraction. GEN is the largest and most expensive — start
# last so we don't burn GPU minutes if the smaller pieces fail health checks.
#
# Environment variables (defaults set in Dockerfile ENV):
#   MODEL_NAME         — HF model id (QuantTrio/Qwen3.5-9B-AWQ)
#   MODEL_PATH         — local cache path (/workspace/models/Qwen3.5-9B-AWQ)
#   DOWNLOAD_MODEL     — 1 to fetch from HF on first run, 0 to require mounted
#   GEN_PORT           — vLLM gen port (8000)
#   EMBED_PORT         — vLLM embed port (8001)
#   GEN_MAX_NUM_SEQS   — concurrent gen requests (32 for H100, drop to 4-8 on consumer)
#   GEN_MAX_MODEL_LEN  — gen context (32768; raise for long V3 traces)
#   GEN_GPU_MEM_UTIL   — fraction of VRAM gen instance can claim (0.55)
#   EMBED_*            — same shape, smaller values
#   MODE               — atlas_only | baseline_only | all
#   SKIP_SMOKE         — 1 to skip the C-Eval pre-flight test

set -euo pipefail

echo "============================================="
echo "ATLAS Benchmark Runner (vLLM) — entrypoint"
echo "============================================="
date
echo "Mode:               $MODE"
echo "Model:              $MODEL_NAME"
echo "Model path:         $MODEL_PATH"
echo "Gen port/seqs/len:  $GEN_PORT / $GEN_MAX_NUM_SEQS / $GEN_MAX_MODEL_LEN"
echo "Embed port/seqs/len: $EMBED_PORT / $EMBED_MAX_NUM_SEQS / $EMBED_MAX_MODEL_LEN"
echo "Benchmark parallel: $BENCHMARK_PARALLEL"
echo "---------------------------------------------"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || echo "no nvidia-smi"
echo "============================================="
echo ""

# 1. Ensure model is present. AWQ is ~12 GiB; pulling on first run is fine for
#    cloud pods. For repeat runs, mount a volume at /workspace/models to avoid the download.
#    Set HF_TOKEN for gated/private repos (passed through to huggingface_hub).
if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
        echo "--- Downloading $MODEL_NAME from HuggingFace (~12 GiB) ---"
        mkdir -p "$MODEL_PATH"
        # huggingface_hub CLI handles sharded weights + retries.
        pip3 install -q huggingface_hub
        python -c "
import os
from huggingface_hub import snapshot_download
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
snapshot_download(
    repo_id='$MODEL_NAME',
    local_dir='$MODEL_PATH',
    local_dir_use_symlinks=False,
    max_workers=8,
    token=token,
)
print('download complete')
"
    else
        echo "ERROR: Model not found at $MODEL_PATH and DOWNLOAD_MODEL=0." >&2
        echo "Mount a volume at /workspace/models or set DOWNLOAD_MODEL=1." >&2
        exit 1
    fi
fi
echo "Model present at $MODEL_PATH ($(du -sh "$MODEL_PATH" | cut -f1))"

# 1a. Clear flashinfer cache — required when an old container layer carried a
#     stale cache that breaks ninja build. Cheap to do unconditionally.
rm -rf "$HOME/.cache/flashinfer" 2>/dev/null || true

# 2. Geometric Lens service (FastAPI). Boots in ~5s — start it first.
echo ""
echo "--- Starting Geometric Lens service on port ${LENS_PORT:-31144} ---"
LENS_LOG=/tmp/lens-service.log
cd /workspace/ATLAS/geometric-lens
GEOMETRIC_LENS_ENABLED=true \
LLAMA_URL="http://localhost:${EMBED_PORT}" \
LLAMA_EMBED_URL="http://localhost:${EMBED_PORT}" \
PROJECT_DATA_DIR=/data/projects \
nohup python -m uvicorn main:app --host 0.0.0.0 --port "${LENS_PORT:-31144}" \
    > "$LENS_LOG" 2>&1 &
LENS_PID=$!
cd /workspace/ATLAS
echo "Lens PID: $LENS_PID (log: $LENS_LOG)"

# Wait for Lens health (fast).
for i in $(seq 1 24); do
    if curl -s --max-time 2 "http://localhost:${LENS_PORT:-31144}/health" 2>/dev/null | grep -qE "ok|healthy"; then
        echo "Lens healthy after ${i}x5s."
        break
    fi
    printf "."
    sleep 5
done
if ! curl -s "http://localhost:${LENS_PORT:-31144}/health" 2>/dev/null | grep -qE "ok|healthy"; then
    echo "WARNING: Lens did not come up — V3 will run with GEOMETRIC_LENS_ENABLED=false."
    tail -30 "$LENS_LOG" >&2 || true
    export GEOMETRIC_LENS_ENABLED=false
fi

# 3. vLLM EMBED instance. Smaller, starts faster than gen.
echo ""
echo "--- Starting vLLM EMBED on port $EMBED_PORT ---"
EMBED_LOG=/tmp/vllm-embed.log
# --runner pooling --convert embed is the current API for serving a generation
# model via /v1/embeddings. The older --task embed still works but is
# deprecated in vLLM 0.17+.
nohup vllm serve "$MODEL_PATH" \
    --served-model-name "${LLAMA_EMBED_MODEL:-qwen3.5-9b-embed}" \
    --runner pooling \
    --convert embed \
    --port "$EMBED_PORT" \
    --host 0.0.0.0 \
    --max-num-seqs "$EMBED_MAX_NUM_SEQS" \
    --max-model-len "$EMBED_MAX_MODEL_LEN" \
    --max-num-batched-tokens "${EMBED_MAX_NUM_BATCHED_TOKENS:-4096}" \
    --gpu-memory-utilization "$EMBED_GPU_MEM_UTIL" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    > "$EMBED_LOG" 2>&1 &
EMBED_PID=$!
echo "vLLM EMBED PID: $EMBED_PID (log: $EMBED_LOG)"

# Wait for embed health (model load takes 30-90s for AWQ).
echo "--- Waiting for vLLM EMBED health (timeout 600s) ---"
for i in $(seq 1 120); do
    if curl -s --max-time 2 "http://localhost:${EMBED_PORT}/health" 2>/dev/null | grep -q ok; then
        echo "vLLM EMBED healthy after ${i}x5s."
        break
    fi
    printf "."
    sleep 5
done
if ! curl -s "http://localhost:${EMBED_PORT}/health" 2>/dev/null | grep -q ok; then
    echo ""
    echo "ERROR: vLLM EMBED did not come up. Tail of log:" >&2
    tail -60 "$EMBED_LOG" >&2
    exit 1
fi

# 4. vLLM GEN instance — main inference engine.
echo ""
echo "--- Starting vLLM GEN on port $GEN_PORT ---"
GEN_LOG=/tmp/vllm-gen.log
nohup vllm serve "$MODEL_PATH" \
    --served-model-name "${LLAMA_GEN_MODEL:-qwen3.5-9b}" \
    --port "$GEN_PORT" \
    --host 0.0.0.0 \
    --max-num-seqs "$GEN_MAX_NUM_SEQS" \
    --max-model-len "$GEN_MAX_MODEL_LEN" \
    --max-num-batched-tokens "${GEN_MAX_NUM_BATCHED_TOKENS:-$GEN_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "$GEN_GPU_MEM_UTIL" \
    --swap-space "${GEN_SWAP_SPACE_GB:-4}" \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    > "$GEN_LOG" 2>&1 &
GEN_PID=$!
echo "vLLM GEN PID: $GEN_PID (log: $GEN_LOG)"

echo "--- Waiting for vLLM GEN health (timeout 600s) ---"
for i in $(seq 1 120); do
    if curl -s --max-time 2 "http://localhost:${GEN_PORT}/health" 2>/dev/null | grep -q ok; then
        echo "vLLM GEN healthy after ${i}x5s."
        break
    fi
    printf "."
    sleep 5
done
if ! curl -s "http://localhost:${GEN_PORT}/health" 2>/dev/null | grep -q ok; then
    echo ""
    echo "ERROR: vLLM GEN did not come up. Tail of log:" >&2
    tail -60 "$GEN_LOG" >&2
    exit 1
fi

# Export URLs for runners.
export LLAMA_GEN_URL="http://localhost:${GEN_PORT}"
export LLAMA_EMBED_URL="http://localhost:${EMBED_PORT}"
# Backwards-compat: many runners still read LLAMA_URL.
export LLAMA_URL="$LLAMA_GEN_URL"
export ATLAS_LLM_URL="$LLAMA_GEN_URL"
export RAG_API_URL="http://localhost:${LENS_PORT:-31144}"
export LENS_URL="$RAG_API_URL"
# Model names must reach preflight + runners; they were used as --served-model-name
# above, and the OpenAI-compatible API rejects unknown names with a 4xx.
export LLAMA_GEN_MODEL="${LLAMA_GEN_MODEL:-qwen3.5-9b}"
export LLAMA_EMBED_MODEL="${LLAMA_EMBED_MODEL:-qwen3.5-9b-embed}"
export BENCHMARK_PARALLEL ATLAS_LLM_PARALLEL ATLAS_PARALLEL_TASKS GEOMETRIC_LENS_ENABLED

# 5a. Pre-flight: hit each service end-to-end before committing GPU time
#     to a long benchmark sweep. Catches misconfiguration early.
echo ""
echo "--- Pre-flight: vLLM gen + embed + Lens end-to-end ---"
if ! ./benchmarks/h200/preflight.sh; then
    echo "Pre-flight failed. Refusing to start the benchmark sweep." >&2
    echo "Inspect the logs above and the service log files in /tmp/." >&2
    exit 1
fi

# 5b. Optional smoke test (set SKIP_SMOKE=1 to skip).
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

# 6. Run the benchmarks.
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

# 7. Archive results.
echo ""
echo "--- Archiving results ---"
mkdir -p "$(dirname "$RESULT_TAR")"
tar czf "$RESULT_TAR" \
    benchmarks/section_*/*/responses.jsonl \
    benchmarks/section_*/*/results.json \
    benchmarks/section_*/*/traces \
    benchmarks/logs \
    benchmark/results \
    2>/dev/null || true
ls -lh "$RESULT_TAR" || true

# 8. Stop vLLM cleanly.
kill "$GEN_PID" "$EMBED_PID" 2>/dev/null || true

echo ""
echo "============================================="
echo "Done. Results: $RESULT_TAR"
echo "============================================="

if [[ "$SHUTDOWN_ON_COMPLETE" == "1" ]]; then
    echo "SHUTDOWN_ON_COMPLETE=1 — exiting in 60s."
    sleep 60
    exit 0
else
    echo "Container will stay alive — rsync results, then stop the pod."
    exec tail -f "$GEN_LOG"
fi
