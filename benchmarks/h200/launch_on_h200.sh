#!/bin/bash
# Runs on the rented H200 SXM.
# Builds llama.cpp container, starts server with parallel 16, runs ATLAS V3.
# Usage (from ATLAS repo root on the H200):
#   ./benchmarks/h200/launch_on_h200.sh [baseline_only|atlas_only|all]
#
# Default: atlas_only.
#
# Why atlas_only: baseline numbers are cited from Qwen's published model card
# (bf16, their internal stack). ATLAS runs on Q6_K/llama.cpp — the same config
# shipped to users — on H200. The delta vs Qwen's bf16 baseline is a
# conservative lower bound on what the pipeline adds, since ATLAS is running
# quantized against a full-precision baseline. Keeps the Geometric Lens in
# distribution (Lens was trained on Q6_K embeddings) without any retraining.

set -euo pipefail

MODE="${1:-atlas_only}"
MODEL_PATH="${MODEL_PATH:-$PWD/models/Qwen3.5-9B-Q6_K.gguf}"
IMAGE_TAG="${IMAGE_TAG:-llama-server:h200}"
SERVER_PORT="${SERVER_PORT:-32735}"
# H100/H200 = Hopper = compute capability 90
CUDA_ARCH="${CUDA_ARCH:-90}"

echo "========================================"
echo "ATLAS V3.1 — H200 SXM Launch"
echo "Mode:        $MODE"
echo "Model:       $MODEL_PATH"
echo "Image:       $IMAGE_TAG"
echo "Port:        $SERVER_PORT"
echo "CUDA_ARCH:   $CUDA_ARCH"
echo "========================================"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: model not found at $MODEL_PATH. Run transfer_to_h200.sh first." >&2
    exit 1
fi

# --- 1. Pick container runtime ---
if command -v podman >/dev/null 2>&1; then
    CRT=podman
elif command -v docker >/dev/null 2>&1; then
    CRT=docker
else
    echo "ERROR: neither podman nor docker found." >&2
    exit 1
fi
echo "Container runtime: $CRT"

# --- 2. Build llama.cpp container for Hopper (H100/H200) ---
if ! $CRT image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    echo ""
    echo "--- Building llama.cpp container (CUDA_ARCH=$CUDA_ARCH) — this takes ~5 min ---"
    $CRT build \
        --build-arg CUDA_ARCH="$CUDA_ARCH" \
        -f inference/Dockerfile.v31 \
        -t "$IMAGE_TAG" \
        inference/
else
    echo "Image $IMAGE_TAG already built, skipping."
fi

# --- 3. Write H200-tuned entrypoint (parallel 16 + huge ctx pool) ---
ENTRYPOINT_SCRIPT="/tmp/atlas_llama_entrypoint.sh"
cat > "$ENTRYPOINT_SCRIPT" <<'EOF'
#!/bin/bash
export LLAMA_NO_MTP=1
export GGML_CUDA_NO_PINNED=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_MODULE_LOADING=LAZY

MODEL="${MODEL_PATH:-/models/Qwen3.5-9B-Q6_K.gguf}"
echo "=== V3.1 Bench on H200 SXM: parallel 16, 16K/slot, hopper arch ==="
exec /usr/local/bin/llama-server \
  -m "$MODEL" -c 262144 \
  -ctk q8_0 -ctv q4_0 \
  --parallel 16 --cont-batching -ngl 99 \
  --host 0.0.0.0 --port 8000 \
  --flash-attn on --mlock \
  -b 4096 -ub 4096 \
  --ctx-checkpoints 0 --no-cache-prompt \
  --embeddings --jinja --no-warmup 2>&1
EOF
chmod +x "$ENTRYPOINT_SCRIPT"

# --- 4. Start server ---
echo ""
echo "--- starting llama-server container on port $SERVER_PORT ---"
$CRT rm -f atlas-llama 2>/dev/null || true
$CRT run -d --name atlas-llama \
    --gpus all \
    -p "${SERVER_PORT}:8000" \
    -v "$MODEL_PATH:/models/$(basename "$MODEL_PATH"):ro" \
    -v "$ENTRYPOINT_SCRIPT:/entrypoint.sh:ro" \
    --entrypoint /entrypoint.sh \
    "$IMAGE_TAG"

# --- 5. Wait for healthy ---
echo ""
echo "--- waiting for server to come up (timeout 300s) ---"
for i in $(seq 1 60); do
    if curl -s --max-time 2 "http://localhost:${SERVER_PORT}/health" 2>/dev/null | grep -q ok; then
        echo "Server is healthy."
        break
    fi
    printf "."
    sleep 5
done

if ! curl -s "http://localhost:${SERVER_PORT}/health" 2>/dev/null | grep -q ok; then
    echo ""
    echo "ERROR: server did not come up." >&2
    $CRT logs --tail 50 atlas-llama
    exit 1
fi

# --- 6. Smoke test: 3 tasks, verify round-trip works ---
echo ""
echo "--- smoke test: 3 C-Eval tasks (should finish in < 2 min on H200) ---"
BENCHMARK_PARALLEL=4 timeout 300 python -m benchmarks.v301_runner \
    --benchmark c_eval --limit 3 --output-dir /tmp/h200_smoke > /tmp/h200_smoke.log 2>&1 \
    && echo "Smoke test OK" \
    || { echo "Smoke test FAILED. Log:"; tail -30 /tmp/h200_smoke.log; exit 1; }
rm -rf /tmp/h200_smoke

# --- 7. Run the benchmarks ---
LOG_DIR="benchmarks/logs/h200_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

export BENCHMARK_PARALLEL=16
export ATLAS_LLM_PARALLEL=1
export ATLAS_PARALLEL_TASKS=16

if [[ "$MODE" == "baseline_only" || "$MODE" == "all" ]]; then
    echo ""
    echo "========================================"
    echo "BASELINE — resuming from current state"
    echo "========================================"
    ./benchmarks/run_full_baseline.sh 2>&1 | tee "$LOG_DIR/baseline.log"
fi

if [[ "$MODE" == "atlas_only" || "$MODE" == "all" ]]; then
    echo ""
    echo "========================================"
    echo "ATLAS V3 PIPELINE"
    echo "========================================"
    # LCB v6 is the primary ATLAS pipeline benchmark.
    # For the other benchmarks we reuse the baseline runner — ATLAS V3 pipeline
    # integration for IFEval/IFBench/GPQA/MMLU-Pro would be a separate dispatch.
    ./benchmarks/run_lcb_v6.sh 2>&1 | tee "$LOG_DIR/atlas_lcb.log"
fi

echo ""
echo "========================================"
echo "Done. Results in benchmarks/section_*/"
echo "Run ./benchmarks/h200/watchdog.sh in another shell to auto-shutdown,"
echo "or tar results manually and rsync back to local."
echo "========================================"
