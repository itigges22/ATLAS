#!/bin/bash
# Runs on a rented H200 (or H100) SXM pod after the ATLAS image is on disk.
#
# Builds the vLLM-based ATLAS image from benchmarks/h200/Dockerfile and
# starts the container with the entrypoint, which brings up vllm-gen,
# vllm-embed, and the Geometric Lens. Pre-flight runs automatically.
#
# Usage (from ATLAS repo root on the H200):
#   ./benchmarks/h200/launch_on_h200.sh [baseline_only|atlas_only|all]
#
# Default: atlas_only. baseline numbers come from Qwen's published model card.

set -euo pipefail

MODE="${1:-atlas_only}"
IMAGE_TAG="${IMAGE_TAG:-atlas-bench:vllm}"
GEN_PORT="${GEN_PORT:-8000}"
EMBED_PORT="${EMBED_PORT:-8001}"
LENS_PORT="${LENS_PORT:-31144}"

echo "========================================"
echo "ATLAS V3.1 (vLLM) — H200 launch"
echo "Mode:        $MODE"
echo "Image:       $IMAGE_TAG"
echo "Gen/Embed:   $GEN_PORT / $EMBED_PORT"
echo "Lens:        $LENS_PORT"
echo "========================================"

# 1. Pick container runtime.
if command -v docker >/dev/null 2>&1; then
    CRT=docker
elif command -v podman >/dev/null 2>&1; then
    CRT=podman
else
    echo "ERROR: neither docker nor podman found." >&2
    exit 1
fi
echo "Container runtime: $CRT"

# 2. Build the ATLAS+vLLM image (idempotent — skips if tag already exists).
if ! $CRT image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    echo ""
    echo "--- Building $IMAGE_TAG (vLLM nightly + transformers patch + Lens deps) ---"
    $CRT build \
        -f benchmarks/h200/Dockerfile \
        -t "$IMAGE_TAG" \
        .
else
    echo "Image $IMAGE_TAG already built, skipping."
fi

# 3. Start the container. The entrypoint pulls the AWQ model on first run
#    (set DOWNLOAD_MODEL=0 + mount /workspace/models to skip), starts both
#    vLLM instances + Lens, runs preflight, then dispatches the benchmark
#    sweep based on $MODE.
echo ""
echo "--- starting container ---"
$CRT rm -f atlas-bench 2>/dev/null || true
$CRT run -d --name atlas-bench \
    --gpus all \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "${GEN_PORT}:8000" \
    -p "${EMBED_PORT}:8001" \
    -p "${LENS_PORT}:31144" \
    -v "${ATLAS_MODELS_DIR:-$PWD/models}:/workspace/models" \
    -v "${ATLAS_RESULTS_DIR:-$PWD/results}:/workspace/results" \
    -v "${ATLAS_HF_CACHE:-$PWD/.cache/huggingface}:/root/.cache/huggingface" \
    -e MODE="$MODE" \
    -e DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}" \
    -e SKIP_SMOKE="${SKIP_SMOKE:-0}" \
    -e SHUTDOWN_ON_COMPLETE="${SHUTDOWN_ON_COMPLETE:-0}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e GEN_MAX_NUM_SEQS="${GEN_MAX_NUM_SEQS:-32}" \
    -e GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-32768}" \
    -e GEN_GPU_MEM_UTIL="${GEN_GPU_MEM_UTIL:-0.55}" \
    -e GEN_MAX_NUM_BATCHED_TOKENS="${GEN_MAX_NUM_BATCHED_TOKENS:-}" \
    -e GEN_SWAP_SPACE_GB="${GEN_SWAP_SPACE_GB:-4}" \
    -e EMBED_MAX_NUM_SEQS="${EMBED_MAX_NUM_SEQS:-8}" \
    -e EMBED_MAX_MODEL_LEN="${EMBED_MAX_MODEL_LEN:-4096}" \
    -e EMBED_GPU_MEM_UTIL="${EMBED_GPU_MEM_UTIL:-0.20}" \
    "$IMAGE_TAG"

echo ""
echo "Container started. Tail with:"
echo "  $CRT logs -f atlas-bench"
echo ""
echo "Pre-flight + smoke + benchmark sweep are all driven by the entrypoint."
echo "When done, results land in /workspace/results inside the container,"
echo "and the tarball is copied to ${ATLAS_RESULTS_DIR:-$PWD/results}/atlas_results.tar.gz."
