#!/bin/bash
# ATLAS llama-server entrypoint — generation + self-embeddings.
#
# Model-agnostic: the model comes from MODEL_PATH and every memory-bound
# runtime knob (context, KV cache types, batch sizes, parallel slots) is
# env-driven, sized per model + GPU by `atlas tier fit --write` (PC-208).
# Self-embeddings are always on — the Geometric Lens C(x)/G(x) scores
# against the loaded model's own hidden-state dimension.
#
# Runs with --fit off: the model, KV cache, and compute buffers must fit
# entirely in VRAM, or llama-server refuses to start. This is deliberate —
# the silent alternative (llama.cpp demoting layers to CPU) generates at a
# fraction of GPU speed while burning host cores.

SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-/tmp/slots}"
mkdir -p "$SLOT_SAVE_PATH"

CTX_LENGTH="${CONTEXT_LENGTH:-163840}"
KV_CACHE_K="${KV_CACHE_TYPE_K:-f16}"
KV_CACHE_V="${KV_CACHE_TYPE_V:-f16}"
KV_FLAGS="-ctk $KV_CACHE_K -ctv $KV_CACHE_V"
PARALLEL="${PARALLEL_SLOTS:-4}"
# Batch sizes (PC-208): ubatch drives the compute-buffer size
# (~ubatch × n_embd × 280 bytes), which is what OOMs first on tight
# VRAM. `atlas tier fit --write` sizes these per model + GPU.
UBATCH_SIZE="${UBATCH_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
MODEL_FILE="${MODEL_PATH:-/models/Qwen3.5-9B-Q6_K.gguf}"
PORT="${PORT:-8080}"

# Backend-specific runtime tuning (V3.1.1 multi-backend support).
# ATLAS_BACKEND is written into .env by `atlas init`; unset defaults to
# cuda so existing deployments don't break.
ATLAS_BACKEND="${ATLAS_BACKEND:-cuda}"

case "$ATLAS_BACKEND" in
  cuda)
    # NVIDIA CUDA runtime knobs (unchanged from V3.1.0).
    #   GGML_CUDA_NO_PINNED=0     — keep pinned host memory for fast H2D
    #   CUDA_DEVICE_MAX_CONNECTIONS=1 — single-stream batching is fine,
    #                                   higher values seen no benefit
    #   CUDA_MODULE_LOADING=LAZY  — defer kernel loading until first use
    export GGML_CUDA_NO_PINNED="${GGML_CUDA_NO_PINNED:-0}"
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
    if [ -n "$ATLAS_GPU_INDEX" ] && [ -z "$CUDA_VISIBLE_DEVICES" ]; then
      export CUDA_VISIBLE_DEVICES="$ATLAS_GPU_INDEX"
    fi
    ;;
  rocm)
    # AMD ROCm/HIP runtime knobs. llama.cpp's HIP backend shares the
    # GGML_CUDA_* names internally (it mirrors the CUDA backend at the
    # GGML layer) so GGML_CUDA_NO_PINNED still applies. The vendor-side
    # CUDA_DEVICE_MAX_CONNECTIONS / CUDA_MODULE_LOADING vars are inert
    # under HIP and don't need to be set.
    export GGML_CUDA_NO_PINNED="${GGML_CUDA_NO_PINNED:-0}"
    if [ -n "$ATLAS_GPU_INDEX" ] && [ -z "$HIP_VISIBLE_DEVICES" ]; then
      export HIP_VISIBLE_DEVICES="$ATLAS_GPU_INDEX"
      # Newer ROCm (5.7+) prefers ROCR_VISIBLE_DEVICES; set both for
      # cross-version compatibility.
      export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-$ATLAS_GPU_INDEX}"
    fi
    # HSA_OVERRIDE_GFX_VERSION: force a specific gfx target. Useful when
    # rocm-smi reports an "unsupported" GPU (e.g., a consumer Vega/RDNA1
    # variant) that should still work with a near-compatible target.
    # Example: ATLAS_HSA_OVERRIDE_GFX_VERSION=10.3.0 makes RDNA1 cards
    # masquerade as RDNA2 for HIP kernel selection.
    if [ -n "$ATLAS_HSA_OVERRIDE_GFX_VERSION" ]; then
      export HSA_OVERRIDE_GFX_VERSION="$ATLAS_HSA_OVERRIDE_GFX_VERSION"
    fi
    ;;
  vulkan)
    # Vulkan universal backend (#114). The same llama-server binary runs
    # on any Vulkan-capable ICD: Mesa RADV (AMD), Mesa ANV (Intel),
    # nvidia-container-toolkit's libGLX_nvidia (NVIDIA), MoltenVK (Apple
    # via QEMU), Adreno (Snapdragon), or lavapipe (CPU software). The
    # compose overlay decides which ICD by setting device passthrough +
    # NVIDIA_DRIVER_CAPABILITIES; the entrypoint itself stays neutral.
    #
    # GGML_VK_VISIBLE_DEVICES: equivalent to CUDA_VISIBLE_DEVICES /
    # HIP_VISIBLE_DEVICES — pins to a specific Vulkan physical device
    # index when the host has multiple ICDs (e.g. iGPU + dGPU).
    if [ -n "$ATLAS_GPU_INDEX" ] && [ -z "$GGML_VK_VISIBLE_DEVICES" ]; then
      export GGML_VK_VISIBLE_DEVICES="$ATLAS_GPU_INDEX"
    fi
    # MESA_VK_DEVICE_SELECT: Mesa-specific selector ("vendorID:deviceID"
    # or "DeviceName"). Operator can set ATLAS_VK_DEVICE_SELECT to force
    # a specific physical device when GGML_VK_VISIBLE_DEVICES isn't
    # granular enough (e.g. two Intel Arc cards).
    if [ -n "$ATLAS_VK_DEVICE_SELECT" ]; then
      export MESA_VK_DEVICE_SELECT="$ATLAS_VK_DEVICE_SELECT"
    fi
    ;;
  metal|sycl)
    echo "Warning: ATLAS_BACKEND=$ATLAS_BACKEND but this entrypoint runs in Docker."
    echo "  Metal requires native install (V3.1.2 planned). SYCL is roadmap."
    echo "  Continuing with default CPU-only behavior; performance will be poor."
    ;;
  *)
    echo "Warning: ATLAS_BACKEND='$ATLAS_BACKEND' unrecognized; treating as cuda."
    export GGML_CUDA_NO_PINNED="${GGML_CUDA_NO_PINNED:-0}"
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
    ;;
esac

# BiasBusters #4 (ASA steering vectors) — always-on once the vector
# file exists at the standard path. The default path lives next to the
# model file on the persistent /models volume so it survives container
# rebuilds. Operator drops the vector once (workflow:
# ATLAS/geometric-lens/asa_calibration/README.md), and every llama-server
# start picks it up automatically. To override the path or scale, set
# ATLAS_CONTROL_VECTOR / ATLAS_CONTROL_VECTOR_SCALE / _LAYER_RANGE.
# Default scale is conservative (0.5) — bump if behavior change is too
# subtle, drop if non-tool tasks degrade.
ATLAS_CONTROL_VECTOR="${ATLAS_CONTROL_VECTOR:-/models/ast_edit_steering.gguf}"
CVECTOR_FLAGS=""
CVECTOR_STATUS="not present at $ATLAS_CONTROL_VECTOR — build it via geometric-lens/asa_calibration/README.md"
if [ -f "$ATLAS_CONTROL_VECTOR" ]; then
  CVECTOR_SCALE="${ATLAS_CONTROL_VECTOR_SCALE:-0.5}"
  CVECTOR_FLAGS="--control-vector-scaled $ATLAS_CONTROL_VECTOR:$CVECTOR_SCALE"
  if [ -n "$ATLAS_CONTROL_VECTOR_LAYER_RANGE" ]; then
    CVECTOR_FLAGS="$CVECTOR_FLAGS --control-vector-layer-range $ATLAS_CONTROL_VECTOR_LAYER_RANGE"
  fi
  CVECTOR_STATUS="$ATLAS_CONTROL_VECTOR (scale=$CVECTOR_SCALE${ATLAS_CONTROL_VECTOR_LAYER_RANGE:+, layers=$ATLAS_CONTROL_VECTOR_LAYER_RANGE})"
fi

MODEL_BASENAME="$(basename "$MODEL_FILE")"
echo "=== ATLAS llama-server — $MODEL_BASENAME ==="
echo "  Backend: $ATLAS_BACKEND${ATLAS_GPU_INDEX:+ (GPU index=$ATLAS_GPU_INDEX)}"
echo "  Model: $MODEL_FILE"
echo "  Context: $CTX_LENGTH | KV: K=$KV_CACHE_K V=$KV_CACHE_V | Parallel: $PARALLEL | Batch: $BATCH_SIZE/$UBATCH_SIZE"
echo "  Embeddings: ENABLED (model self-embeddings for the Geometric Lens)"
echo "  Slot save path: $SLOT_SAVE_PATH"
echo "  ASA steering: $CVECTOR_STATUS"
echo "  GPU fit: --fit off — the model, KV cache, and compute buffers must"
echo "  fit entirely in VRAM. If startup fails below with a CUDA out-of-memory"
echo "  allocation error, the budget above is too large for this GPU: run"
echo "  'atlas tier fit --write' on the host to size it, then recreate this"
echo "  container. See docs/TROUBLESHOOTING.md 'Model + KV cache don't fit'."

exec /usr/local/bin/llama-server \
  -m "$MODEL_FILE" \
  -c $CTX_LENGTH \
  $KV_FLAGS \
  --parallel $PARALLEL \
  --cont-batching \
  -ngl 99 \
  --fit off \
  --host 0.0.0.0 \
  --port $PORT \
  --flash-attn on \
  --mlock \
  -b $BATCH_SIZE \
  -ub $UBATCH_SIZE \
  --slot-save-path "$SLOT_SAVE_PATH" \
  --ctx-checkpoints 0 \
  `# Prompt caching is ON: each agent-loop turn reuses the prior turn's` \
  `# encoded prefix instead of re-encoding the whole conversation, which` \
  `# is what keeps multi-turn sessions fast. Cross-session isolation is` \
  `# handled by the proxy erasing the KV slot at session start (PC-045),` \
  `# not by disabling the cache.` \
  --embeddings \
  --jinja \
  $CVECTOR_FLAGS
