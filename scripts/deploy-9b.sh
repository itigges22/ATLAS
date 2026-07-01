#!/bin/bash
# Deploy Qwen3.5-9B model to K3s cluster
# Replaces the current 14B+spec-decode setup with 9B (no spec decode)
#
# V3.1.1: multi-backend. Pass ATLAS_BACKEND=rocm in the env (or via
# --backend rocm) to deploy the ROCm-built image with HIP env vars
# instead of the CUDA defaults.
#
# Prerequisites:
#   1. Model file: ${ATLAS_MODELS_DIR:-$HOME/models}/Qwen3.5-9B-Q6_K.gguf
#   2. Container:  localhost/llama-server:v3.1-9b           (CUDA build)
#               or localhost/llama-server-rocm:v3.1-9b      (ROCm build)
#
# Changes from 14B+spec-decode:
#   - Image: localhost/llama-server:v3.1-9b (latest llama.cpp with DeltaNet support)
#   - Model: Qwen3.5-9B-Q6_K.gguf (~7.5GB)
#   - No draft model (spec decode not supported for Qwen3.5)
#   - Parallel: 2 (more VRAM headroom without draft model)
#   - Context: 32768 (Qwen3.5 supports 128K, but 32K is practical)
#   - Embeddings: 4096-dim (vs 5120-dim for 14B)

set -e
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

# Parse --backend flag (alternative to ATLAS_BACKEND env var).
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) ATLAS_BACKEND="$2"; shift 2 ;;
        --backend=*) ATLAS_BACKEND="${1#--backend=}"; shift ;;
        -h|--help)
            grep '^#' "$0" | head -25; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
ATLAS_BACKEND="${ATLAS_BACKEND:-cuda}"

case "$ATLAS_BACKEND" in
    cuda) IMAGE_TAG="localhost/llama-server:v3.1-9b" ;;
    rocm) IMAGE_TAG="localhost/llama-server-rocm:v3.1-9b" ;;
    *)
        echo "ERROR: unsupported backend '$ATLAS_BACKEND' (expected: cuda, rocm)"
        exit 1
        ;;
esac

echo "=== Deploying Qwen3.5-9B to K3s (backend: $ATLAS_BACKEND) ==="

# Verify prerequisites
if [ ! -f ${ATLAS_MODELS_DIR:-$HOME/models}/Qwen3.5-9B-Q6_K.gguf ]; then
    echo "ERROR: Model file not found: ${ATLAS_MODELS_DIR:-$HOME/models}/Qwen3.5-9B-Q6_K.gguf"
    exit 1
fi

IMAGE_SHORT="${IMAGE_TAG#localhost/}"
if ! podman images | grep -q "${IMAGE_SHORT%:*}.*${IMAGE_SHORT#*:}"; then
    echo "ERROR: Container image not found: $IMAGE_TAG"
    echo "Build it with: podman build -f inference/Dockerfile.${ATLAS_BACKEND/cuda/v31} -t $IMAGE_TAG ./inference"
    exit 1
fi

echo "1. Importing container image to K3s..."
podman save "$IMAGE_TAG" | sudo k3s ctr images import -

echo "2. Updating ConfigMap with V3.1 entrypoint..."
kubectl delete configmap llama-entrypoint -n atlas 2>/dev/null || true
kubectl create configmap llama-entrypoint \
    --from-file=entrypoint.sh=${ATLAS_DIR:-$(pwd)}/inference/entrypoint-v3.1.sh \
    -n atlas

echo "3. Building env list for backend=$ATLAS_BACKEND..."
# Common env vars (same across backends)
ENV_JSON='[
    {"name": "MODEL_PATH", "value": "/models/Qwen3.5-9B-Q6_K.gguf"},
    {"name": "CONTEXT_LENGTH", "value": "32768"},
    {"name": "PARALLEL_SLOTS", "value": "2"},
    {"name": "KV_CACHE_TYPE_K", "value": "q4_0"},
    {"name": "KV_CACHE_TYPE_V", "value": "q4_0"},
    {"name": "ATLAS_BACKEND", "value": "'"$ATLAS_BACKEND"'"}'

# Backend-specific env vars. CUDA needs the 3 GGML/CUDA tuning knobs;
# ROCm only needs GGML_CUDA_NO_PINNED (the HIP backend honors it via
# the shared GGML compat layer; the CUDA_DEVICE_MAX_CONNECTIONS and
# CUDA_MODULE_LOADING vars are inert under HIP).
if [[ "$ATLAS_BACKEND" == "cuda" ]]; then
    ENV_JSON+=',
    {"name": "GGML_CUDA_NO_PINNED", "value": "0"},
    {"name": "CUDA_DEVICE_MAX_CONNECTIONS", "value": "1"},
    {"name": "CUDA_MODULE_LOADING", "value": "LAZY"}'
elif [[ "$ATLAS_BACKEND" == "rocm" ]]; then
    ENV_JSON+=',
    {"name": "GGML_CUDA_NO_PINNED", "value": "0"}'
    # Optional HSA gfx override — propagate from caller's env if set.
    if [[ -n "${ATLAS_HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
        ENV_JSON+=',
    {"name": "HSA_OVERRIDE_GFX_VERSION", "value": "'"$ATLAS_HSA_OVERRIDE_GFX_VERSION"'"}'
    fi
fi
ENV_JSON+=']'

echo "4. Patching deployment..."
kubectl patch deployment llama-server -n atlas --type='json' -p="$(cat <<EOF
[
  {"op": "replace", "path": "/spec/template/spec/containers/0/image", "value": "$IMAGE_TAG"},
  {"op": "replace", "path": "/spec/template/spec/containers/0/env", "value": $ENV_JSON}
]
EOF
)"

# NOTE: ROCm K8s deployments also need device passthrough + group_add
# for /dev/kfd + /dev/dri (the cluster-level equivalent of our
# docker-compose.rocm.yml). Those go in the Pod spec under
# securityContext + volumeMounts (hostPath) — not patched here yet.
# See ARCHITECTURE.md §8 for the K3s ROCm deployment recipe (TBD).
if [[ "$ATLAS_BACKEND" == "rocm" ]]; then
    echo "  WARNING: ROCm K8s pods need /dev/kfd + /dev/dri hostPath mounts +"
    echo "  groups 'video','render' in securityContext. This script patches"
    echo "  env vars only; ensure the deployment spec already includes them"
    echo "  (or wait for ARCHITECTURE.md K3s ROCm recipe — V3.1.2)."
fi

echo "5. Waiting for rollout..."
kubectl rollout status deployment/llama-server -n atlas --timeout=300s

echo "6. Verifying pod is ready..."
sleep 10
POD=$(kubectl get pods -n atlas -l app=llama-server -o jsonpath='{.items[0].metadata.name}')
echo "Pod: $POD"
kubectl logs "$POD" -n atlas --tail=20

echo ""
echo "=== Deployment complete (backend=$ATLAS_BACKEND) ==="
echo "Pod IP: $(kubectl get pod $POD -n atlas -o jsonpath='{.status.podIP}')"
echo "NodePort: 32735"
