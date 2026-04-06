#!/bin/bash
# Deploy Qwen3.5-9B model to K3s cluster
# Replaces the current 14B+spec-decode setup with 9B (no spec decode)
#
# Prerequisites:
#   1. Model file: ${ATLAS_MODELS_DIR:-$HOME/models}/Qwen3.5-9B-Q6_K.gguf
#   2. Container:  localhost/llama-server:v3.1-9b
#
# Changes:
#   - Image: localhost/llama-server:v3.1-9b (latest llama.cpp with DeltaNet support)
#   - Model: Qwen3.5-9B-Q6_K.gguf (~7.5GB)
#   - No draft model (spec decode not supported for Qwen3.5)
#   - Parallel: 2 (more VRAM headroom without draft model)
#   - Context: 32768 (Qwen3.5 supports 128K, but 32K is practical)
#   - Embeddings: 4096-dim (vs 5120-dim for 14B)

set -e
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

echo "=== Deploying Qwen3.5-9B to K3s ==="

# Verify prerequisites
if [ ! -f ${ATLAS_MODELS_DIR:-$HOME/models}/Qwen3.5-9B-Q6_K.gguf ]; then
    echo "ERROR: Model file not found: ${ATLAS_MODELS_DIR:-$HOME/models}/Qwen3.5-9B-Q6_K.gguf"
    exit 1
fi

if ! podman images | grep -q "v3.1-9b"; then
    echo "ERROR: Container image not found: localhost/llama-server:v3.1-9b"
    exit 1
fi

echo "1. Importing container image to K3s..."
podman save localhost/llama-server:v3.1-9b | sudo k3s ctr images import -

echo "2. Updating ConfigMap with V3.1 entrypoint..."
kubectl delete configmap llama-entrypoint -n atlas 2>/dev/null || true
kubectl create configmap llama-entrypoint \
    --from-file=entrypoint.sh=${ATLAS_DIR:-$(pwd)}/llama-server/entrypoint-v3.1-9b.sh \
    -n atlas

echo "3. Patching deployment..."
kubectl patch deployment llama-server -n atlas --type='json' -p='[
  {"op": "replace", "path": "/spec/template/spec/containers/0/image", "value": "localhost/llama-server:v3.1-9b"},
  {"op": "replace", "path": "/spec/template/spec/containers/0/env", "value": [
    {"name": "MODEL_PATH", "value": "/models/Qwen3.5-9B-Q6_K.gguf"},
    {"name": "CONTEXT_LENGTH", "value": "32768"},
    {"name": "GPU_LAYERS", "value": "99"},
    {"name": "PARALLEL_SLOTS", "value": "2"},
    {"name": "GGML_CUDA_NO_PINNED", "value": "0"},
    {"name": "CUDA_DEVICE_MAX_CONNECTIONS", "value": "1"},
    {"name": "CUDA_MODULE_LOADING", "value": "LAZY"},
    {"name": "KV_CACHE_TYPE", "value": "q4_0"}
  ]}
]'

echo "4. Waiting for rollout..."
kubectl rollout status deployment/llama-server -n atlas --timeout=300s

echo "5. Verifying pod is ready..."
sleep 10
POD=$(kubectl get pods -n atlas -l app=llama-server -o jsonpath='{.items[0].metadata.name}')
echo "Pod: $POD"
kubectl logs "$POD" -n atlas --tail=20

echo ""
echo "=== Deployment complete ==="
echo "Pod IP: $(kubectl get pod $POD -n atlas -o jsonpath='{.status.podIP}')"
echo "NodePort: 32735"
