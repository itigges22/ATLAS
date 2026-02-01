#!/bin/bash
set -euo pipefail

# ATLAS Installation Script
# Installs K3s, NVIDIA GPU Operator, and deploys all services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/config.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if running as root or with sudo
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root or with sudo"
        exit 1
    fi

    # Check for NVIDIA GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA driver not found. Install NVIDIA drivers first."
        exit 1
    fi

    # Check GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [[ $GPU_MEM -lt 15000 ]]; then
        log_warn "GPU has ${GPU_MEM}MB VRAM. 16GB+ recommended for $ATLAS_MAIN_MODEL."
    fi

    # Check system memory
    SYS_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $SYS_MEM -lt 24 ]]; then
        log_warn "System has ${SYS_MEM}GB RAM. 32GB+ recommended."
    fi

    # Validate config
    if ! validate_config; then
        log_error "Configuration validation failed"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Install K3s
install_k3s() {
    if command -v k3s &> /dev/null; then
        log_info "K3s already installed"
        return
    fi

    log_info "Installing K3s..."
    curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644

    # Wait for K3s to be ready
    log_info "Waiting for K3s to be ready..."
    sleep 10
    kubectl wait --for=condition=Ready nodes --all --timeout=120s

    log_info "K3s installed successfully"
}

# Install NVIDIA GPU Operator
install_gpu_operator() {
    if kubectl get namespace gpu-operator &> /dev/null; then
        log_info "GPU Operator namespace exists, checking status..."
        if kubectl get pods -n gpu-operator | grep -q "Running"; then
            log_info "GPU Operator already running"
            return
        fi
    fi

    log_info "Installing NVIDIA GPU Operator..."

    # Add Helm repo
    if ! command -v helm &> /dev/null; then
        log_info "Installing Helm..."
        curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    fi

    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
    helm repo update

    # Install GPU Operator
    kubectl create namespace gpu-operator || true
    helm install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator \
        --set driver.enabled=false \
        --wait --timeout 10m

    # Wait for GPU to be available
    log_info "Waiting for GPU to be available in cluster..."
    for i in {1..30}; do
        if kubectl get nodes -o json | grep -q "nvidia.com/gpu"; then
            log_info "GPU available in cluster"
            return
        fi
        sleep 10
    done

    log_error "GPU not detected in cluster after 5 minutes"
    exit 1
}

# Create namespaces and secrets
setup_namespace() {
    log_info "Setting up namespace and secrets..."

    # Create namespace if not default
    if [[ "$ATLAS_NAMESPACE" != "default" ]]; then
        kubectl create namespace "$ATLAS_NAMESPACE" || true
    fi

    # Create secrets if they don't exist
    if ! kubectl get secret atlas-secrets -n "$ATLAS_NAMESPACE" &> /dev/null; then
        # Use JWT secret from config
        API_SECRET=$(openssl rand -hex 32)

        kubectl create secret generic atlas-secrets -n "$ATLAS_NAMESPACE" \
            --from-literal=jwt-secret="$ATLAS_JWT_SECRET" \
            --from-literal=api-secret="$API_SECRET" || true
    fi

    log_info "Namespace and secrets ready"
}

# Build container images
build_images() {
    log_info "Building container images..."

    "$SCRIPT_DIR/build-containers.sh"

    log_info "Container images built"
}

# Deploy manifests
deploy_manifests() {
    log_info "Deploying ATLAS services..."

    # Deploy Atlas infrastructure first (Redis is dependency)
    log_info "Deploying Atlas infrastructure..."
    kubectl apply -f "$K8S_DIR/atlas/manifests/redis-deployment.yaml"

    # Deploy core infrastructure
    log_info "Deploying core infrastructure..."
    kubectl apply -f "$K8S_DIR/manifests/qdrant-deployment.yaml"
    kubectl apply -f "$K8S_DIR/manifests/embedding-deployment.yaml"

    # Wait for dependencies
    log_info "Waiting for infrastructure services..."
    kubectl wait --for=condition=Ready pod -l app=redis -n "$ATLAS_NAMESPACE" --timeout=120s || true
    kubectl wait --for=condition=Ready pod -l app=qdrant -n "$ATLAS_NAMESPACE" --timeout=120s || true
    kubectl wait --for=condition=Ready pod -l app=embedding-service -n "$ATLAS_NAMESPACE" --timeout=120s || true

    # Deploy main services
    log_info "Deploying main services..."
    kubectl apply -f "$K8S_DIR/manifests/llama-deployment.yaml"
    kubectl apply -f "$K8S_DIR/manifests/api-portal-deployment.yaml"
    kubectl apply -f "$K8S_DIR/manifests/rag-api-deployment.yaml"
    kubectl apply -f "$K8S_DIR/manifests/llm-proxy-deployment.yaml"

    # Deploy Atlas services
    log_info "Deploying Atlas services..."
    kubectl apply -f "$K8S_DIR/atlas/manifests/sandbox-deployment.yaml"
    kubectl apply -f "$K8S_DIR/atlas/manifests/task-worker-deployment.yaml"
    kubectl apply -f "$K8S_DIR/atlas/manifests/dashboard-deployment.yaml"

    # Apply training CronJob if enabled
    if [[ "$ATLAS_ENABLE_TRAINING" == "true" ]]; then
        kubectl apply -f "$K8S_DIR/atlas/manifests/training-cronjob.yaml" || true
    fi

    log_info "Manifests deployed"
}

# Wait for all services
wait_for_services() {
    log_info "Waiting for all services to be ready..."

    # Service names as defined in deployments
    SERVICES="redis qdrant embedding-service llama-server api-portal rag-api llm-proxy sandbox task-worker atlas-dashboard"

    for svc in $SERVICES; do
        log_info "Waiting for $svc..."
        kubectl wait --for=condition=Ready pod -l app=$svc -n "$ATLAS_NAMESPACE" --timeout=300s || {
            log_warn "$svc not ready within timeout, continuing..."
        }
    done

    log_info "All services deployed"
}

# Main
main() {
    echo "=========================================="
    echo "  ATLAS Installation Script"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  Models dir:  $ATLAS_MODELS_DIR"
    echo "  Data dir:    $ATLAS_DATA_DIR"
    echo "  Namespace:   $ATLAS_NAMESPACE"
    echo ""

    check_prerequisites
    install_k3s
    install_gpu_operator
    setup_namespace
    build_images
    deploy_manifests
    wait_for_services

    echo ""
    echo "=========================================="
    echo "  Installation Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Download models: ./scripts/download-models.sh"
    echo "  2. Verify installation: ./scripts/verify-install.sh"
    echo ""
    echo "Service endpoints:"
    echo "  API Portal:  http://${ATLAS_NODE_IP}:${ATLAS_API_PORTAL_NODEPORT}"
    echo "  LLM Proxy:   http://${ATLAS_NODE_IP}:${ATLAS_LLM_PROXY_NODEPORT}"
    echo "  RAG API:     http://${ATLAS_NODE_IP}:${ATLAS_RAG_API_NODEPORT}"
    echo "  Dashboard:   http://${ATLAS_NODE_IP}:${ATLAS_DASHBOARD_NODEPORT}"
    echo ""
}

main "$@"
