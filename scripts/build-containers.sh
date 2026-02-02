#!/bin/bash
set -euo pipefail

# ATLAS Container Builder
# Builds all container images and imports to K3s
# Note: Importing to K3s requires sudo (will prompt if not root)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/config.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Detect container runtime
detect_runtime() {
    if command -v podman &> /dev/null; then
        echo "podman"
    elif command -v docker &> /dev/null; then
        echo "docker"
    else
        log_error "No container runtime found. Install podman or docker."
        exit 1
    fi
}

build_image() {
    local name="$1"
    local dir="$2"
    local runtime="$3"

    log_info "Building $name..."

    if [[ ! -d "$dir" ]]; then
        log_warn "Directory not found: $dir - skipping"
        return 0
    fi

    if [[ ! -f "$dir/Dockerfile" ]]; then
        log_warn "Dockerfile not found in $dir - skipping"
        return 0
    fi

    $runtime build -t "${ATLAS_REGISTRY}/$name:${ATLAS_IMAGE_TAG}" "$dir"

    log_info "$name built successfully"
}

import_to_k3s() {
    local name="$1"
    local runtime="$2"

    log_info "Importing $name to K3s..."

    # Check if image exists
    if ! $runtime image exists "${ATLAS_REGISTRY}/$name:${ATLAS_IMAGE_TAG}" 2>/dev/null; then
        log_warn "Image ${ATLAS_REGISTRY}/$name:${ATLAS_IMAGE_TAG} not found, skipping import"
        return 0
    fi

    # K3s containerd socket requires root access
    # Use full path since sudo doesn't inherit PATH
    if [[ $EUID -eq 0 ]]; then
        $runtime save "${ATLAS_REGISTRY}/$name:${ATLAS_IMAGE_TAG}" | /usr/local/bin/k3s ctr images import -
    else
        $runtime save "${ATLAS_REGISTRY}/$name:${ATLAS_IMAGE_TAG}" | sudo /usr/local/bin/k3s ctr images import -
    fi

    log_info "$name imported to K3s"
}

main() {
    echo "=========================================="
    echo "  ATLAS Container Builder"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  Registry:    $ATLAS_REGISTRY"
    echo "  Image tag:   $ATLAS_IMAGE_TAG"
    echo ""

    RUNTIME=$(detect_runtime)
    log_info "Using container runtime: $RUNTIME"

    # Core services (in k8s root)
    declare -a CORE_IMAGES=(
        "llama-server:$K8S_DIR/llama-server"
        "rag-api:$K8S_DIR/rag-api"
        "api-portal:$K8S_DIR/api-portal"
        "embedding-service:$K8S_DIR/embedding-service"
        "llm-proxy:$K8S_DIR/llm-proxy"
    )

    # Atlas services (in k8s/atlas)
    declare -a ATLAS_IMAGES=(
        "sandbox:$K8S_DIR/atlas/sandbox"
        "task-worker:$K8S_DIR/atlas/task-worker"
        "atlas-dashboard:$K8S_DIR/atlas/dashboard"
        "atlas-trainer:$K8S_DIR/atlas/trainer"
    )

    # Build all core images
    echo ""
    echo "Building core service images..."
    for entry in "${CORE_IMAGES[@]}"; do
        name="${entry%%:*}"
        dir="${entry#*:}"
        build_image "$name" "$dir" "$RUNTIME"
    done

    # Build all atlas images
    echo ""
    echo "Building Atlas service images..."
    for entry in "${ATLAS_IMAGES[@]}"; do
        name="${entry%%:*}"
        dir="${entry#*:}"
        build_image "$name" "$dir" "$RUNTIME"
    done

    # Import to K3s
    echo ""
    echo "Importing to K3s..."
    if [[ $EUID -ne 0 ]]; then
        log_warn "K3s import requires sudo - you may be prompted for your password"
    fi
    for entry in "${CORE_IMAGES[@]}"; do
        name="${entry%%:*}"
        import_to_k3s "$name" "$RUNTIME"
    done

    for entry in "${ATLAS_IMAGES[@]}"; do
        name="${entry%%:*}"
        import_to_k3s "$name" "$RUNTIME"
    done

    echo ""
    echo "=========================================="
    echo "  Build Complete!"
    echo "=========================================="
    echo ""
    echo "Images built and imported:"
    if [[ $EUID -eq 0 ]]; then
        /usr/local/bin/k3s ctr images list 2>/dev/null | grep "$ATLAS_REGISTRY" || echo "  (use 'sudo k3s ctr images list' to verify)"
    else
        sudo /usr/local/bin/k3s ctr images list 2>/dev/null | grep "$ATLAS_REGISTRY" || echo "  (use 'sudo k3s ctr images list' to verify)"
    fi
    echo ""
}

main "$@"
