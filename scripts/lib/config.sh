#!/bin/bash
# ATLAS Config Loader
# Source this in scripts: source "$SCRIPT_DIR/lib/config.sh"

# Get paths relative to this library file
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SCRIPTS_DIR="$(dirname "$_LIB_DIR")"
K8S_DIR="$(dirname "$_SCRIPTS_DIR")"

# Colors for output (if not already defined)
RED="${RED:-\033[0;31m}"
GREEN="${GREEN:-\033[0;32m}"
YELLOW="${YELLOW:-\033[1;33m}"
NC="${NC:-\033[0m}"

# Load config file
load_config() {
    local config_file="${ATLAS_CONFIG_FILE:-$K8S_DIR/atlas.conf}"

    if [[ ! -f "$config_file" ]]; then
        # Try example file as fallback
        if [[ -f "$K8S_DIR/atlas.conf.example" ]]; then
            echo -e "${YELLOW}[WARN]${NC} atlas.conf not found, using atlas.conf.example defaults"
            config_file="$K8S_DIR/atlas.conf.example"
        else
            echo -e "${RED}[ERROR]${NC} No configuration file found"
            echo "Create one with: cp atlas.conf.example atlas.conf"
            exit 1
        fi
    fi

    # Source the config
    source "$config_file"

    # Handle auto-detection for node IP
    if [[ "${ATLAS_NODE_IP:-auto}" == "auto" ]]; then
        ATLAS_NODE_IP=$(hostname -I | awk '{print $1}')
    fi

    # Handle auto-generation for JWT secret
    if [[ "${ATLAS_JWT_SECRET:-auto}" == "auto" ]]; then
        # Generate deterministic secret based on hostname (or random for new installs)
        if [[ -f "$K8S_DIR/.jwt_secret" ]]; then
            ATLAS_JWT_SECRET=$(cat "$K8S_DIR/.jwt_secret")
        else
            ATLAS_JWT_SECRET=$(openssl rand -hex 32)
            echo "$ATLAS_JWT_SECRET" > "$K8S_DIR/.jwt_secret"
            chmod 600 "$K8S_DIR/.jwt_secret"
        fi
    fi

    # Export all ATLAS_ variables
    export "${!ATLAS_@}"

    # Also export K8S_DIR for scripts
    export K8S_DIR
}

# Validate required config
validate_config() {
    local errors=0

    # Check port conflicts (NodePorts must be unique)
    local ports=(
        "$ATLAS_API_PORTAL_NODEPORT"
        "$ATLAS_LLM_PROXY_NODEPORT"
        "$ATLAS_RAG_API_NODEPORT"
        "$ATLAS_DASHBOARD_NODEPORT"
        "$ATLAS_LLAMA_NODEPORT"
        "$ATLAS_QDRANT_NODEPORT"
        "$ATLAS_QDRANT_GRPC_NODEPORT"
        "$ATLAS_EMBEDDING_NODEPORT"
        "$ATLAS_SANDBOX_NODEPORT"
    )

    local seen=()
    for port in "${ports[@]}"; do
        if [[ " ${seen[*]} " =~ " ${port} " ]]; then
            echo -e "${RED}[ERROR]${NC} Duplicate NodePort: $port"
            errors=$((errors + 1))
        fi
        seen+=("$port")
    done

    # Validate NodePort range (30000-32767)
    for port in "${ports[@]}"; do
        if [[ $port -lt 30000 ]] || [[ $port -gt 32767 ]]; then
            echo -e "${RED}[ERROR]${NC} NodePort $port out of range (30000-32767)"
            errors=$((errors + 1))
        fi
    done

    return $errors
}

# Print current config (for debugging)
print_config() {
    echo "ATLAS Configuration:"
    echo "===================="
    env | grep "^ATLAS_" | sort
}

# Get config value with default
get_config() {
    local key="$1"
    local default="${2:-}"
    local var="ATLAS_$key"
    echo "${!var:-$default}"
}

# Auto-load config when sourced
load_config
