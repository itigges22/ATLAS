#!/bin/bash
set -euo pipefail

# ATLAS Model Downloader (vLLM / AWQ).
#
# Pulls the QuantTrio/Qwen3.5-9B-AWQ weights into ATLAS_MODELS_DIR. vLLM
# loads from this directory directly. Total ~12 GiB.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Optional config sourcing — gracefully no-op if config.sh is missing.
if [[ -f "$SCRIPT_DIR/lib/config.sh" ]]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/lib/config.sh" || true
fi

: "${ATLAS_MODELS_DIR:=$PWD/models}"
: "${ATLAS_MODEL_REPO:=QuantTrio/Qwen3.5-9B-AWQ}"
: "${ATLAS_MODEL_DIR_NAME:=Qwen3.5-9B-AWQ}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

main() {
    local target="$ATLAS_MODELS_DIR/$ATLAS_MODEL_DIR_NAME"

    echo "=========================================="
    echo "  ATLAS Model Downloader (vLLM / AWQ)"
    echo "=========================================="
    echo ""
    echo "Repo:    $ATLAS_MODEL_REPO"
    echo "Target:  $target"
    echo ""

    if [[ -d "$target" && -n "$(ls -A "$target" 2>/dev/null)" ]]; then
        log_info "Target already populated, skipping download"
        log_info "Force a re-download by removing $target first"
        return
    fi

    if ! command -v huggingface-cli >/dev/null 2>&1; then
        log_warn "huggingface-cli not found, installing..."
        pip install -q huggingface_hub
    fi

    mkdir -p "$target"
    log_info "Downloading $ATLAS_MODEL_REPO (~12 GiB) — sharded, parallel"

    # HF_TOKEN passed through automatically if exported.
    huggingface-cli download "$ATLAS_MODEL_REPO" \
        --local-dir "$target" \
        --local-dir-use-symlinks False

    echo ""
    log_info "Download complete:"
    du -sh "$target" || true
    ls "$target" | head -10
    echo ""
    echo "=========================================="
    echo "  Done. Set ATLAS_MODEL_PATH=$target"
    echo "  to point vLLM at it (already the default"
    echo "  in atlas.conf.example and .env.example)."
    echo "=========================================="
}

main "$@"
