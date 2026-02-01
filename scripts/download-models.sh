#!/bin/bash
set -euo pipefail

# ATLAS Model Downloader
# Downloads Qwen3-14B and draft model for speculative decoding

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/config.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Model URLs (Hugging Face)
# Note: Filenames are case-sensitive on Hugging Face
QWEN3_14B_Q4_URL="https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf"
QWEN3_14B_Q6_URL="https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q6_K.gguf"
QWEN3_0_6B_URL="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"

download_model() {
    local url="$1"
    local filename="$2"
    local filepath="$ATLAS_MODELS_DIR/$filename"

    if [[ -f "$filepath" ]]; then
        log_info "$filename already exists, skipping download"
        return
    fi

    log_info "Downloading $filename..."
    log_info "This may take a while (8-12GB file)"

    mkdir -p "$ATLAS_MODELS_DIR"

    # Use curl with resume support
    curl -L -C - -o "$filepath.tmp" "$url"
    mv "$filepath.tmp" "$filepath"

    log_info "$filename downloaded successfully"
}

verify_model() {
    local filepath="$1"
    local min_size="$2"

    if [[ ! -f "$filepath" ]]; then
        return 1
    fi

    local size=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null)
    if [[ $size -lt $min_size ]]; then
        log_error "File $filepath is too small (${size} bytes), may be corrupted"
        return 1
    fi

    return 0
}

main() {
    echo "=========================================="
    echo "  ATLAS Model Downloader"
    echo "=========================================="
    echo ""
    echo "Models directory: $ATLAS_MODELS_DIR"
    echo "Main model:       $ATLAS_MAIN_MODEL"
    echo "Draft model:      ${ATLAS_DRAFT_MODEL:-disabled}"
    echo ""

    # Check for huggingface-cli (optional, for faster downloads)
    if command -v huggingface-cli &> /dev/null; then
        log_info "HuggingFace CLI found, using for downloads"
        HF_CLI=true
    else
        log_info "Using curl for downloads (install huggingface-cli for faster downloads)"
        HF_CLI=false
    fi

    # Select model based on VRAM
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")

    if [[ $GPU_MEM -ge 20000 ]]; then
        log_info "20GB+ VRAM detected, downloading Q6_K quantization"
        MAIN_MODEL_URL="$QWEN3_14B_Q6_URL"
        MAIN_MODEL_FILE="Qwen3-14B-Q6_K.gguf"
    else
        log_info "Using Q4_K_M quantization for ${GPU_MEM}MB VRAM"
        MAIN_MODEL_URL="$QWEN3_14B_Q4_URL"
        MAIN_MODEL_FILE="Qwen3-14B-Q4_K_M.gguf"
    fi

    # Download main model
    download_model "$MAIN_MODEL_URL" "$MAIN_MODEL_FILE"

    # Download draft model for speculative decoding (if enabled)
    if [[ "$ATLAS_ENABLE_SPECULATIVE" == "true" ]] && [[ -n "$ATLAS_DRAFT_MODEL" ]]; then
        download_model "$QWEN3_0_6B_URL" "Qwen3-0.6B-Q8_0.gguf"
    else
        log_info "Speculative decoding disabled, skipping draft model"
    fi

    # Verify downloads
    echo ""
    log_info "Verifying downloads..."

    if verify_model "$ATLAS_MODELS_DIR/$MAIN_MODEL_FILE" 5000000000; then
        log_info "Main model verified: $MAIN_MODEL_FILE"
    else
        log_error "Main model verification failed"
        exit 1
    fi

    if [[ "$ATLAS_ENABLE_SPECULATIVE" == "true" ]] && [[ -n "$ATLAS_DRAFT_MODEL" ]]; then
        # Qwen3-0.6B-Q8_0 is ~639MB
        if verify_model "$ATLAS_MODELS_DIR/$ATLAS_DRAFT_MODEL" 500000000; then
            log_info "Draft model verified: $ATLAS_DRAFT_MODEL"
        else
            log_warn "Draft model verification failed (speculative decoding may not work)"
        fi
    fi

    # Create symlink for default model
    ln -sf "$ATLAS_MODELS_DIR/$MAIN_MODEL_FILE" "$ATLAS_MODELS_DIR/default.gguf"

    # Create LoRA adapter directory
    mkdir -p "$ATLAS_LORA_DIR"
    log_info "LoRA adapter directory created: $ATLAS_LORA_DIR"

    echo ""
    echo "=========================================="
    echo "  Model Download Complete!"
    echo "=========================================="
    echo ""
    echo "Models available:"
    ls -lh "$ATLAS_MODELS_DIR"/*.gguf 2>/dev/null || echo "  No .gguf files found"
    echo ""
}

main "$@"
