#!/bin/bash
set -euo pipefail

# ATLAS Model Downloader
# Downloads model files (Qwen3.5-9B for V3.1, or Qwen3-14B for legacy V3.0)

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

# Lens weights (cost_field, gx_xgboost, metric_tensor) — published in the
# user's HuggingFace dataset alongside the Qwen3.5-9B 4096-dim training run.
ATLAS_LENS_HF_BASE="https://huggingface.co/datasets/itigges22/ATLAS/resolve/main/models"
LENS_FILES=(
    "cost_field.pt"
    "cost_field.safetensors"
    "gx_weights.json"
    "gx_xgboost.json"
    "gx_xgboost.pkl"
    "metric_tensor.pt"
    "metric_tensor.safetensors"
)

# Manifest of model files we know how to fetch automatically.
# Add an entry here when a new model file becomes publicly available.
declare -A KNOWN_MODEL_URLS=(
    ["Qwen3-14B-Q4_K_M.gguf"]="$QWEN3_14B_Q4_URL"
    ["Qwen3-14B-Q6_K.gguf"]="$QWEN3_14B_Q6_URL"
    ["Qwen3-0.6B-Q8_0.gguf"]="$QWEN3_0_6B_URL"
    # Qwen3.5-9B-Q6_K.gguf — V3.1 default per inference/entrypoint-v3.1-9b.sh
    # and docker-compose.yml. NO PUBLIC URL KNOWN. Set ATLAS_MODEL_URL to
    # override or place the file at $ATLAS_MODELS_DIR/Qwen3.5-9B-Q6_K.gguf
    # by hand. See PC-018 in ISSUES.md.
)

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

download_lens_weights() {
    # Fetch the Geometric Lens weights from the ATLAS HF dataset.
    # Idempotent: skips files that are already present.
    local lens_dir
    lens_dir="$(cd "$SCRIPT_DIR/.." && pwd)/geometric-lens/geometric_lens/models"
    mkdir -p "$lens_dir"
    log_info "Lens weights directory: $lens_dir"
    for fname in "${LENS_FILES[@]}"; do
        local dest="$lens_dir/$fname"
        if [[ -f "$dest" ]]; then
            log_info "  $fname already present, skipping"
            continue
        fi
        log_info "  downloading $fname"
        curl -fL -C - -o "$dest.tmp" "$ATLAS_LENS_HF_BASE/$fname"
        mv "$dest.tmp" "$dest"
    done
    log_info "Lens weights ready"
}

resolve_model_url() {
    # Echo the URL for $1 (model filename), or return non-zero if unknown.
    # Resolution order:
    #   1. ATLAS_MODEL_URL env var (explicit override — wins)
    #   2. KNOWN_MODEL_URLS manifest above
    local fname="$1"
    if [[ -n "${ATLAS_MODEL_URL:-}" ]]; then
        echo "$ATLAS_MODEL_URL"
        return 0
    fi
    if [[ -n "${KNOWN_MODEL_URLS[$fname]:-}" ]]; then
        echo "${KNOWN_MODEL_URLS[$fname]}"
        return 0
    fi
    return 1
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

    # Subcommand: --lens fetches lens weights only and exits.
    if [[ "${1:-}" == "--lens" ]]; then
        download_lens_weights
        exit 0
    fi

    # Check for huggingface-cli (optional, for faster downloads)
    if command -v huggingface-cli &> /dev/null; then
        log_info "HuggingFace CLI found, using for downloads"
        HF_CLI=true
    else
        log_info "Using curl for downloads (install huggingface-cli for faster downloads)"
        HF_CLI=false
    fi

    # Pick which model file to fetch.
    # Priority: ATLAS_MODEL_FILE env > ATLAS_MAIN_MODEL config > VRAM autoselect.
    if [[ -n "${ATLAS_MODEL_FILE:-}" ]]; then
        MAIN_MODEL_FILE="$ATLAS_MODEL_FILE"
        log_info "Using ATLAS_MODEL_FILE=$MAIN_MODEL_FILE"
    elif [[ -n "${ATLAS_MAIN_MODEL:-}" && "$ATLAS_MAIN_MODEL" == *.gguf ]]; then
        MAIN_MODEL_FILE="$ATLAS_MAIN_MODEL"
        log_info "Using ATLAS_MAIN_MODEL=$MAIN_MODEL_FILE"
    else
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
        if [[ $GPU_MEM -ge 20000 ]]; then
            log_info "20GB+ VRAM detected, defaulting to Q6_K quantization"
            MAIN_MODEL_FILE="Qwen3-14B-Q6_K.gguf"
        else
            log_info "Defaulting to Q4_K_M quantization for ${GPU_MEM}MB VRAM"
            MAIN_MODEL_FILE="Qwen3-14B-Q4_K_M.gguf"
        fi
    fi

    # Resolve URL via manifest. Fail loudly if unknown rather than silently
    # downloading the wrong file (PC-018).
    if ! MAIN_MODEL_URL="$(resolve_model_url "$MAIN_MODEL_FILE")"; then
        log_error "No download URL known for $MAIN_MODEL_FILE."
        log_error ""
        log_error "Options:"
        log_error "  1. Place the file manually at:"
        log_error "       $ATLAS_MODELS_DIR/$MAIN_MODEL_FILE"
        log_error "  2. Set ATLAS_MODEL_URL=<url> and re-run this script."
        log_error "  3. Pick a model from the manifest in this script:"
        for known in "${!KNOWN_MODEL_URLS[@]}"; do
            log_error "       - $known"
        done
        log_error ""
        log_error "Note: Qwen3.5-9B-Q6_K.gguf is the V3.1 default but is not"
        log_error "publicly hosted at a URL we know about. The Geometric Lens"
        log_error "weights in the ATLAS HF dataset were trained on its 4096-dim"
        log_error "embeddings; using a different model family will silently"
        log_error "produce wrong scores (4096-dim weights vs 5120-dim embeddings)."
        exit 1
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
