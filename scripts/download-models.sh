#!/bin/bash
set -euo pipefail

# ATLAS Model Downloader
# Downloads the GGUF model + Geometric Lens weights for the Docker Compose
# deployment. K3s users should call this through `scripts/install.sh` which
# layers atlas.conf on top.
#
# Config resolution (first hit wins):
#   1. Existing env vars (set by caller — e.g. atlas-bootstrap.sh)
#   2. .env in repo root (Docker Compose convention)
#   3. .env.example (the v3.1 defaults)
#
# We deliberately do NOT source scripts/lib/config.sh here. That library is
# K3s-oriented (auto-writes .jwt_secret to the repo root, requires
# atlas.conf, validates NodePorts) and explodes on a Docker Compose install
# when the repo lives at /opt/atlas owned by root. PC-051.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Source .env (or .env.example) for ATLAS_* defaults. Only pulls in the
# keys this script reads — ATLAS_MODELS_DIR, ATLAS_MODEL_FILE,
# ATLAS_MAIN_MODEL, ATLAS_DRAFT_MODEL, ATLAS_ENABLE_SPECULATIVE,
# ATLAS_LORA_DIR. If a key is already set in the environment, the env
# value wins (so callers like atlas-bootstrap.sh can override).
load_env_defaults() {
    local env_file
    if [[ -f "$REPO_ROOT/.env" ]]; then
        env_file="$REPO_ROOT/.env"
    elif [[ -f "$REPO_ROOT/.env.example" ]]; then
        env_file="$REPO_ROOT/.env.example"
        log_warn ".env not found — using .env.example defaults"
    else
        log_error "Neither .env nor .env.example present at $REPO_ROOT"
        exit 1
    fi
    # Read line-by-line and only export ATLAS_* vars not already set.
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^ATLAS_[A-Z0-9_]+$ ]] || continue
        # Strip surrounding quotes and trailing whitespace from value
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        if [[ -z "${!key:-}" ]]; then
            export "$key=$value"
        fi
    done < <(grep -E '^[A-Z][A-Z0-9_]+=' "$env_file")
}

load_env_defaults

# Runtime paths. Model selection is required and comes from .env / atlas init.
: "${ATLAS_MODELS_DIR:=$REPO_ROOT/models}"
: "${ATLAS_MODEL_FILE:=}"
: "${ATLAS_MAIN_MODEL:=$ATLAS_MODEL_FILE}"
: "${ATLAS_DRAFT_MODEL:=}"
: "${ATLAS_ENABLE_SPECULATIVE:=false}"
: "${ATLAS_LORA_DIR:=$ATLAS_MODELS_DIR/lora}"

# Model URLs (Hugging Face)
# Note: Filenames are case-sensitive on Hugging Face
QWEN35_9B_Q6_URL="https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q6_K.gguf"
QWEN3_14B_Q4_URL="https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf"
QWEN3_14B_Q6_URL="https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q6_K.gguf"
QWEN3_0_6B_URL="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"

# Manifest of model files we know how to fetch automatically.
# Add an entry here when a new model file becomes publicly available.
declare -A KNOWN_MODEL_URLS=(
    # V3.1 default — published by unsloth.
    ["Qwen3.5-9B-Q6_K.gguf"]="$QWEN35_9B_Q6_URL"
    # Legacy V3.0 (Qwen3-14B) — kept so K3s benchmark runs still work.
    ["Qwen3-14B-Q4_K_M.gguf"]="$QWEN3_14B_Q4_URL"
    ["Qwen3-14B-Q6_K.gguf"]="$QWEN3_14B_Q6_URL"
    ["Qwen3-0.6B-Q8_0.gguf"]="$QWEN3_0_6B_URL"
)

# SHA-256 per model file, mirroring the `sha256` fields in
# atlas/cli/commands/model_registry.py (HF x-linked-etag values). The
# Python installer (`atlas model install`) already verifies these; this
# map closes the same gap on the shell path. Files without an entry get
# the size sanity check only — add the hash when a new model lands in
# the registry.
declare -A KNOWN_MODEL_SHA256=(
    ["Qwen3.5-9B-Q6_K.gguf"]="91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59"
)

sha256_of() {
    # sha256sum on Linux, shasum on macOS.
    if command -v sha256sum &>/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

download_model() {
    local url="$1"
    local filename="$2"
    local filepath="$ATLAS_MODELS_DIR/$filename"

    if [[ -f "$filepath" ]]; then
        log_info "$filename already exists, skipping download"
        return
    fi

    log_info "Downloading $filename from $url"
    log_info "This may take 5-15 min depending on network (file is ~7-12GB)"

    mkdir -p "$ATLAS_MODELS_DIR"

    # `-#` forces a progress bar even when stdout is piped (default curl
    # only shows progress on a tty). Without this the user stares at a
    # blank screen for 10 minutes wondering if the download is hung.
    # `-C -` resumes a partial download if .tmp already exists from a
    # previous interrupted run.
    if ! curl -L -# -C - --fail -o "$filepath.tmp" "$url"; then
        log_error "curl failed downloading $filename — see output above."
        log_error "Recovery: re-run this script (curl resumes from .tmp)."
        return 1
    fi

    local expected="${KNOWN_MODEL_SHA256[$filename]:-}"
    if [[ -n "$expected" ]]; then
        log_info "Verifying SHA-256 of $filename (multi-GB file — this takes a moment)"
        local actual
        actual=$(sha256_of "$filepath.tmp")
        if [[ "$actual" != "$expected" ]]; then
            log_error "SHA-256 mismatch for $filename:"
            log_error "  expected $expected"
            log_error "  got      $actual"
            log_error "The download is corrupt or the upstream file changed."
            log_error "Removing the partial file; re-run to download fresh."
            rm -f "$filepath.tmp"
            return 1
        fi
        log_info "SHA-256 verified"
    else
        log_warn "No pinned SHA-256 for $filename — size check only"
    fi
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
    # Artifacts are coupled to the selected model. Delegate to the registry
    # installer rather than downloading one architecture's bundle globally.
    local selected="${ATLAS_MODEL_NAME:-${ATLAS_MODEL_FILE%.gguf}}"
    if [[ -z "$selected" ]]; then
        log_error "No model selected. Set ATLAS_MODEL_NAME or run atlas init."
        return 1
    fi
    log_info "Installing compatible Lens/ASA artifacts for $selected"
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m atlas.cli model install-artifacts "$selected" \
        --models-dir "$ATLAS_MODELS_DIR" --no-color
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

    # Pick the explicitly selected model file. Registry/tier recommendation
    # belongs to `atlas init`; this low-level downloader never guesses.
    if [[ -n "${ATLAS_MODEL_FILE:-}" ]]; then
        MAIN_MODEL_FILE="$ATLAS_MODEL_FILE"
        log_info "Using ATLAS_MODEL_FILE=$MAIN_MODEL_FILE"
    elif [[ -n "${ATLAS_MAIN_MODEL:-}" && "$ATLAS_MAIN_MODEL" == *.gguf ]]; then
        MAIN_MODEL_FILE="$ATLAS_MAIN_MODEL"
        log_info "Using ATLAS_MAIN_MODEL=$MAIN_MODEL_FILE"
    else
        log_error "No model selected. Set ATLAS_MODEL_FILE in .env or run atlas init."
        exit 1
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
        log_error "Lens and ASA artifacts are model-specific. Use a registry entry"
        log_error "with compatible artifacts or build them for this model."
        exit 1
    fi

    # Download main model
    download_model "$MAIN_MODEL_URL" "$MAIN_MODEL_FILE"

    # Download draft model for speculative decoding (if enabled)
    if [[ "$ATLAS_ENABLE_SPECULATIVE" == "true" ]] && [[ -n "$ATLAS_DRAFT_MODEL" ]]; then
        if ! DRAFT_MODEL_URL="$(resolve_model_url "$ATLAS_DRAFT_MODEL")"; then
            log_error "No download URL known for selected draft model $ATLAS_DRAFT_MODEL"
            exit 1
        fi
        download_model "$DRAFT_MODEL_URL" "$ATLAS_DRAFT_MODEL"
    else
        log_info "Speculative decoding disabled, skipping draft model"
    fi

    # Verify downloads
    echo ""
    log_info "Verifying downloads..."

    if verify_model "$ATLAS_MODELS_DIR/$MAIN_MODEL_FILE" 100000000; then
        log_info "Main model verified: $MAIN_MODEL_FILE"
    else
        log_error "Main model verification failed"
        exit 1
    fi

    if [[ "$ATLAS_ENABLE_SPECULATIVE" == "true" ]] && [[ -n "$ATLAS_DRAFT_MODEL" ]]; then
        if verify_model "$ATLAS_MODELS_DIR/$ATLAS_DRAFT_MODEL" 100000000; then
            log_info "Draft model verified: $ATLAS_DRAFT_MODEL"
        else
            log_warn "Draft model verification failed (speculative decoding may not work)"
        fi
    fi

    # Create symlink for default model. Relative target (both live in
    # ATLAS_MODELS_DIR) so the link survives the directory being reached
    # via a different path — e.g. a container mount at /models, or a
    # relative ATLAS_MODELS_DIR resolved from another CWD.
    ln -sf "$MAIN_MODEL_FILE" "$ATLAS_MODELS_DIR/default.gguf"

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
