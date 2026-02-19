#!/bin/bash
#
# Nightly Training Pipeline
# Runs during low-usage hours to:
# 1. Export successful task completions
# 2. Fine-tune a LoRA adapter
# 3. Validate the new adapter
# 4. Hot-swap into production if validation passes
#
# Usage: ./nightly_train.sh
#

set -e

DATE=$(date +%Y%m%d)
TRAINING_DIR="/data/training"
MODELS_DIR="/models"
LORA_DIR="${MODELS_DIR}/lora"
MIN_EXAMPLES=10

echo "=========================================="
echo "ATLAS Nightly Training Pipeline"
echo "Date: ${DATE}"
echo "=========================================="

# Ensure directories exist
mkdir -p "${TRAINING_DIR}" "${LORA_DIR}"

# Step 1: Export training data
echo ""
echo "[Step 1/4] Exporting training data..."
EXPORT_FILE="${TRAINING_DIR}/${DATE}.jsonl"

python3 /app/export_training_data.py --output "${EXPORT_FILE}" --min-quality 0.6

if [ ! -f "${EXPORT_FILE}" ]; then
    echo "Error: Failed to export training data"
    exit 1
fi

EXAMPLE_COUNT=$(wc -l < "${EXPORT_FILE}")
echo "Exported ${EXAMPLE_COUNT} training examples"

if [ "${EXAMPLE_COUNT}" -lt "${MIN_EXAMPLES}" ]; then
    echo "Not enough training examples (${EXAMPLE_COUNT} < ${MIN_EXAMPLES}). Skipping training."
    exit 0
fi

# Step 2: Fine-tune LoRA adapter
echo ""
echo "[Step 2/4] Fine-tuning LoRA adapter..."
ADAPTER_OUTPUT="${LORA_DIR}/${DATE}"

# Note: This requires unsloth or similar library installed
# In production, this would run on a GPU node
if command -v python3 -c "import unsloth" &>/dev/null; then
    python3 -m unsloth.train \
        --model Qwen/Qwen3-14B \
        --data "${EXPORT_FILE}" \
        --output "${ADAPTER_OUTPUT}" \
        --epochs 1 \
        --batch_size 4 \
        --lora_r 16 \
        --lora_alpha 32
else
    echo "Warning: unsloth not installed. Skipping fine-tuning."
    echo "To enable training, install: pip install unsloth"
    # Create a placeholder to test the rest of the pipeline
    mkdir -p "${ADAPTER_OUTPUT}"
    echo '{"r": 16, "alpha": 32}' > "${ADAPTER_OUTPUT}/adapter_config.json"
    echo "Created placeholder adapter for testing"
fi

# Step 3: Validate new adapter
echo ""
echo "[Step 3/4] Validating adapter..."
if python3 /app/validate_adapter.py --adapter "${ADAPTER_OUTPUT}" --threshold 0.6; then
    VALIDATION_PASSED=true
else
    VALIDATION_PASSED=false
fi

# Step 4: Hot-swap if validation passed
echo ""
echo "[Step 4/4] Deployment..."
if [ "${VALIDATION_PASSED}" = true ]; then
    echo "Validation passed! Deploying new adapter..."

    # Update deployment with new adapter path
    if command -v kubectl &>/dev/null; then
        kubectl set env deployment/llama-server LORA_ADAPTER="${ADAPTER_OUTPUT}"
        echo "Adapter deployed: ${ADAPTER_OUTPUT}"
    else
        echo "kubectl not available. Manual deployment required."
        echo "Set LORA_ADAPTER=${ADAPTER_OUTPUT} in llama-server deployment"
    fi

    # Archive old training data
    gzip -f "${EXPORT_FILE}" || true

    echo ""
    echo "=========================================="
    echo "Training pipeline completed successfully!"
    echo "=========================================="
else
    echo "Validation failed. Keeping current adapter."
    echo ""
    echo "=========================================="
    echo "Training pipeline completed with warnings"
    echo "=========================================="
    exit 1
fi
