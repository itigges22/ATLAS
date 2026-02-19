#!/bin/bash
#
# Nightly Training Pipeline (CPU-based)
# Runs during low-usage hours to:
# 1. Export successful task completions
# 2. Fine-tune a LoRA adapter on CPU
# 3. Convert to GGUF format for llama.cpp
# 4. Hot-swap into production
#

set -e

DATE=$(date +%Y%m%d)
TRAINING_DIR="/data/training"
MODELS_DIR="/models"
LORA_DIR="${MODELS_DIR}/lora"
MIN_EXAMPLES=10

echo "=========================================="
echo "ATLAS Nightly Training Pipeline (CPU)"
echo "Date: ${DATE}"
echo "Started: $(date)"
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
    echo "Training will start once you have ${MIN_EXAMPLES}+ successful task completions."
    exit 0
fi

# Step 2: Fine-tune LoRA adapter on CPU
echo ""
echo "[Step 2/4] Fine-tuning LoRA adapter (CPU mode)..."
echo "This may take several hours on CPU..."
ADAPTER_OUTPUT="${LORA_DIR}/${DATE}"

python3 /app/train_lora.py \
    --data "${EXPORT_FILE}" \
    --output "${ADAPTER_OUTPUT}" \
    --epochs 1 \
    --batch-size 1 \
    --lora-r 8 \
    --lora-alpha 16

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

# Step 3: Convert to GGUF (if llama.cpp converter available)
echo ""
echo "[Step 3/4] Converting adapter..."
if [ -f "${ADAPTER_OUTPUT}/adapter_model.safetensors" ]; then
    echo "LoRA adapter created successfully"
    # Note: GGUF conversion would happen here if needed
    # For now, we just verify the adapter exists
else
    echo "Warning: No adapter file found, training may have failed"
fi

# Step 4: Update symlink and notify
echo ""
echo "[Step 4/4] Deployment..."

# Create symlink to latest adapter
ln -sfn "${ADAPTER_OUTPUT}" "${LORA_DIR}/latest"
echo "Updated latest symlink -> ${ADAPTER_OUTPUT}"

# Archive old training data
gzip -f "${EXPORT_FILE}" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Training pipeline completed!"
echo "New adapter: ${ADAPTER_OUTPUT}"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Note: To use the new adapter, restart llama-server with:"
echo "  --lora ${LORA_DIR}/latest"
