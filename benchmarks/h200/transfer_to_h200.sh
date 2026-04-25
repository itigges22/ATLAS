#!/bin/bash
# Transfer the ATLAS benchmark code + AWQ model weights to a rented H200/H100 pod.
# Run locally from the ATLAS repo root: ./benchmarks/h200/transfer_to_h200.sh
#
# Edit H200_USER, H200_HOST, H200_SSH_KEY, REMOTE_DIR below (or set as env vars).

set -euo pipefail

# ============ EDIT THESE ============
H200_USER="${H200_USER:-ubuntu}"
H200_HOST="${H200_HOST:-}"                          # e.g. 203.0.113.5 or H200-vast.ai.example
H200_SSH_KEY="${H200_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-~/ATLAS}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-./models/Qwen3.5-9B-AWQ}"
# ====================================

if [[ -z "$H200_HOST" ]]; then
    echo "ERROR: Set H200_HOST (env var or edit this script)." >&2
    exit 1
fi

if [[ ! -d "$LOCAL_MODEL_DIR" ]]; then
    echo "ERROR: AWQ model not found at $LOCAL_MODEL_DIR" >&2
    echo "Pull it first:  make model" >&2
    exit 1
fi

SSH="ssh -i $H200_SSH_KEY -o StrictHostKeyChecking=accept-new $H200_USER@$H200_HOST"
RSYNC_SSH="ssh -i $H200_SSH_KEY -o StrictHostKeyChecking=accept-new"

echo "========================================"
echo "Transfer to $H200_USER@$H200_HOST"
echo "Remote dir:  $REMOTE_DIR"
echo "Local model: $LOCAL_MODEL_DIR"
echo "========================================"

# 1. Create remote tree
echo ""
echo "--- creating remote directories ---"
$SSH "mkdir -p $REMOTE_DIR/models $REMOTE_DIR/benchmarks $REMOTE_DIR/benchmark"

# 2. Send AWQ model directory (sharded safetensors, ~12 GiB)
echo ""
echo "--- sending AWQ model (~12 GiB, this may take 15-30 min) ---"
rsync -avz --progress -e "$RSYNC_SSH" \
    "$LOCAL_MODEL_DIR" \
    "$H200_USER@$H200_HOST:$REMOTE_DIR/models/"

# 3. Send code + current state + scripts
echo ""
echo "--- sending code and benchmark state ---"
rsync -avz --progress -e "$RSYNC_SSH" \
    --exclude '.cache' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'traces' \
    --include '*/' \
    --include 'responses.jsonl' \
    --include 'results.json' \
    --include 'config.yaml' \
    --include '*.py' \
    --include '*.sh' \
    --include '*.md' \
    --include 'eval_libs/***' \
    --include '.gitkeep' \
    --exclude '*' \
    ./benchmarks \
    "$H200_USER@$H200_HOST:$REMOTE_DIR/"

rsync -avz --progress -e "$RSYNC_SSH" \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --include '*.py' \
    --include '*/' \
    --exclude '*' \
    ./benchmark \
    "$H200_USER@$H200_HOST:$REMOTE_DIR/"

# Geometric Lens (needed for V3 pipeline scoring)
rsync -avz --progress -e "$RSYNC_SSH" \
    --exclude '__pycache__' --exclude '*.pyc' \
    ./geometric-lens \
    "$H200_USER@$H200_HOST:$REMOTE_DIR/"

# 4. Confirm
echo ""
echo "--- verifying remote ---"
$SSH "cd $REMOTE_DIR && du -sh models/Qwen3.5-9B-AWQ/ && echo '---' && ls benchmarks/section_*/*/responses.jsonl 2>/dev/null | head -5 && echo '---' && du -sh ."

echo ""
echo "========================================"
echo "Transfer complete. Next step: SSH to the pod and run"
echo "  cd $REMOTE_DIR && ./benchmarks/h200/launch_on_h200.sh"
echo "========================================"
