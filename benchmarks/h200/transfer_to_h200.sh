#!/bin/bash
# Transfer the ATLAS benchmark state and model file to a rented H200 instance.
# Run locally from the ATLAS repo root: ./benchmarks/h200/transfer_to_h200.sh
#
# Edit H200_USER, H200_HOST, H200_SSH_KEY, REMOTE_DIR below before running.

set -euo pipefail

# ============ EDIT THESE ============
H200_USER="${H200_USER:-ubuntu}"
H200_HOST="${H200_HOST:-}"                          # e.g. 203.0.113.5 or H200-vast.ai.example
H200_SSH_KEY="${H200_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-~/ATLAS}"
# ====================================

if [[ -z "$H200_HOST" ]]; then
    echo "ERROR: Set H200_HOST (env var or edit this script)." >&2
    exit 1
fi

SSH="ssh -i $H200_SSH_KEY -o StrictHostKeyChecking=accept-new $H200_USER@$H200_HOST"
RSYNC_SSH="ssh -i $H200_SSH_KEY -o StrictHostKeyChecking=accept-new"

echo "========================================"
echo "Transfer to H200: $H200_USER@$H200_HOST"
echo "Remote dir:       $REMOTE_DIR"
echo "========================================"

# 1. Create remote tree
echo ""
echo "--- creating remote directories ---"
$SSH "mkdir -p $REMOTE_DIR/models $REMOTE_DIR/benchmarks $REMOTE_DIR/benchmark $REMOTE_DIR/inference"

# 2. Send model (this is the big one, ~7GB)
echo ""
echo "--- sending model (~7GB, this may take 10-20 min) ---"
rsync -avz --progress -e "$RSYNC_SSH" \
    /home/isaac/models/Qwen3.5-9B-Q6_K.gguf \
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

rsync -avz --progress -e "$RSYNC_SSH" \
    ./inference/Dockerfile.v31 \
    "$H200_USER@$H200_HOST:$REMOTE_DIR/inference/"

# 4. Confirm
echo ""
echo "--- verifying remote ---"
$SSH "cd $REMOTE_DIR && ls -la models/ && echo '---' && ls benchmarks/section_*/*/responses.jsonl 2>/dev/null | head -5 && echo '---' && du -sh ."

echo ""
echo "========================================"
echo "Transfer complete. Next step: SSH to the H200 and run"
echo "  cd $REMOTE_DIR && ./benchmarks/h200/launch_on_h200.sh"
echo "========================================"
