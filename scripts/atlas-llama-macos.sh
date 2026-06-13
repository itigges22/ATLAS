#!/usr/bin/env bash
# ATLAS macOS native llama-server launcher (#32 hybrid path).
#
# Starts the Metal-accelerated llama-server built by
# scripts/atlas-setup-macos.sh, using the same flags as the docker
# entrypoint (inference/entrypoint-v3.1-9b.sh) so behavior is
# identical to the linux + cuda/rocm path. Reads config from .env in
# the ATLAS root.
#
# Run this in its own terminal (it stays in the foreground). Stop with
# Ctrl-C; on stop, the docker stack's proxy will start serving 502s
# until you re-launch.
#
# Usage:
#   ./scripts/atlas-llama-macos.sh
#   ./scripts/atlas-llama-macos.sh --port 8081       # override port
#   ./scripts/atlas-llama-macos.sh --rebuild         # re-run setup first

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATLAS_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_PREFIX="$HOME/.atlas/macos"
LLAMA_SERVER="$DEFAULT_PREFIX/bin/llama-server-metal"

# Flag parsing — just the user-facing ones, everything else comes from .env
OVERRIDE_PORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) OVERRIDE_PORT="$2"; shift 2;;
    --rebuild)
      bash "$SCRIPT_DIR/atlas-setup-macos.sh" --rebuild
      shift;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done

# ---------------------------------------------------------------------------
# Sanity checks — fail fast with clear messages rather than letting
# llama-server crash with a confusing error.
# ---------------------------------------------------------------------------

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "native llama-server not found at $LLAMA_SERVER" >&2
  echo "  Run ./scripts/atlas-setup-macos.sh first." >&2
  exit 1
fi

if [[ ! -f "$ATLAS_ROOT/.env" ]]; then
  echo ".env not found at $ATLAS_ROOT/.env" >&2
  echo "  Run 'atlas init' first to generate it." >&2
  exit 1
fi

# Load .env (export every assignment for the subshell). Use `set -a` so
# vars get exported automatically.
set -a
# shellcheck disable=SC1091
source "$ATLAS_ROOT/.env"
set +a

# ---------------------------------------------------------------------------
# Resolve the runtime knobs. Mirrors inference/entrypoint-v3.1-9b.sh
# defaults so behavior matches the Docker path.
# ---------------------------------------------------------------------------

CTX_LENGTH="${ATLAS_CTX_SIZE:-${CONTEXT_LENGTH:-32768}}"
KV_CACHE_K="${KV_CACHE_TYPE_K:-q8_0}"
KV_CACHE_V="${KV_CACHE_TYPE_V:-q4_0}"
PARALLEL="${PARALLEL_SLOTS:-1}"
PORT="${OVERRIDE_PORT:-${ATLAS_LLAMA_PORT:-8080}}"

# Resolve model path. ATLAS_MODELS_DIR is "./models" (relative to atlas root)
# or an absolute path. ATLAS_MODEL_FILE is the .gguf filename.
MODELS_DIR="${ATLAS_MODELS_DIR:-./models}"
if [[ "$MODELS_DIR" != /* ]]; then
  MODELS_DIR="$ATLAS_ROOT/$MODELS_DIR"
fi
MODEL_FILE="$MODELS_DIR/${ATLAS_MODEL_FILE:?ATLAS_MODEL_FILE not set in .env}"

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "model file not found: $MODEL_FILE" >&2
  echo "  Run 'atlas model install ${ATLAS_MODEL_NAME:-<name>}' to download." >&2
  exit 1
fi

# ASA steering vector (#4 BiasBusters). Optional — only loaded if
# present at the conventional path the docker entrypoint uses.
CVECTOR_FLAGS=""
CVECTOR_PATH="$MODELS_DIR/${ATLAS_MODEL_NAME:-model}_ast_edit_steering.gguf"
if [[ -f "$CVECTOR_PATH" ]]; then
  CVECTOR_FLAGS="--control-vector-scaled $CVECTOR_PATH 1.0"
fi

# ---------------------------------------------------------------------------
# Banner — same shape as the docker entrypoint for diff-friendly logs
# ---------------------------------------------------------------------------

cat <<EOF
ATLAS llama-server (native macOS Metal) — #32 hybrid path
  Model:                $MODEL_FILE
  Context length:       $CTX_LENGTH
  Parallel slots:       $PARALLEL
  KV cache K / V:       $KV_CACHE_K / $KV_CACHE_V
  Port:                 $PORT
  ASA steering:         ${CVECTOR_FLAGS:-disabled}
  Binary:               $LLAMA_SERVER

EOF

# ---------------------------------------------------------------------------
# Launch. Same flags as the docker entrypoint with two differences:
#   --host 0.0.0.0   bind on all interfaces so Docker Desktop's
#                    host.docker.internal proxy can reach us
#   no --mlock       optional on Mac (unified memory makes it less
#                    impactful; can be added back if perf testing
#                    shows it helps)
# Slot save path: tmp dir so we don't pollute the repo.
# ---------------------------------------------------------------------------

SLOT_SAVE_PATH="${TMPDIR:-/tmp}/atlas-slots"
mkdir -p "$SLOT_SAVE_PATH"

# MTP (Multi-Token Prediction) flags — enabled via ATLAS_ENABLE_MTP=1 in .env
MTP_FLAGS=""
if [ "${ATLAS_ENABLE_MTP:-0}" = "1" ]; then
  MTP_DRAFT_MAX="${ATLAS_MTP_DRAFT_N_MAX:-2}"
  MTP_FLAGS="--spec-type draft-mtp --spec-draft-n-max $MTP_DRAFT_MAX"
  echo "  MTP: ENABLED (spec-type draft-mtp, draft-n-max=$MTP_DRAFT_MAX)"
fi

exec "$LLAMA_SERVER" \
  -m "$MODEL_FILE" \
  -c "$CTX_LENGTH" \
  -ctk "$KV_CACHE_K" -ctv "$KV_CACHE_V" \
  --parallel "$PARALLEL" \
  --cont-batching \
  -ngl 99 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --flash-attn on \
  -b 4096 \
  -ub 4096 \
  --slot-save-path "$SLOT_SAVE_PATH" \
  --ctx-checkpoints 0 \
  --no-cache-prompt \
  --embeddings \
  --jinja \
  $MTP_FLAGS \
  $CVECTOR_FLAGS
