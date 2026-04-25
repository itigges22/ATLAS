#!/bin/bash
# Periodically rsync benchmark snapshots from a RunPod / Lambda / cloud pod
# to this local machine. Designed to run alongside a long benchmark sweep
# so each `--label=periodic` tarball that snapshot.sh produces lands here
# within $POLL_INTERVAL_SEC of being written, surviving any pod reclaim
# from there on.
#
# Why pull (not push from pod):
#   - No SSH exposure on this machine — pod is the SSH server.
#   - Works through NAT / dynamic IP / coffee-shop wifi.
#   - Standard SSH key auth; nothing to set up beyond the pod side.
#   - rsync is incremental — re-runs are cheap, no duplicate downloads.
#
# Usage:
#   scripts/pull_pod_snapshots.sh <ssh-target> [options]
#
# Required:
#   <ssh-target>          ssh user@host (RunPod gives root@<host> on connect)
#
# Options:
#   -p, --port PORT        SSH port (RunPod assigns a high port, e.g. 12345)
#   -i, --inbox DIR        local destination (default: ./inbox/)
#   -k, --key PATH         SSH private key (default: ssh's default)
#   -n, --interval SEC     polling interval (default: 300 = 5 min)
#   --once                 pull once and exit (no polling loop)
#   --remote PATH          remote results dir (default: /workspace/results)
#   --rehydrate            on each pull, automatically rehydrate any new
#                          *_final.tar.gz into ./rehydrated/
#   -h, --help             show this help
#
# Examples:
#   # Continuous pull every 5 min (RunPod typical SSH on port 12345)
#   scripts/pull_pod_snapshots.sh root@149.36.0.42 -p 12345
#
#   # One-shot pull, custom key, custom inbox
#   scripts/pull_pod_snapshots.sh root@host -p 22 -k ~/.ssh/runpod_key \
#       -i ~/atlas-results --once
#
#   # Pull + auto-rehydrate finals into ./rehydrated/<run_id>/
#   scripts/pull_pod_snapshots.sh root@host -p 22 --rehydrate
#
# Environment overrides (alternative to flags):
#   ATLAS_PULL_TARGET, ATLAS_PULL_PORT, ATLAS_PULL_INBOX, ATLAS_PULL_KEY,
#   ATLAS_PULL_INTERVAL, ATLAS_PULL_REMOTE, ATLAS_PULL_REHYDRATE=1

set -uo pipefail

# Defaults
SSH_TARGET="${ATLAS_PULL_TARGET:-}"
SSH_PORT="${ATLAS_PULL_PORT:-22}"
INBOX="${ATLAS_PULL_INBOX:-./inbox}"
SSH_KEY="${ATLAS_PULL_KEY:-}"
INTERVAL="${ATLAS_PULL_INTERVAL:-300}"
REMOTE_PATH="${ATLAS_PULL_REMOTE:-/workspace/results}"
REHYDRATE="${ATLAS_PULL_REHYDRATE:-0}"
ONCE=0

usage() {
    awk '/^# ?/{sub(/^# ?/, ""); print; next} /^[^#]/{exit}' "$0"
    exit "${1:-0}"
}

# Parse args.
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)      SSH_PORT="$2"; shift 2 ;;
        -i|--inbox)     INBOX="$2"; shift 2 ;;
        -k|--key)       SSH_KEY="$2"; shift 2 ;;
        -n|--interval)  INTERVAL="$2"; shift 2 ;;
        --remote)       REMOTE_PATH="$2"; shift 2 ;;
        --rehydrate)    REHYDRATE=1; shift ;;
        --once)         ONCE=1; shift ;;
        -h|--help)      usage 0 ;;
        --)             shift; POSITIONAL+=("$@"); break ;;
        -*)             echo "unknown flag: $1" >&2; usage 2 ;;
        *)              POSITIONAL+=("$1"); shift ;;
    esac
done

if [[ -z "$SSH_TARGET" && ${#POSITIONAL[@]} -gt 0 ]]; then
    SSH_TARGET="${POSITIONAL[0]}"
fi
if [[ -z "$SSH_TARGET" ]]; then
    echo "ERROR: ssh target required (e.g. root@<pod-host>)" >&2
    usage 2
fi

# Sanity-check tools.
for tool in rsync ssh; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: $tool not found on PATH" >&2
        exit 2
    fi
done

mkdir -p "$INBOX"
INBOX_ABS=$(cd "$INBOX" && pwd)
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Build ssh / rsync flags.
SSH_FLAGS=(-p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
[[ -n "$SSH_KEY" ]] && SSH_FLAGS+=(-i "$SSH_KEY")
SSH_CMD="ssh ${SSH_FLAGS[*]}"

RSYNC_FLAGS=(
    -av                   # archive + verbose
    --partial             # resume interrupted transfers
    --append-verify       # if remote file grew, append + verify
    --inplace             # don't write to a temp file first
    --include='atlas_results_*.tar.gz'
    --include='manifest.json'
    --include='pip_freeze.txt'
    --include='logs/'
    --include='logs/**'
    --exclude='*'         # ignore everything else
)

echo "============================================="
echo "ATLAS spot-snapshot puller"
echo "  pod:       $SSH_TARGET (port $SSH_PORT)"
echo "  remote:    $REMOTE_PATH/"
echo "  local:     $INBOX_ABS/"
echo "  interval:  ${INTERVAL}s"
echo "  rehydrate: $([ $REHYDRATE -eq 1 ] && echo on || echo off)"
echo "  mode:      $([ $ONCE -eq 1 ] && echo one-shot || echo continuous)"
echo "============================================="

# Track which finals we've already rehydrated so we don't redo it every poll.
REHYDRATED_LOG="$INBOX_ABS/.rehydrated"
touch "$REHYDRATED_LOG"

pull_once() {
    local stamp
    stamp=$(date +%H:%M:%S)
    echo ""
    echo "[$stamp] pulling..."

    # rsync exits non-zero on transient SSH errors; catch + report but don't
    # die — the next poll will retry.
    if ! rsync "${RSYNC_FLAGS[@]}" \
        -e "$SSH_CMD" \
        "${SSH_TARGET}:${REMOTE_PATH}/" "$INBOX_ABS/"; then
        echo "[$stamp] rsync failed (will retry next poll)"
        return 1
    fi

    # Report what landed.
    local newest
    newest=$(ls -1t "$INBOX_ABS"/atlas_results_*.tar.gz 2>/dev/null | head -3)
    if [[ -n "$newest" ]]; then
        echo "[$stamp] latest 3 tarballs in $INBOX_ABS:"
        ls -lh $newest 2>/dev/null | awk '{printf "             %s  %s  %s\n", $5, $6" "$7" "$8, $NF}'
    fi

    # Optional auto-rehydrate of *final* tarballs.
    if [[ "$REHYDRATE" -eq 1 ]]; then
        local final
        for final in "$INBOX_ABS"/atlas_results_*_final.tar.gz; do
            [[ -f "$final" ]] || continue
            local fname
            fname=$(basename "$final")
            if grep -qxF "$fname" "$REHYDRATED_LOG"; then
                continue  # already rehydrated
            fi
            echo "[$stamp] rehydrating new final: $fname"
            "$REPO_ROOT/scripts/rehydrate_results.sh" "$final" 2>&1 | tail -5
            echo "$fname" >> "$REHYDRATED_LOG"
        done
    fi

    return 0
}

# Trap Ctrl-C so the loop exits cleanly.
trap 'echo ""; echo "[stop] interrupted"; exit 0' SIGINT SIGTERM

if [[ "$ONCE" -eq 1 ]]; then
    pull_once
    exit $?
fi

while true; do
    pull_once || true
    echo ""
    echo "[sleep] next poll in ${INTERVAL}s (Ctrl-C to stop)"
    sleep "$INTERVAL"
done
