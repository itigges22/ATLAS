#!/bin/bash
# Lay out a tarball produced by benchmarks/h200/snapshot.sh into the local
# repo's benchmarks/section_*/ tree, ready for build_v31_report.py.
#
# Usage:
#   ./scripts/rehydrate_results.sh <tarball.tar.gz> [-o <out_dir>] [--inplace]
#
#   -o <out_dir>   write under <out_dir>/ (default: ./rehydrated/<run_id>/)
#   --inplace      extract directly into the live repo (overwrites
#                  benchmarks/section_*/* — use with care)
#
# Tarballs created by snapshot.sh contain paths like:
#   benchmarks/section_a_knowledge_stem/mmlu_pro/responses.jsonl
#   benchmarks/section_a_knowledge_stem/mmlu_pro/results.json
#   benchmark/results/lcb_v6_*/...
#   results/manifest.json
#   results/pip_freeze.txt
#   results/logs/*.log
#
# Extract to a clean dir to compare with prior runs without trampling the
# active layout.

set -uo pipefail

TARBALL=""
OUT=""
INPLACE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--out) OUT="$2"; shift 2 ;;
        --inplace) INPLACE=1; shift ;;
        -h|--help)
            sed -n '1,/^$/p' "$0" | sed 's/^# //;s/^#//' >&2
            exit 0
            ;;
        *)
            if [[ -z "$TARBALL" ]]; then
                TARBALL="$1"; shift
            else
                echo "rehydrate_results.sh: unexpected arg: $1" >&2
                exit 2
            fi
            ;;
    esac
done

if [[ -z "$TARBALL" ]]; then
    echo "Usage: $0 <tarball.tar.gz> [-o <out_dir>] [--inplace]" >&2
    exit 2
fi
if [[ ! -f "$TARBALL" ]]; then
    echo "rehydrate_results.sh: tarball not found: $TARBALL" >&2
    exit 2
fi

# Sanity-check the tarball.
if ! tar tzf "$TARBALL" >/dev/null 2>&1; then
    echo "rehydrate_results.sh: not a valid gzipped tarball: $TARBALL" >&2
    exit 2
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)

# Pull run_id out of the tarball name if we can: atlas_results_<RUNID>_<LABEL>.tar.gz
RUN_ID=$(basename "$TARBALL" | sed -n 's/^atlas_results_\([^_]*\)_.*\.tar\.gz$/\1/p')
[[ -z "$RUN_ID" ]] && RUN_ID="unknown"

if [[ "$INPLACE" -eq 1 ]]; then
    DEST="$REPO"
    echo "WARNING: --inplace will overwrite live results in $REPO/benchmarks/section_*/"
    echo "Consider: git stash or move existing responses.jsonl aside first."
    read -r -p "Continue? [y/N] " ans
    if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
        echo "aborted"
        exit 1
    fi
else
    DEST="${OUT:-$REPO/rehydrated/$RUN_ID}"
    mkdir -p "$DEST"
fi

echo "Extracting $TARBALL"
echo "  run_id: $RUN_ID"
echo "  dest:   $DEST"

# tar -C extracts relative to that directory. The tarball uses paths
# rooted at the repo (benchmarks/...) and at /workspace/results -> results/.
tar xzf "$TARBALL" -C "$DEST"

echo ""
echo "=== Rehydrated tree ==="
{
    find "$DEST/benchmarks" -name "responses.jsonl" 2>/dev/null | head -10
    find "$DEST/benchmarks" -name "results.json" 2>/dev/null | head -10
    find "$DEST/results" -maxdepth 1 -type f 2>/dev/null | head -5
} | sed "s|^$DEST/|  |"

# Summarize what's actually in there.
RESPONSE_COUNT=$(find "$DEST/benchmarks" -name "responses.jsonl" 2>/dev/null | wc -l)
RESULTS_COUNT=$(find "$DEST/benchmarks" -name "results.json" 2>/dev/null | wc -l)
LCB_COUNT=$(find "$DEST/benchmark/results" -name "v3_events.jsonl" 2>/dev/null | wc -l)

echo ""
echo "=== Summary ==="
echo "  responses.jsonl files: $RESPONSE_COUNT"
echo "  results.json files:    $RESULTS_COUNT"
echo "  v3_events.jsonl runs:  $LCB_COUNT"

if [[ -f "$DEST/results/manifest.json" ]]; then
    echo ""
    echo "=== Manifest fingerprint ==="
    python3 - "$DEST/results/manifest.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"  run_id:        {m.get('run_id')}")
print(f"  snapshot_utc:  {m.get('snapshot_utc')}")
print(f"  git.sha:       {(m.get('git', {}).get('sha') or '')[:12]}")
print(f"  git.dirty:     {m.get('git', {}).get('dirty')}")
print(f"  vllm:          {m.get('python', {}).get('packages', {}).get('vllm')}")
print(f"  transformers:  {m.get('python', {}).get('packages', {}).get('transformers')}")
print(f"  model bytes:   {m.get('model', {}).get('total_bytes')}")
print(f"  hostname:      {m.get('hardware', {}).get('hostname')}")
PYEOF
else
    echo ""
    echo "WARN: no manifest.json — tarball may predate the snapshot pipeline"
fi

echo ""
echo "Next: scripts/build_v31_report.py --baseline <baseline_dir> --atlas <atlas_dir>"
