#!/bin/bash
# Build a comprehensive results tarball + optionally upload it to a durable
# location (HuggingFace dataset). Called from entrypoint.sh:
#   - At container start (just the manifest, no results yet)
#   - Every $SNAPSHOT_INTERVAL_SEC during the benchmark sweep
#   - On SIGTERM (spot reclaim)
#   - At end (final tarball, with --label=final)
#
# Usage: snapshot.sh [--label=LABEL]
#   --label  short tag included in the tarball filename (default: timestamp)
#
# Environment:
#   ATLAS_REPO              source dir (default /workspace/ATLAS)
#   RESULT_DIR              where tarballs land (default /workspace/results)
#   ATLAS_RUN_ID            identifies this run; default = container start UTC
#   HF_TOKEN                if set + ATLAS_HF_DATASET set, push to HF dataset
#   ATLAS_HF_DATASET        e.g. "itigges22/ATLAS" — receives runs/<RUN_ID>/...
#   SNAPSHOT_QUIET          1 = silence stdout (used by periodic background loop)
#
# This is the ONE place we tar results. Anything we want preserved on spot
# reclaim has to be globbed here.

set -uo pipefail

ATLAS_REPO="${ATLAS_REPO:-/workspace/ATLAS}"
RESULT_DIR="${RESULT_DIR:-/workspace/results}"
mkdir -p "$RESULT_DIR"

# Parse args.
LABEL=""
for arg in "$@"; do
    case "$arg" in
        --label=*) LABEL="${arg#--label=}" ;;
        *) echo "snapshot.sh: unknown arg: $arg" >&2; exit 2 ;;
    esac
done
[[ -z "$LABEL" ]] && LABEL=$(date -u +%Y%m%dT%H%M%SZ)

# Determine RUN_ID. Persisted in a file so the SIGTERM handler / periodic
# loop / final call all share the same id.
RUN_ID_FILE="$RESULT_DIR/.run_id"
if [[ -z "${ATLAS_RUN_ID:-}" ]]; then
    if [[ -s "$RUN_ID_FILE" ]]; then
        ATLAS_RUN_ID=$(cat "$RUN_ID_FILE")
    else
        ATLAS_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
        echo "$ATLAS_RUN_ID" > "$RUN_ID_FILE"
    fi
fi
export ATLAS_RUN_ID

TAR_NAME="atlas_results_${ATLAS_RUN_ID}_${LABEL}.tar.gz"
TAR_PATH="$RESULT_DIR/$TAR_NAME"

quiet() {
    if [[ "${SNAPSHOT_QUIET:-0}" != "1" ]]; then
        echo "$@"
    fi
}

quiet "[snapshot] label=$LABEL run_id=$ATLAS_RUN_ID -> $TAR_PATH"

# 1) Refresh the reproducibility manifest. Idempotent; cheap.
ATLAS_MANIFEST_PATH="$RESULT_DIR/manifest.json" \
ATLAS_REPO="$ATLAS_REPO" \
python3 "$ATLAS_REPO/benchmarks/h200/manifest.py" >/dev/null 2>&1 || \
    quiet "[snapshot] WARN: manifest.py failed; continuing"

# 2) Capture vLLM / Lens / smoke logs (live tail, may grow during run).
LOG_STAGE="$RESULT_DIR/logs"
mkdir -p "$LOG_STAGE"
for f in /tmp/vllm-gen.log /tmp/vllm-embed.log /tmp/lens-service.log /tmp/smoke.log; do
    if [[ -f "$f" ]]; then
        cp -f "$f" "$LOG_STAGE/" 2>/dev/null || true
    fi
done

# 3) Build the tarball. Using a file-list so missing paths don't tank the
# whole archive. tar's `-T -` reads paths from stdin; `--ignore-failed-read`
# tolerates files that disappear between glob expansion and archival.
cd "$ATLAS_REPO"

LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT

# Per-benchmark artifacts
{
    # responses + results — the actual per-task records
    find benchmarks/section_*/*/ -maxdepth 1 -name "responses.jsonl" 2>/dev/null
    find benchmarks/section_*/*/ -maxdepth 1 -name "results.json" 2>/dev/null
    find benchmarks/section_*/*/ -maxdepth 1 -name "REPORT.md" 2>/dev/null
    find benchmarks/section_*/*/ -maxdepth 1 -name "sample_questions.jsonl" 2>/dev/null
    find benchmarks/section_*/*/ -maxdepth 1 -name "*.bak.*" 2>/dev/null  # keep prior runs too

    # Per-task traces (V3 pipeline emits these per task)
    find benchmarks/section_*/*/traces -type f 2>/dev/null

    # Sweep logs
    find benchmarks/logs -type f 2>/dev/null

    # Aggregate report (if generated)
    [[ -f "benchmarks/AGGREGATE_REPORT.md" ]] && echo "benchmarks/AGGREGATE_REPORT.md"

    # V3 ablation results (legacy path, harmless if absent)
    find benchmark/results -type f 2>/dev/null

    # V3 telemetry (route_decisions, failure_embeddings, lens_feedback_events)
    find benchmark/v3 -type f -name "*.jsonl" 2>/dev/null
    find telemetry -type f 2>/dev/null  # if a runner writes here directly

    # Cluster-level snapshot files we generated above
    echo "$RESULT_DIR/manifest.json"
    [[ -f "$RESULT_DIR/pip_freeze.txt" ]] && echo "$RESULT_DIR/pip_freeze.txt"
    find "$RESULT_DIR/logs" -type f 2>/dev/null
} | sort -u > "$LIST"

LIST_COUNT=$(wc -l < "$LIST")
quiet "[snapshot] archiving $LIST_COUNT files"

if [[ "$LIST_COUNT" -eq 0 ]]; then
    quiet "[snapshot] WARN: no files matched; tarball will only contain the manifest"
fi

# Use --transform so paths from $RESULT_DIR end up under results/ inside
# the tarball (instead of the absolute path).
tar czf "$TAR_PATH" \
    --ignore-failed-read \
    --transform "s,^${RESULT_DIR#/}/,results/,;s,^/,," \
    -T "$LIST" 2>/dev/null || {
        quiet "[snapshot] tar exited non-zero (likely some files vanished mid-archive); continuing"
    }

# 4) Symlink "latest" for convenience.
ln -sf "$TAR_NAME" "$RESULT_DIR/atlas_results_latest.tar.gz"
SIZE=$(stat -c%s "$TAR_PATH" 2>/dev/null || echo 0)
quiet "[snapshot] wrote $TAR_PATH ($SIZE bytes)"

# 5) Optional upload to HuggingFace dataset.
if [[ -n "${HF_TOKEN:-}" && -n "${ATLAS_HF_DATASET:-}" ]]; then
    quiet "[snapshot] uploading to HF dataset $ATLAS_HF_DATASET"
    python3 - <<PYEOF || quiet "[snapshot] HF upload failed (continuing)"
import os, sys
try:
    from huggingface_hub import HfApi
except Exception as e:
    print(f"huggingface_hub not installed: {e}", file=sys.stderr); sys.exit(1)
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj="$TAR_PATH",
    path_in_repo=f"runs/${ATLAS_RUN_ID}/${TAR_NAME}",
    repo_id=os.environ["ATLAS_HF_DATASET"],
    repo_type="dataset",
    commit_message=f"snapshot ${LABEL} (${ATLAS_RUN_ID})",
)
print(f"uploaded runs/${ATLAS_RUN_ID}/${TAR_NAME}")
PYEOF
fi

# 6) Prune old tarballs: keep last 5 (intermediate snapshots) + any --label=final.
# Final tarballs are kept regardless. Periodic snapshots accumulate; keep the
# 5 most recent so a long run doesn't fill the disk.
find "$RESULT_DIR" -maxdepth 1 -name "atlas_results_*.tar.gz" \
    ! -name "*final*" -printf "%T@ %p\n" 2>/dev/null | \
    sort -n | head -n -5 | awk '{print $2}' | xargs -r rm -f

quiet "[snapshot] done"
exit 0
