#!/usr/bin/env python3
"""Build a V3.1 benchmark report from rehydrated baseline + ATLAS results.

Reads the per-benchmark `results.json` (and `responses.jsonl` for sanity)
that the v301_runner emits, plus the `manifest.json` from the snapshot
pipeline, and produces a single self-contained markdown report:

  AGGREGATE_REPORT.md
    - Methodology summary (model, sampling params, pipelines, hardware)
    - Per-benchmark table: baseline %, ATLAS %, delta, 95% CI overlap
    - Reproducibility manifest fingerprint (git SHA, vLLM ver, model SHA)
    - Footnotes per benchmark for known caveats (extraction failures, etc.)

Usage:
    python3 scripts/build_v31_report.py \
        --baseline rehydrated/<RUN_ID>/ \
        --atlas    rehydrated/<RUN_ID>/ \
        --out      reports/V3.1_BENCHMARKS.md

Or with separate baseline-only / atlas-only tarballs after rehydrate:
    --baseline rehydrated/<baseline_run_id>/
    --atlas    rehydrated/<atlas_run_id>/

Single source of truth for the published V3.1 numbers. Does not run any
inference — pure assembly from cached results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Stable display order; appended-only, not alphabetized.
BENCHMARK_DISPLAY_ORDER = [
    ("c_eval",          "section_a_knowledge_stem",  "C-Eval"),
    ("mmlu_pro",        "section_a_knowledge_stem",  "MMLU-Pro"),
    ("mmlu_redux",      "section_a_knowledge_stem",  "MMLU-Redux"),
    ("gpqa_diamond",    "section_a_knowledge_stem",  "GPQA-Diamond"),
    ("supergpqa",       "section_a_knowledge_stem",  "SuperGPQA"),
    ("ifeval",          "section_b_instruction_following", "IFEval"),
    ("ifbench",         "section_b_instruction_following", "IFBench"),
    ("livecodebench_v6","section_d_reasoning_coding", "LiveCodeBench v6"),
]


def load_results(run_dir: Path, bench: str, section: str) -> Optional[Dict[str, Any]]:
    """results.json under benchmarks/<section>/<bench>/. Returns None if missing."""
    p = run_dir / "benchmarks" / section / bench / "results.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"WARN: could not load {p}: {e}", file=sys.stderr)
        return None


def load_responses_count(run_dir: Path, bench: str, section: str) -> int:
    """Count of records in responses.jsonl (sanity check vs results)."""
    p = run_dir / "benchmarks" / section / bench / "responses.jsonl"
    if not p.exists():
        return 0
    try:
        return sum(1 for line in p.open() if line.strip())
    except Exception:
        return 0


def load_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    """results/manifest.json under the rehydrated dir."""
    p = run_dir / "results" / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"WARN: could not load manifest {p}: {e}", file=sys.stderr)
        return None


def fmt_pct(v: Any, digits: int = 1) -> str:
    """Format a number as a %, accept None / int / float / str gracefully."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}%"
    except Exception:
        return str(v)


def fmt_ci(lo: Any, hi: Any) -> str:
    if lo is None or hi is None:
        return "—"
    try:
        return f"[{float(lo):.1f}, {float(hi):.1f}]"
    except Exception:
        return f"[{lo}, {hi}]"


def fmt_delta(b: Optional[float], a: Optional[float]) -> str:
    if b is None or a is None:
        return "—"
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}pp"


def manifest_fingerprint(m: Optional[Dict[str, Any]]) -> str:
    if not m:
        return "<no manifest>"
    g = m.get("git", {})
    pkgs = m.get("python", {}).get("packages", {})
    model = m.get("model", {})
    hw = m.get("hardware", {})
    return "\n".join([
        f"- **run_id**: `{m.get('run_id')}`",
        f"- **snapshot_utc**: `{m.get('snapshot_utc')}`",
        f"- **git SHA**: `{(g.get('sha') or '')[:12]}` (dirty: `{g.get('dirty')}`)",
        f"- **vLLM**: `{pkgs.get('vllm', '?')}`",
        f"- **transformers**: `{pkgs.get('transformers', '?')}`",
        f"- **torch**: `{pkgs.get('torch', '?')}`",
        f"- **model**: `{model.get('path')}` ({(model.get('total_bytes') or 0) // (1024**3)} GiB, {sum(1 for k in (model.get('files') or {}) if k.endswith('.safetensors'))} shards)",
        f"- **hostname**: `{hw.get('hostname')}`",
    ])


def render_report(
    baseline_run: Path,
    atlas_run: Path,
    benchmark_filter: Optional[List[str]] = None,
) -> str:
    baseline_manifest = load_manifest(baseline_run)
    atlas_manifest = load_manifest(atlas_run)

    rows = []
    notes: List[str] = []
    benchmarks_to_render = (
        BENCHMARK_DISPLAY_ORDER
        if not benchmark_filter
        else [b for b in BENCHMARK_DISPLAY_ORDER if b[0] in benchmark_filter]
    )

    for bench_id, section, display in benchmarks_to_render:
        b_results = load_results(baseline_run, bench_id, section)
        a_results = load_results(atlas_run, bench_id, section)
        b_count = load_responses_count(baseline_run, bench_id, section)
        a_count = load_responses_count(atlas_run, bench_id, section)

        if b_results is None and a_results is None:
            # Skip benchmarks that aren't in either run.
            continue

        b_score = b_results.get("accuracy") if b_results else None
        a_score = a_results.get("accuracy") if a_results else None
        b_lo = b_results.get("ci_95_low") if b_results else None
        b_hi = b_results.get("ci_95_high") if b_results else None
        a_lo = a_results.get("ci_95_low") if a_results else None
        a_hi = a_results.get("ci_95_high") if a_results else None

        rows.append(
            "| {disp} | {bs} | {bci} | {as_} | {aci} | {delta} | {n_b}/{n_a} |".format(
                disp=display,
                bs=fmt_pct(b_score),
                bci=fmt_ci(b_lo, b_hi),
                as_=fmt_pct(a_score),
                aci=fmt_ci(a_lo, a_hi),
                delta=fmt_delta(b_score, a_score),
                n_b=b_count or "—",
                n_a=a_count or "—",
            )
        )

        # Per-benchmark notes — extraction failures, baseline source, etc.
        if a_results and (a_results.get("extraction_failures") or 0) > 0:
            notes.append(
                f"- **{display}**: ATLAS run had {a_results['extraction_failures']} "
                "extraction failures (model wandered without committing to a final answer letter)."
            )
        if b_results and (b_results.get("extraction_failures") or 0) > 0:
            notes.append(
                f"- **{display}**: baseline run had {b_results['extraction_failures']} extraction failures."
            )

    if not rows:
        return "# V3.1 Benchmark Report\n\nNo results found in the provided directories.\n"

    out = []
    out.append("# ATLAS V3.1 Benchmark Report")
    out.append("")
    out.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Benchmark | Baseline | Baseline 95% CI | ATLAS V3 | ATLAS 95% CI | Δ | n (B/A) |")
    out.append("|---|---|---|---|---|---|---|")
    out.extend(rows)
    out.append("")

    if notes:
        out.append("### Notes")
        out.append("")
        out.extend(notes)
        out.append("")

    out.append("## Reproducibility")
    out.append("")
    out.append("### Baseline run")
    out.append("")
    out.append(manifest_fingerprint(baseline_manifest))
    out.append("")
    out.append("### ATLAS run")
    out.append("")
    out.append(manifest_fingerprint(atlas_manifest))
    out.append("")

    # Method summary, deferred to the runner config that produced these.
    out.append("## Methodology")
    out.append("")
    if atlas_manifest:
        env = atlas_manifest.get("env") or {}
        out.append("ATLAS V3 sampling (from manifest env):")
        for k in ("BENCHMARK_PARALLEL", "ATLAS_LLM_PARALLEL", "ATLAS_PARALLEL_TASKS",
                  "GEN_MAX_NUM_SEQS", "GEN_MAX_MODEL_LEN", "GEN_GPU_MEM_UTIL",
                  "EMBED_MAX_NUM_SEQS", "EMBED_MAX_MODEL_LEN", "EMBED_GPU_MEM_UTIL",
                  "MODE", "GEOMETRIC_LENS_ENABLED"):
            if k in env:
                out.append(f"- `{k}` = `{env[k]}`")
        out.append("")
    out.append(
        "Per-benchmark `temperature`, `top_p`, `top_k`, `presence_penalty`, "
        "`max_tokens`, `seed` are recorded inside each `results.json` (`temperature`, "
        "`max_tokens`, `seed`, `bootstrap_n` fields). The full per-task records "
        "are in `responses.jsonl` for every benchmark, available alongside this "
        "report in the same rehydrated tree."
    )
    out.append("")

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", required=True,
        help="Rehydrated baseline run dir (e.g. rehydrated/<run_id>/)",
    )
    parser.add_argument(
        "--atlas", required=True,
        help="Rehydrated ATLAS run dir",
    )
    parser.add_argument(
        "--out",
        default="reports/V3.1_BENCHMARKS.md",
        help="Where to write the report (default: reports/V3.1_BENCHMARKS.md)",
    )
    parser.add_argument(
        "--benchmark", action="append", default=None,
        help="Restrict to specific benchmark id (e.g. --benchmark mmlu_pro). "
             "Can be passed multiple times.",
    )
    args = parser.parse_args()

    baseline_run = Path(args.baseline).resolve()
    atlas_run = Path(args.atlas).resolve()
    if not baseline_run.exists():
        print(f"ERROR: baseline dir not found: {baseline_run}", file=sys.stderr)
        return 2
    if not atlas_run.exists():
        print(f"ERROR: atlas dir not found: {atlas_run}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = render_report(baseline_run, atlas_run, args.benchmark)
    out_path.write_text(report)
    print(f"wrote {out_path} ({len(report)} bytes, {report.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
