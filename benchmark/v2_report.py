"""
V2 Benchmark Report Generator.

Reads phase results from a V2 benchmark run directory and generates
a comprehensive markdown report.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def _load_phase_results(run_dir: Path, phase_name: str) -> Optional[dict]:
    """Load results.json for a phase, or None if not found."""
    results_file = run_dir / phase_name / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def _count_telemetry(run_dir: Path) -> dict:
    """Count telemetry records and compute aggregates."""
    tel_dir = run_dir / "telemetry"
    stats = {
        "total_records": 0,
        "pass_count": 0,
        "fail_count": 0,
        "routes": {},
        "difficulties": {},
        "total_tokens": 0,
        "total_time_ms": 0,
    }

    tel_file = tel_dir / "route_decisions.jsonl"
    if not tel_file.exists():
        return stats

    with open(tel_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            stats["total_records"] += 1
            if rec.get("result") == "PASS":
                stats["pass_count"] += 1
            else:
                stats["fail_count"] += 1

            route = rec.get("route_selected", "UNKNOWN")
            stats["routes"][route] = stats["routes"].get(route, 0) + 1

            diff = rec.get("difficulty_bin", "UNKNOWN")
            stats["difficulties"][diff] = stats["difficulties"].get(diff, 0) + 1

            stats["total_tokens"] += rec.get("tokens_generated", 0)
            stats["total_time_ms"] += rec.get("generation_time_ms", 0)

    return stats


def _count_embeddings(run_dir: Path) -> dict:
    """Count failure embedding records."""
    embed_file = run_dir / "telemetry" / "failure_embeddings.jsonl"
    counts = {"PASS": 0, "FAIL": 0}
    if embed_file.exists():
        with open(embed_file) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                label = rec.get("label", "UNKNOWN")
                counts[label] = counts.get(label, 0) + 1
    return counts


def _load_learning_curve(run_dir: Path) -> Optional[dict]:
    """Load learning curve data from telemetry."""
    lc_file = run_dir / "telemetry" / "learning_curve.json"
    if lc_file.exists():
        with open(lc_file) as f:
            return json.load(f)
    return None


def _learning_curve_section(run_dir: Path) -> str:
    """Generate the Geometric Lens learning curve report section."""
    lc = _load_learning_curve(run_dir)
    if not lc or not lc.get("epochs"):
        return ""

    epochs = lc["epochs"]
    baseline_rate = epochs[0]["pass_rate"] if epochs else 0

    section = "\n## Geometric Lens Learning Curve\n\n"
    section += "| Epoch | Tasks | Training Data | AUC | pass@1 | vs Baseline |\n"
    section += "|-------|-------|---------------|-----|--------|-------------|\n"

    cumulative_data = 0
    for ep in epochs:
        idx = ep["epoch"]
        total = ep["total_tasks"]
        rate = ep["pass_rate"]
        delta = rate - baseline_rate

        metrics = ep.get("retrain_metrics") or {}
        auc = metrics.get("val_auc", "N/A")
        auc_str = f"{auc:.3f}" if isinstance(auc, float) else str(auc)

        if idx == 0:
            cumulative_data = total
            section += f"| {idx} (baseline) | {total} | 0 | N/A | {rate*100:.1f}% | -- |\n"
        else:
            section += f"| {idx} | {total} | {cumulative_data} | {auc_str} | {rate*100:.1f}% | {delta*100:+.1f}% |\n"
            cumulative_data += total

    # Retrain validation metrics table
    retrain_epochs = [ep for ep in epochs if ep.get("retrain_metrics") and not ep["retrain_metrics"].get("skipped")]
    if retrain_epochs:
        section += "\n### Retrain Validation Metrics\n\n"
        section += "| After Epoch | Train Size | Val AUC | Val Accuracy | Fail Ratio |\n"
        section += "|-------------|-----------|---------|-------------|------------|\n"
        for ep in retrain_epochs:
            m = ep["retrain_metrics"]
            section += (
                f"| {ep['epoch']} | {m.get('train_size', '?')} | "
                f"{m.get('val_auc', 0):.3f} | "
                f"{m.get('val_accuracy', 0)*100:.1f}% | "
                f"{m.get('fail_ratio', 0)*100:.1f}% |\n"
            )

    # Energy-outcome correlation improvement table
    rho_epochs = [ep for ep in epochs if ep.get("retrain_metrics") and "spearman_rho" in ep.get("retrain_metrics", {})]
    if rho_epochs:
        section += "\n### Energy-Outcome Correlation Improvement\n\n"
        section += "| Retrain | Spearman \u03c1 | Direction Correct? |\n"
        section += "|---------|-----------|-------------------|\n"
        section += "| Original (hand-labeled) | -0.05 | NO |\n"
        for ep in rho_epochs:
            m = ep["retrain_metrics"]
            rho = m["spearman_rho"]
            direction = "YES" if rho > 0 else "NO"
            section += f"| After Epoch {ep['epoch']} | {rho:.3f} | {direction} |\n"

    return section


def _load_best_of_k_summary(run_dir) -> Optional[dict]:
    """Load best-of-k selection summary from telemetry."""
    bok_file = Path(run_dir) / "telemetry" / "best_of_k_summary.json"
    if bok_file.exists():
        with open(bok_file) as f:
            return json.load(f)
    return None


def _best_of_k_section(run_dir) -> str:
    """Generate the Best-of-K Selection Analysis report section."""
    bok = _load_best_of_k_summary(run_dir)
    if not bok or not bok.get("total_tasks"):
        return ""

    section = "\n## Best-of-K Selection Analysis\n\n"

    # Table 1: Selection Accuracy
    section += "### Selection Accuracy\n\n"
    section += "| Metric | Value |\n"
    section += "|--------|-------|\n"
    section += f"| Tasks with >= 1 pass candidate | {bok.get('tasks_with_pass_candidate', 0)}/{bok['total_tasks']} ({bok.get('oracle_pass_rate', 0)*100:.1f}%) |\n"
    section += f"| Lens picked a passer | {bok.get('lens_picked_passer', 0)}/{bok.get('tasks_with_pass_candidate', 0)} |\n"
    section += f"| Selection accuracy | {bok.get('selection_accuracy', 0)*100:.1f}% |\n"

    # Table 2: Effective pass@1 vs Oracle pass@k
    section += "\n### Effective pass@1 vs Oracle pass@k\n\n"
    section += "| Metric | Score |\n"
    section += "|--------|-------|\n"
    section += f"| Oracle pass@k (upper bound) | {bok.get('oracle_pass_rate', 0)*100:.1f}% |\n"
    section += f"| Lens-selected effective pass@1 | {bok.get('effective_pass_rate', 0)*100:.1f}% |\n"
    gap = bok.get('oracle_pass_rate', 0) - bok.get('effective_pass_rate', 0)
    section += f"| Gap to oracle | {gap*100:.1f}% |\n"

    # Table 3: Sandbox Call Efficiency
    section += "\n### Sandbox Call Efficiency\n\n"
    section += "| Metric | Value |\n"
    section += "|--------|-------|\n"
    section += f"| Avg sandbox calls to find PASS | {bok.get('avg_sandbox_calls', 0):.1f} |\n"
    section += f"| Avg unique solutions per task | {bok.get('avg_unique_solutions', 0):.1f} |\n"
    section += f"| Avg energy spread (std dev) | {bok.get('avg_energy_std', 0):.2f} |\n"

    # Table 4: Energy Distribution
    ed = bok.get("energy_distribution", {})
    if ed:
        section += "\n### Energy Distribution\n\n"
        section += "| Category | Mean Energy | Std Dev |\n"
        section += "|----------|-------------|--------|\n"
        section += f"| Selected (returned to user) | {ed.get('selected_mean', 0):.2f} | {ed.get('selected_std', 0):.2f} |\n"
        section += f"| Rejected (discarded) | {ed.get('rejected_mean', 0):.2f} | {ed.get('rejected_std', 0):.2f} |\n"
        section += f"| PASS candidates (ground truth) | {ed.get('pass_mean', 0):.2f} | {ed.get('pass_std', 0):.2f} |\n"
        section += f"| FAIL candidates (ground truth) | {ed.get('fail_mean', 0):.2f} | {ed.get('fail_std', 0):.2f} |\n"

    return section


def generate_report(run_dir: Path) -> Path:
    """
    Generate the V2 benchmark markdown report.

    Args:
        run_dir: Path to the V2 benchmark run directory.

    Returns:
        Path to the generated report file.
    """
    run_dir = Path(run_dir)

    # Load run metadata
    meta_file = run_dir / "run_meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
    else:
        meta = {}

    run_id = meta.get("run_id", run_dir.name)
    start_time = meta.get("start_time", "unknown")
    end_time = meta.get("end_time", "unknown")

    # Calculate runtime
    try:
        dt_start = datetime.fromisoformat(start_time)
        dt_end = datetime.fromisoformat(end_time)
        duration = dt_end - dt_start
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        runtime_str = f"{hours}h {minutes}m"
    except (ValueError, TypeError):
        runtime_str = "unknown"

    # Load phase results
    p1 = _load_phase_results(run_dir, "phase1_lcb_mode_b")
    p2 = _load_phase_results(run_dir, "phase2_lcb_mode_a")
    p3 = _load_phase_results(run_dir, "phase3_gpqa_mode_b")
    p4 = _load_phase_results(run_dir, "phase4_ifbench_mode_b")
    p5b = _load_phase_results(run_dir, "phase5_custom/mode_b_pass1")
    p5a = _load_phase_results(run_dir, "phase5_custom/mode_a_pass1")
    p6 = _load_phase_results(run_dir, "phase6_scicode_mode_b")

    # Telemetry
    tel = _count_telemetry(run_dir)
    embeds = _count_embeddings(run_dir)

    def _fmt(phase_data):
        if phase_data is None:
            return "---", "---", "---"
        total = phase_data.get("total_tasks", 0)
        passed = phase_data.get("passed_tasks", 0)
        rate = phase_data.get("pass_rate", 0)
        return str(total), str(passed), f"{rate*100:.1f}%"

    p1_t, p1_p, p1_r = _fmt(p1)
    p2_t, p2_p, p2_r = _fmt(p2)
    p3_t, p3_p, p3_r = _fmt(p3)
    p4_t, p4_p, p4_r = _fmt(p4)
    p5b_t, p5b_p, p5b_r = _fmt(p5b)
    p5a_t, p5a_p, p5a_r = _fmt(p5a)
    p6_t, p6_p, p6_r = _fmt(p6)

    # Build report
    report = f"""# ATLAS V2 Benchmark Report

**Generated:** {datetime.utcnow().isoformat()}
**Run ID:** {run_id}
**Total Runtime:** {runtime_str}
**Benchmark Suite:** LiveCodeBench v5, GPQA Diamond, IFBench, Custom, SciCode
**Hardware:** RTX 5060 Ti 16GB, Qwen3-14B-Q4_K_M, 5 vCores, 14GB RAM

*All results from a single benchmark run. Not averaged across multiple runs; variance unknown.*

---

## Headline Results

| Benchmark | Tasks | V2 pass@1 (Mode B) | V2 pass@1 (Mode A) | Notes |
|-----------|-------|--------------------|--------------------|-------|
| LiveCodeBench v5 | {p1_t} | {p1_r} | {p2_r} | Primary code gen benchmark |
| GPQA Diamond | {p3_t} | {p3_r} | --- | Scientific reasoning |
| IFBench | {p4_t} | {p4_r} | --- | Instruction following * |
| Custom | {p5b_t} | {p5b_r} | {p5a_r} | V1 baseline: 66.0% |
| SciCode (sub-problems) | {p6_t} | {p6_r} | --- | Stretch goal |

> **Unvalidated:** `evaluate_ifbench_loose()` defaults to True for ~11/15
> instruction categories. This score reflects incomplete evaluation logic, not
> model capability. Not comparable to official IFBench scores.

## V2 Telemetry Summary

| Metric | Value |
|--------|-------|
| Total generations | {tel['total_records']} |
| Total tokens | {tel['total_tokens']:,} |
| PASS count | {tel['pass_count']} |
| FAIL count | {tel['fail_count']} |
| Avg tokens/task | {tel['total_tokens'] // max(tel['total_records'], 1):,} |
| Throughput (tasks/hr) | {tel['total_records'] / max(tel['total_time_ms'] / 3600000, 0.001):.0f} |

## Route Distribution

| Route | Count | Percentage |
|-------|-------|------------|
"""
    total_routes = max(sum(tel["routes"].values()), 1)
    for route, count in sorted(tel["routes"].items()):
        pct = 100 * count / total_routes
        report += f"| {route} | {count} | {pct:.1f}% |\n"

    report += f"""
## Difficulty Distribution

| Difficulty | Count | Percentage |
|-----------|-------|------------|
"""
    total_diff = max(sum(tel["difficulties"].values()), 1)
    for diff, count in sorted(tel["difficulties"].items()):
        pct = 100 * count / total_diff
        report += f"| {diff} | {count} | {pct:.1f}% |\n"

    report += f"""
## Failure Embeddings Collected

| Label | Count |
|-------|-------|
| PASS | {embeds.get('PASS', 0)} |
| FAIL | {embeds.get('FAIL', 0)} |
| Ready for Lens retraining | {'YES' if embeds.get('FAIL', 0) >= 200 and embeds.get('PASS', 0) >= 200 else 'NO'} |

## Context: Industry Comparison (Published Results)

For reference only -- not a direct comparison (different hardware, inference stack):

| Model | LiveCodeBench | GPQA Diamond | SciCode (sub-prob) |
|-------|-------------|--------------|-------------------|
| Claude Sonnet 4.5 | ~55% | ~60% | ~40% |
| GPT-4o | ~50% | ~53% | ~35% |
| **ATLAS V2 (Qwen3-14B)** | **{p1_r}** | **{p3_r}** | **{p6_r}** |
"""

    # Learning curve section
    lc_section = _learning_curve_section(run_dir)
    if lc_section:
        report += lc_section

    # Best-of-K selection section
    bok_section = _best_of_k_section(run_dir)
    if bok_section:
        report += bok_section

    report_path = run_dir / "v2_benchmark_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report generated: {report_path}")
    return report_path
