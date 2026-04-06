"""V3 Ablation Analysis — statistical analysis for ablation studies.

Reads unified V3 telemetry (v3_events.jsonl), computes per-condition
pass rates, bootstrap significance tests between adjacent conditions,
and generates publication-ready ablation tables.

Config: N/A (standalone analysis tool)
Usage: python -m benchmark.v3.ablation_analysis <telemetry_dir>
"""

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskOutcome:
    """Minimal per-task outcome for analysis."""
    task_id: str
    passed: bool
    latency_ms: float = 0.0
    tokens_generated: int = 0
    energy: float = 0.0


@dataclass
class ConditionResult:
    """Aggregated results for one (condition, seed) pair."""
    condition: str
    seed: int
    outcomes: List[TaskOutcome] = field(default_factory=list)

    @property
    def n_tasks(self) -> int:
        return len(self.outcomes)

    @property
    def n_pass(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def pass_rate(self) -> float:
        return self.n_pass / self.n_tasks if self.n_tasks > 0 else 0.0


@dataclass
class BootstrapResult:
    """Result of bootstrap significance test between two conditions."""
    condition_a: str
    condition_b: str
    mean_delta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_resamples: int
    significant: bool  # ci doesn't cross 0


# ---------------------------------------------------------------------------
# Telemetry loading
# ---------------------------------------------------------------------------

def load_telemetry(path: Path) -> List[dict]:
    """Load JSONL telemetry file into list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def extract_outcomes(records: List[dict]) -> List[TaskOutcome]:
    """Extract TaskOutcome list from telemetry records.

    Expects each record to have at minimum:
      - task_id: str
      - passed: bool (or final_passed, sandbox_passed)
    Optional fields: latency (dict or float), tokens_generated, energy.
    """
    outcomes = []
    for rec in records:
        task_id = rec.get("task_id", "")
        if not task_id:
            continue

        passed = rec.get("passed",
                         rec.get("final_passed",
                                 rec.get("sandbox_passed", False)))

        latency = 0.0
        lat_field = rec.get("latency", {})
        if isinstance(lat_field, dict):
            latency = lat_field.get("phase3_total",
                                    lat_field.get("total", 0.0))
        elif isinstance(lat_field, (int, float)):
            latency = float(lat_field)

        tokens = rec.get("tokens_generated",
                         rec.get("total_tokens", 0))

        energy = rec.get("selected_energy",
                         rec.get("energy", 0.0))

        outcomes.append(TaskOutcome(
            task_id=task_id,
            passed=bool(passed),
            latency_ms=latency,
            tokens_generated=tokens,
            energy=energy,
        ))
    return outcomes


def load_condition(telemetry_dir: Path,
                   condition: str,
                   seed: int) -> ConditionResult:
    """Load results for a specific (condition, seed) pair.

    Looks for files matching patterns:
      - {condition}_seed{seed}.jsonl
      - v3_events_{condition}_{seed}.jsonl
      - {condition}/{seed}/v3_events.jsonl
    """
    patterns = [
        f"{condition}_seed{seed}.jsonl",
        f"v3_events_{condition}_{seed}.jsonl",
        f"{condition}/seed{seed}/v3_events.jsonl",
        f"{condition}/{seed}/v3_events.jsonl",
    ]

    for pattern in patterns:
        path = telemetry_dir / pattern
        if path.exists():
            records = load_telemetry(path)
            outcomes = extract_outcomes(records)
            return ConditionResult(
                condition=condition, seed=seed, outcomes=outcomes)

    return ConditionResult(condition=condition, seed=seed)


# ---------------------------------------------------------------------------
# Bootstrap significance testing
# ---------------------------------------------------------------------------

def _pass_rate_from_outcomes(outcomes: List[TaskOutcome]) -> float:
    """Compute pass rate from a list of outcomes."""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.passed) / len(outcomes)


def bootstrap_delta(outcomes_a: List[TaskOutcome],
                    outcomes_b: List[TaskOutcome],
                    n_resamples: int = 10000,
                    seed: int = 42) -> BootstrapResult:
    """Bootstrap test for pass rate difference between two conditions.

    Computes: delta = pass_rate(B) - pass_rate(A)
    Positive delta means B is better.

    Uses paired bootstrap: resamples by task_id to preserve per-task pairing.
    Falls back to unpaired if task_ids don't match.

    Args:
        outcomes_a: Outcomes from condition A.
        outcomes_b: Outcomes from condition B.
        n_resamples: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with delta, CI, and p-value.
    """
    rng = random.Random(seed)

    # Try paired bootstrap
    map_a = {o.task_id: o.passed for o in outcomes_a}
    map_b = {o.task_id: o.passed for o in outcomes_b}
    shared_ids = sorted(set(map_a.keys()) & set(map_b.keys()))

    if len(shared_ids) >= min(len(outcomes_a), len(outcomes_b)) * 0.8:
        # Paired: resample task IDs
        pairs = [(map_a[tid], map_b[tid]) for tid in shared_ids]
        observed_delta = (sum(b for _, b in pairs) - sum(a for a, _ in pairs)) / len(pairs)

        deltas = []
        for _ in range(n_resamples):
            sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
            rate_a = sum(a for a, _ in sample) / len(sample)
            rate_b = sum(b for _, b in sample) / len(sample)
            deltas.append(rate_b - rate_a)
    else:
        # Unpaired: resample independently
        rate_a = _pass_rate_from_outcomes(outcomes_a)
        rate_b = _pass_rate_from_outcomes(outcomes_b)
        observed_delta = rate_b - rate_a

        deltas = []
        for _ in range(n_resamples):
            sample_a = [outcomes_a[rng.randrange(len(outcomes_a))]
                        for _ in range(len(outcomes_a))]
            sample_b = [outcomes_b[rng.randrange(len(outcomes_b))]
                        for _ in range(len(outcomes_b))]
            d = _pass_rate_from_outcomes(sample_b) - _pass_rate_from_outcomes(sample_a)
            deltas.append(d)

    deltas.sort()
    ci_lower = deltas[int(0.025 * n_resamples)]
    ci_upper = deltas[int(0.975 * n_resamples)]

    # Two-sided p-value: proportion of resamples on the other side of 0
    if observed_delta >= 0:
        p_value = sum(1 for d in deltas if d <= 0) / n_resamples
    else:
        p_value = sum(1 for d in deltas if d >= 0) / n_resamples
    p_value = min(p_value * 2, 1.0)  # two-sided

    significant = (ci_lower > 0) or (ci_upper < 0)

    return BootstrapResult(
        condition_a="A",
        condition_b="B",
        mean_delta=observed_delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_resamples=n_resamples,
        significant=significant,
    )


def multi_seed_pass_rate(results: List[ConditionResult]) -> Tuple[float, float]:
    """Compute mean and std of pass rates across seeds.

    Args:
        results: List of ConditionResult for different seeds.

    Returns:
        (mean_rate, std_rate)
    """
    if not results:
        return 0.0, 0.0
    rates = [r.pass_rate for r in results]
    mean = sum(rates) / len(rates)
    if len(rates) > 1:
        std = statistics.stdev(rates)
    else:
        std = 0.0
    return mean, std


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def ablation_table(conditions: Dict[str, List[ConditionResult]]) -> str:
    """Generate markdown ablation results table.

    Args:
        conditions: Dict mapping condition name to list of per-seed results.

    Returns:
        Markdown table string.
    """
    lines = [
        "## Ablation Results",
        "",
        "| Condition | Seeds | Tasks | Pass Rate (mean) | Std | 95% CI |",
        "|-----------|-------|-------|-------------------|-----|--------|",
    ]

    for cond_name in sorted(conditions.keys()):
        results = conditions[cond_name]
        n_seeds = len(results)
        n_tasks = results[0].n_tasks if results else 0
        mean_rate, std_rate = multi_seed_pass_rate(results)

        # Bootstrap CI across seeds on the mean
        if n_seeds >= 2:
            rates = [r.pass_rate for r in results]
            ci_lo = mean_rate - 1.96 * std_rate / math.sqrt(n_seeds)
            ci_hi = mean_rate + 1.96 * std_rate / math.sqrt(n_seeds)
            ci_str = f"[{ci_lo:.1%}, {ci_hi:.1%}]"
        else:
            ci_str = "—"

        lines.append(
            f"| {cond_name} | {n_seeds} | {n_tasks} | "
            f"{mean_rate:.1%} | {std_rate:.1%} | {ci_str} |"
        )

    return "\n".join(lines)


def pairwise_significance(conditions: Dict[str, List[ConditionResult]],
                          pairs: Optional[List[Tuple[str, str]]] = None,
                          n_resamples: int = 10000) -> str:
    """Generate pairwise bootstrap significance table.

    Args:
        conditions: Dict mapping condition name to list of per-seed results.
        pairs: List of (cond_a, cond_b) pairs to compare.
               Defaults to adjacent conditions (A→B, B→C, ...).
        n_resamples: Bootstrap resamples per comparison.

    Returns:
        Markdown table string.
    """
    if pairs is None:
        sorted_conds = sorted(conditions.keys())
        pairs = [(sorted_conds[i], sorted_conds[i+1])
                 for i in range(len(sorted_conds) - 1)]

    lines = [
        "## Pairwise Significance Tests",
        "",
        f"Bootstrap resamples: {n_resamples:,}",
        "",
        "| Comparison | Delta (pp) | 95% CI | p-value | Significant |",
        "|------------|-----------|--------|---------|-------------|",
    ]

    for cond_a, cond_b in pairs:
        results_a = conditions.get(cond_a, [])
        results_b = conditions.get(cond_b, [])

        if not results_a or not results_b:
            lines.append(f"| {cond_a} → {cond_b} | — | — | — | — |")
            continue

        # Pool outcomes across seeds for each condition
        all_a = [o for r in results_a for o in r.outcomes]
        all_b = [o for r in results_b for o in r.outcomes]

        result = bootstrap_delta(all_a, all_b, n_resamples=n_resamples)
        result.condition_a = cond_a
        result.condition_b = cond_b

        delta_pp = result.mean_delta * 100
        ci_lo_pp = result.ci_lower * 100
        ci_hi_pp = result.ci_upper * 100
        sig = "Yes" if result.significant else "No"

        lines.append(
            f"| {cond_a} → {cond_b} | "
            f"{delta_pp:+.1f} | [{ci_lo_pp:+.1f}, {ci_hi_pp:+.1f}] | "
            f"{result.p_value:.4f} | {sig} |"
        )

    return "\n".join(lines)


def phase_waterfall(conditions: Dict[str, List[ConditionResult]]) -> str:
    """Generate phase contribution waterfall (text-based).

    Shows incremental contribution of each condition over the previous one.

    Args:
        conditions: Dict mapping condition name to list of per-seed results.

    Returns:
        Markdown waterfall string.
    """
    sorted_conds = sorted(conditions.keys())
    if not sorted_conds:
        return ""

    lines = [
        "## Phase Contribution Waterfall",
        "",
        "| Phase | Pass Rate | Delta | Cumulative |",
        "|-------|-----------|-------|------------|",
    ]

    prev_rate = 0.0
    for cond in sorted_conds:
        results = conditions[cond]
        mean_rate, _ = multi_seed_pass_rate(results)
        delta = mean_rate - prev_rate

        bar_len = int(mean_rate * 40)
        bar = "█" * bar_len

        lines.append(
            f"| {cond} | {mean_rate:.1%} | "
            f"{delta:+.1f}pp | {bar} |"
        )
        prev_rate = mean_rate

    return "\n".join(lines)


def latency_summary(conditions: Dict[str, List[ConditionResult]]) -> str:
    """Generate per-condition latency summary.

    Args:
        conditions: Dict mapping condition name to list of per-seed results.

    Returns:
        Markdown table string.
    """
    lines = [
        "## Latency Summary",
        "",
        "| Condition | Median (ms) | P50 | P90 | P99 | Mean |",
        "|-----------|-------------|-----|-----|-----|------|",
    ]

    for cond in sorted(conditions.keys()):
        results = conditions[cond]
        all_latencies = [o.latency_ms for r in results for o in r.outcomes
                         if o.latency_ms > 0]

        if not all_latencies:
            lines.append(f"| {cond} | — | — | — | — | — |")
            continue

        all_latencies.sort()
        n = len(all_latencies)
        median = all_latencies[n // 2]
        p50 = all_latencies[int(0.50 * n)]
        p90 = all_latencies[int(0.90 * n)]
        p99 = all_latencies[min(int(0.99 * n), n - 1)]
        mean = sum(all_latencies) / n

        lines.append(
            f"| {cond} | {median:.0f} | {p50:.0f} | "
            f"{p90:.0f} | {p99:.0f} | {mean:.0f} |"
        )

    return "\n".join(lines)


def full_report(conditions: Dict[str, List[ConditionResult]],
                n_resamples: int = 10000) -> str:
    """Generate complete ablation report.

    Args:
        conditions: Dict mapping condition name to list of per-seed results.
        n_resamples: Bootstrap resamples for significance tests.

    Returns:
        Full markdown report.
    """
    sections = [
        "# ATLAS V3.1 Ablation Report\n",
        ablation_table(conditions),
        "",
        pairwise_significance(conditions, n_resamples=n_resamples),
        "",
        phase_waterfall(conditions),
        "",
        latency_summary(conditions),
    ]
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m benchmark.v3.ablation_analysis <telemetry_dir>")
        print("Expects per-condition JSONL files in the directory.")
        sys.exit(1)

    telemetry_dir = Path(sys.argv[1])
    condition_names = ["A", "B", "C", "D", "E", "F"]
    seeds = [42, 43, 44]

    conditions: Dict[str, List[ConditionResult]] = {}
    for cond in condition_names:
        cond_results = []
        for seed in seeds:
            result = load_condition(telemetry_dir, cond, seed)
            if result.n_tasks > 0:
                cond_results.append(result)
        if cond_results:
            conditions[cond] = cond_results

    if not conditions:
        print(f"No telemetry data found in {telemetry_dir}")
        sys.exit(1)

    report = full_report(conditions)
    print(report)
