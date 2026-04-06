"""Tests for V3 ablation analysis module."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.v3.ablation_analysis import (
    TaskOutcome,
    ConditionResult,
    BootstrapResult,
    bootstrap_delta,
    multi_seed_pass_rate,
    extract_outcomes,
    load_telemetry,
    load_condition,
    ablation_table,
    pairwise_significance,
    phase_waterfall,
    latency_summary,
    full_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_outcomes(n_tasks: int, n_pass: int,
                   prefix: str = "task") -> list:
    """Create a list of TaskOutcome with given pass count."""
    outcomes = []
    for i in range(n_tasks):
        outcomes.append(TaskOutcome(
            task_id=f"{prefix}_{i:03d}",
            passed=(i < n_pass),
            latency_ms=100.0 + i * 10,
            tokens_generated=500,
            energy=5.0 if i < n_pass else 15.0,
        ))
    return outcomes


def _make_condition(condition: str, seed: int,
                    n_tasks: int, n_pass: int) -> ConditionResult:
    return ConditionResult(
        condition=condition,
        seed=seed,
        outcomes=_make_outcomes(n_tasks, n_pass),
    )


# ---------------------------------------------------------------------------
# TaskOutcome / ConditionResult
# ---------------------------------------------------------------------------

class TestConditionResult:
    def test_pass_rate(self):
        cr = _make_condition("A", 42, 100, 60)
        assert cr.pass_rate == 0.6
        assert cr.n_tasks == 100
        assert cr.n_pass == 60

    def test_empty(self):
        cr = ConditionResult(condition="A", seed=42)
        assert cr.pass_rate == 0.0
        assert cr.n_tasks == 0

    def test_all_pass(self):
        cr = _make_condition("A", 42, 50, 50)
        assert cr.pass_rate == 1.0


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class TestBootstrapDelta:
    def test_identical_conditions(self):
        """Same outcomes should yield delta near 0."""
        outcomes = _make_outcomes(100, 50)
        result = bootstrap_delta(outcomes, outcomes, n_resamples=1000)
        assert abs(result.mean_delta) < 0.01
        assert not result.significant

    def test_clearly_different(self):
        """Large difference should be significant."""
        outcomes_a = _make_outcomes(100, 30, prefix="a")
        outcomes_b = _make_outcomes(100, 70, prefix="b")
        result = bootstrap_delta(outcomes_a, outcomes_b, n_resamples=5000)
        assert result.mean_delta > 0.3
        assert result.significant
        assert result.p_value < 0.05

    def test_paired_bootstrap(self):
        """When task IDs overlap, uses paired resampling."""
        outcomes_a = _make_outcomes(100, 40)
        outcomes_b = _make_outcomes(100, 60)
        result = bootstrap_delta(outcomes_a, outcomes_b, n_resamples=1000)
        assert result.mean_delta > 0.15
        assert result.ci_lower > 0  # should be significant

    def test_small_n(self):
        """Works with small sample sizes."""
        outcomes_a = _make_outcomes(10, 3)
        outcomes_b = _make_outcomes(10, 7)
        result = bootstrap_delta(outcomes_a, outcomes_b, n_resamples=500)
        assert result.mean_delta > 0.0

    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        outcomes_a = _make_outcomes(50, 20)
        outcomes_b = _make_outcomes(50, 30)
        r1 = bootstrap_delta(outcomes_a, outcomes_b, seed=123)
        r2 = bootstrap_delta(outcomes_a, outcomes_b, seed=123)
        assert r1.mean_delta == r2.mean_delta
        assert r1.ci_lower == r2.ci_lower


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

class TestMultiSeedPassRate:
    def test_single_seed(self):
        results = [_make_condition("A", 42, 100, 60)]
        mean, std = multi_seed_pass_rate(results)
        assert mean == 0.6
        assert std == 0.0

    def test_multiple_seeds(self):
        results = [
            _make_condition("A", 42, 100, 55),
            _make_condition("A", 43, 100, 60),
            _make_condition("A", 44, 100, 65),
        ]
        mean, std = multi_seed_pass_rate(results)
        assert abs(mean - 0.6) < 0.001
        assert std > 0

    def test_empty(self):
        mean, std = multi_seed_pass_rate([])
        assert mean == 0.0
        assert std == 0.0


# ---------------------------------------------------------------------------
# Telemetry loading
# ---------------------------------------------------------------------------

class TestTelemetryLoading:
    def test_load_telemetry(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as f:
            f.write(json.dumps({"task_id": "t1", "passed": True}) + "\n")
            f.write(json.dumps({"task_id": "t2", "passed": False}) + "\n")
            path = Path(f.name)

        records = load_telemetry(path)
        assert len(records) == 2
        assert records[0]["task_id"] == "t1"
        path.unlink()

    def test_extract_outcomes(self):
        records = [
            {"task_id": "t1", "passed": True, "latency": {"phase3_total": 500}},
            {"task_id": "t2", "final_passed": False, "tokens_generated": 200},
            {"task_id": "t3", "sandbox_passed": True, "energy": 3.5},
        ]
        outcomes = extract_outcomes(records)
        assert len(outcomes) == 3
        assert outcomes[0].passed is True
        assert outcomes[0].latency_ms == 500
        assert outcomes[1].passed is False
        assert outcomes[1].tokens_generated == 200
        assert outcomes[2].energy == 3.5

    def test_extract_skips_no_task_id(self):
        records = [{"passed": True}]
        outcomes = extract_outcomes(records)
        assert len(outcomes) == 0

    def test_load_condition_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "A_seed42.jsonl"
            with open(path, "w") as f:
                for i in range(10):
                    f.write(json.dumps({
                        "task_id": f"t{i}",
                        "passed": i < 6,
                    }) + "\n")

            result = load_condition(Path(tmpdir), "A", 42)
            assert result.n_tasks == 10
            assert result.n_pass == 6
            assert result.condition == "A"
            assert result.seed == 42

    def test_load_condition_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_condition(Path(tmpdir), "X", 99)
            assert result.n_tasks == 0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def _make_conditions(self):
        return {
            "A": [_make_condition("A", 42, 100, 40),
                  _make_condition("A", 43, 100, 42)],
            "B": [_make_condition("B", 42, 100, 55),
                  _make_condition("B", 43, 100, 58)],
            "C": [_make_condition("C", 42, 100, 70),
                  _make_condition("C", 43, 100, 72)],
        }

    def test_ablation_table(self):
        conditions = self._make_conditions()
        table = ablation_table(conditions)
        assert "Ablation Results" in table
        assert "| A |" in table
        assert "| B |" in table
        assert "| C |" in table

    def test_pairwise_significance(self):
        conditions = self._make_conditions()
        table = pairwise_significance(conditions, n_resamples=500)
        assert "Pairwise Significance" in table
        assert "A → B" in table
        assert "B → C" in table

    def test_phase_waterfall(self):
        conditions = self._make_conditions()
        wf = phase_waterfall(conditions)
        assert "Waterfall" in wf
        assert "█" in wf

    def test_latency_summary(self):
        conditions = self._make_conditions()
        summary = latency_summary(conditions)
        assert "Latency Summary" in summary
        assert "P90" in summary

    def test_full_report(self):
        conditions = self._make_conditions()
        report = full_report(conditions, n_resamples=500)
        assert "Ablation Report" in report
        assert "Ablation Results" in report
        assert "Pairwise Significance" in report
        assert "Waterfall" in report
        assert "Latency" in report

    def test_empty_conditions(self):
        table = ablation_table({})
        assert "Ablation Results" in table

    def test_single_condition(self):
        conditions = {"A": [_make_condition("A", 42, 100, 50)]}
        table = ablation_table(conditions)
        assert "50.0%" in table
