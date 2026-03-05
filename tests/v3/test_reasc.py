"""Tests for V3 ReASC (Feature 2B) — Early Stopping via Token Confidence."""

import json
import math
from pathlib import Path

import pytest

from benchmark.v3.reasc import (
    ReASC,
    ReASCConfig,
    ReASCEvent,
    compute_bottom_10_confidence,
    should_early_stop,
)
from benchmark.v3.budget_forcing import normalize_energy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_telemetry(tmp_path):
    d = tmp_path / "telemetry"
    d.mkdir()
    return d


@pytest.fixture
def reasc_enabled(tmp_telemetry):
    cfg = ReASCConfig(enabled=True, confidence_threshold=-0.5)
    return ReASC(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def reasc_disabled(tmp_telemetry):
    cfg = ReASCConfig(enabled=False)
    return ReASC(cfg, telemetry_dir=tmp_telemetry)


# ---------------------------------------------------------------------------
# Test: Disabled noop
# ---------------------------------------------------------------------------

class TestDisabledNoop:

    def test_never_early_stops(self, reasc_disabled):
        stop, reason = reasc_disabled.evaluate(
            raw_energy=2.0, logprobs=[-0.1] * 100, task_id="t1"
        )
        assert stop is False
        assert reason == "disabled"

    def test_no_telemetry_when_disabled(self, reasc_disabled, tmp_telemetry):
        reasc_disabled.evaluate(raw_energy=2.0, logprobs=[-0.1] * 100, task_id="t1")
        events_file = tmp_telemetry / "reasc_events.jsonl"
        assert not events_file.exists()


# ---------------------------------------------------------------------------
# Test: compute_bottom_10_confidence
# ---------------------------------------------------------------------------

class TestComputeBottom10Confidence:

    def test_uniform_logprobs(self):
        """Uniform logprobs should return that value."""
        logprobs = [-0.3] * 100
        conf = compute_bottom_10_confidence(logprobs)
        assert abs(conf - (-0.3)) < 0.001

    def test_bottom_10_is_worst(self):
        """Bottom 10% should be the most negative values."""
        # 90 easy tokens + 10 hard tokens
        logprobs = [-0.1] * 90 + [-2.0] * 10
        conf = compute_bottom_10_confidence(logprobs)
        assert abs(conf - (-2.0)) < 0.001

    def test_single_token(self):
        """Single token: bottom 10% = the only token."""
        conf = compute_bottom_10_confidence([-1.5])
        assert abs(conf - (-1.5)) < 0.001

    def test_two_tokens(self):
        """Two tokens: bottom 10% = 1 token (min 1)."""
        conf = compute_bottom_10_confidence([-0.1, -3.0])
        assert abs(conf - (-3.0)) < 0.001

    def test_empty_logprobs(self):
        conf = compute_bottom_10_confidence([])
        assert conf == 0.0

    def test_ten_tokens(self):
        """10 tokens: bottom 10% = 1 token."""
        logprobs = [-0.1] * 9 + [-5.0]
        conf = compute_bottom_10_confidence(logprobs)
        assert abs(conf - (-5.0)) < 0.001

    def test_twenty_tokens(self):
        """20 tokens: bottom 10% = 2 tokens."""
        logprobs = [-0.1] * 18 + [-1.0, -2.0]
        conf = compute_bottom_10_confidence(logprobs)
        expected = (-1.0 + -2.0) / 2
        assert abs(conf - expected) < 0.001

    def test_all_zero(self):
        """Perfect confidence (logprob 0 = prob 1)."""
        conf = compute_bottom_10_confidence([0.0] * 50)
        assert conf == 0.0

    def test_highly_confident(self):
        """Very confident model: logprobs near 0."""
        logprobs = [-0.01] * 95 + [-0.1] * 5
        conf = compute_bottom_10_confidence(logprobs)
        # Bottom 10% of 100 = 10 tokens, including the 5 at -0.1 and 5 at -0.01
        assert conf > -0.1


# ---------------------------------------------------------------------------
# Test: should_early_stop function
# ---------------------------------------------------------------------------

class TestShouldEarlyStop:

    def test_easy_and_confident_stops(self):
        stop, reason = should_early_stop(0.05, -0.3)
        assert stop is True
        assert reason == "easy_and_confident"

    def test_hard_does_not_stop(self):
        stop, reason = should_early_stop(0.20, -0.3)
        assert stop is False
        assert "energy" in reason

    def test_low_confidence_does_not_stop(self):
        stop, reason = should_early_stop(0.05, -0.8)
        assert stop is False
        assert "confidence" in reason

    def test_boundary_energy(self):
        # At exactly the threshold
        stop, _ = should_early_stop(0.10, -0.3)
        assert stop is False  # >= threshold, should NOT stop

    def test_boundary_confidence(self):
        # At exactly the threshold
        stop, _ = should_early_stop(0.05, -0.5)
        assert stop is False  # <= threshold, should NOT stop

    def test_custom_thresholds(self):
        stop, _ = should_early_stop(
            0.15, -0.3,
            energy_threshold=0.20,
            confidence_threshold=-0.4,
        )
        assert stop is True

    def test_both_fail(self):
        stop, reason = should_early_stop(0.50, -2.0)
        assert stop is False


# ---------------------------------------------------------------------------
# Test: AC-2B-1 — Early stop triggers on >= 30% of easy tasks (structural)
# ---------------------------------------------------------------------------

class TestAC2B1EarlyStopRate:

    def test_easy_confident_tasks_trigger_stop(self, reasc_enabled):
        """Easy tasks with high confidence should trigger early stopping."""
        # Simulate 10 easy tasks with high confidence
        easy_energies = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.2, 4.5, 4.7, 4.9]
        high_confidence_logprobs = [-0.1] * 100  # Very confident

        stop_count = 0
        for e in easy_energies:
            stop, _ = reasc_enabled.evaluate(
                raw_energy=e, logprobs=high_confidence_logprobs
            )
            if stop:
                stop_count += 1

        assert stop_count >= 3, f"Only {stop_count}/10 easy tasks early-stopped"

    def test_hard_tasks_never_trigger(self, reasc_enabled):
        """Hard tasks should never trigger early stopping."""
        hard_energies = [10.0, 12.0, 14.0, 16.0, 18.0]
        for e in hard_energies:
            stop, _ = reasc_enabled.evaluate(
                raw_energy=e, logprobs=[-0.1] * 100
            )
            assert stop is False, f"Hard task (energy={e}) triggered early stop"


# ---------------------------------------------------------------------------
# Test: AC-2B-2 — Early-stopped tasks have >= 95% accuracy (structural)
# ---------------------------------------------------------------------------

class TestAC2B2Accuracy:

    def test_only_stops_when_both_gates_pass(self, reasc_enabled):
        """Early stopping requires BOTH low energy AND high confidence."""
        # Low energy, low confidence → should NOT stop
        stop, _ = reasc_enabled.evaluate(
            raw_energy=3.0, logprobs=[-3.0] * 100
        )
        assert stop is False, "Stopped on low-confidence candidate"

        # High energy, high confidence → should NOT stop
        stop, _ = reasc_enabled.evaluate(
            raw_energy=12.0, logprobs=[-0.1] * 100
        )
        assert stop is False, "Stopped on hard problem"

    def test_dual_gate_prevents_false_positives(self, reasc_enabled):
        """The dual gate (energy + confidence) should be conservative."""
        # Only when BOTH conditions are met
        stop, _ = reasc_enabled.evaluate(
            raw_energy=2.0, logprobs=[-0.1] * 100
        )
        assert stop is True  # Both gates pass


# ---------------------------------------------------------------------------
# Test: AC-2B-3 — Wall-clock reduction (structural)
# ---------------------------------------------------------------------------

class TestAC2B3TimeReduction:

    def test_early_stop_avoids_additional_generations(self, reasc_enabled):
        """When early stopped, no additional candidates should be generated.

        This test validates the structural property — actual time reduction
        depends on runtime integration.
        """
        stop, reason = reasc_enabled.evaluate(
            raw_energy=2.0, logprobs=[-0.1] * 100
        )
        assert stop is True
        assert reason == "easy_and_confident"
        # In the runner, this means k=1 (only probe candidate used)


# ---------------------------------------------------------------------------
# Test: ReASC class
# ---------------------------------------------------------------------------

class TestReASCClass:

    def test_evaluate_easy_confident(self, reasc_enabled):
        stop, reason = reasc_enabled.evaluate(
            raw_energy=2.0, logprobs=[-0.1] * 100, task_id="t1"
        )
        assert stop is True

    def test_evaluate_hard(self, reasc_enabled):
        stop, reason = reasc_enabled.evaluate(
            raw_energy=14.0, logprobs=[-0.1] * 100, task_id="t1"
        )
        assert stop is False

    def test_evaluate_unconfident(self, reasc_enabled):
        stop, reason = reasc_enabled.evaluate(
            raw_energy=2.0, logprobs=[-3.0] * 100, task_id="t1"
        )
        assert stop is False

    def test_compute_confidence(self, reasc_enabled):
        conf = reasc_enabled.compute_confidence([-0.1] * 100)
        assert abs(conf - (-0.1)) < 0.001

    def test_no_crash_without_telemetry_dir(self):
        cfg = ReASCConfig(enabled=True, confidence_threshold=-0.5)
        reasc = ReASC(cfg, telemetry_dir=None)
        stop, _ = reasc.evaluate(raw_energy=2.0, logprobs=[-0.1] * 100, task_id="t1")
        assert isinstance(stop, bool)

    def test_custom_thresholds(self, tmp_telemetry):
        cfg = ReASCConfig(
            enabled=True,
            confidence_threshold=-1.0,
            energy_threshold=0.20,
        )
        reasc = ReASC(cfg, telemetry_dir=tmp_telemetry)
        # Energy 6.0 → normalized ~0.148 (< 0.20), confidence -0.5 (> -1.0)
        stop, _ = reasc.evaluate(raw_energy=6.0, logprobs=[-0.5] * 100)
        assert stop is True


# ---------------------------------------------------------------------------
# Test: Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_event_logged_on_evaluate(self, reasc_enabled, tmp_telemetry):
        reasc_enabled.evaluate(
            raw_energy=5.0, logprobs=[-0.2] * 50, task_id="LCB_001"
        )
        events_file = tmp_telemetry / "reasc_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "LCB_001"
        assert data["raw_energy"] == 5.0
        assert "normalized_energy" in data
        assert "bottom_10_confidence" in data
        assert "early_stopped" in data
        assert "reason" in data
        assert "timestamp" in data

    def test_multiple_events(self, reasc_enabled, tmp_telemetry):
        for i in range(5):
            reasc_enabled.evaluate(
                raw_energy=float(i + 3), logprobs=[-0.2] * 50, task_id=f"t{i}"
            )
        events_file = tmp_telemetry / "reasc_events.jsonl"
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_no_telemetry_without_task_id(self, reasc_enabled, tmp_telemetry):
        reasc_enabled.evaluate(raw_energy=5.0, logprobs=[-0.2] * 50)
        events_file = tmp_telemetry / "reasc_events.jsonl"
        assert not events_file.exists()


# ---------------------------------------------------------------------------
# Test: Data structures
# ---------------------------------------------------------------------------

class TestDataStructures:

    def test_event_to_dict(self):
        e = ReASCEvent(
            task_id="t1", raw_energy=5.0, normalized_energy=0.07,
            bottom_10_confidence=-0.3, early_stopped=True,
            reason="easy_and_confident",
        )
        d = e.to_dict()
        assert d["task_id"] == "t1"
        assert d["early_stopped"] is True
        assert d["reason"] == "easy_and_confident"
        assert "timestamp" in d

    def test_config_defaults(self):
        cfg = ReASCConfig()
        assert cfg.enabled is False
        assert cfg.confidence_threshold == -0.5
        assert cfg.energy_threshold == 0.10
        assert cfg.energy_midpoint == 9.5
        assert cfg.energy_steepness == 0.5


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_logprobs(self, reasc_enabled):
        """Empty logprobs should not crash."""
        stop, _ = reasc_enabled.evaluate(
            raw_energy=2.0, logprobs=[], task_id="t1"
        )
        # Confidence = 0.0, which is > -0.5, and energy is easy
        assert stop is True

    def test_single_logprob(self, reasc_enabled):
        stop, _ = reasc_enabled.evaluate(
            raw_energy=2.0, logprobs=[-0.1], task_id="t1"
        )
        assert stop is True

    def test_very_negative_logprobs(self, reasc_enabled):
        """Very uncertain model should not early stop."""
        stop, _ = reasc_enabled.evaluate(
            raw_energy=2.0, logprobs=[-10.0] * 100, task_id="t1"
        )
        assert stop is False

    def test_mixed_logprobs(self, reasc_enabled):
        """Mix of confident and unconfident tokens."""
        # 90 confident + 10 very uncertain
        logprobs = [-0.05] * 90 + [-5.0] * 10
        stop, _ = reasc_enabled.evaluate(
            raw_energy=2.0, logprobs=logprobs, task_id="t1"
        )
        # Bottom 10% average = -5.0, well below -0.5 threshold
        assert stop is False
