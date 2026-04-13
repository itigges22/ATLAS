"""Tests for V3 Lens Feedback — online recalibration during benchmarks."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.v3.lens_feedback import LensFeedbackCollector, LensFeedbackConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def telemetry_dir(tmp_path):
    d = tmp_path / "telemetry"
    d.mkdir()
    return d


@pytest.fixture
def dummy_embedding():
    return [0.1] * 4096


def make_collector(telemetry_dir, enabled=True, interval=5, min_pass=2,
                   min_fail=2):
    return LensFeedbackCollector(
        LensFeedbackConfig(
            enabled=enabled,
            retrain_interval=interval,
            min_pass=min_pass,
            min_fail=min_fail,
            rag_api_url="http://localhost:31144",
        ),
        telemetry_dir=telemetry_dir,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_defaults(self):
        cfg = LensFeedbackConfig()
        assert cfg.enabled is False
        assert cfg.retrain_interval == 50
        assert cfg.min_pass == 5
        assert cfg.min_fail == 5
        assert cfg.domain == "LCB"
        assert cfg.use_replay is True
        assert cfg.use_ewc is True
        assert cfg.retrain_epochs == 50

    def test_custom_values(self):
        cfg = LensFeedbackConfig(enabled=True, retrain_interval=25)
        assert cfg.enabled is True
        assert cfg.retrain_interval == 25


class TestDisabledNoop:
    def test_record_noop_when_disabled(self, telemetry_dir, dummy_embedding):
        collector = make_collector(telemetry_dir, enabled=False)
        collector.record(dummy_embedding, "PASS", "task_1")
        assert len(collector._buffer) == 0
        assert len(collector._all_data) == 0

    def test_no_telemetry_when_disabled(self, telemetry_dir, dummy_embedding):
        collector = make_collector(telemetry_dir, enabled=False)
        for i in range(10):
            collector.record(dummy_embedding, "PASS", f"task_{i}")
        telemetry_file = telemetry_dir / "lens_feedback_events.jsonl"
        assert not telemetry_file.exists()


class TestRecordAndRetrain:
    def test_accumulates_buffer(self, telemetry_dir, dummy_embedding):
        collector = make_collector(telemetry_dir, interval=10)
        for i in range(3):
            collector.record(dummy_embedding, "PASS", f"task_{i}")
        assert len(collector._buffer) == 3
        assert len(collector._all_data) == 3
        assert collector._retrain_count == 0

    @patch("benchmark.v3.lens_feedback.urllib.request.urlopen")
    def test_triggers_retrain_at_threshold(self, mock_urlopen,
                                            telemetry_dir, dummy_embedding):
        """Buffer hitting interval triggers retrain POST."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "ok",
            "metrics": {
                "pass_energy_mean": 3.0,
                "fail_energy_mean": 12.0,
                "val_auc": 0.95,
            },
        }).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        collector = make_collector(telemetry_dir, interval=5,
                                   min_pass=2, min_fail=2)
        # Add 3 PASS + 2 FAIL = 5 total
        for i in range(3):
            collector.record(dummy_embedding, "PASS", f"pass_{i}")
        for i in range(2):
            collector.record(dummy_embedding, "FAIL", f"fail_{i}")

        assert collector._retrain_count == 1
        assert len(collector._buffer) == 0  # Cleared after retrain
        assert len(collector._all_data) == 5  # Kept for next retrain
        mock_urlopen.assert_called_once()

    @patch("benchmark.v3.lens_feedback.urllib.request.urlopen")
    def test_skips_retrain_insufficient_labels(self, mock_urlopen,
                                                telemetry_dir,
                                                dummy_embedding):
        """Buffer hitting interval but not enough PASS/FAIL skips retrain."""
        collector = make_collector(telemetry_dir, interval=5,
                                   min_pass=3, min_fail=3)
        # All PASS — not enough FAIL
        for i in range(5):
            collector.record(dummy_embedding, "PASS", f"pass_{i}")

        assert collector._retrain_count == 0
        mock_urlopen.assert_not_called()


class TestRecalibrationMath:
    def test_midpoint_steepness_computation(self, telemetry_dir):
        collector = make_collector(telemetry_dir)
        collector._recompute_normalization(pass_mean=3.0, fail_mean=12.0)

        assert collector.current_midpoint == pytest.approx(7.5)
        # steepness = 4.0 / (12.0 - 3.0) = 4/9 ≈ 0.4444
        assert collector.current_steepness == pytest.approx(4.0 / 9.0)
        assert collector.needs_propagation is True

    def test_small_separation_clamped(self, telemetry_dir):
        """Near-zero separation should not cause division by zero."""
        collector = make_collector(telemetry_dir)
        collector._recompute_normalization(pass_mean=5.0, fail_mean=5.01)

        assert collector.current_midpoint == pytest.approx(5.005)
        # steepness = 4.0 / max(0.01, 0.1) = 4.0 / 0.1 = 40.0
        assert collector.current_steepness == pytest.approx(40.0)

    def test_zero_separation_clamped(self, telemetry_dir):
        """Exact same means should use 0.1 floor."""
        collector = make_collector(telemetry_dir)
        collector._recompute_normalization(pass_mean=5.0, fail_mean=5.0)

        assert collector.current_midpoint == pytest.approx(5.0)
        assert collector.current_steepness == pytest.approx(40.0)


class TestComponentPropagation:
    def test_updates_blend_asc_and_budget_forcing(self, telemetry_dir):
        collector = make_collector(telemetry_dir)
        collector._recompute_normalization(pass_mean=3.0, fail_mean=12.0)

        mock_blend = MagicMock()
        mock_blend.config = MagicMock()
        mock_blend.config.energy_midpoint = 9.5
        mock_blend.config.energy_steepness = 0.5

        mock_bf = MagicMock()
        mock_bf.config = MagicMock()
        mock_bf.config.energy_midpoint = 9.5
        mock_bf.config.energy_steepness = 0.5

        collector.apply_to_components(mock_blend, mock_bf)

        assert mock_blend.config.energy_midpoint == pytest.approx(7.5)
        assert mock_blend.config.energy_steepness == pytest.approx(4.0 / 9.0)
        assert mock_bf.config.energy_midpoint == pytest.approx(7.5)
        assert mock_bf.config.energy_steepness == pytest.approx(4.0 / 9.0)
        assert collector.needs_propagation is False

    def test_noop_without_propagation_needed(self, telemetry_dir):
        collector = make_collector(telemetry_dir)
        # needs_propagation is False by default, so apply should be a noop
        assert collector.needs_propagation is False

        mock_blend = MagicMock()
        mock_blend.config.energy_midpoint = 9.5
        mock_blend.config.energy_steepness = 0.5

        collector.apply_to_components(mock_blend, None)

        # Values should remain unchanged
        assert mock_blend.config.energy_midpoint == 9.5
        assert mock_blend.config.energy_steepness == 0.5


class TestTelemetry:
    @patch("benchmark.v3.lens_feedback.urllib.request.urlopen")
    def test_retrain_event_logged(self, mock_urlopen, telemetry_dir,
                                   dummy_embedding):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "ok",
            "metrics": {
                "pass_energy_mean": 4.0,
                "fail_energy_mean": 15.0,
                "val_auc": 0.92,
            },
        }).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        collector = make_collector(telemetry_dir, interval=4,
                                   min_pass=1, min_fail=1)
        collector.record(dummy_embedding, "PASS", "t1")
        collector.record(dummy_embedding, "PASS", "t2")
        collector.record(dummy_embedding, "FAIL", "t3")
        collector.record(dummy_embedding, "FAIL", "t4")

        telemetry_file = telemetry_dir / "lens_feedback_events.jsonl"
        assert telemetry_file.exists()

        events = []
        with open(telemetry_file) as f:
            for line in f:
                events.append(json.loads(line))

        retrain_events = [e for e in events if e["type"] == "retrain"]
        assert len(retrain_events) == 1
        event = retrain_events[0]
        assert event["retrain_count"] == 1
        assert event["pass_energy_mean"] == 4.0
        assert event["fail_energy_mean"] == 15.0
        assert event["new_midpoint"] == pytest.approx(9.5)
        assert "timestamp" in event

    @patch("benchmark.v3.lens_feedback.urllib.request.urlopen")
    def test_propagation_event_logged(self, mock_urlopen, telemetry_dir,
                                       dummy_embedding):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "ok",
            "metrics": {
                "pass_energy_mean": 3.0,
                "fail_energy_mean": 12.0,
                "val_auc": 0.90,
            },
        }).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        collector = make_collector(telemetry_dir, interval=3,
                                   min_pass=1, min_fail=1)
        collector.record(dummy_embedding, "PASS", "t1")
        collector.record(dummy_embedding, "FAIL", "t2")
        collector.record(dummy_embedding, "FAIL", "t3")

        mock_blend = MagicMock()
        mock_blend.config = MagicMock()
        mock_bf = MagicMock()
        mock_bf.config = MagicMock()
        collector.apply_to_components(mock_blend, mock_bf)

        telemetry_file = telemetry_dir / "lens_feedback_events.jsonl"
        events = []
        with open(telemetry_file) as f:
            for line in f:
                events.append(json.loads(line))

        prop_events = [e for e in events if e["type"] == "propagation"]
        assert len(prop_events) == 1
        assert prop_events[0]["midpoint"] == pytest.approx(7.5)
