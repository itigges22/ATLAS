"""Tests for V3 Metacognitive Model (Feature 3F)."""

import json
from pathlib import Path
from typing import Optional, Tuple

import pytest

from benchmark.v3.metacognitive import (
    BenchmarkResult,
    FailurePattern,
    MetacognitiveConfig,
    MetacognitiveEvent,
    MetacognitiveProfile,
    parse_patterns,
)


# ---------------------------------------------------------------------------
# Mock callables
# ---------------------------------------------------------------------------

PATTERN_RESPONSE = (
    "PATTERN 1:\n"
    "DESCRIPTION: Incorrect shift direction on bitwise operations\n"
    "FREQUENCY: 0.73\n"
    "COMPENSATION: Always verify shift direction with a concrete example\n"
    "\n"
    "PATTERN 2:\n"
    "DESCRIPTION: Off-by-one in boundary checks\n"
    "FREQUENCY: 0.55\n"
    "COMPENSATION: Use inclusive upper bounds and test edge values\n"
)


class MockLLM:
    """Mock LLM that returns predefined pattern analysis."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list = []

    def __call__(self, prompt: str, temperature: float,
                 max_tokens: int, seed: Optional[int]) -> Tuple[str, int, float]:
        self.calls.append({
            "prompt": prompt, "temperature": temperature,
            "max_tokens": max_tokens, "seed": seed,
        })
        return self.response, 100, 50.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_telemetry(tmp_path):
    """Provide a temporary telemetry directory."""
    d = tmp_path / "telemetry"
    d.mkdir()
    return d


@pytest.fixture
def mc_enabled(tmp_telemetry):
    """MetacognitiveProfile instance with enabled=True."""
    cfg = MetacognitiveConfig(enabled=True)
    return MetacognitiveProfile(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def mc_disabled(tmp_telemetry):
    """MetacognitiveProfile instance with enabled=False."""
    cfg = MetacognitiveConfig(enabled=False)
    return MetacognitiveProfile(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def mc_with_profile(tmp_telemetry):
    """MetacognitiveProfile with pre-loaded patterns."""
    cfg = MetacognitiveConfig(enabled=True)
    mc = MetacognitiveProfile(cfg, telemetry_dir=tmp_telemetry)
    mc._profile = {
        "bitwise": [
            FailurePattern(
                pattern="Incorrect shift direction",
                frequency=0.73,
                compensation="Verify shift direction with concrete example",
            ),
            FailurePattern(
                pattern="Unsigned vs signed confusion",
                frequency=0.45,
                compensation="Always cast to unsigned before shifting",
            ),
        ],
        "sorting": [
            FailurePattern(
                pattern="Unstable sort assumption",
                frequency=0.60,
                compensation="Use stable sort when order matters",
            ),
        ],
    }
    return mc


@pytest.fixture
def sample_results():
    """Benchmark results with mixed pass/fail across categories."""
    results = []
    # 6 bitwise failures, 4 passes
    for i in range(10):
        results.append(BenchmarkResult(
            task_id=f"bit_{i}",
            category="bitwise",
            passed=i >= 6,
            error=f"shift error {i}" if i < 6 else "",
        ))
    # 3 sorting failures (below threshold of 5)
    for i in range(5):
        results.append(BenchmarkResult(
            task_id=f"sort_{i}",
            category="sorting",
            passed=i >= 3,
            error=f"sort error {i}" if i < 3 else "",
        ))
    # 7 graph failures, 3 passes
    for i in range(10):
        results.append(BenchmarkResult(
            task_id=f"graph_{i}",
            category="graph",
            passed=i >= 7,
            error=f"graph error {i}" if i < 7 else "",
        ))
    return results


# ---------------------------------------------------------------------------
# Test: Disabled noop behavior
# ---------------------------------------------------------------------------

class TestDisabledNoop:
    """When enabled=False, MetacognitiveProfile should be a complete noop."""

    def test_get_warnings_returns_empty(self, mc_disabled):
        warnings = mc_disabled.get_warnings(["bitwise", "sorting"], task_id="t1")
        assert warnings == []

    def test_analyze_benchmark_returns_empty(self, mc_disabled, sample_results):
        result = mc_disabled.analyze_benchmark(results=sample_results, task_id="t1")
        assert result == {}

    def test_no_telemetry_when_disabled(self, mc_disabled, tmp_telemetry):
        mc_disabled.get_warnings(["bitwise"], task_id="t1")
        events_file = tmp_telemetry / "metacognitive_events.jsonl"
        assert not events_file.exists()

    def test_does_not_call_llm_when_disabled(self, mc_disabled, sample_results):
        mock_llm = MockLLM(PATTERN_RESPONSE)
        mc_disabled.analyze_benchmark(
            results=sample_results, llm_call=mock_llm,
        )
        assert len(mock_llm.calls) == 0

    def test_profile_empty_when_disabled(self, mc_disabled):
        assert mc_disabled.total_patterns == 0
        assert mc_disabled.categories == []


# ---------------------------------------------------------------------------
# Test: parse_patterns
# ---------------------------------------------------------------------------

class TestParsePatterns:

    def test_parses_two_patterns(self):
        patterns = parse_patterns(PATTERN_RESPONSE)
        assert len(patterns) == 2

    def test_description_extracted(self):
        patterns = parse_patterns(PATTERN_RESPONSE)
        assert "shift direction" in patterns[0].pattern

    def test_frequency_extracted(self):
        patterns = parse_patterns(PATTERN_RESPONSE)
        assert abs(patterns[0].frequency - 0.73) < 0.01

    def test_compensation_extracted(self):
        patterns = parse_patterns(PATTERN_RESPONSE)
        assert "verify" in patterns[0].compensation.lower()

    def test_percentage_frequency_normalized(self):
        response = (
            "PATTERN 1:\n"
            "DESCRIPTION: Some error\n"
            "FREQUENCY: 73\n"
            "COMPENSATION: Fix it\n"
        )
        patterns = parse_patterns(response)
        assert patterns[0].frequency == pytest.approx(0.73, abs=0.01)

    def test_empty_response(self):
        patterns = parse_patterns("")
        assert patterns == []

    def test_no_pattern_blocks(self):
        patterns = parse_patterns("Just some text without structure.")
        assert patterns == []

    def test_fallback_numbered_items(self):
        response = "1) Off-by-one errors in loop boundaries are very common\n2) Missing edge case for empty input arrays"
        patterns = parse_patterns(response)
        assert len(patterns) == 2

    def test_skips_short_numbered_items(self):
        response = "1) Short\n2) This is a longer pattern description that should be kept"
        patterns = parse_patterns(response)
        assert len(patterns) == 1

    def test_empty_description_skipped(self):
        response = (
            "PATTERN 1:\n"
            "FREQUENCY: 0.5\n"
            "COMPENSATION: Do something\n"
        )
        patterns = parse_patterns(response)
        assert len(patterns) == 0

    def test_case_insensitive(self):
        response = (
            "pattern 1:\n"
            "description: Lower case error\n"
            "frequency: 0.60\n"
            "compensation: Fix it properly\n"
        )
        patterns = parse_patterns(response)
        assert len(patterns) == 1


# ---------------------------------------------------------------------------
# Test: MetacognitiveProfile.get_warnings
# ---------------------------------------------------------------------------

class TestGetWarnings:

    def test_returns_compensations_for_matching_category(self, mc_with_profile):
        warnings = mc_with_profile.get_warnings(["bitwise"])
        assert len(warnings) == 2
        assert any("shift direction" in w.lower() for w in warnings)

    def test_returns_empty_for_unknown_category(self, mc_with_profile):
        warnings = mc_with_profile.get_warnings(["dynamic_programming"])
        assert warnings == []

    def test_multiple_categories(self, mc_with_profile):
        warnings = mc_with_profile.get_warnings(["bitwise", "sorting"])
        assert len(warnings) == 3

    def test_skips_entries_with_negative_effectiveness(self, mc_with_profile):
        mc_with_profile._profile["bitwise"][0].effectiveness = -0.1
        warnings = mc_with_profile.get_warnings(["bitwise"])
        assert len(warnings) == 1
        assert "shift direction" not in warnings[0].lower()

    def test_skips_entries_with_zero_effectiveness(self, mc_with_profile):
        mc_with_profile._profile["bitwise"][0].effectiveness = 0.0
        warnings = mc_with_profile.get_warnings(["bitwise"])
        assert len(warnings) == 1

    def test_keeps_entries_with_positive_effectiveness(self, mc_with_profile):
        mc_with_profile._profile["bitwise"][0].effectiveness = 0.5
        warnings = mc_with_profile.get_warnings(["bitwise"])
        assert len(warnings) == 2

    def test_keeps_entries_with_none_effectiveness(self, mc_with_profile):
        mc_with_profile._profile["bitwise"][0].effectiveness = None
        warnings = mc_with_profile.get_warnings(["bitwise"])
        assert len(warnings) == 2

    def test_skips_entries_without_compensation(self, mc_with_profile):
        mc_with_profile._profile["bitwise"][0].compensation = ""
        warnings = mc_with_profile.get_warnings(["bitwise"])
        assert len(warnings) == 1

    def test_empty_categories_returns_empty(self, mc_with_profile):
        warnings = mc_with_profile.get_warnings([])
        assert warnings == []


# ---------------------------------------------------------------------------
# Test: MetacognitiveProfile.analyze_benchmark
# ---------------------------------------------------------------------------

class TestAnalyzeBenchmark:

    def test_without_llm_creates_rate_patterns(self, mc_enabled, sample_results):
        new_patterns = mc_enabled.analyze_benchmark(
            results=sample_results, task_id="run1",
        )
        # bitwise (6 failures out of 10) and graph (7 failures out of 10)
        # should both exceed min_failures_per_category=5
        assert "bitwise" in new_patterns or "graph" in new_patterns

    def test_with_llm_extracts_patterns(self, mc_enabled, sample_results):
        mock_llm = MockLLM(PATTERN_RESPONSE)
        new_patterns = mc_enabled.analyze_benchmark(
            results=sample_results, llm_call=mock_llm, task_id="run1",
        )
        assert len(mock_llm.calls) > 0
        assert len(new_patterns) > 0

    def test_skips_categories_below_threshold(self, tmp_telemetry):
        cfg = MetacognitiveConfig(enabled=True, min_failures_per_category=10)
        mc = MetacognitiveProfile(cfg, telemetry_dir=tmp_telemetry)
        results = [
            BenchmarkResult(task_id=f"t{i}", category="small", passed=False)
            for i in range(5)
        ]
        new_patterns = mc.analyze_benchmark(results=results)
        assert new_patterns == {}

    def test_filters_by_min_pattern_frequency(self, tmp_telemetry):
        cfg = MetacognitiveConfig(
            enabled=True, min_failures_per_category=1,
            min_pattern_frequency=0.9,
        )
        mc = MetacognitiveProfile(cfg, telemetry_dir=tmp_telemetry)
        results = [
            BenchmarkResult(task_id=f"t{i}", category="test", passed=False)
            for i in range(5)
        ]
        # Without LLM, rate pattern is 5/5 = 1.0 which is >= 0.9
        new_patterns = mc.analyze_benchmark(results=results)
        assert "test" in new_patterns

    def test_does_not_duplicate_patterns(self, mc_enabled):
        results = [
            BenchmarkResult(task_id=f"t{i}", category="cat", passed=False)
            for i in range(10)
        ]
        mc_enabled.analyze_benchmark(results=results, task_id="run1")
        count_after_first = mc_enabled.total_patterns
        # Analyze again with same results
        mc_enabled.analyze_benchmark(results=results, task_id="run2")
        count_after_second = mc_enabled.total_patterns
        assert count_after_second == count_after_first

    def test_profile_updated_after_analysis(self, mc_enabled, sample_results):
        assert mc_enabled.total_patterns == 0
        mc_enabled.analyze_benchmark(results=sample_results)
        assert mc_enabled.total_patterns > 0

    def test_returns_empty_for_all_passing(self, mc_enabled):
        results = [
            BenchmarkResult(task_id=f"t{i}", category="easy", passed=True)
            for i in range(10)
        ]
        new_patterns = mc_enabled.analyze_benchmark(results=results)
        assert new_patterns == {}


# ---------------------------------------------------------------------------
# Test: update_effectiveness
# ---------------------------------------------------------------------------

class TestUpdateEffectiveness:

    def test_updates_matching_pattern(self, mc_with_profile):
        mc_with_profile.update_effectiveness("bitwise", "Incorrect shift direction", 0.8)
        pattern = mc_with_profile._profile["bitwise"][0]
        assert pattern.effectiveness == 0.8

    def test_no_error_for_missing_category(self, mc_with_profile):
        mc_with_profile.update_effectiveness("nonexistent", "pattern", 0.5)

    def test_no_error_for_missing_pattern(self, mc_with_profile):
        mc_with_profile.update_effectiveness("bitwise", "nonexistent pattern", 0.5)
        # Original patterns unchanged
        assert mc_with_profile._profile["bitwise"][0].effectiveness is None


# ---------------------------------------------------------------------------
# Test: Profile persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_and_load(self, tmp_path):
        profile_path = str(tmp_path / "profile.json")
        cfg = MetacognitiveConfig(enabled=True, profile_path=profile_path)

        # Create and populate
        mc1 = MetacognitiveProfile(cfg)
        mc1._profile["test_cat"] = [
            FailurePattern(
                pattern="Test pattern",
                frequency=0.7,
                compensation="Test compensation",
                effectiveness=0.5,
            ),
        ]
        mc1._save(profile_path)

        # Load in new instance
        mc2 = MetacognitiveProfile(cfg)
        assert "test_cat" in mc2._profile
        assert len(mc2._profile["test_cat"]) == 1
        p = mc2._profile["test_cat"][0]
        assert p.pattern == "Test pattern"
        assert p.frequency == 0.7
        assert p.compensation == "Test compensation"
        assert p.effectiveness == 0.5

    def test_load_missing_file(self, tmp_path):
        profile_path = str(tmp_path / "nonexistent.json")
        cfg = MetacognitiveConfig(enabled=True, profile_path=profile_path)
        mc = MetacognitiveProfile(cfg)
        assert mc.total_patterns == 0

    def test_load_invalid_json(self, tmp_path):
        profile_path = str(tmp_path / "bad.json")
        with open(profile_path, "w") as f:
            f.write("not valid json {{{")
        cfg = MetacognitiveConfig(enabled=True, profile_path=profile_path)
        mc = MetacognitiveProfile(cfg)
        assert mc.total_patterns == 0

    def test_analyze_auto_saves(self, tmp_path):
        profile_path = str(tmp_path / "auto_save.json")
        cfg = MetacognitiveConfig(
            enabled=True, profile_path=profile_path,
            min_failures_per_category=1,
        )
        mc = MetacognitiveProfile(cfg)
        results = [
            BenchmarkResult(task_id=f"t{i}", category="cat", passed=False)
            for i in range(5)
        ]
        mc.analyze_benchmark(results=results)
        assert Path(profile_path).exists()
        with open(profile_path) as f:
            data = json.load(f)
        assert "cat" in data


# ---------------------------------------------------------------------------
# Test: Properties
# ---------------------------------------------------------------------------

class TestProperties:

    def test_profile_returns_copy(self, mc_with_profile):
        p = mc_with_profile.profile
        p["new_key"] = []
        assert "new_key" not in mc_with_profile._profile

    def test_categories(self, mc_with_profile):
        cats = mc_with_profile.categories
        assert "bitwise" in cats
        assert "sorting" in cats

    def test_total_patterns(self, mc_with_profile):
        assert mc_with_profile.total_patterns == 3

    def test_empty_profile(self, mc_enabled):
        assert mc_enabled.total_patterns == 0
        assert mc_enabled.categories == []


# ---------------------------------------------------------------------------
# Test: AC-3F-1 — Stores failure patterns per category
# ---------------------------------------------------------------------------

class TestAC3F1PatternStorage:
    """AC-3F-1: System stores failure patterns organized by category."""

    def test_patterns_stored_by_category(self, mc_with_profile):
        profile = mc_with_profile.profile
        assert "bitwise" in profile
        assert "sorting" in profile
        for cat, patterns in profile.items():
            for p in patterns:
                assert isinstance(p, FailurePattern)
                assert len(p.pattern) > 0


# ---------------------------------------------------------------------------
# Test: AC-3F-2 — Injects compensating constraints
# ---------------------------------------------------------------------------

class TestAC3F2Compensation:
    """AC-3F-2: System injects compensating constraints for known weaknesses."""

    def test_compensations_returned(self, mc_with_profile):
        warnings = mc_with_profile.get_warnings(["bitwise"])
        assert len(warnings) > 0
        for w in warnings:
            assert isinstance(w, str)
            assert len(w) > 0


# ---------------------------------------------------------------------------
# Test: AC-3F-3 — Learns from benchmark runs
# ---------------------------------------------------------------------------

class TestAC3F3Learning:
    """AC-3F-3: System learns new patterns from benchmark analysis."""

    def test_new_patterns_discovered(self, mc_enabled, sample_results):
        mock_llm = MockLLM(PATTERN_RESPONSE)
        new_patterns = mc_enabled.analyze_benchmark(
            results=sample_results, llm_call=mock_llm,
        )
        assert len(new_patterns) > 0
        assert mc_enabled.total_patterns > 0


# ---------------------------------------------------------------------------
# Test: AC-3F-4 — Tracks effectiveness
# ---------------------------------------------------------------------------

class TestAC3F4Effectiveness:
    """AC-3F-4: System tracks and uses effectiveness to filter patterns."""

    def test_update_and_filter(self, mc_with_profile):
        mc_with_profile.update_effectiveness(
            "bitwise", "Incorrect shift direction", -0.2,
        )
        warnings = mc_with_profile.get_warnings(["bitwise"])
        # Should skip the negative-effectiveness pattern
        assert not any("shift direction" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Test: Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_lookup_event_written(self, mc_with_profile, tmp_telemetry):
        mc_with_profile.get_warnings(["bitwise"], task_id="LCB_001")
        events_file = tmp_telemetry / "metacognitive_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "LCB_001"
        assert data["operation"] == "lookup"
        assert data["num_warnings"] == 2
        assert "bitwise" in data["category"]
        assert "timestamp" in data

    def test_analyze_event_written(self, mc_enabled, sample_results, tmp_telemetry):
        mc_enabled.analyze_benchmark(
            results=sample_results, task_id="run1",
        )
        events_file = tmp_telemetry / "metacognitive_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "run1"
        assert data["operation"] == "analyze"

    def test_multiple_events_appended(self, mc_with_profile, tmp_telemetry):
        for i in range(3):
            mc_with_profile.get_warnings(["bitwise"], task_id=f"t{i}")
        events_file = tmp_telemetry / "metacognitive_events.jsonl"
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_no_telemetry_without_task_id(self, mc_with_profile, tmp_telemetry):
        mc_with_profile.get_warnings(["bitwise"])
        events_file = tmp_telemetry / "metacognitive_events.jsonl"
        assert not events_file.exists()

    def test_no_crash_without_telemetry_dir(self):
        cfg = MetacognitiveConfig(enabled=True)
        mc = MetacognitiveProfile(cfg, telemetry_dir=None)
        mc._profile["test"] = [
            FailurePattern(pattern="p", compensation="c"),
        ]
        warnings = mc.get_warnings(["test"], task_id="t1")
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Test: Data structures
# ---------------------------------------------------------------------------

class TestDataStructures:

    def test_failure_pattern_to_dict(self):
        fp = FailurePattern(
            pattern="Shift direction error",
            frequency=0.73,
            compensation="Verify with example",
            discovered_at="2026-02-24T00:00:00Z",
            effectiveness=0.8,
        )
        d = fp.to_dict()
        assert d["pattern"] == "Shift direction error"
        assert d["frequency"] == 0.73
        assert d["compensation"] == "Verify with example"
        assert d["discovered_at"] == "2026-02-24T00:00:00Z"
        assert d["effectiveness"] == 0.8

    def test_failure_pattern_none_effectiveness_excluded(self):
        fp = FailurePattern(pattern="test", effectiveness=None)
        d = fp.to_dict()
        assert "effectiveness" not in d

    def test_failure_pattern_from_dict(self):
        d = {
            "pattern": "Test pattern",
            "frequency": 0.5,
            "compensation": "Fix it",
            "discovered_at": "2026-01-01T00:00:00Z",
            "effectiveness": 0.6,
        }
        fp = FailurePattern.from_dict(d)
        assert fp.pattern == "Test pattern"
        assert fp.frequency == 0.5
        assert fp.compensation == "Fix it"
        assert fp.effectiveness == 0.6

    def test_failure_pattern_from_dict_defaults(self):
        fp = FailurePattern.from_dict({})
        assert fp.pattern == ""
        assert fp.frequency == 0.0
        assert fp.compensation == ""
        assert fp.effectiveness is None

    def test_benchmark_result_to_dict(self):
        r = BenchmarkResult(
            task_id="t1", category="bitwise", passed=False,
            code="def solve(): pass", error="wrong answer",
        )
        d = r.to_dict()
        assert d["task_id"] == "t1"
        assert d["category"] == "bitwise"
        assert d["passed"] is False

    def test_event_to_dict(self):
        e = MetacognitiveEvent(
            task_id="t1", operation="lookup",
            category="bitwise", num_warnings=3, num_patterns=5,
        )
        d = e.to_dict()
        assert d["task_id"] == "t1"
        assert d["operation"] == "lookup"
        assert d["category"] == "bitwise"
        assert d["num_warnings"] == 3
        assert d["num_patterns"] == 5
        assert "timestamp" in d

    def test_profile_to_dict(self, mc_with_profile):
        d = mc_with_profile.to_dict()
        assert "bitwise" in d
        assert "sorting" in d
        assert len(d["bitwise"]) == 2
        assert len(d["sorting"]) == 1

    def test_config_defaults(self):
        cfg = MetacognitiveConfig()
        assert cfg.enabled is False
        assert cfg.min_failures_per_category == 5
        assert cfg.min_pattern_frequency == 0.5
        assert cfg.profile_path == ""
