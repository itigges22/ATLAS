"""Tests for V3 Constraint Refinement (Feature 3B)."""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from benchmark.v3.constraint_refinement import (
    ConstraintRefinementConfig,
    ConstraintRefinementEvent,
    ConstraintRefiner,
    RefinedHypothesis,
    RefinementResult,
    cosine_distance,
    filter_by_distance,
    parse_hypotheses,
)
from benchmark.v3.failure_analysis import FailureAnalysis


# ---------------------------------------------------------------------------
# Mock callables
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns predefined responses."""

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


class MockEmbed:
    """Mock embedding callable that returns vectors based on text."""

    def __init__(self, dim: int = 10):
        self.dim = dim
        self.calls: list = []

    def __call__(self, text: str) -> List[float]:
        self.calls.append(text)
        # Return a vector based on text hash for determinism
        h = hash(text) % 1000
        return [float(h % (i + 1)) / 100.0 for i in range(self.dim)]


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
def cr_enabled(tmp_telemetry):
    """ConstraintRefiner instance with enabled=True."""
    cfg = ConstraintRefinementConfig(enabled=True)
    return ConstraintRefiner(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def cr_disabled(tmp_telemetry):
    """ConstraintRefiner instance with enabled=False."""
    cfg = ConstraintRefinementConfig(enabled=False)
    return ConstraintRefiner(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def sample_failure_analysis():
    """A FailureAnalysis result for testing."""
    return FailureAnalysis(
        categories={0: "wrong_algorithm", 1: "edge_case_miss"},
        violated_constraints=["Must handle empty input"],
        common_pattern="All solutions assume non-empty input",
        new_constraints=["Check for empty input before processing"],
    )


@pytest.fixture
def structured_llm_response():
    """An LLM response in the structured hypothesis format."""
    return (
        "HYPOTHESIS 1:\n"
        "APPROACH: Use dynamic programming with memoization\n"
        "RATIONALE: Avoids the exponential blowup from recursive approach\n"
        "CONSTRAINTS:\n"
        "- Must handle empty input\n"
        "- NEW: Use bottom-up DP to avoid stack overflow\n"
        "\n"
        "HYPOTHESIS 2:\n"
        "APPROACH: Sort first, then use two pointers\n"
        "RATIONALE: Sorting reduces the problem to linear scan\n"
        "CONSTRAINTS:\n"
        "- Must handle empty input\n"
        "- NEW: Sort the array first for O(n log n) complexity\n"
        "- NEW: Handle duplicate elements in the sorted array\n"
    )


# ---------------------------------------------------------------------------
# Test: Disabled noop behavior
# ---------------------------------------------------------------------------

class TestDisabledNoop:
    """When enabled=False, ConstraintRefiner should be a complete noop."""

    def test_returns_empty_result(self, cr_disabled, sample_failure_analysis):
        result = cr_disabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=["Must sort"],
            llm_call=MockLLM("should not run"),
        )
        assert result.hypotheses == []
        assert result.filtered_count == 0
        assert result.total_generated == 0

    def test_no_telemetry_when_disabled(self, cr_disabled, sample_failure_analysis, tmp_telemetry):
        cr_disabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=[], llm_call=MockLLM("x"), task_id="t1",
        )
        events_file = tmp_telemetry / "constraint_refinement_events.jsonl"
        assert not events_file.exists()

    def test_does_not_call_llm_when_disabled(self, cr_disabled, sample_failure_analysis):
        mock_llm = MockLLM("should not run")
        cr_disabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=[], llm_call=mock_llm,
        )
        assert len(mock_llm.calls) == 0


# ---------------------------------------------------------------------------
# Test: cosine_distance
# ---------------------------------------------------------------------------

class TestCosineDistance:

    def test_identical_vectors(self):
        a = [1.0, 2.0, 3.0]
        dist = cosine_distance(a, a)
        assert abs(dist) < 1e-10, f"Identical vectors should have distance 0, got {dist}"

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        dist = cosine_distance(a, b)
        assert abs(dist - 1.0) < 1e-10, f"Orthogonal vectors should have distance 1.0, got {dist}"

    def test_opposite_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        dist = cosine_distance(a, b)
        assert abs(dist - 2.0) < 1e-10, f"Opposite vectors should have distance 2.0, got {dist}"

    def test_similar_vectors_low_distance(self):
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.1]
        dist = cosine_distance(a, b)
        assert dist < 0.01, f"Similar vectors should have low distance, got {dist}"

    def test_empty_vectors(self):
        dist = cosine_distance([], [])
        assert dist == 1.0, "Empty vectors should return default 1.0"

    def test_mismatched_lengths(self):
        dist = cosine_distance([1.0, 2.0], [1.0, 2.0, 3.0])
        assert dist == 1.0, "Mismatched lengths should return default 1.0"

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        dist = cosine_distance(a, b)
        assert dist == 1.0, "Zero vector should return default 1.0"

    def test_range_is_0_to_2(self):
        """Cosine distance should always be in [0, 2]."""
        test_cases = [
            ([1.0, 0.0], [1.0, 0.0]),      # identical
            ([1.0, 0.0], [0.0, 1.0]),      # orthogonal
            ([1.0, 0.0], [-1.0, 0.0]),     # opposite
            ([3.0, 4.0], [1.0, 2.0]),      # arbitrary
        ]
        for a, b in test_cases:
            dist = cosine_distance(a, b)
            assert 0.0 - 1e-10 <= dist <= 2.0 + 1e-10, (
                f"Distance {dist} out of range for {a}, {b}"
            )

    def test_symmetric(self):
        a = [1.0, 3.0, 5.0]
        b = [2.0, 4.0, 6.0]
        assert abs(cosine_distance(a, b) - cosine_distance(b, a)) < 1e-10


# ---------------------------------------------------------------------------
# Test: parse_hypotheses
# ---------------------------------------------------------------------------

class TestParseHypotheses:

    def test_parses_structured_response(self, structured_llm_response):
        original = ["Must handle empty input"]
        hypotheses = parse_hypotheses(structured_llm_response, original)
        assert len(hypotheses) == 2

    def test_includes_original_constraints(self, structured_llm_response):
        original = ["Must handle empty input"]
        hypotheses = parse_hypotheses(structured_llm_response, original)
        for h in hypotheses:
            assert "Must handle empty input" in h.constraints

    def test_extracts_approach(self, structured_llm_response):
        hypotheses = parse_hypotheses(structured_llm_response, [])
        assert len(hypotheses[0].approach) > 0
        assert "dynamic programming" in hypotheses[0].approach.lower()

    def test_extracts_rationale(self, structured_llm_response):
        hypotheses = parse_hypotheses(structured_llm_response, [])
        assert len(hypotheses[0].rationale) > 0

    def test_extracts_new_constraints(self, structured_llm_response):
        hypotheses = parse_hypotheses(structured_llm_response, [])
        for h in hypotheses:
            assert len(h.new_constraints) >= 1

    def test_new_constraints_in_full_list(self, structured_llm_response):
        original = ["Must handle empty input"]
        hypotheses = parse_hypotheses(structured_llm_response, original)
        for h in hypotheses:
            for nc in h.new_constraints:
                assert nc in h.constraints

    def test_empty_response(self):
        hypotheses = parse_hypotheses("", ["c1"])
        assert hypotheses == []

    def test_no_hypotheses_found(self):
        hypotheses = parse_hypotheses("Just some random text.", ["c1"])
        assert hypotheses == []

    def test_single_hypothesis(self):
        response = (
            "HYPOTHESIS 1:\n"
            "APPROACH: Brute force\n"
            "RATIONALE: Simple and correct\n"
            "CONSTRAINTS:\n"
            "- NEW: Check all pairs\n"
        )
        hypotheses = parse_hypotheses(response, [])
        assert len(hypotheses) == 1
        assert "Brute force" in hypotheses[0].approach


# ---------------------------------------------------------------------------
# Test: filter_by_distance
# ---------------------------------------------------------------------------

class TestFilterByDistance:

    def test_no_failed_embeddings_keeps_all(self):
        hypotheses = [
            RefinedHypothesis(approach="a", embedding=[1.0, 0.0]),
            RefinedHypothesis(approach="b", embedding=[0.0, 1.0]),
        ]
        result = filter_by_distance(hypotheses, [], 0.15)
        assert len(result) == 2

    def test_no_embedding_keeps_hypothesis(self):
        """Hypotheses without embeddings pass through."""
        hypotheses = [
            RefinedHypothesis(approach="a", embedding=[]),
            RefinedHypothesis(approach="b"),
        ]
        failed = [[1.0, 0.0]]
        result = filter_by_distance(hypotheses, failed, 0.15)
        assert len(result) == 2

    def test_close_hypothesis_filtered(self):
        """Hypothesis too close to a failure should be filtered."""
        failed_emb = [1.0, 0.0, 0.0]
        # Nearly identical embedding
        hypotheses = [
            RefinedHypothesis(approach="a", embedding=[1.0, 0.001, 0.0]),
        ]
        result = filter_by_distance(hypotheses, [failed_emb], min_distance=0.15)
        assert len(result) == 0

    def test_distant_hypothesis_kept(self):
        """Hypothesis far from failures should pass."""
        failed_emb = [1.0, 0.0, 0.0]
        # Orthogonal embedding
        hypotheses = [
            RefinedHypothesis(approach="a", embedding=[0.0, 1.0, 0.0]),
        ]
        result = filter_by_distance(hypotheses, [failed_emb], min_distance=0.15)
        assert len(result) == 1

    def test_mixed_filtering(self):
        """Some hypotheses pass, some are filtered."""
        failed = [[1.0, 0.0]]
        hypotheses = [
            RefinedHypothesis(approach="close", embedding=[1.0, 0.001]),  # too close
            RefinedHypothesis(approach="far", embedding=[0.0, 1.0]),      # far enough
        ]
        result = filter_by_distance(hypotheses, failed, min_distance=0.15)
        assert len(result) == 1
        assert result[0].approach == "far"

    def test_custom_min_distance(self):
        """Higher min_distance filters more aggressively."""
        failed = [[1.0, 0.0]]
        hypotheses = [
            RefinedHypothesis(approach="a", embedding=[0.9, 0.4]),
        ]
        # With low threshold, passes
        result_low = filter_by_distance(hypotheses, failed, min_distance=0.01)
        assert len(result_low) == 1
        # With high threshold, filtered
        result_high = filter_by_distance(hypotheses, failed, min_distance=0.99)
        assert len(result_high) == 0


# ---------------------------------------------------------------------------
# Test: ConstraintRefiner.refine (enabled, with mock LLM)
# ---------------------------------------------------------------------------

class TestRefineEnabled:

    def test_basic_refinement(self, cr_enabled, sample_failure_analysis, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = cr_enabled.refine(
            problem="Find two sum",
            failure_analysis=sample_failure_analysis,
            original_constraints=["Must handle empty input"],
            llm_call=mock_llm,
            task_id="t1",
        )
        assert isinstance(result, RefinementResult)
        assert result.total_generated > 0
        assert result.refinement_time_ms > 0

    def test_hypotheses_generated(self, cr_enabled, sample_failure_analysis, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=["c1"], llm_call=mock_llm,
        )
        assert len(result.hypotheses) > 0

    def test_llm_called_once(self, cr_enabled, sample_failure_analysis):
        mock_llm = MockLLM("HYPOTHESIS 1:\nAPPROACH: test\nCONSTRAINTS:\n- NEW: x constraint here")
        cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=[], llm_call=mock_llm,
        )
        assert len(mock_llm.calls) == 1

    def test_no_llm_returns_empty(self, cr_enabled, sample_failure_analysis):
        result = cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=[],
        )
        assert result.hypotheses == []

    def test_embeddings_computed(self, cr_enabled, sample_failure_analysis, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        mock_embed = MockEmbed(dim=5)
        result = cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=["c1"], llm_call=mock_llm,
            embed_call=mock_embed,
        )
        for h in result.hypotheses:
            if h.approach:
                assert len(h.embedding) == 5

    def test_filtering_applied(self, cr_enabled, sample_failure_analysis, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        mock_embed = MockEmbed(dim=5)
        # Provide a failed embedding that is close to what MockEmbed returns
        failed_embs = [mock_embed("similar text")]
        result = cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=["c1"], llm_call=mock_llm,
            embed_call=mock_embed, failed_embeddings=failed_embs,
        )
        # Some may have been filtered
        assert result.total_generated >= len(result.hypotheses)

    def test_metacognitive_warnings_included(self, cr_enabled, sample_failure_analysis):
        mock_llm = MockLLM("HYPOTHESIS 1:\nAPPROACH: test\nCONSTRAINTS:\n- NEW: new thing here")
        cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=[], llm_call=mock_llm,
            metacognitive_warnings=["Beware of integer overflow"],
        )
        prompt = mock_llm.calls[0]["prompt"]
        assert "integer overflow" in prompt

    def test_llm_params(self, cr_enabled, sample_failure_analysis):
        mock_llm = MockLLM("response")
        cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=[], llm_call=mock_llm,
        )
        call = mock_llm.calls[0]
        assert call["temperature"] == 0.5
        assert call["max_tokens"] == 2048
        assert call["seed"] is None  # no seed for diversity


# ---------------------------------------------------------------------------
# Test: AC-3B-1 — Generates refined constraint sets
# ---------------------------------------------------------------------------

class TestAC3B1GeneratesHypotheses:
    """AC-3B-1: Generates refined constraint sets from failure analysis."""

    def test_hypotheses_contain_original_constraints(self, cr_enabled, sample_failure_analysis, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        original = ["Must handle empty input"]
        result = cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=original, llm_call=mock_llm,
        )
        for h in result.hypotheses:
            assert "Must handle empty input" in h.constraints

    def test_hypotheses_have_new_constraints(self, cr_enabled, sample_failure_analysis, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=["c1"], llm_call=mock_llm,
        )
        has_new = any(len(h.new_constraints) > 0 for h in result.hypotheses)
        assert has_new


# ---------------------------------------------------------------------------
# Test: AC-3B-2 — Each hypothesis has a different approach
# ---------------------------------------------------------------------------

class TestAC3B2DifferentApproaches:
    """AC-3B-2: Each hypothesis suggests a different approach."""

    def test_hypotheses_have_approaches(self, cr_enabled, sample_failure_analysis, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=[], llm_call=mock_llm,
        )
        for h in result.hypotheses:
            assert len(h.approach) > 0


# ---------------------------------------------------------------------------
# Test: AC-3B-3 — Geometric distance filtering
# ---------------------------------------------------------------------------

class TestAC3B3DistanceFiltering:
    """AC-3B-3: Hypotheses too close to failures are filtered."""

    def test_filtered_count_tracked(self, cr_enabled, sample_failure_analysis, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        mock_embed = MockEmbed(dim=5)
        # Use the same embed for failures so some hypotheses may be filtered
        failed = [mock_embed("test approach")]
        result = cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=[], llm_call=mock_llm,
            embed_call=mock_embed, failed_embeddings=failed,
        )
        assert result.filtered_count >= 0
        assert result.total_generated == len(result.hypotheses) + result.filtered_count


# ---------------------------------------------------------------------------
# Test: AC-3B-4 — Includes rationale
# ---------------------------------------------------------------------------

class TestAC3B4Rationale:
    """AC-3B-4: Each hypothesis includes rationale for avoiding failures."""

    def test_hypotheses_have_rationale(self, cr_enabled, sample_failure_analysis, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=[], llm_call=mock_llm,
        )
        for h in result.hypotheses:
            assert len(h.rationale) > 0


# ---------------------------------------------------------------------------
# Test: Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_event_written_to_jsonl(self, cr_enabled, sample_failure_analysis, tmp_telemetry, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=["c1"], llm_call=mock_llm, task_id="LCB_001",
        )
        events_file = tmp_telemetry / "constraint_refinement_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "LCB_001"
        assert data["num_hypotheses_generated"] > 0
        assert "num_hypotheses_viable" in data
        assert "num_filtered_by_distance" in data
        assert "refinement_time_ms" in data
        assert "timestamp" in data

    def test_multiple_events_appended(self, cr_enabled, sample_failure_analysis, tmp_telemetry):
        mock_llm = MockLLM("HYPOTHESIS 1:\nAPPROACH: test\nCONSTRAINTS:\n- NEW: something here now")
        for i in range(3):
            cr_enabled.refine(
                problem="test", failure_analysis=sample_failure_analysis,
                original_constraints=[], llm_call=mock_llm, task_id=f"t{i}",
            )
        events_file = tmp_telemetry / "constraint_refinement_events.jsonl"
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_no_telemetry_without_task_id(self, cr_enabled, sample_failure_analysis, tmp_telemetry):
        mock_llm = MockLLM("response")
        cr_enabled.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=[], llm_call=mock_llm,
        )
        events_file = tmp_telemetry / "constraint_refinement_events.jsonl"
        assert not events_file.exists()

    def test_no_crash_without_telemetry_dir(self, sample_failure_analysis):
        cfg = ConstraintRefinementConfig(enabled=True)
        cr = ConstraintRefiner(cfg, telemetry_dir=None)
        mock_llm = MockLLM("HYPOTHESIS 1:\nAPPROACH: test\nCONSTRAINTS:\n- NEW: a new constraint")
        result = cr.refine(
            problem="test", failure_analysis=sample_failure_analysis,
            original_constraints=[], llm_call=mock_llm, task_id="t1",
        )
        assert isinstance(result, RefinementResult)


# ---------------------------------------------------------------------------
# Test: Data structures
# ---------------------------------------------------------------------------

class TestDataStructures:

    def test_refined_hypothesis_to_dict(self):
        h = RefinedHypothesis(
            constraints=["c1", "c2", "new_c"],
            new_constraints=["new_c"],
            approach="Dynamic programming",
            rationale="Avoids exponential blowup",
            embedding=[1.0, 2.0, 3.0],
        )
        d = h.to_dict()
        assert d["num_constraints"] == 3
        assert d["num_new_constraints"] == 1
        assert "Dynamic" in d["approach"]
        assert d["has_embedding"] is True

    def test_refined_hypothesis_no_embedding(self):
        h = RefinedHypothesis(approach="test")
        d = h.to_dict()
        assert d["has_embedding"] is False

    def test_refinement_result_to_dict(self):
        r = RefinementResult(
            hypotheses=[RefinedHypothesis(approach="a")],
            filtered_count=1, total_generated=2,
            refinement_time_ms=100.0,
        )
        d = r.to_dict()
        assert d["num_hypotheses"] == 1
        assert d["filtered_count"] == 1
        assert d["total_generated"] == 2
        assert d["refinement_time_ms"] == 100.0

    def test_event_to_dict(self):
        e = ConstraintRefinementEvent(
            task_id="t1", num_hypotheses_generated=3,
            num_hypotheses_viable=2, num_filtered_by_distance=1,
            refinement_time_ms=200.0,
        )
        d = e.to_dict()
        assert d["task_id"] == "t1"
        assert d["num_hypotheses_generated"] == 3
        assert d["num_filtered_by_distance"] == 1
        assert "timestamp" in d

    def test_config_defaults(self):
        cfg = ConstraintRefinementConfig()
        assert cfg.enabled is False
        assert cfg.num_hypotheses == 3
        assert cfg.min_cosine_distance == 0.15
        assert cfg.refinement_temperature == 0.5
        assert cfg.refinement_max_tokens == 2048
