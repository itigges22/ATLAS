"""Tests for V3 Failure Analysis (Feature 3A)."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from benchmark.v3.failure_analysis import (
    FAILURE_CATEGORIES,
    FailingCandidate,
    FailureAnalysis,
    FailureAnalysisConfig,
    FailureAnalysisEvent,
    FailureAnalyzer,
    format_candidates_with_errors,
    parse_common_pattern,
    parse_failure_categories,
    parse_new_constraints,
    parse_violated_constraints,
)


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
    """Mock embedding callable that returns fixed-size vectors."""

    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.calls: list = []

    def __call__(self, text: str) -> List[float]:
        self.calls.append(text)
        # Return a simple deterministic embedding based on text length
        return [float(len(text) % 10) / 10.0] * self.dim


class MockEmbedFailing:
    """Mock embedding callable that raises an exception."""

    def __call__(self, text: str) -> List[float]:
        raise RuntimeError("Embedding service unavailable")


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
def fa_enabled(tmp_telemetry):
    """FailureAnalyzer instance with enabled=True."""
    cfg = FailureAnalysisConfig(enabled=True)
    return FailureAnalyzer(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def fa_disabled(tmp_telemetry):
    """FailureAnalyzer instance with enabled=False."""
    cfg = FailureAnalysisConfig(enabled=False)
    return FailureAnalyzer(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def sample_candidates():
    """A set of failing candidates for testing."""
    return [
        FailingCandidate(
            code="def solve(n):\n    return n * 2",
            error_output="AssertionError: expected 3 got 4",
            index=0,
        ),
        FailingCandidate(
            code="def solve(n):\n    return n + 1",
            error_output="AssertionError: expected 6 got 3",
            index=1,
        ),
    ]


@pytest.fixture
def structured_llm_response():
    """An LLM response in the structured format the parser expects."""
    return (
        "CATEGORY:\n"
        "Solution 1: wrong_algorithm\n"
        "Solution 2: implementation_bug\n"
        "\n"
        "VIOLATED:\n"
        "- Must handle negative numbers\n"
        "- Must return sorted output\n"
        "\n"
        "COMMON:\n"
        "All solutions assume positive input. They fail when given negative numbers.\n"
        "\n"
        "NEW_CONSTRAINTS:\n"
        "- Handle negative input by taking absolute value first\n"
        "- Validate input range before processing\n"
    )


# ---------------------------------------------------------------------------
# Test: Disabled noop behavior
# ---------------------------------------------------------------------------

class TestDisabledNoop:
    """When enabled=False, FailureAnalyzer should be a complete noop."""

    def test_returns_empty_analysis(self, fa_disabled, sample_candidates):
        result = fa_disabled.analyze(
            problem="test problem",
            candidates=sample_candidates,
            original_constraints=["must sort"],
        )
        assert result.categories == {}
        assert result.violated_constraints == []
        assert result.common_pattern == ""
        assert result.new_constraints == []
        assert result.failure_embeddings == []
        assert result.raw_analysis == ""

    def test_no_telemetry_when_disabled(self, fa_disabled, sample_candidates, tmp_telemetry):
        fa_disabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], task_id="t1",
        )
        events_file = tmp_telemetry / "failure_analysis_events.jsonl"
        assert not events_file.exists()

    def test_does_not_call_llm_when_disabled(self, fa_disabled, sample_candidates):
        mock_llm = MockLLM("should not be called")
        fa_disabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        assert len(mock_llm.calls) == 0

    def test_does_not_call_embed_when_disabled(self, fa_disabled, sample_candidates):
        mock_embed = MockEmbed()
        fa_disabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], embed_call=mock_embed,
        )
        assert len(mock_embed.calls) == 0


# ---------------------------------------------------------------------------
# Test: FAILURE_CATEGORIES
# ---------------------------------------------------------------------------

class TestFailureCategories:

    def test_has_six_categories(self):
        assert len(FAILURE_CATEGORIES) == 6

    def test_expected_categories_present(self):
        expected = {
            "wrong_algorithm", "implementation_bug", "edge_case_miss",
            "time_limit", "format_error", "partial_correct",
        }
        assert set(FAILURE_CATEGORIES.keys()) == expected

    def test_descriptions_are_nonempty(self):
        for cat, desc in FAILURE_CATEGORIES.items():
            assert len(desc) > 10, f"Category '{cat}' has too short a description"


# ---------------------------------------------------------------------------
# Test: parse_failure_categories
# ---------------------------------------------------------------------------

class TestParseFailureCategories:

    def test_structured_format_solution_n(self):
        response = "Solution 1: wrong_algorithm\nSolution 2: implementation_bug"
        cats = parse_failure_categories(response, 2)
        assert cats[0] == "wrong_algorithm"
        assert cats[1] == "implementation_bug"

    def test_structured_format_candidate_n(self):
        response = "Candidate 1: edge_case_miss\nCandidate 2: time_limit"
        cats = parse_failure_categories(response, 2)
        assert cats[0] == "edge_case_miss"
        assert cats[1] == "time_limit"

    def test_numbered_format(self):
        response = "1) wrong_algorithm\n2) format_error"
        cats = parse_failure_categories(response, 2)
        assert cats[0] == "wrong_algorithm"
        assert cats[1] == "format_error"

    def test_fallback_to_mention(self):
        """When no structured format found, picks first mentioned category."""
        response = "The code has a clear implementation_bug in the loop."
        cats = parse_failure_categories(response, 2)
        assert len(cats) >= 1
        assert "implementation_bug" in cats.values()

    def test_invalid_category_ignored(self):
        response = "Solution 1: nonexistent_category"
        cats = parse_failure_categories(response, 1)
        assert len(cats) == 0

    def test_out_of_range_index_falls_back(self):
        """Out-of-range "Solution 10" triggers fallback: category mention assigned to first candidate."""
        response = "Solution 10: wrong_algorithm"
        cats = parse_failure_categories(response, 2)
        # Fallback path finds "wrong_algorithm" in text, assigns to candidate 0
        assert cats.get(0) == "wrong_algorithm"

    def test_empty_response(self):
        cats = parse_failure_categories("", 2)
        assert cats == {}

    def test_case_insensitive(self):
        response = "SOLUTION 1: WRONG_ALGORITHM"
        cats = parse_failure_categories(response, 1)
        assert cats.get(0) == "wrong_algorithm"


# ---------------------------------------------------------------------------
# Test: parse_violated_constraints
# ---------------------------------------------------------------------------

class TestParseViolatedConstraints:

    def test_standard_format(self):
        response = "VIOLATED:\n- Must handle empty input\n- Must return sorted list\nCOMMON:"
        violated = parse_violated_constraints(response)
        assert len(violated) == 2
        assert "Must handle empty input" in violated[0]

    def test_no_violated_section(self):
        """Response without any text matching 'VIOLATED' (case-insensitive) returns empty."""
        response = "This response mentions no relevant keywords."
        violated = parse_violated_constraints(response)
        assert violated == []

    def test_short_lines_skipped(self):
        """Lines shorter than 5 chars are filtered."""
        response = "VIOLATED:\n- Yes\n- This is a real constraint\nCOMMON:"
        violated = parse_violated_constraints(response)
        assert len(violated) == 1
        assert "real constraint" in violated[0]

    def test_strips_bullet_markers(self):
        response = "VIOLATED:\n* Constraint one is important\n- Constraint two matters\nCOMMON:"
        violated = parse_violated_constraints(response)
        for v in violated:
            assert not v.startswith("-")
            assert not v.startswith("*")


# ---------------------------------------------------------------------------
# Test: parse_common_pattern
# ---------------------------------------------------------------------------

class TestParseCommonPattern:

    def test_standard_format(self):
        response = (
            "COMMON:\n"
            "All solutions assume the input is sorted.\n"
            "They fail on unsorted arrays.\n"
            "NEW_CONSTRAINTS:"
        )
        pattern = parse_common_pattern(response)
        assert "sorted" in pattern
        assert len(pattern) > 0

    def test_no_common_section(self):
        """Response without any text matching 'COMMON' (case-insensitive) returns empty."""
        response = "Just some analysis with no relevant keywords here."
        pattern = parse_common_pattern(response)
        assert pattern == ""

    def test_multiline_joined(self):
        response = "COMMON:\nLine one.\nLine two.\nLine three.\nNEW_CONSTRAINTS:"
        pattern = parse_common_pattern(response)
        assert "Line one" in pattern
        assert "Line two" in pattern
        assert "Line three" in pattern

    def test_empty_common_section(self):
        response = "COMMON:\n\nNEW_CONSTRAINTS:"
        pattern = parse_common_pattern(response)
        assert pattern == ""


# ---------------------------------------------------------------------------
# Test: parse_new_constraints
# ---------------------------------------------------------------------------

class TestParseNewConstraints:

    def test_standard_format(self):
        response = "NEW_CONSTRAINTS:\n- Validate input range is non-negative\n- Check for empty array"
        constraints = parse_new_constraints(response)
        assert len(constraints) >= 1
        assert any("non-negative" in c for c in constraints)

    def test_numbered_format(self):
        response = "NEW_CONSTRAINTS:\n1. Always validate boundary conditions\n2. Handle the zero case explicitly"
        constraints = parse_new_constraints(response)
        assert len(constraints) == 2

    def test_no_section(self):
        response = "Some analysis without new constraints section."
        constraints = parse_new_constraints(response)
        assert constraints == []

    def test_short_lines_skipped(self):
        """Lines shorter than 10 chars are filtered."""
        response = "NEW_CONSTRAINTS:\n- Short\n- This is a sufficiently long constraint"
        constraints = parse_new_constraints(response)
        assert len(constraints) == 1
        assert "sufficiently" in constraints[0]

    def test_plural_section_header(self):
        """Parser handles both NEW_CONSTRAINT and NEW_CONSTRAINTS."""
        response = "NEW_CONSTRAINT:\n- Validate all edge cases before processing"
        constraints = parse_new_constraints(response)
        assert len(constraints) == 1


# ---------------------------------------------------------------------------
# Test: format_candidates_with_errors
# ---------------------------------------------------------------------------

class TestFormatCandidatesWithErrors:

    def test_formats_single_candidate(self):
        candidates = [FailingCandidate(
            code="def f(): pass", error_output="TypeError", index=0,
        )]
        text = format_candidates_with_errors(candidates)
        assert "Solution 1:" in text
        assert "def f(): pass" in text
        assert "TypeError" in text

    def test_formats_multiple_candidates(self):
        candidates = [
            FailingCandidate(code="code_a", error_output="error_a", index=0),
            FailingCandidate(code="code_b", error_output="error_b", index=1),
        ]
        text = format_candidates_with_errors(candidates)
        assert "Solution 1:" in text
        assert "Solution 2:" in text
        assert "code_a" in text
        assert "code_b" in text

    def test_empty_candidates(self):
        text = format_candidates_with_errors([])
        assert text == ""

    def test_code_in_python_block(self):
        candidates = [FailingCandidate(
            code="def f(): pass", error_output="err", index=0,
        )]
        text = format_candidates_with_errors(candidates)
        assert "```python" in text
        assert "```" in text


# ---------------------------------------------------------------------------
# Test: FailureAnalyzer.analyze (enabled, with mock LLM)
# ---------------------------------------------------------------------------

class TestAnalyzeEnabled:

    def test_basic_analysis(self, fa_enabled, sample_candidates, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = fa_enabled.analyze(
            problem="Find the sum of two numbers",
            candidates=sample_candidates,
            original_constraints=["Must handle negative numbers"],
            llm_call=mock_llm,
            task_id="t1",
        )
        assert isinstance(result, FailureAnalysis)
        assert result.raw_analysis == structured_llm_response
        assert result.analysis_time_ms > 0

    def test_categories_parsed(self, fa_enabled, sample_candidates, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        assert len(result.categories) > 0

    def test_violated_constraints_parsed(self, fa_enabled, sample_candidates, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        assert len(result.violated_constraints) > 0

    def test_common_pattern_parsed(self, fa_enabled, sample_candidates, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        assert len(result.common_pattern) > 0

    def test_new_constraints_parsed(self, fa_enabled, sample_candidates, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        assert len(result.new_constraints) > 0

    def test_llm_called_once(self, fa_enabled, sample_candidates):
        mock_llm = MockLLM("Some analysis")
        fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        assert len(mock_llm.calls) == 1

    def test_llm_receives_correct_params(self, fa_enabled, sample_candidates):
        mock_llm = MockLLM("analysis")
        fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        call = mock_llm.calls[0]
        assert call["temperature"] == 0.3
        assert call["max_tokens"] == 2048
        assert call["seed"] == 42

    def test_empty_candidates_returns_empty(self, fa_enabled):
        result = fa_enabled.analyze(
            problem="test", candidates=[],
            original_constraints=[], llm_call=MockLLM("ignored"),
        )
        assert result.categories == {}
        assert result.raw_analysis == ""

    def test_no_llm_still_runs(self, fa_enabled, sample_candidates):
        """Without llm_call, analysis still returns a result."""
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[],
        )
        assert isinstance(result, FailureAnalysis)
        assert result.raw_analysis == ""

    def test_max_candidates_to_analyze(self, tmp_telemetry):
        cfg = FailureAnalysisConfig(enabled=True, max_candidates_to_analyze=2)
        fa = FailureAnalyzer(cfg, telemetry_dir=tmp_telemetry)
        many_candidates = [
            FailingCandidate(code=f"code_{i}", error_output=f"err_{i}", index=i)
            for i in range(10)
        ]
        mock_llm = MockLLM("analysis")
        fa.analyze(
            problem="test", candidates=many_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        # Only 2 candidates should appear in the prompt
        prompt = mock_llm.calls[0]["prompt"]
        assert "code_0" in prompt
        assert "code_1" in prompt
        assert "code_2" not in prompt

    def test_embeddings_computed(self, fa_enabled, sample_candidates):
        mock_embed = MockEmbed(dim=10)
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], embed_call=mock_embed,
        )
        assert len(result.failure_embeddings) == 2
        assert len(result.failure_embeddings[0]) == 10

    def test_embed_failure_does_not_crash(self, fa_enabled, sample_candidates):
        mock_embed = MockEmbedFailing()
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], embed_call=mock_embed,
        )
        assert result.failure_embeddings == []


# ---------------------------------------------------------------------------
# Test: AC-3A-1 — Categorizes failure types
# ---------------------------------------------------------------------------

class TestAC3A1FailureCategories:
    """AC-3A-1: System categorizes failure types into 6 categories."""

    def test_all_categories_recognizable(self, fa_enabled):
        """Parser should recognize every valid category."""
        for i, cat in enumerate(FAILURE_CATEGORIES.keys(), 1):
            response = f"Solution 1: {cat}"
            cats = parse_failure_categories(response, 1)
            assert cats.get(0) == cat, f"Failed to parse category '{cat}'"

    def test_categories_assigned_to_candidates(self, fa_enabled, sample_candidates, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        for idx, cat in result.categories.items():
            assert cat in FAILURE_CATEGORIES, f"Unknown category: {cat}"


# ---------------------------------------------------------------------------
# Test: AC-3A-2 — Identifies violated constraints
# ---------------------------------------------------------------------------

class TestAC3A2ViolatedConstraints:
    """AC-3A-2: System identifies which constraints were violated."""

    def test_violated_constraints_extracted(self, fa_enabled, sample_candidates, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=["Must handle negative numbers"],
            llm_call=mock_llm,
        )
        assert len(result.violated_constraints) > 0

    def test_violated_constraints_are_strings(self, fa_enabled, sample_candidates, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        for vc in result.violated_constraints:
            assert isinstance(vc, str)
            assert len(vc) > 5


# ---------------------------------------------------------------------------
# Test: AC-3A-3 — Finds common failure pattern
# ---------------------------------------------------------------------------

class TestAC3A3CommonPattern:
    """AC-3A-3: System identifies common patterns across failures."""

    def test_common_pattern_extracted(self, fa_enabled, sample_candidates, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        assert len(result.common_pattern) > 0


# ---------------------------------------------------------------------------
# Test: AC-3A-4 — Generates new constraints
# ---------------------------------------------------------------------------

class TestAC3A4NewConstraints:
    """AC-3A-4: System generates new constraints to prevent failures."""

    def test_new_constraints_generated(self, fa_enabled, sample_candidates, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        assert len(result.new_constraints) > 0

    def test_new_constraints_are_meaningful(self, fa_enabled, sample_candidates, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        result = fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        for nc in result.new_constraints:
            assert isinstance(nc, str)
            assert len(nc) > 10


# ---------------------------------------------------------------------------
# Test: Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_event_written_to_jsonl(self, fa_enabled, sample_candidates, tmp_telemetry, structured_llm_response):
        mock_llm = MockLLM(structured_llm_response)
        fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm, task_id="LCB_001",
        )
        events_file = tmp_telemetry / "failure_analysis_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "LCB_001"
        assert data["num_candidates"] == 2
        assert "categories_found" in data
        assert "analysis_time_ms" in data
        assert "timestamp" in data

    def test_multiple_events_appended(self, fa_enabled, sample_candidates, tmp_telemetry):
        mock_llm = MockLLM("analysis")
        for i in range(3):
            fa_enabled.analyze(
                problem="test", candidates=sample_candidates,
                original_constraints=[], llm_call=mock_llm, task_id=f"t{i}",
            )
        events_file = tmp_telemetry / "failure_analysis_events.jsonl"
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_no_telemetry_without_task_id(self, fa_enabled, sample_candidates, tmp_telemetry):
        mock_llm = MockLLM("analysis")
        fa_enabled.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm,
        )
        events_file = tmp_telemetry / "failure_analysis_events.jsonl"
        assert not events_file.exists()

    def test_no_crash_without_telemetry_dir(self, sample_candidates):
        cfg = FailureAnalysisConfig(enabled=True)
        fa = FailureAnalyzer(cfg, telemetry_dir=None)
        mock_llm = MockLLM("analysis")
        result = fa.analyze(
            problem="test", candidates=sample_candidates,
            original_constraints=[], llm_call=mock_llm, task_id="t1",
        )
        assert isinstance(result, FailureAnalysis)


# ---------------------------------------------------------------------------
# Test: Data structures
# ---------------------------------------------------------------------------

class TestDataStructures:

    def test_failing_candidate_to_dict(self):
        c = FailingCandidate(
            code="def solve(): return 42",
            error_output="AssertionError: got 42 expected 43",
            index=3,
        )
        d = c.to_dict()
        assert d["index"] == 3
        assert d["code_length"] == len("def solve(): return 42")
        assert "AssertionError" in d["error_preview"]

    def test_failing_candidate_error_preview_truncated(self):
        c = FailingCandidate(code="x", error_output="E" * 500, index=0)
        d = c.to_dict()
        assert len(d["error_preview"]) == 200

    def test_failure_analysis_to_dict(self):
        fa = FailureAnalysis(
            categories={0: "wrong_algorithm", 1: "edge_case_miss"},
            violated_constraints=["Must sort", "Must handle empty"],
            common_pattern="All assume sorted input",
            new_constraints=["Validate input order"],
            failure_embeddings=[[1.0, 2.0], [3.0, 4.0]],
            analysis_time_ms=150.0,
        )
        d = fa.to_dict()
        assert d["categories"] == {0: "wrong_algorithm", 1: "edge_case_miss"}
        assert len(d["violated_constraints"]) == 2
        assert d["common_pattern"] == "All assume sorted input"
        assert len(d["new_constraints"]) == 1
        assert d["num_embeddings"] == 2
        assert d["analysis_time_ms"] == 150.0

    def test_event_to_dict(self):
        e = FailureAnalysisEvent(
            task_id="t1", num_candidates=3,
            categories_found=["wrong_algorithm", "edge_case_miss"],
            num_violated_constraints=2, num_new_constraints=1,
            analysis_time_ms=100.0,
        )
        d = e.to_dict()
        assert d["task_id"] == "t1"
        assert d["num_candidates"] == 3
        assert d["categories_found"] == ["wrong_algorithm", "edge_case_miss"]
        assert "timestamp" in d

    def test_config_defaults(self):
        cfg = FailureAnalysisConfig()
        assert cfg.enabled is False
        assert cfg.max_candidates_to_analyze == 5
        assert cfg.analysis_temperature == 0.3
        assert cfg.analysis_max_tokens == 2048

    def test_get_category_descriptions(self, fa_enabled):
        descs = fa_enabled.get_category_descriptions()
        assert descs == FAILURE_CATEGORIES
        assert len(descs) == 6
