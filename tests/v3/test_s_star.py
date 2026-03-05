"""Tests for V3 S* Tiebreaking (Feature 2C) — Distinguishing Input Generation."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from benchmark.v3.s_star import (
    DISTINGUISHING_INPUT_PROMPT,
    CandidateScore,
    SStar,
    SStarConfig,
    SStarEvent,
    TiebreakResult,
    get_top2_by_energy,
    parse_distinguishing_inputs,
    score_candidates_on_inputs,
    should_tiebreak,
)


# ---------------------------------------------------------------------------
# Mock callables
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns predefined responses."""

    def __init__(self, response: str = "INPUT: 0\nINPUT: -1\nINPUT: 1000000"):
        self.response = response
        self.calls: list = []

    def __call__(self, prompt: str, temperature: float,
                 max_tokens: int, seed: Optional[int]) -> Tuple[str, int, float]:
        self.calls.append({
            "prompt": prompt, "temperature": temperature,
            "max_tokens": max_tokens, "seed": seed,
        })
        return self.response, 50, 100.0


class MockSandbox:
    """Mock sandbox that returns predefined pass/fail results."""

    def __init__(self, results: Optional[dict] = None):
        # results maps (code_snippet, input) to (passed, stdout, stderr)
        self._results = results or {}
        self.calls: list = []
        self.default_pass = True

    def __call__(self, code: str, test_input: str) -> Tuple[bool, str, str]:
        self.calls.append({"code": code, "input": test_input})
        key = (code, test_input)
        if key in self._results:
            return self._results[key]
        return self.default_pass, "", ""


def make_mock_sandbox_a_wins():
    """Create sandbox where candidate A always passes, B sometimes fails."""
    sandbox = MockSandbox()
    sandbox._results = {}

    def call(code: str, test_input: str) -> Tuple[bool, str, str]:
        sandbox.calls.append({"code": code, "input": test_input})
        if "solution_a" in code:
            return True, "correct", ""
        elif "solution_b" in code:
            if test_input == "-1":
                return False, "", "error"
            return True, "correct", ""
        return True, "", ""

    return call, sandbox


def make_mock_sandbox_b_wins():
    """Create sandbox where candidate B wins."""
    def call(code: str, test_input: str) -> Tuple[bool, str, str]:
        if "solution_a" in code:
            if test_input == "-1":
                return False, "", "error"
            return True, "correct", ""
        return True, "correct", ""
    return call


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_telemetry(tmp_path):
    d = tmp_path / "telemetry"
    d.mkdir()
    return d


@pytest.fixture
def sstar_enabled(tmp_telemetry):
    cfg = SStarConfig(enabled=True, energy_delta=1.0)
    return SStar(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def sstar_disabled(tmp_telemetry):
    cfg = SStarConfig(enabled=False)
    return SStar(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def close_candidates():
    """Two candidates with close energy scores (within delta)."""
    return [
        CandidateScore(code="def solution_a(): pass", raw_energy=5.0, index=0),
        CandidateScore(code="def solution_b(): pass", raw_energy=5.5, index=1),
    ]


@pytest.fixture
def far_candidates():
    """Two candidates with far energy scores (outside delta)."""
    return [
        CandidateScore(code="def solution_a(): pass", raw_energy=5.0, index=0),
        CandidateScore(code="def solution_b(): pass", raw_energy=10.0, index=1),
    ]


# ---------------------------------------------------------------------------
# Test: Disabled noop
# ---------------------------------------------------------------------------

class TestDisabledNoop:

    def test_returns_lowest_energy(self, sstar_disabled, close_candidates):
        result = sstar_disabled.tiebreak(
            close_candidates, "problem", task_id="t1"
        )
        assert result.triggered is False
        assert result.winner_index == 0  # Lower energy
        assert result.reason == "disabled"

    def test_no_telemetry_when_disabled(self, sstar_disabled, close_candidates, tmp_telemetry):
        sstar_disabled.tiebreak(close_candidates, "problem")
        events_file = tmp_telemetry / "s_star_events.jsonl"
        assert not events_file.exists()


# ---------------------------------------------------------------------------
# Test: get_top2_by_energy
# ---------------------------------------------------------------------------

class TestGetTop2:

    def test_returns_two_lowest(self):
        candidates = [
            CandidateScore(code="c", raw_energy=10.0, index=0),
            CandidateScore(code="b", raw_energy=5.0, index=1),
            CandidateScore(code="a", raw_energy=3.0, index=2),
        ]
        top2 = get_top2_by_energy(candidates)
        assert len(top2) == 2
        assert top2[0].raw_energy == 3.0
        assert top2[1].raw_energy == 5.0

    def test_single_candidate(self):
        candidates = [CandidateScore(code="a", raw_energy=5.0, index=0)]
        top2 = get_top2_by_energy(candidates)
        assert len(top2) == 1

    def test_empty_list(self):
        top2 = get_top2_by_energy([])
        assert len(top2) == 0

    def test_preserves_index(self):
        candidates = [
            CandidateScore(code="b", raw_energy=8.0, index=1),
            CandidateScore(code="a", raw_energy=3.0, index=0),
        ]
        top2 = get_top2_by_energy(candidates)
        assert top2[0].index == 0
        assert top2[1].index == 1


# ---------------------------------------------------------------------------
# Test: should_tiebreak
# ---------------------------------------------------------------------------

class TestShouldTiebreak:

    def test_within_delta(self):
        a = CandidateScore(code="", raw_energy=5.0, index=0)
        b = CandidateScore(code="", raw_energy=5.5, index=1)
        assert should_tiebreak(a, b, delta=1.0) is True

    def test_outside_delta(self):
        a = CandidateScore(code="", raw_energy=5.0, index=0)
        b = CandidateScore(code="", raw_energy=7.0, index=1)
        assert should_tiebreak(a, b, delta=1.0) is False

    def test_exact_delta(self):
        a = CandidateScore(code="", raw_energy=5.0, index=0)
        b = CandidateScore(code="", raw_energy=6.0, index=1)
        assert should_tiebreak(a, b, delta=1.0) is True

    def test_equal_energies(self):
        a = CandidateScore(code="", raw_energy=5.0, index=0)
        b = CandidateScore(code="", raw_energy=5.0, index=1)
        assert should_tiebreak(a, b, delta=1.0) is True

    def test_custom_delta(self):
        a = CandidateScore(code="", raw_energy=5.0, index=0)
        b = CandidateScore(code="", raw_energy=7.5, index=1)
        assert should_tiebreak(a, b, delta=3.0) is True
        assert should_tiebreak(a, b, delta=2.0) is False


# ---------------------------------------------------------------------------
# Test: parse_distinguishing_inputs
# ---------------------------------------------------------------------------

class TestParseDistinguishingInputs:

    def test_structured_format(self):
        response = "INPUT: 0\nINPUT: -1\nINPUT: 999"
        inputs = parse_distinguishing_inputs(response)
        assert inputs == ["0", "-1", "999"]

    def test_numbered_fallback(self):
        response = "1. [1, 2, 3]\n2. []\n3. [-1, 0, 1]"
        inputs = parse_distinguishing_inputs(response)
        assert len(inputs) == 3

    def test_bare_lines_fallback(self):
        response = "[1, 2, 3]\n[]\n[-1, 0, 1]"
        inputs = parse_distinguishing_inputs(response)
        assert len(inputs) == 3

    def test_max_inputs_limit(self):
        response = "\n".join(f"INPUT: {i}" for i in range(10))
        inputs = parse_distinguishing_inputs(response, max_inputs=5)
        assert len(inputs) == 5

    def test_empty_response(self):
        inputs = parse_distinguishing_inputs("")
        assert inputs == []

    def test_ignores_code_blocks(self):
        response = "INPUT: 5\n```python\nprint(5)\n```\nINPUT: 10"
        inputs = parse_distinguishing_inputs(response)
        assert inputs == ["5", "10"]

    def test_ignores_comments(self):
        response = "INPUT: 5\n# This is a comment\nINPUT: 10"
        inputs = parse_distinguishing_inputs(response)
        assert inputs == ["5", "10"]

    def test_mixed_case_input(self):
        response = "input: 5\nInput: 10\nINPUT: 15"
        inputs = parse_distinguishing_inputs(response)
        assert len(inputs) == 3

    def test_empty_input_values_skipped(self):
        response = "INPUT: \nINPUT: 5\nINPUT: "
        inputs = parse_distinguishing_inputs(response)
        assert inputs == ["5"]


# ---------------------------------------------------------------------------
# Test: score_candidates_on_inputs
# ---------------------------------------------------------------------------

class TestScoreCandidates:

    def test_a_wins(self):
        def sandbox(code, inp):
            if code.startswith("GOOD"):
                return True, "", ""
            return False, "", "error"
        score_a, score_b = score_candidates_on_inputs(
            "GOOD code", "BAD code", ["1", "2", "3"], sandbox
        )
        assert score_a == 3
        assert score_b == 0

    def test_tie(self):
        def sandbox(code, inp):
            return True, "", ""
        score_a, score_b = score_candidates_on_inputs(
            "def a(): pass", "def b(): pass", ["1", "2"], sandbox
        )
        assert score_a == 2
        assert score_b == 2

    def test_empty_inputs(self):
        def sandbox(code, inp):
            return True, "", ""
        score_a, score_b = score_candidates_on_inputs(
            "def a(): pass", "def b(): pass", [], sandbox
        )
        assert score_a == 0
        assert score_b == 0

    def test_b_wins(self):
        def sandbox(code, inp):
            if "b" in code:
                return True, "", ""
            return False, "", "error"
        score_a, score_b = score_candidates_on_inputs(
            "def a(): pass", "def b(): pass", ["1", "2"], sandbox
        )
        assert score_a == 0
        assert score_b == 2


# ---------------------------------------------------------------------------
# Test: AC-2C-1 — Tiebreak triggers on >= 15% of tasks (structural)
# ---------------------------------------------------------------------------

class TestAC2C1TriggerRate:

    def test_close_energies_trigger_tiebreak(self, sstar_enabled):
        """When candidates have similar energies, tiebreak should trigger."""
        mock_llm = MockLLM("INPUT: 0\nINPUT: -1\nINPUT: 100")
        mock_sandbox = MockSandbox()

        candidates = [
            CandidateScore(code="def a(): pass", raw_energy=5.0, index=0),
            CandidateScore(code="def b(): pass", raw_energy=5.8, index=1),
        ]
        result = sstar_enabled.tiebreak(
            candidates, "Two sum problem", mock_llm, mock_sandbox
        )
        assert result.triggered is True

    def test_far_energies_skip_tiebreak(self, sstar_enabled):
        """When candidates have very different energies, no tiebreak."""
        candidates = [
            CandidateScore(code="def a(): pass", raw_energy=3.0, index=0),
            CandidateScore(code="def b(): pass", raw_energy=12.0, index=1),
        ]
        result = sstar_enabled.tiebreak(candidates, "problem")
        assert result.triggered is False
        assert result.reason == "clear_winner"

    def test_tiebreak_rate_on_mixed_tasks(self, sstar_enabled):
        """Simulate mix of tasks — some close, some far energies."""
        mock_llm = MockLLM("INPUT: test")
        mock_sandbox = MockSandbox()

        task_energies = [
            (5.0, 5.3),   # close → tiebreak
            (3.0, 12.0),  # far → no tiebreak
            (7.0, 7.5),   # close → tiebreak
            (4.0, 4.9),   # close → tiebreak
            (2.0, 15.0),  # far → no tiebreak
            (6.0, 6.2),   # close → tiebreak
            (1.0, 8.0),   # far → no tiebreak
            (8.0, 8.1),   # close → tiebreak
            (9.0, 9.8),   # close → tiebreak
            (3.0, 20.0),  # far → no tiebreak
        ]

        triggered_count = 0
        for e_a, e_b in task_energies:
            candidates = [
                CandidateScore(code="a", raw_energy=e_a, index=0),
                CandidateScore(code="b", raw_energy=e_b, index=1),
            ]
            result = sstar_enabled.tiebreak(
                candidates, "problem", mock_llm, mock_sandbox
            )
            if result.triggered:
                triggered_count += 1

        # At least 15% of tasks should trigger
        assert triggered_count >= 2, f"Only {triggered_count}/10 triggered tiebreak"


# ---------------------------------------------------------------------------
# Test: AC-2C-2 — Correct tiebreaks >= 70% (structural with mock)
# ---------------------------------------------------------------------------

class TestAC2C2CorrectTiebreaks:

    def test_better_candidate_wins(self, sstar_enabled):
        """When sandbox distinguishes candidates, the better one wins."""
        mock_llm = MockLLM("INPUT: 0\nINPUT: -1\nINPUT: 100")

        # A passes all, B fails on -1
        def sandbox(code, inp):
            if "solution_b" in code and inp == "-1":
                return False, "", "error"
            return True, "ok", ""

        candidates = [
            CandidateScore(code="def solution_a(): pass", raw_energy=5.0, index=0),
            CandidateScore(code="def solution_b(): pass", raw_energy=5.5, index=1),
        ]
        result = sstar_enabled.tiebreak(
            candidates, "problem", mock_llm, sandbox
        )
        assert result.triggered is True
        assert result.winner_index == 0  # A has higher score
        assert result.scores[0] > result.scores[1]

    def test_underdog_can_win(self, sstar_enabled):
        """Candidate with slightly higher energy can win via tiebreak."""
        mock_llm = MockLLM("INPUT: 0\nINPUT: -1\nINPUT: 100")

        # B passes all, A fails on -1
        def sandbox(code, inp):
            if "solution_a" in code and inp == "-1":
                return False, "", "error"
            return True, "ok", ""

        candidates = [
            CandidateScore(code="def solution_a(): pass", raw_energy=5.0, index=0),
            CandidateScore(code="def solution_b(): pass", raw_energy=5.5, index=1),
        ]
        result = sstar_enabled.tiebreak(
            candidates, "problem", mock_llm, sandbox
        )
        assert result.triggered is True
        assert result.winner_index == 1  # B wins despite higher energy

    def test_tie_goes_to_lower_energy(self, sstar_enabled):
        """When scores are tied, lower energy candidate wins."""
        mock_llm = MockLLM("INPUT: 0\nINPUT: -1")

        def sandbox(code, inp):
            return True, "ok", ""

        candidates = [
            CandidateScore(code="def solution_a(): pass", raw_energy=5.0, index=0),
            CandidateScore(code="def solution_b(): pass", raw_energy=5.5, index=1),
        ]
        result = sstar_enabled.tiebreak(
            candidates, "problem", mock_llm, sandbox
        )
        assert result.triggered is True
        assert result.winner_index == 0  # Tie goes to lower energy


# ---------------------------------------------------------------------------
# Test: AC-2C-3 — Lens accuracy improvement (structural)
# ---------------------------------------------------------------------------

class TestAC2C3Accuracy:

    def test_tiebreak_can_override_lens(self, sstar_enabled):
        """S* can override Lens-only selection when sandbox reveals issues."""
        mock_llm = MockLLM("INPUT: edge_case")

        # Lens says A is better (lower energy), but B passes more tests
        def sandbox(code, inp):
            if "solution_a" in code:
                return False, "", "edge case failure"
            return True, "ok", ""

        candidates = [
            CandidateScore(code="def solution_a(): pass", raw_energy=5.0, index=0),
            CandidateScore(code="def solution_b(): pass", raw_energy=5.8, index=1),
        ]
        result = sstar_enabled.tiebreak(
            candidates, "problem", mock_llm, sandbox
        )
        # S* should override Lens selection
        assert result.winner_index == 1


# ---------------------------------------------------------------------------
# Test: AC-2C-4 — Time < 5s per tiebreak (structural)
# ---------------------------------------------------------------------------

class TestAC2C4TimeLimit:

    def test_tiebreak_is_fast(self, sstar_enabled):
        """Tiebreak with mock callables should be very fast."""
        mock_llm = MockLLM("INPUT: 0\nINPUT: -1\nINPUT: 100")
        mock_sandbox = MockSandbox()

        candidates = [
            CandidateScore(code="a", raw_energy=5.0, index=0),
            CandidateScore(code="b", raw_energy=5.5, index=1),
        ]
        result = sstar_enabled.tiebreak(
            candidates, "problem", mock_llm, mock_sandbox
        )
        assert result.time_ms < 5000  # < 5 seconds


# ---------------------------------------------------------------------------
# Test: SStar class
# ---------------------------------------------------------------------------

class TestSStarClass:

    def test_single_candidate(self, sstar_enabled):
        """Single candidate should return without tiebreak."""
        candidates = [CandidateScore(code="a", raw_energy=5.0, index=0)]
        result = sstar_enabled.tiebreak(candidates, "problem")
        assert result.triggered is False
        assert result.winner_index == 0
        assert result.reason == "insufficient_candidates"

    def test_empty_candidates(self, sstar_enabled):
        result = sstar_enabled.tiebreak([], "problem")
        assert result.triggered is False
        assert result.winner_index == -1

    def test_missing_callables(self, sstar_enabled, close_candidates):
        """Should fall back to Lens without callables."""
        result = sstar_enabled.tiebreak(close_candidates, "problem")
        assert result.triggered is False
        assert result.reason == "missing_callables"

    def test_no_inputs_generated(self, sstar_enabled, close_candidates):
        """If LLM generates no parseable inputs, fall back to Lens."""
        mock_llm = MockLLM("")  # Empty response → no inputs
        mock_sandbox = MockSandbox()
        result = sstar_enabled.tiebreak(
            close_candidates, "problem", mock_llm, mock_sandbox, task_id="t1"
        )
        assert result.triggered is True
        assert result.reason == "no_inputs_generated"
        assert result.winner_index == 0  # Falls back to lower energy

    def test_custom_delta(self, tmp_telemetry):
        cfg = SStarConfig(enabled=True, energy_delta=0.5)
        sstar = SStar(cfg, telemetry_dir=tmp_telemetry)

        # Delta 0.6 > 0.5: close enough for tiebreak
        candidates = [
            CandidateScore(code="a", raw_energy=5.0, index=0),
            CandidateScore(code="b", raw_energy=5.6, index=1),
        ]
        mock_llm = MockLLM("INPUT: 0")
        mock_sandbox = MockSandbox()
        result = sstar.tiebreak(candidates, "problem", mock_llm, mock_sandbox)
        assert result.triggered is False  # 0.6 > 0.5

    def test_no_crash_without_telemetry_dir(self, close_candidates):
        cfg = SStarConfig(enabled=True)
        sstar = SStar(cfg, telemetry_dir=None)
        mock_llm = MockLLM("INPUT: 0")
        mock_sandbox = MockSandbox()
        result = sstar.tiebreak(close_candidates, "problem",
                               mock_llm, mock_sandbox, task_id="t1")
        assert isinstance(result, TiebreakResult)

    def test_three_candidates(self, sstar_enabled):
        """With 3+ candidates, only top-2 are considered."""
        mock_llm = MockLLM("INPUT: 0")
        mock_sandbox = MockSandbox()
        candidates = [
            CandidateScore(code="c", raw_energy=10.0, index=2),
            CandidateScore(code="a", raw_energy=5.0, index=0),
            CandidateScore(code="b", raw_energy=5.5, index=1),
        ]
        result = sstar_enabled.tiebreak(
            candidates, "problem", mock_llm, mock_sandbox
        )
        assert result.triggered is True
        assert result.winner_index in [0, 1]  # Third candidate ignored


# ---------------------------------------------------------------------------
# Test: Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_event_logged_on_tiebreak(self, sstar_enabled, tmp_telemetry):
        mock_llm = MockLLM("INPUT: 0\nINPUT: -1")
        mock_sandbox = MockSandbox()
        candidates = [
            CandidateScore(code="a", raw_energy=5.0, index=0),
            CandidateScore(code="b", raw_energy=5.5, index=1),
        ]
        sstar_enabled.tiebreak(
            candidates, "problem", mock_llm, mock_sandbox, task_id="LCB_001"
        )
        events_file = tmp_telemetry / "s_star_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "LCB_001"
        assert data["triggered"] is True
        assert "energy_delta" in data
        assert "scores_a" in data
        assert "scores_b" in data
        assert "timestamp" in data

    def test_event_logged_on_clear_winner(self, sstar_enabled, tmp_telemetry, far_candidates):
        sstar_enabled.tiebreak(far_candidates, "problem", task_id="t2")
        events_file = tmp_telemetry / "s_star_events.jsonl"
        assert events_file.exists()
        data = json.loads(events_file.read_text().strip())
        assert data["triggered"] is False
        assert data["reason"] == "clear_winner"

    def test_no_telemetry_without_task_id(self, sstar_enabled, tmp_telemetry, close_candidates):
        mock_llm = MockLLM("INPUT: 0")
        mock_sandbox = MockSandbox()
        sstar_enabled.tiebreak(close_candidates, "problem", mock_llm, mock_sandbox)
        events_file = tmp_telemetry / "s_star_events.jsonl"
        assert not events_file.exists()


# ---------------------------------------------------------------------------
# Test: Data structures
# ---------------------------------------------------------------------------

class TestDataStructures:

    def test_candidate_to_dict(self):
        c = CandidateScore(code="def f(): return 42", raw_energy=5.0, index=0)
        d = c.to_dict()
        assert d["index"] == 0
        assert d["raw_energy"] == 5.0
        assert d["code_length"] == len("def f(): return 42")

    def test_tiebreak_result_to_dict(self):
        r = TiebreakResult(
            triggered=True, winner_index=1, scores=[2, 3],
            num_inputs=3, time_ms=500.0, reason="tiebreak_complete",
        )
        d = r.to_dict()
        assert d["triggered"] is True
        assert d["scores"] == [2, 3]

    def test_event_to_dict(self):
        e = SStarEvent(
            task_id="t1", triggered=True, energy_delta=0.5,
            candidate_a_energy=5.0, candidate_b_energy=5.5,
            winner_index=0, num_inputs=3, scores_a=3, scores_b=1,
            time_ms=100.0, reason="tiebreak_complete",
        )
        d = e.to_dict()
        assert d["task_id"] == "t1"
        assert d["triggered"] is True
        assert "timestamp" in d

    def test_config_defaults(self):
        cfg = SStarConfig()
        assert cfg.enabled is False
        assert cfg.energy_delta == 1.0
        assert cfg.max_distinguishing_inputs == 5
        assert cfg.num_inputs_to_generate == 3

    def test_prompt_template(self):
        prompt = DISTINGUISHING_INPUT_PROMPT.format(
            problem="Two sum", candidate_a="def a(): pass",
            candidate_b="def b(): pass", n=3,
        )
        assert "Two sum" in prompt
        assert "Solution A" in prompt
        assert "Solution B" in prompt
        assert "edge-case" in prompt
