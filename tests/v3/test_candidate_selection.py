"""Tests for V3 candidate selection strategies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.v3.candidate_selection import (
    CandidateInfo,
    select_lens,
    select_random,
    select_logprob,
    select_oracle,
    select_candidate,
)


def _make_candidates():
    """Create a test set of candidates with varying properties."""
    return [
        CandidateInfo(index=0, code="def f(): return 1", energy=5.0, passed=True,
                       logprobs=[-0.1, -0.2, -0.3]),
        CandidateInfo(index=1, code="def f(): return 2", energy=3.0, passed=True,
                       logprobs=[-0.5, -0.6, -0.7]),
        CandidateInfo(index=2, code="def f(): return 3", energy=8.0, passed=False,
                       logprobs=[-0.01, -0.02]),
        CandidateInfo(index=3, code="def f(): return 4", energy=12.0, passed=True,
                       logprobs=[-1.0, -1.5, -2.0]),
    ]


class TestSelectLens:
    def test_selects_lowest_energy_passing(self):
        candidates = _make_candidates()
        winner = select_lens(candidates)
        assert winner is not None
        assert winner.index == 1  # energy=3.0, lowest among passing

    def test_ignores_failing(self):
        candidates = _make_candidates()
        # Candidate 2 has energy=8.0 but is failing
        winner = select_lens(candidates)
        assert winner.index != 2

    def test_no_passing_returns_none(self):
        candidates = [CandidateInfo(index=0, code="x", energy=1.0, passed=False)]
        assert select_lens(candidates) is None

    def test_empty_returns_none(self):
        assert select_lens([]) is None


class TestSelectRandom:
    def test_returns_passing_candidate(self):
        candidates = _make_candidates()
        for seed in range(10):
            winner = select_random(candidates, seed=seed)
            assert winner is not None
            assert winner.passed is True

    def test_deterministic_with_seed(self):
        candidates = _make_candidates()
        w1 = select_random(candidates, seed=42)
        w2 = select_random(candidates, seed=42)
        assert w1.index == w2.index

    def test_no_passing_returns_none(self):
        candidates = [CandidateInfo(index=0, code="x", energy=1.0, passed=False)]
        assert select_random(candidates, seed=42) is None


class TestSelectLogprob:
    def test_selects_highest_mean_logprob(self):
        candidates = _make_candidates()
        winner = select_logprob(candidates)
        assert winner is not None
        # Candidate 0: mean = (-0.1-0.2-0.3)/3 = -0.2
        # Candidate 1: mean = (-0.5-0.6-0.7)/3 = -0.6
        # Candidate 3: mean = (-1.0-1.5-2.0)/3 = -1.5
        assert winner.index == 0  # highest mean logprob

    def test_fallback_to_lens_without_logprobs(self):
        candidates = [
            CandidateInfo(index=0, code="x", energy=5.0, passed=True, logprobs=None),
            CandidateInfo(index=1, code="y", energy=3.0, passed=True, logprobs=None),
        ]
        winner = select_logprob(candidates)
        assert winner is not None
        assert winner.index == 1  # falls back to lens (lowest energy)

    def test_no_passing_returns_none(self):
        candidates = [CandidateInfo(index=0, code="x", energy=1.0, passed=False)]
        assert select_logprob(candidates) is None


class TestSelectOracle:
    def test_returns_any_passing(self):
        candidates = _make_candidates()
        winner = select_oracle(candidates)
        assert winner is not None
        assert winner.passed is True

    def test_no_passing_returns_none(self):
        candidates = [CandidateInfo(index=0, code="x", energy=1.0, passed=False)]
        assert select_oracle(candidates) is None


class TestSelectCandidate:
    def test_lens_strategy(self):
        candidates = _make_candidates()
        winner = select_candidate(candidates, strategy="lens")
        assert winner.index == 1

    def test_random_strategy(self):
        candidates = _make_candidates()
        winner = select_candidate(candidates, strategy="random", seed=42)
        assert winner is not None
        assert winner.passed is True

    def test_logprob_strategy(self):
        candidates = _make_candidates()
        winner = select_candidate(candidates, strategy="logprob")
        assert winner.index == 0

    def test_oracle_strategy(self):
        candidates = _make_candidates()
        winner = select_candidate(candidates, strategy="oracle")
        assert winner.passed is True

    def test_unknown_strategy_raises(self):
        candidates = _make_candidates()
        try:
            select_candidate(candidates, strategy="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "invalid" in str(e)
