"""V3 Candidate Selection Strategies — baseline selectors for ablation.

Provides four selection strategies for choosing among passing candidates:
- lens: Select by lowest C(x) energy (current V3 default)
- random: Uniform random selection (baseline)
- logprob: Select by highest mean token log-probability (baseline)
- oracle: Always select a passing candidate if one exists (ceiling)

Config: ATLAS_V3_SELECTION_STRATEGY in atlas.conf
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CandidateInfo:
    """Minimal candidate info needed for selection."""
    index: int
    code: str
    energy: float
    passed: bool
    logprobs: Optional[List[float]] = None


def select_lens(candidates: List[CandidateInfo]) -> Optional[CandidateInfo]:
    """Select the passing candidate with lowest C(x) energy.

    This is the default V3 strategy — the Geometric Lens picks the
    candidate it believes is most likely correct.
    """
    passing = [c for c in candidates if c.passed]
    if not passing:
        return None
    return min(passing, key=lambda c: c.energy)


def select_random(candidates: List[CandidateInfo],
                  seed: Optional[int] = None) -> Optional[CandidateInfo]:
    """Select a passing candidate uniformly at random.

    Baseline: proves whether structured selection (Lens) outperforms
    naive uniform selection from k candidates.
    """
    passing = [c for c in candidates if c.passed]
    if not passing:
        return None
    rng = random.Random(seed)
    return rng.choice(passing)


def select_logprob(candidates: List[CandidateInfo]) -> Optional[CandidateInfo]:
    """Select the passing candidate with highest mean token log-probability.

    Baseline: model confidence (generation probability) as a selection
    signal. If logprobs are unavailable, falls back to lens selection.
    """
    passing = [c for c in candidates if c.passed]
    if not passing:
        return None

    # Filter to candidates with logprobs
    with_logprobs = [c for c in passing if c.logprobs]
    if not with_logprobs:
        return select_lens(candidates)

    def mean_logprob(c):
        return sum(c.logprobs) / len(c.logprobs) if c.logprobs else float('-inf')

    return max(with_logprobs, key=mean_logprob)


def select_oracle(candidates: List[CandidateInfo]) -> Optional[CandidateInfo]:
    """Select any passing candidate (oracle ceiling measurement).

    This represents the theoretical ceiling: if you could always identify
    a correct candidate from the pool, what would pass@1 be? Equivalent
    to pass@k but reported as pass@1 for comparison.
    """
    passing = [c for c in candidates if c.passed]
    if not passing:
        return None
    return passing[0]


# Strategy registry
STRATEGIES = {
    "lens": select_lens,
    "random": select_random,
    "logprob": select_logprob,
    "oracle": select_oracle,
}


def select_candidate(candidates: List[CandidateInfo],
                     strategy: str = "lens",
                     seed: Optional[int] = None) -> Optional[CandidateInfo]:
    """Select a candidate using the specified strategy.

    Args:
        candidates: List of candidate info objects.
        strategy: One of "lens", "random", "logprob", "oracle".
        seed: Random seed (only used by "random" strategy).

    Returns:
        Selected candidate, or None if no passing candidates.
    """
    if strategy == "random":
        return select_random(candidates, seed=seed)

    selector = STRATEGIES.get(strategy)
    if selector is None:
        raise ValueError(f"Unknown selection strategy: {strategy}. "
                         f"Choose from: {list(STRATEGIES.keys())}")
    return selector(candidates)
