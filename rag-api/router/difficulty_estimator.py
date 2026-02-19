"""Fuse signals into a single difficulty score D(x)."""

import logging
import os

from models.route import SignalBundle, DifficultyBin, difficulty_to_bin

logger = logging.getLogger(__name__)

# 3-signal weights (when Geometric Lens is disabled)
WEIGHTS_3_SIGNAL = {
    "pattern_cache": 0.40,
    "retrieval_confidence": 0.30,
    "query_complexity": 0.30,
    "geometric_energy": 0.00,
}

# 4-signal weights (when Geometric Lens is enabled)
WEIGHTS_4_SIGNAL = {
    "pattern_cache": 0.30,
    "retrieval_confidence": 0.25,
    "query_complexity": 0.20,
    "geometric_energy": 0.25,
}


def _get_weights() -> dict:
    """Get appropriate weights based on whether Geometric Lens is active."""
    lens_enabled = os.environ.get("GEOMETRIC_LENS_ENABLED", "false").lower() in ("true", "1", "yes")
    return WEIGHTS_4_SIGNAL if lens_enabled else WEIGHTS_3_SIGNAL


def estimate_difficulty(
    signals: SignalBundle,
    weights: dict = None,
) -> float:
    """
    Compute fused difficulty score D(x) in [0, 1].

    With 3 signals: D(x) = w1*(1-s_p) + w2*(1-r_c) + w3*q_c
    With 4 signals: D(x) = w1*(1-s_p) + w2*(1-r_c) + w3*q_c + w4*g_e

    Higher = more difficult = needs more compute.
    """
    w = weights or _get_weights()

    # Component 1: Pattern cache (inverted — high match = low difficulty)
    pattern_component = 1.0 - signals.pattern_cache_score

    # Component 2: Retrieval confidence (inverted — low confidence = high difficulty)
    retrieval_component = 1.0 - signals.retrieval_confidence

    # Component 3: Query complexity (direct — high = complex)
    complexity_component = signals.query_complexity

    # Component 4: Geometric energy (direct — high energy = bug-prone)
    geometric_component = signals.geometric_energy

    difficulty = (
        w["pattern_cache"] * pattern_component
        + w["retrieval_confidence"] * retrieval_component
        + w["query_complexity"] * complexity_component
        + w["geometric_energy"] * geometric_component
    )

    # Clamp to [0, 1]
    difficulty = min(1.0, max(0.0, difficulty))

    logger.debug(
        f"Difficulty estimated: D(x)={difficulty:.3f} "
        f"(pattern={pattern_component:.3f} retrieval={retrieval_component:.3f} "
        f"complexity={complexity_component:.3f} geometric={geometric_component:.3f})"
    )

    return difficulty


def get_difficulty_bin(difficulty: float) -> DifficultyBin:
    """Convenience wrapper for difficulty_to_bin."""
    return difficulty_to_bin(difficulty)
