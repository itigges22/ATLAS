"""Ebbinghaus decay scoring for cached patterns."""

import math
import logging

from models.pattern import Pattern, PatternScore, HALF_LIVES

logger = logging.getLogger(__name__)


def compute_score(pattern: Pattern, similarity: float) -> PatternScore:
    """
    Compute composite Ebbinghaus score for a pattern.

    score = similarity × 0.5^(days_since_last_access / half_life) × log(1 + access_count)

    Three multiplicative terms:
    1. similarity — BM25 relevance to current query (0.0-1.0)
    2. decay — temporal recency via Ebbinghaus forgetting curve
    3. frequency_boost — logarithmic access count (prevents runaway)
    """
    # Temporal decay: 0.5^(days / half_life)
    days = pattern.days_since_access()
    half_life = pattern.half_life_days
    if half_life <= 0:
        half_life = 14.0  # Fallback

    decay = math.pow(0.5, days / half_life)

    # Frequency boost: log(1 + access_count)
    freq = math.log(1 + pattern.access_count)
    # Minimum boost of 1.0 so new patterns aren't zeroed out
    freq = max(freq, 1.0)

    # Composite score
    composite = similarity * decay * freq

    return PatternScore(
        pattern=pattern,
        similarity=similarity,
        decay_factor=decay,
        frequency_boost=freq,
        composite_score=composite,
    )


def compute_storage_score(pattern: Pattern) -> float:
    """
    Compute storage score for sorted set ordering.
    Uses surprise_score as initial boost + decay + frequency.

    This is the score used for STM/LTM sorted set ordering (without query similarity).
    """
    days = pattern.days_since_access()
    half_life = pattern.half_life_days

    decay = math.pow(0.5, days / half_life)
    freq = max(math.log(1 + pattern.access_count), 1.0)

    # Surprise gives initial boost (high-retry = more valuable)
    surprise_boost = 1.0 + pattern.surprise_score

    return surprise_boost * decay * freq
