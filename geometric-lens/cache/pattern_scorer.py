"""Ebbinghaus decay scoring for cached patterns."""

import math
import logging

from models.pattern import Pattern, PatternScore, HALF_LIVES

logger = logging.getLogger(__name__)

# Saturation anchor for the frequency boost. At this many accesses the
# frequency term is at 1.0; below it the term grows logarithmically.
REF_ACCESS = 50.0


def _freq_boost(access_count: int) -> float:
    """Saturating log frequency in [0, 1]. 0 → 0, 1 → ~0.18, 5 → ~0.46, 50 → 1.0."""
    return min(math.log1p(max(access_count, 0)) / math.log1p(REF_ACCESS), 1.0)


def compute_score(pattern: Pattern, similarity: float) -> PatternScore:
    """
    Compute composite Ebbinghaus score for a pattern, in [0, 1].

    composite = similarity * decay * freq

    Three multiplicative terms, each in [0, 1]:
    1. similarity — BM25 relevance to the current query
    2. decay — temporal recency via Ebbinghaus forgetting curve
    3. freq — saturating log frequency with a 0.5 baseline so fresh
       patterns aren't zeroed out
    """
    days = pattern.days_since_access()
    half_life = pattern.half_life_days if pattern.half_life_days > 0 else 14.0

    decay = math.pow(0.5, days / half_life)
    # 0.5 baseline keeps fresh patterns (access_count=0) retrievable while
    # letting frequently-used patterns saturate the term to 1.0
    freq = 0.5 + 0.5 * _freq_boost(pattern.access_count)

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
    Compute storage score for STM/LTM sorted-set ordering.

    Used purely for relative ordering; not directly comparable to
    compute_score's composite (which is gated against an absolute threshold).
    """
    days = pattern.days_since_access()
    half_life = pattern.half_life_days if pattern.half_life_days > 0 else 14.0

    decay = math.pow(0.5, days / half_life)
    boost = _freq_boost(pattern.access_count)
    surprise_boost = 1.0 + pattern.surprise_score

    return surprise_boost * decay * (1.0 + boost)
