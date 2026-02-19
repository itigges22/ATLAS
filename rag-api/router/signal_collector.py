"""Collect routing signals from Pattern Cache, PageIndex, and query heuristics."""

import logging
from typing import List, Optional

from models.route import SignalBundle

logger = logging.getLogger(__name__)


def compute_query_complexity(query: str) -> float:
    """
    Heuristic query complexity score in [0,1].

    Formula: min(1.0, (tokens/500 + lines/50 + code_blocks/3) / 3)
    No LLM call needed — pure heuristic.
    """
    tokens = len(query.split())
    lines = query.count("\n") + 1
    code_blocks = query.count("```")

    raw = (tokens / 500 + lines / 50 + code_blocks / 3) / 3
    return min(1.0, max(0.0, raw))


def compute_pattern_cache_score(scored_patterns) -> float:
    """
    Extract pattern cache score from scored patterns (read path result).

    Uses BM25 similarity weighted by pattern tier:
    - STM/LTM patterns (learned from actual tasks): full weight
    - Persistent/seed patterns (universal): discounted to 0.3x
      (they match everything broadly and don't indicate task-specific familiarity)

    If no patterns matched, returns 0.0 (high difficulty signal).
    """
    if not scored_patterns:
        return 0.0

    best_score = 0.0
    for ps in scored_patterns[:3]:
        sim = ps.similarity
        tier = ps.pattern.tier.value
        # Discount seed/persistent patterns — they match broadly
        if tier == "persistent":
            sim *= 0.3
        if sim > best_score:
            best_score = sim

    return min(1.0, max(0.0, best_score))


def compute_retrieval_confidence(chunks: List[dict]) -> float:
    """
    Estimate retrieval confidence from PageIndex/BM25 results.

    Uses the number of chunks and their scores as a signal.
    High confidence = we found relevant code context.
    """
    if not chunks:
        return 0.0

    # Use the number of chunks returned relative to expected
    count_signal = min(1.0, len(chunks) / 5.0)

    # If chunks have a score field, use the top score
    top_score = 0.0
    for chunk in chunks[:3]:
        score = chunk.get("score", 0.0)
        if score > top_score:
            top_score = score

    # Normalize score (BM25 scores can be >1, tree scores are 0-10)
    score_signal = min(1.0, top_score / 5.0) if top_score > 0 else count_signal

    # Blend count and score signals
    return 0.4 * count_signal + 0.6 * score_signal


def compute_geometric_energy(query: str) -> float:
    """
    Compute geometric energy signal from the Geometric Lens.

    Returns normalized energy in [0,1]:
    - Low energy (near 0) = query is in the "correct code" attractor basin
    - High energy (near 1) = query is in a bug-prone region

    Returns 0.0 if Geometric Lens is disabled or unavailable.
    """
    try:
        from geometric_lens.service import get_geometric_energy, is_enabled
        if not is_enabled():
            return 0.0
        return get_geometric_energy(query)
    except Exception as e:
        logger.debug(f"Geometric energy unavailable: {e}")
        return 0.0


def collect_signals(
    query: str,
    scored_patterns=None,
    chunks: Optional[List[dict]] = None,
) -> SignalBundle:
    """
    Collect all routing signals into a SignalBundle.

    Args:
        query: The user's query text.
        scored_patterns: Output from retrieve_cached_patterns() (may be None/empty).
        chunks: Retrieved PageIndex/BM25 chunks (may be None/empty).

    Returns:
        SignalBundle with all 4 signals (geometric_energy is 0.0 if lens disabled).
    """
    s_p = compute_pattern_cache_score(scored_patterns)
    q_c = compute_query_complexity(query)
    r_c = compute_retrieval_confidence(chunks or [])
    g_e = compute_geometric_energy(query)

    bundle = SignalBundle(
        pattern_cache_score=s_p,
        retrieval_confidence=r_c,
        query_complexity=q_c,
        geometric_energy=g_e,
    )

    logger.debug(
        f"Signals collected: cache={s_p:.3f} retrieval={r_c:.3f} "
        f"complexity={q_c:.3f} geometric={g_e:.3f}"
    )

    return bundle
