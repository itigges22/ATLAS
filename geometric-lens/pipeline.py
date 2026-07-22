"""Geometric Lens pipeline — Pattern Cache writes and Confidence Router feedback.

Reached from main.py's /internal/patterns/write (which v3-service calls after a
successful candidate) and the router feedback path. The retrieval orchestration
this module was named for is gone: PageIndex/BM25 retrieval served only the lens's
own /v1/chat/completions, which had no callers.
"""

import os
import logging
from typing import List, Optional
from datetime import datetime, timezone

from config import config

logger = logging.getLogger(__name__)


def is_routing_enabled() -> bool:
    """Check if confidence routing is enabled (ROUTING_ENABLED env var)."""
    return os.environ.get("ROUTING_ENABLED", "true").lower() in ("true", "1", "yes")


# ──────────────────────────────────────────────────────────────
# Pattern Cache: Read Path
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# Pattern Cache: Write Path
# ──────────────────────────────────────────────────────────────

async def write_pattern_async(
    query: str,
    solution: str,
    retry_count: int,
    max_retries: int,
    error_context: Optional[str],
    source_files: List[str],
    active_pattern_ids: Optional[List[str]] = None,
):
    """
    Write path: extract and store a pattern from a successful task completion.
    Runs ASYNC — does not block the response pipeline.
    """
    from cache.pattern_store import get_pattern_store
    from cache.pattern_extractor import extract_pattern
    from cache.pattern_scorer import compute_storage_score
    from cache.co_occurrence import CoOccurrenceGraph
    from cache.consolidator import update_category_surprise

    store = get_pattern_store()
    if not store.available:
        return

    try:
        # Extract pattern via LLM
        pattern = await extract_pattern(
            query=query,
            solution=solution,
            retry_count=retry_count,
            max_retries=max_retries,
            error_context=error_context,
            source_files=source_files,
            llama_url=config.llama.base_url,
        )

        if not pattern:
            logger.warning("Pattern extraction returned None, skipping write")
            return

        # Compute storage score and store
        score = compute_storage_score(pattern)
        store.store_pattern(pattern, score=score)
        store.record_write()

        logger.info(
            f"Pattern written: {pattern.id} type={pattern.type.value} "
            f"surprise={pattern.surprise_score:.2f} score={score:.3f}"
        )

        # Update co-occurrence graph
        pattern_ids = [pattern.id]
        if active_pattern_ids:
            pattern_ids.extend(active_pattern_ids)

        if len(pattern_ids) >= 2:
            cooccur = CoOccurrenceGraph()
            cooccur.record_co_occurrence(pattern_ids)

        # Update category surprise
        update_category_surprise(pattern.type, pattern.surprise_score)

    except Exception as e:
        logger.error(f"Pattern write failed: {e}")


async def record_pattern_outcome(
    pattern_ids: List[str],
    success: bool,
):
    """Record whether injected patterns led to task success or failure."""
    from cache.pattern_store import get_pattern_store
    from cache.pattern_scorer import compute_storage_score

    store = get_pattern_store()
    if not store.available:
        return

    for pid in pattern_ids:
        pattern = store.get_pattern(pid)
        if pattern:
            if success:
                pattern.success_count += 1
                pattern.last_success = datetime.now(timezone.utc).isoformat()
            else:
                pattern.failure_count += 1
            score = compute_storage_score(pattern)
            store.update_pattern(pattern, score=score)


# ──────────────────────────────────────────────────────────────
# Main completion pipeline
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# Confidence Router: Feedback Recording
# ──────────────────────────────────────────────────────────────

def record_route_feedback(
    route_value: str,
    difficulty_bin_value: str,
    success: bool,
):
    """Record a routing outcome to update Thompson Sampling state."""
    if not is_routing_enabled():
        return

    try:
        from router.feedback_recorder import record_outcome
        from models.route import Route, DifficultyBin

        route = Route(route_value)
        d_bin = DifficultyBin(difficulty_bin_value)
        record_outcome(d_bin, route, success)
    except Exception as e:
        logger.error(f"Failed to record route feedback: {e}")


