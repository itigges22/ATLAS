"""Geometric Lens pipeline — the Pattern Cache read and write paths."""

import logging
from typing import List, Optional
from datetime import datetime, timezone

from config import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Pattern Cache: Read Path
# ──────────────────────────────────────────────────────────────

# Similarity assigned by the type match: full weight when the task
# classifies to the pattern's type, a floor otherwise so a very recent /
# frequently-used pattern of another type can still surface.
TYPE_MATCH_SIMILARITY = 1.0
TYPE_MISMATCH_SIMILARITY = 0.3

# Minimum composite score (compute_score: similarity × decay × freq) for a
# pattern to be served. A fresh type-matched pattern scores 0.5; a
# type-mismatched one 0.15; two half-lives of decay put a type match at
# 0.125 — so the floor drops only long-unused mismatches.
PATTERN_RELEVANCE_THRESHOLD = 0.1


async def retrieve_cached_patterns(task: str, top_k: int = 3):
    """
    Read path: type + recency matching over the pattern cache.

    Flow:
    1. Classify the task text to a PatternType (same heuristic the write
       path uses — no LLM call)
    2. Score every STM + persistent pattern through compute_score with
       similarity 1.0 on a type match, 0.3 otherwise
    3. Expand 1 hop through the co-occurrence graph: a pattern linked to a
       type-matched one inherits similarity proportional to the edge weight
    4. Return the top-k above the relevance threshold
    """
    from cache.pattern_store import get_pattern_store
    from cache.pattern_scorer import compute_score
    from cache.pattern_extractor import classify_pattern_type
    from cache.co_occurrence import CoOccurrenceGraph

    store = get_pattern_store()
    if not store.available:
        return []

    candidates = store.get_all_patterns()
    if not candidates:
        store.record_miss()
        return []

    task_type = classify_pattern_type(task, None)
    by_id = {p.id: p for p in candidates}

    scored = {
        p.id: compute_score(
            p,
            TYPE_MATCH_SIMILARITY if p.type == task_type
            else TYPE_MISMATCH_SIMILARITY,
        )
        for p in by_id.values()
    }

    cooccur = CoOccurrenceGraph()
    for pattern in by_id.values():
        if pattern.type != task_type:
            continue
        for linked_id, edge_weight in cooccur.get_linked_patterns(
            pattern.id, top_k=3, max_depth=1
        ):
            linked = by_id.get(linked_id)
            if linked is None:
                continue
            inherited = TYPE_MATCH_SIMILARITY * edge_weight
            if inherited > scored[linked_id].similarity:
                scored[linked_id] = compute_score(linked, inherited)

    ranked = sorted(
        scored.values(), key=lambda ps: ps.composite_score, reverse=True
    )
    result = [
        ps for ps in ranked[:top_k]
        if ps.composite_score >= PATTERN_RELEVANCE_THRESHOLD
    ]

    if result:
        store.record_hit()
        logger.info(
            f"Pattern cache HIT: {len(result)} patterns "
            f"(task type={task_type.value}, "
            f"top score={result[0].composite_score:.3f})"
        )
    else:
        store.record_miss()

    return result


async def record_pattern_access(scored_patterns):
    """Update last_accessed and access_count for retrieved patterns."""
    from cache.pattern_store import get_pattern_store
    from cache.pattern_scorer import compute_storage_score

    store = get_pattern_store()
    if not store.available:
        return

    now = datetime.now(timezone.utc).isoformat()
    for ps in scored_patterns:
        p = ps.pattern
        p.last_accessed = now
        p.access_count += 1
        score = compute_storage_score(p)
        store.update_pattern(p, score=score)


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

