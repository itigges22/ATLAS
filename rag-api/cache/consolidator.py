"""STM → LTM promotion, LTM pruning, category surprise tracking."""

import logging
from typing import Dict

from cache.pattern_store import get_pattern_store
from cache.pattern_scorer import compute_storage_score
from cache.co_occurrence import CoOccurrenceGraph
from models.pattern import PatternType

logger = logging.getLogger(__name__)

# Promotion criteria (from Memoria/Moltbook)
MIN_ACCESS_COUNT = 3       # Must be retrieved at least 3 times
MIN_SUCCESS_RATE = 0.6     # At least 60% success rate
MIN_AGE_DAYS = 3.0         # Must survive at least 3 days

# Pruning threshold
LTM_PRUNE_SCORE = 0.01    # Remove LTM patterns below this score

# Category surprise tracking
_category_surprise: Dict[str, float] = {}


async def run_consolidation():
    """
    Run the consolidation cycle:
    1. Scan STM for promotion candidates
    2. Promote qualifying patterns to LTM
    3. Prune decayed LTM patterns
    """
    store = get_pattern_store()
    if not store.available:
        return

    promoted = 0
    pruned = 0

    # Phase 1: STM → LTM promotion
    stm_patterns = store.get_stm_patterns(limit=100)
    for pattern in stm_patterns:
        if _should_promote(pattern):
            score = compute_storage_score(pattern)
            if store.promote_to_ltm(pattern.id, score):
                promoted += 1

    # Phase 2: LTM pruning
    ltm_patterns = store.get_ltm_patterns(limit=500)
    cooccur = CoOccurrenceGraph()
    for pattern in ltm_patterns:
        score = compute_storage_score(pattern)
        if score < LTM_PRUNE_SCORE:
            store.delete_pattern(pattern.id)
            cooccur.cleanup_pattern(pattern.id)
            pruned += 1

    if promoted > 0 or pruned > 0:
        logger.info(
            f"Consolidation: promoted {promoted} STM→LTM, pruned {pruned} from LTM"
        )


def _should_promote(pattern) -> bool:
    """Check if a pattern meets STM → LTM promotion criteria."""
    if pattern.access_count < MIN_ACCESS_COUNT:
        return False
    if pattern.success_rate() < MIN_SUCCESS_RATE:
        return False
    if pattern.age_days() < MIN_AGE_DAYS:
        return False
    return True


def update_category_surprise(pattern_type: PatternType, surprise: float):
    """Update running category surprise score (momentum from Titans)."""
    key = pattern_type.value
    current = _category_surprise.get(key, 0.0)
    # Exponential moving average with η=0.3
    eta = 0.3
    _category_surprise[key] = eta * current + (1 - eta) * surprise


def get_category_surprise(pattern_type: PatternType) -> float:
    """Get current category surprise level."""
    return _category_surprise.get(pattern_type.value, 0.0)
