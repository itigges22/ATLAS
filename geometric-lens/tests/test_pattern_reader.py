"""Pattern-cache read path: type + recency matching in pipeline.retrieve_cached_patterns."""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _reset_singletons():
    import sqlite_store
    from cache import pattern_store as pattern_store_mod

    sqlite_store.SQLitePool._instance = None
    pattern_store_mod._store = None


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Pattern store on a per-test database, singletons reset."""
    import sqlite_store

    monkeypatch.setattr(sqlite_store, "DB_PATH", str(tmp_path / "state.db"))
    _reset_singletons()
    from cache.pattern_store import get_pattern_store
    yield get_pattern_store()
    _reset_singletons()


def _make_pattern(pid, ptype, days_since_access=0.0):
    from models.pattern import Pattern

    accessed = datetime.now(timezone.utc) - timedelta(days=days_since_access)
    return Pattern(
        id=pid,
        type=ptype,
        content=f"content for {pid}",
        summary=f"summary for {pid}",
        context_query=f"query for {pid}",
        last_accessed=accessed.isoformat(),
    )


# A task the classify heuristic maps to BUG_FIX ("if .+ is None").
BUG_FIX_TASK = "guard the parser: if value is None the loop crashes"


def test_type_match_outranks_mismatch(store):
    from models.pattern import PatternType
    from pipeline import retrieve_cached_patterns

    store.store_pattern(_make_pattern("idiom-1", PatternType.IDIOM))
    store.store_pattern(_make_pattern("bugfix-1", PatternType.BUG_FIX))

    result = asyncio.run(retrieve_cached_patterns(BUG_FIX_TASK, top_k=2))

    assert [ps.pattern.id for ps in result] == ["bugfix-1", "idiom-1"]
    assert result[0].similarity == 1.0
    assert result[1].similarity == pytest.approx(0.3)
    assert store.get_stats()["hits"] == 1


def test_threshold_drops_decayed_mismatches(store):
    from models.pattern import PatternType
    from pipeline import retrieve_cached_patterns

    # Type mismatch AND ~4 half-lives of decay: composite far below the
    # relevance threshold, so nothing is served and a miss is recorded.
    store.store_pattern(
        _make_pattern("stale-idiom", PatternType.IDIOM, days_since_access=60))

    result = asyncio.run(retrieve_cached_patterns(BUG_FIX_TASK, top_k=3))

    assert result == []
    assert store.get_stats()["misses"] == 1


def test_top_k_caps_the_result(store):
    from models.pattern import PatternType
    from pipeline import retrieve_cached_patterns

    for i in range(5):
        store.store_pattern(_make_pattern(f"bugfix-{i}", PatternType.BUG_FIX))

    result = asyncio.run(retrieve_cached_patterns(BUG_FIX_TASK, top_k=3))
    assert len(result) == 3


def test_empty_store_records_miss(store):
    from pipeline import retrieve_cached_patterns

    result = asyncio.run(retrieve_cached_patterns(BUG_FIX_TASK, top_k=3))
    assert result == []
    assert store.get_stats()["misses"] == 1


def test_co_occurrence_boosts_linked_pattern(store):
    from cache.co_occurrence import CoOccurrenceGraph
    from models.pattern import PatternType
    from pipeline import retrieve_cached_patterns

    store.store_pattern(_make_pattern("bugfix-1", PatternType.BUG_FIX))
    store.store_pattern(_make_pattern("idiom-1", PatternType.IDIOM))
    CoOccurrenceGraph().record_co_occurrence(["bugfix-1", "idiom-1"])

    result = asyncio.run(retrieve_cached_patterns(BUG_FIX_TASK, top_k=2))

    # The linked idiom inherits the type-match similarity through the
    # full-weight edge instead of its own 0.3 mismatch floor.
    by_id = {ps.pattern.id: ps for ps in result}
    assert by_id["idiom-1"].similarity == pytest.approx(1.0)


def test_record_pattern_access_updates_stats(store):
    from models.pattern import PatternType
    from pipeline import record_pattern_access, retrieve_cached_patterns

    store.store_pattern(_make_pattern("bugfix-1", PatternType.BUG_FIX))
    result = asyncio.run(retrieve_cached_patterns(BUG_FIX_TASK, top_k=1))
    asyncio.run(record_pattern_access(result))

    assert store.get_pattern("bugfix-1").access_count == 1
