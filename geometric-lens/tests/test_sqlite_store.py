"""SQLite state store: schema init, patterns."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXPECTED_TABLES = {"patterns", "co_occurrence", "store_metadata"}


def _reset_singletons():
    import sqlite_store
    from cache import pattern_store as pattern_store_mod

    sqlite_store.SQLitePool._instance = None
    pattern_store_mod._store = None


@pytest.fixture
def store(tmp_path, monkeypatch):
    """sqlite_store pointed at a per-test database, singletons reset."""
    import sqlite_store

    monkeypatch.setattr(sqlite_store, "DB_PATH", str(tmp_path / "state.db"))
    _reset_singletons()
    yield sqlite_store
    _reset_singletons()


def _table_names(pool):
    with pool.get_connection() as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")
        return {row["name"] for row in cur.fetchall()}


# ── Schema ──────────────────────────────────────────────────────────


def test_schema_init_creates_all_tables(store):
    pool = store.get_db_pool()
    assert EXPECTED_TABLES <= _table_names(pool)


def test_schema_init_is_idempotent(store):
    pool = store.get_db_pool()
    with pool.get_connection() as conn:
        conn.execute(
            "INSERT INTO patterns (id, data, tier, score) "
            "VALUES ('p1', '{}', 'stm', 0.5)")

    # Re-run initialization against the existing database file.
    store.SQLitePool._instance = None
    pool2 = store.get_db_pool()
    assert EXPECTED_TABLES <= _table_names(pool2)
    with pool2.get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) AS c FROM patterns")
        assert cur.fetchone()["c"] == 1  # existing rows survive re-init


# ── Pattern store ───────────────────────────────────────────────────


def _make_pattern(pid="pat-1", tier=None):
    from models.pattern import Pattern, PatternType, PatternTier

    return Pattern(
        id=pid,
        type=PatternType.BUG_FIX,
        tier=tier or PatternTier.STM,
        content="if x is None: raise ValueError('x')",
        summary="null check",
        context_query="null check pattern",
    )


def test_pattern_store_crud(store):
    from cache.pattern_store import get_pattern_store

    ps = get_pattern_store()
    assert ps.available

    pattern = _make_pattern()
    assert ps.store_pattern(pattern, score=0.4)

    got = ps.get_pattern(pattern.id)
    assert got is not None
    assert got.id == pattern.id
    assert got.content == pattern.content

    got.summary = "updated summary"
    assert ps.update_pattern(got, score=0.9)
    assert ps.get_pattern(pattern.id).summary == "updated summary"

    assert ps.delete_pattern(pattern.id)
    assert ps.get_pattern(pattern.id) is None


def test_pattern_store_tier_listing_and_scores(store):
    from models.pattern import PatternTier
    from cache.pattern_store import get_pattern_store

    ps = get_pattern_store()
    ps.store_pattern(_make_pattern("stm-low"), score=0.1)
    ps.store_pattern(_make_pattern("stm-high"), score=0.9)
    ps.store_pattern(
        _make_pattern("seed-1", tier=PatternTier.PERSISTENT), score=0.5)

    stm = ps.get_stm_patterns()
    assert [p.id for p in stm] == ["stm-high", "stm-low"]  # score-descending
    assert [p.id for p in ps.get_persistent_patterns()] == ["seed-1"]
    assert ps.stm_size() == 2
    assert ps.persistent_size() == 1
    assert {p.id for p in ps.get_all_patterns()} == {
        "stm-high", "stm-low", "seed-1"}
