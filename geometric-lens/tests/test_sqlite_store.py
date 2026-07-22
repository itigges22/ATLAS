"""SQLite state store: schema init, Thompson state, patterns, co-occurrence."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# The task-queue and request-metrics tables were dropped with the lens's
# /v1/tasks/* and /v1/chat/completions endpoints — nothing wrote or read them
# afterwards.
EXPECTED_TABLES = {
    "patterns", "co_occurrence", "thompson_state", "routing_stats",
    "store_metadata",
}


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
            "INSERT INTO store_metadata (key, value) VALUES ('probe', 7)")

    # Re-run initialization against the existing database file.
    store.SQLitePool._instance = None
    pool2 = store.get_db_pool()
    assert EXPECTED_TABLES <= _table_names(pool2)
    with pool2.get_connection() as conn:
        cur = conn.execute(
            "SELECT value FROM store_metadata WHERE key = 'probe'")
        assert cur.fetchone()["value"] == 7  # existing rows survive re-init


# ── Thompson state ──────────────────────────────────────────────────


def test_thompson_upsert_and_read(store):
    from models.route import Route, DifficultyBin
    from router.feedback_recorder import record_outcome
    from router.route_selector import _load_thompson_state

    record_outcome(DifficultyBin.LOW, Route.FAST_PATH, success=True)
    record_outcome(DifficultyBin.LOW, Route.FAST_PATH, success=True)
    record_outcome(DifficultyBin.LOW, Route.FAST_PATH, success=False)

    pool = store.get_db_pool()
    with pool.get_connection() as conn:
        cur = conn.execute(
            "SELECT alpha, beta FROM thompson_state "
            "WHERE difficulty_bin = ? AND route = ?",
            (DifficultyBin.LOW.value, Route.FAST_PATH.value))
        row = cur.fetchone()
    assert row["alpha"] == 2.0
    assert row["beta"] == 1.0

    # Loader adds the Beta(1,1) prior on top of the raw counts.
    states = _load_thompson_state(DifficultyBin.LOW)
    assert states[Route.FAST_PATH] == (3.0, 2.0)
    # Routes with no outcomes read as the uniform prior.
    assert states[Route.HARD_PATH] == (1.0, 1.0)


def test_thompson_reset(store):
    from models.route import Route, DifficultyBin
    from router.feedback_recorder import record_outcome
    from router.route_selector import reset_thompson_state

    record_outcome(DifficultyBin.HIGH, Route.STANDARD, success=True)
    reset_thompson_state()

    pool = store.get_db_pool()
    with pool.get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) AS c FROM thompson_state")
        assert cur.fetchone()["c"] == 0


# ── Task queue ──────────────────────────────────────────────────────










# ── Metrics ─────────────────────────────────────────────────────────




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


def test_pattern_store_crud_and_version_bump(store):
    from cache.pattern_store import get_pattern_store

    ps = get_pattern_store()
    assert ps.available
    v0 = ps.get_version()

    pattern = _make_pattern()
    assert ps.store_pattern(pattern, score=0.4)
    v1 = ps.get_version()
    assert v1 > v0

    got = ps.get_pattern(pattern.id)
    assert got is not None
    assert got.id == pattern.id
    assert got.content == pattern.content

    got.summary = "updated summary"
    assert ps.update_pattern(got, score=0.9)
    v2 = ps.get_version()
    assert v2 > v1
    assert ps.get_pattern(pattern.id).summary == "updated summary"

    assert ps.delete_pattern(pattern.id)
    assert ps.get_version() > v2
    assert ps.get_pattern(pattern.id) is None


def test_pattern_store_tier_listing_and_scores(store):
    from models.pattern import PatternTier
    from cache.pattern_store import get_pattern_store

    ps = get_pattern_store()
    ps.store_pattern(_make_pattern("stm-low"), score=0.1)
    ps.store_pattern(_make_pattern("stm-high"), score=0.9)
    ps.store_pattern(_make_pattern("ltm-1", tier=PatternTier.LTM), score=0.5)

    stm = ps.get_stm_patterns()
    assert [p.id for p in stm] == ["stm-high", "stm-low"]  # score-descending
    assert [p.id for p in ps.get_ltm_patterns()] == ["ltm-1"]
    assert ps.stm_size() == 2
    assert ps.ltm_size() == 1
