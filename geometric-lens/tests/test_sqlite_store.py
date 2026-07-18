"""SQLite state store: schema init, Thompson state, task queue, metrics, patterns."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXPECTED_TABLES = {
    "patterns", "co_occurrence", "thompson_state", "routing_stats",
    "tasks", "metrics_daily", "metrics_recent_tasks", "store_metadata",
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
    with pool.get_connection() as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = 'tasks' AND name NOT LIKE 'sqlite_%'")
        names = {row["name"] for row in cur.fetchall()}
    assert "idx_tasks_status_priority_created" in names


def test_schema_init_is_idempotent(store):
    pool = store.get_db_pool()
    with pool.get_connection() as conn:
        conn.execute(
            "INSERT INTO tasks (id, priority, status, data) "
            "VALUES ('t1', 'p1', 'pending', '{}')")

    # Re-run initialization against the existing database file.
    store.SQLitePool._instance = None
    pool2 = store.get_db_pool()
    assert EXPECTED_TABLES <= _table_names(pool2)
    with pool2.get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) AS c FROM tasks")
        assert cur.fetchone()["c"] == 1  # existing rows survive re-init


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


def _submit(pool, task_id, priority):
    with pool.get_connection() as conn:
        conn.execute(
            "INSERT INTO tasks (id, priority, status, data) "
            "VALUES (?, ?, 'pending', ?)",
            (task_id, priority, json.dumps({"id": task_id})))


def _next_pending(pool, priority):
    with pool.get_connection() as conn:
        cur = conn.execute(
            "SELECT id FROM tasks WHERE status = 'pending' AND priority = ? "
            "ORDER BY created_at, rowid LIMIT 1", (priority,))
        row = cur.fetchone()
    return row["id"] if row else None


def test_task_queue_fifo_within_priority(store):
    pool = store.get_db_pool()
    for tid in ("a", "b", "c"):
        _submit(pool, tid, "p1")
    _submit(pool, "urgent", "p0")

    order = []
    while True:
        tid = _next_pending(pool, "p1")
        if tid is None:
            break
        order.append(tid)
        with pool.get_connection() as conn:
            conn.execute(
                "UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
    assert order == ["a", "b", "c"]  # FIFO within a priority
    # Other priorities are untouched.
    assert _next_pending(pool, "p0") == "urgent"


def test_task_status_transitions_and_queue_stats(store):
    pool = store.get_db_pool()
    _submit(pool, "t1", "p0")
    _submit(pool, "t2", "p1")
    _submit(pool, "t3", "p1")

    def pending_counts():
        with pool.get_connection() as conn:
            cur = conn.execute(
                "SELECT priority, COUNT(*) AS count FROM tasks "
                "WHERE status = 'pending' GROUP BY priority")
            return {row["priority"]: row["count"] for row in cur.fetchall()}

    assert pending_counts() == {"p0": 1, "p1": 2}

    # pending -> running -> completed
    for status in ("running", "completed"):
        with pool.get_connection() as conn:
            conn.execute(
                "UPDATE tasks SET status = ? WHERE id = 't2'", (status,))
        with pool.get_connection() as conn:
            cur = conn.execute("SELECT status FROM tasks WHERE id = 't2'")
            assert cur.fetchone()["status"] == status

    # Completed tasks leave the pending pool but stay queryable.
    assert pending_counts() == {"p0": 1, "p1": 1}


# ── Metrics ─────────────────────────────────────────────────────────


def test_recent_tasks_trims_to_100(store, tmp_path, monkeypatch):
    # main's import-time side effects need a writable project dir.
    monkeypatch.setenv("PROJECT_DATA_DIR", str(tmp_path / "projects"))
    # `import main` by bare name collides with v3-service's main when
    # another suite imported it first (both are flat top-level modules);
    # load the lens main under a unique sys.modules key instead.
    import importlib.util
    main = sys.modules.get("lens_main")
    if main is None:
        spec = importlib.util.spec_from_file_location(
            "lens_main",
            os.path.join(os.path.dirname(__file__), "..", "main.py"))
        main = importlib.util.module_from_spec(spec)
        sys.modules["lens_main"] = main
        spec.loader.exec_module(main)

    for i in range(120):
        main.log_request_metrics("chat_completion", success=(i % 2 == 0),
                                 tokens=i, model=f"m{i}")

    pool = store.get_db_pool()
    with pool.get_connection() as conn:
        cur = conn.execute(
            "SELECT task_record FROM metrics_recent_tasks ORDER BY id")
        records = [json.loads(row["task_record"]) for row in cur.fetchall()]

    assert len(records) == 100
    # Oldest 20 trimmed, newest kept, insertion order preserved.
    assert records[0]["model"] == "m20"
    assert records[-1]["model"] == "m119"

    # Daily counters aggregated in metrics_daily.
    with pool.get_connection() as conn:
        cur = conn.execute("SELECT key, value FROM metrics_daily")
        daily = {row["key"]: row["value"] for row in cur.fetchall()}
    assert daily["tasks_total"] == 120
    assert daily["tasks_success"] == 60
    assert daily["tokens_total"] == sum(range(120))


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
