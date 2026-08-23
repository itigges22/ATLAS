"""Lens state policy: one frozen snapshot, shared by every case, never written.

A paired A/B run reads the pattern cache on both arms. Three things mutate
it, and only the first is obvious:

  the write path  adds patterns after a solve;
  the read path   updates last_accessed and access_count, which retrieval
                  scores on;
  the read path   also bumps hit/miss counters, synchronously, without going
                  through the async spawn helper the other two use.

So without a freeze, the patterns an early case touches change what a later
case is served, the two arms are no longer being run against the same cache,
and case order becomes a variable in the result.

In the 2026-08-23 acquisition the writes happened to fail with HTTP 403 while
the reads succeeded and recorded their accesses. The state was neither frozen
nor reset -- it was drifting in one direction only, by accident. These tests
hold the policy that replaced that accident.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LENS = os.path.join(ROOT, "geometric-lens")

# Both services lay their modules out flat, so `main`, `pipeline`, `config`
# and `scoring` name a different file in each. tests/v3/conftest.py puts
# v3-service on sys.path, which makes a bare `import main` here resolve to
# the V3 service -- passing when this file runs alone and failing the moment
# it runs beside the V3 tests. Load the lens by path, with the colliding
# names evicted for the duration of the import.
_COLLIDING = ("main", "pipeline", "config", "scoring", "adapters",
              "symbols", "planning", "structured_log", "contract")


def _load_lens_main():
    import importlib

    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.pop(name)
                     for name in _COLLIDING if name in sys.modules}
    sys.path.insert(0, LENS)
    try:
        return importlib.import_module("main")
    finally:
        sys.path[:] = saved_path
        for name in _COLLIDING:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def _main(monkeypatch, value):
    """Load geometric-lens main with the env var set, without booting it."""
    if value is None:
        monkeypatch.delenv("ATLAS_LENS_ONLINE_LEARNING", raising=False)
    else:
        monkeypatch.setenv("ATLAS_LENS_ONLINE_LEARNING", value)
    return _load_lens_main()


@pytest.mark.parametrize("value,expected", [
    (None, True),      # production default: the cache learns
    ("1", True),
    ("true", True),
    ("0", False),
    ("false", False),
    ("no", False),
    ("off", False),
    ("OFF", False),
    (" 0 ", False),
])
def test_flag_parsing(monkeypatch, value, expected):
    main = _main(monkeypatch, value)
    assert main.online_learning_enabled() is expected


def test_default_is_on_so_production_is_unchanged(monkeypatch):
    main = _main(monkeypatch, None)
    assert main.online_learning_enabled() is True


def test_frozen_spawns_no_mutating_task(monkeypatch):
    """The gate sits on the spawn helper, so a new mutating path is frozen
    by default rather than frozen only if someone remembered."""
    main = _main(monkeypatch, "0")
    started = []

    class _Coro:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    coro = _Coro()
    monkeypatch.setattr(main, "_pattern_write_tasks", set())
    main._spawn_pattern_task(coro)
    assert started == []
    assert coro.closed, "a frozen run left a coroutine un-awaited"
    assert not main._pattern_write_tasks


def test_unfrozen_still_spawns(monkeypatch):
    main = _main(monkeypatch, "1")
    import asyncio

    created = []
    monkeypatch.setattr(main, "_pattern_write_tasks", set())

    class _Task:
        def add_done_callback(self, cb):
            pass

    def _fake_create_task(coro):
        created.append(coro)
        coro.close()
        return _Task()

    monkeypatch.setattr(asyncio, "create_task", _fake_create_task)

    async def _noop():
        return None

    main._spawn_pattern_task(_noop())
    assert len(created) == 1


def test_health_states_the_policy(monkeypatch):
    """A frozen cache and a cache whose writes are silently failing look the
    same from outside. The run has to be able to check which it got."""
    main = _main(monkeypatch, "0")
    monkeypatch.setattr(main, "_db_state", lambda: {"connected": True})
    monkeypatch.setattr(main, "_llama_state", lambda: {"reachable": True})
    monkeypatch.setitem(main._BOOT_STATE, "lens_enabled", False)
    assert main.health()["online_learning"] is False

    main = _main(monkeypatch, "1")
    monkeypatch.setattr(main, "_db_state", lambda: {"connected": True})
    monkeypatch.setattr(main, "_llama_state", lambda: {"reachable": True})
    assert main.health()["online_learning"] is True


def test_read_path_stat_counters_are_frozen_too(monkeypatch):
    """The read path bumps hit/miss counters synchronously.

    They never pass through _spawn_pattern_task, so gating only the spawn
    helper left a pure read still writing. Measured on the 42-case rehearsal:
    booting the lens on a snapshot changed nothing, and three pattern-context
    reads moved the state.

    The store here has no connection pool, so reaching the write raises. That
    is the signal: frozen returns quietly, unfrozen blows up on the pool it
    tried to use.
    """
    saved_path = list(sys.path)
    saved_modules = {n: sys.modules.pop(n) for n in _COLLIDING if n in sys.modules}
    sys.path.insert(0, LENS)
    try:
        import importlib
        store_mod = importlib.import_module("cache.pattern_store")

        store = store_mod.PatternStore.__new__(store_mod.PatternStore)
        store._available = True
        store._pool = None

        monkeypatch.setenv("ATLAS_LENS_ONLINE_LEARNING", "0")
        store._incr_stat("hits")  # frozen: returns before touching the pool

        monkeypatch.setenv("ATLAS_LENS_ONLINE_LEARNING", "1")
        reached = []
        original = store_mod.PatternStore._incr_stat

        class _Pool:
            def get_connection(self):
                reached.append(True)
                raise RuntimeError("no pool in this test")

        store._pool = _Pool()
        original(store, "hits")  # errors are swallowed by design
        assert reached, "unfrozen _incr_stat never reached the store"
    finally:
        sys.path[:] = saved_path
        for n in _COLLIDING:
            sys.modules.pop(n, None)
        sys.modules.update(saved_modules)


def test_every_mutating_path_goes_through_the_gate():
    """Structural: no mutating coroutine is awaited around the helper.

    record_pattern_access, write_pattern_async and record_pattern_outcome
    each change stored state. If one is scheduled without _spawn_pattern_task
    it escapes the freeze, and the escape is invisible in a passing run.
    """
    import ast

    src = open(os.path.join(LENS, "main.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    mutators = {"record_pattern_access", "write_pattern_async",
                "record_pattern_outcome"}
    gated = set()
    stray = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name == "_spawn_pattern_task":
            for arg in node.args:
                inner = getattr(getattr(arg, "func", None), "id", None)
                if inner in mutators:
                    gated.add(inner)
        elif name in mutators:
            # Allowed only as the argument of a _spawn_pattern_task call,
            # which the branch above already counted.
            stray.append((name, node.lineno))

    ungated = [s for s in stray
               if not _is_inside_spawn(tree, s[1])]
    assert not ungated, (
        f"state-mutating call(s) scheduled outside the freeze gate: {ungated}")
    assert gated == mutators, (
        f"expected every mutator to be gated; gated={sorted(gated)}")


def _is_inside_spawn(tree, lineno):
    import ast

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "_spawn_pattern_task":
            continue
        if node.lineno <= lineno <= (node.end_lineno or node.lineno):
            return True
    return False
