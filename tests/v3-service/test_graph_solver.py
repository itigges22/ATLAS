"""Tests for the fan-in/out complexity signal (Phase 4) and the in-process
Datalog solver layer (Phase 5)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

from graph.types import CodeGraph, CallsFact, DefinesFact  # noqa: E402
from graph import analyses  # noqa: E402
from graph.datalog import Datalog, Var, reachable_pairs, solver_reaches  # noqa: E402


def _g():
    # main -> helper -> leaf ; main -> util ; recurse -> recurse (self-loop)
    return CodeGraph(
        defines=[DefinesFact("a.py", n, "function", i)
                 for i, n in enumerate(["main", "helper", "leaf", "util", "recurse"])],
        calls=[
            CallsFact("main", "helper"), CallsFact("helper", "leaf"),
            CallsFact("main", "util"), CallsFact("recurse", "recurse"),
        ],
    )


class TestComplexity:
    def test_fan_in_out(self):
        c = analyses.complexity(_g())
        assert c["per_node"]["main"]["fan_out"] == 2   # helper, util
        assert c["per_node"]["main"]["fan_in"] == 0
        assert c["per_node"]["helper"]["fan_in"] == 1   # called by main
        assert c["per_node"]["helper"]["fan_out"] == 1  # calls leaf
        assert c["max_fan_out"] == 2
        assert c["n_edges"] == 4

    def test_dispatch(self):
        assert analyses.run_analysis(_g(), "complexity")["max_fan_out"] == 2


class TestDatalogEngine:
    def test_generic_transitive_rule(self):
        dl = Datalog()
        for a, b in [("a", "b"), ("b", "c"), ("c", "d")]:
            dl.add_fact("edge", a, b)
        x, y, z = Var("X"), Var("Y"), Var("Z")
        dl.add_rule(("tc", [x, y]), [("edge", [x, y])])
        dl.add_rule(("tc", [x, y]), [("edge", [x, z]), ("tc", [z, y])])
        dl.run()
        pairs = {tuple(t) for t in dl.facts["tc"]}
        assert ("a", "d") in pairs and ("a", "c") in pairs and ("b", "d") in pairs
        assert ("d", "a") not in pairs

    def test_unsafe_rule_does_not_crash(self):
        # A rule with a head variable not bound by the body is unsafe; run()
        # must skip it rather than raise KeyError (public-API robustness).
        dl = Datalog()
        dl.add_fact("calls", "a", "b")
        x, y = Var("X"), Var("Y")
        dl.add_rule(("p", [x, y]), [("calls", [x])])  # Y unbound
        dl.run()  # must not raise
        assert dl.facts.get("p", set()) == set()

    def test_query_bindings(self):
        dl = Datalog()
        dl.add_fact("calls", "main", "helper")
        dl.add_fact("calls", "main", "util")
        res = dl.query("calls", "main", Var("X"))
        callees = sorted(b["X"] for b in res)
        assert callees == ["helper", "util"]


class TestSolverMatchesNative:
    def test_closure_relation(self):
        pairs = set(reachable_pairs(_g()))
        assert ("main", "leaf") in pairs   # transitive
        assert ("recurse", "recurse") in pairs  # self-loop
        assert ("leaf", "main") not in pairs

    def test_reaches_agrees_with_native_for_all_pairs(self):
        g = _g()
        names = sorted({d.name for d in g.defines}
                       | {c.caller for c in g.calls} | {c.callee for c in g.calls})
        for a in names:
            for b in names:
                assert solver_reaches(g, a, b) == analyses.reachability(g, a, b), (a, b)

    def test_runaway_guard(self):
        # A dense cycle still terminates under the iteration cap.
        g = CodeGraph(calls=[CallsFact("a", "b"), CallsFact("b", "a")])
        pairs = set(reachable_pairs(g))
        assert ("a", "a") in pairs and ("b", "b") in pairs and ("a", "b") in pairs
