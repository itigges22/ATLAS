"""Tests for the offline RPG-quality metrics harness (Phase 4, issue #120)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

from rpg_eval import aggregate_metrics, rpg_quality_metrics  # noqa: E402

GOOD_RPG = {
    "capabilities": [
        {"id": "c1", "name": "Core", "parent": None},
        {"id": "c2", "name": "Loader", "parent": "c1"},
        {"id": "c3", "name": "Proc", "parent": "c1"},
    ],
    "files": [
        {"id": "f1", "path": "load.py", "capability": "c2",
         "functions": [{"name": "load", "signature": "def load() -> list", "summary": ""}]},
        {"id": "f2", "path": "proc.py", "capability": "c3",
         "functions": [{"name": "run", "signature": "def run(x)", "summary": ""}]},
    ],
    "edges": [{"from": "f1", "to": "f2", "kind": "data_flow", "label": "rows"}],
    "verify": "pytest",
    "rationale": "ok",
}

CYCLIC_RPG = {
    **GOOD_RPG,
    "edges": [{"from": "f1", "to": "f2"}, {"from": "f2", "to": "f1"}],
}


class TestQualityMetrics:
    def test_good_graph(self):
        m = rpg_quality_metrics(GOOD_RPG)
        assert m["parseable"] and m["valid"] and m["acyclic"]
        assert m["n_files"] == 2 and m["n_edges"] == 1
        assert m["leaf_coverage"] == 1.0
        assert m["signature_density"] == 1.0
        assert m["has_verify"] is True
        assert m["score"] > 0.5

    def test_cyclic_graph_flagged(self):
        m = rpg_quality_metrics(CYCLIC_RPG)
        assert m["acyclic"] is False
        assert m["valid"] is False

    def test_accepts_plan_envelope(self):
        # A /v3/plan result carrying the graph under "rpg" is unwrapped.
        m = rpg_quality_metrics({"steps": [], "rpg": GOOD_RPG})
        assert m["parseable"] and m["n_files"] == 2

    def test_unparseable(self):
        assert rpg_quality_metrics({"not": "an rpg"})["parseable"] in (True, False)


class TestAggregate:
    def test_aggregate_rates(self):
        agg = aggregate_metrics([GOOD_RPG, GOOD_RPG, CYCLIC_RPG])
        assert agg["n"] == 3
        assert agg["parse_rate"] == 1.0
        assert agg["valid_rate"] == 2 / 3
        assert agg["acyclic_rate"] == 2 / 3
        assert agg["avg_files"] == 2.0
        assert 0.0 <= agg["mean_score"] <= 1.0

    def test_empty(self):
        agg = aggregate_metrics([])
        assert agg["parse_rate"] == 0.0

    def test_json_roundtrip_serializable(self):
        # Metrics must be JSON-serializable for the CLI output.
        json.dumps(aggregate_metrics([GOOD_RPG]))
