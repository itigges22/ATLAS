"""Offline RPG-quality metrics + evaluation harness (V3.2 Phase 4, issue #120).

The live RepoCraft-style comparison (RPG-on vs flat planner: functional
coverage, test accuracy, code scale, L6 localization turns) requires a GPU +
model and is run via the live benchmark stack — see the runbook in
docs/reports/RPG_WAVELET_PLANNING_V3_2.md (Phase 4). This module is the
*offline* half: it scores RPG artifacts (the JSON the planner emits) on
graph-quality metrics so a benchmark run — or a fixture set — can be summarized
without re-deriving the graph logic, and so regressions in plan shape are
catchable in CI.

Usage:
    python rpg_eval.py artifacts/*.json        # aggregate over RPG JSON files
    python rpg_eval.py --jsonl plans.jsonl     # one RPG (or {"rpg": ...}) per line
"""

from __future__ import annotations

import glob
import json
import sys
from typing import List

import rpg as rpgmod


def rpg_quality_metrics(rpg_dict: dict) -> dict:
    """Graph-quality metrics for a single RPG artifact (a dict as emitted by
    rpg.RPG.to_dict, or a plan envelope carrying it under "rpg")."""
    if "rpg" in rpg_dict and isinstance(rpg_dict["rpg"], dict):
        rpg_dict = rpg_dict["rpg"]
    g = rpgmod.parse_rpg(json.dumps(rpg_dict))
    if g is None:
        return {"parseable": False}

    ok, issues = rpgmod.validate_rpg(g)
    _, acyclic = rpgmod._topo_order([f.id for f in g.files], g.edges)

    caps = g.capabilities
    parents = {c.parent for c in caps if c.parent}
    leaves = [c for c in caps if c.id not in parents]
    mapped = {f.capability for f in g.files if f.capability}
    leaf_covered = sum(1 for c in leaves if c.id in mapped)

    files_with_sigs = sum(1 for f in g.files if f.functions)

    return {
        "parseable": True,
        "valid": ok,
        "acyclic": acyclic,
        "n_capabilities": len(caps),
        "n_files": len(g.files),
        "n_edges": len(g.edges),
        "n_functions": sum(len(f.functions) for f in g.files),
        "leaf_coverage": (leaf_covered / len(leaves)) if leaves else 1.0,
        "signature_density": (files_with_sigs / len(g.files)) if g.files else 0.0,
        "has_verify": bool(g.verify),
        "score": rpgmod.score_rpg(g),
        "n_issues": len(issues),
    }


def aggregate_metrics(rpgs: List[dict]) -> dict:
    """Aggregate per-RPG metrics across a set (e.g. a benchmark run's plans)."""
    per = [rpg_quality_metrics(r) for r in rpgs]
    parseable = [m for m in per if m.get("parseable")]
    n = len(per)
    if not parseable:
        return {"n": n, "parse_rate": 0.0}

    def _mean(key: str) -> float:
        vals = [m[key] for m in parseable if key in m]
        return sum(vals) / len(vals) if vals else 0.0

    def _rate(key: str) -> float:
        return sum(1 for m in parseable if m.get(key)) / len(parseable)

    return {
        "n": n,
        "parse_rate": len(parseable) / n,
        "valid_rate": _rate("valid"),
        "acyclic_rate": _rate("acyclic"),
        "has_verify_rate": _rate("has_verify"),
        "mean_leaf_coverage": _mean("leaf_coverage"),
        "mean_signature_density": _mean("signature_density"),
        "mean_score": _mean("score"),
        "avg_files": _mean("n_files"),
        "avg_edges": _mean("n_edges"),
        "avg_functions": _mean("n_functions"),
    }


def _load_artifacts(args: List[str]) -> List[dict]:
    out: List[dict] = []
    if args and args[0] == "--jsonl":
        for path in args[1:]:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        return out
    # Expand globs (shells that don't expand, or quoted globs).
    paths: List[str] = []
    for a in args:
        expanded = glob.glob(a)
        paths.extend(expanded if expanded else [a])
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    artifacts = _load_artifacts(argv)
    if not artifacts:
        print("no RPG artifacts found", file=sys.stderr)
        return 1
    metrics = aggregate_metrics(artifacts)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
