"""Executable proof of genericity, replacing the source-code vocabulary grep.

A synthetic adapter with arbitrary criteria (alpha/beta/gamma) is driven
through the REAL policy. If any browser or game assumption were baked in,
these would fail — which a grep over identifiers could never establish.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import contract as C  # noqa: E402

SYNTH_CAPS = ["alpha", "beta", "gamma"]


def _synth(observations, requirements=None, strength=C.BEHAVIORAL, **kw):
    return C.build(
        contract_id="synthetic.v1", contract_version="1.2.3",
        adapter_id="synthetic_adapter", adapter_version="0.9.0",
        requirements=requirements or [C.requirement("alpha"), C.requirement("beta"),
                                      C.requirement("gamma", required=False, weight=0.5)],
        observations=observations, capabilities=SYNTH_CAPS,
        evidence_strength=strength, artifact_scope="synthetic.txt",
        project_snapshot_hash="deadbeef", **kw)


def test_coverage_and_quality_work_with_arbitrary_criteria():
    rec = _synth({"alpha": C.observation(C.DEMONSTRATED),
                  "beta": C.observation(C.DEMONSTRATED),
                  "gamma": C.observation(C.DEMONSTRATED)})
    assert rec["requirements_complete"] is True
    assert rec["quality_score"] == 1.0
    assert rec["closure_eligible"] is True
    assert not rec["missing_required"]


def test_a_missing_required_criterion_blocks_completeness():
    rec = _synth({"alpha": C.observation(C.DEMONSTRATED),
                  "beta": C.observation(C.REFUTED)})
    assert rec["requirements_complete"] is False
    assert rec["missing_required"] == ["beta"]
    assert rec["closure_eligible"] is False


def test_optional_criteria_move_quality_without_being_required():
    without = _synth({"alpha": C.observation(C.DEMONSTRATED),
                      "beta": C.observation(C.DEMONSTRATED)})
    with_opt = _synth({"alpha": C.observation(C.DEMONSTRATED),
                       "beta": C.observation(C.DEMONSTRATED),
                       "gamma": C.observation(C.DEMONSTRATED)})
    assert without["requirements_complete"] is True
    assert with_opt["quality_score"] > without["quality_score"]
    assert C.rank_key(with_opt) > C.rank_key(without)
    # Required complete but quality below threshold: closure is a POLICY call.
    assert without["closure_eligible"] is False


def test_an_unmeasurable_required_criterion_can_never_complete():
    """Inability to observe is not evidence of absence."""
    rec = C.build("synthetic.v1", "1.2.3", "synthetic_adapter", "0.9.0",
                  requirements=[C.requirement("alpha"), C.requirement("delta")],
                  observations={"alpha": C.observation(C.DEMONSTRATED)},
                  capabilities=SYNTH_CAPS,          # delta not measurable
                  evidence_strength=C.BEHAVIORAL)
    assert "delta" in rec["missing_required"]
    assert rec["requirements_complete"] is False


def test_an_adapter_cannot_claim_a_criterion_it_cannot_measure():
    rec = C.build("synthetic.v1", "1.2.3", "synthetic_adapter", "0.9.0",
                  requirements=[C.requirement("alpha")],
                  observations={"alpha": C.observation(C.DEMONSTRATED),
                                "omega": C.observation(C.DEMONSTRATED)},
                  capabilities=SYNTH_CAPS,
                  evidence_strength=C.BEHAVIORAL)
    assert rec["overclaimed"] == ["omega"]
    assert rec["requirements_complete"] is False


def test_strength_and_completeness_are_independent_dimensions():
    weak_but_complete = _synth({"alpha": C.observation(C.DEMONSTRATED),
                                "beta": C.observation(C.DEMONSTRATED),
                                "gamma": C.observation(C.DEMONSTRATED)},
                               strength=C.SYNTAX)
    assert weak_but_complete["requirements_complete"] is True
    assert weak_but_complete["quality_score"] == 1.0
    assert weak_but_complete["closure_eligible"] is False, \
        "syntax-strength verification must never close, however complete it claims to be"


def test_two_different_contracts_are_not_score_compared():
    canvas = _synth({"alpha": C.observation(C.DEMONSTRATED)})
    api = C.build("http_api.v1", "1.0.0", "api_adapter", "0.1.0",
                  requirements=[C.requirement("route_reachable")],
                  observations={"route_reachable": C.observation(C.DEMONSTRATED)},
                  capabilities=["route_reachable", "status_code", "schema_match"],
                  evidence_strength=C.BEHAVIORAL, artifact_scope="api.py",
                  project_snapshot_hash="deadbeef")
    assert not C.comparable(canvas, api)
    winner, incomparable = C.select([canvas, api])
    assert winner in (canvas, api)
    assert incomparable, "the other rubric must be reported, not silently ranked"


def test_unsupported_records_never_win():
    good = _synth({"alpha": C.observation(C.DEMONSTRATED),
                   "beta": C.observation(C.DEMONSTRATED)})
    unsup = _synth({}, supported=False)
    winner, _ = C.select([unsup, good])
    assert winner is good
    assert C.select([unsup])[0] is None, "an all-unsupported pool has no verified winner"


def test_contract_and_adapter_versions_survive_serialization():
    rec = _synth({"alpha": C.observation(C.DEMONSTRATED)})
    back = json.loads(json.dumps(rec))
    for field in ("schema_version", "contract_id", "contract_version",
                  "adapter_id", "adapter_version", "artifact_scope",
                  "project_snapshot_hash", "quality_score",
                  "requirements_complete", "closure_eligible"):
        assert field in back, field
    assert back["schema_version"] == C.SCHEMA_VERSION


def test_the_policy_module_imports_nothing_domain_specific():
    import ast as _ast
    tree = _ast.parse(Path(C.__file__).read_text())
    # Compare CODE, not commentary: the docstrings deliberately name the
    # domains whose coupling was removed, and a check that cannot tell prose
    # from logic is not a check — a lesson this repo has learned twice.
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            stripped = [n for n in node.body
                        if not (isinstance(n, _ast.Expr)
                                and isinstance(n.value, _ast.Constant)
                                and isinstance(n.value.value, str))]
            code = "\n".join(_ast.dump(n) for n in stripped).lower()
            for word in ("canvas", "snake", "collision", "food", "keydown", "browser"):
                assert word not in code, f"'{word}' leaked into {node.name}()"
