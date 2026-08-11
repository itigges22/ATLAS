"""Executable genericity + policy proofs for the evidence contract.

A synthetic adapter with arbitrary criteria (alpha/beta/gamma) drives the
REAL policy. If a browser or game assumption were baked in, these fail —
which a grep over identifiers could never establish.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import contract as C  # noqa: E402

CAPS = ["alpha", "beta", "gamma"]
CTX = "ctx-abc123"
SCOPE = "synthetic.txt"

TASK = C.task_contract("synthetic.v1", "1.2.3",
                       [C.requirement("alpha"), C.requirement("beta"),
                        C.requirement("gamma", required=False, weight=0.5)])


def _rec(obs, task=TASK, strength=C.BEHAVIORAL, adapter="synth", ver="0.9.0",
         ctx=CTX, scope=SCOPE, content="c1", **kw):
    return C.build(task, adapter, ver, obs, CAPS, strength,
                   artifact_scope=scope, evaluation_context_hash=ctx,
                   candidate_content_hash=content, **kw)


D = C.observation(C.DEMONSTRATED)
R = C.observation(C.REFUTED)


def _expected(**kw):
    base = {"contract_id": "synthetic.v1", "contract_version": "1.2.3",
            "artifact_scope": SCOPE, "evaluation_context_hash": CTX,
            "calibration_id": "", "adapter_id": "synth", "adapter_version": "0.9.0"}
    base.update(kw)
    return base


# ---- 1. plurality must not choose the rubric ------------------------------

def test_wrong_contract_plurality_cannot_outvote_the_right_one():
    right = _rec({"alpha": D, "beta": D})
    other_task = C.task_contract("other.v1", "9.9.9", [C.requirement("alpha")])
    wrong = [C.build(other_task, "synth", "0.9.0", {"alpha": D}, CAPS, C.BEHAVIORAL,
                     artifact_scope=SCOPE, evaluation_context_hash=CTX,
                     candidate_content_hash=f"w{i}") for i in range(10)]
    res = C.select(wrong + [right], _expected())
    assert res["best_record"] is right, "ten misrouted candidates must not outvote one correct one"
    assert len(res["incomparable"]) == 10


def test_no_matching_record_yields_no_verified_winner():
    other_task = C.task_contract("other.v1", "1.0.0", [C.requirement("alpha")])
    foreign = C.build(other_task, "synth", "0.9.0", {"alpha": D}, CAPS, C.BEHAVIORAL,
                      artifact_scope=SCOPE, evaluation_context_hash=CTX)
    res = C.select([foreign], _expected())
    assert res["best_record"] is None and res["incomparable"] == [foreign]


# ---- 2. comparison identity ----------------------------------------------

def test_empty_identity_fields_never_make_records_comparable():
    a = _rec({"alpha": D}, ctx="", scope="")
    assert not C.comparable(a, a), "empty identity must not compare, even to itself"
    with pytest.raises(C.ContractError):
        C.select([a], _expected(evaluation_context_hash=""))


def test_candidate_hashes_differ_while_context_matches():
    a = _rec({"alpha": D, "beta": D}, content="cand-a")
    b = _rec({"alpha": D}, content="cand-b")
    assert a["candidate_content_hash"] != b["candidate_content_hash"]
    assert C.comparable(a, b), "candidates must stay comparable within one context"


def test_different_adapter_versions_are_incomparable_without_calibration():
    a = _rec({"alpha": D, "beta": D}, ver="0.9.0")
    b = _rec({"alpha": D, "beta": D}, ver="1.0.0")
    assert not C.comparable(a, b)
    ca = _rec({"alpha": D, "beta": D}, ver="0.9.0", calibration_id="cal-1")
    cb = _rec({"alpha": D, "beta": D}, ver="1.0.0", calibration_id="cal-1")
    assert C.comparable(ca, cb), "an explicit shared calibration permits comparison"


# ---- 3. failed executions -------------------------------------------------

def test_a_timed_out_run_is_never_complete_or_a_winner():
    dead = _rec({"alpha": D, "beta": D, "gamma": D}, execution_status=C.EXEC_TIMEOUT)
    assert dead["requirements_complete"] is False
    assert dead["closure_eligible"] is False
    healthy = _rec({"alpha": D})
    res = C.select([dead, healthy], _expected())
    assert res["best_record"] is healthy, "a dead run must not outrank a healthy one"
    assert dead in res["ineligible"]
    assert dead["observations"]["alpha"]["status"] == C.DEMONSTRATED, \
        "partial observations are preserved for diagnostics"


# ---- 4. required coverage vs optional quality -----------------------------

def test_required_coverage_outranks_heavily_weighted_optional_quality():
    task = C.task_contract("synthetic.v1", "1.2.3",
                           [C.requirement("alpha"), C.requirement("beta"),
                            C.requirement("gamma", required=False, weight=9.0)])
    mostly_required = _rec({"alpha": D, "beta": D, "gamma": R}, task=task)
    all_optional = _rec({"alpha": R, "beta": R, "gamma": D}, task=task)
    assert all_optional["overall_quality_score"] > mostly_required["overall_quality_score"]
    assert C.rank_key(mostly_required) > C.rank_key(all_optional)


# ---- 5. closure floor belongs to the contract -----------------------------

def test_syntax_cannot_close_a_behavioural_contract():
    rec = _rec({"alpha": D, "beta": D, "gamma": D}, strength=C.SYNTAX)
    assert rec["requirements_complete"] is True
    assert rec["closure_eligible"] is False


def test_syntax_closes_a_syntax_scoped_contract():
    schema_task = C.task_contract("json_schema.v1", "1.0.0",
                                  [C.requirement("alpha")],
                                  minimum_closure_strength=C.SYNTAX)
    rec = _rec({"alpha": D}, task=schema_task, strength=C.SYNTAX)
    assert rec["closure_eligible"] is True, \
        "a universal behavioural floor is not prompt-agnostic"


def test_an_oracle_floor_rejects_merely_behavioural_evidence():
    algo = C.task_contract("algorithmic.v1", "1.0.0", [C.requirement("alpha")],
                           minimum_closure_strength=C.ORACLE)
    assert _rec({"alpha": D}, task=algo, strength=C.BEHAVIORAL)["closure_eligible"] is False
    assert _rec({"alpha": D}, task=algo, strength=C.ORACLE)["closure_eligible"] is True


# ---- 6. validation + materialisation --------------------------------------

@pytest.mark.parametrize("bad", [
    {"reqs": [C.requirement("a"), C.requirement("a")]},
    {"reqs": [C.requirement("")]},
    {"reqs": [C.requirement("a", weight=float("inf"))]},
    {"reqs": [C.requirement("a", weight=-1)]},
])
def test_malformed_requirements_are_rejected(bad):
    task = C.task_contract("x.v1", "1", bad["reqs"])
    with pytest.raises(C.ContractError):
        C.build(task, "synth", "0.9", {}, CAPS, C.BEHAVIORAL,
                artifact_scope=SCOPE, evaluation_context_hash=CTX)


@pytest.mark.parametrize("obs", [
    {"alpha": {"status": "made_up", "confidence": 1.0}},
    {"alpha": C.observation(C.DEMONSTRATED, confidence=1.5)},
    {"omega": C.observation(C.DEMONSTRATED)},
    {"omega": C.observation(C.REFUTED)},
])
def test_malformed_or_overreaching_observations_are_rejected(obs):
    with pytest.raises(C.ContractError):
        _rec(obs)


def test_unknown_statuses_are_rejected():
    with pytest.raises(C.ContractError):
        _rec({"alpha": D}, execution_status="exploded")
    with pytest.raises(C.ContractError):
        _rec({"alpha": D}, strength="vibes")


def test_derived_observations_are_materialised_for_telemetry():
    task = C.task_contract("synthetic.v1", "1.2.3",
                           [C.requirement("alpha"), C.requirement("delta")])
    rec = _rec({"alpha": D}, task=task)
    back = json.loads(json.dumps(rec))
    assert back["observations"]["delta"]["status"] == C.NOT_APPLICABLE, \
        "telemetry must be able to explain why completion failed"
    assert "delta" in back["missing_required"]


def test_unobserved_survives_serialization():
    task = C.task_contract("synthetic.v1", "1.2.3",
                           [C.requirement("alpha"), C.requirement("beta")])
    rec = json.loads(json.dumps(_rec({"alpha": D}, task=task)))
    assert rec["observations"]["beta"]["status"] == C.UNOBSERVED
    assert rec["schema_version"] == C.SCHEMA_VERSION


# ---- 7. tie-break ---------------------------------------------------------

def test_exact_evidence_ties_reach_the_supplied_tie_break():
    a = _rec({"alpha": D, "beta": D}, content="a")
    b = _rec({"alpha": D, "beta": D}, content="b")
    res = C.select([a, b], _expected(),
                   tie_break=lambda r: -{"a": 1, "b": 0}[r["candidate_content_hash"]])
    assert len(res["tied"]) == 2, "the pipeline must see the tie"
    assert res["best_record"] is b, "the supplied lens tie-break decides, not list order"


def test_without_a_tie_break_selection_is_stable():
    a = _rec({"alpha": D, "beta": D}, content="a")
    b = _rec({"alpha": D, "beta": D}, content="b")
    assert C.select([a, b], _expected())["best_record"] is a


# ---- genericity -----------------------------------------------------------

def test_the_policy_functions_carry_no_domain_vocabulary():
    import ast as _ast
    tree = _ast.parse(Path(C.__file__).read_text())
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            body = [n for n in node.body
                    if not (isinstance(n, _ast.Expr) and isinstance(n.value, _ast.Constant)
                            and isinstance(n.value.value, str))]
            code = "\n".join(_ast.dump(n) for n in body).lower()
            for word in ("canvas", "snake", "collision", "food", "keydown", "browser"):
                assert word not in code, f"'{word}' leaked into {node.name}()"


# ---- all-failed pools -----------------------------------------------------

def _dead(status, content="d"):
    return _rec({"alpha": D, "beta": D, "gamma": D},
                execution_status=status, content=content)


@pytest.mark.parametrize("status", [C.EXEC_TIMEOUT, C.EXEC_CRASH, C.EXEC_ERROR])
def test_an_all_failed_pool_has_no_winner(status):
    """Ranking alone let the best corpse win when every record had died."""
    res = C.select([_dead(status, "a"), _dead(status, "b")], _expected())
    assert res["best_record"] is None
    assert res["verified_winner"] is None
    assert len(res["ineligible"]) == 2
    assert "failed to execute" in res["selection_reason"]


def test_unsupported_plus_timeout_yields_no_winner():
    unsup = _rec({"alpha": D}, supported=False, content="u")
    res = C.select([unsup, _dead(C.EXEC_TIMEOUT, "t")], _expected())
    assert res["best_record"] is None and len(res["ineligible"]) == 2


def test_a_healthy_partial_beats_a_failed_complete_looking_record():
    partial = _rec({"alpha": D}, content="p")          # healthy, incomplete
    dead = _dead(C.EXEC_TIMEOUT, "d")                  # "complete" but dead
    res = C.select([dead, partial], _expected())
    assert res["best_record"] is partial
    assert res["closure_eligible"] is False, "best evidence is not the same as verified"
    assert res["verified_winner"] is None


def test_a_healthy_partial_is_never_called_a_verified_winner():
    res = C.select([_rec({"alpha": D}, content="p")], _expected())
    assert res["best_record"] is not None
    assert res["verified_winner"] is None, \
        "an incomplete record must not become a top-level verified pass"


def test_a_complete_healthy_record_is_a_verified_winner():
    res = C.select([_rec({"alpha": D, "beta": D, "gamma": D})], _expected())
    assert res["verified_winner"] is res["best_record"]
    assert res["closure_eligible"] is True


def test_failed_records_keep_their_observations_for_diagnostics():
    dead = _dead(C.EXEC_CRASH)
    res = C.select([dead], _expected())
    assert res["ineligible"][0]["observations"]["alpha"]["status"] == C.DEMONSTRATED


# ---- complete identity validation -----------------------------------------

@pytest.mark.parametrize("missing", ["contract_id", "contract_version",
                                     "artifact_scope", "evaluation_context_hash"])
def test_expected_identity_must_be_complete(missing):
    with pytest.raises(C.ContractError):
        C.select([], _expected(**{missing: ""}))


def test_calibration_without_adapter_identity_is_still_valid():
    exp = _expected(adapter_id="", adapter_version="", calibration_id="cal-9")
    C.select([], exp)          # must not raise


def test_no_adapter_identity_and_no_calibration_is_rejected():
    with pytest.raises(C.ContractError):
        C.select([], _expected(adapter_id="", adapter_version="", calibration_id=""))


@pytest.mark.parametrize("field", ["contract_version", "adapter_id", "adapter_version"])
def test_records_with_incomplete_identity_are_incomparable(field):
    rec = _rec({"alpha": D, "beta": D})
    rec[field] = ""
    res = C.select([rec], _expected())
    assert res["best_record"] is None
    assert rec in res["incomparable"]
