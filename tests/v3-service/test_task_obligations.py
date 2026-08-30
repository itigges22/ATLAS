"""Task-owned obligations, and the two bounds an adapter is held to.

The rule under test is one sentence: what a run OWES is the task's to say,
what a verifier CAN SHOW is the adapter's, and a record may only claim the
intersection. Everything here is a way of getting that wrong.

Nothing in this file authorizes a delivery. contract records are inert data
until an authorization consumer exists, and none does in this build.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import adapters as A  # noqa: E402
import contract as C  # noqa: E402
import obligations as O  # noqa: E402


def _record(adapter, obligations_, accepted=True, probe=None):
    return A.contract_record(
        adapter=adapter, accepted=accepted, probe=probe,
        contract_id="c.v1", contract_version="1", artifact_scope="solve.py",
        evaluation_context_hash="ctx", candidate_content_hash="hash",
        task_obligations=obligations_)


# --- the taxonomy is closed and its floors are stated ------------------------

def test_every_kind_has_exactly_one_answer_about_its_floor():
    """A kind is fixed-floor, baseline-floored, or unsatisfiable. Never two."""
    for kind in O.KINDS:
        if kind in O._UNSATISFIABLE_KINDS:
            with pytest.raises(O.ObligationError):
                O.required_strength(kind)
            continue
        if kind in O._DYNAMIC_STRENGTH_KINDS:
            # Needs the baseline's own strength and refuses to invent one.
            with pytest.raises(O.ObligationError):
                O.required_strength(kind)
            assert O.required_strength(kind, C.BEHAVIORAL) == C.BEHAVIORAL
            continue
        assert O.required_strength(kind) in C.STRENGTH_ORDER
        # A fixed floor cannot be raised by handing it a baseline.
        with pytest.raises(O.ObligationError):
            O.required_strength(kind, C.ORACLE)


def test_a_declared_command_is_behavioral_and_not_an_oracle():
    """Exit zero says the command ran and succeeded against these bytes. It
    does not say an answer was checked against a reference, and calling an
    arbitrary exit-zero command an oracle is how "it ran" became "it is right".

    How strongly it counts is the kind the CLIENT typed for that exact command,
    so the floor comes from the obligation rather than from the kind -- and
    oracle is not among the kinds a client may type.
    """
    for kind in (C.SYNTAX, C.RUNTIME, C.BEHAVIORAL):
        assert O.required_strength(O.KIND_DECLARED_COMMAND, kind) == kind
    with pytest.raises(O.ObligationError):
        O.required_strength(O.KIND_DECLARED_COMMAND, C.ORACLE)
    with pytest.raises(O.ObligationError):
        O.required_strength(O.KIND_DECLARED_COMMAND)
    assert O.required_strength(O.KIND_DECLARED_EXAMPLE) == C.ORACLE


def test_an_unknown_kind_or_strength_fails_closed():
    with pytest.raises(O.ObligationError):
        O.obligation(kind="something_new", subject="x")
    with pytest.raises(O.ObligationError):
        O.obligation_id("something_new", "x")
    with pytest.raises(O.ObligationError):
        O.obligation(kind=O.KIND_BASELINE_PRESERVED, subject="x",
                     baseline_strength="very_strong")
    forged = dict(O.obligation(kind=O.KIND_SYNTACTIC_VALIDITY, subject="x"))
    forged["required_strength"] = "very_strong"
    with pytest.raises(O.ObligationError):
        O.validate([forged])
    with pytest.raises(O.ObligationError):
        _record(A.ADAPTER_PYTHON_COMPILE, [forged])


def test_an_obligation_never_carries_its_subject_text():
    """A declared command is a subject, and a command string in a log is a
    content leak. The rule is uniform so no exception can leak one."""
    secret = "pytest --token=hunter2 -q"
    o = O.obligation(kind=O.KIND_DECLARED_COMMAND, subject=secret,
                     baseline_strength=C.RUNTIME)
    rendered = repr(o)
    assert secret not in rendered
    assert "hunter2" not in rendered
    assert o["id"].startswith(O.KIND_DECLARED_COMMAND + ":")


def test_obligation_ids_are_deterministic_and_kind_scoped():
    a = O.obligation_id(O.KIND_ARTIFACT_EXISTS, "solve.py")
    b = O.obligation_id(O.KIND_ARTIFACT_EXISTS, "solve.py")
    c = O.obligation_id(O.KIND_SYNTACTIC_VALIDITY, "solve.py")
    assert a == b
    assert a != c, "one subject under two kinds is two obligations"


def test_the_closure_floor_is_the_strongest_required_obligation():
    obs = [O.obligation(kind=O.KIND_ARTIFACT_EXISTS, subject="solve.py"),
           O.obligation(kind=O.KIND_DECLARED_COMMAND, subject="pytest -q",
                        baseline_strength=C.BEHAVIORAL)]
    assert O.closure_floor(obs) == C.BEHAVIORAL


def test_an_unsupported_required_obligation_puts_closure_out_of_reach():
    """Not absent: unreachable. A task that owes something nothing measured
    must not close because the thing was never scored."""
    obs = [O.obligation(kind=O.KIND_UNSUPPORTED, subject="a thing we cannot name")]
    assert O.closure_floor(obs) == C.STRENGTH_ORDER[-1]
    rec = _record(A.ADAPTER_PYTHON_COMPILE, obs)
    assert rec["closure_eligible"] is False


# --- capability is an upper bound, never a substitute ------------------------

def test_a_behavioral_obligation_presented_to_a_syntax_only_adapter_stays_incomplete():
    command = O.obligation(kind=O.KIND_DECLARED_COMMAND, subject="pytest -q",
                           baseline_strength=C.BEHAVIORAL)
    rec = _record(A.ADAPTER_PYTHON_COMPILE, [command])
    assert rec["requirements_complete"] is False
    assert rec["closure_eligible"] is False
    assert rec["missing_required"] == [command["id"]]
    assert rec["observations"][command["id"]]["status"] == C.NOT_APPLICABLE
    assert command["id"] not in rec["capabilities"]


def test_a_syntax_obligation_with_a_real_syntax_evaluator_records_both():
    """The matching case: capability owns the kind, and the run reached the
    floor. Capability and observation are both recorded, so a reader can tell
    a measured pass from an unmeasured one."""
    syntax = O.obligation(kind=O.KIND_SYNTACTIC_VALIDITY, subject="solve.py")
    rec = _record(A.ADAPTER_PYTHON_COMPILE, [syntax])
    assert syntax["id"] in rec["capabilities"]
    assert rec["observations"][syntax["id"]]["status"] == C.DEMONSTRATED
    assert rec["requirements_complete"] is True
    assert rec["evidence_strength"] == C.SYNTAX


def test_a_syntax_evaluator_that_rejected_the_artifact_demonstrates_nothing():
    syntax = O.obligation(kind=O.KIND_SYNTACTIC_VALIDITY, subject="solve.py")
    rec = _record(A.ADAPTER_PYTHON_COMPILE, [syntax], accepted=False)
    assert rec["requirements_complete"] is False
    assert rec["execution_ok"] is False
    assert rec["closure_eligible"] is False


def test_behavioral_capability_does_not_satisfy_an_unrelated_command():
    """A browser probe that ran the artifact cleanly owns structural validity
    and nothing else. The client's command is a different obligation and the
    probe never executed it."""
    probe = {"supported": True, "runtime_clean": True,
             **{c: True for c in A.BROWSER_REQUIRED + A.BROWSER_OPTIONAL}}
    command = O.obligation(kind=O.KIND_DECLARED_COMMAND, subject="npm test",
                           baseline_strength=C.BEHAVIORAL)
    syntax = O.obligation(kind=O.KIND_SYNTACTIC_VALIDITY, subject="game.js")
    rec = _record(A.ADAPTER_BROWSER_CANVAS_JS, [syntax, command], probe=probe)
    assert rec["evidence_strength"] == C.BEHAVIORAL
    assert rec["observations"][syntax["id"]]["status"] == C.DEMONSTRATED
    assert rec["observations"][command["id"]]["status"] == C.NOT_APPLICABLE
    assert rec["requirements_complete"] is False


def test_an_unsupported_adapter_cannot_become_closure_eligible():
    """Unsupported is unverifiable, never failed and never vacuously complete
    -- including when the task owes nothing this build can name."""
    for adapter in (A.ADAPTER_UNSUPPORTED,
                    A.ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED):
        syntax = O.obligation(kind=O.KIND_SYNTACTIC_VALIDITY, subject="a.py")
        rec = _record(adapter, [syntax])
        assert rec["supported"] is False
        assert rec["closure_eligible"] is False
        expected = {"contract_id": rec["contract_id"],
                    "contract_version": rec["contract_version"],
                    "artifact_scope": rec["artifact_scope"],
                    "evaluation_context_hash": rec["evaluation_context_hash"],
                    "adapter_id": rec["adapter_id"],
                    "adapter_version": rec["adapter_version"]}
        sel = C.select([rec], expected)
        assert sel["best_record"] is None
        assert sel["ineligible"] == [rec]


def test_an_oracle_obligation_is_out_of_reach_of_a_compile():
    example = O.obligation(kind=O.KIND_DECLARED_EXAMPLE, subject="case-1")
    rec = _record(A.ADAPTER_PYTHON_COMPILE, [example])
    assert rec["observations"][example["id"]]["status"] == C.NOT_APPLICABLE
    assert rec["closure_eligible"] is False


def test_the_oracle_adapter_owns_declared_examples():
    example = O.obligation(kind=O.KIND_DECLARED_EXAMPLE, subject="case-1")
    rec = _record(A.ADAPTER_ALGORITHMIC_IO, [example])
    assert rec["evidence_strength"] == C.ORACLE
    assert rec["observations"][example["id"]]["status"] == C.DEMONSTRATED
    assert rec["closure_eligible"] is True


def test_the_oracle_route_needs_a_trusted_declared_case_source():
    """algorithmic_io is reachable only under has_io_oracle, and
    pipeline._trusted_oracle grants that only when EVERY case declares trusted
    provenance. A model-generated suite routes to the compile instead."""
    assert A.select_adapter("solve.py", "print(1)", False) == A.ADAPTER_PYTHON_COMPILE
    assert A.select_adapter("solve.py", "print(1)", True) == A.ADAPTER_ALGORITHMIC_IO
    src = (Path(__file__).resolve().parents[2] / "v3-service" / "pipeline.py").read_text()
    assert "_has_oracle = _trusted_oracle(self_tests)" in src


# --- an existing baseline is not replaced on weaker evidence -----------------

def test_a_baseline_obligation_takes_the_strength_it_already_has():
    behavioural_baseline = O.obligation(
        kind=O.KIND_BASELINE_PRESERVED, subject="solve.py",
        baseline_strength=C.BEHAVIORAL)
    assert behavioural_baseline["required_strength"] == C.BEHAVIORAL
    rec = _record(A.ADAPTER_PYTHON_COMPILE, [behavioural_baseline])
    # A compile is not evidence that behaviour survived.
    assert rec["closure_eligible"] is False
    assert rec["observations"][behavioural_baseline["id"]]["status"] == C.NOT_APPLICABLE


def test_a_syntax_baseline_may_be_preserved_by_a_syntax_verifier():
    syntax_baseline = O.obligation(
        kind=O.KIND_BASELINE_PRESERVED, subject="solve.py",
        baseline_strength=C.SYNTAX)
    rec = _record(A.ADAPTER_PYTHON_COMPILE, [syntax_baseline])
    assert rec["observations"][syntax_baseline["id"]]["status"] == C.NOT_APPLICABLE, (
        "preservation is a kind no verifier here owns; owning syntactic "
        "validity is not owning the claim that a baseline survived")
    assert rec["closure_eligible"] is False


# --- legacy traffic gains no structured authority ----------------------------

def test_an_absent_obligation_set_is_not_an_empty_one():
    """A caller that stated no knowledge gets the adapter's own criteria and no
    task authority. An empty list is a caller saying it owes nothing, which is
    a different claim -- and vacuous completeness is exactly what must not be
    reachable by omission."""
    legacy = A.contract_record(
        adapter=A.ADAPTER_PYTHON_COMPILE, accepted=True, probe=None,
        contract_id="c.v1", contract_version="1", artifact_scope="s",
        evaluation_context_hash="ctx", candidate_content_hash="h")
    assert legacy["requirements"], "legacy records are measured against something"
    assert legacy["closure_eligible"] is False, (
        "a compile cannot close a task whose floor it never learned")


def test_legacy_and_structured_records_of_the_same_run_are_distinguishable():
    syntax = O.obligation(kind=O.KIND_SYNTACTIC_VALIDITY, subject="solve.py")
    legacy = A.contract_record(
        adapter=A.ADAPTER_PYTHON_COMPILE, accepted=True, probe=None,
        contract_id="c.v1", contract_version="1", artifact_scope="s",
        evaluation_context_hash="ctx", candidate_content_hash="h")
    structured = A.contract_record(
        adapter=A.ADAPTER_PYTHON_COMPILE, accepted=True, probe=None,
        contract_id="c.v1", contract_version="1", artifact_scope="s",
        evaluation_context_hash="ctx", candidate_content_hash="h",
        task_obligations=[syntax])
    assert [r["id"] for r in legacy["requirements"]] != \
           [r["id"] for r in structured["requirements"]]


# --- the correction did not change what legacy traffic selects ---------------

def test_correcting_the_registry_left_legacy_selection_where_it_was():
    """Every SELECTABLE python_compile record is an accepted one -- a rejected
    candidate is EXEC_ERROR and never enters the pool -- so the corrected
    requirement moves every comparable record by the same amount and the
    winner is unchanged. Closure stays out of reach: the adapter's floor is
    behavioural and a compile demonstrates syntax.
    """
    recs = []
    for h in ("h1", "h2", "h3"):
        recs.append(A.contract_record(
            adapter=A.ADAPTER_PYTHON_COMPILE, accepted=True, probe=None,
            contract_id="c.v1", contract_version="1", artifact_scope="s",
            evaluation_context_hash="ctx", candidate_content_hash=h))
    expected = {"contract_id": "c.v1", "contract_version": "1",
                "artifact_scope": "s", "evaluation_context_hash": "ctx",
                "adapter_id": A.ADAPTER_PYTHON_COMPILE,
                "adapter_version": A.LIVE_ADAPTER_VERSION}
    sel = C.select(recs, expected)
    assert sel["best_record"] is recs[0], "stable order among exact ties"
    assert sel["verified_winner"] is None
    assert sel["closure_eligible"] is False


def test_a_rejected_candidate_is_ineligible_rather_than_low_scoring():
    rec = A.contract_record(
        adapter=A.ADAPTER_PYTHON_COMPILE, accepted=False, probe=None,
        contract_id="c.v1", contract_version="1", artifact_scope="s",
        evaluation_context_hash="ctx", candidate_content_hash="h")
    assert rec["execution_ok"] is False
    expected = {"contract_id": "c.v1", "contract_version": "1",
                "artifact_scope": "s", "evaluation_context_hash": "ctx",
                "adapter_id": A.ADAPTER_PYTHON_COMPILE,
                "adapter_version": A.LIVE_ADAPTER_VERSION}
    sel = C.select([rec], expected)
    assert sel["ineligible"] == [rec]
    assert sel["best_record"] is None


def test_the_registry_invariant_still_rejects_an_unevaluable_declaration():
    with pytest.raises(A.AdapterRegistryError):
        A.check_registry({"x": {"support": A.SUPPORT_ALWAYS,
                                "capabilities": ["c"], "requirements": [("c", True)],
                                "evaluators": {}, "unmeasurable": [],
                                "obligation_kinds": []}})
    with pytest.raises(A.AdapterRegistryError):
        A.check_registry({"x": {"support": A.SUPPORT_ALWAYS,
                                "capabilities": [], "requirements": [],
                                "evaluators": {}, "unmeasurable": [],
                                "obligation_kinds": ["not_a_kind"]}})
    with pytest.raises(A.AdapterRegistryError):
        A.check_registry({"x": {"support": A.SUPPORT_NEVER,
                                "capabilities": [], "requirements": [],
                                "evaluators": {}, "unmeasurable": [],
                                "obligation_kinds": [O.KIND_SYNTACTIC_VALIDITY]}})
    with pytest.raises(A.AdapterRegistryError):
        A.check_registry({"x": {"support": A.SUPPORT_ALWAYS,
                                "capabilities": [], "requirements": [],
                                "evaluators": {}, "unmeasurable": []}})
