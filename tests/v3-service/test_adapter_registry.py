"""An adapter may declare only what it can actually evaluate.

Three declarations used to be derived rather than made:

  supported     inferred from set membership, so an adapter became
                "supported" for an artifact class by not being listed anywhere;
  requirements  built FROM capabilities -- literally
                `[requirement(c) for c in _CAPABILITIES[adapter]]` -- so an
                adapter invented the task's obligations out of its own reach,
                which is the one thing contract.py's docstring says it must
                never do;
  evaluators    implicit in a branch of _observations, so "this adapter can
                measure X" and "something here computes X" were separate
                facts nothing compared.

The sealed Stage-A acquisition is what that cost: 100 of 103 candidate
evaluations ran under `python_compile`, which declares four BROWSER criteria
it cannot measure, and every one recorded
`missing_required: ["temporal_progress", "input_causality"]` with
`capabilities: []`. Closure was unreachable for every Python candidate in the
run -- not because the corpus lacked an oracle, but because no Python route
can reach it at all.

These tests pin the registry as a structure. They do NOT change what any
adapter reports: correcting the CONTENT of python_compile's declaration
necessarily moves records into contract.select's ineligible bucket, which
changes selection, and this commit must be authority-inert.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import adapters as A
import obligations as O  # noqa: E402
import contract as C  # noqa: E402

ALL_EIGHT = [
    A.ADAPTER_ALGORITHMIC_IO, A.ADAPTER_CSS_SYNTAX,
    A.ADAPTER_BROWSER_CANVAS_JS, A.ADAPTER_BROWSER_INLINE_SCRIPT,
    A.ADAPTER_PYTHON_COMPILE, A.ADAPTER_JAVASCRIPT_COMPILE,
    A.ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED, A.ADAPTER_UNSUPPORTED,
]


# --- every adapter is declared, for all three facts -------------------------

def test_every_adapter_is_registered():
    assert set(A.ALL_ADAPTERS) == set(ALL_EIGHT), (
        "the registry and the adapter identities disagree")


@pytest.mark.parametrize("adapter", ALL_EIGHT)
def test_every_adapter_declares_support_explicitly(adapter):
    assert adapter in A.SUPPORT_DECLARATION, (
        f"{adapter} does not declare whether it supports an artifact; "
        "membership inference is what let an adapter be supported by omission")
    assert A.SUPPORT_DECLARATION[adapter] in A.SUPPORT_KINDS


@pytest.mark.parametrize("adapter", ALL_EIGHT)
def test_every_declared_capability_has_an_evaluator(adapter):
    caps = set(A._capabilities(adapter))
    evals = set(A.evaluators_for(adapter))
    missing = sorted(caps - evals)
    assert not missing, (
        f"{adapter} claims it can measure {missing} with nothing that computes them")


@pytest.mark.parametrize("adapter", ALL_EIGHT)
def test_no_evaluator_without_a_declared_capability(adapter):
    caps = set(A._capabilities(adapter))
    extra = sorted(set(A.evaluators_for(adapter)) - caps)
    assert not extra, (
        f"{adapter} evaluates {extra} without declaring it measurable")


@pytest.mark.parametrize("adapter", ALL_EIGHT)
def test_requirements_are_declared_not_derived_from_capability(adapter):
    """An obligation is the TASK's. An adapter that reads its own reach and
    calls the result a requirement has decided what the user needed."""
    assert adapter in A.REQUIREMENT_DECLARATION, (
        f"{adapter} has no declared requirement set")


@pytest.mark.parametrize("adapter", ALL_EIGHT)
def test_every_requirement_is_measurable_or_registered_unmeasurable(adapter):
    caps = set(A._capabilities(adapter))
    declared = {r["id"] for r in A._requirements(adapter)}
    quarantined = set(A.unmeasurable_requirements(adapter))
    orphan = sorted(declared - caps - quarantined)
    assert not orphan, (
        f"{adapter} requires {orphan} with no evaluator and no registered "
        "reason; a requirement it cannot measure must be declared as such")


@pytest.mark.parametrize("adapter", ALL_EIGHT)
def test_a_criterion_is_never_both_measurable_and_unmeasurable(adapter):
    both = sorted(set(A._capabilities(adapter)) & set(A.unmeasurable_requirements(adapter)))
    assert not both, f"{adapter} declares {both} as both measurable and not"


# --- the quarantine is lifted, and what replaced it --------------------------

def test_no_adapter_demands_a_criterion_it_cannot_measure():
    """The defect the sealed run paid for is gone at the source.

    python_compile required four browser criteria it cannot observe, so 100 of
    103 candidate evaluations carried missing_required
    ["temporal_progress","input_causality"] and no Python candidate could reach
    closure by any route. It required them because an adapter used to declare
    the TASK's obligations; obligations.py owns that now, so an adapter has
    nothing to over-declare.
    """
    for adapter in A.ALL_ADAPTERS:
        assert not A.unmeasurable_requirements(adapter), (
            f"{adapter} still quarantines "
            f"{sorted(A.unmeasurable_requirements(adapter))}")
        declared = {cid for cid, _ in A.REQUIREMENT_DECLARATION[adapter]}
        assert declared <= set(A._capabilities(adapter)), (
            f"{adapter} requires {sorted(declared - set(A._capabilities(adapter)))} "
            "with nothing to measure it")


def test_the_corrected_obligation_capability_matrix():
    """What each verifier may now speak for, stated once.

    A compile owns structural validity and nothing above it. The browser probe
    runs the artifact, so it owns the same and no more: behavioural reach is
    not permission to answer an unrelated command obligation. algorithmic_io
    owns declared examples, and select_adapter reaches it only under a trusted
    declared case source. The two unsupported adapters own nothing.
    """
    assert A.obligation_capabilities(A.ADAPTER_PYTHON_COMPILE) == [O.KIND_SYNTACTIC_VALIDITY]
    assert A.obligation_capabilities(A.ADAPTER_JAVASCRIPT_COMPILE) == [O.KIND_SYNTACTIC_VALIDITY]
    assert A.obligation_capabilities(A.ADAPTER_CSS_SYNTAX) == [O.KIND_SYNTACTIC_VALIDITY]
    assert A.obligation_capabilities(A.ADAPTER_BROWSER_CANVAS_JS) == [O.KIND_SYNTACTIC_VALIDITY]
    assert A.obligation_capabilities(A.ADAPTER_BROWSER_INLINE_SCRIPT) == [O.KIND_SYNTACTIC_VALIDITY]
    assert A.obligation_capabilities(A.ADAPTER_ALGORITHMIC_IO) == [
        O.KIND_SYNTACTIC_VALIDITY, O.KIND_DECLARED_EXAMPLE]
    assert A.obligation_capabilities(A.ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED) == []
    assert A.obligation_capabilities(A.ADAPTER_UNSUPPORTED) == []


def test_no_adapter_claims_a_kind_nothing_can_evaluate():
    """unsupported is a real answer, never an evaluable one."""
    for adapter in A.ALL_ADAPTERS:
        assert O.KIND_UNSUPPORTED not in A.obligation_capabilities(adapter)


def test_a_behavioral_obligation_on_a_syntax_only_adapter_is_never_complete():
    """The rule this pins: capability is an upper bound, never a substitute for
    the obligation. A syntax-only verifier facing a behavioural obligation is
    unverifiable -- it is NOT syntax-authorized.

    It is pinned against a real declared-command obligation now. It used to
    pass off the quarantine: python_compile happened to require browser
    criteria, so every record was incomplete for a reason that had nothing to
    do with the obligation in front of it.
    """
    command = O.obligation(kind=O.KIND_DECLARED_COMMAND, subject="pytest -q")
    rec = A.contract_record(
        adapter=A.ADAPTER_PYTHON_COMPILE, accepted=True, probe=None,
        contract_id="c.v1", contract_version="1", artifact_scope="s",
        evaluation_context_hash="ctx", candidate_content_hash="h",
        task_obligations=[command])
    assert rec["evidence_strength"] == C.SYNTAX
    assert rec["requirements_complete"] is False
    assert rec["closure_eligible"] is False
    assert rec["missing_required"] == [command["id"]]
    # Unmeasured, not refuted: the compile did not observe the command fail.
    assert rec["observations"][command["id"]]["status"] == C.NOT_APPLICABLE
    assert rec["required_coverage_score"] == 0.0
    assert rec["missing_required"], "a behavioural obligation vanished"


def test_an_unsupported_adapter_is_neither_failed_nor_vacuously_complete():
    for adapter in (A.ADAPTER_UNSUPPORTED, A.ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED):
        rec = A.contract_record(
            adapter=adapter, accepted=True, probe=None,
            contract_id="c.v1", contract_version="1", artifact_scope="s",
            evaluation_context_hash="ctx", candidate_content_hash="h")
        assert rec["supported"] is False
        assert rec["requirements_complete"] is False, "vacuously complete"
        assert rec["closure_eligible"] is False
        assert rec["execution_status"] != C.EXEC_ERROR, (
            "unsupported was reported as a failure")


# --- a new adapter cannot slip through --------------------------------------

def test_an_adapter_declaring_an_unevaluable_criterion_fails_the_registry():
    with pytest.raises(A.AdapterRegistryError) as exc:
        A.check_registry({
            "fake_adapter": {
                "support": A.SUPPORT_ALWAYS,
                "capabilities": ["a_thing_nobody_computes"],
                "requirements": ["a_thing_nobody_computes"],
                "evaluators": {},          # no evaluator for the declared capability
                "unmeasurable": [],
            }})
    assert "a_thing_nobody_computes" in str(exc.value)


def test_an_adapter_requiring_something_unregistered_fails_the_registry():
    with pytest.raises(A.AdapterRegistryError):
        A.check_registry({
            "fake_adapter": {
                "support": A.SUPPORT_ALWAYS,
                "capabilities": [],
                "requirements": ["behaviour_it_cannot_see"],
                "evaluators": {},
                "unmeasurable": [],        # not quarantined either
            }})


def test_an_adapter_with_no_support_declaration_fails_the_registry():
    with pytest.raises(A.AdapterRegistryError):
        A.check_registry({
            "fake_adapter": {
                "capabilities": [], "requirements": [], "evaluators": {},
                "unmeasurable": [],
            }})


def test_an_adapter_claiming_a_criterion_both_ways_fails_the_registry():
    with pytest.raises(A.AdapterRegistryError):
        A.check_registry({
            "fake_adapter": {
                "support": A.SUPPORT_ALWAYS,
                "capabilities": ["x"],
                "requirements": ["x"],
                "evaluators": {"x": lambda accepted, probe: C.UNOBSERVED},
                "unmeasurable": ["x"],
            }})


def test_the_live_registry_satisfies_its_own_invariant():
    A.check_registry(A.REGISTRY)      # must not raise


# --- repair candidates are not a hole ---------------------------------------

def test_repair_candidates_carry_a_contract_record():
    """The sealed run captured three repair candidates with `record=None`:
    adapter None, an empty record, and no way to say what was measured."""
    import ast
    src = (Path(__file__).resolve().parents[2] / "v3-service" / "pipeline.py").read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None)
        if name != "note_candidate":
            continue
        kw = {k.arg: k.value for k in node.keywords}
        role = kw.get("role")
        rec = kw.get("record")
        role_v = getattr(role, "value", None)
        if isinstance(rec, ast.Constant) and rec.value is None:
            offenders.append((role_v, node.lineno))
    assert not offenders, (
        f"pool members captured with no contract record: {offenders}")
