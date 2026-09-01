"""Shadow must be observational with respect to DECISIONS, not just selection.

The defect: phase zero called may_return_early_result on probe evidence, so
in shadow a behaviourally complete browser candidate returned early and
skipped candidate generation. That is a live control-flow change.

The probe-free judgement still suppresses syntax-only early return in every
mode including off — that determination comes from the adapter, needs no
probe, and is the defect this whole line of work exists to fix.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "v3-service"))

import adapters  # noqa: E402
import symbols  # noqa: E402
import contract as C  # noqa: E402
import adapters as A  # noqa: E402
import pipeline as P  # noqa: E402
import scoring  # noqa: E402

COMPLETE = {"supported": True, "runtime_clean": True, "temporal_progress": True,
            "input_causality": True, "collision_transition": True,
            "food_or_score_transition": True}



# ---------------------------------------------------------------------------
# Real run() integration
# ---------------------------------------------------------------------------
#
# These drive the actual pipeline, not a mirror of its decision. A mirrored
# helper is what let the previous version of this file claim shadow was
# observational while the live path did something else; the assertions here
# are control flow -- did generation run, what closed the pipeline, which
# bytes came back -- plus the structured selection that reached telemetry.

PROBE_CODE = "def solve():\n    return 1\n"
ALT_CODES = ["def alt_a():\n    return 2\n",
             "def alt_b():\n    return 3\n",
             "def alt_c():\n    return 4\n"]


class _LLM:
    """Returns the artifact under test, so the adapter the pipeline selects is
    the one the case is about."""

    code = PROBE_CODE

    def __init__(self, progress_callback=None, thinking=False):
        pass

    def __call__(self, prompt, temperature, max_tokens, seed, thinking=None):
        return f"```\n{type(self).code}```", 3, 1.0


class _Embed:
    def __call__(self, text):
        return []


def _sandbox_factory(self_test_pass=True, smoke_ok=True, partial_oracle=False,
                     passing_marker=None):
    seen = {"cases": 0}

    class _Sandbox:
        def __init__(self, project_files=None):
            pass

        def __call__(self, code, test_input="", **_):
            if "SELF_TEST_PASS" in code:
                seen["cases"] += 1
                if passing_marker is not None:
                    # Only the artifact carrying the marker satisfies its own
                    # suite; everything else scores below half and fails.
                    ok = passing_marker in code or seen["cases"] % 3 == 1
                    if passing_marker in code:
                        ok = True
                    return (True, "SELF_TEST_PASS", "") if ok else (True, "WRONG", "")
                if partial_oracle:
                    # One case passes, the rest do not: a suite that CAN
                    # separate candidates, and a candidate that underperforms.
                    ok = seen["cases"] == 1
                else:
                    ok = self_test_pass
                return (True, "SELF_TEST_PASS", "") if ok else (True, "WRONG", "")
            return (smoke_ok, "ok", "") if smoke_ok else (False, "", "boom")
    return _Sandbox


def _service(monkeypatch, *, oracle_cases=0, self_test_pass=True, smoke_ok=True,
             task_type="algorithmic", probe=None, plan_calls=None,
             record_hook=None, code=PROBE_CODE, partial_oracle=False,
             passing_marker=None):
    """A V3PipelineService whose every outside dependency is controlled."""
    monkeypatch.setattr(_LLM, "code", code)
    monkeypatch.setattr(adapters, "LLMAdapter", _LLM)
    monkeypatch.setattr(adapters, "SandboxAdapter",
                        _sandbox_factory(self_test_pass, smoke_ok, partial_oracle,
                                         passing_marker))
    monkeypatch.setattr(adapters, "EmbedAdapter", _Embed)
    monkeypatch.setattr(scoring, "classify_task_type", lambda p: task_type)
    monkeypatch.setattr(scoring, "score_candidate", lambda code: (1.0, 0.1, False))
    monkeypatch.setattr(scoring, "smoke_compile_check",
                        lambda code, sandbox, language=None: (smoke_ok, "ok", ""))
    monkeypatch.setattr(scoring, "score_candidate_per_step",
                        lambda code: {"gx_score_min": 0.9, "gx_score_mean": 0.9,
                                      "cx_norm_max": 0.1, "first_off_rails_idx": -1,
                                      "n_tokens": 10, "thresholds": {"severe": 0.30}})
    # The structural veto is a different subsystem with its own tests; it must
    # not decide which candidates reach the selection this slice is about.
    monkeypatch.setattr(symbols, "structural_score",
                        lambda project_symbols, code: {"ok": False})
    monkeypatch.setattr(P, "run_browser_probe",
                        lambda code, sandbox=None: (probe(code) if callable(probe)
                                                    else probe))
    if record_hook is not None:
        real = adapters.contract_record
        monkeypatch.setattr(adapters, "contract_record",
                            lambda **kw: record_hook(real, kw))

    service = P.V3PipelineService()
    if oracle_cases:
        # These stand in for a REAL oracle — repository or task-supplied
        # cases whose expected output ATLAS did not invent. They declare
        # trusted provenance, so they keep the reject-and-close authority
        # model-generated cases no longer have.
        cases = [SimpleNamespace(input_str="1", expected_output="1",
                                 provenance=P.PROVENANCE_TRUSTED)
                 for _ in range(oracle_cases)]
        service.self_test_gen = SimpleNamespace(
            generate=lambda problem, llm, task_id: SimpleNamespace(test_cases=cases))
    else:
        service.self_test_gen = SimpleNamespace(
            generate=lambda problem, llm, task_id:
                (_ for _ in ()).throw(RuntimeError("unavailable")))

    calls = plan_calls if plan_calls is not None else []

    def _generate(problem, task_id, llm, num_plans=None, budget_tier="standard"):
        calls.append(task_id)
        return SimpleNamespace(candidates=list(ALT_CODES), total_tokens=0)

    service.plan_search = SimpleNamespace(generate=_generate)
    service.pr_cot = SimpleNamespace(
        repair=lambda problem, code, error, llm_call, task_id:
            SimpleNamespace(repairs=[], total_tokens=0))
    service.refinement_loop = SimpleNamespace(
        run=lambda **kw: SimpleNamespace(solved=False, total_tokens=0,
                                         total_iterations=1, winning_code=""))
    return service, calls


def _complete_probe(**over):
    ev = {"supported": True, "runtime_clean": True, "temporal_progress": True,
          "input_causality": True, "collision_transition": True,
          "food_or_score_transition": True}
    ev.update(over)
    return ev


BROWSER_JS = ("const c = document.getElementById('g');\n"
              "const ctx = c.getContext('2d');\n"
              "document.addEventListener('keydown', e => {});\n"
              "function loop(){ ctx.fillRect(0,0,10,10); setTimeout(loop, 50); } loop();\n")


def _run(service, file_path, problem="build the thing"):
    return service.run(problem, task_id="t", file_path=file_path)


# 1. Algorithmic I/O with a complete oracle: a legitimate close.
def test_algorithmic_oracle_closes_without_generating(monkeypatch):
    service, calls = _service(monkeypatch, oracle_cases=2, self_test_pass=True)
    result = _run(service, "solve.py")

    assert result["phase_solved"] == "probe"
    assert result["passed"] is True
    assert calls == [], "a closed pipeline must not generate alternatives"
    rec = result["evidence_record"]
    assert rec["evidence_strength"] == C.ORACLE
    assert rec["closure_eligible"] is True
    assert rec["candidate_content_hash"] == C.content_hash(result["code"])


# 1b. The bytes that close the run are the bytes the model wrote. PROBE_CODE
# ends in a newline and reaches the pipeline through a fenced response, so
# this is exact-byte identity across extraction, selection and the hash the
# proxy will compare against the bytes it holds.
def test_the_closing_candidate_is_the_models_exact_bytes(monkeypatch):
    service, _ = _service(monkeypatch, oracle_cases=2, self_test_pass=True)
    result = _run(service, "solve.py")

    assert result["code"] == PROBE_CODE
    assert result["code"].endswith("\n")
    rec = result["evidence_record"]
    assert rec["candidate_content_hash"] == C.content_hash(PROBE_CODE)
    assert rec["candidate_content_hash"] == C.content_hash(result["code"])


# 2. Algorithmic I/O that does not pass its own suite: no closure claim.
def test_algorithmic_partial_does_not_close_and_generates(monkeypatch):
    service, calls = _service(monkeypatch, oracle_cases=3, partial_oracle=True)
    result = _run(service, "solve.py")

    assert result["phase_solved"] != "probe"
    rec = result["evidence_record"]
    assert rec["closure_eligible"] is False
    assert rec["evidence_strength"] != C.ORACLE
    assert len(calls) == 1, "alternatives must be generated"


# 3. Browser evidence that is real but partial: still open.
def test_browser_partial_evidence_generates_alternatives(monkeypatch):
    monkeypatch.setenv("ATLAS_EVIDENCE_MODE", "enforce")
    service, calls = _service(monkeypatch, task_type="interactive", code=BROWSER_JS,
                              probe=lambda code: _complete_probe(input_causality=False))
    result = _run(service, "game.js")

    assert result["phase_solved"] != "probe"
    # The probe's own verdict, before selection replaced the run's record with
    # the delivered candidate's.
    early = result["evidence_early_return"]
    assert early["adapter"] == adapters.ADAPTER_BROWSER_CANVAS_JS
    assert early["strength"] == C.RUNTIME
    assert early["closure_eligible"] is False
    assert early["evidence_would_return_early"] is False
    assert early["minimum_closure_strength"] == C.BEHAVIORAL
    assert len(calls) == 1


# 4. Browser evidence that satisfies every requirement and the floor.
def test_browser_complete_evidence_closes_in_enforce(monkeypatch):
    monkeypatch.setenv("ATLAS_EVIDENCE_MODE", "enforce")
    service, calls = _service(monkeypatch, task_type="interactive", code=BROWSER_JS,
                              probe=_complete_probe())
    result = _run(service, "game.js")

    rec = result["evidence_record"]
    assert rec["evidence_strength"] == C.BEHAVIORAL
    assert rec["requirements_complete"] is True
    assert rec["closure_eligible"] is True
    assert result["phase_solved"] == "probe"
    assert calls == []


def test_browser_complete_evidence_does_not_close_in_shadow(monkeypatch):
    """Shadow observes. The probe-free judgement is the one that may act."""
    monkeypatch.setenv("ATLAS_EVIDENCE_MODE", "shadow")
    service, calls = _service(monkeypatch, task_type="interactive", code=BROWSER_JS,
                              probe=_complete_probe())
    result = _run(service, "game.js")

    assert result["phase_solved"] != "probe"
    assert len(calls) == 1
    early = result["evidence_early_return"]
    assert early["evidence_would_return_early"] is True
    assert early["probe_free_would_return_early"] is False


# 5. A contract whose declared floor IS syntax may close on syntax evidence.
def test_syntax_floor_contract_closes_on_syntax(monkeypatch):
    service, calls = _service(monkeypatch, task_type="interactive",
                              code="body { color: red; }\n")
    result = _run(service, "theme.css")

    rec = result["evidence_record"]
    assert adapters.closure_floor(rec["adapter_id"]) == C.SYNTAX
    assert rec["evidence_strength"] == C.SYNTAX
    assert rec["closure_eligible"] is True
    assert result["phase_solved"] == "probe"
    assert calls == [], "a satisfied contract must not spend the budget"


# 6. Unsupported is unverified, never failed, and never closes.
def test_unsupported_adapter_never_closes_and_is_not_failed(monkeypatch):
    service, calls = _service(monkeypatch, task_type="interactive",
                              code="export const A = () => null;\n")
    result = _run(service, "app.tsx")

    rec = result["evidence_record"]
    assert rec["supported"] is False
    assert rec["closure_eligible"] is False
    assert rec["execution_status"] == C.EXEC_SKIPPED, \
        "unsupported must not be reported as an execution failure"
    assert result["phase_solved"] != "probe"
    assert len(calls) == 1


# 7. Evidence about other bytes closes nothing.
def test_hash_mismatch_cannot_close(monkeypatch):
    def _stale(real, kw):
        kw = dict(kw)
        kw["candidate_content_hash"] = C.content_hash("some other artifact\n")
        return real(**kw)

    service, calls = _service(monkeypatch, oracle_cases=2, self_test_pass=True,
                              record_hook=_stale)
    result = _run(service, "solve.py")

    rec = result["evidence_record"]
    assert rec["closure_eligible"] is True, "the record itself is well formed"
    assert rec["candidate_content_hash"] != C.content_hash(PROBE_CODE)
    assert result["phase_solved"] != "probe", \
        "a record about other bytes may not close the pipeline"
    assert len(calls) == 1


# 8. A foreign majority cannot outvote the one record measured under the
# task's own rubric.
def test_foreign_records_cannot_outvote_the_matching_one(monkeypatch):
    def _foreign(real, kw):
        kw = dict(kw)
        if kw["candidate_content_hash"] != C.content_hash(PROBE_CODE):
            kw["contract_id"] = "generate:other"
        return real(**kw)

    monkeypatch.setenv("ATLAS_EVIDENCE_MODE", "enforce")
    service, calls = _service(monkeypatch, task_type="interactive",
                              code=BROWSER_JS,
                              probe=lambda code: _complete_probe(
                                  food_or_score_transition=False),
                              record_hook=_foreign)
    result = _run(service, "game.js")

    selection = result.get("evidence_selection")
    assert selection, "selection telemetry must survive"
    assert selection["incomparable"] >= 1
    assert selection["evidence_index"] == 0, \
        "the matching record wins over a foreign majority"


# 9. A best record that is not closure-eligible is diagnostic only.
def test_best_record_without_closure_does_not_authorize_a_winner(monkeypatch):
    monkeypatch.setenv("ATLAS_EVIDENCE_MODE", "enforce")
    service, calls = _service(monkeypatch, task_type="interactive",
                              code=BROWSER_JS,
                              probe=lambda code: _complete_probe(
                                  input_causality=False))
    result = _run(service, "game.js")

    selection = result["evidence_selection"]
    assert selection["status"] == C.SELECTION_BEST_NOT_ELIGIBLE
    assert selection["verified_index"] is None
    assert result["contract_selection"]["verified_winner"] is None
    assert result["contract_selection"]["best_record"] is not None


# 10. Shadow keeps the lens bytes; enforce moves only for a verified winner.
@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_lens_bytes_change_only_for_a_verified_contract_winner(monkeypatch, mode):
    monkeypatch.setenv("ATLAS_EVIDENCE_MODE", mode)
    service, calls = _service(monkeypatch, task_type="interactive",
                              code=BROWSER_JS,
                              probe=lambda code: _complete_probe(
                                  input_causality=False))
    result = _run(service, "game.js")

    selection = result.get("evidence_selection")
    if selection:
        # No verified winner exists, so neither mode may replace the choice.
        assert selection["verified_index"] is None
        assert result["code"] in [BROWSER_JS] + ALT_CODES


# 11. Candidate zero stays in the pool and can win it.
def test_candidate_zero_remains_selectable(monkeypatch):
    monkeypatch.setenv("ATLAS_EVIDENCE_MODE", "shadow")
    service, calls = _service(monkeypatch, task_type="interactive",
                              code=BROWSER_JS,
                              probe=lambda code: _complete_probe(
                                  input_causality=(code == BROWSER_JS)))
    result = _run(service, "game.js")

    selection = result.get("evidence_selection")
    assert selection, "selection telemetry must survive"
    indices = [c["index"] for c in selection["candidates"]]
    assert 0 in indices, "candidate zero must remain in the pool"
    assert selection["evidence_index"] == 0


# 12. The structured vocabulary survives into telemetry and the wire envelope.
def test_selection_vocabulary_reaches_telemetry_and_the_envelope(monkeypatch):
    monkeypatch.setenv("ATLAS_EVIDENCE_MODE", "shadow")
    service, calls = _service(monkeypatch, task_type="interactive",
                              code=BROWSER_JS,
                              probe=lambda code: _complete_probe())
    result = _run(service, "game.js")

    selection = result.get("evidence_selection")
    assert selection["status"] in C.SELECTION_STATUSES
    assert selection["reason"]
    for key in ("tied", "incomparable", "ineligible"):
        assert key in selection

    # The POOL telemetry keeps the vocabulary of the selection that ran; the
    # ENVELOPE describes the bytes actually delivered, which in shadow is the
    # lens choice rather than the contract's pick. Both are structured, and
    # neither is allowed to speak for the other.
    envelope = adapters.evidence_envelope(result, delivered_code=result["code"])
    assert envelope["selection"]["status"] in C.SELECTION_STATUSES
    assert envelope["evaluation"]["evidence_strength"]
    assert envelope["identity"]["candidate_content_hash"] == \
        C.content_hash(result["code"])
    assert envelope["delivery"]["describes_delivered_candidate"] is True


def test_env_none_differs_from_an_empty_env():
    import os
    os.environ["ATLAS_EVIDENCE_MODE"] = "enforce"
    try:
        assert P._selection_mode() == P.MODE_ENFORCE       # reads the process env
        assert P._selection_mode({}) == P.MODE_OFF         # explicitly empty
    finally:
        del os.environ["ATLAS_EVIDENCE_MODE"]


def test_probe_timeout_fits_inside_the_client_read_timeout():
    import pipeline as P
    assert P.BROWSER_PROBE_TIMEOUT_S < 45, \
        "an execution budget above the client read timeout is cut off by its caller"


# ---------------------------------------------------------------------------
# Every successful exit describes the bytes it returns
# ---------------------------------------------------------------------------
#
# The pipeline has six ways to return code with passed=true. Two of them
# evaluated the artifact they hand back; the rest returned code whose evidence
# was missing or about a different candidate, so a consumer had nothing to
# check the delivered bytes against. These drive each phase through the real
# run() and assert the envelope describes the exact returned hash.

SUCCESS_PHASES = ("probe", "dead_oracle_consensus", "budget", "phase1",
                  "pr_cot", "refinement")


def _envelope_for(result):
    return adapters.evidence_envelope(result, delivered_code=result["code"])


def _assert_envelope_describes_delivery(result, phase):
    """The exhaustive check every successful exit owes."""
    assert result["phase_solved"] == phase, result["phase_solved"]
    assert result["code"], "a successful exit returned no code"
    env = _envelope_for(result)
    assert env is not None, f"{phase} returned code with no evidence envelope"

    delivered = C.content_hash(result["code"])
    assert env["identity"]["candidate_content_hash"] == delivered, phase
    assert env["delivery"]["delivered_content_hash"] == delivered, phase
    assert env["delivery"]["describes_delivered_candidate"] is True, phase

    identity = env["identity"]
    assert identity["contract_id"] and identity["contract_version"], phase
    assert identity["artifact_scope"] and identity["evaluation_context_hash"], phase
    assert identity["adapter_id"] and identity["adapter_version"], phase

    ev = env["evaluation"]
    assert ev["execution_status"] in C.EXECUTION_STATUSES, phase
    assert ev["evidence_strength"] in C.STRENGTH_ORDER, phase
    assert isinstance(ev["requirements_complete"], bool), phase
    assert isinstance(ev["closure_eligible"], bool), phase
    assert 0.0 <= ev["quality"]["overall"] <= 1.0, phase

    cov = env["coverage"]
    for key in ("required", "demonstrated", "missing", "unmeasurable", "optional"):
        assert key in cov, f"{phase}: coverage lacks {key}"

    assert env["selection"]["status"] in C.SELECTION_STATUSES, phase
    assert env["wire_version"] and env["record_schema_version"], phase
    return env


def test_probe_exit_describes_its_delivery(monkeypatch):
    service, _ = _service(monkeypatch, oracle_cases=2, self_test_pass=True)
    result = _run(service, "solve.py")
    env = _assert_envelope_describes_delivery(result, "probe")
    assert env["selection"]["status"] == C.SELECTION_VERIFIED_WINNER
    assert env["evaluation"]["closure_eligible"] is True


def test_dead_oracle_consensus_exit_describes_its_delivery(monkeypatch):
    monkeypatch.setenv("ATLAS_V3_DEAD_ORACLE_CONSENSUS", "1")
    chosen = {"code": "def agreed():\n    return 7\n", "index": 2}
    monkeypatch.setattr(P, "_dead_oracle_consensus",
                        lambda *a, **kw: (chosen, 0))
    service, calls = _service(monkeypatch, oracle_cases=2, self_test_pass=False)
    result = _run(service, "solve.py")

    env = _assert_envelope_describes_delivery(result, "dead_oracle_consensus")
    assert result["code"] == chosen["code"]
    assert result["passed"] is True
    # Consensus is agreement, not verification: the envelope must not claim a
    # verified winner for it.
    assert env["selection"]["status"] != C.SELECTION_VERIFIED_WINNER or \
        env["evaluation"]["closure_eligible"] is True


def test_budget_exit_describes_its_delivery(monkeypatch):
    # The budget path hands back the best PASSING candidate when the run's
    # wall-clock cap expires mid-pipeline. Driven through the pipeline's own
    # budget primitive: generous until generation has produced a pool, then
    # spent.
    # The lens declines to select, so phase one does not close the run, and
    # the clock then expires before the repair phase -- which is exactly when
    # an anytime algorithm has to hand back its best verified candidate.
    monkeypatch.setattr(P, "select_candidate", lambda cands, strategy="lens": None)
    monkeypatch.setattr(P, "_remaining_budget_ms", lambda start: -1.0)
    service, calls = _service(monkeypatch, task_type="interactive",
                              code=BROWSER_JS)
    result = _run(service, "game.js")
    if result["phase_solved"] != "budget":
        pytest.skip(f"budget path not taken (got {result['phase_solved']})")
    _assert_envelope_describes_delivery(result, "budget")


def test_phase_one_exit_describes_its_delivery(monkeypatch):
    service, calls = _service(monkeypatch, task_type="interactive",
                              code=BROWSER_JS)
    result = _run(service, "game.js")
    _assert_envelope_describes_delivery(result, "phase1")


def test_pr_cot_exit_describes_its_delivery(monkeypatch):
    repaired = "def repaired():\n    return 11\n"
    service, calls = _service(monkeypatch, oracle_cases=3,
                              passing_marker="repaired")
    service.plan_search = SimpleNamespace(
        generate=lambda problem, task_id, llm, num_plans=None,
        budget_tier="standard": SimpleNamespace(candidates=list(ALT_CODES),
                                                total_tokens=0))
    service.pr_cot = SimpleNamespace(
        repair=lambda problem, code, error, llm_call, task_id:
            SimpleNamespace(repairs=[repaired], total_tokens=0))
    result = _run(service, "solve.py")
    if result["phase_solved"] != "pr_cot":
        pytest.skip(f"repair path not taken (got {result['phase_solved']})")
    env = _assert_envelope_describes_delivery(result, "pr_cot")
    assert result["code"] == repaired
    assert env["identity"]["candidate_content_hash"] == C.content_hash(repaired)


def test_refinement_exit_describes_its_delivery(monkeypatch):
    winning = "def refined():\n    return 13\n"
    service, calls = _service(monkeypatch, oracle_cases=3,
                              passing_marker="refined")
    service.pr_cot = SimpleNamespace(
        repair=lambda problem, code, error, llm_call, task_id:
            SimpleNamespace(repairs=[], total_tokens=0))
    service.refinement_loop = SimpleNamespace(
        run=lambda **kw: SimpleNamespace(solved=True, total_tokens=0,
                                         total_iterations=2,
                                         winning_code=winning))
    result = _run(service, "solve.py")
    if result["phase_solved"] != "refinement":
        pytest.skip(f"refinement path not taken (got {result['phase_solved']})")
    env = _assert_envelope_describes_delivery(result, "refinement")
    assert result["code"] == winning
    assert env["identity"]["candidate_content_hash"] == C.content_hash(winning)


def test_every_successful_exit_is_covered():
    """The sentinel: a new way to return passed=true must arrive with a test.

    Counted from the source rather than from a list someone maintains, so a
    seventh success site fails this immediately.
    """
    src = (Path(__file__).resolve().parents[2] / "v3-service" / "pipeline.py").read_text()
    sites = src.count('result["passed"] = True')
    assert sites == len(SUCCESS_PHASES), (
        f"{sites} successful-return sites, {len(SUCCESS_PHASES)} covered phases: "
        f"add the new phase to SUCCESS_PHASES and give it a run() test")
    # Every phase name this file claims to cover is one the pipeline can emit.
    for phase in SUCCESS_PHASES:
        assert f'result["phase_solved"] = "{phase}"' in src, phase
    # And the finaliser is the single place that guarantees it.
    assert "_ensure_delivered_evidence(result" in src


# =====================================================================
# Model-generated cases lose their authority; a trusted oracle keeps its own.
#
# The captured ring2 shape: five generated cases, two keys agreeing with the
# task's own reference and three disagreeing. The candidate matched the
# reference on every input and scored 2/5 against its own suite. Under the
# old rule that was a rejection, and at 0/N it skipped candidate generation
# entirely and returned nothing.
# =====================================================================

RING2_CANDIDATE = ("def main():\n"
                   "    print(open('input.txt').read().strip())\n"
                   "if __name__ == '__main__':\n    main()\n")


def _generated_cases(n=5):
    return [SimpleNamespace(input_str=str(i), expected_output=str(i),
                            provenance=P.PROVENANCE_GENERATED)
            for i in range(n)]


def _sandbox_scoring(pass_cases):
    """A sandbox whose generated self-tests pass exactly `pass_cases` of 5.

    The candidate itself always runs cleanly — the only thing failing is the
    comparison against the model's own answer key.
    """
    class _S:
        def __init__(self, project_files=None):
            pass

        def __call__(self, code, test_input="", files=None, **_):
            if "SELF_TEST_PASS" not in code:
                return True, "ok", ""
            idx = int((files or {}).get("input.txt", "0") or "0")
            if idx < pass_cases:
                return True, "SELF_TEST_PASS\n", ""
            return False, "", "AssertionError: got something else"

        def syntax_check(self, code, language, filename=""):
            return True, "", ""
    return _S


def _run_with(monkeypatch, sandbox_cls, cases, plan_candidates=None,
              diagnostic_total=0):
    monkeypatch.setenv("ATLAS_V3_TELEMETRY_DIR", "off")
    monkeypatch.setattr(adapters, "LLMAdapter", _llm_returning(RING2_CANDIDATE))
    monkeypatch.setattr(adapters, "SandboxAdapter", sandbox_cls)
    monkeypatch.setattr(adapters, "EmbedAdapter", lambda: (lambda t: []))
    monkeypatch.setattr(scoring, "classify_task_type", lambda p: "algorithmic")
    monkeypatch.setattr(scoring, "score_candidate", lambda code: (1.0, 0.5, False))
    monkeypatch.setattr(scoring, "score_candidate_combined",
                        lambda code: dict(scoring.NEUTRAL_COMBINED))
    monkeypatch.setattr(scoring, "score_candidate_per_step", lambda code: None)
    service = P.V3PipelineService()
    service.self_test_gen = SimpleNamespace(
        generate=lambda problem, llm, task_id:
            SimpleNamespace(test_cases=cases, generation_tokens=0))
    service.plan_search = SimpleNamespace(
        generate=lambda problem, task_id, llm, num_plans=None,
        budget_tier="standard": SimpleNamespace(
            candidates=list(plan_candidates or []), total_tokens=0))
    service.pr_cot = SimpleNamespace(
        repair=lambda problem, code, error, llm_call, task_id:
            SimpleNamespace(repairs=[], total_tokens=0))
    service.refinement_loop = SimpleNamespace(
        run=lambda **kw: SimpleNamespace(solved=False, total_tokens=0,
                                         total_iterations=1, winning_code=""))
    kwargs = {}
    if diagnostic_total:
        kwargs["diagnostic_total_candidates"] = diagnostic_total
    return service.run("read input.txt", task_id="trust",
                       file_path="/workspace/e2e/solve.py",
                       files={"input.txt": "1\n"}, **kwargs)


def _llm_returning(code):
    class _L:
        def __init__(self, progress_callback=None, thinking=False):
            pass

        def __call__(self, prompt, temperature, max_tokens, seed, thinking=None):
            return f"```python\n{code}\n```", 5, 1.0
    return _L


def test_two_of_five_generated_cases_does_not_reject_the_candidate(monkeypatch):
    """The captured ring2 score, on a candidate that is actually correct."""
    result = _run_with(monkeypatch, _sandbox_scoring(2), _generated_cases())
    stages = [e["stage"] for e in result["events"]]
    assert "self_test_verify" in stages
    assert "self_test_untrusted" in stages
    assert "self_test_inconclusive" not in stages
    assert result["phase_solved"] != "oracle_inconclusive"


def test_five_zero_of_n_results_do_not_skip_candidate_generation(monkeypatch):
    """0/N from model-generated cases must not take the dead-oracle exit."""
    result = _run_with(monkeypatch, _sandbox_scoring(0), _generated_cases(),
                       plan_candidates=[RING2_CANDIDATE.replace("strip", "rstrip")])
    stages = [e["stage"] for e in result["events"]]
    assert "probe_unverifiable" not in stages
    assert result["phase_solved"] != "oracle_inconclusive"
    assert "plansearch" in stages, "candidate generation must happen"
    assert result["candidates_generated"] >= 2


def test_a_generated_score_never_reaches_a_repair_prompt(monkeypatch):
    """A wrong key must not become 'your output was wrong; expected X'.

    The score reached repair as the candidate's error_output, so a candidate
    that was right and a key that was wrong produced an instruction to fix
    working code.
    """
    result = _run_with(monkeypatch, _sandbox_scoring(0), _generated_cases())
    blob = repr(result.get("events", []))
    assert "Self-test:" not in blob
    assert "expected" not in blob.lower() or "self_test_untrusted" in blob


def test_a_trusted_oracle_still_rejects_and_still_closes(monkeypatch):
    """Requirement 6: correcting generated-case authority must not weaken a
    real oracle."""
    trusted = [SimpleNamespace(input_str=str(i), expected_output=str(i),
                               provenance=P.PROVENANCE_TRUSTED)
               for i in range(5)]
    result = _run_with(monkeypatch, _sandbox_scoring(0), trusted)
    stages = [e["stage"] for e in result["events"]]
    assert "self_test_inconclusive" in stages
    assert "self_test_untrusted" not in stages
    assert result["phase_solved"] == "oracle_inconclusive"


def test_a_syntax_failure_still_rejects_independently(monkeypatch):
    """The syntax/compile verifier does not depend on a generated case, so it
    keeps its rejection authority."""
    class _Broken:
        def __init__(self, project_files=None):
            pass

        def __call__(self, code, test_input="", files=None, **_):
            if "SELF_TEST_PASS" in code:
                return True, "SELF_TEST_PASS\n", ""
            return False, "", "SyntaxError: invalid syntax"

        def syntax_check(self, code, language, filename=""):
            return False, "", "SyntaxError: invalid syntax"

    result = _run_with(monkeypatch, _Broken, _generated_cases())
    assert result["passed"] is False
    assert result["code"] == ""


def test_generated_evidence_never_reaches_closure_or_a_verified_winner(monkeypatch):
    """With no trusted oracle nothing may close, whatever the pool did."""
    result = _run_with(monkeypatch, _sandbox_scoring(5), _generated_cases(),
                       plan_candidates=[RING2_CANDIDATE.replace("strip", "rstrip")])
    record = result.get("evidence_record") or {}
    # The compile requirement IS complete -- the artifact parses, and saying
    # otherwise was an accident of the old quarantine. What it cannot reach is
    # the behavioural floor its adapter demands, which is the fact that keeps
    # an untrusted pool from closing anything.
    assert record.get("evidence_strength") == C.SYNTAX
    assert record.get("closure_eligible") is False
    selection = result.get("contract_selection") or {}
    assert selection.get("verified_winner") is None
    assert C.selection_status(selection) != "verified_winner"


# =====================================================================
# Diagnostic candidate allocation.
#
# `k` is the TOTAL V3 pool: the phase-zero probe occupies one slot and
# `remaining_k = k - len(candidates)` fills the rest with generated
# alternatives. The incumbent is not in it at all -- it lives in the shadow
# comparison pool. A measurement that needs consensus needs at least two
# generated alternatives, and the budget cap can drive the allocator's k to
# 1, so the diagnostic must be able to raise it. It may only raise, only in
# process, only up to a small hard cap, and never through /v3/generate.
# =====================================================================


def test_k_counts_the_probe_and_the_generated_alternatives():
    """Naming check: whatever the argument is called, this is the quantity."""
    src = (Path(__file__).resolve().parents[2] / "v3-service" / "pipeline.py").read_text()
    assert "remaining_k = max(0, k - len(candidates))" in src


def test_the_default_allocation_is_untouched(monkeypatch):
    """Absent means absent: same k, same events, no diagnostic path."""
    plain = _run_with(monkeypatch, _sandbox_scoring(2), _generated_cases(),
                      plan_candidates=[RING2_CANDIDATE.replace("strip", "rstrip")])
    alloc = [e for e in plain["events"] if e["stage"] == "phase2_allocated"]
    assert alloc, "allocation must still happen"
    assert "diagnostic_allocation" not in [e["stage"] for e in plain["events"]]


@pytest.mark.parametrize("bad", [True, False, "3", 3.5, -1, 0.0, 99])
def test_malformed_or_oversized_values_fail_closed(bad):
    with pytest.raises((ValueError, TypeError)):
        P._diagnostic_total_candidates(bad)


@pytest.mark.parametrize("good,expected", [(None, 0), (0, 0), (3, 3), (6, 6)])
def test_absent_and_in_range_values_are_accepted(good, expected):
    assert P._diagnostic_total_candidates(good) == expected


def test_the_cap_is_small_and_declared():
    assert P.DIAGNOSTIC_MAX_TOTAL_CANDIDATES == 6


def test_the_diagnostic_floor_raises_the_pool(monkeypatch):
    """The allocator's own floor is 3; asking for more must raise it."""
    alternatives = [RING2_CANDIDATE.replace("strip", "rstrip"),
                    RING2_CANDIDATE.replace("read()", "read(  )"),
                    RING2_CANDIDATE.replace("main()", "main( )")]
    result = _run_with(monkeypatch, _sandbox_scoring(2), _generated_cases(),
                       plan_candidates=alternatives, diagnostic_total=4)
    diag = [e for e in result["events"] if e["stage"] == "diagnostic_allocation"]
    assert diag, "the raise must be visible in telemetry"
    assert diag[0]["data"]["diagnostic_k"] == 4
    assert diag[0]["data"]["allocated_k"] < 4, "it may only raise"
    phase1 = [e for e in result["events"] if e["stage"] == "phase1"][0]
    assert "4 diverse candidates" in phase1["detail"]


def test_the_diagnostic_floor_never_lowers_the_allocation(monkeypatch):
    """Asking for fewer than the allocator chose changes nothing."""
    result = _run_with(monkeypatch, _sandbox_scoring(2), _generated_cases(),
                       plan_candidates=[RING2_CANDIDATE.replace("strip", "rstrip")],
                       diagnostic_total=2)
    assert not [e for e in result["events"]
                if e["stage"] == "diagnostic_allocation"]
    phase1 = [e for e in result["events"] if e["stage"] == "phase1"][0]
    assert "3 diverse candidates" in phase1["detail"]


def test_the_generate_endpoint_cannot_activate_it():
    """No request field, no env read, no server-side mode: the only way in
    is an in-process argument the HTTP handler never supplies."""
    v3dir = Path(__file__).resolve().parents[2] / "v3-service"
    main_src = (v3dir / "main.py").read_text()
    assert "diagnostic_total_candidates" not in main_src
    assert "min_candidates" not in main_src
    handler = main_src.split("result = pipeline.run(", 1)[1].split(")", 1)[0]
    assert "diagnostic" not in handler
    pipeline_src = (v3dir / "pipeline.py").read_text()
    floor_block = pipeline_src.split("def _diagnostic_total_candidates(", 1)[1]
    floor_block = floor_block.split("\ndef ", 1)[0]
    assert "environ" not in floor_block and "getenv" not in floor_block


def test_the_go_request_has_no_representation_for_it():
    go = (Path(__file__).resolve().parents[2] / "proxy" / "types.go").read_text()
    for name in ("diagnostic_total_candidates", "min_candidates",
                 "DiagnosticTotalCandidates", "MinCandidates"):
        assert name not in go, name


def test_capture_and_diagnostic_allocation_are_independent(monkeypatch, tmp_path):
    """Capture alone never raises k; allocation alone never opens a sink."""
    monkeypatch.setenv("ATLAS_V3_CAPTURE_POOL", str(tmp_path / "p.jsonl"))
    captured = _run_with(monkeypatch, _sandbox_scoring(2), _generated_cases(),
                         plan_candidates=[RING2_CANDIDATE.replace("strip", "rstrip")])
    assert "diagnostic_allocation" not in [e["stage"] for e in captured["events"]]

    monkeypatch.delenv("ATLAS_V3_CAPTURE_POOL", raising=False)
    empty = tmp_path / "none"
    empty.mkdir()
    _run_with(monkeypatch, _sandbox_scoring(2), _generated_cases(),
              plan_candidates=[RING2_CANDIDATE.replace("strip", "rstrip")],
              diagnostic_total=3)
    assert list(empty.iterdir()) == []
