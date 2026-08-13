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


def _sandbox_factory(self_test_pass=True, smoke_ok=True, partial_oracle=False):
    seen = {"cases": 0}

    class _Sandbox:
        def __init__(self, project_files=None):
            pass

        def __call__(self, code, test_input=""):
            if "SELF_TEST_PASS" in code:
                seen["cases"] += 1
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
             record_hook=None, code=PROBE_CODE, partial_oracle=False):
    """A V3PipelineService whose every outside dependency is controlled."""
    monkeypatch.setattr(_LLM, "code", code)
    monkeypatch.setattr(adapters, "LLMAdapter", _LLM)
    monkeypatch.setattr(adapters, "SandboxAdapter",
                        _sandbox_factory(self_test_pass, smoke_ok, partial_oracle))
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
        cases = [SimpleNamespace(input_str="1", expected_output="1")
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

    envelope = adapters.evidence_envelope(result, delivered_code=result["code"])
    assert envelope["selection"]["status"] == selection["status"]
    assert envelope["evaluation"]["evidence_strength"]
    assert envelope["identity"]["candidate_content_hash"]


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
