"""D1: a vetoed candidate is a failing candidate.

The lens / structural / call-graph vetoes remove sandbox-passing candidates
from selection. Before the fix they left ``passed=True`` on the candidate
dict, so the phase-3 pool (``not c.get("passed")``) excluded them, selection
excluded them, and the final energy fallback could return the vetoed
candidate itself — the exact stub the veto existed to block.

These tests drive the full ``V3PipelineService.run`` orchestration with all
service adapters faked, reproduce the all-passing-candidates-vetoed scenario,
and assert:
  * vetoed candidates are marked (``passed=False`` + ``vetoed_by``) and enter
    phase-3 repair like any failing candidate (PR-CoT sees the veto reason);
  * the energy fallback never returns a vetoed candidate — a non-vetoed
    sandbox-failing candidate wins even at higher energy, and an all-vetoed
    set returns no code at all.
"""

import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "v3-service"))

import adapters  # noqa: E402
import pipeline as v3pipeline  # noqa: E402
import scoring  # noqa: E402

STUB_CODES = [
    "def stub_a():\n    pass\n",
    "def stub_b():\n    pass\n",
    "def stub_c():\n    pass\n",
]
FAILING_CODE = "def broken():  # FAILS\n    pass\n"
REPAIRED_CODE = "def repaired():\n    return 42\n"

# Per-step lens stub: gx_min sits below the calibrated severe band, which
# must veto any sandbox-passing candidate carrying it.
VETO_PER_STEP = {
    "gx_score_min": 0.01,
    "gx_score_mean": 0.02,
    "cx_norm_max": 0.5,
    "first_off_rails_idx": 0,
    "n_tokens": 10,
    "thresholds": {"severe": 0.30},
}


class FakeLLM:
    """Never produces code — forces the probe to fail so phase 1 runs."""

    def __init__(self, progress_callback=None, thinking=False):
        pass

    def __call__(self, prompt, temperature, max_tokens, seed, thinking=None):
        # An unclosed <think> block extracts to no code at all, so the
        # probe fails every budget tier and phase 1 must run.
        return "<think>still thinking", 3, 1.0


class FakeSandbox:
    """Everything executes fine except code carrying the FAILS marker."""

    def __init__(self, project_files=None):
        self.project_files = project_files or {}

    def __call__(self, code, test_input=""):
        if "FAILS" in code:
            return False, "", "boom: genuine sandbox failure"
        return True, "ok", ""


class FakeEmbed:
    def __call__(self, text):
        return []


class RecordingPRCoT:
    def __init__(self, repairs):
        self.repairs = repairs
        self.calls = []

    def repair(self, problem, code, error, llm_call, task_id):
        self.calls.append({"code": code, "error": error})
        return SimpleNamespace(repairs=list(self.repairs), total_tokens=7)


class RecordingRefinement:
    def __init__(self):
        self.calls = []

    def run(self, problem, failing_candidates, original_constraints,
            llm_call, sandbox_run, embed_call, task_id):
        self.calls.append(list(failing_candidates))
        return SimpleNamespace(solved=False, total_tokens=0,
                               total_iterations=2, winning_code="")


class FakeSelfTestGen:
    def generate(self, problem, llm, task_id):
        raise RuntimeError("self-test generation unavailable in this test")


def _make_service(monkeypatch, plan_codes, pr_cot):
    # Keep the test hermetic: never probe the host's /data/telemetry.
    monkeypatch.setenv("ATLAS_V3_TELEMETRY_DIR", "off")
    monkeypatch.setattr(adapters, "LLMAdapter", FakeLLM)
    monkeypatch.setattr(adapters, "SandboxAdapter", FakeSandbox)
    monkeypatch.setattr(adapters, "EmbedAdapter", FakeEmbed)
    monkeypatch.setattr(scoring, "classify_task_type", lambda p: "algorithmic")
    # Sandbox-passing stubs get LOW energy, the honest failure HIGH energy —
    # so an energy-sorted fallback would prefer the vetoed stubs.
    monkeypatch.setattr(
        scoring, "score_candidate",
        lambda code: (9.0, 0.9, False) if "FAILS" in code else (1.0, 0.1, False))
    monkeypatch.setattr(
        scoring, "score_candidate_per_step", lambda code: dict(VETO_PER_STEP))

    service = v3pipeline.V3PipelineService()
    service.self_test_gen = FakeSelfTestGen()
    service.plan_search = SimpleNamespace(
        generate=lambda problem, task_id, llm, num_plans=None, budget_tier="standard":
            SimpleNamespace(candidates=list(plan_codes), total_tokens=0))
    service.pr_cot = pr_cot
    service.refinement_loop = RecordingRefinement()
    return service


def test_all_passing_candidates_vetoed_enters_repair_and_repair_wins(monkeypatch):
    """Every sandbox-passing candidate is lens-vetoed → phase-3 repair must
    run over the vetoed pool and its verified repair must win."""
    pr_cot = RecordingPRCoT(repairs=[REPAIRED_CODE])
    service = _make_service(monkeypatch, STUB_CODES, pr_cot)

    result = service.run("write a real dashboard", task_id="d1-repair")

    stages = [e["stage"] for e in result["events"]]
    # All three candidates were vetoed...
    assert stages.count("lens_veto") == 3
    # ...and the pipeline entered phase-3 repair with all of them in the pool
    # (the veto log's "falling through to phase-3 repair" is now true).
    phase3 = next(e for e in result["events"] if e["stage"] == "phase3")
    assert phase3["data"]["failing"] == 3
    # PR-CoT ran against a vetoed candidate and saw the veto reason as error.
    assert pr_cot.calls, "PR-CoT repair never ran on the vetoed pool"
    assert pr_cot.calls[0]["code"] in STUB_CODES
    assert "lens veto" in pr_cot.calls[0]["error"]
    # The verified repair — not any vetoed stub — is the returned solution.
    assert result["passed"] is True
    assert result["phase_solved"] == "pr_cot"
    assert result["code"] == REPAIRED_CODE


def test_energy_fallback_returns_no_unverified_candidate(monkeypatch):
    """Repair exhausted and nothing passed: no code is returned at all.

    This used to hand back the honest sandbox failure — the highest-energy
    non-vetoed candidate — on the reasoning that it beat a vetoed stub. It
    does not beat the caller's baseline, which is the model's own
    syntax-gated write. Measured across one 28-session run: 0 of 44
    candidates passed the sandbox and this path returned a failing one 11
    times, each written over the model's own content.

    The vetoed candidates must still reach repair's failing pool, which is
    what the rest of this asserts.
    """
    pr_cot = RecordingPRCoT(repairs=[])  # repair produces nothing
    service = _make_service(
        monkeypatch, [STUB_CODES[0], STUB_CODES[1], FAILING_CODE], pr_cot)

    result = service.run("write a real dashboard", task_id="d1-fallback-mixed")

    stages = [e["stage"] for e in result["events"]]
    assert stages.count("lens_veto") == 2
    assert result["passed"] is False
    assert not result.get("code"), "an unverified candidate must not be returned"
    assert "fallback_unverified" in stages
    # Refinement still saw the vetoed candidates in its failing pool.
    assert service.refinement_loop.calls
    pool_codes = {c.code for c in service.refinement_loop.calls[0]}
    assert STUB_CODES[0] in pool_codes and STUB_CODES[1] in pool_codes


def test_energy_fallback_returns_nothing_when_all_candidates_vetoed(monkeypatch):
    """All candidates vetoed and repair exhausted → no code is returned
    (the caller substitutes its baseline), never a vetoed stub."""
    pr_cot = RecordingPRCoT(repairs=[])
    service = _make_service(monkeypatch, STUB_CODES, pr_cot)

    result = service.run("write a real dashboard", task_id="d1-fallback-all")

    stages = [e["stage"] for e in result["events"]]
    assert stages.count("lens_veto") == 3
    assert "fallback_all_vetoed" in stages
    assert result["passed"] is False
    assert result["code"] == ""
    # Repair still ran before the empty fallback.
    assert pr_cot.calls
