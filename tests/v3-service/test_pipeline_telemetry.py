"""D8: the live service writes stage telemetry.

V3PipelineService used to construct every stage without telemetry_dir, so
all documented telemetry/*.jsonl was bench-only and the live orchestrator
was unmeasurable. The service now resolves ATLAS_V3_TELEMETRY_DIR, passes
it to the stages, and pipeline.run appends one pipeline_summary.jsonl line
per task — fail-soft in every direction.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "v3-service"))

import adapters  # noqa: E402
import pipeline as v3pipeline  # noqa: E402
import scoring  # noqa: E402

VETO_PER_STEP = {
    "gx_score_min": 0.01,
    "gx_score_mean": 0.02,
    "cx_norm_max": 0.5,
    "first_off_rails_idx": 0,
    "n_tokens": 10,
    "thresholds": {"severe": 0.30},
}


class FakeLLM:
    def __init__(self, progress_callback=None, thinking=False):
        pass

    def __call__(self, prompt, temperature, max_tokens, seed, thinking=None):
        return "<think>still thinking", 3, 1.0


class PassingSandbox:
    def __init__(self, project_files=None):
        pass

    def __call__(self, code, test_input=""):
        return True, "ok", ""


class FakeEmbed:
    def __call__(self, text):
        return []


def _make_service(monkeypatch):
    monkeypatch.setattr(adapters, "LLMAdapter", FakeLLM)
    monkeypatch.setattr(adapters, "SandboxAdapter", PassingSandbox)
    monkeypatch.setattr(adapters, "EmbedAdapter", FakeEmbed)
    monkeypatch.setattr(scoring, "classify_task_type", lambda p: "algorithmic")
    monkeypatch.setattr(scoring, "score_candidate", lambda code: (1.0, 0.1, False))
    monkeypatch.setattr(
        scoring, "score_candidate_per_step", lambda code: dict(VETO_PER_STEP))

    service = v3pipeline.V3PipelineService()
    service.self_test_gen = SimpleNamespace(
        generate=lambda problem, llm, task_id:
            (_ for _ in ()).throw(RuntimeError("unavailable")))
    service.plan_search = SimpleNamespace(
        generate=lambda problem, task_id, llm, num_plans=None:
            SimpleNamespace(candidates=["def a():\n    pass\n",
                                        "def b():\n    pass\n",
                                        "def c():\n    pass\n"],
                            total_tokens=0))
    service.pr_cot = SimpleNamespace(
        repair=lambda problem, code, error, llm_call, task_id:
            SimpleNamespace(repairs=[], total_tokens=0))
    service.refinement_loop = SimpleNamespace(
        run=lambda **kw: SimpleNamespace(solved=False, total_tokens=0,
                                         total_iterations=1, winning_code=""))
    service.derivation_chains = SimpleNamespace(
        solve=lambda **kw: SimpleNamespace(solved=False, total_tokens=0,
                                           final_code="", reason="exhausted"))
    return service


def test_env_dir_reaches_stages_and_summary_is_written(monkeypatch, tmp_path):
    tdir = tmp_path / "telemetry"
    monkeypatch.setenv("ATLAS_V3_TELEMETRY_DIR", str(tdir))
    service = _make_service(monkeypatch)

    # The resolved dir is wired through to the stages the service builds.
    assert service.telemetry_dir == tdir
    fresh = v3pipeline.V3PipelineService()
    assert fresh.plan_search.telemetry_dir == tdir
    assert fresh.s_star.telemetry_dir == tdir
    assert fresh.refinement_loop.telemetry_dir == tdir

    result = service.run("write a real dashboard", task_id="d8-summary")

    summary_file = tdir / "pipeline_summary.jsonl"
    assert summary_file.exists()
    lines = summary_file.read_text().strip().splitlines()
    assert len(lines) == 1
    line = json.loads(lines[0])
    assert line["schema"] == "v3_pipeline_summary_v1"
    assert line["task_id"] == "d8-summary"
    assert line["passed"] is False
    assert line["phase_solved"] == "none"
    # Fully mocked runs can complete inside a millisecond and round to 0.
    assert line["total_time_ms"] >= 0

    phases = {p["phase"]: p for p in line["phases"]}
    # Probe ran, generation ran, all sandbox-passers were vetoed, repair
    # ran, and the fallback closed the run.
    for expected in ("probe", "generation", "sandbox", "veto",
                     "repair_pr_cot", "fallback"):
        assert expected in phases, f"missing phase {expected}: {line['phases']}"
    assert phases["fallback"]["outcome"] == "fallback_all_vetoed"
    for p in line["phases"]:
        assert p["duration_ms"] >= 0

    assert len(line["veto_events"]) == 3
    assert all(v["stage"] == "lens_veto" for v in line["veto_events"])

    # And the events in the result agree with what was summarized.
    assert result["phase_solved"] == "none"


def test_disable_value_turns_telemetry_off(monkeypatch, tmp_path):
    monkeypatch.setenv("ATLAS_V3_TELEMETRY_DIR", "off")
    service = _make_service(monkeypatch)
    assert service.telemetry_dir is None

    result = service.run("write a real dashboard", task_id="d8-off")
    assert result["phase_solved"] == "none"
    assert not list(tmp_path.iterdir())


def test_unwritable_dir_disables_without_breaking_generation(monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file, not a directory")
    monkeypatch.setenv("ATLAS_V3_TELEMETRY_DIR", str(blocker / "telemetry"))
    service = _make_service(monkeypatch)
    assert service.telemetry_dir is None

    result = service.run("write a real dashboard", task_id="d8-unwritable")
    assert result["phase_solved"] == "none"
