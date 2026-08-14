"""D8: the live service writes stage telemetry.

V3PipelineService used to construct every stage without telemetry_dir, so
all documented telemetry/*.jsonl was bench-only and the live orchestrator
was unmeasurable. The service now resolves ATLAS_V3_TELEMETRY_DIR, passes
it to the stages, and pipeline.run appends one pipeline_summary.jsonl line
per task — fail-soft in every direction.
"""

import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

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

    def __call__(self, code, test_input="", **_):
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
        generate=lambda problem, task_id, llm, num_plans=None, budget_tier="standard":
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
    return service


def test_env_dir_reaches_stages_and_summary_is_written(monkeypatch, tmp_path):
    tdir = tmp_path / "telemetry"
    monkeypatch.setenv("ATLAS_V3_TELEMETRY_DIR", str(tdir))
    service = _make_service(monkeypatch)

    # The resolved dir is wired through to the stages the service builds.
    assert service.telemetry_dir == tdir
    fresh = v3pipeline.V3PipelineService()
    assert fresh.plan_search.telemetry_dir == tdir
    assert fresh.pr_cot.telemetry_dir == tdir
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


# =====================================================================
# Benchmark-only candidate-pool capture.
#
# Suite A retained one number per verification -- a boolean -- so a
# candidate that scored 9/10 and one that scored 0/10 reached the contract
# record identically, and every rejected candidate's bytes were gone by the
# time the run ended. That made the measured 0/N pattern unattributable:
# nothing retained could separate a wrong candidate from a wrong answer
# key. Capture writes the pool, the per-case expected/actual pairs and the
# selection identities to one file, off unless a path is configured, and
# changes no decision on the way.
# =====================================================================

# No trailing newline: extract_code strips the fenced body, so this is the
# byte string the probe actually enters the pipeline with, and capture is
# required to reproduce it exactly rather than approximately.
CAP_ZERO = "def solve(x):\n    return x + 1"
CAP_ONE = "def solve(x):\n    return x + 2"
CAP_TWO = "def solve(x):\n    return x + 3"

CAP_PASS = (True, "SELF_TEST_PASS\n", "")
CAP_INPUTS = ("0", "1", "2", "3", "4")

# Well clear of the severe band: the lens veto is not what this measures,
# and a vetoed pool leaves no selection to summarize.
CAP_PER_STEP = {
    "gx_score_min": 0.80,
    "gx_score_mean": 0.90,
    "cx_norm_max": 0.5,
    "first_off_rails_idx": -1,
    "n_tokens": 10,
    "thresholds": {"severe": 0.30, "severe_mean": 0.40},
}

# Per candidate, per case input: what the sandbox reports back. Candidate
# zero scores 1/5 rather than 0/5 on purpose — a clean 0/N sends the run
# down the dead-oracle fast return and no candidates are ever generated,
# which is the Suite A shape but leaves nothing to select between.
CAP_SANDBOX_TABLE = {
    # 1/5: below half, rejected, and each failure a different kind.
    (CAP_ZERO, "0"): CAP_PASS,
    (CAP_ZERO, "1"): (False, "", "AssertionError: got 2"),
    (CAP_ZERO, "2"): (False, "", "Traceback (most recent call last):\n"
                                 "NameError: name 'q' is not defined"),
    (CAP_ZERO, "3"): (False, "", "execution timed out after 15s"),
    (CAP_ZERO, "4"): (False, "", "AssertionError: got 5"),
    # 3/5: the only candidate that reaches the pool.
    (CAP_ONE, "0"): CAP_PASS,
    (CAP_ONE, "1"): CAP_PASS,
    (CAP_ONE, "2"): CAP_PASS,
    (CAP_ONE, "3"): (False, "", "AssertionError: got 5"),
    (CAP_ONE, "4"): (False, "", "AssertionError: got 6"),
    # 2/5: a partial below half — rejected, and distinguishable from 1/5
    # only because capture kept the score.
    (CAP_TWO, "0"): CAP_PASS,
    (CAP_TWO, "1"): CAP_PASS,
    (CAP_TWO, "2"): (False, "", "AssertionError: got 5"),
    (CAP_TWO, "3"): (False, "", "AssertionError: got 6"),
    (CAP_TWO, "4"): (False, "", "AssertionError: got 7"),
}


class CaptureSandbox:
    """Runs candidates and their generated cases from a fixed table."""

    def __init__(self, project_files=None):
        pass

    def __call__(self, code, test_input="", language="python", timeout=15, **_):
        if "SELF_TEST_PASS" not in code:
            return True, "ok", ""
        candidate = next((c for c in (CAP_ZERO, CAP_ONE, CAP_TWO)
                          if code.startswith(c)), None)
        case = next((i for i in CAP_INPUTS if f"_i='{i}'" in code), None)
        if candidate is None or case is None:
            return False, "", "unmatched self-test"
        return CAP_SANDBOX_TABLE[(candidate, case)]

    def syntax_check(self, code, language, filename=""):
        return True, "", ""


class CaptureLLM:
    """Returns candidate zero's source for every probe attempt."""

    def __init__(self, progress_callback=None, thinking=False):
        pass

    def __call__(self, prompt, temperature, max_tokens, seed, thinking=None):
        return f"```python\n{CAP_ZERO}\n```", 5, 1.0


def _capture_cases():
    return [SimpleNamespace(input_str=i, expected_output=str(int(i) + 1))
            for i in CAP_INPUTS]


def _capture_service(monkeypatch):
    """A service whose pool is candidate zero plus two generated rivals."""
    monkeypatch.setattr(adapters, "LLMAdapter", CaptureLLM)
    monkeypatch.setattr(adapters, "SandboxAdapter", CaptureSandbox)
    monkeypatch.setattr(adapters, "EmbedAdapter", FakeEmbed)
    monkeypatch.setattr(scoring, "classify_task_type", lambda p: "algorithmic")
    monkeypatch.setattr(scoring, "score_candidate", lambda code: (1.0, 0.1, False))
    monkeypatch.setattr(scoring, "score_candidate_combined",
                        lambda code: dict(scoring.NEUTRAL_COMBINED))
    monkeypatch.setattr(scoring, "score_candidate_per_step",
                        lambda code: dict(CAP_PER_STEP))
    monkeypatch.setenv("ATLAS_V3_TELEMETRY_DIR", "off")

    service = v3pipeline.V3PipelineService()
    service.self_test_gen = SimpleNamespace(
        generate=lambda problem, llm, task_id:
            SimpleNamespace(test_cases=_capture_cases(), generation_tokens=0))
    service.plan_search = SimpleNamespace(
        generate=lambda problem, task_id, llm, num_plans=None,
        budget_tier="standard":
            SimpleNamespace(candidates=[CAP_ONE, CAP_TWO], total_tokens=0))
    service.pr_cot = SimpleNamespace(
        repair=lambda problem, code, error, llm_call, task_id:
            SimpleNamespace(repairs=[], total_tokens=0))
    service.refinement_loop = SimpleNamespace(
        run=lambda **kw: SimpleNamespace(solved=False, total_tokens=0,
                                         total_iterations=1, winning_code=""))
    return service


def _capture_records(path: Path):
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


def _of_type(records, kind):
    return [r for r in records if r.get("type") == kind]


def _run_captured(monkeypatch, tmp_path, task_id="cap"):
    sink = tmp_path / "pool.jsonl"
    monkeypatch.setenv(v3pipeline.CAPTURE_ENV, str(sink))
    service = _capture_service(monkeypatch)
    result = service.run("add one to the number on stdin", task_id=task_id,
                         file_path="/workspace/e2e/solve.py")
    return result, sink


def test_capture_is_inert_without_the_environment_variable(monkeypatch, tmp_path):
    """Unset means no sink is opened and no file appears, not an empty file."""
    monkeypatch.delenv(v3pipeline.CAPTURE_ENV, raising=False)
    capture = v3pipeline._PoolCapture.from_env()
    assert capture.enabled is False
    assert capture.path is None

    service = _capture_service(monkeypatch)
    service.run("add one", task_id="cap-off", file_path="/workspace/e2e/solve.py")
    assert list(tmp_path.iterdir()) == []


def test_exact_candidate_bytes_round_trip_with_hash_and_length(monkeypatch, tmp_path):
    """The pool is retained as bytes, not as a summary of bytes."""
    _, sink = _run_captured(monkeypatch, tmp_path)
    evaluations = _of_type(_capture_records(sink), "candidate_evaluation")
    assert evaluations, "no candidate_evaluation records written"

    seen = set()
    for rec in evaluations:
        raw = base64.b64decode(rec["code_b64"])
        assert raw.decode("utf-8") in (CAP_ZERO, CAP_ONE, CAP_TWO)
        assert rec["code_sha256"] == hashlib.sha256(raw).hexdigest()
        assert rec["code_bytes"] == len(raw)
        seen.add(raw.decode("utf-8"))
    assert seen == {CAP_ZERO, CAP_ONE, CAP_TWO}


def test_per_case_expected_and_actual_outcomes_are_retained(monkeypatch, tmp_path):
    """Input, generated expected, observed actual and a classified outcome --
    the four fields that decide whether the answer key or the code was wrong."""
    _, sink = _run_captured(monkeypatch, tmp_path)
    evaluations = _of_type(_capture_records(sink), "candidate_evaluation")
    by_code = {base64.b64decode(r["code_b64"]).decode(): r for r in evaluations}

    zero = by_code[CAP_ZERO]["oracle"]
    assert zero["cases_total"] == 5
    assert zero["cases_passed"] == 1
    outcomes = {c["input"]: c["outcome"] for c in zero["cases"]}
    assert outcomes == {"0": "pass", "1": "wrong_answer",
                        "2": "execution_error", "3": "timeout",
                        "4": "wrong_answer"}
    wrong = next(c for c in zero["cases"] if c["input"] == "1")
    assert wrong["expected"] == "2"
    assert wrong["actual"] == "2"
    assert wrong["passed"] is False

    one = by_code[CAP_ONE]["oracle"]
    assert (one["cases_passed"], one["cases_total"]) == (3, 5)
    two = by_code[CAP_TWO]["oracle"]
    assert (two["cases_passed"], two["cases_total"]) == (2, 5)


def test_passed_and_total_survive_the_boolean_the_pipeline_still_uses(
        monkeypatch, tmp_path):
    """Production keeps its accept/reject boolean; capture keeps the score
    that boolean discards."""
    _, sink = _run_captured(monkeypatch, tmp_path)
    evaluations = _of_type(_capture_records(sink), "candidate_evaluation")
    by_code = {base64.b64decode(r["code_b64"]).decode(): r for r in evaluations}
    # 2/5 and 1/5 are both rejected, and the record still tells them apart.
    assert by_code[CAP_TWO]["accepted"] is False
    assert by_code[CAP_ZERO]["accepted"] is False
    assert by_code[CAP_TWO]["oracle"]["cases_passed"] == 2
    assert by_code[CAP_ZERO]["oracle"]["cases_passed"] == 1


def test_candidate_zero_is_identified_as_such(monkeypatch, tmp_path):
    """Candidate zero is the probe, not merely index 0 of a sorted pool."""
    _, sink = _run_captured(monkeypatch, tmp_path)
    evaluations = _of_type(_capture_records(sink), "candidate_evaluation")
    zeros = [r for r in evaluations if r["role"] == "candidate_zero"]
    assert len(zeros) == 1
    assert base64.b64decode(zeros[0]["code_b64"]).decode() == CAP_ZERO
    assert {r["role"] for r in evaluations} >= {"candidate_zero", "generated"}


def test_selection_summary_names_only_captured_identities(monkeypatch, tmp_path):
    """A selection that references a hash no capture record carries cannot be
    adjudicated offline."""
    result, sink = _run_captured(monkeypatch, tmp_path)
    records = _capture_records(sink)
    summaries = _of_type(records, "selection_summary")
    assert len(summaries) == 1
    summary = summaries[0]
    known = {r["code_sha256"] for r in _of_type(records, "candidate_evaluation")}
    assert set(summary["pool"]) <= known
    assert summary["service_returned_candidate_hash"] in known
    assert summary["service_returned_candidate_hash"] == \
        hashlib.sha256(result["code"].encode()).hexdigest()
    for key in ("lens_index", "evidence_index", "verified_index",
                "selection_status", "selection_reason", "tied_count",
                "incomparable_count", "ineligible_count", "session_id",
                "phase"):
        assert key in summary, key


def test_capture_status_closes_the_file(monkeypatch, tmp_path):
    _, sink = _run_captured(monkeypatch, tmp_path)
    status = _of_type(_capture_records(sink), "capture_status")
    assert len(status) == 1
    assert status[0]["limit_reached"] is False
    assert status[0]["write_error"] == ""
    assert status[0]["records_written"] > 0
    assert status[0]["bytes_written"] > 0
    assert status[0]["max_bytes"] == v3pipeline.CAPTURE_DEFAULT_MAX_BYTES


def test_capture_changes_no_decision_and_no_public_surface(monkeypatch, tmp_path):
    """Off versus on must be indistinguishable to every caller."""
    monkeypatch.delenv(v3pipeline.CAPTURE_ENV, raising=False)
    off = _capture_service(monkeypatch).run(
        "add one", task_id="cap-cmp", file_path="/workspace/e2e/solve.py")
    _, sink = _run_captured(monkeypatch, tmp_path, task_id="cap-cmp")
    on = _capture_service(monkeypatch).run(
        "add one", task_id="cap-cmp", file_path="/workspace/e2e/solve.py")

    def comparable(r):
        return {k: v for k, v in r.items()
                if k not in ("total_time_ms", "events")}

    assert comparable(off) == comparable(on)
    assert [(e.get("stage"), e.get("detail")) for e in off["events"]] == \
        [(e.get("stage"), e.get("detail")) for e in on["events"]]

    # No candidate's source may ride out on an event, and the two rejected
    # candidates must not appear anywhere outside the sink.
    blob = json.dumps(on["events"])
    for code in (CAP_ZERO, CAP_TWO):
        assert code not in blob
        assert code[:20] not in blob
    assert sink.exists()


def test_a_capture_failure_never_changes_the_result(monkeypatch, tmp_path):
    """An unwritable sink degrades the diagnostic, never the pipeline."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file, not a directory")
    monkeypatch.setenv(v3pipeline.CAPTURE_ENV, str(blocker / "pool.jsonl"))
    broken = _capture_service(monkeypatch).run(
        "add one", task_id="cap-fail", file_path="/workspace/e2e/solve.py")

    monkeypatch.delenv(v3pipeline.CAPTURE_ENV, raising=False)
    clean = _capture_service(monkeypatch).run(
        "add one", task_id="cap-fail", file_path="/workspace/e2e/solve.py")
    assert broken["passed"] == clean["passed"]
    assert broken["code"] == clean["code"]
    assert broken["phase_solved"] == clean["phase_solved"]


def test_a_relative_path_or_missing_parent_disables_capture(monkeypatch, tmp_path):
    monkeypatch.setenv(v3pipeline.CAPTURE_ENV, "pool.jsonl")
    assert v3pipeline._PoolCapture.from_env().enabled is False

    monkeypatch.setenv(v3pipeline.CAPTURE_ENV, str(tmp_path / "gone" / "p.jsonl"))
    assert v3pipeline._PoolCapture.from_env().enabled is False


def test_capture_refuses_to_follow_a_pre_existing_symlink(monkeypatch, tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_text("")
    link = tmp_path / "pool.jsonl"
    link.symlink_to(target)
    monkeypatch.setenv(v3pipeline.CAPTURE_ENV, str(link))
    capture = v3pipeline._PoolCapture.from_env()
    assert capture.enabled is False
    assert target.read_text() == ""


def test_the_sink_is_owner_only(monkeypatch, tmp_path):
    _, sink = _run_captured(monkeypatch, tmp_path)
    mode = stat.S_IMODE(sink.stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_the_byte_cap_never_emits_a_partial_line(monkeypatch, tmp_path):
    """A record that will not fit is not written at all."""
    sink = tmp_path / "pool.jsonl"
    monkeypatch.setenv(v3pipeline.CAPTURE_ENV, str(sink))
    monkeypatch.setattr(v3pipeline, "CAPTURE_DEFAULT_MAX_BYTES", 900)
    service = _capture_service(monkeypatch)
    service.run("add one", task_id="cap-limit",
                file_path="/workspace/e2e/solve.py")

    text = sink.read_text()
    assert text.endswith("\n")
    records = _capture_records(sink)
    for line in text.splitlines():
        json.loads(line)          # every line is a complete record
    assert len(sink.read_bytes()) <= 900
    marker = _of_type(records, "capture_status")
    assert marker and marker[-1]["limit_reached"] is True


def test_concurrent_writers_append_complete_unique_records(tmp_path):
    """Two V3 worker processes share one sink; flock is the only reason the
    lines do not interleave."""
    sink = tmp_path / "pool.jsonl"
    writer = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(PROJECT_ROOT / "v3-service")!r})
        os.environ["ATLAS_V3_CAPTURE_POOL"] = {str(sink)!r}
        import pipeline
        cap = pipeline._PoolCapture.from_env()
        assert cap.enabled
        tag = sys.argv[1]
        for i in range(200):
            cap.write({{"type": "candidate_evaluation", "session_id": tag,
                       "candidate_index": i, "code_b64": "A" * 400}})
        cap.close()
    """)
    script = tmp_path / "writer.py"
    script.write_text(writer)
    procs = [subprocess.Popen([sys.executable, str(script), f"w{i}"])
             for i in range(4)]
    for p in procs:
        assert p.wait() == 0

    lines = [line for line in sink.read_text().splitlines() if line.strip()]
    parsed = [json.loads(line) for line in lines]
    evaluations = [r for r in parsed if r["type"] == "candidate_evaluation"]
    assert len(evaluations) == 800
    assert len({(r["session_id"], r["candidate_index"])
                for r in evaluations}) == 800


def test_capture_output_is_excluded_from_git():
    """The sink is a benchmark artifact; a committed pool is a code leak."""
    check = subprocess.run(
        ["git", "check-ignore", "-q",
         "redteam/runs/diagnostic/pool.jsonl"],
        cwd=str(PROJECT_ROOT))
    assert check.returncode == 0, "the capture location is not gitignored"


def test_candidate_bytes_reach_no_serialiser_or_emitter():
    """Source sentinel: the writer is referenced only where the pool exists.
    A capture call inside an emitter, a response builder or a log formatter is
    how benchmark-only bytes become production output."""
    v3dir = PROJECT_ROOT / "v3-service"
    owners = sorted(p.name for p in v3dir.glob("*.py")
                    if "_PoolCapture" in p.read_text()
                    or "CAPTURE_ENV" in p.read_text())
    assert owners == ["pipeline.py"], owners

    source = (v3dir / "pipeline.py").read_text()
    assert "code_b64" in source
    for module in ("main.py", "adapters.py", "contract.py", "scoring.py",
                   "structured_log.py"):
        text = (v3dir / module).read_text()
        assert "code_b64" not in text, module
        assert "ATLAS_V3_CAPTURE_POOL" not in text, module
