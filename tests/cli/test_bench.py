"""Tests for `atlas bench` — the first step of the model-onboarding loop
(bench → lens build --from-results → asa build).

The V3 runner itself needs a live llama-server, so it is faked at the
subprocess boundary: these tests pin down argument validation, the exact
runner invocation (command, env, cwd), and the exit-code contract for
failure / empty / successful runs.
"""

import json

import pytest

from atlas.commands import bench


class _FakeProc:
    """Stands in for the subprocess.Popen handle bench() drives."""

    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.terminated = False

    def wait(self):
        return self.returncode

    def terminate(self):
        self.terminated = True


def _fake_popen(monkeypatch, lines, returncode=0):
    """Replace subprocess.Popen inside bench() and capture its call."""
    calls = []

    def fake(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return _FakeProc(lines, returncode)

    # bench() does `import subprocess` at function scope, so patch the
    # module-level attribute it resolves to.
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", fake)
    return calls


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_negative_tasks_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        bench.main(["--tasks", "-3"])
    assert exc.value.code == 2
    assert "--tasks must be >= 0" in capsys.readouterr().err


def test_unknown_strategy_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        bench.main(["--strategy", "vibes"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Runner invocation contract
# ---------------------------------------------------------------------------

def test_runner_invocation_command_env_and_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "_atlas_root", lambda: tmp_path)
    calls = _fake_popen(monkeypatch, lines=[], returncode=1)

    bench.main(["--tasks", "5", "--strategy", "lens", "--run-id", "r1"])

    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert cmd[1:3] == ["-m", "benchmark.v3_runner"]
    assert "--baseline" in cmd
    assert cmd[cmd.index("--run-id") + 1] == "r1"
    assert cmd[cmd.index("--selection-strategy") + 1] == "lens"
    assert cmd[cmd.index("--max-tasks") + 1] == "5"
    # Repo root as cwd so the benchmark package resolves regardless of
    # where the operator invoked the CLI from.
    assert calls[0]["cwd"] == str(tmp_path)
    # Generation stays serialized — the safe default for any model.
    assert calls[0]["env"]["ATLAS_PARALLEL_TASKS"] == "1"


def test_zero_tasks_omits_max_tasks_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "_atlas_root", lambda: tmp_path)
    calls = _fake_popen(monkeypatch, lines=[], returncode=1)

    bench.main(["--run-id", "r2"])

    assert "--max-tasks" not in calls[0]["cmd"]


# ---------------------------------------------------------------------------
# Exit-code contract
# ---------------------------------------------------------------------------

def test_runner_failure_exits_nonzero_and_shows_tail(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(bench, "_atlas_root", lambda: tmp_path)
    _fake_popen(monkeypatch,
                lines=["Loading LiveCodeBench", "boom: connection refused"],
                returncode=3)

    rc = bench.main(["--run-id", "r3"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "exited with code 3" in out
    # The captured tail must surface the actual failure line.
    assert "connection refused" in out


def test_clean_exit_without_tasks_is_an_error(tmp_path, monkeypatch, capsys):
    """Runner exits 0 having processed nothing (aborted pre-flight):
    stale on-disk results must not be summarized as this run's output."""
    monkeypatch.setattr(bench, "_atlas_root", lambda: tmp_path)
    _fake_popen(monkeypatch, lines=[], returncode=0)

    rc = bench.main(["--run-id", "r4"])

    assert rc == 1
    assert "without processing any tasks" in capsys.readouterr().out


def test_successful_run_reports_results_and_exits_zero(tmp_path, monkeypatch,
                                                       capsys):
    monkeypatch.setattr(bench, "_atlas_root", lambda: tmp_path)
    per_task = tmp_path / "benchmark" / "results" / "r5" / "v3_lcb" / "per_task"
    per_task.mkdir(parents=True)
    (per_task / "t1.json").write_text(json.dumps({"passed": True}))
    (per_task / "t2.json").write_text(json.dumps({"passed": False}))

    _fake_popen(monkeypatch,
                lines=["[1/2] LCB t1: PASS",
                       "[2/2] LCB t2: FAIL",
                       "BENCHMARK COMPLETE"],
                returncode=0)

    rc = bench.main(["--run-id", "r5"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "pass@1: 1/2" in out
    # The next onboarding step is printed for the operator.
    assert "lens build --force --from-results" in out


def test_corrupt_result_files_are_counted_as_failures(tmp_path, monkeypatch,
                                                      capsys):
    monkeypatch.setattr(bench, "_atlas_root", lambda: tmp_path)
    per_task = tmp_path / "benchmark" / "results" / "r6" / "v3_lcb" / "per_task"
    per_task.mkdir(parents=True)
    (per_task / "good.json").write_text(json.dumps({"passed": True}))
    (per_task / "bad.json").write_text("{not json")

    _fake_popen(monkeypatch,
                lines=["[1/1] LCB t1: PASS", "BENCHMARK COMPLETE"],
                returncode=0)

    rc = bench.main(["--run-id", "r6"])

    assert rc == 0
    assert "pass@1: 1/2" in capsys.readouterr().out
