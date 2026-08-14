"""Agreement, not the model's guess, decides when the answer key is wrong.

The self-test compares a candidate's output to an expected value the same
model produced. For a task it cannot reliably solve, producing the answer IS
the task, so the key is wrong and correct code fails its own suite.

Measured in one run: 42 self-test verifications, every one 0/N, and V3
selected a candidate in 0 of 12 sessions — every session shipped the model's
own draft. A candidate pulled from those logs passed immediately when given a
correct expected value, so the code was never the problem.

CodeT (Chen et al., 2022), already cited for this pipeline, does not use
generated tests as an oracle. It runs candidates on the generated INPUTS and
clusters them by agreement. These tests pin that behaviour.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import pipeline  # noqa: E402

CORRECT_A = ("def main():\n"
             "    d = [int(x) for x in open('input.txt')]\n"
             "    print(sum(1 for i in range(1, len(d)) if d[i] > d[i-1]))\n"
             "main()\n")
CORRECT_B = ("def main():\n"
             "    n = [int(v) for v in open('input.txt').read().split()]\n"
             "    print(len([1 for i in range(1, len(n)) if n[i] > n[i-1]]))\n"
             "main()\n")
WRONG = ("import sys\n"
         "def main():\n"
         "    print(len([int(x) for x in sys.stdin]))\n"
         "main()\n")


def _sandbox(code):
    work = tempfile.mkdtemp()
    Path(work, "t.py").write_text(code)
    run = subprocess.run([sys.executable, "t.py"], cwd=work,
                         capture_output=True, text=True, timeout=60)
    return run.returncode == 0, run.stdout, run.stderr


def _case(expected="THIS KEY IS WRONG"):
    return type("TC", (), {"input_str": "199\n200\n208\n210\n200",
                           "expected_output": expected})()


def _case3(expected):
    return type("TC", (), {"input_str": "1\n2\n3",
                           "expected_output": expected})()


def _cands(*codes):
    return [{"index": i, "code": c} for i, c in enumerate(codes)]


def test_agreeing_candidates_win_despite_a_wrong_key():
    """The measured situation: the key is wrong, the code is not."""
    winners = pipeline._consensus_winners(
        _cands(CORRECT_A, WRONG, CORRECT_B), [_case()], _sandbox, lambda *a, **k: None)
    assert sorted(c["index"] for c in winners) == [0, 2]


def test_one_candidate_cannot_agree_with_itself():
    """Agreement between independent programs is evidence; a single program
    reproducing its own output is not."""
    assert pipeline._consensus_winners(
        _cands(CORRECT_A), [_case()], _sandbox, lambda *a, **k: None) == []


def test_total_disagreement_selects_nothing():
    """Three answers, three clusters of one — no majority to trust."""
    a = "print(1)\n"
    b = "print(2)\n"
    c = "print(3)\n"
    assert pipeline._consensus_winners(
        _cands(a, b, c), [_case()], _sandbox, lambda *a, **k: None) == []


def test_no_test_cases_selects_nothing():
    assert pipeline._consensus_winners(
        _cands(CORRECT_A, CORRECT_B), [], _sandbox, lambda *a, **k: None) == []


def test_a_crashing_candidate_does_not_join_a_cluster():
    boom = "raise SystemExit(1)\n"
    winners = pipeline._consensus_winners(
        _cands(CORRECT_A, boom, CORRECT_B), [_case()], _sandbox, lambda *a, **k: None)
    assert sorted(c["index"] for c in winners) == [0, 2]


def test_the_probe_reports_output_rather_than_asserting():
    probe = pipeline._make_output_probe(CORRECT_A, _case())
    assert pipeline._CONSENSUS_MARK in probe
    assert "SELF_TEST_PASS" not in probe
    assert "THIS KEY IS WRONG" not in probe, "the probe must not see the key"


def test_the_consensus_stage_is_registered():
    """An unregistered stage contributes no phase row, so the run summary
    loses the reason a candidate was selected."""
    assert pipeline._STAGE_PHASE["consensus"] == "selection"


# --- verification holds candidates to the ENVIRONMENT's contract -----------
#
# The self-test used to pick its shape from the candidate's own code, so a
# stdin-reading candidate was handed a stdin the real run never provides. It
# passed verification and was selected in 3 sessions; all 3 then failed the
# task, printing 0 because no stdin arrived. The caller runs the program with
# the task's files on disk and no stdin, and verification has to do the same.

STDIN_CANDIDATE = "import sys\nprint(len([x for x in sys.stdin]))\n"
FILE_CANDIDATE = "print(len(open('input.txt').read().split()))\n"


def _run(built):
    """`built` is a self-test (wrapper, files) pair or a bare probe body."""
    body, files = built if isinstance(built, tuple) else (built, {})
    work = tempfile.mkdtemp()
    for name, content in (files or {}).items():
        Path(work, name).write_text(content)
    Path(work, "t.py").write_text(body)
    return subprocess.run([sys.executable, "t.py"], cwd=work,
                          capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=60)


def test_the_task_input_file_is_recognised():
    assert pipeline._task_input_file({"input.txt": "1\n"}) == "input.txt"
    assert pipeline._task_input_file({"solve.py": "x=1"}) == ""
    assert pipeline._task_input_file(None) == ""


def test_a_stdin_candidate_fails_the_task_contract():
    """The measured regression: it passed on a stdin the caller never gives."""
    body = pipeline._make_self_test(STDIN_CANDIDATE, _case3("3"), "input.txt")
    assert "SELF_TEST_PASS" not in _run(body).stdout


def test_a_file_candidate_passes_the_task_contract():
    body = pipeline._make_self_test(FILE_CANDIDATE, _case3("3"), "input.txt")
    assert "SELF_TEST_PASS" in _run(body).stdout


def test_without_a_task_input_file_the_old_shapes_stand():
    """Tasks that really are stdin-driven must keep working."""
    body, _files = pipeline._make_self_test(STDIN_CANDIDATE, _case3("3"))
    assert "_s.stdin=" in body


def test_the_probe_uses_the_same_contract():
    probe = pipeline._make_output_probe(STDIN_CANDIDATE, _case(), "input.txt")
    assert "open('input.txt','w')" in probe
    # stdin is attached-and-empty under the task contract; the case input
    # must never ride it.
    assert "_s.stdin=_o.StringIO('')" in probe
    assert "199" not in probe.split("open(")[0], "case input stays out of the stdin setup"
