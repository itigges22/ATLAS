"""Contracts for V3 plan verification scoring."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "v3-service"))

import main as v3main  # noqa: E402


def _plan(command):
    return {
        "steps": [
            {"id": "s1", "action": "read_file", "target": "README.md"},
            {"id": "s2", "action": "run_command", "target": command},
        ],
        "verify_step": "s2",
        "rationale": "Inspect, then verify.",
    }


def test_plan_scorer_recognizes_language_specific_linters():
    for command in (
        "markdownlint README.md",
        "shellcheck scripts/setup.sh",
        "golangci-lint run ./...",
    ):
        score, reasons = v3main._score_plan(_plan(command), "fix README.md")
        assert score >= 0.9 - 1e-9
        assert "verify_step references a real verification command" in reasons


def test_plan_scorer_does_not_treat_recon_as_verification():
    score, reasons = v3main._score_plan(
        _plan("grep -n typo README.md"), "fix README.md"
    )
    assert score == pytest.approx(0.8)
    assert "verify_step doesn't reference a verification command" in reasons


# --- planning to recreate what already exists ---------------------------------
#
# Measured on aoc_sonar: the winning plan's step 1 was `write_file input.txt`
# — "create the necessary input data" — against a 2000-line fixture already on
# disk. The model tried to retype it from memory, degenerated into repeating
# one line, had its stream cut mid-JSON, and the run died on three unparseable
# responses. aoc_course executed the same step successfully and corrupted the
# fixture. Both plans scored 1.00, because nothing looked at what was there.

_CLOBBER_PLAN = {
    "steps": [
        {"id": "s1", "action": "write_file", "target": "input.txt",
         "why": "create the necessary input data"},
        {"id": "s2", "action": "write_file", "target": "solve.py", "why": "the solution"},
        {"id": "s3", "action": "run_command", "target": "python solve.py", "why": "verify"},
    ],
    "verify_step": "s3",
    "rationale": "creates input, implements, verifies",
}


def test_a_plan_that_recreates_an_existing_file_scores_below_one_that_does_not():
    clean, _ = v3main._score_plan(_CLOBBER_PLAN, "solve the puzzle", frozenset())
    clobber, reasons = v3main._score_plan(
        _CLOBBER_PLAN, "solve the puzzle", {"input.txt"})
    assert clobber < clean, (clobber, clean)
    assert any("already exist" in r for r in reasons), reasons
    assert any("input.txt" in r for r in reasons), reasons


def test_creating_a_genuinely_new_file_is_not_penalised():
    _, reasons = v3main._score_plan(
        _CLOBBER_PLAN, "solve the puzzle", {"README.md", "notes.txt"})
    assert not any("already exist" in r for r in reasons), reasons


def test_editing_an_existing_file_is_not_penalised():
    """Only CREATE actions clobber. Editing what is there is the correct plan."""
    plan = {
        "steps": [
            {"id": "s1", "action": "edit_file", "target": "input.txt", "why": "fix a line"},
            {"id": "s2", "action": "run_command", "target": "pytest", "why": "verify"},
        ],
        "verify_step": "s2", "rationale": "edit then verify",
    }
    _, reasons = v3main._score_plan(plan, "fix input.txt", {"input.txt"})
    assert not any("already exist" in r for r in reasons), reasons


def test_the_existing_file_set_reads_the_workspace(tmp_path):
    (tmp_path / "input.txt").write_text("199\n200\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "mod.py").write_text("x = 1\n")
    found = v3main._existing_workspace_files(str(tmp_path), {"ctx_only.py": "y = 2"})
    assert "input.txt" in found
    assert "sub/mod.py" in found
    assert "ctx_only.py" in found      # what the proxy shipped counts too
