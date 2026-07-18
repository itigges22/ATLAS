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
