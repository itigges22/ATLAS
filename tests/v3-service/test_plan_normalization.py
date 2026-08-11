"""A plan must not manufacture an edit it could have avoided.

Audited on the literal prompt "Build me a snake game.", the planner emitted
`write_file index.html` ... `edit_file index.html "Link the CSS and JS
files"`. index.html is greenfield in that same plan and the paths are known
at planning time, so the links belong in the initial write. The split
creates an exact-span edit for a quantized model that measurably cannot do
them — one dogfood session died looping on that exact edit_file/index.html
pair. Its verify_step was `python3 -m http.server 8000`, which is setup and
cannot fail on an inert page.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import planning  # noqa: E402

SNAKE_PLAN = {
    "steps": [
        {"id": "s1", "action": "write_file", "target": "index.html", "why": "Create the canvas page."},
        {"id": "s2", "action": "write_file", "target": "game.js", "why": "Game logic."},
        {"id": "s3", "action": "write_file", "target": "style.css", "why": "Styling."},
        {"id": "s4", "action": "edit_file", "target": "index.html", "why": "Link the CSS and JS files."},
        {"id": "s5", "action": "run_command", "target": "python3 -m http.server 8000", "why": "Serve it."},
    ],
    "verify_step": "s5",
}


def test_greenfield_write_then_edit_is_collapsed():
    plan, notes = planning.normalize_plan(dict(SNAKE_PLAN))
    actions = [(s["action"], s["target"]) for s in plan["steps"]]
    assert ("edit_file", "index.html") not in actions, actions
    assert len(plan["steps"]) == 4
    # The collapsed intent survives on the initial write.
    s1 = next(s for s in plan["steps"] if s["id"] == "s1")
    assert "Link the CSS and JS" in s1["why"]
    assert any("collapsed" in n for n in notes)


def test_server_start_only_verification_is_flagged_and_penalised():
    plan, notes = planning.normalize_plan(dict(SNAKE_PLAN))
    assert plan.get("verify_is_setup_only") is True
    assert any("setup" in n for n in notes)
    score, reasons = planning._score_plan(plan, "Build me a snake game.")
    assert any("setup, not verification" in r for r in reasons), reasons


def test_an_edit_to_a_preexisting_file_is_untouched():
    plan = {
        "steps": [
            {"id": "s1", "action": "read_file", "target": "app.py", "why": "Look."},
            {"id": "s2", "action": "edit_file", "target": "app.py", "why": "Fix the bug."},
            {"id": "s3", "action": "run_command", "target": "pytest tests/", "why": "Verify."},
        ],
        "verify_step": "s3",
    }
    out, notes = planning.normalize_plan(dict(plan))
    assert len(out["steps"]) == 3, "an edit to a file this plan did not create must stand"
    assert not notes


def test_a_real_verification_is_not_penalised():
    plan = {
        "steps": [
            {"id": "s1", "action": "write_file", "target": "solve.py", "why": "Write it."},
            {"id": "s2", "action": "run_command", "target": "python3 solve.py", "why": "Run it."},
        ],
        "verify_step": "s2",
    }
    out, _ = planning.normalize_plan(dict(plan))
    assert not out.get("verify_is_setup_only")
