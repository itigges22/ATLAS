"""Syntax validity is not behavioural verification.

The defect: one `passed` boolean carried every kind of evidence, so a
compile smoke on browser JavaScript returned from phase 0 with
candidates_generated=1 and PlanSearch / DivSampling / consensus / ranking
never ran. ATLAS paid orchestration latency without applying any of its
test-time compute.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import evidence  # noqa: E402
import pipeline  # noqa: E402

WORKING = """
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
let snake = [{x:10,y:10}]; let dx = 1, dy = 0;
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowUp') { dx = 0; dy = -1; }
  if (e.key === 'ArrowRight') { dx = 1; dy = 0; }
});
function loop(){
  const h = {x: snake[0].x + dx, y: snake[0].y + dy};
  if (h.x < 0 || h.x >= 20 || h.y < 0 || h.y >= 20) { alert('over'); return; }
  snake.unshift(h); snake.pop();
  ctx.fillRect(0,0,400,400);
  snake.forEach(p => ctx.fillRect(p.x*20, p.y*20, 18, 18));
  setTimeout(loop, 60);
}
loop();
"""

INERT = """
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
ctx.fillRect(0, 0, 400, 400);
document.addEventListener('keydown', function (e) { /* ignored */ });
"""

ANIMATION_IGNORING_INPUT = """
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
let p = 0;
document.addEventListener('keydown', function (e) { /* ignored */ });
function loop(){ p = (p + 5) % 400; ctx.fillRect(p, 10, 10, 10); setTimeout(loop, 40); }
loop();
"""

THROWS = "const c = document.getElementById('gameCanvas'); c.getContext('2d'); null.boom();"


def _probe(code):
    d = Path(tempfile.mkdtemp())
    h = d / "h.js"; h.write_text(evidence.js_probe_source())
    f = d / "a.js"; f.write_text(code)
    runs = {}
    for mode in ("baseline", "input"):
        p = subprocess.run(["node", str(h), str(f), mode],
                           capture_output=True, text=True, timeout=120)
        runs[mode] = evidence.parse_probe_output(p.stdout)
    return evidence.combine_runs(runs["baseline"], runs["input"])


# ---- evidence model -------------------------------------------------------

def test_syntax_evidence_never_permits_early_return():
    assert not evidence.may_return_early(evidence.SYNTAX, [], 0.0)
    assert not evidence.may_return_early(evidence.RUNTIME, [], 0.5)


LOOP_AND_INPUT_ONLY = """
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
let x = 5, dx = 1, dy = 0;
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowUp') { dx = 0; dy = -1; }
  if (e.key === 'ArrowRight') { dx = 1; dy = 0; }
});
function loop(){
  x += dx;                       // moves, responds to input
  ctx.fillRect(0, 0, 400, 400);  // ...but never collides, never scores
  ctx.fillRect((x % 20) * 20, (dy ? 40 : 20), 18, 18);
  setTimeout(loop, 60);
}
loop();
"""


def test_loop_and_input_without_collision_or_score_must_not_return_early():
    """The exact defect: no REQUIRED criterion is missing, so an earlier rule
    let this close the pipeline at 0.75 — measuring itself inferior to the
    bare model's 1.00 and then generating no alternatives."""
    ev = _probe(LOOP_AND_INPUT_ONLY)
    st, missing, score = evidence.grade_interactive(ev)
    assert st == evidence.BEHAVIORAL_PARTIAL, (st, ev)
    assert not missing, "both required criteria ARE satisfied — that is the trap"
    assert score < 1.0
    assert not evidence.may_return_early(st, missing, score), \
        "partial behaviour must proceed to candidate generation"


def test_pins_the_observed_production_comparison():
    """Bare 1.00 vs ATLAS 0.75, measured on the real artifacts."""
    bare = {"supported": True, "runtime_clean": True, "temporal_progress": True,
            "input_causality": True, "collision_transition": True,
            "food_or_score_transition": True}
    atlas = {**bare, "food_or_score_transition": False}
    bst, bmiss, bscore = evidence.grade_interactive(bare)
    ast_, amiss, ascore = evidence.grade_interactive(atlas)
    assert bscore == 1.0 and ascore == 0.75
    assert evidence.may_return_early(bst, bmiss, bscore)
    assert not evidence.may_return_early(ast_, amiss, ascore)


def test_partial_behaviour_with_missing_required_does_not_return_early():
    """Replacing compile_passed with dom_probe_passed and keeping an
    unconditional return would reproduce the defect one level up."""
    assert not evidence.may_return_early(
        evidence.BEHAVIORAL_PARTIAL, ["input_causality"], 0.5)
    assert not evidence.may_return_early(evidence.BEHAVIORAL_PARTIAL, [], 0.75)


def test_an_oracle_pass_still_earns_the_fast_path():
    assert evidence.may_return_early(evidence.BEHAVIORAL_COMPLETE, [], 1.0)


def test_interactive_artifacts_are_detected_by_artifact_not_task_wording():
    assert pipeline._is_interactive_artifact("game.js", WORKING)
    assert pipeline._is_interactive_artifact("index.html", "<canvas>")
    assert not pipeline._is_interactive_artifact("solve.py", "print(1)")


# ---- the probe ------------------------------------------------------------

def test_working_game_demonstrates_required_behaviours():
    st, missing, score = evidence.grade_interactive(_probe(WORKING))
    assert not missing, (st, missing)
    assert evidence.at_least(st, evidence.BEHAVIORAL_PARTIAL)
    assert score > 0.4


def test_inert_javascript_does_not_pass():
    st, missing, score = evidence.grade_interactive(_probe(INERT))
    assert missing
    assert not evidence.may_return_early(st, missing, score)
    assert score == 0.0


def test_animation_ignoring_input_cannot_satisfy_causality():
    ev = _probe(ANIMATION_IGNORING_INPUT)
    assert ev["temporal_progress"] is True
    assert ev["input_causality"] is False, "a timer moving pixels is not input causality"
    st, missing, _ = evidence.grade_interactive(ev)
    assert "input_causality" in missing


def test_runtime_exception_fails_the_probe():
    ev = _probe(THROWS)
    assert ev["runtime_clean"] is False
    st, missing, _ = evidence.grade_interactive(ev)
    assert st == evidence.SYNTAX and missing


def test_unsupported_artifacts_are_unverified_not_failed():
    assert not evidence.js_is_instrumentable("const fs = require('fs');")
    st, missing, score = evidence.grade_interactive({"supported": False})
    assert st == evidence.SYNTAX and missing and score == 0.0


# ---- selection ------------------------------------------------------------

def test_behaviour_outranks_the_lens():
    better = {"behavior_score": 0.75, "missing_required": [], "energy": 9.0,
              "evidence_strength": evidence.BEHAVIORAL_PARTIAL,
              "behavior": {"runtime_clean": True}}
    prettier = {"behavior_score": 0.25, "missing_required": ["input_causality"],
                "energy": 0.1, "evidence_strength": evidence.RUNTIME,
                "behavior": {"runtime_clean": True}}
    assert max([better, prettier], key=evidence.rank_key) is better
