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

import adapters  # noqa: E402
import contract as C  # noqa: E402

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
    h = d / "h.js"; h.write_text(adapters.js_probe_source())
    f = d / "a.js"; f.write_text(code)
    runs = {}
    for mode in ("baseline", "input"):
        p = subprocess.run(["node", str(h), str(f), mode],
                           capture_output=True, text=True, timeout=120)
        runs[mode] = adapters.parse_probe_output(p.stdout)
    return adapters.combine_runs(runs["baseline"], runs["input"])


# ---- evidence model -------------------------------------------------------


def _record(ev, accepted=True):
    """The record the production path builds from one probe trace."""
    return adapters.contract_record(
        adapter=adapters.ADAPTER_BROWSER_CANVAS_JS, accepted=accepted, probe=ev,
        contract_id="probe", contract_version="1", artifact_scope="a.js",
        evaluation_context_hash=C.content_hash("ctx"),
        candidate_content_hash=C.content_hash("bytes"))



def test_syntax_evidence_never_permits_early_return():
    """Below the contract's floor is below it, whatever else is true."""
    for ev in (None, {"supported": True, "runtime_clean": False}):
        rec = _record(ev)
        assert rec["evidence_strength"] in (C.SYNTAX, C.RUNTIME)
        assert rec["closure_eligible"] is False


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
    rec = _record(ev)
    assert rec["evidence_strength"] == C.BEHAVIORAL, (rec, ev)
    assert not rec["missing_required"], \
        "both required criteria ARE satisfied — that is the trap"
    assert rec["overall_quality_score"] < 1.0
    assert rec["closure_eligible"] is False, \
        "partial behaviour must proceed to candidate generation"


def test_pins_the_observed_production_comparison():
    """Bare 1.00 vs ATLAS 0.75, measured on the real artifacts."""
    bare = {"supported": True, "runtime_clean": True, "temporal_progress": True,
            "input_causality": True, "collision_transition": True,
            "food_or_score_transition": True}
    atlas = {**bare, "food_or_score_transition": False}
    bare_rec, atlas_rec = _record(bare), _record(atlas)
    assert bare_rec["overall_quality_score"] == 1.0
    assert atlas_rec["overall_quality_score"] == 0.75
    assert bare_rec["closure_eligible"] is True
    assert atlas_rec["closure_eligible"] is False


def test_partial_behaviour_with_missing_required_does_not_return_early():
    """Replacing compile_passed with dom_probe_passed and keeping an
    unconditional return would reproduce the defect one level up."""
    missing_required = _record({"supported": True, "runtime_clean": True,
                                "temporal_progress": True,
                                "collision_transition": True})
    assert "input_causality" in missing_required["missing_required"]
    assert missing_required["closure_eligible"] is False
    below_quality = _record({"supported": True, "runtime_clean": True,
                             "temporal_progress": True, "input_causality": True,
                             "collision_transition": True})
    assert not below_quality["missing_required"]
    assert below_quality["overall_quality_score"] < 1.0
    assert below_quality["closure_eligible"] is False


def test_an_oracle_pass_still_earns_the_fast_path():
    rec = adapters.contract_record(
        adapter=adapters.ADAPTER_ALGORITHMIC_IO, accepted=True,
        contract_id="probe", contract_version="1", artifact_scope="s.py",
        evaluation_context_hash=C.content_hash("ctx"),
        candidate_content_hash=C.content_hash("bytes"))
    assert rec["evidence_strength"] == C.ORACLE
    assert rec["closure_eligible"] is True


# ---- the probe ------------------------------------------------------------

def test_working_game_demonstrates_required_behaviours():
    rec = _record(_probe(WORKING))
    assert not rec["missing_required"], rec
    assert C.STRENGTH_ORDER.index(rec["evidence_strength"]) >= \
        C.STRENGTH_ORDER.index(C.BEHAVIORAL)
    assert rec["overall_quality_score"] > 0.4


def test_inert_javascript_does_not_pass():
    rec = _record(_probe(INERT))
    assert rec["missing_required"]
    assert rec["closure_eligible"] is False
    assert rec["overall_quality_score"] == 0.0


def test_animation_ignoring_input_cannot_satisfy_causality():
    ev = _probe(ANIMATION_IGNORING_INPUT)
    assert ev["temporal_progress"] is True
    assert ev["input_causality"] is False, "a timer moving pixels is not input causality"
    assert "input_causality" in _record(ev)["missing_required"]


def test_runtime_exception_fails_the_probe():
    ev = _probe(THROWS)
    assert ev["runtime_clean"] is False
    rec = _record(ev)
    assert rec["evidence_strength"] == C.SYNTAX and rec["missing_required"]


def test_unsupported_artifacts_are_unverified_not_failed():
    assert not adapters.js_is_instrumentable("const fs = require('fs');")
    rec = _record({"supported": False})
    assert rec["evidence_strength"] == C.SYNTAX
    assert rec["missing_required"]
    assert rec["overall_quality_score"] == 0.0
    assert rec["supported"] is False
    assert rec["execution_status"] == C.EXEC_SKIPPED, "unsupported is not failed"


# ---- selection ------------------------------------------------------------

def test_behaviour_outranks_the_lens():
    """Ranking is the contract's, over records: coverage first, then quality."""
    better = _record({"supported": True, "runtime_clean": True,
                      "temporal_progress": True, "input_causality": True,
                      "collision_transition": True})
    prettier = _record({"supported": True, "runtime_clean": True,
                        "temporal_progress": True})
    assert C.select([prettier, better], better)["best_record"] is better
