"""Evidence strength comes from the verifier that ran, not the extension.

The first cut keyed off file extension and mapped every .py to
behavioral_complete. That is wrong for Pygame, Tkinter, curses and Flask:
they receive a compile smoke and nothing more, and would have closed the
pipeline claiming behaviour nobody demonstrated. It also routed .css through
a JavaScript probe and treated every .js as a canvas game.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import adapters as A  # noqa: E402
import contract as C  # noqa: E402


CANVAS_GAME = """
const c = document.getElementById('gameCanvas');
const ctx = c.getContext('2d');
document.addEventListener('keydown', e => {});
function loop(){ ctx.fillRect(0,0,10,10); setTimeout(loop, 50); } loop();
"""
NODE_SCRIPT = "const fs = require('fs');\nmodule.exports = function add(a,b){return a+b;};"
PLAIN_JS_HELPERS = "export function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }"
PYGAME = "import pygame\npygame.init()\nscreen = pygame.display.set_mode((640,480))"
TKINTER = "from tkinter import Tk\nroot = Tk()\nroot.mainloop()"
FLASK = "from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef home(): return 'hi'"
ALGO_PY = "import sys\nprint(sum(int(x) for x in open('input.txt')))"
HTML_INLINE = "<html><body><canvas id='c'></canvas><script>const x=document.getElementById('c');x.getContext('2d');document.addEventListener('keydown',()=>{});setTimeout(function f(){},10);</script></body></html>"
HTML_STATIC = "<html><body><h1>Hello</h1></body></html>"

_SCOPE = "static/game.js"
_CTX = None  # filled below, after contract import


def _record(adapter, accepted=True, probe=None):
    """The production path: raw observation inputs in, contract record out."""
    return A.contract_record(adapter=adapter, accepted=accepted, probe=probe,
                             contract_id="generate:js", contract_version="1",
                             artifact_scope=_SCOPE,
                             evaluation_context_hash=C.content_hash("ctx"),
                             candidate_content_hash=C.content_hash("bytes"))


def test_plain_js_is_not_automatically_a_canvas_game():
    assert A.select_adapter("util.js", PLAIN_JS_HELPERS) == A.ADAPTER_JAVASCRIPT_COMPILE
    assert A.select_adapter("build.js", NODE_SCRIPT) == A.ADAPTER_JAVASCRIPT_COMPILE
    assert A.select_adapter("game.js", CANVAS_GAME) == A.ADAPTER_BROWSER_CANVAS_JS


def test_interactive_python_never_gets_complete_evidence_from_compile():
    for src in (PYGAME, TKINTER, FLASK):
        adapter = A.select_adapter("app.py", src)
        assert adapter == A.ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED, src[:24]
        rec = _record(adapter, True)
        assert rec["evidence_strength"] == C.SYNTAX
        assert rec["supported"] is False
        assert rec["closure_eligible"] is False, \
            "compile smoke cannot close a Pygame artifact"


def test_algorithmic_python_with_an_oracle_keeps_the_fast_path():
    adapter = A.select_adapter("solve.py", ALGO_PY, has_io_oracle=True)
    assert adapter == A.ADAPTER_ALGORITHMIC_IO
    rec = _record(adapter, True)
    assert rec["evidence_strength"] == C.ORACLE
    assert rec["closure_eligible"] is True


def test_algorithmic_python_without_an_oracle_is_only_syntax():
    adapter = A.select_adapter("solve.py", ALGO_PY, has_io_oracle=False)
    assert adapter == A.ADAPTER_PYTHON_COMPILE
    assert _record(adapter, True)["closure_eligible"] is False


def test_css_is_never_sent_through_the_javascript_probe():
    adapter = A.select_adapter("style.css", "body { color: red; }")
    assert adapter == A.ADAPTER_CSS_SYNTAX
    rec = _record(adapter, True)
    assert rec["evidence_strength"] == C.SYNTAX
    # A stylesheet's contract closes on syntax: there is no behaviour to demand.
    assert A.closure_floor(adapter) == C.SYNTAX
    assert rec["closure_eligible"] is True


def test_jsx_and_tsx_are_unsupported_until_transpiled():
    for name in ("App.jsx", "App.tsx", "app.ts"):
        assert A.select_adapter(name, "const A = () => <div/>;") == A.ADAPTER_UNSUPPORTED


def test_html_routes_by_whether_it_has_instrumentable_inline_script():
    assert A.select_adapter("index.html", HTML_INLINE) == A.ADAPTER_BROWSER_INLINE_SCRIPT
    assert A.select_adapter("index.html", HTML_STATIC) == A.ADAPTER_UNSUPPORTED
    assert "getContext" in A.extract_inline_script(HTML_INLINE)


def test_browser_probe_that_could_not_run_is_unverified_not_failed():
    rec = _record(A.ADAPTER_BROWSER_CANVAS_JS, True, None)
    assert rec["supported"] is False
    assert rec["evidence_strength"] == C.SYNTAX
    assert rec["execution_status"] == C.EXEC_SKIPPED, \
        "the smoke check still passed — unverified, not failed"
    assert rec["closure_eligible"] is False


def test_complete_browser_behaviour_may_close_the_pipeline():
    full = {"supported": True, "runtime_clean": True, "temporal_progress": True,
            "input_causality": True, "collision_transition": True,
            "food_or_score_transition": True}
    rec = _record(A.ADAPTER_BROWSER_CANVAS_JS, True, full)
    assert rec["evidence_strength"] == C.BEHAVIORAL
    assert rec["overall_quality_score"] == 1.0
    assert rec["closure_eligible"] is True


def test_partial_browser_behaviour_may_not():
    partial = {"supported": True, "runtime_clean": True, "temporal_progress": True,
               "input_causality": True, "collision_transition": True,
               "food_or_score_transition": False}
    rec = _record(A.ADAPTER_BROWSER_CANVAS_JS, True, partial)
    assert rec["evidence_strength"] == C.BEHAVIORAL
    assert rec["overall_quality_score"] == 0.75
    assert rec["closure_eligible"] is False


# ---------------------------------------------------------------------------
# Direct contract-record production (evidence.py retirement, step 1)
# ---------------------------------------------------------------------------
#
# adapters.py no longer imports evidence.py: it declares its own capabilities
# and derives strength from the OBSERVATIONS rather than from that module's
# graded string. These characterize the swap over the heterogeneous adapter
# matrix -- every adapter, supported and unsupported, accepted and rejected,
# with and without a probe trace -- against the retiring implementation.
#
# The comparison itself is test-only and disappears with evidence.py; the
# production path never calls it, which the import sentinel proves.



def _probe(**flags):
    ev = {"supported": True, "runtime_clean": True,
          "temporal_progress": False, "input_causality": False,
          "collision_transition": False, "food_or_score_transition": False}
    ev.update(flags)
    return ev


def _matrix():
    """Every observation shape the pipeline can hand this layer: (name,
    adapter, smoke verdict, probe trace)."""
    cases = []
    for adapter in (A.ADAPTER_ALGORITHMIC_IO, A.ADAPTER_PYTHON_COMPILE, A.ADAPTER_JAVASCRIPT_COMPILE,
                    A.ADAPTER_CSS_SYNTAX, A.ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED, A.ADAPTER_UNSUPPORTED):
        for smoke in (True, False):
            cases.append((f"{adapter}:smoke={smoke}", adapter, smoke, None))
    for adapter in (A.ADAPTER_BROWSER_CANVAS_JS, A.ADAPTER_BROWSER_INLINE_SCRIPT):
        for smoke in (True, False):
            cases.append((f"{adapter}:no-probe:smoke={smoke}", adapter, smoke, None))
            for label, ev in (
                    ("all", _probe(temporal_progress=True, input_causality=True,
                                   collision_transition=True,
                                   food_or_score_transition=True)),
                    ("required-only", _probe(temporal_progress=True,
                                             input_causality=True)),
                    ("partial-required", _probe(temporal_progress=True)),
                    ("dirty-runtime", _probe(runtime_clean=False)),
                    ("unsupported-probe", _probe(supported=False))):
                cases.append((f"{adapter}:{label}:smoke={smoke}", adapter, smoke, ev))
    return cases


# The characterization values below were captured from the retiring
# implementation before it was deleted, and are asserted as literals now that
# there is nothing left to compare against. Adapter routing, supported vs
# unsupported, execution status and evidence strength for every observation
# shape the pipeline can produce.
CHARACTERIZED = {
    "algorithmic_io:smoke=True": ("oracle", "ok", True),
    "algorithmic_io:smoke=False": ("syntax", "error", False),
    "python_compile:smoke=True": ("syntax", "ok", True),
    "python_compile:smoke=False": ("syntax", "error", False),
    "javascript_compile:smoke=True": ("syntax", "ok", True),
    "javascript_compile:smoke=False": ("syntax", "error", False),
    "css_syntax:smoke=True": ("syntax", "ok", True),
    "css_syntax:smoke=False": ("syntax", "error", False),
    "interactive_python_unsupported:smoke=True": ("syntax", "skipped", False),
    "interactive_python_unsupported:smoke=False": ("syntax", "error", False),
    "unsupported:smoke=True": ("syntax", "skipped", False),
    "unsupported:smoke=False": ("syntax", "error", False),
}


def test_direct_production_matches_the_characterized_behaviour():
    """Adapter routing, supported/unsupported, execution status and evidence
    strength for every shape the pipeline can produce."""
    for name, adapter, smoke, probe in _matrix():
        rec = _record(adapter, smoke, probe)
        assert rec["adapter_id"] == adapter, name
        if name in CHARACTERIZED:
            want = CHARACTERIZED[name]
            got = (rec["evidence_strength"], rec["execution_status"], rec["supported"])
            assert got == want, f"{name}: {got} != {want}"
            continue
        # Browser families, keyed by what the probe demonstrated.
        if probe is None:
            assert rec["supported"] is False
            assert rec["execution_status"] in (C.EXEC_SKIPPED, C.EXEC_ERROR), name
        elif not probe.get("supported", True):
            assert rec["supported"] is False, name
        elif not probe.get("runtime_clean", True):
            assert rec["evidence_strength"] == C.SYNTAX, name
        elif all(probe.get(c) for c in A.BROWSER_REQUIRED):
            assert rec["evidence_strength"] == C.BEHAVIORAL, name
        else:
            assert rec["evidence_strength"] == C.RUNTIME, name


def test_direct_production_preserves_coverage_and_closure():
    """Criterion observations, required/missing/unmeasurable coverage, quality
    and closure eligibility follow from what the adapter reported."""
    for name, adapter, smoke, probe in _matrix():
        rec = _record(adapter, smoke, probe)
        caps = set(A._capabilities(adapter))
        obs = rec["observations"]

        # An adapter may only report on what it can measure.
        for cid, o in obs.items():
            if o["status"] in (C.DEMONSTRATED, C.REFUTED):
                assert cid in caps, f"{name}: {cid} reported outside capabilities"
        # Everything it cannot measure is unmeasurable, never silently missing.
        for r in rec["requirements"]:
            if r["required"] and r["id"] not in caps:
                assert obs[r["id"]]["status"] == C.NOT_APPLICABLE, name
                assert r["id"] in rec["missing_required"], name

        behavior = probe or {}
        for cid in A.BROWSER_REQUIRED:
            if cid in caps:
                demonstrated = obs[cid]["status"] == C.DEMONSTRATED
                assert demonstrated == bool(behavior.get(cid)), f"{name}:{cid}"

        # Closure follows contract policy, not the adapter's opinion.
        assert rec["closure_eligible"] == (
            rec["requirements_complete"] and rec["supported"]
            and rec["execution_status"] == C.EXEC_OK
            and C.STRENGTH_ORDER.index(rec["evidence_strength"])
            >= C.STRENGTH_ORDER.index(A.closure_floor(adapter))
            and rec["overall_quality_score"] >= 1.0), name
        assert 0.0 <= rec["overall_quality_score"] <= 1.0, name


def test_direct_production_carries_identity_and_hashes():
    for name, adapter, smoke, probe in _matrix():
        rec = _record(adapter, smoke, probe)
        C.require_identity(rec, name)
        assert rec["contract_id"] == "generate:js"
        assert rec["contract_version"] == "1"
        assert rec["artifact_scope"] == _SCOPE
        assert rec["evaluation_context_hash"] == C.content_hash("ctx")
        assert rec["candidate_content_hash"] == C.content_hash("bytes")
        assert rec["adapter_version"] == A.LIVE_ADAPTER_VERSION


def test_adapter_ids_are_the_ones_records_carry():
    """The wire values are part of the contract identity, so they are pinned
    as literals rather than compared against another copy."""
    assert A.ADAPTER_BROWSER_CANVAS_JS == "browser_canvas_js"
    assert A.ADAPTER_BROWSER_INLINE_SCRIPT == "browser_inline_script"
    assert A.ADAPTER_JAVASCRIPT_COMPILE == "javascript_compile"
    assert A.ADAPTER_CSS_SYNTAX == "css_syntax"
    assert A.ADAPTER_ALGORITHMIC_IO == "algorithmic_io"
    assert A.ADAPTER_PYTHON_COMPILE == "python_compile"
    assert A.ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED == "interactive_python_unsupported"
    assert A.ADAPTER_UNSUPPORTED == "unsupported"
    assert A.BROWSER_REQUIRED == ["temporal_progress", "input_causality"]
    assert A.BROWSER_OPTIONAL == ["collision_transition", "food_or_score_transition"]


def test_probe_mechanics_live_only_here():
    """The browser probe's machinery has exactly one home."""
    v3 = Path(__file__).resolve().parents[2] / "v3-service"
    for fn in ("def select_adapter(", "def js_is_instrumentable(",
               "def extract_inline_script(", "def js_probe_source_inline(",
               "def parse_probe_output(", "def combine_runs("):
        owners = [p.name for p in v3.glob("*.py") if fn in p.read_text()]
        assert owners == ["adapters.py"], f"{fn.strip()} owners: {owners}"
