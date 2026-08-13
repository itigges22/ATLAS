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

import evidence as E  # noqa: E402

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


def test_plain_js_is_not_automatically_a_canvas_game():
    assert E.select_adapter("util.js", PLAIN_JS_HELPERS) == E.JAVASCRIPT_COMPILE
    assert E.select_adapter("build.js", NODE_SCRIPT) == E.JAVASCRIPT_COMPILE
    assert E.select_adapter("game.js", CANVAS_GAME) == E.BROWSER_CANVAS_JS


def test_interactive_python_never_gets_complete_evidence_from_compile():
    for src in (PYGAME, TKINTER, FLASK):
        adapter = E.select_adapter("app.py", src)
        assert adapter == E.INTERACTIVE_PYTHON_UNSUPPORTED, src[:24]
        res = E.result_from_adapter(adapter, smoke_passed=True)
        assert res["strength"] == E.SYNTAX
        assert res["supported"] is False
        assert not E.may_return_early_result(res), "compile smoke cannot close a Pygame artifact"


def test_algorithmic_python_with_an_oracle_keeps_the_fast_path():
    adapter = E.select_adapter("solve.py", ALGO_PY, has_io_oracle=True)
    assert adapter == E.ALGORITHMIC_IO
    res = E.result_from_adapter(adapter, smoke_passed=True)
    assert res["strength"] == E.BEHAVIORAL_COMPLETE
    assert E.may_return_early_result(res)


def test_algorithmic_python_without_an_oracle_is_only_syntax():
    adapter = E.select_adapter("solve.py", ALGO_PY, has_io_oracle=False)
    assert adapter == E.PYTHON_COMPILE
    assert not E.may_return_early_result(E.result_from_adapter(adapter, True))


def test_css_is_never_sent_through_the_javascript_probe():
    adapter = E.select_adapter("style.css", "body { color: red; }")
    assert adapter == E.CSS_SYNTAX
    res = E.result_from_adapter(adapter, smoke_passed=True)
    assert res["strength"] == E.SYNTAX
    assert not E.may_return_early_result(res)


def test_jsx_and_tsx_are_unsupported_until_transpiled():
    for name in ("App.jsx", "App.tsx", "app.ts"):
        assert E.select_adapter(name, "const A = () => <div/>;") == E.UNSUPPORTED


def test_html_routes_by_whether_it_has_instrumentable_inline_script():
    assert E.select_adapter("index.html", HTML_INLINE) == E.BROWSER_INLINE_SCRIPT
    assert E.select_adapter("index.html", HTML_STATIC) == E.UNSUPPORTED
    assert "getContext" in E.extract_inline_script(HTML_INLINE)


def test_browser_probe_that_could_not_run_is_unverified_not_failed():
    res = E.result_from_adapter(E.BROWSER_CANVAS_JS, smoke_passed=True, probe_evidence=None)
    assert res["supported"] is False
    assert res["strength"] == E.SYNTAX
    assert res["accepted"] is True, "smoke still passed — it is unverified, not failed"
    assert not E.may_return_early_result(res)


def test_complete_browser_behaviour_may_close_the_pipeline():
    full = {"supported": True, "runtime_clean": True, "temporal_progress": True,
            "input_causality": True, "collision_transition": True,
            "food_or_score_transition": True}
    res = E.result_from_adapter(E.BROWSER_CANVAS_JS, True, full)
    assert res["strength"] == E.BEHAVIORAL_COMPLETE
    assert E.may_return_early_result(res)


def test_partial_browser_behaviour_may_not():
    partial = {"supported": True, "runtime_clean": True, "temporal_progress": True,
               "input_causality": True, "collision_transition": True,
               "food_or_score_transition": False}
    res = E.result_from_adapter(E.BROWSER_CANVAS_JS, True, partial)
    assert res["strength"] == E.BEHAVIORAL_PARTIAL
    assert res["behavior_score"] == 0.75
    assert not E.may_return_early_result(res)


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

import contract as C  # noqa: E402
import adapters as A  # noqa: E402

_SCOPE = "static/game.js"
_CTX = C.content_hash("build something")
_CODE = C.content_hash("const a = 1;\n")


def _record(live_record):
    return A.contract_record(live_record, contract_id="generate:js",
                             contract_version="1", artifact_scope=_SCOPE,
                             evaluation_context_hash=_CTX,
                             candidate_content_hash=_CODE)


def _retiring_strength(adapter, strength):
    """What the retiring bridge mapped a prototype grade onto."""
    if strength == E.NONE:
        return None
    if adapter == E.ALGORITHMIC_IO and strength == E.BEHAVIORAL_COMPLETE:
        return C.ORACLE
    return {E.BEHAVIORAL_COMPLETE: C.BEHAVIORAL, E.BEHAVIORAL_PARTIAL: C.BEHAVIORAL,
            E.RUNTIME: C.RUNTIME, E.SYNTAX: C.SYNTAX}[strength]


def _retiring_expectation(live_record):
    """(strength, execution_status, supported) as the retiring path produced."""
    strength = _retiring_strength(live_record["adapter"], live_record["strength"])
    supported = bool(live_record.get("supported", True))
    if strength is None:
        return C.SYNTAX, C.EXEC_ERROR, False
    if not supported:
        return strength, C.EXEC_SKIPPED, False
    return strength, C.EXEC_OK, True


def _probe(**flags):
    ev = {"supported": True, "runtime_clean": True,
          "temporal_progress": False, "input_causality": False,
          "collision_transition": False, "food_or_score_transition": False}
    ev.update(flags)
    return ev


def _matrix():
    """Every record shape result_from_adapter can actually emit."""
    cases = []
    for adapter in (E.ALGORITHMIC_IO, E.PYTHON_COMPILE, E.JAVASCRIPT_COMPILE,
                    E.CSS_SYNTAX, E.INTERACTIVE_PYTHON_UNSUPPORTED, E.UNSUPPORTED):
        for smoke in (True, False):
            cases.append((f"{adapter}:smoke={smoke}",
                          E.result_from_adapter(adapter, smoke)))
    for adapter in (E.BROWSER_CANVAS_JS, E.BROWSER_INLINE_SCRIPT):
        for smoke in (True, False):
            cases.append((f"{adapter}:no-probe:smoke={smoke}",
                          E.result_from_adapter(adapter, smoke)))
            for label, ev in (
                    ("all", _probe(temporal_progress=True, input_causality=True,
                                   collision_transition=True,
                                   food_or_score_transition=True)),
                    ("required-only", _probe(temporal_progress=True,
                                             input_causality=True)),
                    ("partial-required", _probe(temporal_progress=True)),
                    ("dirty-runtime", _probe(runtime_clean=False)),
                    ("unsupported-probe", _probe(supported=False))):
                cases.append((f"{adapter}:{label}:smoke={smoke}",
                              E.result_from_adapter(adapter, smoke, ev)))
    return cases


def test_direct_production_matches_the_retiring_implementation():
    """Adapter routing, supported/unsupported, execution status and evidence
    strength are preserved for every shape the pipeline can produce."""
    for name, live_record in _matrix():
        rec = _record(live_record)
        want_strength, want_exec, want_supported = _retiring_expectation(live_record)
        assert rec["adapter_id"] == live_record["adapter"], name
        assert rec["evidence_strength"] == want_strength, name
        assert rec["execution_status"] == want_exec, name
        assert rec["supported"] is want_supported, name


def test_direct_production_preserves_coverage_and_closure():
    """Criterion observations, required/missing/unmeasurable coverage, quality
    and closure eligibility follow from what the adapter reported."""
    for name, live_record in _matrix():
        rec = _record(live_record)
        adapter = live_record["adapter"]
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

        behavior = live_record.get("behavior") or {}
        for cid in A.BROWSER_REQUIRED:
            if cid in caps:
                demonstrated = obs[cid]["status"] == C.DEMONSTRATED
                assert demonstrated == bool(behavior.get(cid)), f"{name}:{cid}"

        # Closure follows contract policy, not the adapter's opinion.
        assert rec["closure_eligible"] == (
            rec["requirements_complete"] and rec["supported"]
            and rec["execution_status"] == C.EXEC_OK
            and C.STRENGTH_ORDER.index(rec["evidence_strength"])
            >= C.STRENGTH_ORDER.index(C.BEHAVIORAL)
            and rec["overall_quality_score"] >= 1.0), name
        assert 0.0 <= rec["overall_quality_score"] <= 1.0, name


def test_direct_production_carries_identity_and_hashes():
    for name, live_record in _matrix():
        rec = _record(live_record)
        C.require_identity(rec, name)
        assert rec["contract_id"] == "generate:js"
        assert rec["contract_version"] == "1"
        assert rec["artifact_scope"] == _SCOPE
        assert rec["evaluation_context_hash"] == _CTX
        assert rec["candidate_content_hash"] == _CODE
        assert rec["adapter_version"] == A.LIVE_ADAPTER_VERSION


def test_adapter_ids_and_criteria_match_the_retiring_module():
    """While evidence.py exists, the ids this layer declares must be the ones
    the pipeline's records actually carry."""
    assert A.ADAPTER_BROWSER_CANVAS_JS == E.BROWSER_CANVAS_JS
    assert A.ADAPTER_BROWSER_INLINE_SCRIPT == E.BROWSER_INLINE_SCRIPT
    assert A.ADAPTER_JAVASCRIPT_COMPILE == E.JAVASCRIPT_COMPILE
    assert A.ADAPTER_CSS_SYNTAX == E.CSS_SYNTAX
    assert A.ADAPTER_ALGORITHMIC_IO == E.ALGORITHMIC_IO
    assert A.ADAPTER_PYTHON_COMPILE == E.PYTHON_COMPILE
    assert A.ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED == E.INTERACTIVE_PYTHON_UNSUPPORTED
    assert A.ADAPTER_UNSUPPORTED == E.UNSUPPORTED
    assert A.BROWSER_REQUIRED == E.INTERACTIVE_REQUIRED
    assert A.BROWSER_OPTIONAL == E.INTERACTIVE_OPTIONAL


def test_production_record_path_never_calls_the_retiring_module():
    """The sentinel: no import, and no call, even indirectly. Only the test
    above may reference evidence.py from this layer."""
    import re
    src = (Path(__file__).resolve().parents[2] / "v3-service" / "adapters.py").read_text()
    assert "import evidence" not in src, "adapters.py imports the retiring module"
    assert "_live." not in src, "adapters.py still holds the retiring alias"
    # Attribute access on the module, ignoring prose that names the file.
    hit = re.search(r"\bevidence\.(?!py)\w+", src)
    assert hit is None, f"adapters.py calls into evidence.py: {hit and hit.group(0)}"
