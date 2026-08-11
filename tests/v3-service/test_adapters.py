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
