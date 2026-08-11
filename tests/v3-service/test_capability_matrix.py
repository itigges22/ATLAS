"""Heterogeneous capability matrix: where can ATLAS produce trustworthy evidence?

Everything in this line of work was shaped by one Snake prompt, so the risk
is an architecture built around canvas games by accident. This matrix asks a
narrower question than the benchmark: across task families, which adapter is
selected, what strength can it honestly claim, and where must it say
unsupported or inconclusive instead of guessing.

Deterministic fixtures, no model, no sandbox. The invariant that matters
most is the false-behavioral_complete rate: zero.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import evidence as E  # noqa: E402

# --- fixtures by family ----------------------------------------------------

ALGO_PY = "print(sum(int(x) for x in open('input.txt')))"
CLI_PY = "import sys\nprint('usage: tool <file>') if len(sys.argv) < 2 else print('ok')"
ALGO_JS = "const fs=require('fs');console.log(fs.readFileSync(0,'utf8').trim().length);"
NODE_MODULE = "const fs=require('fs');module.exports=(a,b)=>a+b;"
CANVAS_GAME = ("const c=document.getElementById('gameCanvas');const x=c.getContext('2d');"
               "document.addEventListener('keydown',e=>{});"
               "function loop(){x.fillRect(0,0,9,9);setTimeout(loop,50);}loop();")
DOM_APP = ("const btn=document.querySelector('#go');"
           "btn.addEventListener('click',()=>{document.getElementById('out').textContent='hi';});")
STATIC_HTML = "<html><body><h1>Docs</h1><p>Hello</p></body></html>"
HTML_INLINE = ("<html><body><canvas id='c'></canvas><script>"
               "const q=document.getElementById('c');q.getContext('2d');"
               "document.addEventListener('keydown',()=>{});setTimeout(function f(){},9);"
               "</script></body></html>")
HTML_EXTERNAL = "<html><body><canvas id='c'></canvas><script src='game.js'></script></body></html>"
CSS = "body { margin: 0; background: #111; }"
JSX = "export default function App(){ return <div className='x'>hi</div>; }"
FLASK_API = ("from flask import Flask\napp=Flask(__name__)\n"
             "@app.route('/health')\ndef h(): return {'ok': True}")
PYGAME = "import pygame\npygame.init()\nscreen=pygame.display.set_mode((320,240))"
TKINTER = "from tkinter import Tk\nroot=Tk()\nroot.mainloop()"
CONFIG_JSON = '{"debug": false, "retries": 3}'
UNSUPPORTED_LANG = "package main\nimport \"fmt\"\nfunc main(){ fmt.Println(1) }"

# task_family, fixture, path, has_oracle, expected_adapter, may_claim_behaviour
MATRIX = [
    ("algorithmic_io_python",   ALGO_PY,          "solve.py",   True,  E.ALGORITHMIC_IO,                 True),
    ("algorithmic_io_python_no_oracle", ALGO_PY,  "solve.py",   False, E.PYTHON_COMPILE,                 False),
    ("cli_python",              CLI_PY,           "tool.py",    False, E.PYTHON_COMPILE,                 False),
    ("algorithmic_io_js",       ALGO_JS,          "solve.js",   False, E.JAVASCRIPT_COMPILE,             False),
    ("node_module",             NODE_MODULE,      "util.js",    False, E.JAVASCRIPT_COMPILE,             False),
    ("canvas_game",             CANVAS_GAME,      "game.js",    False, E.BROWSER_CANVAS_JS,              True),
    ("dom_app_no_canvas",       DOM_APP,          "app.js",     False, E.JAVASCRIPT_COMPILE,             False),
    ("static_html",             STATIC_HTML,      "index.html", False, E.UNSUPPORTED,                    False),
    ("html_inline_script",      HTML_INLINE,      "index.html", False, E.BROWSER_INLINE_SCRIPT,          True),
    ("html_external_script",    HTML_EXTERNAL,    "index.html", False, E.UNSUPPORTED,                    False),
    ("css",                     CSS,              "style.css",  False, E.CSS_SYNTAX,                     False),
    ("react_jsx",               JSX,              "App.jsx",    False, E.UNSUPPORTED,                    False),
    ("backend_api",             FLASK_API,        "api.py",     False, E.INTERACTIVE_PYTHON_UNSUPPORTED, False),
    ("pygame",                  PYGAME,           "game.py",    False, E.INTERACTIVE_PYTHON_UNSUPPORTED, False),
    ("tkinter",                 TKINTER,          "ui.py",      False, E.INTERACTIVE_PYTHON_UNSUPPORTED, False),
    ("config_data",             CONFIG_JSON,      "config.json", False, E.UNSUPPORTED,                   False),
    ("unsupported_language",    UNSUPPORTED_LANG, "main.go",    False, E.UNSUPPORTED,                    False),
]


@pytest.mark.parametrize("family,code,path,oracle,expected_adapter,may_claim", MATRIX,
                         ids=[m[0] for m in MATRIX])
def test_adapter_routing(family, code, path, oracle, expected_adapter, may_claim):
    assert E.select_adapter(path, code, oracle) == expected_adapter, family


@pytest.mark.parametrize("family,code,path,oracle,expected_adapter,may_claim", MATRIX,
                         ids=[m[0] for m in MATRIX])
def test_no_family_claims_behaviour_it_cannot_demonstrate(
        family, code, path, oracle, expected_adapter, may_claim):
    """The invariant that matters: false behavioral_complete rate is zero.

    Every family gets a smoke pass and NO behavioural probe result. Only the
    families with a real oracle may reach complete on that basis.
    """
    res = E.result_from_adapter(expected_adapter, smoke_passed=True, probe_evidence=None)
    if may_claim and expected_adapter == E.ALGORITHMIC_IO:
        assert res["strength"] == E.BEHAVIORAL_COMPLETE
        assert E.may_return_early_result(res)
    else:
        assert res["strength"] in (E.SYNTAX, E.RUNTIME), family
        assert not E.may_return_early_result(res), family


@pytest.mark.parametrize("family,code,path,oracle,expected_adapter,may_claim", MATRIX,
                         ids=[m[0] for m in MATRIX])
def test_unsupported_is_never_represented_as_failed(
        family, code, path, oracle, expected_adapter, may_claim):
    res = E.result_from_adapter(expected_adapter, smoke_passed=True, probe_evidence=None)
    assert res["accepted"] is True, f"{family}: smoke passed, so it is unverified not failed"


def test_the_browser_probe_is_never_dispatched_outside_its_capability():
    """An adapter must not execute an artifact outside what it declares."""
    for family, code, path, oracle, adapter, _ in MATRIX:
        if adapter in (E.BROWSER_CANVAS_JS, E.BROWSER_INLINE_SCRIPT):
            continue
        target = E.extract_inline_script(code) if path.endswith(".html") else code
        assert not E.js_is_instrumentable(target), \
            f"{family} routed to {adapter} but looks instrumentable — routing is inconsistent"


def test_the_generic_policy_functions_carry_no_domain_vocabulary():
    """Strength ordering, coverage and ranking must be prompt-agnostic.

    KNOWN LIMITATION, asserted rather than hidden: the required/optional
    CRITERIA NAMES are still browser-game specific --
    `collision_transition` and `food_or_score_transition` sit in the shared
    contract instead of being supplied by the adapter that can measure them.
    That is the Snake-shaped residue this matrix exists to surface, and it
    must move into an adapter-declared contract before a second behavioural
    adapter is written. The policy FUNCTIONS below are clean today; this test
    fails the moment domain words spread into them.
    """
    src = Path(E.__file__).read_text()
    for fn in ("def at_least(", "def may_return_early(", "def rank_key(",
               "def selection_mode(", "def probing_enabled(", "def selection_enabled("):
        start = src.index(fn)
        nxt = src.find("\n\n", start)
        body = src[start:nxt if nxt != -1 else len(src)]
        for word in ("snake", "food", "collision", "game"):
            assert word not in body.lower(), f"'{word}' leaked into {fn}"


def test_known_limitation_criteria_names_are_still_browser_game_specific():
    """Pin the residue so it cannot be forgotten or silently widened."""
    assert E.INTERACTIVE_OPTIONAL == ["collision_transition", "food_or_score_transition"]
    assert E.INTERACTIVE_REQUIRED == ["temporal_progress", "input_causality"]


def test_multi_file_behaviour_is_not_claimed_from_one_file():
    """HTML whose behaviour lives in an external file must not be graded from
    the HTML alone — that requires project-aware evaluation, which does not
    exist yet."""
    assert E.select_adapter("index.html", HTML_EXTERNAL) == E.UNSUPPORTED
    res = E.result_from_adapter(E.UNSUPPORTED, smoke_passed=True)
    assert res["supported"] is False
    assert not E.may_return_early_result(res)
