"""Gate 1, reproducible: the behavioural probe against the LIVE sandbox.

Opt-in, because it needs a running sandbox and must not make ordinary CI
environment-dependent:

    ATLAS_LIVE_SANDBOX=1 python3 -m pytest tests/live -q -s

No mocked network calls. It drives the production SandboxAdapter, records
the sandbox/runtime identity and per-arm duration, and asserts the minimum
behavioural distinctions the evidence policy depends on. It exists because
the capability was once "verified" by a stub that accepted a keyword the
real adapter rejected — a manual observation nobody can re-run is the same
class of gap.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import adapters  # noqa: E402
import adapters  # noqa: E402
import contract as _contract  # noqa: E402
import pipeline as P  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("ATLAS_LIVE_SANDBOX", "0") != "1",
    reason="live sandbox test: set ATLAS_LIVE_SANDBOX=1 to run")

RESPONSIVE = """
const c=document.getElementById('gameCanvas');const x=c.getContext('2d');
let px=5,dx=1,dy=0;
document.addEventListener('keydown',e=>{ if(e.key==='ArrowUp'){dx=0;dy=-1;} if(e.key==='ArrowRight'){dx=1;dy=0;} });
function loop(){ px+=dx; x.fillRect(0,0,400,400); x.fillRect((px%20)*20,(dy?40:20),18,18); setTimeout(loop,60);} loop();
"""
IGNORES_INPUT = """
const c=document.getElementById('gameCanvas');const x=c.getContext('2d');let p=0;
document.addEventListener('keydown',function(e){});
function loop(){p=(p+5)%400;x.fillRect(p,10,10,10);setTimeout(loop,40);} loop();
"""
INERT = ("const c=document.getElementById('gameCanvas');const x=c.getContext('2d');"
         "x.fillRect(0,0,400,400);document.addEventListener('keydown',()=>{});")
THROWS = "const c=document.getElementById('gameCanvas');c.getContext('2d');null.boom();"
NODE_MODULE = "const fs=require('fs');module.exports=(a,b)=>a+b;"


class _AlwaysFails:
    """Forced sandbox failure that still returns parseable stdout."""
    def __call__(self, code, test_input="", language="python", timeout=15):
        return False, ('{"runtime_clean":true,"supported":true,"early":"a",'
                       '"trace":"a|b","ended":false,"textSets":0}'), "forced failure"


def _sandbox_languages():
    url = adapters.SANDBOX_URL.rstrip("/") + "/languages"
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r).get("languages", {})


@pytest.fixture(scope="module")
def live_sandbox():
    try:
        langs = _sandbox_languages()
    except Exception as exc:                       # noqa: BLE001
        pytest.skip(f"live sandbox unreachable at {adapters.SANDBOX_URL}: {exc}")
    if "javascript" not in langs:
        pytest.skip(f"sandbox reports no javascript support: {sorted(langs)}")
    print(f"\n[live] sandbox={adapters.SANDBOX_URL} node={langs.get('javascript')}")
    return adapters.SandboxAdapter()


def _probe(sandbox, code):
    t0 = time.time()
    ev = P.run_browser_probe(code, sandbox=sandbox)
    return ev, int((time.time() - t0) * 1000)


def test_six_fixtures_against_the_live_sandbox(live_sandbox):
    report = {}
    for name, code in (("responsive", RESPONSIVE), ("ignores_input", IGNORES_INPUT),
                       ("inert", INERT), ("throws", THROWS), ("node_module", NODE_MODULE)):
        ev, ms = _probe(live_sandbox, code)
        if ev is None:
            report[name] = {"status": "inconclusive", "ms": ms}
            continue
        strength, missing, score = adapters.grade_interactive(ev)
        report[name] = {"status": "graded", "strength": strength, "score": score,
                        "missing_required": missing, "ms": ms,
                        "behavior": {k: v for k, v in ev.items() if k != "error"}}
    ev, ms = _probe(_AlwaysFails(), RESPONSIVE)
    report["sandbox_failure"] = {"status": "inconclusive" if ev is None else "LEAKED",
                                 "ms": ms}
    print("\n[live] " + json.dumps(report, indent=1, sort_keys=True))

    # The minimum distinctions the evidence policy rests on.
    assert report["responsive"]["behavior"]["temporal_progress"] is True
    assert report["responsive"]["behavior"]["input_causality"] is True

    assert report["ignores_input"]["behavior"]["temporal_progress"] is True
    assert report["ignores_input"]["behavior"]["input_causality"] is False, \
        "a timer moving pixels must never read as input causality"

    assert report["inert"]["score"] == 0.0
    assert report["inert"]["missing_required"]

    assert report["throws"]["behavior"]["runtime_clean"] is False
    assert report["throws"]["strength"] == adapters.SYNTAX, \
        "a runtime exception must never yield behavioural evidence"

    assert report["node_module"]["status"] == "inconclusive", \
        "an uninstrumentable Node module must never reach the probe"
    assert report["node_module"]["ms"] < 50, "it should not even be dispatched"

    assert report["sandbox_failure"]["status"] == "inconclusive", \
        "sandbox failure must be inconclusive even with parseable stdout"

    # Nothing here may close the pipeline: none is behaviourally complete.
    for name in ("responsive", "ignores_input", "inert", "throws"):
        r = report[name]
        if r["status"] != "graded":
            continue
        res = adapters.result_from_adapter(adapters.ADAPTER_BROWSER_CANVAS_JS, True, r["behavior"])
        assert not adapters.may_return_early_result(res), name
