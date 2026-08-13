"""The probe must work through the PRODUCTION adapter, not just a stub.

SandboxAdapter.__call__ took no `language` and hardcoded "python" in the
request body. The probe's call therefore raised TypeError, which
run_browser_probe converted to "inconclusive" — so no real browser probe
ever produced evidence, while an isolation test using a permissive stub
passed. That production/stub gap is why these assert on the adapter's
actual serialized request.
"""

import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import adapters  # noqa: E402
import adapters as A  # noqa: E402
import pipeline as P  # noqa: E402
import pipeline as P  # noqa: E402


class _Resp:
    def __init__(self, payload):
        self._p = json.dumps(payload).encode()
    def read(self): return self._p
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _capture(payload=None):
    sent = {}
    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data.decode())
        return _Resp(payload or {"success": True, "stdout": "{}", "stderr": ""})
    return sent, fake_urlopen


def test_python_execution_still_defaults_to_python():
    sent, fake = _capture()
    with mock.patch("urllib.request.urlopen", fake):
        adapters.SandboxAdapter()("print(1)")
    assert sent["body"]["language"] == "python"


def test_javascript_reaches_execute_with_the_right_language():
    sent, fake = _capture()
    with mock.patch("urllib.request.urlopen", fake):
        adapters.SandboxAdapter()("console.log(1)", language="javascript", timeout=30)
    assert sent["body"]["language"] == "javascript"
    assert sent["body"]["timeout"] == 30


CANVAS = ("const c=document.getElementById('gameCanvas');const x=c.getContext('2d');"
          "document.addEventListener('keydown',e=>{});"
          "function loop(){x.fillRect(0,0,10,10);setTimeout(loop,50);} loop();")


def test_probe_drives_the_real_adapter_without_typeerror():
    """The exact defect: the production signature rejected the probe's call."""
    payload = {"success": True, "stdout":
               '{"runtime_clean":true,"supported":true,"ended":false,'
               '"textSets":0,"early":"a|b","trace":"a|b|c|d"}', "stderr": ""}
    sent, fake = _capture(payload)
    with mock.patch("urllib.request.urlopen", fake):
        ev = P.run_browser_probe(CANVAS, sandbox=adapters.SandboxAdapter())
    assert ev is not None, "the production adapter must accept the probe's call"
    assert sent["body"]["language"] == "javascript"


def test_sandbox_failure_with_parseable_stdout_is_still_inconclusive():
    payload = {"success": False, "stdout":
               '{"runtime_clean":true,"supported":true,"ended":false,'
               '"textSets":0,"early":"a","trace":"a|b"}', "stderr": "timeout"}
    _sent, fake = _capture(payload)
    with mock.patch("urllib.request.urlopen", fake):
        assert P.run_browser_probe(CANVAS, sandbox=adapters.SandboxAdapter()) is None


# ---- modes ---------------------------------------------------------------

def test_the_three_modes_have_exactly_the_intended_behaviour():
    assert (P._probing_enabled(P.MODE_OFF), P._selection_enabled(P.MODE_OFF)) == (False, False)
    assert (P._probing_enabled(P.MODE_SHADOW), P._selection_enabled(P.MODE_SHADOW)) == (True, False)
    assert (P._probing_enabled(P.MODE_ENFORCE), P._selection_enabled(P.MODE_ENFORCE)) == (True, True)


def test_enforcement_without_probing_is_unrepresentable():
    for mode in (P.MODE_OFF, P.MODE_SHADOW, P.MODE_ENFORCE, "nonsense", ""):
        m = mode if mode in (P.MODE_OFF, P.MODE_SHADOW, P.MODE_ENFORCE) else P.MODE_OFF
        assert not (P._selection_enabled(m) and not P._probing_enabled(m))


def test_unknown_mode_falls_back_to_off():
    assert P._selection_mode({"ATLAS_EVIDENCE_MODE": "wat"}) == P.MODE_OFF
    assert P._selection_mode({}) == P.MODE_OFF
    assert P._selection_mode({"ATLAS_EVIDENCE_MODE": "ENFORCE"}) == P.MODE_ENFORCE
