"""The behavioural probe must not execute model code in the service.

It previously shelled out to `node` directly from v3-service, running
model-generated JavaScript with the service's filesystem, environment,
network and process capabilities. The instrumentability regex is a routing
hint, not a containment boundary. The probe now runs inside the same
isolated sandbox that already executes untrusted candidate code, and returns
"inconclusive" rather than a verdict whenever it cannot.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import adapters as A  # noqa: E402
import pipeline as P  # noqa: E402
import pipeline as P  # noqa: E402

CANVAS = """
const c = document.getElementById('gameCanvas');
const ctx = c.getContext('2d');
document.addEventListener('keydown', e => {});
function loop(){ ctx.fillRect(0,0,10,10); setTimeout(loop, 50); } loop();
"""


def test_probe_never_spawns_a_process_from_the_service():
    src = Path(P.__file__).read_text()
    body = src[src.index("def run_browser_probe("):src.index("def _make_output_probe(")]
    # Strip the docstring: it NAMES the removed mechanism to explain the fix,
    # and a token check that cannot tell prose from code is not a check.
    import re as _re
    code_only = _re.sub(r'"""[\s\S]*?"""', "", body)
    for forbidden in ("subprocess.", "Popen(", "os.system(", "os.popen(", '"node"', "'node'"):
        assert forbidden not in code_only, f"probe must not reach {forbidden} directly"


def test_inline_harness_has_no_filesystem_or_argv_access():
    src = A.js_probe_source_inline()
    assert "process.argv" not in src
    assert "require('fs')" not in src
    assert "__ARTIFACT__" in src


def test_no_sandbox_means_inconclusive_not_verified():
    assert P.run_browser_probe(CANVAS, sandbox=None) is None


def test_sandbox_failure_is_inconclusive():
    def broken(_code, **_kw):
        raise RuntimeError("sandbox down")
    assert P.run_browser_probe(CANVAS, sandbox=broken) is None


def test_adapter_without_language_support_is_inconclusive():
    def legacy(_code):           # no language kwarg
        return True, "", ""
    assert P.run_browser_probe(CANVAS, sandbox=legacy) is None


def test_uninstrumentable_code_is_never_probed():
    calls = []

    def sb(code, **kw):
        calls.append(code)
        return True, "{}", ""
    assert P.run_browser_probe("const fs = require('fs');", sandbox=sb) is None
    assert not calls, "a Node script must not be sent to the probe at all"


def test_probe_runs_both_arms_through_the_sandbox():
    seen = []

    payload = ('{"runtime_clean":true,"supported":true,"ended":false,'
               '"textSets":0,"early":"a","trace":"a|b"}')

    def sb(code, language=None, timeout=None):
        seen.append(language)
        return True, payload, ""
    P.run_browser_probe(CANVAS, sandbox=sb)
    assert seen == ["javascript", "javascript"], seen
