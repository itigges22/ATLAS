"""JavaScript extraction (issue #39, Phase 6) with golden parity against
chiasmus's own extractor, captured via npx tsx on the same fixture."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import graph  # noqa: E402

pytestmark = pytest.mark.skipif(not graph.js_available(),
                                reason="tree-sitter JavaScript grammar not installed")

JS_FIXTURE = """import { clean } from "./helpers.js";
import defaultExport from "./def.js";

export function main() {
  const p = new Pipeline();
  return p.run(read());
}

class Pipeline {
  run(data) {
    return this.load(data);
  }
  load(data) {
    return clean(data);
  }
}

const read = () => {
  return [];
};

function helper() {
  return transform();
}
"""

GOLDEN = {
    "defines": [["Pipeline", "class"], ["helper", "function"], ["load", "method"],
                ["main", "function"], ["read", "function"], ["run", "method"]],
    "calls": [["helper", "transform"], ["load", "clean"], ["main", "read"],
              ["main", "run"], ["run", "load"]],
    "imports": [["clean", "./helpers.js"], ["defaultExport", "./def.js"]],
}


class TestJsGoldenParity:
    def test_matches_chiasmus(self):
        g = graph.extract_file("pipe.js", JS_FIXTURE)
        got = {
            "defines": sorted([[d.name, d.kind] for d in g.defines]),
            "calls": sorted([[c.caller, c.callee] for c in g.calls]),
            "imports": sorted([[i.name, i.source] for i in g.imports]),
        }
        for key, expected in GOLDEN.items():
            assert got[key] == sorted(expected), f"{key} diverged from chiasmus"

    def test_new_expression_not_a_call(self):
        # `new Pipeline()` must not appear as a call edge.
        g = graph.extract_file("pipe.js", JS_FIXTURE)
        assert all(c.callee != "Pipeline" for c in g.calls)

    def test_method_contains(self):
        g = graph.extract_file("pipe.js", JS_FIXTURE)
        pairs = {(c.parent, c.child) for c in g.contains}
        assert ("Pipeline", "run") in pairs and ("Pipeline", "load") in pairs


class TestMixedProject:
    def test_python_and_js_in_one_graph(self):
        files = {
            "a.py": "def py_fn():\n    return 1\n",
            "b.js": "function js_fn() {\n  return helper();\n}\n",
        }
        g = graph.build_graph(files)
        names = {d.name for d in g.defines}
        assert {"py_fn", "js_fn"} <= names
        # analyses work across the merged multi-language graph
        assert graph.callees(g, "js_fn") == ["helper"]

    def test_is_supported(self):
        assert graph.is_js("x.js") and graph.is_js("x.mjs")
        assert graph.is_supported("x.py") and graph.is_supported("x.jsx")
        assert not graph.is_supported("x.md")
