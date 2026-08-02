"""structural_edit node-size precondition.

A replacement many times the size of the node it replaces is not an edit of
that node — it is the whole file wearing a selector. Size alone is not the
test, because writing a real body over a `pass` stub is also many times the
node; what marks the blob is that its content already exists elsewhere in the
file.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import main  # noqa: E402

pytestmark = pytest.mark.skipif(
    not getattr(main, "_STRUCTURAL_EDIT_AVAILABLE", False),
    reason="tree-sitter not installed in this environment",
)

structural_edit = main.structural_edit

# --- node-size precondition ---------------------------------------------------
#
# 2026-08-02 dogfooding: `function:index` (3 lines, `return
# render_template_string(HTML_TEMPLATE)`) was replaced with a 258-line
# `HTML_TEMPLATE = """..."""` assignment. That is valid Python, so the
# post-splice compile() passed and the splice landed — deleting the app's only
# @app.route. The file still parsed and the agent reported success.
#
# The check for this existed, but inside the `except SyntaxError` handler, so
# it only ever fired when the blob ALSO failed to compile.

_FLASK = (
    'from flask import Flask, render_template_string\n'
    'app = Flask(__name__)\n'
    'HTML_TEMPLATE = """\n'
    + "".join(f'<div class="row-{i}">some markup content here</div>\n' for i in range(60))
    + '"""\n\n\n'
    "@app.route('/')\n"
    'def index():\n'
    '    return render_template_string(HTML_TEMPLATE)\n'
)


def test_moving_the_template_into_the_route_is_refused():
    template = _FLASK[_FLASK.index('HTML_TEMPLATE'):_FLASK.index("@app.route")]
    res = structural_edit(path="app.py", source_text=_FLASK,
                          selector="function:index", content=template)
    assert res["success"] is False
    err = res["error"]
    assert "is only" in err and "your replacement is" in err
    assert "HTML_TEMPLATE" in err          # names where the bulk actually lives
    assert "replace_lines" in err          # and a tool that can reach into it
    assert "was NOT modified" in err


def test_it_fires_even_though_the_replacement_is_valid_python():
    """The whole point: compile() succeeds on the blob, so a check that runs
    only on SyntaxError never sees it."""
    template = _FLASK[_FLASK.index('HTML_TEMPLATE'):_FLASK.index("@app.route")]
    compile(template, "app.py", "exec")  # valid on its own
    assert structural_edit(path="app.py", source_text=_FLASK,
                           selector="function:index", content=template)["success"] is False


def test_a_freshly_written_template_inlined_into_the_route_is_also_refused():
    """Not every blob is a copy. A model that writes the markup out fresh
    inside an inline render_template_string literal has made the same mistake,
    and the reuse signal cannot see it — the file having a template-sized
    string constant is what says no selector reaches the real target.
    """
    fresh = ("@app.route('/')\ndef index():\n"
             '    return render_template_string("""\n'
             + "".join(f"  <div>brand new line {i}</div>\n" for i in range(120))
             + '""")\n')
    res = structural_edit(path="app.py", source_text=_FLASK,
                          selector="function:index", content=fresh)
    assert res["success"] is False, "a fresh 120-line inline template must be refused"
    assert "does not live here" in res["error"]


def test_a_fresh_implementation_over_a_stub_is_allowed():
    """Size alone must not refuse: writing a real body over `pass` is many
    times the node and is ordinary work. A file with no template-sized string
    constant and no reused content has neither signal."""
    stub = "def helper():\n    pass\n\n\ndef other():\n    return 1\n"
    for n in (25, 60, 150):
        body = "def helper():\n" + "".join(
            f"    value_number_{i} = {i} * 3\n" for i in range(n)) + "    return 0\n"
        res = structural_edit(path="a.py", source_text=stub,
                              selector="function:helper", content=body)
        assert res["success"] is True, (n, res.get("error"))


def test_a_node_sized_replacement_is_untouched():
    res = structural_edit(path="app.py", source_text=_FLASK, selector="function:index",
                          content="@app.route('/')\ndef index():\n    return 'hi'\n")
    assert res["success"] is True, res.get("error")
