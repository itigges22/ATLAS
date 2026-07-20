"""#147: the structural veto must catch an edit that calls a name the file
neither imports nor defines (render_template with only render_template_string
imported), using the candidate's OWN imports — i.e. with EMPTY project
symbols, the case the edit path hits when it sends no project_context."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import main  # noqa: E402

pytestmark = pytest.mark.skipif(
    not getattr(main, "_AST_EDIT_AVAILABLE", False),
    reason="tree-sitter not installed",
)


def test_unimported_call_is_unresolved_with_empty_project():
    # The exact #147 shape: import render_template_string, call render_template.
    code = (
        "from flask import Flask, render_template_string\n"
        "app = Flask(__name__)\n"
        "@app.route('/')\n"
        "def index():\n"
        "    return render_template('index.html')\n"
    )
    struct = main.structural_score(set(), code)  # empty project symbols
    assert struct["ok"]
    assert "render_template" in struct["unresolved_calls"], struct
    assert struct["n_unresolved"] >= 1


def test_imported_name_resolves():
    # Calling the name that IS imported must NOT be flagged.
    code = (
        "from flask import render_template_string\n"
        "def index():\n"
        "    return render_template_string('<b>x</b>')\n"
    )
    struct = main.structural_score(set(), code)
    assert struct["ok"]
    assert "render_template_string" not in struct["unresolved_calls"], struct


def test_name_imported_elsewhere_in_file_passes():
    # A helper defined at top level in the same file resolves via local defs.
    code = (
        "def helper():\n    return 1\n"
        "def index():\n    return helper()\n"
    )
    struct = main.structural_score(set(), code)
    assert struct["ok"]
    assert "helper" not in struct["unresolved_calls"], struct


def test_project_symbol_credits_cross_file_call():
    # A name supplied by project symbols is credited (lenient cross-file).
    code = "def index():\n    return shared_util()\n"
    struct = main.structural_score({"shared_util"}, code)
    assert struct["ok"]
    assert "shared_util" not in struct["unresolved_calls"], struct


def test_local_variable_call_not_flagged():
    # #147 review #4: fn is a local variable, not a top-level def — must NOT flag.
    code = (
        "def run():\n"
        "    fn = build_pipeline()\n"
        "    return fn()\n"
        "def build_pipeline():\n    return lambda: 1\n"
    )
    struct = main.structural_score(set(), code)
    assert struct["ok"]
    assert "fn" not in struct["unresolved_calls"], struct


def test_function_parameter_call_not_flagged():
    code = "def apply(handler, evt):\n    return handler(evt)\n"
    struct = main.structural_score(set(), code)
    assert "handler" not in struct["unresolved_calls"], struct


def test_loop_and_comprehension_target_not_flagged():
    code = (
        "def dispatch(handlers, evt):\n"
        "    for h in handlers:\n"
        "        h(evt)\n"
        "    return [g() for g in handlers]\n"
    )
    struct = main.structural_score(set(), code)
    assert "h" not in struct["unresolved_calls"], struct
    assert "g" not in struct["unresolved_calls"], struct


def test_with_as_target_not_flagged():
    code = "def f():\n    with open('x') as fh:\n        return fh()\n"
    struct = main.structural_score(set(), code)
    assert "fh" not in struct["unresolved_calls"], struct


def test_genuine_nameerror_still_flagged_amid_locals():
    # render_template is neither imported, defined, bound, nor builtin —
    # must still flag even though the file has locals.
    code = (
        "from flask import render_template_string\n"
        "def index():\n"
        "    tmpl = load()\n"
        "    return render_template(tmpl)\n"
        "def load():\n    return 'x'\n"
    )
    struct = main.structural_score(set(), code)
    assert "render_template" in struct["unresolved_calls"], struct
    assert "tmpl" not in struct["unresolved_calls"], struct
