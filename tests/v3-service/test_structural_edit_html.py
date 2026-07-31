"""structural_edit HTML selector tests — regression for <script>/<style> matching.

tree-sitter-html parses <script> and <style> as dedicated script_element /
style_element nodes (raw JS/CSS bodies), NOT generic `element` nodes, so the
generic element query matched them 0 times. These confirm the dedicated-node
queries match (and that bare tags still work)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import main  # noqa: E402

pytestmark = pytest.mark.skipif(
    not getattr(main, "_STRUCTURAL_EDIT_AVAILABLE", False),
    reason="tree-sitter not installed in this environment",
)

HTML = (
    "<!DOCTYPE html>\n<html>\n<head>\n  <style>body { margin: 0; }</style>\n"
    "</head>\n<body>\n  <canvas id=\"gameCanvas\"></canvas>\n"
    "  <script src=\"/static/game.js\"></script>\n</body>\n</html>\n"
)


def test_script_selector_matches_script_element():
    res = main.structural_edit("templates/index.html", HTML, "<script>",
                         "<script>\n  const c = 1; // inline\n</script>")
    assert res.get("success"), res
    # the src-based script was replaced
    assert "/static/game.js" not in res["new_content"]
    assert "inline" in res["new_content"]


def test_style_selector_matches_style_element():
    res = main.structural_edit("templates/index.html", HTML, "<style>",
                         "<style>body { margin: 8px; }</style>")
    assert res.get("success"), res
    assert "margin: 8px" in res["new_content"]


def test_bare_element_selector_still_works():
    res = main.structural_edit("templates/index.html", HTML, "<canvas>",
                         "<canvas id=\"gameCanvas\" width=\"400\"></canvas>")
    assert res.get("success"), res
    assert "width=\"400\"" in res["new_content"]


def test_attribute_selector_rejected_with_guidance():
    q, _, err = main._ast_selector_to_query('<script src="x">', "html")
    assert q is None and err and "bare tag" in err


def test_html_tag_on_python_file_names_the_escape_hatch():
    """An HTML tag selector on a .py file is the Flask-template case.

    A live session editing a Flask app reached for `<script>` on the .py file
    whose script lives inside a template string, and got a selector list that
    did not address what it was trying to do. The Python grammar sees one
    string literal, so no selector can reach inside: the message has to name
    what does work instead.
    """
    q, _, err = main._ast_selector_to_query("<script>", "python")
    assert q is None and err
    assert "HTML-only" in err
    assert "function:NAME" in err and "edit_file" in err


def test_non_tag_unknown_python_selector_still_lists_selectors():
    q, _, err = main._ast_selector_to_query("def:index", "python")
    assert q is None and err
    assert "function:NAME, class:NAME" in err
