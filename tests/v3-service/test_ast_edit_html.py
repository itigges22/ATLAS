"""ast_edit HTML selector tests — regression for <script>/<style> matching.

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
    not getattr(main, "_AST_EDIT_AVAILABLE", False),
    reason="tree-sitter not installed in this environment",
)

HTML = (
    "<!DOCTYPE html>\n<html>\n<head>\n  <style>body { margin: 0; }</style>\n"
    "</head>\n<body>\n  <canvas id=\"gameCanvas\"></canvas>\n"
    "  <script src=\"/static/game.js\"></script>\n</body>\n</html>\n"
)


def test_script_selector_matches_script_element():
    res = main.ast_edit("templates/index.html", HTML, "<script>",
                         "<script>\n  const c = 1; // inline\n</script>")
    assert res.get("success"), res
    # the src-based script was replaced
    assert "/static/game.js" not in res["new_content"]
    assert "inline" in res["new_content"]


def test_style_selector_matches_style_element():
    res = main.ast_edit("templates/index.html", HTML, "<style>",
                         "<style>body { margin: 8px; }</style>")
    assert res.get("success"), res
    assert "margin: 8px" in res["new_content"]


def test_bare_element_selector_still_works():
    res = main.ast_edit("templates/index.html", HTML, "<canvas>",
                         "<canvas id=\"gameCanvas\" width=\"400\"></canvas>")
    assert res.get("success"), res
    assert "width=\"400\"" in res["new_content"]


def test_attribute_selector_rejected_with_guidance():
    q, _, err = main._ast_selector_to_query('<script src="x">', "html")
    assert q is None and err and "bare tag" in err
