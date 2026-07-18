"""Regression tests for v3-service candidate selection + self-test builder.

Covers:
  * _candidate_by_index — S*/lens winners are reported by the candidate's
    ORIGINAL index; the `passing` list is sorted and filtered, so positional
    indexing selected the wrong candidate (or raised IndexError).
  * _make_self_test — the stdin-form test builder must not corrupt multiline
    string literals inside the candidate (the old per-line indent under
    `try:` changed content inside triple-quoted strings).
"""

import contextlib
import io
import sys
from pathlib import Path

# v3-service/main.py imports from project root, so we add it the same way
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "v3-service"))

import main as v3main  # noqa: E402


# --- _candidate_by_index ------------------------------------------------------

def _passing_fixture():
    # Sorted by energy and filtered (candidate 1 vetoed) — original indexes
    # no longer line up with list positions.
    return [
        {"index": 2, "code": "code-two", "energy": 0.1},
        {"index": 0, "code": "code-zero", "energy": 0.4},
    ]


def test_candidate_by_index_matches_original_index_not_position():
    passing = _passing_fixture()

    winner = v3main._candidate_by_index(passing, 0)

    assert winner is not None
    assert winner["code"] == "code-zero"


def test_candidate_by_index_returns_none_for_filtered_out_index():
    passing = _passing_fixture()

    # Candidate 1 was vetoed out of `passing`; positional indexing would have
    # silently returned candidate at slot 1 ("code-zero") instead.
    assert v3main._candidate_by_index(passing, 1) is None


def test_candidate_by_index_out_of_range_returns_none():
    passing = _passing_fixture()

    # Old code: passing[2] → IndexError (swallowed upstream, losing the win).
    assert v3main._candidate_by_index(passing, 5) is None


# --- _make_self_test ----------------------------------------------------------

class _TC:
    def __init__(self, input_str, expected_output):
        self.input_str = input_str
        self.expected_output = expected_output


def _run_generated(test_code):
    """Exec a generated self-test the way the sandbox would (module import):
    fresh namespace, captured stdout. Restores sys.stdin afterwards."""
    buf = io.StringIO()
    old_stdin = sys.stdin
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(test_code, "solution.py", "exec"), {"__name__": "solution"})
    finally:
        sys.stdin = old_stdin
    return buf.getvalue()


def test_make_self_test_stdin_form_preserves_multiline_string_literals():
    # Indent-sensitive triple-quoted literal: the old builder prefixed every
    # line with four spaces, changing TEMPLATE's content and the output.
    code = (
        'TEMPLATE = """\n'
        "line1\n"
        "line2\n"
        '"""\n'
        "import sys\n"
        "data = sys.stdin.read().strip()\n"
        "print(len(TEMPLATE.splitlines()) + int(data))\n"
    )
    tc = _TC("3", "6")  # TEMPLATE has 3 splitlines entries; 3 + 3 = 6

    test_code = _make(code, tc)
    out = _run_generated(test_code)

    assert "SELF_TEST_PASS" in out


def test_make_self_test_stdin_form_fails_on_wrong_output():
    code = "import sys\nprint(sys.stdin.read().strip())\n"
    tc = _TC("hello", "goodbye")

    test_code = _make(code, tc)
    try:
        out = _run_generated(test_code)
    except AssertionError:
        return  # expected: the generated assertion fired
    assert "SELF_TEST_PASS" not in out


def test_make_self_test_function_form_unchanged():
    code = "def double(x):\n    return x * 2\n"
    tc = _TC("21", "42")

    test_code = _make(code, tc)
    out = _run_generated(test_code)

    assert "SELF_TEST_PASS" in out


def _make(code, tc):
    return v3main._make_self_test(code, tc)
