"""A self-test has to call the function the test case is about.

The generator took the first `def` in the file, which is the entry point
only when the solution defines no helpers above it. `def parse(...)`
followed by `def solve(...)` meant every case called parse with the case
input and compared it to the final answer.

Measured across a 28-session run: 0 of 44 candidates passed the sandbox,
and the self-test results were 0/5, 0/4, 0/3 — never partial. Imperfect
code fails some cases; only a harness fault fails all of them uniformly.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import pipeline  # noqa: E402


def test_a_helper_defined_first_is_not_the_entry_point():
    """The exact shape that produced 0/N: parse above solve."""
    code = ("def parse(lines):\n"
            "    return [int(x) for x in lines]\n\n"
            "def solve(lines):\n"
            "    return sum(parse(lines))\n")
    assert pipeline._entry_function(code) == "solve"


def test_the_only_function_is_the_entry_point():
    assert pipeline._entry_function("def solve(x):\n    return x\n") == "solve"


@pytest.mark.parametrize("name", ["solve", "main", "run"])
def test_a_conventional_name_wins_among_roots(name):
    code = (f"def alpha(x):\n    return x\n\n"
            f"def {name}(x):\n    return alpha(x)\n\n"
            f"def zulu(x):\n    return x\n")
    assert pipeline._entry_function(code) == name


def test_without_a_convention_the_last_uncalled_function_wins():
    """Helpers are conventionally defined above their caller."""
    code = ("def prep(x):\n    return x\n\n"
            "def compute(x):\n    return prep(x) + 1\n")
    assert pipeline._entry_function(code) == "compute"


def test_mutual_calls_still_yield_a_name():
    """Everything called by something else leaves no root; picking nothing
    would drop the test case entirely, so the last def stands in."""
    code = ("def a(x):\n    return b(x)\n\n"
            "def b(x):\n    return a(x)\n")
    assert pipeline._entry_function(code) == "b"


def test_unparseable_code_falls_back_to_the_old_behaviour():
    """A candidate that does not parse still gets a name rather than an
    exception — the sandbox is what should report the syntax error."""
    assert pipeline._entry_function("def parse(l:\n    broken") == "parse"


def test_no_functions_returns_none():
    assert pipeline._entry_function("print(42)\n") is None


def test_the_generated_test_calls_the_entry_point():
    """End to end: the assertion body names solve, not parse."""
    code = ("def parse(lines):\n"
            "    return [int(x) for x in lines]\n\n"
            "def solve(lines):\n"
            "    return sum(parse(lines))\n")
    tc = type("TC", (), {"input_str": "[1, 2, 3]", "expected_output": "6"})()
    body = pipeline._make_self_test(code, tc)
    assert "solve(" in body
    assert "_r=parse(" not in body


# --- the two self-test shapes are not interchangeable ----------------------

SCRIPT = ("import sys\n\n"
          "def main():\n"
          "    d = [int(x) for x in sys.stdin]\n"
          "    print(len(d))\n\n"
          "main()\n")

FUNCTION = ("def parse(lines):\n"
            "    return [int(x) for x in lines]\n\n"
            "def solve(lines):\n"
            "    return sum(parse(lines))\n")


def _case():
    return type("TC", (), {"input_str": "199\n200\n208",
                           "expected_output": "1"})()


def test_a_stdin_script_is_driven_through_stdin():
    """Measured on aoc_sonar: candidates are `def main():` reading sys.stdin
    and printing. The function path called main(case) — a TypeError, with no
    return value to compare — so every case failed. The stdin/stdout path
    already existed and was never reached, because the choice keyed on a
    function merely existing."""
    body = pipeline._make_self_test(SCRIPT, _case())
    assert "_s.stdin" in body, "a stdin-reading script must be fed stdin"
    assert "main(" not in body.split("_src=")[0]


def test_a_pure_function_is_still_called_directly():
    body = pipeline._make_self_test(FUNCTION, _case())
    assert "solve(" in body
    assert "_s.stdin" not in body


def test_a_zero_argument_function_cannot_take_the_case():
    assert not pipeline._entry_takes_case_input(
        "def solve():\n    return 1\n", "solve")


def test_a_function_that_returns_nothing_has_no_answer_to_compare():
    assert not pipeline._entry_takes_case_input(
        "def solve(x):\n    print(x)\n", "solve")


def test_a_function_taking_the_case_and_returning_qualifies():
    assert pipeline._entry_takes_case_input(
        "def solve(x):\n    return x + 1\n", "solve")


def test_reading_stdin_disqualifies_whatever_the_signature_says():
    assert not pipeline._entry_takes_case_input(
        "import sys\ndef solve(x):\n    return sys.stdin.read()\n", "solve")
