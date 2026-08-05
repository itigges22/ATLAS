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
