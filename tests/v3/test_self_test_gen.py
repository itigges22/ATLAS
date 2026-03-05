# tests/v3/test_self_test_gen.py
"""Tests for V3 Self-Test Generation."""

import json
from pathlib import Path
from typing import Optional, Tuple

import pytest

from benchmark.v3.self_test_gen import (
    SelfTestGenConfig,
    SelfTestGen,
    GeneratedTestCase,
    SelfTestResult,
    _strip_think_tags,
    _strip_markdown_fences,
    _parse_structured_blocks,
    _parse_numbered_blocks,
)


# ---------------------------------------------------------------------------
# Mock callables
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns predefined test case responses."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list = []

    def __call__(self, prompt: str, temperature: float,
                 max_tokens: int, seed: Optional[int]) -> Tuple[str, int, float]:
        self.calls.append({
            "prompt": prompt, "temperature": temperature,
            "max_tokens": max_tokens, "seed": seed,
        })
        return self.response, 100, 50.0


SAMPLE_LLM_RESPONSE = """\
TEST CASE 1:
DESCRIPTION: Simple case with small input
INPUT: 3
OUTPUT: 6

TEST CASE 2:
DESCRIPTION: Edge case with zero
INPUT: 0
OUTPUT: 0

TEST CASE 3:
DESCRIPTION: Large input boundary
INPUT: 1000000
OUTPUT: 2000000
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_telemetry(tmp_path):
    d = tmp_path / "telemetry"
    d.mkdir()
    return d


@pytest.fixture
def gen_enabled(tmp_telemetry):
    return SelfTestGen(
        SelfTestGenConfig(enabled=True, num_test_cases=3),
        telemetry_dir=tmp_telemetry,
    )


@pytest.fixture
def gen_disabled(tmp_telemetry):
    return SelfTestGen(
        SelfTestGenConfig(enabled=False),
        telemetry_dir=tmp_telemetry,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelfTestGenConfig:
    def test_defaults(self):
        cfg = SelfTestGenConfig()
        assert cfg.enabled is False
        assert cfg.num_test_cases == 5
        assert cfg.majority_threshold == 0.6

    def test_custom(self):
        cfg = SelfTestGenConfig(enabled=True, num_test_cases=3)
        assert cfg.enabled is True
        assert cfg.num_test_cases == 3


class TestGeneratedTestCase:
    def test_to_dict(self):
        tc = GeneratedTestCase(
            input_str="5", expected_output="10",
            description="simple doubling",
        )
        d = tc.to_dict()
        assert d["input_str"] == "5"
        assert d["expected_output"] == "10"
        assert d["description"] == "simple doubling"


class TestParsing:
    def test_parse_structured_response(self, gen_enabled):
        cases = gen_enabled.parse_test_cases(SAMPLE_LLM_RESPONSE)
        assert len(cases) == 3
        assert cases[0].input_str == "3"
        assert cases[0].expected_output == "6"
        assert cases[1].input_str == "0"
        assert cases[2].input_str == "1000000"

    def test_parse_empty_response(self, gen_enabled):
        cases = gen_enabled.parse_test_cases("")
        assert len(cases) == 0

    def test_parse_malformed_response(self, gen_enabled):
        cases = gen_enabled.parse_test_cases("no test cases here")
        assert len(cases) == 0

    def test_parse_caps_limit(self, gen_enabled):
        # gen_enabled has num_test_cases=3, should cap at 3
        response = SAMPLE_LLM_RESPONSE + """
TEST CASE 4:
DESCRIPTION: Extra
INPUT: 99
OUTPUT: 198

TEST CASE 5:
DESCRIPTION: Extra2
INPUT: 1
OUTPUT: 2
"""
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) <= 3


class TestThinkTagStripping:
    """Test handling of <think> tags, including unclosed ones."""

    def test_closed_think_tags_stripped(self, gen_enabled):
        response = "<think>Let me think about this...</think>\nTEST CASE 1:\nINPUT: 5\nOUTPUT: 10"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].input_str == "5"

    def test_unclosed_think_tag_stripped(self, gen_enabled):
        """Model hit token limit mid-think, no closing </think>."""
        response = "<think>I need to reason about this problem. The key insight is that we"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 0

    def test_unclosed_think_with_content_after(self, gen_enabled):
        """Content before unclosed <think> should be preserved."""
        response = (
            "TEST CASE 1:\nINPUT: 5\nOUTPUT: 10\n\n"
            "<think>Now let me think about the next one but I ran out of tok"
        )
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].input_str == "5"

    def test_closed_think_between_cases(self, gen_enabled):
        response = (
            "<think>Reasoning about the problem...</think>\n"
            "TEST CASE 1:\nINPUT: 3\nOUTPUT: 6\n\n"
            "TEST CASE 2:\nINPUT: 0\nOUTPUT: 0\n"
        )
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 2

    def test_strip_think_tags_direct(self):
        assert _strip_think_tags("<think>stuff</think>hello") == "hello"
        assert _strip_think_tags("<think>stuff without close") == ""
        assert _strip_think_tags("hello<think>mid</think>world") == "helloworld"
        assert _strip_think_tags("before<think>unclosed") == "before"


class TestAlternativeHeaderFormats:
    """Test case header format variations the model might use."""

    def test_title_case_test_case(self, gen_enabled):
        response = "Test Case 1:\nInput: 5\nOutput: 10\n\nTest Case 2:\nInput: 0\nOutput: 0\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 2
        assert cases[0].input_str == "5"

    def test_short_format_test_n(self, gen_enabled):
        response = "Test 1:\nInput: 5\nOutput: 10\n\nTest 2:\nInput: 0\nOutput: 0\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 2

    def test_all_lowercase(self, gen_enabled):
        response = "test case 1:\ninput: 5\noutput: 10\n\ntest case 2:\ninput: 0\noutput: 0\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 2

    def test_hash_numbered(self, gen_enabled):
        response = "Test Case #1:\nInput: 42\nOutput: 84\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].input_str == "42"

    def test_dash_separator(self, gen_enabled):
        response = "Test Case 1 -\nInput - 5\nOutput - 10\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1


class TestAlternativeFieldLabels:
    """Test INPUT/OUTPUT label variations."""

    def test_expected_output(self, gen_enabled):
        response = "TEST CASE 1:\nINPUT: 5\nEXPECTED OUTPUT: 10\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].expected_output == "10"

    def test_expected_only(self, gen_enabled):
        response = "TEST CASE 1:\nINPUT: 5\nExpected: 10\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].expected_output == "10"

    def test_stdin_stdout(self, gen_enabled):
        response = "TEST CASE 1:\nSTDIN: 5\nSTDOUT: 10\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].input_str == "5"
        assert cases[0].expected_output == "10"

    def test_mixed_case_labels(self, gen_enabled):
        response = "Test Case 1:\nInput: hello\nExpected Output: world\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].input_str == "hello"
        assert cases[0].expected_output == "world"

    def test_explanation_as_description(self, gen_enabled):
        response = "TEST CASE 1:\nExplanation: tests basic case\nINPUT: 5\nOUTPUT: 10\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].description == "tests basic case"


class TestMarkdownCodeBlocks:
    """Test handling of markdown code fences in values."""

    def test_fenced_code_block(self, gen_enabled):
        response = (
            "TEST CASE 1:\nINPUT:\n```\n5\n```\n"
            "OUTPUT:\n```\n10\n```\n"
        )
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].input_str == "5"
        assert cases[0].expected_output == "10"

    def test_fenced_with_language(self, gen_enabled):
        response = (
            "TEST CASE 1:\nINPUT:\n```python\n5\n```\n"
            "OUTPUT:\n```python\n10\n```\n"
        )
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].input_str == "5"
        assert cases[0].expected_output == "10"

    def test_inline_backticks(self, gen_enabled):
        response = "TEST CASE 1:\nINPUT: `5`\nOUTPUT: `10`\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].input_str == "5"
        assert cases[0].expected_output == "10"

    def test_strip_markdown_fences_direct(self):
        assert _strip_markdown_fences("```\nhello\n```") == "hello"
        assert _strip_markdown_fences("```python\ncode\n```") == "code"
        assert _strip_markdown_fences("`value`") == "value"
        assert _strip_markdown_fences("plain") == "plain"


class TestNumberedBlockFallback:
    """Test fallback parsing for numbered blocks without TEST CASE headers."""

    def test_dot_numbered(self, gen_enabled):
        response = "1. Input: 5\nOutput: 10\n\n2. Input: 0\nOutput: 0\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 2
        assert cases[0].input_str == "5"
        assert cases[1].input_str == "0"

    def test_paren_numbered(self, gen_enabled):
        response = "1) Input: 5\nOutput: 10\n\n2) Input: 3\nOutput: 6\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 2

    def test_colon_numbered(self, gen_enabled):
        response = "1: Input: 5\nOutput: 10\n\n2: Input: 3\nOutput: 6\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 2

    def test_fallback_not_used_when_structured_works(self, gen_enabled):
        """Structured parser should take priority over fallback."""
        cases = gen_enabled.parse_test_cases(SAMPLE_LLM_RESPONSE)
        assert len(cases) == 3
        assert cases[0].description == "Simple case with small input"

    def test_numbered_without_input_output_labels(self, gen_enabled):
        """Numbered blocks without INPUT/OUTPUT labels should not parse."""
        response = "1. Just some text\n2. More text\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 0


class TestMultilineValues:
    """Test cases where input/output span multiple lines."""

    def test_multiline_input(self, gen_enabled):
        response = "TEST CASE 1:\nINPUT: 3\n1 2 3\nOUTPUT: 6\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].input_str == "3\n1 2 3"

    def test_multiline_output(self, gen_enabled):
        response = "TEST CASE 1:\nINPUT: 3\nOUTPUT: 1\n2\n3\n"
        cases = gen_enabled.parse_test_cases(response)
        assert len(cases) == 1
        assert cases[0].expected_output == "1\n2\n3"


class TestGenerate:
    def test_disabled_returns_empty(self, gen_disabled):
        llm = MockLLM(SAMPLE_LLM_RESPONSE)
        result = gen_disabled.generate("double the number", llm_call=llm)
        assert len(result.test_cases) == 0
        assert len(llm.calls) == 0

    def test_enabled_generates_tests(self, gen_enabled):
        llm = MockLLM(SAMPLE_LLM_RESPONSE)
        result = gen_enabled.generate(
            "Given n, return 2*n", llm_call=llm, task_id="test_1",
        )
        assert len(result.test_cases) == 3
        assert len(llm.calls) == 1
        assert result.generation_tokens == 100

    def test_no_llm_returns_empty(self, gen_enabled):
        result = gen_enabled.generate("problem", llm_call=None)
        assert len(result.test_cases) == 0
        assert result.reason == "missing_callable"

    def test_telemetry_logged(self, gen_enabled, tmp_telemetry):
        llm = MockLLM(SAMPLE_LLM_RESPONSE)
        gen_enabled.generate("problem", llm_call=llm, task_id="t1")
        events_file = tmp_telemetry / "self_test_gen_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["task_id"] == "t1"
        assert event["num_cases_generated"] == 3
