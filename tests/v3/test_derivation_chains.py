"""Tests for V3 Derivation Chains (Feature 3D)."""

import json
from pathlib import Path
from typing import Optional, Tuple

import pytest

from benchmark.v3.derivation_chains import (
    DerivationChainEvent,
    DerivationChains,
    DerivationChainsConfig,
    DerivationResult,
    SubProblem,
    VerifiedStep,
    extract_code,
    parse_decomposition,
)


# ---------------------------------------------------------------------------
# Mock callables
# ---------------------------------------------------------------------------

DECOMPOSITION_RESPONSE = (
    "SUB-PROBLEM 1:\n"
    "DESCRIPTION: Parse the input string into a list of integers\n"
    "INPUT: A string of space-separated integers\n"
    "OUTPUT: A list of integers\n"
    "TEST: assert parse('1 2 3') == [1, 2, 3]\n"
    "\n"
    "SUB-PROBLEM 2:\n"
    "DESCRIPTION: Sort the list using merge sort\n"
    "INPUT: A list of integers\n"
    "OUTPUT: A sorted list of integers\n"
    "TEST: assert sort([3,1,2]) == [1,2,3]\n"
)

CODE_RESPONSE = "```python\ndef solve(s):\n    return sorted(int(x) for x in s.split())\n```"

COMPOSITION_RESPONSE = (
    "```python\n"
    "def solve(s):\n"
    "    nums = [int(x) for x in s.split()]\n"
    "    return sorted(nums)\n"
    "```"
)


class MockLLM:
    """Mock LLM that returns decomposition, step code, or composition."""

    def __init__(self):
        self.calls: list = []
        self._call_count = 0

    def __call__(self, prompt: str, temperature: float,
                 max_tokens: int, seed: Optional[int]) -> Tuple[str, int, float]:
        self.calls.append({
            "prompt": prompt, "temperature": temperature,
            "max_tokens": max_tokens, "seed": seed,
        })
        self._call_count += 1

        if "Compose these verified" in prompt:
            return COMPOSITION_RESPONSE, 80, 30.0
        if "Decompose this into" in prompt:
            return DECOMPOSITION_RESPONSE, 120, 50.0
        return CODE_RESPONSE, 60, 25.0


class MockLLMEmpty:
    """Mock LLM that returns empty decomposition (no SUB-PROBLEM blocks)."""

    def __init__(self):
        self.calls: list = []

    def __call__(self, prompt: str, temperature: float,
                 max_tokens: int, seed: Optional[int]) -> Tuple[str, int, float]:
        self.calls.append(prompt)
        return "I could not decompose this problem.", 50, 20.0


class MockSandbox:
    """Mock sandbox that always passes verification."""

    def __init__(self):
        self.calls: list = []

    def __call__(self, code: str, test_case: str) -> Tuple[bool, str, str]:
        self.calls.append({"code": code, "test_case": test_case})
        return True, "OK", ""


class MockSandboxAlwaysFail:
    """Mock sandbox that always fails verification."""

    def __init__(self):
        self.calls: list = []

    def __call__(self, code: str, test_case: str) -> Tuple[bool, str, str]:
        self.calls.append({"code": code, "test_case": test_case})
        return False, "", "AssertionError: wrong answer"


class MockSandboxFailThenPass:
    """Mock sandbox that fails N times then passes."""

    def __init__(self, fail_count: int = 1):
        self.calls: list = []
        self._fail_count = fail_count
        self._call_count = 0

    def __call__(self, code: str, test_case: str) -> Tuple[bool, str, str]:
        self.calls.append({"code": code, "test_case": test_case})
        self._call_count += 1
        if self._call_count <= self._fail_count:
            return False, "", "AssertionError"
        return True, "OK", ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_telemetry(tmp_path):
    """Provide a temporary telemetry directory."""
    d = tmp_path / "telemetry"
    d.mkdir()
    return d


@pytest.fixture
def dc_enabled(tmp_telemetry):
    """DerivationChains instance with enabled=True."""
    cfg = DerivationChainsConfig(enabled=True, max_sub_problems=5)
    return DerivationChains(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def dc_disabled(tmp_telemetry):
    """DerivationChains instance with enabled=False."""
    cfg = DerivationChainsConfig(enabled=False)
    return DerivationChains(cfg, telemetry_dir=tmp_telemetry)


# ---------------------------------------------------------------------------
# Test: Disabled noop behavior
# ---------------------------------------------------------------------------

class TestDisabledNoop:
    """When enabled=False, DerivationChains should be a complete noop."""

    def test_returns_not_triggered(self, dc_disabled):
        result = dc_disabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(),
        )
        assert result.triggered is False
        assert result.solved is False
        assert result.reason == "disabled"

    def test_preserves_task_id(self, dc_disabled):
        result = dc_disabled.solve(problem="test", task_id="my_task")
        assert result.task_id == "my_task"

    def test_no_telemetry_when_disabled(self, dc_disabled, tmp_telemetry):
        dc_disabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(), task_id="t1",
        )
        events_file = tmp_telemetry / "derivation_chains_events.jsonl"
        assert not events_file.exists()

    def test_does_not_call_llm_when_disabled(self, dc_disabled):
        mock_llm = MockLLM()
        dc_disabled.solve(problem="test", llm_call=mock_llm, sandbox_run=MockSandbox())
        assert len(mock_llm.calls) == 0

    def test_does_not_call_sandbox_when_disabled(self, dc_disabled):
        mock_sandbox = MockSandbox()
        dc_disabled.solve(problem="test", llm_call=MockLLM(), sandbox_run=mock_sandbox)
        assert len(mock_sandbox.calls) == 0

    def test_empty_sub_problems_and_steps(self, dc_disabled):
        result = dc_disabled.solve(problem="test")
        assert result.sub_problems == []
        assert result.verified_steps == []
        assert result.final_code == ""


# ---------------------------------------------------------------------------
# Test: parse_decomposition
# ---------------------------------------------------------------------------

class TestParseDecomposition:

    def test_parses_two_sub_problems(self):
        sps = parse_decomposition(DECOMPOSITION_RESPONSE)
        assert len(sps) == 2

    def test_description_extracted(self):
        sps = parse_decomposition(DECOMPOSITION_RESPONSE)
        assert "Parse the input" in sps[0].description
        assert "Sort the list" in sps[1].description

    def test_input_format_extracted(self):
        sps = parse_decomposition(DECOMPOSITION_RESPONSE)
        assert "space-separated" in sps[0].input_format

    def test_output_format_extracted(self):
        sps = parse_decomposition(DECOMPOSITION_RESPONSE)
        assert "list of integers" in sps[0].output_format

    def test_test_case_extracted(self):
        sps = parse_decomposition(DECOMPOSITION_RESPONSE)
        assert "assert" in sps[0].test_case

    def test_respects_max_steps(self):
        sps = parse_decomposition(DECOMPOSITION_RESPONSE, max_steps=1)
        assert len(sps) == 1

    def test_empty_response(self):
        sps = parse_decomposition("")
        assert sps == []

    def test_no_sub_problem_blocks(self):
        sps = parse_decomposition("Just some text without any structure.")
        assert sps == []

    def test_index_assigned_sequentially(self):
        sps = parse_decomposition(DECOMPOSITION_RESPONSE)
        assert sps[0].index == 0
        assert sps[1].index == 1

    def test_case_insensitive_headers(self):
        response = (
            "sub-problem 1:\n"
            "description: Lower case test\n"
            "input: stdin\n"
            "output: stdout\n"
            "test: assert True\n"
        )
        sps = parse_decomposition(response)
        assert len(sps) == 1
        assert "Lower case" in sps[0].description

    def test_skips_empty_description(self):
        response = (
            "SUB-PROBLEM 1:\n"
            "INPUT: something\n"
            "OUTPUT: something\n"
        )
        sps = parse_decomposition(response)
        assert len(sps) == 0

    def test_many_sub_problems_capped(self):
        blocks = ""
        for i in range(1, 10):
            blocks += (
                f"SUB-PROBLEM {i}:\n"
                f"DESCRIPTION: Step {i}\n"
                f"INPUT: data\n"
                f"OUTPUT: result\n"
                f"TEST: assert True\n\n"
            )
        sps = parse_decomposition(blocks, max_steps=3)
        assert len(sps) == 3


# ---------------------------------------------------------------------------
# Test: extract_code
# ---------------------------------------------------------------------------

class TestExtractCode:

    def test_python_block(self):
        response = "Here is the code:\n```python\ndef solve(): return 42\n```"
        code = extract_code(response)
        assert code == "def solve(): return 42"

    def test_plain_code_block(self):
        response = "```\ndef solve(): return 42\n```"
        code = extract_code(response)
        assert code == "def solve(): return 42"

    def test_no_block_returns_stripped(self):
        response = "def solve(): return 42"
        code = extract_code(response)
        assert code == "def solve(): return 42"

    def test_multiple_python_blocks_takes_last(self):
        response = (
            "```python\ndef helper(): pass\n```\n"
            "```python\ndef solve(): return 1\n```"
        )
        code = extract_code(response)
        assert code == "def solve(): return 1"

    def test_strips_think_tags(self):
        response = "<think>reasoning</think>\n```python\ndef solve(): pass\n```"
        code = extract_code(response)
        assert code == "def solve(): pass"
        assert "<think>" not in code

    def test_empty_response(self):
        code = extract_code("")
        assert code == ""

    def test_prefers_python_over_plain(self):
        response = (
            "```\nplain code\n```\n"
            "```python\npython code\n```"
        )
        code = extract_code(response)
        assert code == "python code"


# ---------------------------------------------------------------------------
# Test: DerivationChains.solve — happy path
# ---------------------------------------------------------------------------

class TestSolveHappyPath:

    def test_full_solve_succeeds(self, dc_enabled):
        mock_llm = MockLLM()
        mock_sandbox = MockSandbox()
        result = dc_enabled.solve(
            problem="Sort a space-separated list of integers",
            failure_context="Previous attempts used wrong algorithm",
            llm_call=mock_llm,
            sandbox_run=mock_sandbox,
            task_id="t1",
        )
        assert result.triggered is True
        assert result.solved is True
        assert result.reason == "composed"
        assert len(result.final_code) > 0
        assert len(result.sub_problems) == 2
        assert len(result.verified_steps) == 2

    def test_all_steps_verified(self, dc_enabled):
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(), task_id="t1",
        )
        for vs in result.verified_steps:
            assert vs.verified is True
            assert vs.attempts >= 1

    def test_tokens_tracked(self, dc_enabled):
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(),
        )
        assert result.total_tokens > 0

    def test_time_tracked(self, dc_enabled):
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(),
        )
        assert result.total_time_ms > 0


# ---------------------------------------------------------------------------
# Test: DerivationChains.solve — stops when step fails
# ---------------------------------------------------------------------------

class TestSolveStopsOnFailure:

    def test_stops_when_step_fails(self, dc_enabled):
        mock_sandbox = MockSandboxAlwaysFail()
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=mock_sandbox, task_id="t1",
        )
        assert result.solved is False
        assert "failed" in result.reason

    def test_reason_includes_step_index(self, dc_enabled):
        mock_sandbox = MockSandboxAlwaysFail()
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=mock_sandbox,
        )
        assert result.reason == "step_0_failed"

    def test_retries_before_failing(self, tmp_telemetry):
        cfg = DerivationChainsConfig(enabled=True, max_attempts_per_step=3)
        dc = DerivationChains(cfg, telemetry_dir=tmp_telemetry)
        mock_sandbox = MockSandboxAlwaysFail()
        dc.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=mock_sandbox,
        )
        # 2 sub-problems but should fail on first, attempting 3 times
        assert len(mock_sandbox.calls) == 3

    def test_succeeds_after_retry(self, dc_enabled):
        mock_sandbox = MockSandboxFailThenPass(fail_count=1)
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=mock_sandbox,
        )
        assert result.solved is True
        assert result.verified_steps[0].attempts == 2


# ---------------------------------------------------------------------------
# Test: DerivationChains.solve — decomposition fails
# ---------------------------------------------------------------------------

class TestSolveDecompositionFails:

    def test_empty_decomposition_returns_not_solved(self, dc_enabled):
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLMEmpty(),
            sandbox_run=MockSandbox(), task_id="t1",
        )
        assert result.solved is False
        assert result.reason == "decomposition_failed"
        assert result.triggered is True


# ---------------------------------------------------------------------------
# Test: Missing callables
# ---------------------------------------------------------------------------

class TestMissingCallables:

    def test_missing_llm(self, dc_enabled):
        result = dc_enabled.solve(
            problem="test", llm_call=None,
            sandbox_run=MockSandbox(),
        )
        assert result.reason == "missing_callables"
        assert result.triggered is False

    def test_missing_sandbox(self, dc_enabled):
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=None,
        )
        assert result.reason == "missing_callables"
        assert result.triggered is False

    def test_both_missing(self, dc_enabled):
        result = dc_enabled.solve(problem="test")
        assert result.reason == "missing_callables"


# ---------------------------------------------------------------------------
# Test: max_sub_problems config
# ---------------------------------------------------------------------------

class TestMaxSubProblems:

    def test_limits_sub_problems(self, tmp_telemetry):
        cfg = DerivationChainsConfig(enabled=True, max_sub_problems=1)
        dc = DerivationChains(cfg, telemetry_dir=tmp_telemetry)
        result = dc.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(),
        )
        assert len(result.sub_problems) <= 1

    def test_max_attempts_per_step(self, tmp_telemetry):
        cfg = DerivationChainsConfig(enabled=True, max_attempts_per_step=2)
        dc = DerivationChains(cfg, telemetry_dir=tmp_telemetry)
        mock_sandbox = MockSandboxAlwaysFail()
        dc.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=mock_sandbox,
        )
        # Should attempt exactly 2 times on first sub-problem before failing
        assert len(mock_sandbox.calls) == 2


# ---------------------------------------------------------------------------
# Test: AC-3D-1 — Decomposes into verified sub-steps
# ---------------------------------------------------------------------------

class TestAC3D1Decomposition:
    """AC-3D-1: System decomposes problems into sequential sub-steps."""

    def test_produces_sub_problems(self, dc_enabled):
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(),
        )
        assert len(result.sub_problems) > 0

    def test_sub_problems_have_descriptions(self, dc_enabled):
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(),
        )
        for sp in result.sub_problems:
            assert len(sp.description) > 0


# ---------------------------------------------------------------------------
# Test: AC-3D-2 — Each step is sandbox-verified
# ---------------------------------------------------------------------------

class TestAC3D2Verification:
    """AC-3D-2: Each sub-step solution is sandbox-verified before proceeding."""

    def test_all_steps_verified_on_success(self, dc_enabled):
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(),
        )
        for vs in result.verified_steps:
            assert vs.verified is True

    def test_sandbox_called_for_each_step(self, dc_enabled):
        mock_sandbox = MockSandbox()
        dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=mock_sandbox,
        )
        assert len(mock_sandbox.calls) >= 2


# ---------------------------------------------------------------------------
# Test: AC-3D-3 — Stops on step failure
# ---------------------------------------------------------------------------

class TestAC3D3StopsOnFailure:
    """AC-3D-3: Chain stops when a sub-step cannot be verified."""

    def test_fails_early(self, dc_enabled):
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandboxAlwaysFail(),
        )
        assert result.solved is False
        assert "step_0_failed" in result.reason

    def test_partial_steps_recorded(self, dc_enabled):
        """When step 1 fails, step 0 may still be verified."""
        # Sandbox passes first sub-problem (1 call), fails on second (3 attempts)
        call_count = 0

        class SandboxPassFirstOnly:
            def __init__(self):
                self.calls = []

            def __call__(self, code, test_case):
                self.calls.append(code)
                nonlocal call_count
                call_count += 1
                if call_count <= 1:
                    return True, "OK", ""
                return False, "", "Error"

        mock_sandbox = SandboxPassFirstOnly()
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=mock_sandbox,
        )
        assert result.solved is False
        assert len(result.verified_steps) == 1
        assert result.verified_steps[0].verified is True
        assert result.reason == "step_1_failed"


# ---------------------------------------------------------------------------
# Test: AC-3D-4 — Composes final solution from verified steps
# ---------------------------------------------------------------------------

class TestAC3D4Composition:
    """AC-3D-4: Verified sub-solutions are composed into a complete solution."""

    def test_final_code_present(self, dc_enabled):
        result = dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(),
        )
        assert len(result.final_code) > 0
        assert "def " in result.final_code or "solve" in result.final_code

    def test_composition_uses_llm(self, dc_enabled):
        mock_llm = MockLLM()
        dc_enabled.solve(
            problem="test", llm_call=mock_llm,
            sandbox_run=MockSandbox(),
        )
        # Should have: 1 decomposition + 2 steps + 1 composition = 4 calls
        assert len(mock_llm.calls) == 4
        # Last call should contain "Compose"
        last_prompt = mock_llm.calls[-1]["prompt"]
        assert "Compose" in last_prompt


# ---------------------------------------------------------------------------
# Test: Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_event_written_to_jsonl(self, dc_enabled, tmp_telemetry):
        dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(), task_id="LCB_001",
        )
        events_file = tmp_telemetry / "derivation_chains_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "LCB_001"
        assert data["triggered"] is True
        assert data["solved"] is True
        assert data["num_sub_problems"] == 2
        assert data["num_verified"] == 2
        assert data["total_tokens"] > 0
        assert "timestamp" in data
        assert "reason" in data

    def test_multiple_events_appended(self, dc_enabled, tmp_telemetry):
        for i in range(3):
            dc_enabled.solve(
                problem="test", llm_call=MockLLM(),
                sandbox_run=MockSandbox(), task_id=f"t{i}",
            )
        events_file = tmp_telemetry / "derivation_chains_events.jsonl"
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_no_telemetry_without_task_id(self, dc_enabled, tmp_telemetry):
        dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(),
        )
        events_file = tmp_telemetry / "derivation_chains_events.jsonl"
        assert not events_file.exists()

    def test_no_crash_without_telemetry_dir(self):
        cfg = DerivationChainsConfig(enabled=True)
        dc = DerivationChains(cfg, telemetry_dir=None)
        result = dc.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandbox(), task_id="t1",
        )
        assert isinstance(result, DerivationResult)

    def test_failed_solve_logs_telemetry(self, dc_enabled, tmp_telemetry):
        dc_enabled.solve(
            problem="test", llm_call=MockLLM(),
            sandbox_run=MockSandboxAlwaysFail(), task_id="fail_task",
        )
        events_file = tmp_telemetry / "derivation_chains_events.jsonl"
        assert events_file.exists()
        data = json.loads(events_file.read_text().strip())
        assert data["solved"] is False
        assert data["task_id"] == "fail_task"


# ---------------------------------------------------------------------------
# Test: Data structures
# ---------------------------------------------------------------------------

class TestDataStructures:

    def test_sub_problem_to_dict(self):
        sp = SubProblem(
            description="Parse input" * 30,
            input_format="stdin",
            output_format="list",
            test_case="assert parse('1 2') == [1,2]",
            index=2,
        )
        d = sp.to_dict()
        assert d["index"] == 2
        assert len(d["description"]) <= 200
        assert d["has_test_case"] is True

    def test_sub_problem_no_test_case(self):
        sp = SubProblem(description="Step", test_case="")
        d = sp.to_dict()
        assert d["has_test_case"] is False

    def test_verified_step_to_dict(self):
        vs = VerifiedStep(
            sub_problem=SubProblem(description="Step", index=1),
            solution_code="def solve(): pass",
            verified=True,
            attempts=2,
        )
        d = vs.to_dict()
        assert d["sub_problem_index"] == 1
        assert d["verified"] is True
        assert d["attempts"] == 2
        assert d["code_length"] == len("def solve(): pass")

    def test_derivation_result_to_dict(self):
        r = DerivationResult(
            task_id="t1", triggered=True, solved=True,
            final_code="def solve(): pass",
            sub_problems=[SubProblem(description="s1"), SubProblem(description="s2")],
            verified_steps=[
                VerifiedStep(
                    sub_problem=SubProblem(description="s1"),
                    solution_code="code1", verified=True, attempts=1,
                ),
            ],
            total_tokens=300, total_time_ms=500.0, reason="composed",
        )
        d = r.to_dict()
        assert d["task_id"] == "t1"
        assert d["triggered"] is True
        assert d["solved"] is True
        assert d["num_sub_problems"] == 2
        assert d["num_verified_steps"] == 1
        assert d["total_tokens"] == 300
        assert d["reason"] == "composed"

    def test_derivation_result_defaults(self):
        r = DerivationResult()
        assert r.task_id == ""
        assert r.triggered is False
        assert r.solved is False
        assert r.final_code == ""
        assert r.sub_problems == []
        assert r.verified_steps == []
        assert r.total_tokens == 0

    def test_event_to_dict(self):
        e = DerivationChainEvent(
            task_id="t1", triggered=True, solved=False,
            num_sub_problems=3, num_verified=1,
            total_tokens=200, total_time_ms=800.0,
            reason="step_2_failed",
        )
        d = e.to_dict()
        assert d["task_id"] == "t1"
        assert d["triggered"] is True
        assert d["solved"] is False
        assert d["num_sub_problems"] == 3
        assert d["num_verified"] == 1
        assert d["reason"] == "step_2_failed"
        assert "timestamp" in d

    def test_config_defaults(self):
        cfg = DerivationChainsConfig()
        assert cfg.enabled is False
        assert cfg.max_sub_problems == 5
        assert cfg.max_attempts_per_step == 3
        assert cfg.decomposition_temperature == 0.3
        assert cfg.step_generation_temperature == 0.4
        assert cfg.decomposition_max_tokens == 2048
        assert cfg.step_max_tokens == 4096


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_failure_context_included_in_prompt(self, dc_enabled):
        mock_llm = MockLLM()
        dc_enabled.solve(
            problem="test",
            failure_context="Off-by-one error in loop",
            llm_call=mock_llm,
            sandbox_run=MockSandbox(),
        )
        first_prompt = mock_llm.calls[0]["prompt"]
        assert "Off-by-one" in first_prompt

    def test_empty_failure_context(self, dc_enabled):
        mock_llm = MockLLM()
        dc_enabled.solve(
            problem="test",
            failure_context="",
            llm_call=mock_llm,
            sandbox_run=MockSandbox(),
        )
        first_prompt = mock_llm.calls[0]["prompt"]
        assert "Previous failures" not in first_prompt

    def test_single_sub_problem_solve(self, tmp_telemetry):
        """Test with decomposition that yields exactly one sub-problem."""
        single_response = (
            "SUB-PROBLEM 1:\n"
            "DESCRIPTION: Just solve it directly\n"
            "INPUT: integer\n"
            "OUTPUT: integer\n"
            "TEST: assert solve(1) == 2\n"
        )

        class SingleLLM:
            def __init__(self):
                self.calls = []

            def __call__(self, prompt, temp, max_tok, seed):
                self.calls.append(prompt)
                if "decompos" in prompt.lower() or "sub-problem" in prompt.lower():
                    return single_response, 50, 20.0
                if "Compose" in prompt:
                    return CODE_RESPONSE, 40, 15.0
                return CODE_RESPONSE, 30, 10.0

        cfg = DerivationChainsConfig(enabled=True)
        dc = DerivationChains(cfg, telemetry_dir=tmp_telemetry)
        result = dc.solve(
            problem="test", llm_call=SingleLLM(),
            sandbox_run=MockSandbox(),
        )
        assert result.solved is True
        assert len(result.sub_problems) == 1
        assert len(result.verified_steps) == 1
