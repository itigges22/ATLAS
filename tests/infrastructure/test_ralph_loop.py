"""
Tests for Ralph Loop code generation engine.

Validates retry logic, temperature escalation,
error accumulation, and success criteria.
"""

import sys
import time
import pytest

# Add atlas task-worker to path for imports
sys.path.insert(0, "/home/nobase/k8s/atlas/task-worker")


class TestRalphLoopImport:
    """Test Ralph Loop module can be imported."""

    def test_module_imports(self):
        """ralph_loop module should import successfully."""
        from ralph_loop import RalphLoop, StopReason, RalphLoopResult
        assert RalphLoop is not None
        assert StopReason is not None
        assert RalphLoopResult is not None

    def test_can_instantiate_class(self):
        """Should be able to instantiate RalphLoop class."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult

        # Create mock generator and executor
        def mock_generator(prompt, context, temperature):
            return GenerationResult(
                code="print('hello')",
                tokens_in=10,
                tokens_out=5,
                duration_ms=100,
                temperature=temperature
            )

        def mock_executor(code, context):
            return ExecutionResult(
                success=True,
                compile_success=True,
                tests_run=1,
                tests_passed=1,
                lint_score=8.0,
                stdout="hello",
                stderr=""
            )

        loop = RalphLoop(
            generator=mock_generator,
            executor=mock_executor,
            max_attempts=5
        )
        assert loop is not None
        assert loop.max_attempts == 5


class TestRalphLoopExecution:
    """Test Ralph Loop execution behavior."""

    def test_simple_task_succeeds_first_attempt(self):
        """Simple task should succeed on first attempt."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult, StopReason

        def mock_generator(prompt, context, temperature):
            return GenerationResult(
                code="def add(a, b): return a + b",
                tokens_in=20,
                tokens_out=10,
                duration_ms=50,
                temperature=temperature
            )

        def mock_executor(code, context):
            return ExecutionResult(
                success=True,
                compile_success=True,
                tests_run=1,
                tests_passed=1,
                lint_score=8.0,
                stdout="",
                stderr=""
            )

        loop = RalphLoop(
            generator=mock_generator,
            executor=mock_executor,
            max_attempts=5,
            require_tests_pass=True
        )

        result = loop.run(prompt="Write add function", context={})

        assert result.success is True, "Simple task should succeed"
        assert result.stop_reason == StopReason.SUCCESS
        assert len(result.attempts) == 1, "Should succeed on first attempt"
        assert result.final_code is not None

    def test_returns_generated_code_on_success(self):
        """Successful result should include generated code."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult

        expected_code = "def multiply(a, b): return a * b"

        def mock_generator(prompt, context, temperature):
            return GenerationResult(
                code=expected_code,
                tokens_in=20,
                tokens_out=15,
                duration_ms=50,
                temperature=temperature
            )

        def mock_executor(code, context):
            return ExecutionResult(
                success=True,
                compile_success=True,
                tests_run=1,
                tests_passed=1,
                lint_score=8.0,
                stdout="",
                stderr=""
            )

        loop = RalphLoop(generator=mock_generator, executor=mock_executor)
        result = loop.run(prompt="Test", context={})

        assert result.final_code == expected_code, "Should return generated code"


class TestRalphLoopRetry:
    """Test retry behavior."""

    def test_retry_triggers_on_test_failure(self):
        """Failed tests should trigger retry."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult

        attempt_count = [0]

        def mock_generator(prompt, context, temperature):
            attempt_count[0] += 1
            return GenerationResult(
                code="def broken(): pass",
                tokens_in=10,
                tokens_out=5,
                duration_ms=50,
                temperature=temperature
            )

        def mock_executor(code, context):
            # First two attempts fail, third succeeds
            if attempt_count[0] < 3:
                return ExecutionResult(
                    success=False,
                    compile_success=True,
                    tests_run=1,
                    tests_passed=0,
                    lint_score=5.0,
                    stdout="",
                    stderr="AssertionError"
                )
            return ExecutionResult(
                success=True,
                compile_success=True,
                tests_run=1,
                tests_passed=1,
                lint_score=8.0,
                stdout="",
                stderr=""
            )

        loop = RalphLoop(
            generator=mock_generator,
            executor=mock_executor,
            max_attempts=5
        )

        result = loop.run(prompt="Test", context={})

        assert result.success is True
        assert len(result.attempts) == 3, "Should take 3 attempts"


class TestRalphLoopTemperature:
    """Test temperature escalation."""

    def test_temperature_starts_at_base(self):
        """First attempt should use base temperature."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult

        recorded_temps = []

        def mock_generator(prompt, context, temperature):
            recorded_temps.append(temperature)
            return GenerationResult(
                code="code",
                tokens_in=10,
                tokens_out=5,
                duration_ms=50,
                temperature=temperature
            )

        def mock_executor(code, context):
            return ExecutionResult(
                success=True,
                compile_success=True,
                tests_run=1,
                tests_passed=1,
                lint_score=8.0,
                stdout="",
                stderr=""
            )

        loop = RalphLoop(
            generator=mock_generator,
            executor=mock_executor,
            base_temperature=0.3
        )

        loop.run(prompt="Test", context={})

        assert len(recorded_temps) >= 1
        assert recorded_temps[0] == 0.3, f"First temp should be 0.3, got {recorded_temps[0]}"

    def test_temperature_escalates_on_retry(self):
        """Temperature should increase with each retry."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult

        recorded_temps = []
        attempt_count = [0]

        def mock_generator(prompt, context, temperature):
            recorded_temps.append(temperature)
            attempt_count[0] += 1
            return GenerationResult(
                code="code",
                tokens_in=10,
                tokens_out=5,
                duration_ms=50,
                temperature=temperature
            )

        def mock_executor(code, context):
            # Fail first 3 attempts
            if attempt_count[0] < 4:
                return ExecutionResult(
                    success=False,
                    compile_success=True,
                    tests_run=1,
                    tests_passed=0,
                    lint_score=5.0,
                    stdout="",
                    stderr="fail"
                )
            return ExecutionResult(
                success=True,
                compile_success=True,
                tests_run=1,
                tests_passed=1,
                lint_score=8.0,
                stdout="",
                stderr=""
            )

        loop = RalphLoop(
            generator=mock_generator,
            executor=mock_executor,
            base_temperature=0.3,
            temperature_increment=0.1,
            max_attempts=5
        )

        loop.run(prompt="Test", context={})

        assert len(recorded_temps) >= 3
        # Temperatures should increase
        for i in range(1, len(recorded_temps)):
            assert recorded_temps[i] > recorded_temps[i-1], \
                f"Temperature should increase: {recorded_temps}"


class TestRalphLoopErrorAccumulation:
    """Test error context accumulation."""

    def test_error_context_from_previous_attempts(self):
        """Previous errors should be included in subsequent prompts."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult

        received_prompts = []
        attempt_count = [0]

        def mock_generator(prompt, context, temperature):
            received_prompts.append(prompt)
            attempt_count[0] += 1
            return GenerationResult(
                code="code",
                tokens_in=10,
                tokens_out=5,
                duration_ms=50,
                temperature=temperature
            )

        def mock_executor(code, context):
            if attempt_count[0] < 3:
                return ExecutionResult(
                    success=False,
                    compile_success=True,
                    tests_run=1,
                    tests_passed=0,
                    lint_score=5.0,
                    stdout="",
                    stderr="TypeError: bad input",
                    error_type="TypeError",
                    error_message="bad input"
                )
            return ExecutionResult(
                success=True,
                compile_success=True,
                tests_run=1,
                tests_passed=1,
                lint_score=8.0,
                stdout="",
                stderr=""
            )

        loop = RalphLoop(
            generator=mock_generator,
            executor=mock_executor,
            max_attempts=5
        )

        loop.run(prompt="Original prompt", context={})

        # Later prompts should mention previous errors
        assert len(received_prompts) >= 2
        # Second prompt should have error context
        assert "Previous Attempts" in received_prompts[1] or "TypeError" in received_prompts[1], \
            f"Error context should be in prompt: {received_prompts[1][:200]}"


class TestRalphLoopLimits:
    """Test max attempts and timeout."""

    def test_max_retries_enforced(self):
        """Should stop after max attempts."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult, StopReason

        def mock_generator(prompt, context, temperature):
            return GenerationResult(
                code="bad code",
                tokens_in=10,
                tokens_out=5,
                duration_ms=50,
                temperature=temperature
            )

        def mock_executor(code, context):
            # Always fail
            return ExecutionResult(
                success=False,
                compile_success=True,
                tests_run=1,
                tests_passed=0,
                lint_score=3.0,
                stdout="",
                stderr="fail"
            )

        loop = RalphLoop(
            generator=mock_generator,
            executor=mock_executor,
            max_attempts=3
        )

        result = loop.run(prompt="Test", context={})

        assert result.success is False
        assert result.stop_reason == StopReason.MAX_ATTEMPTS
        assert len(result.attempts) == 3, "Should have exactly 3 attempts"


class TestRalphLoopEarlyTermination:
    """Test early termination on unrecoverable errors."""

    def test_syntax_error_fails_compilation(self):
        """Syntax errors should be caught early."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult

        def mock_generator(prompt, context, temperature):
            # Generate code with syntax error
            return GenerationResult(
                code="def broken(\n  # missing close paren",
                tokens_in=10,
                tokens_out=5,
                duration_ms=50,
                temperature=temperature
            )

        def mock_executor(code, context):
            # Should not be called for syntax errors
            return ExecutionResult(
                success=True,
                compile_success=True,
                tests_run=1,
                tests_passed=1,
                lint_score=8.0,
                stdout="",
                stderr=""
            )

        loop = RalphLoop(
            generator=mock_generator,
            executor=mock_executor,
            max_attempts=5
        )

        result = loop.run(prompt="Test", context={})

        # Should fail due to syntax error
        assert any(not a.execution.compile_success for a in result.attempts), \
            "At least one attempt should fail compilation"


class TestRalphLoopMetrics:
    """Test metrics tracking."""

    def test_attempt_count_in_result(self):
        """Result should include attempt count."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult

        attempt_count = [0]

        def mock_generator(prompt, context, temperature):
            attempt_count[0] += 1
            return GenerationResult(
                code="code",
                tokens_in=10,
                tokens_out=5,
                duration_ms=50,
                temperature=temperature
            )

        def mock_executor(code, context):
            if attempt_count[0] < 2:
                return ExecutionResult(
                    success=False, compile_success=True, tests_run=1, tests_passed=0,
                    lint_score=5.0, stdout="", stderr="fail"
                )
            return ExecutionResult(
                success=True, compile_success=True, tests_run=1, tests_passed=1,
                lint_score=8.0, stdout="", stderr=""
            )

        loop = RalphLoop(generator=mock_generator, executor=mock_executor)
        result = loop.run(prompt="Test", context={})

        assert len(result.attempts) == 2

    def test_total_time_tracked(self):
        """Total duration should be tracked."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult

        def mock_generator(prompt, context, temperature):
            time.sleep(0.1)  # Small delay
            return GenerationResult(
                code="code", tokens_in=10, tokens_out=5,
                duration_ms=100, temperature=temperature
            )

        def mock_executor(code, context):
            return ExecutionResult(
                success=True, compile_success=True, tests_run=1, tests_passed=1,
                lint_score=8.0, stdout="", stderr=""
            )

        loop = RalphLoop(generator=mock_generator, executor=mock_executor)
        result = loop.run(prompt="Test", context={})

        assert result.total_duration_ms > 0, "Duration should be tracked"
        assert result.total_duration_ms >= 100, "Duration should be at least generator delay"

    def test_total_tokens_tracked(self):
        """Total tokens should be accumulated."""
        from ralph_loop import RalphLoop, GenerationResult, ExecutionResult

        attempt_count = [0]

        def mock_generator(prompt, context, temperature):
            attempt_count[0] += 1
            return GenerationResult(
                code="code", tokens_in=100, tokens_out=50,
                duration_ms=50, temperature=temperature
            )

        def mock_executor(code, context):
            if attempt_count[0] < 2:
                return ExecutionResult(
                    success=False, compile_success=True, tests_run=1, tests_passed=0,
                    lint_score=5.0, stdout="", stderr="fail"
                )
            return ExecutionResult(
                success=True, compile_success=True, tests_run=1, tests_passed=1,
                lint_score=8.0, stdout="", stderr=""
            )

        loop = RalphLoop(generator=mock_generator, executor=mock_executor)
        result = loop.run(prompt="Test", context={})

        # 2 attempts * (100 + 50) tokens
        assert result.total_tokens == 300, f"Expected 300 tokens, got {result.total_tokens}"
