"""
Ralph-Loop: Retry-until-success code generation engine.

Mathematical basis:
  P(success within k attempts) = 1 - (1 - p)^k

  Where p = single-attempt pass rate

  With p=0.65 and k=5: P(success) = 99.5%
  With p=0.65 and k=10: P(success) = 99.997%
"""

import time
import logging
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class StopReason(Enum):
    SUCCESS = "success"
    MAX_ATTEMPTS = "max_attempts"
    TIMEOUT = "timeout"
    UNRECOVERABLE = "unrecoverable_error"

@dataclass
class GenerationResult:
    code: str
    tokens_in: int
    tokens_out: int
    duration_ms: int
    temperature: float
    raw_response: Dict = field(default_factory=dict)

@dataclass
class ExecutionResult:
    success: bool
    compile_success: bool
    tests_run: int
    tests_passed: int
    lint_score: Optional[float]
    stdout: str
    stderr: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class Attempt:
    attempt_number: int
    started_at: float
    completed_at: float
    generation: GenerationResult
    execution: ExecutionResult
    success: bool

@dataclass
class RalphLoopResult:
    success: bool
    stop_reason: StopReason
    attempts: List[Attempt]
    final_code: Optional[str]
    total_duration_ms: int
    total_tokens: int

class RalphLoop:
    """
    Retry-until-success code generation engine.

    Usage:
        loop = RalphLoop(
            generator=my_llm_generator,
            executor=my_code_executor,
            max_attempts=5
        )
        result = loop.run(prompt="Implement function X", context={...})
    """

    def __init__(
        self,
        generator: Callable,  # (prompt, context, temperature) -> GenerationResult
        executor: Callable,   # (code, project_context) -> ExecutionResult
        max_attempts: int = 5,
        timeout_seconds: int = 300,
        base_temperature: float = 0.3,
        temperature_increment: float = 0.1,
        require_tests_pass: bool = True,
        require_lint_pass: bool = False,
        min_lint_score: float = 6.0
    ):
        self.generator = generator
        self.executor = executor
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.base_temperature = base_temperature
        self.temperature_increment = temperature_increment
        self.require_tests_pass = require_tests_pass
        self.require_lint_pass = require_lint_pass
        self.min_lint_score = min_lint_score

    def run(
        self,
        prompt: str,
        context: Dict,
        on_attempt: Optional[Callable] = None  # Callback after each attempt
    ) -> RalphLoopResult:
        """
        Execute the ralph-loop until success or termination.

        Args:
            prompt: The code generation task
            context: RAG context, project info, etc.
            on_attempt: Optional callback(attempt) after each try

        Returns:
            RalphLoopResult with success status and all attempts
        """
        attempts = []
        start_time = time.time()
        total_tokens = 0
        accumulated_errors = []

        for attempt_num in range(1, self.max_attempts + 1):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                return self._make_result(
                    success=False,
                    stop_reason=StopReason.TIMEOUT,
                    attempts=attempts,
                    start_time=start_time,
                    total_tokens=total_tokens
                )

            # Calculate temperature (increases with attempts for diversity)
            temperature = self.base_temperature + (attempt_num - 1) * self.temperature_increment
            temperature = min(temperature, 1.0)  # Cap at 1.0

            # Build enhanced prompt with error feedback
            enhanced_prompt = self._build_prompt(prompt, accumulated_errors)

            attempt_start = time.time()

            try:
                # Generate code
                logger.info(f"Attempt {attempt_num}/{self.max_attempts}, temp={temperature:.2f}")
                gen_result = self.generator(enhanced_prompt, context, temperature)
                total_tokens += gen_result.tokens_in + gen_result.tokens_out

                # Quick syntax check before full execution
                if not self._syntax_valid(gen_result.code):
                    logger.info(f"Attempt {attempt_num}: Syntax error, skipping execution")
                    exec_result = ExecutionResult(
                        success=False,
                        compile_success=False,
                        tests_run=0,
                        tests_passed=0,
                        lint_score=None,
                        stdout="",
                        stderr="Syntax error in generated code",
                        error_type="SyntaxError",
                        error_message="Failed initial syntax check"
                    )
                else:
                    # Execute and verify
                    exec_result = self.executor(gen_result.code, context)

                attempt = Attempt(
                    attempt_number=attempt_num,
                    started_at=attempt_start,
                    completed_at=time.time(),
                    generation=gen_result,
                    execution=exec_result,
                    success=self._is_success(exec_result)
                )
                attempts.append(attempt)

                if on_attempt:
                    on_attempt(attempt)

                # Check success
                if attempt.success:
                    logger.info(f"Success on attempt {attempt_num}")
                    return self._make_result(
                        success=True,
                        stop_reason=StopReason.SUCCESS,
                        attempts=attempts,
                        start_time=start_time,
                        total_tokens=total_tokens,
                        final_code=gen_result.code
                    )

                # Accumulate error for next attempt's context
                accumulated_errors.append({
                    "attempt": attempt_num,
                    "error_type": exec_result.error_type,
                    "error_message": exec_result.error_message,
                    "stderr": exec_result.stderr[:500]  # Truncate
                })

                # Check for unrecoverable errors
                if self._is_unrecoverable(exec_result):
                    logger.warning(f"Unrecoverable error: {exec_result.error_type}")
                    return self._make_result(
                        success=False,
                        stop_reason=StopReason.UNRECOVERABLE,
                        attempts=attempts,
                        start_time=start_time,
                        total_tokens=total_tokens
                    )

            except Exception as e:
                logger.error(f"Attempt {attempt_num} exception: {e}")
                accumulated_errors.append({
                    "attempt": attempt_num,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })

        # Max attempts reached
        return self._make_result(
            success=False,
            stop_reason=StopReason.MAX_ATTEMPTS,
            attempts=attempts,
            start_time=start_time,
            total_tokens=total_tokens
        )

    def _build_prompt(self, original_prompt: str, errors: List[Dict]) -> str:
        """Enhance prompt with error feedback from previous attempts."""
        if not errors:
            return original_prompt

        error_context = "\n\n## Previous Attempts (DO NOT repeat these mistakes):\n"
        for err in errors[-3:]:  # Only include last 3 errors
            error_context += f"\nAttempt {err['attempt']} failed with {err['error_type']}:\n"
            error_context += f"  {err['error_message']}\n"
            if err.get('stderr'):
                error_context += f"  stderr: {err['stderr'][:200]}\n"

        return original_prompt + error_context

    def _syntax_valid(self, code: str) -> bool:
        """Quick syntax check before full execution."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def _is_success(self, result: ExecutionResult) -> bool:
        """Determine if execution result meets success criteria."""
        if not result.compile_success:
            return False

        if self.require_tests_pass and result.tests_passed < result.tests_run:
            return False

        if self.require_lint_pass and (result.lint_score or 0) < self.min_lint_score:
            return False

        return True

    def _is_unrecoverable(self, result: ExecutionResult) -> bool:
        """Detect errors that won't be fixed by retrying."""
        unrecoverable_types = [
            "MissingDependency",      # Can't install packages
            "PermissionError",        # Sandbox restrictions
            "ResourceExhausted",      # Out of memory/time
        ]
        return result.error_type in unrecoverable_types

    def _make_result(
        self,
        success: bool,
        stop_reason: StopReason,
        attempts: List[Attempt],
        start_time: float,
        total_tokens: int,
        final_code: Optional[str] = None
    ) -> RalphLoopResult:
        """Construct final result object."""
        return RalphLoopResult(
            success=success,
            stop_reason=stop_reason,
            attempts=attempts,
            final_code=final_code,
            total_duration_ms=int((time.time() - start_time) * 1000),
            total_tokens=total_tokens
        )
