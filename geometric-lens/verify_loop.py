"""Verify-Repair-Retry loop for code generation.

Closed-loop pipeline that combines:
- Sandbox execution (does the code run?)
- G(x) XGBoost scoring (is the code likely correct?)
- Structured error analysis (what went wrong and how to fix it?)
- Repair prompt construction (inject targeted fix instructions)

Integrates into rag_enhanced_completion() as a post-generation step.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    """Result of the verify-repair-retry loop."""
    passed: bool
    final_code: Optional[str]
    gx_score: float
    cx_energy: float
    cx_normalized: float
    verdict: str
    attempts: int
    max_attempts: int
    total_latency_ms: float
    # History of all attempts for telemetry
    attempt_history: List[Dict[str, Any]]

    def to_dict(self) -> dict:
        return asdict(self)


async def verify_and_repair(
    response_text: str,
    test_code: str = "",
    stdin: str = "",
    expected_output: Optional[str] = None,
    retry_budget: int = 3,
    messages: Optional[List[Dict[str, str]]] = None,
    model: str = "",
    forward_fn=None,
    max_tokens: int = 16384,
    **kwargs,
) -> VerifyResult:
    """Run the closed-loop verify-repair-retry pipeline.

    1. Extract code from LLM response
    2. Run in sandbox
    3. If passed + high G(x): return success
    4. If failed + recoverable: build repair prompt, regenerate
    5. If failed + not recoverable: signal restructure needed
    6. Repeat up to retry_budget times

    Args:
        response_text: The LLM's generated response
        test_code: Test code to validate against
        stdin: stdin input for sandbox
        expected_output: Expected output for mismatch detection
        retry_budget: Maximum number of repair attempts
        messages: Original conversation messages (for repair context)
        model: Model name for regeneration
        forward_fn: Async function to call LLM for regeneration
        max_tokens: Max tokens for regeneration
        **kwargs: Additional args for forward_fn

    Returns:
        VerifyResult with final code, scores, and attempt history
    """
    from sandbox_client import execute_sandbox, extract_code_from_response
    from sandbox_analysis import analyze_sandbox_output, build_repair_prompt

    start = time.monotonic()
    attempt_history = []
    best_code = None
    best_gx = 0.0
    current_response = response_text
    max_attempts = min(retry_budget + 1, 6)  # Cap at 6 total attempts

    for attempt in range(max_attempts):
        attempt_start = time.monotonic()

        # Extract code from response
        code = extract_code_from_response(current_response)
        if code is None:
            attempt_history.append({
                "attempt": attempt + 1,
                "status": "no_code_found",
                "response_length": len(current_response),
            })
            logger.debug(f"Attempt {attempt + 1}: no code found in response")
            break

        # Score with G(x) + C(x) (combined, single embedding call)
        gx_result = _score_code(code)
        gx_score = gx_result.get("gx_score", 0.5)
        cx_energy = gx_result.get("cx_energy", 0.0)
        cx_normalized = gx_result.get("cx_normalized", 0.5)
        verdict = gx_result.get("verdict", "unavailable")

        # Track best code (highest G(x) score)
        if gx_score > best_gx:
            best_gx = gx_score
            best_code = code

        # Run sandbox
        passed, stdout, stderr = await execute_sandbox(
            code=code,
            test_code=test_code,
            stdin=stdin,
        )

        # Analyze sandbox result
        analysis = analyze_sandbox_output(
            passed=passed,
            stdout=stdout,
            stderr=stderr,
            expected_output=expected_output,
            gx_score=gx_score,
        )

        attempt_ms = (time.monotonic() - attempt_start) * 1000
        attempt_record = {
            "attempt": attempt + 1,
            "passed": passed,
            "gx_score": gx_score,
            "cx_energy": cx_energy,
            "verdict": verdict,
            "failure_type": analysis.failure_type.value,
            "failure_line": analysis.failure_line,
            "is_recoverable": analysis.is_recoverable,
            "severity": analysis.severity.value,
            "latency_ms": round(attempt_ms, 1),
        }
        attempt_history.append(attempt_record)

        logger.info(
            f"Verify attempt {attempt + 1}/{max_attempts}: "
            f"passed={passed} G(x)={gx_score:.3f} "
            f"type={analysis.failure_type.value} "
            f"recoverable={analysis.is_recoverable}"
        )

        # Success: sandbox passed
        if passed:
            total_ms = (time.monotonic() - start) * 1000
            return VerifyResult(
                passed=True,
                final_code=code,
                gx_score=gx_score,
                cx_energy=cx_energy,
                cx_normalized=cx_normalized,
                verdict=verdict,
                attempts=attempt + 1,
                max_attempts=max_attempts,
                total_latency_ms=round(total_ms, 1),
                attempt_history=attempt_history,
            )

        # High G(x) but no test code — trust G(x) score
        if not test_code and gx_score >= 0.8 and not stderr:
            total_ms = (time.monotonic() - start) * 1000
            return VerifyResult(
                passed=True,  # Trusted via G(x)
                final_code=code,
                gx_score=gx_score,
                cx_energy=cx_energy,
                cx_normalized=cx_normalized,
                verdict="trusted_gx",
                attempts=attempt + 1,
                max_attempts=max_attempts,
                total_latency_ms=round(total_ms, 1),
                attempt_history=attempt_history,
            )

        # Last attempt — no more retries
        if attempt >= max_attempts - 1:
            break

        # Not recoverable — no point retrying with same approach
        if not analysis.is_recoverable and forward_fn is None:
            break

        # Build repair prompt and regenerate
        if forward_fn is not None and messages is not None:
            repair_context = build_repair_prompt(
                analysis=analysis,
                original_code=code,
                gx_score=gx_score,
            )

            # Append repair context as a new user message
            repair_messages = list(messages) + [
                {"role": "assistant", "content": current_response},
                {"role": "user", "content": repair_context},
            ]

            try:
                result = await forward_fn(
                    repair_messages, model, None, max_tokens, **kwargs
                )
                if isinstance(result, dict):
                    choices = result.get("choices", [])
                    if choices:
                        current_response = choices[0].get("message", {}).get("content", "")
                    else:
                        break
                else:
                    break
            except Exception as e:
                logger.error(f"Repair regeneration failed: {e}")
                break
        else:
            break

    # Exhausted budget — return best attempt
    total_ms = (time.monotonic() - start) * 1000
    return VerifyResult(
        passed=False,
        final_code=best_code,
        gx_score=best_gx,
        cx_energy=cx_energy if 'cx_energy' in dir() else 0.0,
        cx_normalized=cx_normalized if 'cx_normalized' in dir() else 0.5,
        verdict="budget_exhausted",
        attempts=len(attempt_history),
        max_attempts=max_attempts,
        total_latency_ms=round(total_ms, 1),
        attempt_history=attempt_history,
    )


def _score_code(code: str) -> dict:
    """Score code using combined C(x) + G(x) evaluation.

    Falls back gracefully if lens is disabled or models unavailable.
    """
    try:
        from geometric_lens.service import evaluate_combined, is_enabled
        if not is_enabled():
            return {"gx_score": 0.5, "cx_energy": 0.0, "cx_normalized": 0.5, "verdict": "unavailable"}

        return evaluate_combined(f"SOLUTION: {code}")
    except Exception as e:
        logger.error(f"Code scoring failed: {e}")
        return {"gx_score": 0.5, "cx_energy": 0.0, "cx_normalized": 0.5, "verdict": "error"}
