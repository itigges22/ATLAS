"""V3 Derivation Chains (Feature 3D) — Verified Problem Decomposition.

For complex problems where single-pass generation fails even with refined
constraints, decomposes the problem into sub-steps. Each step is sandbox-
verified before becoming the foundation for the next, preventing error
accumulation across the chain.

Config: [derivation_chains] in atlas.conf
Telemetry: telemetry/derivation_chains_events.jsonl

The insight: LLM chains degrade — 87.8% accuracy per step means 27%
confidence after 10 steps. But sandbox-verified steps don't accumulate
errors. Each verified step is as reliable as the original problem statement.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# Type aliases
LLMCallable = Callable[[str, float, int, Optional[int]], Tuple[str, int, float]]
SandboxCallable = Callable[[str, str], Tuple[bool, str, str]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DerivationChainsConfig:
    """Configuration for Verified Derivation Chains."""
    enabled: bool = False
    max_sub_problems: int = 5
    max_attempts_per_step: int = 3
    decomposition_temperature: float = 0.3
    step_generation_temperature: float = 0.4
    decomposition_max_tokens: int = 2048
    step_max_tokens: int = 4096


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubProblem:
    """A decomposed sub-problem."""
    description: str
    input_format: str = ""
    output_format: str = ""
    test_case: str = ""
    index: int = 0

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "description": self.description[:200],
            "has_test_case": bool(self.test_case),
        }


@dataclass
class VerifiedStep:
    """A sub-problem solution that has been sandbox-verified."""
    sub_problem: SubProblem
    solution_code: str
    verified: bool = False
    attempts: int = 0

    def to_dict(self) -> Dict:
        return {
            "sub_problem_index": self.sub_problem.index,
            "verified": self.verified,
            "attempts": self.attempts,
            "code_length": len(self.solution_code),
        }


@dataclass
class DerivationResult:
    """Result of a derivation chain attempt."""
    task_id: str = ""
    triggered: bool = False
    solved: bool = False
    final_code: str = ""
    sub_problems: List[SubProblem] = field(default_factory=list)
    verified_steps: List[VerifiedStep] = field(default_factory=list)
    total_tokens: int = 0
    total_time_ms: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "triggered": self.triggered,
            "solved": self.solved,
            "num_sub_problems": len(self.sub_problems),
            "num_verified_steps": len(self.verified_steps),
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Telemetry event
# ---------------------------------------------------------------------------

@dataclass
class DerivationChainEvent:
    """Telemetry event for a derivation chain attempt."""
    task_id: str
    triggered: bool = False
    solved: bool = False
    num_sub_problems: int = 0
    num_verified: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    reason: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "triggered": self.triggered,
            "solved": self.solved,
            "num_sub_problems": self.num_sub_problems,
            "num_verified": self.num_verified,
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "reason": self.reason,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DECOMPOSITION_PROMPT = """\
This problem has resisted multiple solution attempts:
{problem}

{failure_context}

Decompose this into {max_steps} or fewer sequential sub-problems where:
1. Each sub-problem is independently testable
2. Each sub-problem's output feeds into the next
3. The final sub-problem produces the complete solution

For each sub-problem, use this format:
SUB-PROBLEM 1:
DESCRIPTION: <what this step does>
INPUT: <what input it takes>
OUTPUT: <what output format>
TEST: <a simple test case>"""

STEP_GENERATION_PROMPT = """\
Solve this sub-problem as Python code:

{description}

Input format: {input_format}
Output format: {output_format}

{context}

Write a complete Python function for this sub-step."""

COMPOSITION_PROMPT = """\
Compose these verified sub-solutions into a complete solution:

Problem: {problem}

{verified_steps}

Write the complete Python solution that uses these verified components."""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_decomposition(response: str,
                        max_steps: int = 5) -> List[SubProblem]:
    """Parse sub-problems from decomposition response."""
    sub_problems: List[SubProblem] = []

    pattern = r'SUB-PROBLEM\s+(\d+)\s*:(.*?)(?=SUB-PROBLEM\s+\d+\s*:|$)'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

    for num_str, block in matches:
        sp = SubProblem(description="", index=len(sub_problems))

        desc_match = re.search(r'DESCRIPTION[:\s]*(.*?)(?=\nINPUT|\nOUTPUT|\nTEST|$)',
                                block, re.DOTALL | re.IGNORECASE)
        if desc_match:
            sp.description = desc_match.group(1).strip()

        input_match = re.search(r'INPUT[:\s]*(.*?)(?=\nOUTPUT|\nTEST|$)',
                                 block, re.DOTALL | re.IGNORECASE)
        if input_match:
            sp.input_format = input_match.group(1).strip()

        output_match = re.search(r'OUTPUT[:\s]*(.*?)(?=\nTEST|$)',
                                  block, re.DOTALL | re.IGNORECASE)
        if output_match:
            sp.output_format = output_match.group(1).strip()

        test_match = re.search(r'TEST[:\s]*(.*?)$',
                                block, re.DOTALL | re.IGNORECASE)
        if test_match:
            sp.test_case = test_match.group(1).strip()

        if sp.description:
            sub_problems.append(sp)

    return sub_problems[:max_steps]


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    py_blocks = re.findall(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if py_blocks:
        return py_blocks[-1].strip()
    code_blocks = re.findall(r'```\s*\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DerivationChains:
    """Verified Derivation Chain solver.

    When enabled, decomposes complex problems into sandbox-verified sub-steps.
    Each step is tested before becoming the foundation for the next step.

    When disabled, returns an empty result (noop).

    Args:
        config: DerivationChainsConfig instance.
        telemetry_dir: Directory for JSONL event logs.
    """

    def __init__(self, config: DerivationChainsConfig,
                 telemetry_dir: Optional[Path] = None):
        self.config = config
        self.telemetry_dir = telemetry_dir
        self._events_file: Optional[Path] = None
        if telemetry_dir is not None:
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            self._events_file = telemetry_dir / "derivation_chains_events.jsonl"

    def solve(self, problem: str,
              failure_context: str = "",
              llm_call: Optional[LLMCallable] = None,
              sandbox_run: Optional[SandboxCallable] = None,
              task_id: str = "") -> DerivationResult:
        """Attempt to solve via problem decomposition.

        Args:
            problem: Original problem description.
            failure_context: Description of why prior attempts failed.
            llm_call: LLM callable for generation.
            sandbox_run: Sandbox callable for verification.
            task_id: Task identifier for telemetry.

        Returns:
            DerivationResult with sub-problems, verified steps, and outcome.
        """
        if not self.config.enabled:
            return DerivationResult(task_id=task_id, reason="disabled")

        if llm_call is None or sandbox_run is None:
            return DerivationResult(task_id=task_id, reason="missing_callables")

        start_time = time.time()
        result = DerivationResult(task_id=task_id, triggered=True)
        total_tokens = 0

        # Step 1: Decompose
        decomp_prompt = self._build_decomposition_prompt(problem, failure_context)
        decomp_response, decomp_tokens, _ = llm_call(
            decomp_prompt,
            self.config.decomposition_temperature,
            self.config.decomposition_max_tokens,
            42,
        )
        total_tokens += decomp_tokens

        sub_problems = parse_decomposition(
            decomp_response, self.config.max_sub_problems
        )
        result.sub_problems = sub_problems

        if not sub_problems:
            result.reason = "decomposition_failed"
            result.total_time_ms = (time.time() - start_time) * 1000
            result.total_tokens = total_tokens
            self._log_telemetry(task_id, result)
            return result

        if len(sub_problems) > self.config.max_sub_problems:
            result.reason = "too_many_sub_problems"
            result.total_time_ms = (time.time() - start_time) * 1000
            result.total_tokens = total_tokens
            self._log_telemetry(task_id, result)
            return result

        # Step 2: Solve each sub-problem with verification
        verified_steps: List[VerifiedStep] = []

        for sp in sub_problems:
            context = self._format_verified_context(verified_steps)
            solved_step = False

            for attempt in range(self.config.max_attempts_per_step):
                step_prompt = self._build_step_prompt(sp, context)
                step_response, step_tokens, _ = llm_call(
                    step_prompt,
                    self.config.step_generation_temperature,
                    self.config.step_max_tokens,
                    42 + sp.index + attempt * 100,
                )
                total_tokens += step_tokens
                code = extract_code(step_response)

                # Verify in sandbox
                passed, stdout, stderr = sandbox_run(code, sp.test_case)
                if passed:
                    verified_steps.append(VerifiedStep(
                        sub_problem=sp,
                        solution_code=code,
                        verified=True,
                        attempts=attempt + 1,
                    ))
                    solved_step = True
                    break

            if not solved_step:
                result.verified_steps = verified_steps
                result.reason = f"step_{sp.index}_failed"
                result.total_time_ms = (time.time() - start_time) * 1000
                result.total_tokens = total_tokens
                self._log_telemetry(task_id, result)
                return result

        result.verified_steps = verified_steps

        # Step 3: Compose final solution
        compose_prompt = self._build_composition_prompt(problem, verified_steps)
        compose_response, compose_tokens, _ = llm_call(
            compose_prompt,
            self.config.step_generation_temperature,
            self.config.step_max_tokens,
            42,
        )
        total_tokens += compose_tokens
        final_code = extract_code(compose_response)

        result.final_code = final_code
        result.solved = True
        result.reason = "composed"
        result.total_time_ms = (time.time() - start_time) * 1000
        result.total_tokens = total_tokens

        self._log_telemetry(task_id, result)
        return result

    # -- Helpers ------------------------------------------------------------

    def _build_decomposition_prompt(self, problem: str,
                                     failure_context: str) -> str:
        ctx = f"Previous failures: {failure_context}" if failure_context else ""
        user_content = DECOMPOSITION_PROMPT.format(
            problem=problem, failure_context=ctx,
            max_steps=self.config.max_sub_problems,
        )
        system = "You are an expert at problem decomposition."
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _build_step_prompt(self, sp: SubProblem, context: str) -> str:
        user_content = STEP_GENERATION_PROMPT.format(
            description=sp.description,
            input_format=sp.input_format or "standard",
            output_format=sp.output_format or "standard",
            context=f"Context from previous verified steps:\n{context}" if context else "",
        )
        system = "You are an expert programmer. Think carefully about the approach, then write correct Python code."
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _build_composition_prompt(self, problem: str,
                                   steps: List[VerifiedStep]) -> str:
        steps_text = ""
        for vs in steps:
            steps_text += (
                f"Step {vs.sub_problem.index + 1}: {vs.sub_problem.description}\n"
                f"```python\n{vs.solution_code}\n```\n\n"
            )
        user_content = COMPOSITION_PROMPT.format(
            problem=problem, verified_steps=steps_text,
        )
        system = "You are an expert programmer. Think carefully about how to combine the steps, then write the complete solution."
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _format_verified_context(self, steps: List[VerifiedStep]) -> str:
        if not steps:
            return ""
        parts = []
        for vs in steps:
            parts.append(
                f"Step {vs.sub_problem.index + 1} (verified): "
                f"{vs.sub_problem.description}"
            )
        return '\n'.join(parts)

    def _log_telemetry(self, task_id: str, result: DerivationResult) -> None:
        if not task_id:
            return
        self._log_event(DerivationChainEvent(
            task_id=task_id,
            triggered=result.triggered,
            solved=result.solved,
            num_sub_problems=len(result.sub_problems),
            num_verified=len(result.verified_steps),
            total_tokens=result.total_tokens,
            total_time_ms=result.total_time_ms,
            reason=result.reason,
        ))

    def _log_event(self, event: DerivationChainEvent) -> None:
        if self._events_file is None:
            return
        try:
            with open(self._events_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except OSError:
            pass
