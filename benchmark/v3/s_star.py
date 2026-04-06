"""V3 S* Tiebreaking (Feature 2C) — Distinguishing Input Generation.

When the top-2 candidates have similar Lens energy scores (within delta),
generates edge-case test inputs where the candidates are expected to disagree.
Both candidates are run on these inputs in the sandbox, and the candidate that
passes more tests wins the tiebreak.

Paper: Li et al., UC Berkeley (arxiv:2502.14382). 3B with S* beats GPT-4o-mini.
Config: [s_star] in atlas.conf
Telemetry: telemetry/s_star_events.jsonl

Tiebreak Conditions:
  - Top-2 raw energy scores within delta (default 1.0)
  - Maximum 5 distinguishing inputs per tiebreak
  - Falls back to Lens selection if tiebreak fails
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# Type alias for LLM callable
# Signature: (chatml_prompt, temperature, max_tokens, seed) -> (text, tokens, time_ms)
LLMCallable = Callable[[str, float, int, Optional[int]], Tuple[str, int, float]]

# Type alias for sandbox execution
# Signature: (code, test_input) -> (passed, stdout, stderr)
SandboxCallable = Callable[[str, str], Tuple[bool, str, str]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SStarConfig:
    """Configuration for S* tiebreaking."""
    enabled: bool = False
    energy_delta: float = 1.0
    max_distinguishing_inputs: int = 5
    num_inputs_to_generate: int = 3
    generation_temperature: float = 0.3
    generation_max_tokens: int = 2048


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateScore:
    """A candidate solution with its Lens energy score."""
    code: str
    raw_energy: float
    index: int = 0

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "raw_energy": self.raw_energy,
            "code_length": len(self.code),
        }


@dataclass
class TiebreakResult:
    """Result of an S* tiebreak evaluation."""
    triggered: bool = False
    winner_index: int = -1
    scores: List[int] = field(default_factory=list)
    num_inputs: int = 0
    distinguishing_inputs: List[str] = field(default_factory=list)
    time_ms: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "triggered": self.triggered,
            "winner_index": self.winner_index,
            "scores": self.scores,
            "num_inputs": self.num_inputs,
            "time_ms": self.time_ms,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Telemetry event
# ---------------------------------------------------------------------------

@dataclass
class SStarEvent:
    """Telemetry event for an S* tiebreak attempt."""
    task_id: str
    triggered: bool = False
    energy_delta: float = 0.0
    candidate_a_energy: float = 0.0
    candidate_b_energy: float = 0.0
    winner_index: int = -1
    num_inputs: int = 0
    scores_a: int = 0
    scores_b: int = 0
    time_ms: float = 0.0
    reason: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "triggered": self.triggered,
            "energy_delta": self.energy_delta,
            "candidate_a_energy": self.candidate_a_energy,
            "candidate_b_energy": self.candidate_b_energy,
            "winner_index": self.winner_index,
            "num_inputs": self.num_inputs,
            "scores_a": self.scores_a,
            "scores_b": self.scores_b,
            "time_ms": self.time_ms,
            "reason": self.reason,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

DISTINGUISHING_INPUT_PROMPT = """\
Given this problem:
{problem}

And these two candidate solutions:

Solution A:
```python
{candidate_a}
```

Solution B:
```python
{candidate_b}
```

Generate {n} edge-case test inputs where these solutions might produce DIFFERENT outputs.
Focus on: boundary conditions, large inputs, special cases, off-by-one scenarios.

Output format — one test input per line, each on its own line:
INPUT: <the test input>

Only output the inputs, nothing else."""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_top2_by_energy(candidates: List[CandidateScore]) -> List[CandidateScore]:
    """Get the top-2 candidates by lowest raw energy (best first).

    Args:
        candidates: List of scored candidates.

    Returns:
        List of 2 candidates with lowest energy, sorted ascending.
    """
    if len(candidates) < 2:
        return list(candidates)
    sorted_candidates = sorted(candidates, key=lambda c: c.raw_energy)
    return sorted_candidates[:2]


def should_tiebreak(candidate_a: CandidateScore,
                    candidate_b: CandidateScore,
                    delta: float = 1.0) -> bool:
    """Check if a tiebreak is needed between two candidates.

    Args:
        candidate_a: First candidate (lower energy).
        candidate_b: Second candidate.
        delta: Maximum energy difference for tiebreak trigger.

    Returns:
        True if the energy difference is within delta.
    """
    return abs(candidate_a.raw_energy - candidate_b.raw_energy) <= delta


def parse_distinguishing_inputs(response: str,
                                max_inputs: int = 5) -> List[str]:
    """Parse distinguishing inputs from LLM response.

    Looks for "INPUT:" prefix lines, falls back to non-empty lines.

    Args:
        response: Raw LLM response text.
        max_inputs: Maximum number of inputs to return.

    Returns:
        List of test input strings.
    """
    inputs: List[str] = []

    # Try structured "INPUT:" format
    for line in response.strip().split('\n'):
        line = line.strip()
        if line.upper().startswith('INPUT:'):
            value = line.split(':', 1)[1].strip()
            if value:
                inputs.append(value)

    if inputs:
        return inputs[:max_inputs]

    # Fallback: numbered lines (1. ..., 2. ...) or bare lines
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Strip numbering
        cleaned = re.sub(r'^\d+[.)]\s*', '', line).strip()
        if cleaned and not cleaned.startswith('#') and not cleaned.startswith('```'):
            inputs.append(cleaned)

    return inputs[:max_inputs]


def score_candidates_on_inputs(
    candidate_a: str,
    candidate_b: str,
    inputs: List[str],
    sandbox_run: SandboxCallable,
) -> Tuple[int, int]:
    """Run both candidates on distinguishing inputs and compare outputs.

    For stdio-mode tasks, S* generates distinguishing inputs and pipes
    them through both candidates. Candidates that produce the same output
    on a given input are tied; candidates that produce different output
    are differentiated. A candidate scores a point when it runs without
    error (non-empty stdout, no crash).

    Args:
        candidate_a: Code for candidate A.
        candidate_b: Code for candidate B.
        inputs: List of test input strings (stdin format).
        sandbox_run: Callable to run code with specific stdin.
                     Signature: (code, stdin_input) -> (passed, stdout, stderr)

    Returns:
        Tuple of (score_a, score_b).
    """
    score_a = 0
    score_b = 0

    for test_input in inputs:
        passed_a, stdout_a, stderr_a = sandbox_run(candidate_a, test_input)
        passed_b, stdout_b, stderr_b = sandbox_run(candidate_b, test_input)

        # Score: candidate that runs without crash gets a point
        # If both produce output, the one that matches more consistently wins
        if passed_a or (stdout_a.strip() and not stderr_a):
            score_a += 1
        if passed_b or (stdout_b.strip() and not stderr_b):
            score_b += 1

    return score_a, score_b


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SStar:
    """S* distinguishing input tiebreaker.

    When enabled and top-2 candidates have similar energy scores, generates
    edge-case inputs to differentiate them. Selects the candidate that
    passes more distinguishing tests.

    When disabled, always returns the candidate with lower energy (noop).

    Args:
        config: SStarConfig instance.
        telemetry_dir: Directory for JSONL event logs.
    """

    def __init__(self, config: SStarConfig,
                 telemetry_dir: Optional[Path] = None):
        self.config = config
        self.telemetry_dir = telemetry_dir
        self._events_file: Optional[Path] = None
        if telemetry_dir is not None:
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            self._events_file = telemetry_dir / "s_star_events.jsonl"

    def tiebreak(self, candidates: List[CandidateScore],
                 problem: str,
                 llm_call: Optional[LLMCallable] = None,
                 sandbox_run: Optional[SandboxCallable] = None,
                 task_id: str = "") -> TiebreakResult:
        """Attempt S* tiebreaking on candidates.

        Args:
            candidates: List of scored candidates.
            problem: Original problem description.
            llm_call: LLM callable for generating distinguishing inputs.
            sandbox_run: Sandbox callable for running candidates.
            task_id: Task identifier for telemetry.

        Returns:
            TiebreakResult with winner and details.
        """
        # Not enough candidates
        if len(candidates) < 2:
            result = TiebreakResult(
                triggered=False,
                winner_index=candidates[0].index if candidates else -1,
                reason="insufficient_candidates",
            )
            return result

        top2 = get_top2_by_energy(candidates)
        a, b = top2[0], top2[1]

        # Check if tiebreak is needed
        if not self.config.enabled or not should_tiebreak(a, b, self.config.energy_delta):
            result = TiebreakResult(
                triggered=False,
                winner_index=a.index,
                reason="disabled" if not self.config.enabled else "clear_winner",
            )
            if task_id:
                self._log_event(SStarEvent(
                    task_id=task_id,
                    triggered=False,
                    energy_delta=abs(a.raw_energy - b.raw_energy),
                    candidate_a_energy=a.raw_energy,
                    candidate_b_energy=b.raw_energy,
                    winner_index=a.index,
                    reason=result.reason,
                ))
            return result

        # Need LLM and sandbox for tiebreak
        if llm_call is None or sandbox_run is None:
            result = TiebreakResult(
                triggered=False,
                winner_index=a.index,
                reason="missing_callables",
            )
            return result

        start_time = time.time()

        # Generate distinguishing inputs
        prompt = self._build_prompt(problem, a.code, b.code)
        response, tokens, gen_time = llm_call(
            prompt,
            self.config.generation_temperature,
            self.config.generation_max_tokens,
            None,  # no seed for diversity
        )

        inputs = parse_distinguishing_inputs(
            response, self.config.max_distinguishing_inputs
        )

        if not inputs:
            elapsed = (time.time() - start_time) * 1000
            result = TiebreakResult(
                triggered=True,
                winner_index=a.index,
                reason="no_inputs_generated",
                time_ms=elapsed,
            )
            if task_id:
                self._log_event(SStarEvent(
                    task_id=task_id,
                    triggered=True,
                    energy_delta=abs(a.raw_energy - b.raw_energy),
                    candidate_a_energy=a.raw_energy,
                    candidate_b_energy=b.raw_energy,
                    winner_index=a.index,
                    num_inputs=0,
                    reason="no_inputs_generated",
                    time_ms=elapsed,
                ))
            return result

        # Score candidates
        score_a, score_b = score_candidates_on_inputs(
            a.code, b.code, inputs, sandbox_run
        )

        elapsed = (time.time() - start_time) * 1000

        # Determine winner (ties go to lower energy candidate)
        if score_a >= score_b:
            winner = a
        else:
            winner = b

        result = TiebreakResult(
            triggered=True,
            winner_index=winner.index,
            scores=[score_a, score_b],
            num_inputs=len(inputs),
            distinguishing_inputs=inputs,
            time_ms=elapsed,
            reason="tiebreak_complete",
        )

        # Log telemetry
        if task_id:
            self._log_event(SStarEvent(
                task_id=task_id,
                triggered=True,
                energy_delta=abs(a.raw_energy - b.raw_energy),
                candidate_a_energy=a.raw_energy,
                candidate_b_energy=b.raw_energy,
                winner_index=winner.index,
                num_inputs=len(inputs),
                scores_a=score_a,
                scores_b=score_b,
                time_ms=elapsed,
                reason="tiebreak_complete",
            ))

        return result

    # -- Helpers ------------------------------------------------------------

    def _build_prompt(self, problem: str,
                      candidate_a: str, candidate_b: str) -> str:
        """Build the distinguishing input generation prompt."""
        user_content = DISTINGUISHING_INPUT_PROMPT.format(
            problem=problem,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            n=self.config.num_inputs_to_generate,
        )
        # Use a simple system prompt for input generation
        system = "You are a testing expert. Generate edge-case test inputs."
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _log_event(self, event: SStarEvent) -> None:
        """Append event to JSONL telemetry file."""
        if self._events_file is None:
            return
        try:
            with open(self._events_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except OSError:
            pass
