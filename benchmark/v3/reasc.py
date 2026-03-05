"""V3 ReASC (Feature 2B) — Early Stopping via Token Confidence.

Computes "Bottom 10% Group Confidence" from token logprobs. If the first
candidate has high confidence AND the Geometric Lens rates it as easy
(normalized energy < 0.10), the candidate is accepted without generating
more — saving compute on problems the model already handles well.

Paper: arxiv:2601.02970, Jan 2026.
Config: [reasc] in atlas.conf
Telemetry: telemetry/reasc_events.jsonl

Early Stop Conditions (ALL must be true):
  1. Normalized Lens energy < 0.10  (easy problem)
  2. Bottom-10% token confidence > threshold  (model is confident)
  3. Feature is enabled
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .budget_forcing import normalize_energy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReASCConfig:
    """Configuration for ReASC early stopping."""
    enabled: bool = False
    confidence_threshold: float = -0.5
    energy_threshold: float = 0.10
    energy_midpoint: float = 9.5
    energy_steepness: float = 0.5


# ---------------------------------------------------------------------------
# Telemetry event
# ---------------------------------------------------------------------------

@dataclass
class ReASCEvent:
    """Telemetry event for a ReASC early stopping decision."""
    task_id: str
    raw_energy: float = 0.0
    normalized_energy: float = 0.0
    bottom_10_confidence: float = 0.0
    early_stopped: bool = False
    reason: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "raw_energy": self.raw_energy,
            "normalized_energy": self.normalized_energy,
            "bottom_10_confidence": self.bottom_10_confidence,
            "early_stopped": self.early_stopped,
            "reason": self.reason,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_bottom_10_confidence(logprobs: List[float]) -> float:
    """Compute Bottom 10% Group Confidence from token logprobs.

    Takes the bottom 10% of token logprobs (the least confident tokens)
    and averages them. This captures the model's weakest predictions —
    a high average even among weak tokens indicates strong overall confidence.

    Args:
        logprobs: List of per-token log probabilities (negative values).

    Returns:
        Average of the bottom 10% of logprobs. More negative = less confident.
        Returns 0.0 for empty input.
    """
    if not logprobs:
        return 0.0

    sorted_probs = sorted(logprobs)
    n_bottom = max(1, len(sorted_probs) // 10)
    bottom_10_pct = sorted_probs[:n_bottom]
    return sum(bottom_10_pct) / len(bottom_10_pct)


def should_early_stop(normalized_energy: float,
                      confidence: float,
                      energy_threshold: float = 0.10,
                      confidence_threshold: float = -0.5) -> Tuple[bool, str]:
    """Determine if early stopping should be applied.

    Both conditions must be met:
    1. Normalized energy is below threshold (easy problem)
    2. Confidence exceeds threshold (model is sure)

    Args:
        normalized_energy: Lens energy in [0, 1].
        confidence: Bottom-10% confidence score.
        energy_threshold: Max energy for early stop.
        confidence_threshold: Min confidence for early stop.

    Returns:
        Tuple of (should_stop, reason).
    """
    if normalized_energy >= energy_threshold:
        return False, f"energy {normalized_energy:.3f} >= threshold {energy_threshold}"

    if confidence <= confidence_threshold:
        return False, f"confidence {confidence:.3f} <= threshold {confidence_threshold}"

    return True, "easy_and_confident"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ReASC:
    """ReASC early stopping controller.

    When enabled, evaluates whether a probe candidate is good enough to
    accept without generating additional candidates. Uses Lens energy
    (difficulty) and token confidence (certainty) as dual gates.

    When disabled, never triggers early stopping (noop).

    Args:
        config: ReASCConfig instance.
        telemetry_dir: Directory for JSONL event logs.
    """

    def __init__(self, config: ReASCConfig,
                 telemetry_dir: Optional[Path] = None):
        self.config = config
        self.telemetry_dir = telemetry_dir
        self._events_file: Optional[Path] = None
        if telemetry_dir is not None:
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            self._events_file = telemetry_dir / "reasc_events.jsonl"

    def evaluate(self, raw_energy: float,
                 logprobs: List[float],
                 task_id: str = "") -> Tuple[bool, str]:
        """Evaluate whether to early-stop on this candidate.

        Args:
            raw_energy: Raw C(x) energy from Geometric Lens.
            logprobs: Per-token log probabilities from the candidate.
            task_id: Task identifier for telemetry.

        Returns:
            Tuple of (should_stop, reason).
        """
        if not self.config.enabled:
            return False, "disabled"

        normalized = normalize_energy(
            raw_energy,
            midpoint=self.config.energy_midpoint,
            steepness=self.config.energy_steepness,
        )
        confidence = compute_bottom_10_confidence(logprobs)

        stop, reason = should_early_stop(
            normalized,
            confidence,
            energy_threshold=self.config.energy_threshold,
            confidence_threshold=self.config.confidence_threshold,
        )

        # Log telemetry
        if task_id:
            self._log_event(ReASCEvent(
                task_id=task_id,
                raw_energy=raw_energy,
                normalized_energy=normalized,
                bottom_10_confidence=confidence,
                early_stopped=stop,
                reason=reason,
            ))

        return stop, reason

    def compute_confidence(self, logprobs: List[float]) -> float:
        """Compute confidence score from logprobs without triggering evaluation.

        Useful for debugging/inspection.
        """
        return compute_bottom_10_confidence(logprobs)

    # -- Private helpers ----------------------------------------------------

    def _log_event(self, event: ReASCEvent) -> None:
        """Append event to JSONL telemetry file."""
        if self._events_file is None:
            return
        try:
            with open(self._events_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except OSError:
            pass
