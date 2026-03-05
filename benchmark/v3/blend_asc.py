"""V3 Blend-ASC (Feature 2A) — Adaptive Sampling Budget via Lens Energy.

Allocates candidates (k) proportional to problem difficulty. Easy problems
get fewer candidates with lighter budgets; hard problems get more candidates
with deeper thinking. A single probe candidate is generated first, scored
with the Geometric Lens, and the resulting energy determines k.

Paper: Feng & Odonnat (arxiv:2511.12309). 6.8x fewer samples at equal accuracy.
Config: [blend_asc] in atlas.conf
Telemetry: telemetry/blend_asc_events.jsonl

Adaptive K Table (normalized energy -> k range + budget tier):
  [0.00, 0.10) → k=1-2, nothink   (easy, minimal compute)
  [0.10, 0.20) → k=3-4, standard  (medium, moderate compute)
  [0.20, 0.30) → k=5-7, hard      (difficult, extended thinking)
  [0.30, 1.00] → k=8-12, extreme  (very hard, maximum effort)
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
class BlendASCConfig:
    """Configuration for Blend-ASC adaptive sampling."""
    enabled: bool = False
    default_k: int = 3
    energy_midpoint: float = 9.5
    energy_steepness: float = 0.5


# ---------------------------------------------------------------------------
# Adaptive K table
# ---------------------------------------------------------------------------

@dataclass
class KAllocation:
    """K allocation for a given energy range."""
    energy_low: float
    energy_high: float
    min_k: int
    max_k: int
    budget_tier: str

    def contains(self, energy: float) -> bool:
        """Check if normalized energy falls in this range."""
        return self.energy_low <= energy < self.energy_high

    def to_dict(self) -> Dict:
        return {
            "energy_range": [self.energy_low, self.energy_high],
            "min_k": self.min_k,
            "max_k": self.max_k,
            "budget_tier": self.budget_tier,
        }


# Default adaptive K table from PRD
DEFAULT_K_TABLE: List[KAllocation] = [
    KAllocation(0.00, 0.10, 1, 2, "nothink"),
    KAllocation(0.10, 0.20, 3, 4, "standard"),
    KAllocation(0.20, 0.30, 5, 7, "hard"),
    KAllocation(0.30, 1.01, 8, 12, "extreme"),  # 1.01 to include 1.0
]


# ---------------------------------------------------------------------------
# Telemetry event
# ---------------------------------------------------------------------------

@dataclass
class BlendASCEvent:
    """Telemetry event for a Blend-ASC allocation decision."""
    task_id: str
    raw_energy: float = 0.0
    normalized_energy: float = 0.0
    allocated_k: int = 0
    budget_tier: str = ""
    probe_tokens: int = 0
    probe_time_ms: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "raw_energy": self.raw_energy,
            "normalized_energy": self.normalized_energy,
            "allocated_k": self.allocated_k,
            "budget_tier": self.budget_tier,
            "probe_tokens": self.probe_tokens,
            "probe_time_ms": self.probe_time_ms,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def lookup_k(normalized_energy: float,
             k_table: Optional[List[KAllocation]] = None) -> KAllocation:
    """Look up K allocation for a given normalized energy.

    Args:
        normalized_energy: Energy in [0, 1] from normalize_energy().
        k_table: Custom K table (uses DEFAULT_K_TABLE if None).

    Returns:
        KAllocation matching the energy range.
    """
    table = k_table or DEFAULT_K_TABLE
    for alloc in table:
        if alloc.contains(normalized_energy):
            return alloc
    # Fallback to last entry (extreme) for out-of-range values
    return table[-1] if table else KAllocation(0.0, 1.01, 3, 3, "standard")


def compute_k(normalized_energy: float,
              k_table: Optional[List[KAllocation]] = None) -> Tuple[int, str]:
    """Compute the k value and budget tier for a normalized energy.

    Uses min_k by default (conservative). Callers can escalate to max_k
    based on additional signals.

    Args:
        normalized_energy: Energy in [0, 1].
        k_table: Optional custom K table.

    Returns:
        Tuple of (k, budget_tier).
    """
    alloc = lookup_k(normalized_energy, k_table)
    return alloc.min_k, alloc.budget_tier


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BlendASC:
    """Blend-ASC adaptive sampling budget controller.

    When enabled, generates a probe candidate first, scores it with the
    Geometric Lens, and uses the energy to determine how many candidates (k)
    to generate and at what budget tier.

    When disabled, returns the default k and standard budget tier (noop).

    Args:
        config: BlendASCConfig instance.
        k_table: Custom K allocation table (uses default if None).
        telemetry_dir: Directory for JSONL event logs.
    """

    def __init__(self, config: BlendASCConfig,
                 k_table: Optional[List[KAllocation]] = None,
                 telemetry_dir: Optional[Path] = None):
        self.config = config
        self._k_table = k_table or list(DEFAULT_K_TABLE)
        self.telemetry_dir = telemetry_dir
        self._events_file: Optional[Path] = None
        if telemetry_dir is not None:
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            self._events_file = telemetry_dir / "blend_asc_events.jsonl"

    @property
    def k_table(self) -> List[KAllocation]:
        """The current K allocation table."""
        return self._k_table

    def allocate(self, raw_energy: float,
                 task_id: str = "",
                 probe_tokens: int = 0,
                 probe_time_ms: float = 0.0) -> Tuple[int, str]:
        """Determine k and budget tier based on raw Lens energy.

        Args:
            raw_energy: Raw C(x) energy from Geometric Lens.
            task_id: Task identifier for telemetry.
            probe_tokens: Tokens used for the probe candidate.
            probe_time_ms: Time in ms for the probe generation.

        Returns:
            Tuple of (k, budget_tier).
        """
        if not self.config.enabled:
            return self.config.default_k, "standard"

        normalized = normalize_energy(
            raw_energy,
            midpoint=self.config.energy_midpoint,
            steepness=self.config.energy_steepness,
        )
        k, tier = compute_k(normalized, self._k_table)

        # Log telemetry
        if task_id:
            self._log_event(BlendASCEvent(
                task_id=task_id,
                raw_energy=raw_energy,
                normalized_energy=normalized,
                allocated_k=k,
                budget_tier=tier,
                probe_tokens=probe_tokens,
                probe_time_ms=probe_time_ms,
            ))

        return k, tier

    def get_allocation_for_energy(self, normalized_energy: float) -> KAllocation:
        """Get the full KAllocation for a normalized energy value.

        Useful for inspecting min_k/max_k range without triggering telemetry.
        """
        return lookup_k(normalized_energy, self._k_table)

    def get_table_summary(self) -> List[Dict]:
        """Return the K table as a list of dicts for inspection."""
        return [a.to_dict() for a in self._k_table]

    # -- Private helpers ----------------------------------------------------

    def _log_event(self, event: BlendASCEvent) -> None:
        """Append event to JSONL telemetry file."""
        if self._events_file is None:
            return
        try:
            with open(self._events_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except OSError:
            pass
