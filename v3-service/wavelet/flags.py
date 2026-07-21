"""Feature-flag contract for the V3.2 RPG-style architecture-first planner.

The wavelet substrate (this package) is always importable and side-effect free.
Whether the *planner* uses it is gated by `ATLAS_RPG_PLANNING`, which Phase 1
checks before swapping the flat planner for the RPG two-stage flow. Defining the
env contract here (rather than inline at the call site) keeps the default — off —
in one place. See docs/reports/RPG_WAVELET_PLANNING_V3_2.md.
"""

from __future__ import annotations

import os

ENV_VAR = "ATLAS_RPG_PLANNING"

_TRUTHY = {"1", "true", "yes", "on"}


def rpg_planning_enabled() -> bool:
    """True when RPG-style architecture-first planning is enabled. Default: off."""
    return os.getenv(ENV_VAR, "0").strip().lower() in _TRUTHY
