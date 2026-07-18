"""Feature-flag contract for the structural call-graph layer (issue #39).

The graph package is always importable and side-effect free. Whether the
pipeline uses it (deeper structural veto, multi-hop repair context, graph-scoped
context injection) is gated by `ATLAS_CALL_GRAPH`, checked before any of those
paths diverge from current behavior. Default off.
"""

from __future__ import annotations

import os

ENV_VAR = "ATLAS_CALL_GRAPH"

_TRUTHY = {"1", "true", "yes", "on"}


def call_graph_enabled() -> bool:
    """True when the call-graph layer is enabled. Default: off."""
    return os.getenv(ENV_VAR, "0").strip().lower() in _TRUTHY
