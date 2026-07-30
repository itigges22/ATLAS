"""Lens candidate scoring for the benchmark runner (v3_runner imports
score_candidate)."""

import json
import urllib.request
import urllib.error
from typing import Tuple


def score_candidate(text: str, lens_url: str) -> Tuple[float, float]:
    """Score candidate text through the Geometric Lens.

    Args:
        text: Full text to score (typically "TASK: {prompt}\\n\\nSOLUTION: {response}").
        lens_url: Base URL for geometric-lens (e.g. "http://localhost:31144").

    Returns:
        Tuple of (raw_energy, normalized_energy). Returns the neutral
        sentinel (0.0, 0.5) on ANY failure — transport errors included —
        matching the product path (v3-service treats unscored candidates
        as neutral, and v3_runner's sentinel check recognizes exactly
        this pair). A distinct transport-failure value here would feed
        a fake "real" energy into candidate sorting.
    """
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{lens_url}/internal/lens/score-text",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("energy", 0.0), data.get("normalized", 0.5))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return (0.0, 0.5)
