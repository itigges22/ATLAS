"""Lens candidate scoring for the benchmark runner (v3_runner imports
score_candidate for generated candidates and score_candidate_combined for
the probe the CxGx allocation gate reads)."""

import json
import urllib.request
import urllib.error
from typing import Dict, Tuple


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


NEUTRAL_COMBINED = {
    "cx_energy": 0.0, "cx_normalized": 0.5, "cx_calibrated": False,
    "gx_score": 0.5, "gx_available": False, "verdict": "unavailable",
}


def score_candidate_combined(text: str, lens_url: str) -> Dict:
    """Score candidate text through the Lens's combined C(x)+G(x) endpoint.

    A single embedding extraction feeds both models: C(x) cost-field energy
    (``cx_energy`` raw, ``cx_normalized`` in [0,1], lower = better) and the
    G(x) XGBoost quality classifier (``gx_score`` = P(correct) in [0,1],
    higher = better), so the pair costs no more than C(x) alone. The CxGx
    allocation gate reads both off the probe.

    Fail-soft: any transport error, a disabled lens, or a malformed body
    yields the neutral dict with ``cx_calibrated``/``gx_available`` false,
    which the gate reads as "no signal" and answers with its k=3 floor.
    """
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{lens_url}/internal/lens/gx-score",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data.get("enabled", False):
            return dict(NEUTRAL_COMBINED)
        return {
            "cx_energy": data.get("cx_energy", 0.0),
            "cx_normalized": data.get("cx_normalized", 0.5),
            "cx_calibrated": bool(data.get("cx_calibrated", False)),
            "gx_score": data.get("gx_score", 0.5),
            "gx_available": bool(data.get("gx_available", False)),
            "verdict": data.get("verdict", "unavailable"),
        }
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return dict(NEUTRAL_COMBINED)
