"""Model-specific G(x) operating-threshold calibration."""

import math


def validate_gx_thresholds(value) -> dict:
    """Validate and normalize a deserialized G(x) threshold object."""
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    required = ("off_rails", "low", "severe")
    if not all(isinstance(value.get(key), (int, float))
               and not isinstance(value.get(key), bool)
               for key in required):
        raise ValueError("expected numeric off_rails, low, and severe")
    calibrated = {key: float(value[key]) for key in required}
    if not all(math.isfinite(number) for number in calibrated.values()):
        raise ValueError("thresholds must be finite")
    if not all(0.0 < calibrated[key] <= 1.0 for key in required):
        raise ValueError("thresholds must be in (0, 1]")
    if not (calibrated["severe"] <= calibrated["off_rails"]
            <= calibrated["low"]):
        raise ValueError("expected severe <= off_rails <= low")
    return calibrated


def derive_gx_thresholds(pass_scores) -> dict:
    """Derive {off_rails, low, severe} from one model's PASS scores.

    At least 20 positive samples are required. Returning another model's
    historical defaults would make interventions silently model-dependent.
    """
    import numpy as np

    if pass_scores is None or len(pass_scores) < 20:
        raise ValueError(
            "at least 20 PASS scores are required to calibrate G(x) thresholds"
        )

    def clamp(value):
        return float(min(0.6, max(0.02, value)))

    severe = clamp(float(np.percentile(pass_scores, 5)))
    off_rails = clamp(float(np.percentile(pass_scores, 10)))
    low = clamp(float(np.percentile(pass_scores, 20)))
    off_rails = max(off_rails, severe)
    low = max(low, off_rails)
    return validate_gx_thresholds({
        "off_rails": round(off_rails, 3),
        "low": round(low, 3),
        "severe": round(severe, 3),
    })
