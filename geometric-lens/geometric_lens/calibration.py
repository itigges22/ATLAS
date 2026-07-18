"""Model-specific calibration helpers for Geometric Lens scores."""

import json
import math
import os


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def derive_cx_normalization(pass_energy_mean: float,
                            fail_energy_mean: float) -> dict:
    """Derive a sigmoid calibration from one model's C(x) distribution."""
    pass_mean = float(pass_energy_mean)
    fail_mean = float(fail_energy_mean)
    if not math.isfinite(pass_mean) or not math.isfinite(fail_mean):
        raise ValueError("C(x) energy means must be finite")
    separation = fail_mean - pass_mean
    if separation <= 0.0:
        raise ValueError("C(x) calibration requires FAIL energy > PASS energy")
    return {
        "midpoint": (pass_mean + fail_mean) / 2.0,
        "steepness": 4.0 / max(separation, 0.1),
        "pass_energy_mean": pass_mean,
        "fail_energy_mean": fail_mean,
    }


def validate_cx_normalization(value) -> dict:
    """Validate and normalize a deserialized C(x) calibration object."""
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    required = ("midpoint", "steepness")
    if not all(_is_number(value.get(k)) for k in required):
        raise ValueError("expected numeric midpoint and steepness")
    calibrated = {k: float(value[k]) for k in required}
    if not all(math.isfinite(v) for v in calibrated.values()):
        raise ValueError("midpoint and steepness must be finite")
    if calibrated["steepness"] <= 0.0:
        raise ValueError("steepness must be positive")
    for key in ("pass_energy_mean", "fail_energy_mean"):
        if key in value:
            if not _is_number(value[key]):
                raise ValueError(f"{key} must be numeric")
            number = float(value[key])
            if not math.isfinite(number):
                raise ValueError(f"{key} must be finite")
            calibrated[key] = number
    if all(key in calibrated for key in ("pass_energy_mean",
                                          "fail_energy_mean")):
        if calibrated["fail_energy_mean"] <= calibrated["pass_energy_mean"]:
            raise ValueError("FAIL energy mean must be greater than PASS energy mean")
    return calibrated


def normalize_cx_energy(energy: float, calibration) -> float:
    """Normalize C(x) energy, returning a neutral score if uncalibrated."""
    if calibration is None:
        return 0.5
    calibrated = validate_cx_normalization(calibration)
    z = calibrated["steepness"] * (float(energy) - calibrated["midpoint"])
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 709.0)))
    exp_z = math.exp(max(z, -709.0))
    return exp_z / (1.0 + exp_z)


def save_cx_normalization(save_dir: str, calibration: dict) -> str:
    """Write the selected model's C(x) calibration beside its weights."""
    calibrated = validate_cx_normalization(calibration)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "cx_normalization.json")
    with open(path, "w") as fh:
        json.dump(calibrated, fh, indent=2)
        fh.write("\n")
    return path
