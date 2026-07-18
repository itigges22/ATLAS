"""Unit tests for per-model G(x) threshold derivation (_derive_gx_thresholds).

The thresholds must adapt to the selected model's score scale rather than
borrowing fixed operating points from another model.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geometric_lens.thresholds import (
    derive_gx_thresholds as _derive_gx_thresholds,
    validate_gx_thresholds,
)


def test_mid_scale_scores_yield_matching_cutoffs():
    # Derived thresholds must sit near this model's observed PASS scale.
    rng = np.random.default_rng(0)
    pass_scores = np.clip(rng.normal(0.46, 0.05, 500), 0, 1)
    t = _derive_gx_thresholds(pass_scores)
    assert t["low"] > 0.25, f"low too low for a 0.46-centered model: {t}"
    assert t["severe"] < t["low"], f"severe must be strictest: {t}"
    assert t["severe"] <= t["off_rails"] <= t["low"], f"ordering wrong: {t}"
    # All within the clamp band.
    assert all(0.02 <= t[k] <= 0.6 for k in t), t


def test_high_scale_scores_preserve_ordering():
    # PASS writes clustered high with a broad tail.
    rng = np.random.default_rng(1)
    pass_scores = np.clip(rng.normal(0.85, 0.18, 500), 0, 1)
    t = _derive_gx_thresholds(pass_scores)
    assert t["severe"] <= t["off_rails"] <= t["low"], t
    assert all(0.02 <= t[k] <= 0.6 for k in t), t


def test_small_sample_refuses_to_fabricate_calibration():
    with pytest.raises(ValueError, match="at least 20 PASS"):
        _derive_gx_thresholds(np.array([0.4, 0.5, 0.6]))


def test_none_refuses_to_fabricate_calibration():
    with pytest.raises(ValueError, match="at least 20 PASS"):
        _derive_gx_thresholds(None)


@pytest.mark.parametrize("value", [
    {},
    {"severe": True, "off_rails": 0.2, "low": 0.3},
    {"severe": 0.4, "off_rails": 0.3, "low": 0.2},
    {"severe": np.nan, "off_rails": 0.2, "low": 0.3},
])
def test_invalid_threshold_objects_are_rejected(value):
    with pytest.raises(ValueError):
        validate_gx_thresholds(value)


if __name__ == "__main__":
    test_mid_scale_scores_yield_matching_cutoffs()
    test_high_scale_scores_preserve_ordering()
    test_small_sample_refuses_to_fabricate_calibration()
    test_none_refuses_to_fabricate_calibration()
    print("all gx threshold tests passed")
