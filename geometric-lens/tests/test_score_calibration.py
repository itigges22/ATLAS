import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometric_lens.calibration import (
    derive_cx_normalization,
    normalize_cx_energy,
    save_cx_normalization,
    validate_cx_normalization,
)


def test_derive_cx_normalization_centers_selected_model_distribution():
    calibration = derive_cx_normalization(10.0, 30.0)
    assert calibration["midpoint"] == 20.0
    assert calibration["steepness"] == pytest.approx(0.2)
    assert normalize_cx_energy(20.0, calibration) == pytest.approx(0.5)
    assert normalize_cx_energy(10.0, calibration) < 0.5
    assert normalize_cx_energy(30.0, calibration) > 0.5


def test_missing_calibration_is_neutral_not_another_models_default():
    assert normalize_cx_energy(-1000.0, None) == 0.5
    assert normalize_cx_energy(1000.0, None) == 0.5


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"midpoint": 1.0, "steepness": 0.0},
        {"midpoint": math.inf, "steepness": 1.0},
        {"midpoint": "one", "steepness": 1.0},
        {"midpoint": 1.0, "steepness": True},
        {"midpoint": 1.0, "steepness": 1.0,
         "pass_energy_mean": 2.0, "fail_energy_mean": 1.0},
    ],
)
def test_invalid_cx_calibration_is_rejected(value):
    with pytest.raises(ValueError):
        validate_cx_normalization(value)


def test_derive_rejects_nonseparating_energy_distributions():
    with pytest.raises(ValueError):
        derive_cx_normalization(10.0, 10.0)
    with pytest.raises(ValueError):
        derive_cx_normalization(20.0, 10.0)


def test_save_round_trips_calibration(tmp_path):
    expected = derive_cx_normalization(3.0, 13.0)
    path = save_cx_normalization(str(tmp_path), expected)
    with open(path, encoding="utf-8") as calibration_file:
        saved = json.load(calibration_file)
    assert saved == expected
