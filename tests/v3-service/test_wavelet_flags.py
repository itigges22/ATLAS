"""Tests for the ATLAS_RPG_PLANNING feature-flag contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

from wavelet.flags import rpg_planning_enabled, ENV_VAR  # noqa: E402


def test_default_off(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert rpg_planning_enabled() is False


def test_truthy_values(monkeypatch):
    for v in ("1", "true", "TRUE", "yes", "on", " On "):
        monkeypatch.setenv(ENV_VAR, v)
        assert rpg_planning_enabled() is True


def test_falsy_values(monkeypatch):
    for v in ("0", "false", "no", "off", ""):
        monkeypatch.setenv(ENV_VAR, v)
        assert rpg_planning_enabled() is False
