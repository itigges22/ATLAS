"""V3 hands back its best candidate when its clock runs out.

V3 is an anytime algorithm: the refinement loop is budget-gated, so it expands
to fill ATLAS_V3_TIMEOUT rather than converging early. Measured 2026-08-03 on
one task — a 180s budget produced 9 LLM calls and 5 sandbox verifications, a
420s budget 21 calls and 10. The cap is its terminal condition, not an
interruption.

It returned nothing at that terminal condition. The caller's deadline
cancelled the request, every verified candidate went with it, and the write
fell back to the model's own output: 16 of 41 write calls in one 28-session
run, ~48 of the 56 minutes spent in them.
"""

import os
import time

import pytest

import pipeline


@pytest.fixture(autouse=True)
def _clear_budget_env(monkeypatch):
    monkeypatch.delenv("ATLAS_V3_TIMEOUT", raising=False)


def test_remaining_budget_tracks_the_configured_cap(monkeypatch):
    monkeypatch.setenv("ATLAS_V3_TIMEOUT", "180")
    start = time.time()
    left = pipeline._remaining_budget_ms(start)
    assert 179_000 < left <= 180_000

    # A spent clock goes negative rather than clamping, so a guard reading
    # "< reserve" fires rather than seeing a small positive budget.
    assert pipeline._remaining_budget_ms(time.time() - 300) < 0


def test_a_disabled_cap_reports_no_budget(monkeypatch):
    """0 disables the cap for offline bench runs — the guards must then
    never fire, or they would truncate a run that was asked to be unbounded."""
    monkeypatch.setenv("ATLAS_V3_TIMEOUT", "0")
    assert pipeline._remaining_budget_ms(time.time()) is None


def test_the_cap_default_matches_the_proxy(monkeypatch):
    """The service and the proxy read the same knob. If these drift, the
    service plans against a budget the caller does not honour."""
    monkeypatch.delenv("ATLAS_V3_TIMEOUT", raising=False)
    left = pipeline._remaining_budget_ms(time.time())
    assert 179_000 < left <= 180_000

    monkeypatch.setenv("ATLAS_V3_TIMEOUT", "not-a-number")
    left = pipeline._remaining_budget_ms(time.time())
    assert 179_000 < left <= 180_000


def test_new_stages_are_registered_for_the_summary():
    """A stage missing from _STAGE_PHASE contributes no phase row, so the
    run summary silently loses the reason it stopped."""
    assert pipeline._STAGE_PHASE["budget_exhausted"] == "fallback"
    assert pipeline._STAGE_PHASE["divsampling_stop"] == "generation"
