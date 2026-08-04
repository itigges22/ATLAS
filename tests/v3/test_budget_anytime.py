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


def test_reserve_covers_one_more_generation(monkeypatch):
    """A flat reserve is useless when the unit of work is an LLM call.

    Measured 2026-08-03: the phase-3 entry check passed with ~20s left,
    PR-CoT then spent 31s on a single call, and the cap landed part-way
    through the next one. The reserve has to be sized to what starting more
    work actually costs.
    """
    import types

    monkeypatch.setenv("ATLAS_V3_TIMEOUT", "180")

    # Rebuild the closure's inputs the way _run_impl does.
    def make_guard(avg_call_ms, elapsed_s):
        llm = types.SimpleNamespace(avg_call_ms=avg_call_ms)
        start = time.time() - elapsed_s

        def out_of_budget(reserve_ms=None):
            left = pipeline._remaining_budget_ms(start)
            if left is None:
                return False
            if reserve_ms is None:
                observed = getattr(llm, "avg_call_ms", 0.0) or 0.0
                reserve_ms = max(20000.0, observed * 1.2 + 10000.0)
            return left < reserve_ms

        return out_of_budget

    # 30s calls, 140s spent: 40s left does not cover another 30s call + margin.
    assert make_guard(30_000, 140)() is True
    # Same clock, cheap calls: 40s left is plenty for a 5s call.
    assert make_guard(5_000, 140)() is False
    # Early in the run nothing is short.
    assert make_guard(30_000, 10)() is False


def test_a_disabled_cap_never_trips_the_guard(monkeypatch):
    import types
    monkeypatch.setenv("ATLAS_V3_TIMEOUT", "0")
    llm = types.SimpleNamespace(avg_call_ms=60_000)
    start = time.time() - 10_000

    left = pipeline._remaining_budget_ms(start)
    assert left is None, "an unbounded run must report no budget"


# --- the adapter refuses work it cannot finish -----------------------------

def test_adapter_refuses_a_call_that_cannot_finish(monkeypatch):
    """Boundary checks cannot hold: PR-CoT issues two calls, so a guard that
    reserves one is already wrong by the second. Measured 2026-08-03 — a
    boundary check with ~50s left correctly allowed PR-CoT against a ~34s
    reserve, and PR-CoT spent 44s then started a 21s call past the cap."""
    import adapters

    a = adapters.LLMAdapter(deadline=time.time() + 15)
    # 30s average call, 15s left: starting another one overruns.
    a.call_count, a.total_time_ms = 2, 60_000
    with pytest.raises(adapters.BudgetExhausted):
        a._check_budget()

    # Same observed cost, generous clock: allowed.
    a.deadline = time.time() + 300
    a._check_budget()


def test_adapter_without_a_deadline_is_unbounded():
    """The bench and any caller with the cap disabled must not be truncated."""
    import adapters
    a = adapters.LLMAdapter(deadline=None)
    a.call_count, a.total_time_ms = 5, 500_000
    a._check_budget()


def test_first_call_is_never_refused():
    """Before any call there is nothing observed, so a run must be allowed to
    start rather than returning empty on a short budget."""
    import adapters
    a = adapters.LLMAdapter(deadline=time.time() + 1)
    a._check_budget()
