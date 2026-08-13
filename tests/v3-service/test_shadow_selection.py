"""Candidate zero and generated candidates share ONE evaluation path.

The defect this closes: probe-level evidence was structured, but generated
candidates still went through `verified_sandbox` -> a bare `passed`, entered
the pool on that boolean, and were selected by lens energy. ATLAS could
generate alternatives it had no way to rank by demonstrated behaviour.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import adapters as A  # noqa: E402
import contract as C  # noqa: E402
import pipeline as P  # noqa: E402


def _record(probe, code, accepted=True):
    return A.contract_record(
        adapter=A.ADAPTER_BROWSER_CANVAS_JS, accepted=accepted, probe=probe,
        contract_id="selection", contract_version="1", artifact_scope="a.js",
        evaluation_context_hash=C.content_hash("ctx"),
        candidate_content_hash=C.content_hash(code))


def _trace(**flags):
    ev = {"supported": True, "runtime_clean": True, "temporal_progress": False,
          "input_causality": False, "collision_transition": False,
          "food_or_score_transition": False}
    ev.update(flags)
    return ev


COMPLETE = _trace(temporal_progress=True, input_causality=True,
                  collision_transition=True, food_or_score_transition=True)
PARTIAL = _trace(temporal_progress=True, input_causality=True,
                 collision_transition=True)
MISSING_REQUIRED = _trace(temporal_progress=True)


def test_behaviour_beats_a_prettier_lens_score():
    """Selection ranks demonstrated coverage, and knows nothing about energy."""
    behavioural = _record(PARTIAL, "a")
    pretty = _record(MISSING_REQUIRED, "b")
    assert C.select([pretty, behavioural], behavioural)["best_record"] is behavioural


def test_complete_beats_partial():
    partial = _record(PARTIAL, "a")
    complete = _record(COMPLETE, "b")
    picked = C.select([partial, complete], complete)
    assert picked["best_record"] is complete
    assert picked["verified_winner"] is complete
    assert C.selection_status(picked) == C.SELECTION_VERIFIED_WINNER


def test_candidate_zero_is_preserved_and_can_win():
    """ATLAS must be able to decline to replace a better baseline."""
    baseline = _record(COMPLETE, "zero")
    alt = _record(MISSING_REQUIRED, "one")
    assert C.select([baseline, alt], baseline)["best_record"] is baseline


def test_unsupported_candidate_is_a_fallback_not_a_verified_winner():
    unsupported = _record(None, "a")             # probe could not run
    assert unsupported["execution_status"] == C.EXEC_SKIPPED  # usable, unverified
    assert unsupported["supported"] is False     # but never "verified"
    assert unsupported["closure_eligible"] is False
    picked = C.select([unsupported], unsupported)
    assert picked["verified_winner"] is None
    assert C.selection_status(picked) == C.SELECTION_INELIGIBLE


def test_runtime_health_decides_before_any_lens_score():
    clean = _record(PARTIAL, "a")
    dirty = _record(_trace(temporal_progress=True, input_causality=True,
                           collision_transition=True, runtime_clean=False), "b")
    assert C.select([dirty, clean], clean)["best_record"] is clean


def test_the_mode_defaults_to_off():
    """The old test asserted ATLAS_EVIDENCE_SELECTION, a flag the code no
    longer reads — it could stay green while testing nothing."""
    assert P._selection_mode({}) == P.MODE_OFF
    assert not P._selection_enabled(P._selection_mode({}))
    assert not P._probing_enabled(P._selection_mode({}))
