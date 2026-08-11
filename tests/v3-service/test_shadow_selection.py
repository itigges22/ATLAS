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

import evidence as E  # noqa: E402


def _cand(index, score, missing, strength, energy, runtime_clean=True):
    return {"index": index, "code": f"// {index}", "energy": energy,
            "behavior_score": score, "missing_required": missing,
            "evidence_strength": strength,
            "behavior": {"runtime_clean": runtime_clean}}


def test_behaviour_beats_a_prettier_lens_score():
    behavioural = _cand(1, 0.75, [], E.BEHAVIORAL_PARTIAL, energy=9.0)
    pretty = _cand(2, 0.25, ["input_causality"], E.RUNTIME, energy=0.01)
    assert max([behavioural, pretty], key=E.rank_key) is behavioural


def test_complete_beats_partial():
    partial = _cand(1, 0.75, [], E.BEHAVIORAL_PARTIAL, energy=0.1)
    complete = _cand(2, 1.0, [], E.BEHAVIORAL_COMPLETE, energy=5.0)
    assert max([partial, complete], key=E.rank_key) is complete


def test_candidate_zero_is_preserved_and_can_win():
    """ATLAS must be able to decline to replace a better baseline."""
    baseline = _cand(0, 1.0, [], E.BEHAVIORAL_COMPLETE, energy=3.0)
    alt = _cand(1, 0.5, ["input_causality"], E.RUNTIME, energy=0.001)
    assert max([baseline, alt], key=E.rank_key) is baseline


def test_unsupported_candidate_is_a_fallback_not_a_verified_winner():
    unsupported = E.result_from_adapter(E.BROWSER_CANVAS_JS, True, probe_evidence=None)
    assert unsupported["accepted"] is True      # usable as a fallback
    assert unsupported["supported"] is False    # but never "verified"
    assert not E.may_return_early_result(unsupported)


def test_runtime_health_breaks_ties_before_the_lens():
    clean = _cand(1, 0.5, [], E.RUNTIME, energy=9.0, runtime_clean=True)
    dirty = _cand(2, 0.5, [], E.RUNTIME, energy=0.01, runtime_clean=False)
    assert max([clean, dirty], key=E.rank_key) is clean


def test_evidence_selection_is_off_by_default():
    assert os.environ.get("ATLAS_EVIDENCE_SELECTION", "0") == "0", \
        "evidence ranking must ship in shadow mode until uplift is shown"
