"""Shadow must be observational with respect to DECISIONS, not just selection.

The defect: phase zero called may_return_early_result on probe evidence, so
in shadow a behaviourally complete browser candidate returned early and
skipped candidate generation. That is a live control-flow change.

The probe-free judgement still suppresses syntax-only early return in every
mode including off — that determination comes from the adapter, needs no
probe, and is the defect this whole line of work exists to fix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import evidence as E  # noqa: E402

COMPLETE = {"supported": True, "runtime_clean": True, "temporal_progress": True,
            "input_causality": True, "collision_transition": True,
            "food_or_score_transition": True}


def _decisions(mode, probe_ev):
    """Mirror of the pipeline's decision, without a live pipeline run."""
    probe_free = E.result_from_adapter(E.BROWSER_CANVAS_JS, True, None)
    with_probe = E.result_from_adapter(E.BROWSER_CANVAS_JS, True, probe_ev)
    legacy = E.may_return_early_result(probe_free)
    ev = E.may_return_early_result(with_probe)
    return legacy, ev, (ev if E.selection_enabled(mode) else legacy)


def test_shadow_records_but_does_not_act_on_a_complete_candidate():
    legacy, ev, taken = _decisions(E.SHADOW, COMPLETE)
    assert ev is True, "evidence policy would return early"
    assert taken is False, "shadow must not let it change control flow"
    assert legacy is False


def test_enforce_lets_a_complete_candidate_return_early():
    _legacy, ev, taken = _decisions(E.ENFORCE, COMPLETE)
    assert ev is True and taken is True


def test_off_runs_no_browser_probe():
    assert not E.probing_enabled(E.OFF)


def test_syntax_only_is_suppressed_in_every_mode_including_off():
    """The core fix must not depend on the mode."""
    for mode in (E.OFF, E.SHADOW, E.ENFORCE):
        probe_free = E.result_from_adapter(E.BROWSER_CANVAS_JS, True, None)
        assert not E.may_return_early_result(probe_free), mode


def test_algorithmic_fast_path_survives_in_every_mode():
    """Do not disable the class where ATLAS already performs well."""
    res = E.result_from_adapter(E.ALGORITHMIC_IO, True)
    assert E.may_return_early_result(res)
    for mode in (E.OFF, E.SHADOW, E.ENFORCE):
        taken = E.may_return_early_result(res)
        assert taken is True, mode


def test_env_none_differs_from_an_empty_env():
    import os
    os.environ["ATLAS_EVIDENCE_MODE"] = "enforce"
    try:
        assert E.selection_mode() == E.ENFORCE       # reads the process env
        assert E.selection_mode({}) == E.OFF         # explicitly empty
    finally:
        del os.environ["ATLAS_EVIDENCE_MODE"]


def test_probe_timeout_fits_inside_the_client_read_timeout():
    import pipeline as P
    assert P.BROWSER_PROBE_TIMEOUT_S < 45, \
        "an execution budget above the client read timeout is cut off by its caller"
