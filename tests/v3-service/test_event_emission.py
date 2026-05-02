"""Tests for v3-service typed event emission (PC-061 step B).

Exercises the _emit_event helper directly via a fake wfile. Verifies:
  * legacy {stage, detail} shape always emitted (back-compat)
  * envelope NOT emitted when opt-in is False
  * envelope IS emitted when opt-in is True
  * stage_start → stage_end pairing carries parent_id + duration_ms
  * suffix-based classification routes to the right envelope type
"""

import io
import sys
from pathlib import Path

# v3-service/main.py imports from project root, so we add it the same way
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "v3-service"))

import main as v3main  # noqa: E402
from atlas.cli.events import LegacyEventError, parse_envelope, iter_sse_lines  # noqa: E402


def _read_emitted(buf: io.BytesIO):
    """Replay a fake wfile's bytes through iter_sse_lines and split into
    (legacy_dicts, envelopes). Legacy = parses with json but raises
    LegacyEventError; envelopes = parse_envelope succeeds."""
    import json
    buf.seek(0)
    legacy = []
    envs = []
    for line in iter_sse_lines(buf.readlines()):
        try:
            envs.append(parse_envelope(line))
        except LegacyEventError:
            legacy.append(json.loads(line))
    return legacy, envs


def test_emit_without_opt_in_only_writes_legacy():
    buf = io.BytesIO()
    v3main._emit_event(buf, envelope_opt_in=False,
                         stage="phase2", detail="allocating",
                         stage_start_ids={})
    legacy, envs = _read_emitted(buf)
    assert legacy == [{"stage": "phase2", "detail": "allocating"}]
    assert envs == []


def test_emit_with_opt_in_writes_both_legacy_and_envelope():
    buf = io.BytesIO()
    v3main._emit_event(buf, envelope_opt_in=True,
                         stage="phase2", detail="allocating",
                         stage_start_ids={})
    legacy, envs = _read_emitted(buf)
    assert legacy == [{"stage": "phase2", "detail": "allocating"}]
    assert len(envs) == 1
    ev = envs[0]
    assert ev.type == "stage_start"
    assert ev.stage == "phase2"
    assert ev.payload == {"detail": "allocating"}


def test_emit_pass_suffix_becomes_stage_end_with_success_true():
    buf = io.BytesIO()
    v3main._emit_event(buf, True, "probe_pass", "ok", {})
    _, envs = _read_emitted(buf)
    assert envs[0].type == "stage_end"
    assert envs[0].stage == "probe"  # suffix stripped
    assert envs[0].payload["success"] is True


def test_emit_failed_suffix_becomes_stage_end_with_success_false():
    buf = io.BytesIO()
    v3main._emit_event(buf, True, "derivation_failed", "ran out", {})
    _, envs = _read_emitted(buf)
    assert envs[0].type == "stage_end"
    assert envs[0].stage == "derivation"
    assert envs[0].payload["success"] is False


def test_emit_error_suffix_becomes_error_event():
    buf = io.BytesIO()
    v3main._emit_event(buf, True, "pr_cot_error", "exception", {})
    _, envs = _read_emitted(buf)
    assert envs[0].type == "error"
    assert envs[0].stage == "pr_cot"
    assert envs[0].payload["message"] == "exception"
    assert envs[0].payload["recoverable"] is True


def test_emit_pairs_stage_end_to_stage_start_via_parent_id():
    state: dict = {}
    buf = io.BytesIO()
    v3main._emit_event(buf, True, "phase2", "starting", state)
    v3main._emit_event(buf, True, "phase2_allocated", "done", state)  # informational mid-stage
    v3main._emit_event(buf, True, "phase2_pass", "selected", state)

    _, envs = _read_emitted(buf)
    assert [e.type for e in envs] == ["stage_start", "stage_start", "stage_end"]
    start_ev = envs[0]
    end_ev = envs[2]
    # The pairing is by logical stage name. phase2 → "phase2" in the
    # tracker; phase2_allocated → "phase2_allocated" (different key,
    # no suffix match); phase2_pass strips _pass → "phase2", pairs to
    # the original phase2 stage_start (NOT phase2_allocated, which is
    # tracked under its own distinct logical name).
    assert end_ev.parent_id == start_ev.event_id
    assert end_ev.duration_ms is not None
    assert end_ev.duration_ms >= 0
    # And phase2_allocated's entry is still in the tracker — never
    # got a matching stage_end, which is fine.
    assert "phase2_allocated" in state


def test_emit_handles_disconnect_without_raising():
    """If the client disconnects mid-stream, wfile.write raises. Helper
    must swallow it (don't propagate up into the pipeline thread)."""
    class BrokenWfile:
        def write(self, b):
            raise BrokenPipeError("client gone")
        def flush(self):
            raise BrokenPipeError("client gone")
    v3main._emit_event(BrokenWfile(), True, "phase2", "x", {})  # no raise


def test_classify_handles_all_known_suffixes():
    cases = {
        "probe":               ("stage_start", None),
        "probe_pass":          ("stage_end", True),
        "probe_failed":        ("stage_end", False),
        "probe_error":         ("error", None),
        "probe_retry":         ("stage_start", None),
        "self_test_skip":      ("stage_end", True),
        "sandbox_done":        ("stage_end", True),
        "phase2_allocated":    ("stage_start", None),  # no known suffix
    }
    for stage, expected in cases.items():
        actual = v3main._classify_stage(stage)
        assert actual == expected, f"{stage}: expected {expected}, got {actual}"


def test_logical_stage_strips_known_suffixes_only():
    assert v3main._logical_stage("probe_pass") == "probe"
    assert v3main._logical_stage("derivation_failed") == "derivation"
    assert v3main._logical_stage("pr_cot_error") == "pr_cot"
    # Unknown suffix → unchanged.
    assert v3main._logical_stage("phase2_allocated") == "phase2_allocated"
    assert v3main._logical_stage("phase2") == "phase2"
