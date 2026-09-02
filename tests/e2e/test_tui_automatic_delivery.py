"""The interactive product path: a TUI request selects automatic V3 delivery.

    TUI-shaped request  -->  REAL atlas-proxy  -->  REAL v3-service
    (task_mode work, candidate_policy automatic_v3, NO expected outputs)

The TUI knows nothing structured about which files a task requires, so it
declares none. What it does have is the model's own structured tool call: a
write_file naming one canonical target. Under automatic_v3 that structured
mutation target is what grounds the delivery of the selected candidate to
exactly that path -- and nothing else: no obligation is invented, completion
is unchanged, and the default policy still keeps the model's own bytes.

Same fixtures as test_v3_lens_acceptance.py: scripted fake llama, real proxy,
real v3-service, fake lens, real sandbox executor.
"""
import pytest

from tests.e2e.conftest import drive_agent_turn, start_proxy
from tests.e2e.test_v3_lens_acceptance import (  # noqa: F401  (fixtures)
    CAND_A, T2_CONTENT,
    _agent_body, _assert_no_human_gate_inside_v3, _payload, _write_result,
    fake_lens, fake_llama, proxy, v3_service, workspace,
)

# Exactly what the TUI sends for an ordinary work message: the mode, the
# session's policy -- always explicit, strict included -- and nothing about
# files. The omitted-policy shape is what a legacy or API client sends; the TUI
# never sends it, because an omitted field is the proxy's cue to apply the
# operator default.
TUI_AUTOMATIC_CONTRACT = {"task_mode": "work", "candidate_policy": "automatic_v3"}
TUI_STRICT_CONTRACT = {"task_mode": "work", "candidate_policy": "strict"}
LEGACY_OMITTED_CONTRACT = {"task_mode": "work"}


def test_tui_automatic_selection_delivers_to_the_structured_target(proxy, workspace):  # noqa: F811
    events = drive_agent_turn(
        proxy, _agent_body(workspace, task_contract=TUI_AUTOMATIC_CONTRACT),
        deadline_s=180.0)
    result = _write_result(events)
    assert result["data"].get("success") is True, result["data"]
    payload = _payload(result)
    assert payload.get("v3_used") is True, payload
    written = (workspace / "todo_app.py").read_text()
    assert written == CAND_A, "the selected candidate did not land on the model's own target"
    # One target, the one the tool call named. Nothing else appeared.
    assert sorted(p.name for p in workspace.iterdir()) == ["todo_app.py"]
    _assert_no_human_gate_inside_v3(events)


def test_tui_strict_selection_keeps_the_models_bytes(proxy, workspace):  # noqa: F811
    # The TUI's default and its explicit strict are the same request.
    events = drive_agent_turn(
        proxy, _agent_body(workspace, task_contract=TUI_STRICT_CONTRACT),
        deadline_s=180.0)
    assert any(e["type"] == "v3_select" for e in events), "V3 never selected"
    result = _write_result(events)
    assert result["data"].get("success") is True
    assert _payload(result).get("v3_used") is not True
    assert (workspace / "todo_app.py").read_text() == T2_CONTENT
    _assert_no_human_gate_inside_v3(events)


def test_legacy_omitted_policy_keeps_the_models_bytes_under_operator_strict(proxy, workspace):  # noqa: F811
    events = drive_agent_turn(
        proxy, _agent_body(workspace, task_contract=LEGACY_OMITTED_CONTRACT),
        deadline_s=180.0)
    assert any(e["type"] == "v3_select" for e in events), "V3 never selected"
    result = _write_result(events)
    assert result["data"].get("success") is True
    assert _payload(result).get("v3_used") is not True
    assert (workspace / "todo_app.py").read_text() == T2_CONTENT


# --- the operator default says automatic ------------------------------------
#
# What the TUI displays must be what the proxy applies, and the case that can
# separate them is an operator who set ATLAS_CANDIDATE_POLICY=automatic_v3. A
# TUI request that sent no policy would run under that default while showing
# strict. The TUI sends strict explicitly; the proxy applies it.

@pytest.fixture()
def proxy_operator_automatic(fake_llama, fake_lens, v3_service, sandbox_executor):  # noqa: F811
    port, proc = start_proxy({
        "ATLAS_INFERENCE_URL": f"http://127.0.0.1:{fake_llama}",
        "ATLAS_LENS_URL": f"http://127.0.0.1:{fake_lens}",
        "ATLAS_SANDBOX_URL": f"http://127.0.0.1:{sandbox_executor}",
        "ATLAS_V3_URL": f"http://127.0.0.1:{v3_service}",
        "ATLAS_CANDIDATE_POLICY": "automatic_v3",
    })
    yield port
    proc.terminate()
    proc.wait(timeout=10)


def test_tui_explicit_strict_beats_an_operator_automatic_default(proxy_operator_automatic, workspace):  # noqa: F811
    events = drive_agent_turn(
        proxy_operator_automatic, _agent_body(workspace, task_contract=TUI_STRICT_CONTRACT),
        deadline_s=180.0)
    assert any(e["type"] == "v3_select" for e in events), "V3 never selected"
    result = _write_result(events)
    assert result["data"].get("success") is True
    assert _payload(result).get("v3_used") is not True, "the operator default overrode the TUI's strict"
    assert (workspace / "todo_app.py").read_text() == T2_CONTENT
    _assert_no_human_gate_inside_v3(events)


def test_legacy_omitted_policy_inherits_an_operator_automatic_default(proxy_operator_automatic, workspace):  # noqa: F811
    # Unchanged compatibility for clients that send no policy: the operator's
    # default applies. This is exactly why the TUI never sends this shape.
    events = drive_agent_turn(
        proxy_operator_automatic, _agent_body(workspace, task_contract=LEGACY_OMITTED_CONTRACT),
        deadline_s=180.0)
    result = _write_result(events)
    assert result["data"].get("success") is True
    assert _payload(result).get("v3_used") is True, _payload(result)
    assert (workspace / "todo_app.py").read_text() == CAND_A
    _assert_no_human_gate_inside_v3(events)
