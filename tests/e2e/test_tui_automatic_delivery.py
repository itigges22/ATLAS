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
from tests.e2e.conftest import drive_agent_turn
from tests.e2e.test_v3_lens_acceptance import (  # noqa: F401  (fixtures)
    CAND_A, T2_CONTENT,
    _agent_body, _assert_no_human_gate_inside_v3, _payload, _write_result,
    fake_lens, fake_llama, proxy, v3_service, workspace,
)

# Exactly what the TUI sends for an ordinary work message once the user has
# selected automatic delivery: the mode, the policy, and nothing about files.
TUI_AUTOMATIC_CONTRACT = {"task_mode": "work", "candidate_policy": "automatic_v3"}
TUI_DEFAULT_CONTRACT = {"task_mode": "work"}


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


def test_tui_default_selection_keeps_the_models_bytes(proxy, workspace):  # noqa: F811
    events = drive_agent_turn(
        proxy, _agent_body(workspace, task_contract=TUI_DEFAULT_CONTRACT),
        deadline_s=180.0)
    assert any(e["type"] == "v3_select" for e in events), "V3 never selected"
    result = _write_result(events)
    assert result["data"].get("success") is True
    assert _payload(result).get("v3_used") is not True
    assert (workspace / "todo_app.py").read_text() == T2_CONTENT
    _assert_no_human_gate_inside_v3(events)
