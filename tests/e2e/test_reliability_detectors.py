"""Unit tests for the harness-defect detectors in scripts/e2e-reliability.py.

The detectors decide whether a live session's failure was ATLAS's fault or the
model's, so a detector that silently stops firing turns the reliability number
into a rubber stamp. Each test below is a synthetic event stream reproducing a
defect that was actually observed on 2026-07-31, plus the corresponding clean
stream, so a detector cannot pass by flagging everything.

No live stack: streams are literals and the workspace is a tmp_path.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rel():
    spec = importlib.util.spec_from_file_location(
        "atlas_reliability", REPO / "scripts" / "e2e-reliability.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["atlas_reliability"] = mod
    spec.loader.exec_module(mod)
    return mod


def _session(rel, events, workspace, stream_ok=True):
    return rel.Session(task="t", rep=1, events=events, workspace=workspace,
                       wall_s=1.0, stream_ok=stream_ok)


def _call(name, **args):
    return {"type": "tool_call", "data": {"name": name, "args": args}}


def _ok(tool="read_file"):
    # Real tool_result events carry the tool name (data keys: data, elapsed,
    # error, success, tool). The detectors read it from the RESULT rather than
    # pairing positionally with the calls, because one unanswered call — a
    # client timeout mid-stream — shifted every pair after it.
    return {"type": "tool_result", "data": {"tool": tool, "success": True, "error": ""}}


def _fail(error, tool="read_file"):
    return {"type": "tool_result", "data": {"tool": tool, "success": False, "error": error}}


# --- H1 protocol ----------------------------------------------------------

def test_h1_flags_orphaned_tool_call(rel, tmp_path):
    s = _session(rel, [_call("read_file", path="a.py"), _ok(),
                       _call("edit_file", path="a.py"),
                       {"type": "done", "data": {"summary": "x"}}], tmp_path)
    assert any("orphaned call" in d for d in rel.h1_protocol(s, {"tool_call", "tool_result", "done"}))


def test_h1_flags_event_the_tui_cannot_render(rel, tmp_path):
    s = _session(rel, [{"type": "brand_new_event", "data": {}},
                       {"type": "done", "data": {"summary": "x"}}], tmp_path)
    found = rel.h1_protocol(s, {"done"})
    assert any("cannot render" in d and "brand_new_event" in d for d in found)


def test_h1_does_not_charge_the_runner_cap_to_the_proxy(rel, tmp_path):
    """This runner stops reading at --timeout and appends its own error
    event. The proxy never got to send `done` and the socket closed
    mid-stream, so counting those as two protocol violations scores our
    deadline as its defect."""
    cap = {"type": "error",
           "data": {"error": "harness cap: session exceeded 900s"}}
    s = _session(rel, [_call("read_file", path="a.py"), _ok(), cap],
                 tmp_path, stream_ok=False)
    found = rel.h1_protocol(s, {"tool_call", "tool_result", "done", "error"})
    assert not any("protocol" in d for d in found)
    assert any("timeout" in d and "cap" in d for d in found)


def test_h1_capped_session_tolerates_only_the_in_flight_call(rel, tmp_path):
    cap = {"type": "error",
           "data": {"error": "harness cap: session exceeded 900s"}}
    known = {"tool_call", "tool_result", "done", "error"}

    one = _session(rel, [_call("read_file", path="a.py"), _ok(),
                         _call("edit_file", path="a.py"), cap],
                   tmp_path, stream_ok=False)
    assert not any("orphaned" in d for d in rel.h1_protocol(one, known))

    two = _session(rel, [_call("read_file", path="a.py"),
                         _call("edit_file", path="a.py"), cap],
                   tmp_path, stream_ok=False)
    assert any("orphaned" in d for d in rel.h1_protocol(two, known))


def test_h1_clean_stream_is_clean(rel, tmp_path):
    s = _session(rel, [_call("read_file", path="a.py"), _ok(),
                       {"type": "done", "data": {"summary": "x"}}], tmp_path)
    assert rel.h1_protocol(s, {"tool_call", "tool_result", "done"}) == []


# --- H2 false rejection (the D9 class) ------------------------------------

def test_h2_flags_rejection_blaming_a_file_that_is_fine(rel, tmp_path):
    """V3 authored the break, the gate blocked it, the message named the file.

    The model then hunts a defect that is not on disk. Decided by re-checking
    the file, which is exactly what a human does to catch this.
    """
    (tmp_path / "app.py").write_text("def index():\n    return 'ok'\n")
    s = _session(rel, [_call("edit_file", path="app.py"),
                       _fail("/workspace/app.py has a JavaScript syntax error in "
                             "the <script> block — it was NOT written.\nline 121: "
                             "unexpected `)`"),
                       {"type": "done", "data": {"summary": "x"}}], tmp_path)
    assert any("blamed for a syntax error it does not have"
               in d for d in rel.h2_false_rejection(s))


def test_h2_silent_when_the_file_really_is_broken(rel, tmp_path):
    (tmp_path / "app.py").write_text("def index(:\n")  # genuinely unparseable
    s = _session(rel, [_call("edit_file", path="app.py"),
                       _fail("/workspace/app.py has a Python syntax error — it "
                             "was NOT written."),
                       {"type": "done", "data": {"summary": "x"}}], tmp_path)
    assert rel.h2_false_rejection(s) == []


# --- H3 dead-end steering (the D10 class) ---------------------------------

def test_h3_flags_advice_the_next_call_is_refused_for_taking(rel, tmp_path):
    """Advice offered `<body>`; the model reached for `<script>`.

    Different tag, same unsupported shape — matching the literal string would
    miss the case this exists to catch.
    """
    s = _session(rel, [
        _call("edit_file", path="app.py"),
        _fail("string to replace not found in file. To replace a whole element "
              "use structural_edit with a selector (e.g. `function:NAME`, "
              "`class:NAME`, `<body>`) and the new content."),
        _call("structural_edit", path="app.py", selector="<script>"),
        _fail("unknown selector '<script>' for python. Supported: "
              "function:NAME, class:NAME"),
        {"type": "done", "data": {"summary": "x"}}], tmp_path)
    found = rel.h3_dead_end_steering(s)
    assert any("<tag>" in d for d in found), found


def test_h3_silent_when_advice_matched_the_file(rel, tmp_path):
    s = _session(rel, [
        _call("edit_file", path="app.py"),
        _fail("string to replace not found. Use structural_edit with a "
              "selector (`function:NAME` or `class:NAME`)."),
        _call("structural_edit", path="app.py", selector="function:index"),
        _fail("structural_edit: your replacement is IDENTICAL to the code "
              "already in the file"),
        {"type": "done", "data": {"summary": "x"}}], tmp_path)
    assert rel.h3_dead_end_steering(s) == []


# --- H4 gate escape (the D11 class) ---------------------------------------

def test_h4_flags_exit_with_no_write_on_an_action_prompt(rel, tmp_path):
    s = _session(rel, [_call("read_file", path="app.py"), _ok(),
                       {"type": "text", "data": {"content":
                        "The pause logic is already present in the template."}},
                       {"type": "done", "data": {"summary": ""}}], tmp_path)
    assert rel.h4_gate_escape(s) != []


def test_h4_silent_when_the_breaker_ended_honestly(rel, tmp_path):
    """An honest 'I stopped' is the machinery working, not an escape."""
    s = _session(rel, [_call("read_file", path="app.py"), _ok(),
                       {"type": "done", "data": {"summary":
                        "Stopped after 3 tool failures on the same target "
                        "with no successful changes."}}], tmp_path)
    assert rel.h4_gate_escape(s) == []


def test_h4_silent_when_a_write_landed(rel, tmp_path):
    s = _session(rel, [_call("edit_file", path="app.py"), _ok("edit_file"),
                       {"type": "done", "data": {"summary": "Added the toggle."}}],
                 tmp_path)
    assert rel.h4_gate_escape(s) == []


# --- H5 corrupt write -----------------------------------------------------

def test_h5_flags_unparseable_file_left_on_disk(rel, tmp_path):
    (tmp_path / "broken.py").write_text("def f(:\n")
    task = rel.Task(name="t", prompt="", files={}, check=lambda p: (True, ""))
    s = _session(rel, [], tmp_path)
    assert any("broken.py" in d for d in rel.h5_corrupt_write(s, task))


def test_h5_flags_a_required_file_that_was_deleted(rel, tmp_path):
    task = rel.Task(name="t", prompt="", files={}, check=lambda p: (True, ""),
                    must_exist=("gone.py",))
    s = _session(rel, [], tmp_path)
    assert any("gone.py was deleted" in d for d in rel.h5_corrupt_write(s, task))


def test_h5_clean_workspace_is_clean(rel, tmp_path):
    (tmp_path / "fine.py").write_text("x = 1\n")
    task = rel.Task(name="t", prompt="", files={}, check=lambda p: (True, ""),
                    must_exist=("fine.py",))
    s = _session(rel, [], tmp_path)
    assert rel.h5_corrupt_write(s, task) == []


# --- TUI coverage ---------------------------------------------------------

def test_tui_handled_types_is_populated_and_has_the_core_events(rel):
    """Located by content marker, so a file move must not empty the set."""
    handled = rel.tui_handled_types()
    assert len(handled) > 30, f"only found {len(handled)} — dispatcher not located?"
    for core in ("tool_call", "tool_result", "done", "text", "error"):
        assert core in handled, f"TUI dispatcher has no case for {core!r}"


# --- H7 silent background leak -------------------------------------------

def test_h7_is_silent_when_the_session_announced_the_jobs(rel, tmp_path, monkeypatch):
    """Persistence is deliberate; the defect is persistence nobody was told about.

    An agent loop is one user message, so killing jobs at its end would break
    "start the dev server" then "now curl it". H7 must therefore fire on
    silence, not on the jobs existing.
    """
    monkeypatch.setattr(rel.subprocess, "run",
                        lambda *a, **k: type("P", (), {"stdout": "2", "returncode": 0})())
    announced = _session(rel, [{"type": "done", "data": {"summary":
        "Done.\n\nStill running in the sandbox:\n  abc — python app.py\n"
        "These keep their ports until stopped. Use stop_background to end them."}}],
        tmp_path)
    assert rel.h7_background_leak("atlas-sandbox-1", announced) == []

    silent = _session(rel, [{"type": "done", "data": {"summary": "Added the toggle."}}],
                      tmp_path)
    found = rel.h7_background_leak("atlas-sandbox-1", silent)
    assert found and "silent background leak" in found[0]


# --- H8 anchored on ATLAS-injected text ----------------------------------

def test_h8_flags_old_str_copied_from_the_call_graph_footer(rel, tmp_path):
    """read_file appends a footer that is not on disk.

    A measured session anchored edit_file on "## Call graph (within this
    file)\\n- mean calls: ..." and spent all three of its failures on an edit
    that could never match. Scoring that against the model would make the
    harness understate the defects it exists to find.
    """
    s = _session(rel, [
        _call("edit_file", path="stats.py",
              old_str="\n\n\n## Call graph (within this file)\n"
                      "- mean calls: ValueError, sum, len"),
        _fail("string to replace not found in file."),
        {"type": "done", "data": {"summary": "Stopped after 3 tool failures."}},
    ], tmp_path)
    found = rel.h8_anchored_on_injected_text(s)
    assert found and "Call graph" in found[0]


def test_h8_silent_on_a_normal_old_str(rel, tmp_path):
    s = _session(rel, [
        _call("edit_file", path="stats.py", old_str="def mean(values):"),
        _fail("string to replace not found in file."),
        {"type": "done", "data": {"summary": "x"}},
    ], tmp_path)
    assert rel.h8_anchored_on_injected_text(s) == []


def test_h2_silent_on_a_message_that_blames_the_models_content(rel, tmp_path):
    """"Your content for X has a syntax error" is the CORRECT message.

    It blames the submission, and the file on disk is clean precisely because
    the write was refused. An earlier version of this detector matched it and
    reported a harness defect on a session where ATLAS behaved perfectly —
    a false-positive detector is worse than a missing one, because it sends
    someone chasing a bug that is not there.
    """
    (tmp_path / "store.py").write_text("x = 1\n")
    s = _session(rel, [
        _call("write_file", path="store.py"),
        _fail("Your content for store.py has a syntax error (SyntaxError: "
              "unmatched ')') — it was NOT written. The content is NOT "
              "truncated; it is complete but INVALID."),
        {"type": "done", "data": {"summary": "x"}},
    ], tmp_path)
    assert rel.h2_false_rejection(s) == []


# --- H9 tier appropriateness ---------------------------------------------

def test_h9_flags_v3_running_on_a_question(rel, tmp_path):
    """The tiers exist so the heavy pipeline does not run on everything.

    A wrong answer is a model limit. Spending a multi-minute V3 pipeline on
    "what does this function do" is a product defect that costs real minutes.
    """
    task = rel.Task(name="ask", prompt="what does f do?", files={},
                    check=lambda p, s=None: (True, ""), conversational=True)
    s = _session(rel, [
        {"type": "v3_probe", "data": {"stage": "probe"}},
        {"type": "text", "data": {"content": "It counts duplicates."}},
        {"type": "done", "data": {"summary": ""}},
    ], tmp_path)
    found = rel.h9_tier_misapplied(s, task)
    assert found and "V3 pipeline ran on a question" in found[0]


def test_h9_flags_a_question_that_edited_files(rel, tmp_path):
    task = rel.Task(name="ask", prompt="what does f do?", files={},
                    check=lambda p, s=None: (True, ""), conversational=True)
    s = _session(rel, [
        _call("edit_file", path="orders.py"), _ok("edit_file"),
        {"type": "done", "data": {"summary": "done"}},
    ], tmp_path)
    found = rel.h9_tier_misapplied(s, task)
    assert found and "caused file writes" in found[0]


def test_h9_silent_on_a_clean_conversational_turn(rel, tmp_path):
    task = rel.Task(name="ask", prompt="what does f do?", files={},
                    check=lambda p, s=None: (True, ""), conversational=True)
    s = _session(rel, [
        _call("read_file", path="orders.py"), _ok(),
        {"type": "text", "data": {"content": "It is quadratic."}},
        {"type": "done", "data": {"summary": ""}},
    ], tmp_path)
    assert rel.h9_tier_misapplied(s, task) == []


def test_h9_does_not_constrain_a_work_task(rel, tmp_path):
    """V3 and writes are exactly what a coding task should produce."""
    task = rel.Task(name="fix", prompt="fix the bug", files={},
                    check=lambda p: (True, ""))
    s = _session(rel, [
        {"type": "v3_probe", "data": {}},
        _call("edit_file", path="a.py"), _ok(),
        {"type": "done", "data": {"summary": "fixed"}},
    ], tmp_path)
    assert rel.h9_tier_misapplied(s, task) == []


def test_h4_silent_on_a_conversational_task(rel, tmp_path):
    """A question SHOULD exit without writing.

    Scoring that as a gate escape reported a harness defect on both
    conversational probes for behaving exactly as asked — the inverse of the
    H9 tier check.
    """
    task = rel.Task(name="ask", prompt="what does f do?", files={},
                    check=lambda p, s=None: (True, ""), conversational=True)
    s = _session(rel, [
        _call("read_file", path="orders.py"), _ok(),
        {"type": "text", "data": {"content": "It is quadratic."}},
        {"type": "done", "data": {"summary": ""}},
    ], tmp_path)
    assert rel.h4_gate_escape(s, task) == []


def test_h4_still_fires_on_a_work_task(rel, tmp_path):
    task = rel.Task(name="fix", prompt="fix the bug", files={},
                    check=lambda p: (True, ""))
    s = _session(rel, [_call("read_file", path="a.py"), _ok(),
                       {"type": "done", "data": {"summary": ""}}], tmp_path)
    assert rel.h4_gate_escape(s, task) != []


# --- bug-find check must not accept an invented mechanism ----------------

def test_bugfind_rejects_the_right_file_with_a_wrong_mechanism(rel, tmp_path):
    """The real cycle-6 answer, which an earlier version of the check passed.

    It named planning.py and the symptom correctly, then attributed the cause
    to "how min() is used with a custom key" — there is no min() there, and
    the function it named was the scorer, not the selection loop. Right file,
    invented mechanism, and a loose check called it a pass.
    """
    task = rel.TASKS["bugfind_tiebreak"]
    s = _session(rel, [{"type": "text", "data": {"content":
        "The issue is in `planning.py` within the `_score_plan` function. When two "
        "plans have the same score, the code selects the one with the maximum number "
        "of steps because of how the `min()` function is being used with a custom key."}},
        {"type": "done", "data": {"summary": ""}}], tmp_path)
    passed, _ = task.check(tmp_path, s)
    assert not passed


def test_bugfind_accepts_the_actual_comparison(rel, tmp_path):
    task = rel.TASKS["bugfind_tiebreak"]
    s = _session(rel, [{"type": "text", "data": {"content":
        "planning.py: the selection loop breaks ties with n_steps > best_steps, "
        "so a tie keeps the longer plan. It should be <."}},
        {"type": "done", "data": {"summary": ""}}], tmp_path)
    passed, _ = task.check(tmp_path, s)
    assert passed


# --- H6 service fault ------------------------------------------------------

def test_h6_does_not_charge_the_runner_cap_to_the_proxy(rel, tmp_path):
    """The cap event is this runner's own, appended when it stops reading at
    --timeout. h1_protocol already reports it as the timeout it is; counting
    it again here charged one deadline as two separate proxy defects."""
    cap = {"type": "error",
           "data": {"error": "harness cap: session exceeded 900s"}}
    s = _session(rel, [_call("read_file", path="a.py"), _ok(), cap],
                 tmp_path, stream_ok=False)
    assert rel.h6_service_fault(s) == []


def test_h6_still_reports_a_real_service_fault(rel, tmp_path):
    boom = {"type": "error", "data": {"error": "v3 service: connection refused"}}
    s = _session(rel, [boom], tmp_path, stream_ok=False)
    found = rel.h6_service_fault(s)
    assert any("connection refused" in d for d in found)


def test_h6_ignores_a_parse_failure_the_session_recovered_from(rel, tmp_path):
    """flask_pause rep2, 2026-08-03: the model emitted a 20 KB tool call that
    ran out of tokens mid-JSON, the proxy classified it and told the model,
    and the session went on to pass the task — scored a harness defect for
    it. Recovered model behaviour is the proxy working."""
    err = {"type": "error", "data": {"category": "truncated_tool",
                                     "error": "failed to parse model response"}}
    s = _session(rel, [_call("edit_file", path="app.py"), err,
                       _call("replace_lines", path="app.py"), _ok("replace_lines"),
                       {"type": "done", "data": {"summary": "added the toggle"}}],
                 tmp_path)
    assert rel.h6_service_fault(s) == []


def test_h6_still_reports_a_parse_failure_the_session_died_on(rel, tmp_path):
    err = {"type": "error", "data": {"error": "failed to parse model response"}}
    s = _session(rel, [_call("edit_file", path="app.py"), err], tmp_path,
                 stream_ok=False)
    assert any("parse model response" in d for d in rel.h6_service_fault(s))


# --- the V3-disabled control arm -------------------------------------------
#
# The independent confirmation needs incumbents ATLAS produces with the V3
# capability absent and NOTHING else changed. `bypass_v3` is the proxy's
# existing per-request field for exactly that: it short-circuits the V3
# orchestration while ctx.V3URL stays set, so structural_check,
# embedded_script_check, symbol_index and orphaned_symbols — the mutation
# gates — keep running. Clearing ATLAS_V3_URL instead would silently
# disable those gates too, which is a different experiment.


def _capture_body(rel, monkeypatch, **kwargs):
    """Run one session against a stubbed transport and return the request."""
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return iter([b'data: {"type":"done"}\n\n'])

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(rel.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_bypass_v3_reaches_the_agent_request(rel, monkeypatch, tmp_path):
    import json as _json
    globals()["json"] = _json
    seen = _capture_body(rel, monkeypatch)
    task = rel.Task(name="t", prompt="do it", files={}, check=lambda ws, s=None: (True, ""))
    rel.run_session(task, 0, "http://proxy", tmp_path, "e2e", 30, bypass_v3=True)
    assert seen["body"]["bypass_v3"] is True
    assert seen["body"]["message"] == "do it"


def test_the_field_is_absent_by_default(rel, monkeypatch, tmp_path):
    import json as _json
    globals()["json"] = _json
    seen = _capture_body(rel, monkeypatch)
    task = rel.Task(name="t", prompt="do it", files={}, check=lambda ws, s=None: (True, ""))
    rel.run_session(task, 0, "http://proxy", tmp_path, "e2e", 30)
    assert "bypass_v3" not in seen["body"], \
        "every existing caller's request body must be unchanged"


def test_the_proxy_declares_the_field():
    """Only honest if the server accepts it under this exact name."""
    agent = (REPO / "proxy" / "agent.go").read_text()
    assert 'BypassV3         bool   `json:"bypass_v3,omitempty"`' in agent


# Every symbol through which a request's V3 capability mode can be read. A
# mutation gate that mentions any of them has stopped being mode-independent.
_V3_MODE_SYMBOLS = ("BypassV3", "V3Mode", "effectiveV3Mode", "V3Bypassed",
                    "V3GenerationEnabled", "V3PlanningEnabled")

# The gate implementations, and the v3-service route each one must still reach.
_MUTATION_GATES = {
    "checkStructuralUnresolved": "/internal/structural_check",
    "embeddedScriptOutcome": "/internal/embedded_script_check",
}

# The bridge call that dispatches candidate generation. Everything upstream of
# it is what "V3 generation" means at the proxy.
_GENERATION_BRIDGE = "callV3GenerateStreaming("


def _go_funcs(src):
    """{name: body} for every top-level func in a gofmt-formatted Go file.

    gofmt puts `func` in column 0 and closes a top-level declaration with a
    lone `}` in column 0, so this needs no brace matching and cannot be misled
    by a brace inside a string literal or a comment.
    """
    funcs, lines = {}, src.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("func "):
            continue
        sig = line[len("func "):]
        if sig.startswith("("):                     # method receiver
            sig = sig[sig.index(")") + 1:].lstrip()
        name = re.match(r"[A-Za-z_][A-Za-z0-9_]*", sig)
        if not name:
            continue
        for j in range(i + 1, len(lines)):
            if lines[j] == "}":
                funcs[name.group(0)] = "\n".join(lines[i:j + 1])
                break
    return funcs


def _proxy_funcs():
    """{(file, func): body} across the proxy's non-test sources."""
    out = {}
    for path in sorted((REPO / "proxy").glob("*.go")):
        if path.name.endswith("_test.go"):
            continue
        for name, body in _go_funcs(path.read_text()).items():
            out[(path.name, name)] = body
    return out


def test_disabling_v3_does_not_disable_the_mutation_gates():
    """Turning V3 candidate generation off must leave every mutation gate on.

    Pinned to the property, not to a spelling. The previous version counted
    literal `!ctx.BypassV3` expressions; fb45b74 moved the generation call
    sites to the typed predicate `ctx.V3GenerationEnabled()` and the count
    went to zero, so the test failed while the property it named still held.
    A count would have gone vacuous just as quietly had it been relaxed.

    Three facts, each read out of the source rather than assumed:

      1. the gate implementations name no V3-mode symbol at all -- they gate
         on ctx.V3URL, so they run identically in every mode;
      2. every function that reaches the generation bridge is called only from
         a function that consults ctx.V3GenerationEnabled() first;
      3. the demo-baseline relaxation is off-mode only, so planner_only turns
         generation off and keeps the whole guarded write path.
    """
    sources = {p.name: p.read_text()
               for p in (REPO / "proxy").glob("*.go")
               if not p.name.endswith("_test.go")}

    # 1. the gates are mode-independent
    gates = _go_funcs(sources["gates.go"])
    for fn, route in _MUTATION_GATES.items():
        body = gates.get(fn)
        assert body, f"{fn} is gone; repoint this test at its replacement"
        assert route in body, f"{fn} no longer reaches {route}"
        leaked = [s for s in _V3_MODE_SYMBOLS if s in body]
        assert not leaked, (
            f"{fn} consults {leaked}; a mutation gate must depend on "
            "ctx.V3URL alone so that disabling V3 cannot disable it")

    # 2. every generation dispatch is guarded, before the call, by the typed
    #    predicate -- and by nothing else
    funcs = _proxy_funcs()
    dispatchers = {name for (_f, name), body in funcs.items()
                   if _GENERATION_BRIDGE in body.split("\n", 1)[1]}
    dispatchers.discard(_GENERATION_BRIDGE.rstrip("("))
    assert dispatchers, (
        "no function dispatches V3 candidate generation; repoint this test "
        f"at whatever replaced {_GENERATION_BRIDGE}")
    for dispatcher in sorted(dispatchers):
        callers = {(f, n): b for (f, n), b in funcs.items()
                   if n != dispatcher and dispatcher + "(" in b.split("\n", 1)[1]}
        assert callers, f"{dispatcher} is unreachable from any tool"
        for (fname, caller), body in sorted(callers.items()):
            guard = body.find("ctx.V3GenerationEnabled()")
            call = body.find(dispatcher + "(", body.find("\n"))
            assert guard != -1, (
                f"{fname}:{caller} reaches {dispatcher} without consulting "
                "ctx.V3GenerationEnabled()")
            assert guard < call, (
                f"{fname}:{caller} calls {dispatcher} before it consults "
                "ctx.V3GenerationEnabled()")
            assert "ctx.BypassV3" not in body, (
                f"{fname}:{caller} reads the legacy boolean directly; the "
                "typed predicate is the only authority at a generation site")

    # 3. planner_only disables generation and keeps the gates
    types = _go_funcs(sources["types.go"])
    assert "V3ModeOff" in types["V3Bypassed"], (
        "V3Bypassed must name the off mode it relaxes for")
    assert "V3ModePlannerOnly" not in types["V3Bypassed"], (
        "the demo-baseline relaxation must be off-mode only; planner_only "
        "executes through the ordinary guarded write path")
    assert "V3ModeFull" in types["V3GenerationEnabled"], (
        "only full mode may dispatch candidate generation")
