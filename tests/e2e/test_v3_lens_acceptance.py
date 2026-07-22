"""Deterministic end-to-end acceptance test — the V3/Lens pipeline path.

This is the path that differentiates ATLAS from a plain agent wrapper,
booted for real:

    fake llama (scripted SSE) <-- REAL atlas-proxy --> REAL v3-service
                                        |                    |
                                  fake lens  <---------------+
                                        |                    |
                                  REAL sandbox executor <----+

A write_file that classifies Tier-2 routes through the real proxy→V3
bridge into the real v3-service pipeline: probe (fails on purpose),
Lens-calibrated k allocation, PlanSearch candidate generation via the
fake llama, per-candidate Lens scoring (C(x) via /internal/lens/gx-score
and per-step G(x) via /internal/lens/score-per-step), real sandbox
/execute verification, winner selection by lowest Lens energy, and the
winning candidate written to disk by the proxy. Assertions pin: V3 was
NOT bypassed (v3_used), multiple candidates were tested, Lens scoring
was actually called (the fake lens records requests), winner selection
picked the candidate the Lens preferred, and no stage was silently
skipped (ordered v3_* event subsequence).

Failure modes covered here at the seam level: V3 unreachable, V3
malformed response, V3 timeout (all → documented direct-write
fallback), and Lens unreachable on the optional path (pipeline
completes uncalibrated). Deeper in-pipeline failure cases (no valid
candidate → repair phases, winner-selection edge cases, malformed lens
payloads and missing thresholds) are pinned by the hermetic unit
suites tests/v3-service/test_winner_selection.py and
tests/v3-service/test_lens_calibration.py.
"""

import http.server
import json
import os
import shutil
import subprocess
import threading
import time
import uuid

import pytest

from tests.e2e import conftest
from tests.e2e.conftest import (
    PROXY_BINARY, REPO, drive_agent_turn, free_port, ordered_subsequence,
    sandbox_deps_available, start_proxy, proxy_binary_available,
    SKIP_NO_PROXY_BINARY, wait_for_port,
)

pytestmark = [
    pytest.mark.skipif(
        not proxy_binary_available(), reason=SKIP_NO_PROXY_BINARY),
    pytest.mark.skipif(
        not sandbox_deps_available(),
        reason="sandbox runtime deps missing — "
               "pip install -r sandbox/requirements-runtime.txt"),
]

# The model's write_file content: 12 lines of .py with >=2 logic
# indicators — guaranteed Tier-2, so the proxy routes it through V3.
T2_CONTENT = (
    "def add(items, x):\n"
    "    if x:\n"
    "        items.append(x)\n"
    "    return items\n"
    "\n"
    "def remove(items, x):\n"
    "    if x in items:\n"
    "        items.remove(x)\n"
    "    return items\n"
    "\n"
    "def count(items):\n"
    "    return len(items)\n"
)

# Two valid candidate programs the fake llama serves for the two
# PlanSearch implementation prompts. Both pass the real sandbox's
# /execute; the fake lens gives cand-A the lower energy, so winner
# selection MUST produce cand-A. The 3.0 raw-energy gap (> the 1.0
# S* energy_delta) keeps the S* tiebreak deterministically inactive.
CAND_A = T2_CONTENT + "\n# cand-A\n"
CAND_B = T2_CONTENT + "\n# cand-B\n"

BROKEN_PROBE = "def add(items, x)\n    return items\n"  # SyntaxError


# ---------------------------------------------------------------------------
# Fake llama — serves BOTH the agent loop and v3-service's LLMAdapter
# ---------------------------------------------------------------------------

def _llm_reply(req: dict) -> str:
    messages = req.get("messages", [])
    all_text = "\n".join(m.get("content", "") for m in messages)

    # Agent-loop calls carry response_format/grammar (the JSON envelope
    # constraint); v3-service's LLMAdapter sends neither.
    if "response_format" in req or "grammar" in req:
        tool_results = sum(
            1 for m in messages
            if m.get("role") == "user"
            and m.get("content", "").startswith("[tool result]"))
        if tool_results == 0:
            return json.dumps({"type": "tool_call", "name": "write_file",
                               "args": {"path": "todo_app.py",
                                        "content": T2_CONTENT}})
        return json.dumps({"type": "done",
                           "summary": "Created todo_app.py; the V3 pipeline "
                                      "verified the write."})

    # --- v3-service pipeline prompts, keyed by distinctive substrings
    # (PlanSearch steps run in thread pools, so order is not reliable).
    if "CONSTRAINTS" in all_text and "Identify" in all_text:
        return ("CONSTRAINT SET 1:\n- keep functions pure alpha\n\n"
                "CONSTRAINT SET 2:\n- validate inputs beta\n")
    if "Design a solution plan" in all_text:
        return ("PLAN-ALPHA: implement add/remove/count directly"
                if "alpha" in all_text else
                "PLAN-BETA: implement with input validation")
    if "Implement this plan as Python code:" in all_text:
        code = CAND_A if "PLAN-ALPHA" in all_text else CAND_B
        return f"```python\n{code}```"
    if "test cases" in all_text:
        return "no structured cases"  # unparseable -> self-test loop skipped
    # Probe (the raw problem statement) and its retries: broken code so
    # the pipeline must proceed to candidate generation.
    return f"```python\n{BROKEN_PROBE}```"


class _FakeLlamaHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/health"):
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length))
        if self.path.startswith("/v1/embeddings") or self.path.startswith("/embedding"):
            self.send_response(404)  # EmbedAdapter fail-softs to []
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        reply = _llm_reply(req)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        delta = json.dumps({"choices": [{"delta": {"content": reply}}]})
        usage = json.dumps({"choices": [],
                            "usage": {"total_tokens": 30,
                                      "prompt_tokens": 20,
                                      "completion_tokens": 10}})
        for line in (delta, usage, "[DONE]"):
            self.wfile.write(f"data: {line}\n\n".encode())
        self.wfile.flush()


# ---------------------------------------------------------------------------
# Fake lens — deterministic scores; records every request for assertions
# ---------------------------------------------------------------------------

class _FakeLensHandler(http.server.BaseHTTPRequestHandler):
    calls = []  # (path, text) tuples, appended per request

    def log_message(self, *args):
        pass

    def _reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply({"status": "ok"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(length))
        except ValueError:
            req = {}
        text = req.get("text", "")
        type(self).calls.append((self.path, text))

        if self.path == "/internal/lens/gx-score":
            # cand-A must win: lowest raw energy. Probe scores
            # calibrated 0.15 normalized -> BlendASC allocates k=3.
            if "# cand-A" in text:
                energy, norm = 2.0, 0.20
            elif "# cand-B" in text:
                energy, norm = 5.0, 0.50
            else:
                energy, norm = 9.0, 0.15
            self._reply({"cx_energy": energy, "cx_normalized": norm,
                         "cx_calibrated": True,
                         "gx_score": 0.9, "gx_available": True})
        elif self.path == "/internal/lens/score-per-step":
            # Healthy per-step aggregate: gx_score_min well above the
            # severe threshold, so the lens veto must NOT fire.
            self._reply({
                "enabled": True, "gx_available": True, "n_tokens": 12,
                "latency_ms": 1,
                "aggregate": {"first_off_rails_idx": -1,
                              "gx_score_min": 0.9, "gx_score_mean": 0.92,
                              "cx_norm_max": 0.3, "cx_norm_mean": 0.2},
                "thresholds": {"off_rails": 0.34, "low": 0.34,
                               "severe": 0.28},
            })
        else:  # /internal/patterns/write and anything else
            self._reply({"status": "ok"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fake_llama():
    port = free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port),
                                             _FakeLlamaHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield port
    server.shutdown()


@pytest.fixture(scope="module")
def fake_lens():
    port = free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port),
                                             _FakeLensHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield port
    server.shutdown()


@pytest.fixture(scope="module")
def v3_service(fake_llama, fake_lens, sandbox_executor):
    port = free_port()
    env = {**os.environ,
           "ATLAS_V3_PORT": str(port),
           "ATLAS_INFERENCE_URL": f"http://127.0.0.1:{fake_llama}",
           "ATLAS_LENS_URL": f"http://127.0.0.1:{fake_lens}",
           "ATLAS_SANDBOX_URL": f"http://127.0.0.1:{sandbox_executor}",
           # the sandbox executor enforces internal auth session-wide
           "ATLAS_SERVICE_TOKEN_FILE": conftest._TOKEN_FILE}
    env.pop("ATLAS_CALL_GRAPH", None)
    proc = subprocess.Popen(
        ["python", "main.py"],
        cwd=str(REPO / "v3-service"), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        wait_for_port(port)
    except TimeoutError:
        proc.terminate()
        _, err = proc.communicate(timeout=5)
        pytest.fail(f"v3-service never started: {err.decode()[-2000:]}")
    yield port
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture()
def workspace(workspace_root):
    ws = workspace_root / f"v3e2e-{uuid.uuid4().hex[:8]}"
    ws.mkdir(parents=True)
    yield ws
    shutil.rmtree(ws, ignore_errors=True)


@pytest.fixture()
def proxy(fake_llama, fake_lens, v3_service, sandbox_executor):
    port, proc = start_proxy({
        "ATLAS_INFERENCE_URL": f"http://127.0.0.1:{fake_llama}",
        "ATLAS_LENS_URL": f"http://127.0.0.1:{fake_lens}",
        "ATLAS_SANDBOX_URL": f"http://127.0.0.1:{sandbox_executor}",
        "ATLAS_V3_URL": f"http://127.0.0.1:{v3_service}",
    })
    yield port
    proc.terminate()
    proc.wait(timeout=10)


def _agent_body(workspace, **overrides):
    body = {
        "message": "Create a todo module in todo_app.py",
        "working_dir": str(workspace),
        "mode": "accept-edits",  # write_file auto-approved; V3 is the focus
        "session_id": f"v3e2e-{uuid.uuid4().hex[:8]}",
        # bypass_v3 deliberately ABSENT — the whole point.
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# The V3/Lens acceptance test
# ---------------------------------------------------------------------------

def test_v3_pipeline_with_lens_winner_selection(proxy, workspace):
    _FakeLensHandler.calls.clear()
    events = drive_agent_turn(proxy, _agent_body(workspace),
                              deadline_s=180.0)

    # No stage silently skipped: the v3_* event subsequence of a full
    # pipeline run, in order.
    ordered_subsequence(
        events,
        ("tool_call:write_file",
         lambda ev: ev["type"] == "tool_call"
         and ev["data"].get("name") == "write_file"),
        ("v3_progress:start",
         lambda ev: ev["type"] == "v3_progress"),
        ("v3_probe", lambda ev: ev["type"] == "v3_probe"),
        ("v3_plansearch", lambda ev: ev["type"] == "v3_plansearch"),
        ("v3_lens_per_step",
         lambda ev: ev["type"] == "v3_lens_per_step"),
        ("v3_sandbox", lambda ev: ev["type"] == "v3_sandbox"),
        ("v3_select:selected",
         lambda ev: ev["type"] == "v3_select"),
        ("tool_result:write_file",
         lambda ev: ev["type"] == "tool_result"
         and ev["data"].get("tool") == "write_file"
         and ev["data"].get("success") is True),
        ("done", lambda ev: ev["type"] == "done"),
    )

    # V3 was NOT bypassed, tested multiple candidates, and reports the
    # phase + score of the lens-selected winner.
    result = next(e for e in events
                  if e["type"] == "tool_result"
                  and e["data"].get("tool") == "write_file")
    payload = result["data"]["data"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    assert payload.get("v3_used") is True, payload
    assert payload.get("phase_solved") == "phase1", payload
    assert payload.get("candidates_tested", 0) >= 2, payload

    # Lens scoring genuinely participated: the fake lens saw C(x)
    # requests for both candidates and per-step G(x) requests.
    gx_texts = [t for p, t in _FakeLensHandler.calls
                if p == "/internal/lens/gx-score"]
    per_step = [t for p, t in _FakeLensHandler.calls
                if p == "/internal/lens/score-per-step"]
    assert any("# cand-A" in t for t in gx_texts), "cand-A never scored"
    assert any("# cand-B" in t for t in gx_texts), "cand-B never scored"
    assert len(per_step) >= 2, "per-step G(x) scoring not exercised"

    # Winner selection picked the candidate the Lens preferred (lowest
    # raw C(x) energy), and the proxy wrote exactly that candidate.
    written = (workspace / "todo_app.py").read_text()
    assert "# cand-A" in written, "lens-preferred candidate not written"
    assert "# cand-B" not in written
    assert abs(payload.get("winning_score", -1) - 0.20) < 1e-6, payload


def test_v3_unreachable_falls_back_to_direct_write(fake_llama, fake_lens,
                                                   sandbox_executor,
                                                   workspace):
    port, proc = start_proxy({
        "ATLAS_INFERENCE_URL": f"http://127.0.0.1:{fake_llama}",
        "ATLAS_LENS_URL": f"http://127.0.0.1:{fake_lens}",
        "ATLAS_SANDBOX_URL": f"http://127.0.0.1:{sandbox_executor}",
        "ATLAS_V3_URL": "http://127.0.0.1:9",  # unreachable
    })
    try:
        events = drive_agent_turn(port, _agent_body(workspace))
        result = next(e for e in events
                      if e["type"] == "tool_result"
                      and e["data"].get("tool") == "write_file")
        assert result["data"]["success"] is True
        payload = result["data"]["data"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert not payload.get("v3_used"), (
            "v3_used claimed despite unreachable V3")
        # The model's own content landed (visible fallback, not a skip).
        assert (workspace / "todo_app.py").read_text() == T2_CONTENT
    finally:
        proc.terminate()
        proc.wait(timeout=10)


class _MalformedV3Handler(http.server.BaseHTTPRequestHandler):
    """Answers /v3/generate with garbage SSE (no result frame) and
    /internal/cyclomatic_complexity with a valid no-op."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path.startswith("/internal/cyclomatic_complexity"):
            body = b'{"complexity": 1}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: {\"stage\":\"garbage\"}\n\ndata: [DONE]\n\n")
        self.wfile.flush()


class _StallingV3Handler(_MalformedV3Handler):
    def do_POST(self):
        if self.path.startswith("/internal/cyclomatic_complexity"):
            super().do_POST()
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: {\"stage\":\"plan_search\",\"detail\":\"stalling\"}\n\n")
        self.wfile.flush()
        time.sleep(20)  # far past the 1s cap the test sets


@pytest.mark.parametrize("handler,extra_env", [
    (_MalformedV3Handler, {}),
    (_StallingV3Handler, {"ATLAS_V3_TIMEOUT": "1"}),
], ids=["malformed-response", "timeout"])
def test_v3_failure_modes_fall_back_visibly(handler, extra_env, fake_llama,
                                            fake_lens, sandbox_executor,
                                            workspace):
    v3_port = free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", v3_port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port, proc = start_proxy({
        "ATLAS_INFERENCE_URL": f"http://127.0.0.1:{fake_llama}",
        "ATLAS_LENS_URL": f"http://127.0.0.1:{fake_lens}",
        "ATLAS_SANDBOX_URL": f"http://127.0.0.1:{sandbox_executor}",
        "ATLAS_V3_URL": f"http://127.0.0.1:{v3_port}",
        **extra_env,
    })
    try:
        events = drive_agent_turn(port, _agent_body(workspace),
                                  deadline_s=90.0)
        result = next(e for e in events
                      if e["type"] == "tool_result"
                      and e["data"].get("tool") == "write_file")
        assert result["data"]["success"] is True
        payload = result["data"]["data"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert not payload.get("v3_used")
        assert (workspace / "todo_app.py").read_text() == T2_CONTENT
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        server.shutdown()


def test_lens_unreachable_pipeline_completes_uncalibrated(
        fake_llama, sandbox_executor, workspace):
    """Lens is optional on this path: with it unreachable, v3-service
    must fall back to neutral scores (k=3 default, no vetoes), still
    test candidates in the sandbox, and complete the turn."""
    v3_port = free_port()
    env = {**os.environ,
           "ATLAS_V3_PORT": str(v3_port),
           "ATLAS_INFERENCE_URL": f"http://127.0.0.1:{fake_llama}",
           "ATLAS_LENS_URL": "http://127.0.0.1:9",  # unreachable
           "ATLAS_SERVICE_TOKEN_FILE": conftest._TOKEN_FILE,
           "ATLAS_SANDBOX_URL": f"http://127.0.0.1:{sandbox_executor}"}
    env.pop("ATLAS_CALL_GRAPH", None)
    v3_proc = subprocess.Popen(
        ["python", "main.py"],
        cwd=str(REPO / "v3-service"), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    wait_for_port(v3_port)
    port, proc = start_proxy({
        "ATLAS_INFERENCE_URL": f"http://127.0.0.1:{fake_llama}",
        "ATLAS_LENS_URL": "http://127.0.0.1:9",
        "ATLAS_SANDBOX_URL": f"http://127.0.0.1:{sandbox_executor}",
        "ATLAS_V3_URL": f"http://127.0.0.1:{v3_port}",
    })
    try:
        events = drive_agent_turn(port, _agent_body(workspace),
                                  deadline_s=180.0)
        result = next(e for e in events
                      if e["type"] == "tool_result"
                      and e["data"].get("tool") == "write_file")
        assert result["data"]["success"] is True
        payload = result["data"]["data"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert payload.get("v3_used") is True, (
            "lens outage must not disable V3 itself")
        # A candidate was written; with neutral scores the first passing
        # candidate wins — either tag is acceptable, but ONE of them is.
        written = (workspace / "todo_app.py").read_text()
        assert "# cand-A" in written or "# cand-B" in written
        assert any(e["type"] == "v3_sandbox" for e in events), (
            "sandbox verification stage missing")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        v3_proc.terminate()
        v3_proc.wait(timeout=10)
