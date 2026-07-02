"""Deterministic end-to-end acceptance test — no GPU, no model.

Boots the REAL control plane:

    fake llama-server (scripted SSE)  <-- atlas-proxy binary -->  real
                                                sandbox executor (uvicorn)

then drives one complete agent turn through the real protocol:

    open session -> read_file -> edit_file -> run_command (sandbox
    verification, gated by an interactive permission approve) -> done

and asserts every stage happened, in order, with the file actually
fixed on disk. The model is a four-step script served by the fake
llama-server; everything else (agent loop, guardrails, permission
gate, workspace containment, sandbox execution, SSE protocol) is the
production code path.

Requirements (all provided by the e2e CI job; skipped cleanly when
absent locally):
  - the proxy binary at $ATLAS_PROXY_BINARY (default /tmp/test-atlas-proxy;
    build with `cd proxy && go build -o /tmp/test-atlas-proxy .`)
  - sandbox runtime deps (fastapi/uvicorn/pydantic/defusedxml)

The sandbox executor runs with ATLAS_SANDBOX_WORKSPACE_ROOT pointed at a
pytest tmp dir, so no /workspace mount or sudo is needed.
"""

import http.client
import http.server
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest

PROXY_BINARY = os.environ.get("ATLAS_PROXY_BINARY", "/tmp/test-atlas-proxy")

BUGGY_APP = '''def greeting(name):
    return "Hello, " + nmae


if __name__ == "__main__":
    print(greeting("world"))
'''

OLD_STR = 'return "Hello, " + nmae'
NEW_STR = 'return "Hello, " + name'


def _sandbox_deps_available() -> bool:
    try:
        import fastapi, uvicorn, defusedxml  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.skipif(
        not (os.path.isfile(PROXY_BINARY) and os.access(PROXY_BINARY, os.X_OK)),
        reason=f"atlas-proxy binary not available at {PROXY_BINARY} "
               f"— run `cd proxy && go build -o {PROXY_BINARY} .` first"),
    pytest.mark.skipif(
        not _sandbox_deps_available(),
        reason="sandbox runtime deps missing — "
               "pip install -r sandbox/requirements-runtime.txt"),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"port {port} never came up")


# ---------------------------------------------------------------------------
# Fake llama-server — serves the scripted model turn by turn
# ---------------------------------------------------------------------------

def _model_script(tool_results_seen: int) -> str:
    """The model's next envelope, keyed off how many tool results it has
    been shown — robust to any extra user-role nudges the proxy injects."""
    if tool_results_seen == 0:
        return json.dumps({"type": "tool_call", "name": "read_file",
                           "args": {"path": "app.py"}})
    if tool_results_seen == 1:
        return json.dumps({"type": "tool_call", "name": "edit_file",
                           "args": {"path": "app.py", "old_str": OLD_STR,
                                    "new_str": NEW_STR}})
    if tool_results_seen == 2:
        return json.dumps({"type": "tool_call", "name": "run_command",
                           "args": {"command": "python3 -m py_compile app.py",
                                    "timeout": 30}})
    return json.dumps({"type": "done",
                       "summary": "Fixed the NameError in app.py and "
                                  "verified it compiles."})


class _FakeLlamaHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_GET(self):
        if self.path.startswith("/health"):
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:  # /slots etc. — the prompt-progress poller stops on 404
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length))
        tool_results = sum(
            1 for m in req.get("messages", [])
            if m.get("role") == "user"
            and m.get("content", "").startswith("[tool result]"))
        envelope = _model_script(tool_results)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        delta = json.dumps({"choices": [{"delta": {"content": envelope}}]})
        usage = json.dumps({"choices": [],
                            "usage": {"total_tokens": 20,
                                      "prompt_tokens": 15,
                                      "completion_tokens": 5}})
        for line in (delta, usage, "[DONE]"):
            self.wfile.write(f"data: {line}\n\n".encode())
        self.wfile.flush()


# ---------------------------------------------------------------------------
# Fixtures — fake llama, real sandbox executor, real proxy binary
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fake_llama():
    port = _free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port),
                                             _FakeLlamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.fixture(scope="module")
def workspace_root(tmp_path_factory):
    return tmp_path_factory.mktemp("workspace-root")


@pytest.fixture(scope="module")
def sandbox_executor(tmp_path_factory, workspace_root):
    port = _free_port()
    scratch = tmp_path_factory.mktemp("sandbox-scratch")
    env = {**os.environ,
           "WORKSPACE_BASE": str(scratch),
           "ATLAS_SANDBOX_WORKSPACE_ROOT": str(workspace_root),
           "MAX_EXECUTION_TIME": "60"}
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "executor_server:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "error"],
        cwd=str(repo_root / "sandbox"), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        _wait_for_port(port)
    except TimeoutError:
        proc.terminate()
        _, err = proc.communicate(timeout=5)
        pytest.fail(f"sandbox executor never started: {err.decode()[-2000:]}")
    yield port
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture()
def workspace(workspace_root):
    ws = workspace_root / f"e2e-{uuid.uuid4().hex[:8]}"
    ws.mkdir(parents=True)
    (ws / "app.py").write_text(BUGGY_APP)
    yield ws
    shutil.rmtree(ws, ignore_errors=True)


@pytest.fixture()
def proxy(fake_llama, sandbox_executor):
    port = _free_port()
    # Explicit minimal env: nothing leaks in from the developer's shell
    # or a repo .env — every ATLAS_* value the turn depends on is pinned.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "ATLAS_PROXY_PORT": str(port),
        "ATLAS_INFERENCE_URL": f"http://127.0.0.1:{fake_llama}",
        "ATLAS_SANDBOX_URL": f"http://127.0.0.1:{sandbox_executor}",
        "ATLAS_LENS_URL": "http://127.0.0.1:9",  # dead port — lens fail-soft
        "ATLAS_V3_URL": "http://127.0.0.1:9",    # bypass_v3 skips it anyway
        "ATLAS_KEEP_LLAMA_WARM": "0",
        "ATLAS_PERMISSION_TIMEOUT_SEC": "30",    # fail-safe, not expected
    }
    proc = subprocess.Popen([PROXY_BINARY], env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    try:
        _wait_for_port(port)
    except TimeoutError:
        proc.terminate()
        _, err = proc.communicate(timeout=5)
        pytest.fail(f"proxy never bound: {err.decode()[-2000:]}")
    yield port
    proc.terminate()
    proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# SSE driver
# ---------------------------------------------------------------------------

def _post_json(port: int, path: str, body: dict) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", path, json.dumps(body),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return json.loads(resp.read() or b"{}")
    finally:
        conn.close()


def _drive_agent_turn(port: int, body: dict, deadline_s: float = 90.0):
    """POST /v1/agent, stream events, answer permission prompts inline.

    Returns the ordered event list (each {"type", "data"}).
    """
    events = []
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=deadline_s)
    conn.request("POST", "/v1/agent", json.dumps(body),
                 {"Content-Type": "application/json",
                  "Accept": "text/event-stream"})
    resp = conn.getresponse()
    assert resp.status == 200, resp.read()[:500]

    deadline = time.monotonic() + deadline_s
    buf = b""
    done = False
    while not done:
        assert time.monotonic() < deadline, (
            f"turn did not complete in {deadline_s}s; events so far: "
            f"{[e['type'] for e in events]}")
        chunk = resp.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.decode("utf-8", "replace").splitlines():
                if not line.startswith("data: "):
                    continue  # ": connected" comment, heartbeats
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    done = True
                    continue
                ev = json.loads(payload)
                events.append(ev)
                if ev["type"] == "permission_request":
                    # Approve from a second connection while the agent
                    # SSE stream stays open — the real client flow.
                    answer = _post_json(port, "/v1/permission", {
                        "session_id": body["session_id"],
                        "tool_call_id": ev["data"]["tool_call_id"],
                        "decision": "allow",
                        "scope": "once",
                    })
                    assert answer.get("delivered") is True, answer
    conn.close()
    return events


def _ordered_subsequence(events, *predicates):
    """Assert the predicates match, in order, over the event list.
    Returns the matched events."""
    matched = []
    it = iter(events)
    for name, pred in predicates:
        for ev in it:
            if pred(ev):
                matched.append(ev)
                break
        else:
            raise AssertionError(
                f"stage {name!r} missing (or out of order); event sequence: "
                f"{[e['type'] for e in events]}")
    return matched


# ---------------------------------------------------------------------------
# The acceptance test
# ---------------------------------------------------------------------------

def test_full_agent_turn_read_edit_verify_permission_done(proxy, workspace):
    session = f"e2e-{uuid.uuid4().hex[:8]}"
    events = _drive_agent_turn(proxy, {
        "message": "Fix the NameError in app.py, then verify it compiles "
                   "with python3 -m py_compile app.py",
        "working_dir": str(workspace),
        "mode": "default",
        "session_id": session,
        "bypass_v3": True,
    })

    def tool_call(name):
        return (f"tool_call:{name}",
                lambda ev: ev["type"] == "tool_call"
                and ev["data"].get("name") == name)

    def tool_ok(name):
        return (f"tool_result:{name}",
                lambda ev: ev["type"] == "tool_result"
                and ev["data"].get("tool") == name
                and ev["data"].get("success") is True)

    # Every stage, in order — a silently skipped stage fails here.
    _ordered_subsequence(
        events,
        tool_call("read_file"),
        tool_ok("read_file"),
        tool_call("edit_file"),
        tool_ok("edit_file"),
        tool_call("run_command"),
        ("permission_request",
         lambda ev: ev["type"] == "permission_request"
         and ev["data"].get("tool_name") == "run_command"),
        tool_ok("run_command"),
        ("done", lambda ev: ev["type"] == "done"),
    )

    # The permission prompt paused execution: the approval must precede
    # the run_command result (ordered_subsequence already proved this),
    # and exactly one prompt fired — read_file/edit_file are not
    # destructive and must not prompt.
    prompts = [e for e in events if e["type"] == "permission_request"]
    assert len(prompts) == 1, [e["type"] for e in events]
    assert not any(e["type"] == "permission_denied" for e in events)

    # The edit really landed: the sandbox verified the FIXED file.
    final = (workspace / "app.py").read_text()
    assert NEW_STR in final and "nmae" not in final

    # The verification ran in the real sandbox executor (py_compile
    # writes __pycache__ next to the file).
    assert (workspace / "__pycache__").is_dir(), (
        "run_command did not execute in the workspace")

    # Turn-level accounting reached the client.
    done_ev = next(e for e in events if e["type"] == "done")
    assert done_ev["data"].get("summary")


def test_session_less_destructive_call_is_denied(proxy, workspace):
    """Fail-closed contract: no session_id + default mode means the
    run_command permission prompt cannot be answered — the proxy must
    deny it, not silently execute."""
    events = _drive_agent_turn(proxy, {
        "message": "Fix the NameError in app.py, then verify it compiles "
                   "with python3 -m py_compile app.py",
        "working_dir": str(workspace),
        "mode": "default",
        "session_id": "",
        "bypass_v3": True,
    }, deadline_s=60.0)

    assert any(e["type"] == "permission_denied" for e in events), (
        f"no denial event: {[e['type'] for e in events]}")
    # The command never executed.
    assert not (workspace / "__pycache__").exists()
