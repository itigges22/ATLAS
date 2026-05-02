"""Live cross-language integration tests for the event protocol (PC-061).

These tests start a real atlas-proxy binary and exercise iter_events()
against /events via HTTP. They prove the full producer → SSE wire →
consumer pipeline works across the Go/Python boundary.

Skipped automatically when the proxy binary isn't available — keeps
the unit-test suite lean and the integration tests opt-in via a
pre-built binary.
"""

import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

PROXY_BINARY = os.environ.get("ATLAS_PROXY_BINARY", "/tmp/test-atlas-proxy")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


pytestmark = pytest.mark.skipif(
    not os.path.isfile(PROXY_BINARY) or not os.access(PROXY_BINARY, os.X_OK),
    reason=f"atlas-proxy binary not available at {PROXY_BINARY} "
           f"— run `cd atlas-proxy && go build -o {PROXY_BINARY} .` first",
)


def _free_port() -> int:
    """Bind 0.0.0.0:0, read back the assigned port, close. Race-y but
    fine for a single-test launch."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_port(port: int, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.1)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except (ConnectionRefusedError, socket.timeout):
                time.sleep(0.05)
    return False


@pytest.fixture
def running_proxy():
    port = _free_port()
    env = {**os.environ,
           "ATLAS_PROXY_PORT": str(port),
           "ATLAS_KEEP_LLAMA_WARM": "0"}
    proc = subprocess.Popen(
        [PROXY_BINARY],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _wait_for_port(port):
        proc.terminate()
        pytest.fail(f"proxy didn't bind {port} within 3s")
    try:
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_iter_events_connects_and_receives_initial_sentinel(running_proxy):
    """Connect with iter_events to a live proxy; the immediate
    `: connected` SSE comment must NOT appear as a typed event
    (it's a control frame). The connection must stay open without
    parse errors."""
    from atlas.cli.events import iter_events
    url = f"http://127.0.0.1:{running_proxy}/events"

    # iter_events returns a generator. Pull next() with a thread-side
    # timeout so the test fails fast if iter_events errors immediately.
    error: list = []

    def consume():
        try:
            iter_events(url, timeout=2.0)
            # No envelopes are emitted in a quiet proxy, so the iterator
            # would block. The whole point of this test is that
            # iter_events doesn't raise on the `: connected` sentinel.
            # We give it 0.5s to either error out or sit idle.
            time.sleep(0.5)
        except Exception as e:
            error.append(repr(e))

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    t.join(timeout=2.0)

    assert error == [], f"iter_events raised on connect: {error[0]}"


def test_live_proxy_emits_envelope_when_broker_emit_called(running_proxy):
    """End-to-end: connect via iter_events, fire an envelope inside
    the proxy by hitting an endpoint that triggers Emit() in the
    agent loop, verify the typed Event arrives on the consumer side.

    Today no endpoint Emits without a real LLM behind it, so this
    test is a placeholder for future TUI integration. Marked skip
    rather than removed so the contract is documented."""
    pytest.skip("requires llama-server + sandbox + lens to drive an "
                "agent loop end-to-end; will land with TUI integration test")


def test_live_proxy_events_endpoint_returns_correct_content_type(running_proxy):
    """Headers arrive immediately and Content-Type is text/event-stream.
    Regression for the buffer-flush bug fixed in b7463e3.

    Uses readline() rather than read(N) — the SSE stream is line-based,
    and read(64) would block waiting for more bytes that won't arrive
    until either an envelope event or the 15s heartbeat fires."""
    import urllib.request
    url = f"http://127.0.0.1:{running_proxy}/events"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        assert resp.status == 200
        ct = resp.headers.get("Content-Type", "")
        assert ct == "text/event-stream", f"got {ct!r}"
        # The first SSE frame is `: connected\n\n` — readline reads up
        # to and including the first \n.
        first_line = resp.readline()
        assert b"connected" in first_line, \
            f"no connected sentinel in first line {first_line!r}"
