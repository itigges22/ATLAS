"""No V3 inference may START after the client is gone.

Measured on a real acquisition: case v3c-b-walk4 issued 23 serial inference
calls, and the 23rd STARTED at +569.8s -- the work deadline -- then ran 39.8s
to completion after the agent request had already returned. The relay recorded
it `completed`, never cancelled, and the acquisition's 30s drain expired with
that one call still in flight.

V3 is a synchronous ThreadingHTTPServer. Its only disconnect signal is a
BrokenPipeError on the next SSE write, which sets `disconnected` on the
progress callback, and the pipeline consults that flag at PHASE boundaries.
Nothing consulted it at the point that actually spends GPU: the dispatch of an
inference call. These tests pin that gate at the one chokepoint every V3 call
passes through.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "v3-service"))
import adapters  # noqa: E402


class Callback:
    """Stands in for the handler's emit_progress, which carries the flag."""

    def __init__(self, disconnected=False):
        self.disconnected = disconnected
        self.events = []

    def __call__(self, stage, detail, **kw):
        self.events.append((stage, detail))


def adapter(cb, sent):
    a = adapters.LLMAdapter(progress_callback=cb)

    def _send(*args, **kwargs):
        sent.append(1)
        return {"choices": [{"text": "```python\nx = 1\n```"}],
                "usage": {"total_tokens": 10}}

    a._send = _send
    return a


def test_dispatch_is_refused_once_the_client_is_gone():
    sent = []
    cb = Callback(disconnected=True)
    a = adapter(cb, sent)
    with pytest.raises(adapters.ClientDisconnected):
        a("prompt", 0.7, 128, 42)
    assert sent == [], "an inference call was dispatched after the client disconnected"


def test_a_live_client_still_dispatches():
    sent = []
    cb = Callback(disconnected=False)
    a = adapter(cb, sent)
    out = a("prompt", 0.7, 128, 42)
    assert sent == [1]
    assert out[0]


def test_the_gate_is_evaluated_per_call_not_once():
    sent = []
    cb = Callback(disconnected=False)
    a = adapter(cb, sent)
    a("prompt", 0.7, 128, 1)
    cb.disconnected = True          # the SSE write fails between calls
    with pytest.raises(adapters.ClientDisconnected):
        a("prompt", 0.7, 128, 2)
    assert sent == [1], "a second call was dispatched after mid-run disconnect"


def test_no_callback_means_no_gate():
    """A non-agent caller (bench, CLI) has no SSE client and must be unaffected."""
    sent = []
    a = adapters.LLMAdapter(progress_callback=None)

    def _send(*args, **kwargs):
        sent.append(1)
        return {"choices": [{"text": "```python\nx = 1\n```"}],
                "usage": {"total_tokens": 10}}

    a._send = _send
    assert a("prompt", 0.7, 128, 42)[0]
    assert sent == [1]


def test_the_walk4_shape_starts_no_late_call():
    """23 serial calls; the client goes at the 22nd. The 23rd must not start."""
    sent = []
    cb = Callback()
    a = adapter(cb, sent)
    for i in range(22):
        a("prompt", 0.7, 128, i)
    assert len(sent) == 22
    cb.disconnected = True
    with pytest.raises(adapters.ClientDisconnected):
        a("prompt", 0.7, 128, 22)
    assert len(sent) == 22, "the late call was dispatched anyway"
