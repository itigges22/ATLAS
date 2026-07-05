"""structured_log.py copies stay byte-identical + behave correctly."""

import hashlib
import importlib.util
import logging
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COPIES = [
    REPO / "geometric-lens" / "geometric_lens" / "structured_log.py",
    REPO / "sandbox" / "structured_log.py",
    REPO / "v3-service" / "structured_log.py",
]


def test_copies_byte_identical():
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in COPIES}
    assert len(set(digests.values())) == 1, (
        "structured_log.py copies drifted:\n"
        + "\n".join(f"  {p}: {d[:16]}" for p, d in digests.items()))


def _load(path):
    spec = importlib.util.spec_from_file_location(
        f"sl_{path.parent.name}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_request_id_roundtrip():
    m = _load(COPIES[0])
    m.set_request_id("req-abc")
    assert m.get_request_id() == "req-abc"
    m.set_request_id("")
    assert m.get_request_id() == ""


def test_json_formatter_shape():
    m = _load(COPIES[0])
    m.set_request_id("req-json")
    rec = logging.LogRecord("n", logging.INFO, "f", 1, "hello", None, None)
    out = m.JsonFormatter("svc").format(rec)
    import json
    d = json.loads(out)
    assert d["service"] == "svc" and d["level"] == "INFO"
    assert d["msg"] == "hello" and d["request_id"] == "req-json"
    m.set_request_id("")


def test_request_id_isolated_across_async_tasks():
    """Concurrent async tasks must not see each other's request id — the
    reason this uses contextvars, not threading.local (which would bleed
    between interleaved requests on one event-loop thread)."""
    import asyncio
    m = _load(COPIES[0])
    seen = {}

    async def worker(name):
        m.set_request_id(f"req-{name}")
        await asyncio.sleep(0)          # yield: another task runs here
        await asyncio.sleep(0)
        seen[name] = m.get_request_id()  # must still be our own id

    async def run():
        await asyncio.gather(*(worker(n) for n in ("a", "b", "c", "d")))

    asyncio.run(run())
    assert seen == {"a": "req-a", "b": "req-b", "c": "req-c", "d": "req-d"}, seen
