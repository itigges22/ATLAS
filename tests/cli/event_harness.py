"""SSE event-stream test harness for the atlas event spec.

Consumer/assertion helpers used only by the test suite; the wire spec
itself (EVENT_TYPES, Event, parse_envelope) lives in atlas.events.
"""

from typing import Dict, Iterator, List, Optional

from atlas.events import Event, LegacyEventError, SchemaError, parse_envelope

def iter_sse_lines(stream) -> Iterator[str]:
    """Yield decoded `data:` lines from an SSE byte stream. Handles the
    standard SSE framing: lines prefixed `data: `, blank line as event
    delimiter. `event:` lines are echoed as `<event-name>: <data>` so the
    caller can distinguish stream-control events (`event: result`,
    `event: done`) from plain `data:` events.
    """
    pending_event: Optional[str] = None
    for raw in stream:
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
        else:
            line = raw.rstrip("\n").rstrip("\r")
        if not line:
            pending_event = None  # event boundary
            continue
        if line.startswith(":"):
            continue  # SSE comment / heartbeat
        if line.startswith("event:"):
            pending_event = line[len("event:"):].strip()
            continue
        if line.startswith("data:"):
            data = line[len("data:"):].lstrip()
            if pending_event:
                yield f"{pending_event}: {data}"
            else:
                yield data


def iter_events(url: str, timeout: float = 30.0,
                 headers: Optional[Dict[str, str]] = None) -> Iterator[Event]:
    """Stream typed Event objects from an SSE URL. Yields until the
    server closes the connection or a `done` event is received.

    Legacy `{stage, detail}` events are silently skipped — the caller
    explicitly opted into typed events via the URL or headers.
    """
    import urllib.request
    req_headers = {"Accept": "application/json+envelope, text/event-stream"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for data in iter_sse_lines(resp):
            if (data == "[DONE]" or data.startswith("done: ")
                    or data.startswith("result: ") or data == ""):
                # done/result are stream-control frames (the v3 legacy
                # terminal frame rides `event: result`), not envelopes.
                continue
            try:
                yield parse_envelope(data)
            except LegacyEventError:
                continue  # caller opted into typed; skip legacy frames


# ---------------------------------------------------------------------------
# Sequencing helpers (used by tests, future TUI, future timing analysis)
# ---------------------------------------------------------------------------

def is_terminal(event: Event) -> bool:
    """True if this event is the final one in a stream."""
    return event.type == "done"


def collect(events: Iterator[Event]) -> List[Event]:
    """Drain an event iterator into a list. Convenience wrapper for tests."""
    return list(events)


def assert_monotonic(events: List[Event]) -> None:
    """Raise SchemaError if timestamps are not monotonically non-decreasing.
    Tolerates equal timestamps (events emitted in the same microsecond)."""
    last = float("-inf")
    for ev in events:
        if ev.timestamp < last:
            raise SchemaError(
                f"non-monotonic timestamp at {ev.event_id}: "
                f"{ev.timestamp} < previous {last}")
        last = ev.timestamp
