"""ATLAS event protocol — typed events streaming over SSE (PC-061).

This module is the canonical Python definition of the event envelope
and the consumer helpers. Producers (v3-service, atlas-proxy) emit
JSON in this shape; consumers (TUI, tests, bench CLI) call
`iter_events(url)` to receive typed `Event` objects.

The schema is also documented in docs/PROTOCOL.md; this docstring is
the executable spec.

Envelope shape
--------------

    {
      "event_id":    "evt_<8 hex>",
      "timestamp":   <float — Unix seconds with microsecond precision>,
      "type":        "stage_start" | "stage_end" | "tool_call" |
                     "tool_result" | "metric" | "error" | "done",
      "stage":       <str — pipeline stage name; e.g., "phase2", "pr_cot">,
      "parent_id":   "evt_<8 hex>" | null,   # pairs end-events to start-events
      "duration_ms": <int> | null,           # set on stage_end / tool_result
      "payload":     { ... type-specific ... }
    }

Per-type payload contracts
--------------------------

  stage_start   {detail?: str}
  stage_end     {detail?: str, success: bool, summary?: str}
  tool_call     {name: str, args_summary: str}
  tool_result   {name: str, success: bool, summary?: str}
  metric        {name: str, value: number, unit?: str}
  error         {stage: str, message: str, recoverable: bool}
  done          {success: bool, total_duration_ms: int, summary?: str}

The `done` event is always the last event in a stream. Consumers that
detect EOF without a `done` event should treat the stream as truncated.

Backward compatibility
----------------------

v3-service emits BOTH the new envelope AND the legacy
`{"stage": ..., "detail": ...}` shape for one release window. Consumers
that want envelopes opt in via the `Accept: application/json+envelope`
header or the `?event_format=v2` query param. `parse_envelope` raises
`LegacyEventError` on the legacy shape so callers can fall back
explicitly rather than silently mis-parsing.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Type registry — exhaustive list of legal envelope types
# ---------------------------------------------------------------------------

EVENT_TYPES = (
    "stage_start",
    "stage_end",
    "tool_call",
    "tool_result",
    "metric",
    "error",
    "done",
)


# ---------------------------------------------------------------------------
# Envelope dataclass
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single envelope-shaped event. `payload` is intentionally a dict —
    the per-type fields are documented above and validated in
    `parse_envelope`, but kept as `dict` so producers can add new payload
    fields without breaking the consumer.
    """
    event_id: str
    timestamp: float
    type: str
    stage: str
    payload: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "type": self.type,
            "stage": self.stage,
            "payload": self.payload,
        }
        if self.parent_id is not None:
            d["parent_id"] = self.parent_id
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class EventError(ValueError):
    """Base class for all envelope parsing errors."""


class LegacyEventError(EventError):
    """The blob looked like the legacy {stage, detail} shape, not an
    envelope. Callers can catch this to fall back to legacy handling."""


class SchemaError(EventError):
    """Envelope was malformed — missing required field, wrong type,
    unknown event_type, etc."""


# ---------------------------------------------------------------------------
# Producer helpers (used by tests and any Python producer; production
# emitters live in their own services)
# ---------------------------------------------------------------------------

def new_event_id() -> str:
    """Short, log-readable, session-unique. 8 hex chars from a uuid4 is
    enough — collision risk is negligible for any realistic stream."""
    return "evt_" + uuid.uuid4().hex[:8]


def make_event(type: str, stage: str, payload: Optional[Dict[str, Any]] = None,
                parent_id: Optional[str] = None,
                duration_ms: Optional[int] = None,
                timestamp: Optional[float] = None) -> Event:
    """Build a well-formed Event with sensible defaults.

    Validates `type` is one of EVENT_TYPES — catches typos at producer
    time rather than letting them through to the consumer.
    """
    if type not in EVENT_TYPES:
        raise SchemaError(f"unknown event type {type!r}; "
                          f"must be one of {EVENT_TYPES}")
    return Event(
        event_id=new_event_id(),
        timestamp=timestamp if timestamp is not None else time.time(),
        type=type,
        stage=stage,
        payload=payload or {},
        parent_id=parent_id,
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Parser / validator
# ---------------------------------------------------------------------------

_REQUIRED = ("event_id", "timestamp", "type", "stage", "payload")
_LEGACY_KEYS = {"stage", "detail"}


def parse_envelope(blob: Any) -> Event:
    """Parse a JSON blob (str or dict) into an Event.

    Raises:
      LegacyEventError — the blob is the legacy {stage, detail} shape
      SchemaError       — the blob is malformed
    """
    if isinstance(blob, str):
        try:
            blob = json.loads(blob)
        except json.JSONDecodeError as e:
            raise SchemaError(f"not valid JSON: {e}") from e
    if not isinstance(blob, dict):
        raise SchemaError(f"envelope must be a JSON object, got {type(blob).__name__}")

    # Legacy detection: exactly the legacy keyset, no envelope keys.
    keys = set(blob.keys())
    if keys == _LEGACY_KEYS or (keys <= _LEGACY_KEYS and "stage" in keys
                                  and "type" not in keys
                                  and "event_id" not in keys):
        raise LegacyEventError(
            f"blob is the legacy {{stage, detail}} shape, not an envelope: "
            f"{blob!r}. Opt into v2 events via the "
            f"Accept: application/json+envelope header.")

    for key in _REQUIRED:
        if key not in blob:
            raise SchemaError(f"missing required field {key!r}: {blob!r}")

    if blob["type"] not in EVENT_TYPES:
        raise SchemaError(f"unknown event type {blob['type']!r}; "
                          f"must be one of {EVENT_TYPES}")
    if not isinstance(blob["timestamp"], (int, float)):
        raise SchemaError(f"timestamp must be a number, got "
                          f"{type(blob['timestamp']).__name__}")
    if not isinstance(blob["payload"], dict):
        raise SchemaError(f"payload must be an object, got "
                          f"{type(blob['payload']).__name__}")

    return Event(
        event_id=blob["event_id"],
        timestamp=float(blob["timestamp"]),
        type=blob["type"],
        stage=blob["stage"],
        payload=blob["payload"],
        parent_id=blob.get("parent_id"),
        duration_ms=blob.get("duration_ms"),
    )


# ---------------------------------------------------------------------------
# SSE consumer
# ---------------------------------------------------------------------------

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
            if data == "[DONE]" or data.startswith("done: ") or data == "":
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
