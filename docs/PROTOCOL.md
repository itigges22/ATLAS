# ATLAS Event Protocol (PC-061)

ATLAS services emit a typed JSON event stream over Server-Sent Events (SSE). This document is the wire-format spec — every producer (v3-service Python, atlas-proxy Go) implements it identically, and every consumer (TUI, tests, future web dashboard, bench CLI) reads it through `atlas/cli/events.py`.

This protocol is the foundation for Phase 1's UI/UX work. The Bubbletea TUI, live decode streaming, per-stage spinners, and compute budget visualizer all consume these events.

## Why this exists

Before PC-061, ATLAS had three different progress mechanisms:

- **v3-service** spoke SSE but with an opaque `{"stage": <str>, "detail": <str>}` shape — consumers had to string-parse to know what kind of event they got.
- **atlas-proxy** emitted bracket-tagged stdout lines (`[agent] turn=N type=X name=Y args=Z`) — pseudo-structured but not parseable as data.
- **The benchmark runners** had their own progress format.

A TUI that wants to render "current stage" / "elapsed time" / "tool calls so far" had to triple-implement parsing for all three. PC-061 unifies them into one schema with explicit event types and a consumer library so the TUI is one parser, not three.

## Transport

**Server-Sent Events (SSE)** — `text/event-stream`, server-push only. Cancellation is out of scope for this protocol; clients cancel via a separate POST endpoint (TBD when the TUI lands).

Each event arrives as one SSE frame:

```
data: {"event_id":"evt_aabb1122",...}

```

(Two newlines terminate the frame.) The Python helper `atlas.cli.events.iter_sse_lines` handles the framing.

### SSE control frames (server → client only)

The protocol uses three SSE comment / control patterns. None are envelope events; consumers must skip them via the lines parser:

| Frame | When | Why |
|---|---|---|
| `: connected\n\n` | First body byte after a successful `/events` connection (atlas-proxy only) | Forces the response headers + first body chunk to leave the server immediately. Without it, Go buffers the response until the first envelope or 15s heartbeat fires, and clients with short connect timeouts see "no response received" (PC-061 follow-up). |
| `: heartbeat\n\n` | Every 15s during quiet stretches (atlas-proxy only) | Keeps proxies / load balancers from idling out the connection. |
| `event: result\ndata: {...}\n\n` | Right before stream end on `/v3/run` (v3-service only, legacy) | Carries the final pipeline `result` dict in the legacy back-compat shape. v2 envelope consumers should ignore this and watch for the `done` envelope instead. |

The Python `iter_sse_lines` helper already filters comment lines (any line starting with `:`) automatically. Named-event lines (`event: result`) come through prefixed (`result: <data>`) so the caller can distinguish them.

## Single-session broadcast model (current limitation)

atlas-proxy's `/events` endpoint **broadcasts every envelope event from every concurrent agent session to every connected subscriber.** There is no `session_id` field in the envelope, and no per-session `?session_id=X` filtering on the endpoint.

For Phase 1's single-user TUI consumer, this is fine — most users run one ATLAS at a time, so the broadcast model degenerates to "you see your own session." If multi-user / multi-session use cases emerge:

1. Add a `session_id` field to the envelope (back-compatible — consumers that don't recognize it ignore it)
2. Add a `?session_id=X` filter to `/events` that drops envelopes whose session doesn't match

Documented as a known scope-defining choice rather than a bug. v3-service's `/v3/run` endpoint is per-request streaming, so session interleaving is not an issue there.

## Envelope

Every event is a JSON object with this shape:

```json
{
  "event_id":    "evt_<8 hex chars>",
  "timestamp":   1714600000.123,
  "type":        "stage_start" | "stage_end" | "tool_call" | "tool_result" | "metric" | "error" | "done",
  "stage":       "<pipeline stage name>",
  "parent_id":   "evt_<8 hex chars>",   // optional — pairs end-events to start-events
  "duration_ms": 4523,                  // optional — set on stage_end / tool_result
  "payload":     { ... type-specific ... }
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `event_id` | string | yes | Format: `evt_` + 8 hex chars (uuid4 truncated). Session-unique. |
| `timestamp` | number | yes | Unix seconds with microsecond precision. Producers MUST emit in monotonically non-decreasing order. |
| `type` | string | yes | One of the seven legal values listed below. |
| `stage` | string | yes | Logical pipeline stage (`probe`, `phase2`, `pr_cot`, `agent`, etc.). Suffixes like `_pass`/`_done`/`_failed` are stripped before placing in this field. |
| `payload` | object | yes | Type-specific. Always an object, never null. |
| `parent_id` | string | optional | Set on `stage_end` to point at the matching `stage_start`. Lets consumers compute durations and nest UI elements. |
| `duration_ms` | int | optional | Set on `stage_end` and `tool_result`. Convenience — equivalent to `(this.timestamp - parent.timestamp) * 1000`. |

## Event types

### `stage_start`

A logical stage has begun. Pairs with a later `stage_end` event (typically — long-running stages that crash without an end event are valid; consumers must handle missing `stage_end`).

```json
{
  "type": "stage_start",
  "stage": "phase2",
  "payload": {
    "detail": "allocating candidates"   // optional human-readable
  }
}
```

### `stage_end`

A logical stage finished. `success` is required so consumers can distinguish completed-OK from completed-failed without inspecting the next event.

```json
{
  "type": "stage_end",
  "stage": "phase2",
  "parent_id": "evt_aabb1122",
  "duration_ms": 4523,
  "payload": {
    "success": true,
    "detail": "selected candidate #3",   // optional
    "summary": "..."                     // optional
  }
}
```

### `tool_call`

The agent is invoking a tool. Always followed by exactly one `tool_result` for the same tool name within the same stage (typically a few hundred ms later).

```json
{
  "type": "tool_call",
  "stage": "agent",
  "payload": {
    "name": "edit_file",
    "args_summary": "src/snake.py: replace render() body",   // truncated to ~200 chars
    "turn": 3
  }
}
```

### `tool_result`

```json
{
  "type": "tool_result",
  "stage": "agent",
  "duration_ms": 487,
  "payload": {
    "name": "edit_file",
    "success": true,
    "summary": "..."
  }
}
```

### `metric`

A measured value worth surfacing. Used for `gx_score`, `candidates_generated`, `tokens_used`, etc.

```json
{
  "type": "metric",
  "stage": "lens",
  "payload": {
    "name": "gx_score",
    "value": 0.83,
    "unit": "score"   // optional
  }
}
```

### `error`

Something went wrong. `recoverable: true` means the pipeline continues (the model retries, the stage reruns); `false` means the run is about to terminate with a `done` event reporting `success: false`.

```json
{
  "type": "error",
  "stage": "pr_cot",
  "payload": {
    "stage": "pr_cot",
    "message": "model output was empty",
    "recoverable": true
  }
}
```

### `done`

Always the last event in a stream. Consumers that detect EOF without a `done` event should treat the stream as truncated (network drop, server crash, etc.).

```json
{
  "type": "done",
  "stage": "pipeline",
  "payload": {
    "success": true,
    "total_duration_ms": 12453,
    "summary": "..."   // optional
  }
}
```

## Producer endpoints

| Service | Endpoint | Notes |
|---|---|---|
| atlas-proxy | `GET /events` | Broadcasts all envelope events from any active session to every connected subscriber. Heartbeat every 15s to defeat proxy idle timeouts. |
| v3-service | `POST /v3/run` (existing) | Dual-emits: legacy `{stage, detail}` always; envelope additionally when client opts in via `Accept: application/json+envelope` or `?event_format=v2`. |

## Opting into typed events

For back-compat, v3-service's `/v3/run` keeps emitting the legacy `{stage, detail}` shape unconditionally. Clients that want envelopes opt in via either:

```
Accept: application/json+envelope
```

or appending to the URL:

```
?event_format=v2
```

When the opt-in is present, v3-service emits both shapes (legacy first, then envelope). Consumers that opt in must filter the legacy frames — the Python helper `atlas.cli.events.iter_events()` does this automatically (skips frames that raise `LegacyEventError`).

The dual-emission window stays open for one release after PC-061. Removal of the legacy shape is a separate ticket once downstream consumers (Aider integration, current bench runners) have all migrated.

## Consumer library: `atlas/cli/events.py`

Python consumers should not parse SSE frames themselves. Use:

```python
from atlas.cli.events import iter_events

for ev in iter_events("http://localhost:8090/events"):
    print(ev.type, ev.stage, ev.payload)
    if ev.type == "done":
        break
```

`iter_events(url)` yields `Event` dataclass objects with typed fields. `parse_envelope(blob)` parses one frame; raises `LegacyEventError` if the blob is the v3-service legacy shape, `SchemaError` if it's malformed.

## Stage-name suffix conventions (v3-service)

v3-service uses these suffix conventions to derive envelope `type` from the existing 46 stage names without hand-mapping each one:

| Suffix | Envelope type | Notes |
|---|---|---|
| `_error` | `error` | Wrapped with `recoverable: true` by default. |
| `_failed` | `stage_end` | `payload.success = false`. |
| `_pass` | `stage_end` | `payload.success = true`. |
| `_done` | `stage_end` | `payload.success = true`. |
| `_skip` | `stage_end` | `payload.success = true` (skipped is success in flow). |
| `_retry` | `stage_start` | A re-attempt of the same logical stage. |
| (no suffix) | `stage_start` | Fresh stage entry. |

Suffixes are stripped from the `stage` field, so `stage_start("phase2")` and `stage_end("phase2")` share the same logical name. This lets consumers pair them via `parent_id`.

## Schema versioning

This document describes **v1** of the protocol. Future schema changes:

- **Backward-compatible additions** (new event types, new optional payload fields): bump the `Accept` header version (`application/json+envelope; v=2`). Consumers that don't recognize the version still work — they just ignore unknown event types.
- **Breaking changes** (renamed fields, type changes): require a new endpoint path (`/events/v2`). The old endpoint stays for one release window.

## Test contract

The schema is pinned by `tests/cli/test_events.py` (Python consumer) and `atlas-proxy/events_test.go` (Go producer). Any change to the wire format MUST update both, in lockstep. v3-service's emitter is tested in `tests/v3-service/test_event_emission.py`.
