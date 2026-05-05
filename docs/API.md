# ATLAS API Reference

API endpoints for each ATLAS service. All services communicate over HTTP/JSON. Streaming endpoints use Server-Sent Events (SSE).

> **Ports listed are defaults.** All are configurable via environment variables (see [CONFIGURATION.md](CONFIGURATION.md)). Docker Compose maps container-internal ports to host ports — the ports below are what you hit from the host.

---

## atlas-proxy (Port 8090)

The main entry point. Wraps llama-server with an agent loop, grammar-constrained tool calls, Lens scoring, and sandbox verification.

**This is the public client surface.** The canonical client is [atlas-tui](CLI.md), but the contract below is stable and other front-ends (web UIs, editor plugins, CI bots, custom CLIs) can use it directly. Tracking issue for richer client-facing docs: [PC-063](../ISSUES.md).

There are three primary endpoints for building a client:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/agent` | POST | Send a user message, stream back a turn (tool calls, results, tokens, completion) as SSE |
| `/cancel` | POST | Abort an in-flight `/v1/agent` turn by `session_id` |
| `/events` | GET | Subscribe to a global typed-envelope event broker (PC-061) — same events the TUI's pipeline pane uses |

Plus three legacy / utility endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions (legacy, kept for SDK compatibility) |
| `/v1/models` | GET | List available models (OpenAI-compatible) |
| `/health` | GET | Liveness + counters |

---

### POST /v1/agent

Tool-based agent endpoint. Sends a user message, runs the agent loop (LLM → tool call → tool result → repeat) until the model emits `done` or hits the turn cap, and streams every step back as SSE.

**Request:**
```json
{
  "message": "Add a snake game in Python and verify it runs",
  "working_dir": "/home/me/projects/snake",
  "mode": "default",
  "session_id": "tui-7f3a2c1b"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | (required) | The user's request |
| `working_dir` | string | `"."` | Host-side working directory. Inside the proxy container this is overridden to `ATLAS_WORKSPACE_DIR` (the bind-mount target — `/workspace` by default). The startup wrapper aligns the bind mount to the user's cwd, so writes land in the right place. |
| `mode` | string | `"default"` | Permission mode: `"default"` (prompt for destructive ops), `"accept-edits"` (auto-approve write/edit, prompt for delete/run), `"yolo"` (auto-approve everything) |
| `session_id` | string | `""` | Optional. Required if the client wants to be able to call `/cancel`. The proxy stores a `context.CancelFunc` keyed by this id while the turn is running. |

**Response:** `text/event-stream` of `data: {...}\n\n` lines. The proxy flushes a `: connected\n\n` SSE comment on connect so clients see HTTP/200 immediately, then emits typed events for the duration of the turn, terminated by `data: [DONE]\n\n`.

#### Event types on `/v1/agent`

Every event has the shape `{"type":"<name>","data":{...}}`. Types in emission order for a typical turn:

| Type | When | Payload |
|------|------|---------|
| `turn_start` | At the start of every agent loop iteration | `turn` (int), `messages` (int), `trimmed` (bool, true if conversation history was trimmed for context window) |
| `llm_call_start` | Before each LLM round-trip | `turn`, `messages`, `prompt_tokens` (estimated, chars/4) |
| `llm_prompt_progress` | Every ~250 ms while llama-server is in prompt-eval (before any decoded token) — only emitted when llama-server's `/slots` endpoint is enabled | `processed` (int), `total` (int), `pct` (0–1 float). Stops as soon as `llm_first_token` fires. |
| `llm_first_token` | First streamed delta from llama-server | `prompt_ms` (time-to-first-token in milliseconds) |
| `llm_token` | Each streamed delta | `text` (the delta string — typically a token or two) |
| `llm_call_end` | LLM call finished | `turn`, `tokens` (this call), `total_tokens` (cumulative for the turn), `ms`, `chars`. On error: `error` instead of `tokens`/`chars`. |
| `tool_call` | Model emitted a `{"type":"tool_call",...}` JSON | `name` (string), `args` (raw JSON), `turn` |
| `permission_denied` | User said no via `PermissionFn` callback | `tool` (the tool name) |
| `tool_result` | Tool finished executing | `tool`, `success` (bool), `data` (raw JSON), `error` (string), `elapsed` (Go duration string, e.g. `"245ms"`) |
| `text` | Model emitted a `{"type":"text","content":"..."}` JSON (conversational reply) | `content` (string) |
| `v3_progress` | V3 pipeline stage that doesn't have a dedicated typed event yet (fallback) | `message` (string) — humanized stage label |
| `v3_llm_start` / `v3_llm_end` | V3's internal LLMAdapter started / finished a call (planner, candidate generation, repair, etc.) | `detail` (string), `call` (int), `tokens` (int, on `llm_end`), `elapsed_ms` (int, on `llm_end`), `max_tokens`, `temperature` (on `llm_start`) |
| `v3_token` | V3's internal LLM streamed a token | `text` (delta string) |
| `v3_phase` | V3 phase transition (`phase1`, `phase2`, `phase2_allocated`) | `stage`, `detail`, plus `k` (candidate count) and `tier` on `phase2_allocated` |
| `v3_plansearch` | PlanSearch step (`plansearch`, `plansearch_done`, `plansearch_error`) | `stage`, `detail`, `plans` (int), `candidates` (int, on `_done`), `tokens` (int, on `_done`) |
| `v3_divsampling` | DivSampling step (`divsampling`, `divsampling_done`, `divsampling_error`) | `stage`, `detail`, `slots` (int), `total` (int, on `_done`) |
| `v3_sandbox` | Per-candidate sandbox test (`sandbox_test`, `sandbox_pass`, `sandbox_fail`, `sandbox_done`) | `stage`, `detail`, `index` (int), `elapsed_ms` (int), `energy` (float, on `_pass`), `stderr` (string, first 120 chars on `_fail`), `passed` / `total` (on `_done`) |
| `v3_select` | Candidate selection (`s_star`, `s_star_winner`, `selected`) | `stage`, `detail`, `index` (int), `energy` (float) |
| `v3_repair` | Phase 3 repair strategy (`phase3`, `pr_cot*`, `refinement*`, `derivation*`, `fallback`) | `stage`, `detail`, `strategy` (string: `pr_cot` / `refinement` / `derivation`), `failing` (int), `iterations` (int, on `refinement_pass`), `tokens` (int, on `_pass`) |
| `v3_probe` | Probe phase events (`probe`, `probe_light`, `probe_retry`, `probe_failed`, `probe_scored`, `probe_sandbox`, `probe_pass`) | `stage`, `detail` |
| `v3_self_test` | Self-test generation/verify events (`self_test_gen`, `self_test_done`, `self_test_error`, `self_test_skip`, `self_test_verify`) | `stage`, `detail` |
| `v3_plan` | Plan-pipeline progress (`plan_start`, `plan_candidate`, `plan_candidate_scored`, `plan_candidate_unparseable`, `plan_candidate_error`, `plan_selected`, `plan_failed`). Per-token `token`/`llm_start`/`llm_end` events are filtered out at the proxy. | `stage`, `detail`, `index` (int, per-candidate), `score` (float, on `_scored`/`_selected`), `revision` (int, set when fired during a revise) |
| `plan_loaded` | A winning plan has been generated. Fires once after initial generation and again after each revision. Carries the full step list. | `steps` (array of `{id, action, target, why}`), `verify_step` (string id), `rationale` (string), `winning_score` (float), `revision` (int — 0 for initial plan, 1+ for revisions) |
| `plan_adherence` | Emitted after each tool call, indicating whether the call satisfied an outstanding plan step. Off-plan calls (`matched=false`) accumulate into the off-streak counter that drives auto-revise. | On match: `matched=true`, `step_index`, `step_id`, `step_action`, `satisfied` (steps satisfied so far), `total`. On miss: `matched=false`, `tool`, `off_streak` (consecutive off-plan calls), `satisfied`, `total`. |
| `plan_revise` | The off-streak crossed `planAutoReviseThreshold` (3) — a fresh plan is being generated. The next `plan_loaded` (with `revision>0`) supersedes the prior plan; `Satisfied` flags reset. | `reason` (string), `revision` (int, 1-indexed) |
| `done` | Agent loop ended cleanly | `summary` (string — empty for a `text`-shaped turn) |
| `error` | LLM/parse/turn-cap error | `error` (string) |

After the final event the server writes the SSE sentinel `data: [DONE]\n\n` and closes the response.

> **Why so many event types?** A client that only cares about the user-facing chat surface needs `text`, `tool_call`, `tool_result`, `done`, and `error`. The `llm_*` and `v3_*` events exist so the TUI can show what the model is *doing* during the 5–60 s gap between user input and final reply (encoding the prompt, streaming a tool call, V3 grinding through a probe → PlanSearch → DivSampling → sandbox cycle). Drop them if your UI doesn't care.

#### Stream parsing example (Python)

```python
import json, requests

with requests.post(
    "http://localhost:8090/v1/agent",
    json={"message": "fix the bug in app.py", "session_id": "client-1"},
    stream=True,
    timeout=(10, None),  # connect timeout 10s, no read timeout — turns can be long
) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        body = line[6:]
        if body == "[DONE]":
            break
        evt = json.loads(body)
        t, d = evt["type"], evt["data"]
        if t == "tool_call":
            print(f"→ {d['name']}({d.get('args', {})})")
        elif t == "tool_result":
            print(f"  {'OK' if d['success'] else 'FAIL'} {d.get('elapsed', '')}")
        elif t == "text":
            print(d["content"])
        elif t == "done":
            print(f"✓ {d.get('summary', '')}")
        elif t == "error":
            print(f"✗ {d['error']}")
```

That is the minimum viable client. Full streaming chat with token-level rendering is roughly +20 lines (buffer `llm_token` deltas, flush on `llm_call_end`).

---

### POST /cancel

Abort an in-flight `/v1/agent` turn. Idempotent — repeated calls for the same session return 404.

**Request:**
```json
{"session_id": "tui-7f3a2c1b"}
```

**Response (200):**
```json
{"cancelled": true}
```

**Response (404):**
```json
{"cancelled": false}
```

When cancelled, the agent loop exits via `context.Canceled`, the SSE stream emits its trailing `[DONE]`, and the connection closes cleanly. Any in-flight LLM call to llama-server is also aborted via the cascading request context.

The TUI uses this on `Esc` mid-turn — see [CLI.md → Cancelling a turn](CLI.md).

---

### GET /events

Subscribe to the **global typed-envelope broker** (PC-061). Unlike `/v1/agent` (per-request stream of one turn), `/events` is a long-lived pub/sub feed of structured envelopes from across the proxy: agent loop boundaries, tool calls, V3 stage transitions, metrics. Multiple clients can subscribe simultaneously; slow consumers drop events rather than blocking producers.

**Envelope wire format** (matches `atlas/cli/events.py` exactly):
```json
{
  "event_id": "evt_a1b2c3d4",
  "timestamp": 1714617823.412,
  "type": "stage_start",
  "stage": "llm",
  "payload": {"turn": 1, "messages": 3},
  "parent_id": "evt_...",
  "duration_ms": 245
}
```

**Event types:** `stage_start`, `stage_end`, `tool_call`, `tool_result`, `metric`, `error`, `done`.

**Transport:** SSE. Each line is `data: <json>\n\n`. The server sends `: connected\n\n` immediately on subscribe and a `: heartbeat\n\n` comment every 15 s during quiet stretches to keep proxies/load-balancers from idling out the connection.

**Example:**
```bash
curl -N http://localhost:8090/events
```

Use `/events` when you want a global observability feed (a TUI pipeline pane, a metrics scraper, a debug log viewer). Use `/v1/agent` when you want to drive a specific user turn.

---

### POST /v1/chat/completions (passthrough)

OpenAI-compatible chat completions. Predates `/v1/agent` and is kept for SDK compatibility. The proxy passes these requests through to llama-server unchanged — **no agent loop, no tool calls, no V3 pipeline runs on this endpoint**. The response shape and streaming format is whatever llama-server returns natively.

**For agent turns and tool calls, use `/v1/agent`.** It exposes the full structured event stream (tool calls, V3 progress, permission requests) — all features that used to live on `/v1/chat/completions` were moved there when the legacy Aider-format wrapping was retired.

**Request:**
```json
{
  "model": "Qwen3.5-9B-Q6_K",
  "messages": [
    {"role": "user", "content": "Create a Python hello world script"}
  ],
  "max_tokens": 32768,
  "temperature": 0.3,
  "stream": true
}
```

**Response (SSE stream when `stream: true`):**
```
data: {"id":"atlas-verify","object":"chat.completion.chunk","choices":[{"delta":{"content":"[Turn 1/30] writing hello.py..."}}]}
data: {"id":"atlas-verify","object":"chat.completion.chunk","choices":[{"delta":{"content":"hello.py\n```python\nprint('hello world')\n```"}}]}
data: [DONE]
```

**Response (non-streaming when `stream: false`):**
```json
{
  "id": "atlas-verify",
  "object": "chat.completion",
  "model": "Qwen3.5-9B-Q6_K",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 150, "completion_tokens": 200, "total_tokens": 350},
  "atlas_route": "standard",
  "atlas_gx_score": 0.85,
  "atlas_verdict": "likely_correct",
  "atlas_sandbox_passed": true,
  "atlas_repair_attempt": 0
}
```

The `atlas_*` fields are ATLAS-specific metadata attached to non-streaming responses, omitted when empty.

> **Note:** `/chat/completions` and `/models` (no `/v1/` prefix) are aliases. Any unmatched path is proxied directly to llama-server.

---

### Tools available to the agent loop

Defined in `proxy/tools.go`. Used by the model when responding `{"type":"tool_call","name":"<tool>","args":{...}}`.

| Tool | Purpose |
|------|---------|
| `read_file` | Read a file and return its contents with line numbers |
| `write_file` | Create a new file or replace its full contents (rejected for existing files >100 lines — use `edit_file` instead) |
| `edit_file` | Apply targeted `old_str`/`new_str` edits to an existing file. May route through V3 verification when the file is build-checkable. |
| `delete_file` | Remove a file from the workspace |
| `search_files` | Regex search inside file **contents**. Returns matching lines with file paths and line numbers |
| `find_file` | Regex search by file **name** or relative path. Use to check whether a file exists. (PC-028) |
| `list_directory` | List files and subdirectories at a given path |
| `run_command` | Execute a shell command via bash. Runs inside the atlas-proxy container, which ships with python3, py3-pip, nodejs/npm, gcc, g++, make, and git. (PC-032 will route this through the sandbox container.) |
| `plan_tasks` | Decompose work into parallel tasks with dependencies |

---

### GET /v1/models

OpenAI-compatible model list.

```bash
curl http://localhost:8090/v1/models
```

### GET /health

```bash
curl http://localhost:8090/health
```

```json
{
  "status": "ok",
  "inference": true,
  "lens": true,
  "sandbox": true,
  "port": "8090",
  "stats": {"requests": 42, "repairs": 3, "sandbox_passes": 38, "sandbox_fails": 4}
}
```

---

## V3 Pipeline Service (Port 8070)

Runs the full V3 code generation pipeline: probe, PlanSearch, DivSampling, Budget Forcing, Lens scoring, sandbox testing, and Phase 3 repair. Normally invoked indirectly through the proxy's `edit_file`/`write_file` tools, but the HTTP surface is stable for direct use.

### POST /v3/generate

Run the V3 pipeline for a file generation task. Streams progress events as SSE.

**Request:**
```json
{
  "file_path": "app/page.tsx",
  "baseline_code": "export default function Page() { ... }",
  "project_context": {"package.json": "{...}", "tsconfig.json": "{...}"},
  "framework": "nextjs",
  "build_command": "npx next build",
  "constraints": ["Must use Tailwind CSS", "Must be a client component"],
  "tier": 2,
  "working_dir": "/path/to/project"
}
```

All fields are optional except the task itself. `tier` defaults to 2.

**Response (SSE stream):**
```
data: {"stage": "probe", "detail": "Generating probe candidate..."}
data: {"stage": "probe_scored", "detail": "C(x)=0.72 norm=0.68"}
data: {"stage": "phase2_allocated", "detail": "k=3 tier=standard", "data": {"k": 3, "tier": "standard"}}
data: {"stage": "plansearch", "detail": "Generating 3 plans...", "data": {"plans": 3}}
data: {"stage": "sandbox_test", "detail": "Testing 3 candidates...", "data": {"candidates": 3}}
data: {"stage": "sandbox_pass", "detail": "Candidate 1 passed", "data": {"index": 1, "elapsed_ms": 420, "energy": 0.34}}
data: {"stage": "sandbox_done", "detail": "1/3 passed", "data": {"passed": 1, "total": 3}}
data: {"stage": "llm_start", "detail": "call #4", "data": {"call": 4, "max_tokens": 4096, "temperature": 0.7}}
data: {"stage": "token", "detail": "def "}
data: {"stage": "token", "detail": "merge_sort("}
data: {"stage": "llm_end", "detail": "245 tok · 1820ms", "data": {"call": 4, "tokens": 245, "elapsed_ms": 1820}}

event: result
data: {"code": "...", "passed": true, "phase_solved": "phase1", "candidates_tested": 3, "winning_score": 0.85, "total_tokens": 12500, "total_time_ms": 4200.0}

data: [DONE]
```

Each progress event has the shape `{"stage": "<name>", "detail": "<human-readable>", "data": {...}}`. The `data` object carries structured fields specific to that stage (counts, indices, timings, strategy labels) — the proxy bridge fans these out into the dedicated `v3_*` events on `/v1/agent`. Older stages without `data` enrichment still emit `stage` + `detail` only and continue to flow through `v3_progress`.

The proxy's `tools.go` bridge translates these into `v3_progress` / `v3_token` / `v3_llm_start` / `v3_llm_end` events on the `/v1/agent` stream — direct callers see the raw V3 stages.

<details>
<summary><b>All SSE stage values</b></summary>

The pipeline emits stages as it progresses. Not all stages appear in every run — the pipeline exits early when a candidate passes.

| Phase | Stages |
|-------|--------|
| **Probe (Phase 0)** | `probe`, `probe_light`, `probe_error`, `probe_retry`, `probe_failed`, `self_test_gen`, `self_test_done`, `self_test_error`, `self_test_verify`, `probe_scored`, `probe_sandbox`, `probe_pass` |
| **Generation (Phase 1)** | `phase1`, `phase2` (allocation), `phase2_allocated`, `plansearch`, `plansearch_done`, `plansearch_error`, `divsampling`, `divsampling_done`, `divsampling_error` |
| **Testing (Phase 2)** | `sandbox_test`, `sandbox_pass`, `sandbox_done`, `s_star`, `s_star_winner`, `s_star_error`, `selected` |
| **Repair (Phase 3)** | `phase3`, `pr_cot`, `pr_cot_pass`, `pr_cot_failed`, `pr_cot_error`, `refinement`, `refinement_pass`, `refinement_failed`, `refinement_error`, `derivation`, `derivation_pass`, `derivation_failed`, `derivation_error`, `fallback` |
| **LLM streaming** | `llm_start`, `token`, `llm_end` (one bracketed group per internal LLM call — planner, candidate generation, repair, etc.) |

</details>

### POST /v3/run

Simplified endpoint for running the pipeline on a problem description (used by the CLI).

**Request:**
```json
{
  "problem": "Write a function that finds the longest palindromic substring",
  "task_id": "cli",
  "stream": true,
  "files": {"main.py": "# existing code..."}
}
```

**Response:** Same SSE format as `/v3/generate`.

### POST /v3/plan

Generates a step-by-step plan for a coding task using diverse LLM sampling and heuristic scoring. Used by the proxy's agent loop to seed each turn with explicit step guidance — see [PLAN_MODE.md](PLAN_MODE.md) for the consumer side.

**Request:**
```json
{
  "user_message": "fix the broken index.html template",
  "working_dir": "/workspace",
  "project_context": {
    "app.py": "from flask import Flask...",
    "templates/index.html": "<!DOCTYPE html>..."
  },
  "n_candidates": 3
}
```

| Field | Required | Notes |
|---|---|---|
| `user_message` | yes | The original user request the planner must address. |
| `working_dir` | optional | Used in the planner prompt for path-context. Defaults to `/workspace`. |
| `project_context` | optional | Map of `relative_path → file_content`. Files are truncated to ~200 chars in the planner prompt. |
| `n_candidates` | optional | How many candidate plans to sample. Defaults to 3. Each is sampled at a different temperature (0.3 / 0.5 / 0.7) for diversity. |

**Response:** SSE stream of `{stage, detail, data}` events ending with `event: result\ndata: <plan-json>` and `data: [DONE]`.

Plan-pipeline stages: `plan_start`, `plan_candidate`, `plan_candidate_unparseable`, `plan_candidate_error`, `plan_candidate_scored`, `plan_selected`, `plan_failed`. Per-candidate token streaming flows under `token` / `llm_start` / `llm_end` (filtered out by the proxy bridge before reaching `/v1/agent`).

The final `event: result` payload has the shape:

```json
{
  "steps": [
    {"id": "s1", "action": "read_file", "target": "templates/index.html", "why": "inspect"},
    {"id": "s2", "action": "edit_file", "target": "templates/index.html", "why": "fix structural HTML"},
    {"id": "s3", "action": "run_command", "target": "curl http://localhost:5000/", "why": "verify"}
  ],
  "verify_step": "s3",
  "rationale": "investigate, change, verify.",
  "candidates_tested": 3,
  "winning_score": 1.0,
  "winning_index": 0,
  "reasons": ["step count 3 in range", "verify_step=s3", ...]
}
```

If all candidates fail to parse, the endpoint returns a single-step fallback (`{steps:[{action:"investigate the request and act"}], verify_step:null, winning_index:-1}`) rather than 5xx — callers can detect the fallback by `winning_index<0` or `reasons` containing "all candidates failed".

### GET /health

```bash
curl http://localhost:8070/health
# {"status": "ok", "service": "v3-pipeline"}
```

---

## Geometric Lens (Port 8099)

Energy-based code scoring using C(x) cost field and G(x) quality prediction. Also serves as the RAG API for project indexing and retrieval.

> **Internal port note:** The container runs uvicorn on port 8001. Docker Compose maps this to 8099 on the host.

### POST /internal/lens/gx-score

Score code using combined C(x) + G(x) energy. Single embedding extraction serves both models.

**Request:**
```json
{"text": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    ..."}
```

**Response:**
```json
{
  "cx_energy": 5.2,
  "cx_normalized": 0.32,
  "gx_score": 0.85,
  "verdict": "likely_correct",
  "gx_available": true,
  "enabled": true,
  "latency_ms": 26.4
}
```

| Field | Type | Description |
|-------|------|-------------|
| `cx_energy` | float | Raw cost field energy (lower = more likely correct) |
| `cx_normalized` | float | 0–1 normalized energy |
| `gx_score` | float | 0–1 probability of correctness (G(x) model) |
| `verdict` | string | `"likely_correct"`, `"uncertain"`, or `"likely_incorrect"` |
| `gx_available` | bool | Whether the G(x) model was loaded |
| `enabled` | bool | Whether Geometric Lens is enabled |
| `latency_ms` | float | Execution time in milliseconds |

When Lens is disabled, returns `enabled: false` with neutral defaults (`cx_energy: 0.0`, `gx_score: 0.5`).

**Example:**
```bash
curl http://localhost:8099/internal/lens/gx-score \
  -H "Content-Type: application/json" \
  -d '{"text": "print(\"hello world\")"}'
```

### GET /health

```bash
curl http://localhost:8099/health
# {"status": "healthy", "service": "geometric-lens"}
```

<details>
<summary><b>Additional internal endpoints</b></summary>

These are used internally by other ATLAS services. They are stable but not part of the public API.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/projects/sync` | POST | Sync/index a project codebase |
| `/v1/projects/{id}/status` | GET | Get project index status |
| `/v1/projects` | GET | List indexed projects |
| `/v1/projects/{id}` | DELETE | Delete a project index |
| `/v1/chat/completions` | POST | RAG-augmented chat completions |
| `/v1/models` | GET | List available models |
| `/v1/tasks/submit` | POST | Submit async task |
| `/v1/tasks/{id}/status` | GET | Get task status |
| `/v1/queue/stats` | GET | Task queue statistics |
| `/v1/patterns/write` | POST | Write pattern data |
| `/internal/cache/stats` | GET | Cache statistics |
| `/internal/cache/flush` | POST | Flush cache |
| `/internal/cache/consolidate` | POST | Consolidate cache entries |
| `/internal/router/stats` | GET | Confidence router statistics |
| `/internal/router/reset` | POST | Reset router posteriors |
| `/internal/router/feedback` | POST | Record routing feedback |
| `/internal/lens/stats` | GET | Lens model statistics |
| `/internal/lens/evaluate` | GET/POST | Evaluate text through Lens |
| `/internal/lens/score-text` | POST | Score text (C(x) only) |
| `/internal/lens/retrain` | POST | Retrain cost field model |
| `/internal/lens/reload` | POST | Reload model weights |
| `/internal/lens/correctability` | POST | Correctability evaluation |
| `/internal/sandbox/analyze` | POST | Sandbox result analysis |

</details>

---

## Sandbox (Port 30820 → container 8020)

Isolated code execution with compilation, testing, and linting support.

### POST /execute

Execute code in an isolated environment.

**Request:**
```json
{
  "code": "from utils import greet\nprint(greet('world'))",
  "language": "python",
  "test_code": null,
  "requirements": null,
  "timeout": 30,
  "files": {"utils.py": "def greet(name): return f'hi {name}'"}
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `code` | string | (required) | Code to execute |
| `language` | string | `"python"` | Target language (see supported list below) |
| `test_code` | string | null | Optional test code (e.g. pytest assertions) |
| `requirements` | string[] | null | Python packages to pip install before execution |
| `timeout` | int | 30 | Max execution time in seconds (capped at 60) |
| `files` | object | null | Map of `relative-path → file-content` written into the workspace before execution. Use to ship multi-file project context (e.g. modules the candidate imports). Path traversal (`..`, absolute paths) is rejected. See PC-046. |

**Response:**
```json
{
  "success": true,
  "compile_success": true,
  "tests_run": 1,
  "tests_passed": 1,
  "lint_score": 8.5,
  "stdout": "hello from sandbox\n",
  "stderr": "",
  "error_type": null,
  "error_message": null,
  "execution_time_ms": 45
}
```

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Overall pass/fail |
| `compile_success` | bool | Whether compilation succeeded (always true for interpreted languages) |
| `tests_run` | int | Number of tests executed |
| `tests_passed` | int | Number of tests that passed |
| `lint_score` | float? | Pylint score 0–10 (Python only, null for other languages) |
| `stdout` | string | Stdout output (truncated to last 4000 chars) |
| `stderr` | string | Stderr output (truncated to last 2000 chars) |
| `error_type` | string? | Error classification (e.g. `SyntaxError`, `CompileError`, `Timeout`, `ImportError`) |
| `error_message` | string? | First 500 chars of error details |
| `execution_time_ms` | int | Execution time in milliseconds |

### POST /syntax-check

Check syntax without executing code.

**Request:**
```json
{
  "code": "def foo(:\n    pass",
  "language": "python",
  "filename": "main.py"
}
```

**Response:**
```json
{
  "valid": false,
  "errors": ["SyntaxError: invalid syntax (line 1)"],
  "language": "python",
  "check_time_ms": 12
}
```

### GET /languages

List supported languages with installed runtime versions.

```bash
curl http://localhost:30820/languages
```

```json
{
  "languages": {
    "python": "Python 3.11.2",
    "javascript": "v20.11.0",
    "typescript": "5.3.3",
    "go": "go1.22.0",
    "rust": "rustc 1.77.0",
    "c": "gcc 13.2.0",
    "cpp": "g++ 13.2.0",
    "bash": "GNU bash 5.2.21"
  }
}
```

**Supported languages:** `python` (aliases: `py`, `python3`), `javascript` (`js`, `node`), `typescript` (`ts`), `go` (`golang`), `rust` (`rs`), `c`, `cpp` (`c++`), `bash` (`sh`, `shell`)

### GET /health

```bash
curl http://localhost:30820/health
# {"status": "healthy"}
```

---

## llama-server (Port 8080)

Standard llama.cpp server API. See [llama.cpp documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

### POST /v1/chat/completions

OpenAI-compatible chat completions with `response_format` support for grammar-constrained JSON output.

**Example:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q6_K",
    "messages": [{"role":"user","content":"/nothink\nSay hello"}],
    "max_tokens": 50,
    "response_format": {"type": "json_object"}
  }'
```

### POST /completion

Raw completion endpoint (no chat template). Used internally by the benchmark runner.

### POST /embedding

Generate embeddings for input text. Used by Geometric Lens for C(x)/G(x) scoring.

### GET /health

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

> llama-server exposes many more endpoints. See the [llama.cpp server docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for the full API reference.

---

## Building a non-TUI client

A minimal client needs three things:

1. **POST `/v1/agent`** with `{message, working_dir, mode, session_id}` and parse the SSE stream. See the Python example above.
2. **POST `/cancel`** with `{session_id}` when the user wants to abort.
3. *(Optional)* **GET `/events`** in a background goroutine/thread for the global typed-envelope feed if you want a pipeline-progress sidebar.

The TUI ([atlas tui](CLI.md)) is a Go reference implementation (~3 kloc) — its `model.go` shows how to handle every event type, and `panes.go` shows one approach to rendering them. Browse `tui/` in the repo for a complete worked example.

PC-063 tracks producing a fully-worked web client recipe and an OpenAPI spec generated from `tools.go`. Until then, this document and the TUI source are the canonical reference.
