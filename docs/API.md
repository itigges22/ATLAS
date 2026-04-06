# ATLAS API Reference

API endpoints for each ATLAS service. All services communicate over HTTP/JSON. Streaming endpoints use Server-Sent Events (SSE).

> **Ports listed are defaults.** All are configurable via environment variables (see [CONFIGURATION.md](CONFIGURATION.md)). Docker Compose maps container-internal ports to host ports — the ports below are what you hit from the host.

---

## atlas-proxy (Port 8090)

The main entry point. Wraps llama-server with an agent loop, grammar-constrained tool calls, Lens scoring, and sandbox verification. This is what Aider connects to.

### POST /v1/chat/completions

OpenAI-compatible chat completions. When `ATLAS_AGENT_LOOP=1` (default), the proxy runs an internal agent loop with structured tool calls instead of forwarding directly to the LLM.

**Request:**
```json
{
  "model": "Qwen3.5-9B-Q6_K",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Create a Python hello world script"}
  ],
  "max_tokens": 32768,
  "temperature": 0.3,
  "stream": true,
  "stop": []
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
  "created": 1712345678,
  "model": "Qwen3.5-9B-Q6_K",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 200,
    "total_tokens": 350
  },
  "atlas_route": "standard",
  "atlas_gx_score": 0.85,
  "atlas_verdict": "likely_correct",
  "atlas_sandbox_passed": true,
  "atlas_repair_attempt": 0
}
```

The `atlas_*` fields are ATLAS-specific metadata attached to non-streaming responses. They are omitted when empty/zero.

**Example:**
```bash
curl -N http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-9B-Q6_K","messages":[{"role":"user","content":"hi"}],"max_tokens":100,"stream":true}'
```

### POST /v1/agent

Tool-based agent endpoint. Sends a message and receives a stream of tool calls and results.

**Request:**
```json
{
  "message": "Create a snake game in Python",
  "working_dir": "/path/to/project",
  "mode": "default"
}
```

`mode` options: `"default"`, `"accept-edits"`, `"yolo"`

**Response:** SSE stream of tool call events, terminated by `data: [DONE]`.

### GET /v1/models

Returns available models (OpenAI-compatible).

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
  "stats": {
    "requests": 42,
    "repairs": 3,
    "sandbox_passes": 38,
    "sandbox_fails": 4
  }
}
```

> **Note:** `/chat/completions` and `/models` (without `/v1/` prefix) are aliases for their `/v1/` counterparts. Any unmatched path is proxied directly to llama-server.

---

## V3 Pipeline Service (Port 8070)

Runs the full V3 code generation pipeline: probe, PlanSearch, DivSampling, Budget Forcing, Lens scoring, sandbox testing, and Phase 3 repair.

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
data: {"stage": "probe_sandbox", "detail": "Testing probe..."}
data: {"stage": "plansearch", "detail": "Generating 3 plans..."}
data: {"stage": "divsampling", "detail": "Generating candidates..."}
data: {"stage": "sandbox_test", "detail": "Testing 3 candidates..."}
data: {"stage": "sandbox_pass", "detail": "Candidate 1 passed"}

event: result
data: {"code": "...", "passed": true, "phase_solved": "phase1", "candidates_tested": 3, "winning_score": 0.85, "total_tokens": 12500, "total_time_ms": 4200.0}

data: [DONE]
```

<details>
<summary><b>All SSE stage values</b></summary>

The pipeline emits stages as it progresses. Not all stages appear in every run — the pipeline exits early when a candidate passes.

| Phase | Stages |
|-------|--------|
| **Probe (Phase 0)** | `probe`, `probe_light`, `probe_error`, `probe_retry`, `probe_failed`, `self_test_gen`, `self_test_done`, `self_test_error`, `self_test_verify`, `probe_scored`, `probe_sandbox`, `probe_pass` |
| **Generation (Phase 1)** | `phase1`, `phase2` (allocation), `phase2_allocated`, `plansearch`, `plansearch_done`, `plansearch_error`, `divsampling`, `divsampling_done`, `divsampling_error` |
| **Testing (Phase 2)** | `sandbox_test`, `sandbox_pass`, `sandbox_done`, `s_star`, `s_star_winner`, `s_star_error`, `selected` |
| **Repair (Phase 3)** | `phase3`, `pr_cot`, `pr_cot_pass`, `pr_cot_failed`, `pr_cot_error`, `refinement`, `refinement_pass`, `refinement_failed`, `refinement_error`, `derivation`, `derivation_pass`, `derivation_failed`, `derivation_error`, `fallback` |

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
  "code": "print('hello from sandbox')",
  "language": "python",
  "test_code": null,
  "requirements": null,
  "timeout": 30
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `code` | string | (required) | Code to execute |
| `language` | string | `"python"` | Target language (see supported list below) |
| `test_code` | string | null | Optional test code (e.g. pytest assertions) |
| `requirements` | string[] | null | Python packages to pip install before execution |
| `timeout` | int | 30 | Max execution time in seconds (capped at 60) |

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
