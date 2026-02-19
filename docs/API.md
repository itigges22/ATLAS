# ATLAS API Reference

The ATLAS RAG API runs on port 8001 (K3s NodePort 31144) and provides OpenAI-compatible chat completions with RAG enhancement, project management, and internal monitoring endpoints.

## Authentication

All `/v1/` endpoints require a Bearer token in the `Authorization` header. The API validates keys against the API Portal service.

```bash
-H "Authorization: Bearer sk-your-api-key"
```

Keys are cached in-memory for 60 seconds after successful validation. Internal endpoints (`/internal/` and `/health`) do not require authentication.

---

## Core Endpoints

### Health Check

```
GET /health
```

Returns service health status. No authentication required.

```bash
curl http://localhost:31144/health
```

```json
{"status": "healthy", "service": "rag-api"}
```

---

### Chat Completions

```
POST /v1/chat/completions
```

OpenAI-compatible chat completions. When `project_id` is provided, the request is enhanced with RAG context from the synced project (PageIndex retrieval + Pattern Cache). When omitted, the request is forwarded directly to llama-server.

Supports streaming (`"stream": true`) via SSE.

**Request body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | | Model name (passed to llama-server) |
| `messages` | array | yes | | Chat messages (`role` + `content`) |
| `project_id` | string | no | null | Project ID for RAG enhancement |
| `max_tokens` | int | no | 16384 | Maximum tokens to generate |
| `temperature` | float | no | null | Sampling temperature |
| `top_p` | float | no | null | Nucleus sampling parameter |
| `stream` | bool | no | false | Enable SSE streaming |
| `tools` | array | no | null | Tool definitions for function calling |

```bash
curl -X POST http://localhost:31144/v1/chat/completions \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b",
    "messages": [{"role": "user", "content": "Write a Python fibonacci function"}],
    "max_tokens": 4096
  }'
```

---

### Project Sync

```
POST /v1/projects/sync
```

Syncs a project's codebase for RAG indexing. Builds a PageIndex (AST tree + BM25 index) and generates LLM summaries for tree nodes. Skips re-indexing if the project hash matches a previous sync.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_name` | string | yes | Human-readable project name |
| `project_hash` | string | yes | Content hash for change detection |
| `files` | array | yes | List of `{path, content}` objects |
| `metadata` | object | no | Optional project metadata |

```bash
curl -X POST http://localhost:31144/v1/projects/sync \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "my-project",
    "project_hash": "abc123",
    "files": [
      {"path": "main.py", "content": "def hello(): ..."},
      {"path": "utils.py", "content": "import os ..."}
    ]
  }'
```

```json
{
  "project_id": "proj_abc123",
  "status": "synced",
  "stats": {"files_indexed": 2, "chunks_created": 15, "loc_indexed": 120},
  "sync_time_ms": 3400
}
```

---

### List Projects

```
GET /v1/projects
```

Returns all synced projects with their status and last sync time.

```bash
curl http://localhost:31144/v1/projects \
  -H "Authorization: Bearer sk-your-key"
```

---

### Project Status

```
GET /v1/projects/{project_id}/status
```

Returns detailed status for a specific project including file count, chunk count, LOC, and expiration time.

```bash
curl http://localhost:31144/v1/projects/proj_abc123/status \
  -H "Authorization: Bearer sk-your-key"
```

---

### Delete Project

```
DELETE /v1/projects/{project_id}
```

Deletes a project and all its indexed data (PageIndex, BM25 index, in-memory cache).

```bash
curl -X DELETE http://localhost:31144/v1/projects/proj_abc123 \
  -H "Authorization: Bearer sk-your-key"
```

---

### List Models

```
GET /v1/models
```

Proxies to llama-server's model list endpoint. Returns available models.

```bash
curl http://localhost:31144/v1/models \
  -H "Authorization: Bearer sk-your-key"
```

---

## Task Queue Endpoints

### Submit Task

```
POST /v1/tasks/submit
```

Submits a task for asynchronous processing via the Redis task queue.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | (required) | Task prompt |
| `type` | string | `code_generation` | Task type |
| `priority` | string | `p1` | Priority: `p0` (interactive), `p1` (fire-forget), `p2` (batch) |
| `project_id` | string | null | Optional project context |
| `max_attempts` | int | 5 | Maximum retry attempts |
| `require_tests_pass` | bool | true | Whether tests must pass |
| `test_code` | string | null | Test code to validate against |

```bash
curl -X POST http://localhost:31144/v1/tasks/submit \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a binary search function", "priority": "p1"}'
```

---

### Task Status

```
GET /v1/tasks/{task_id}/status
```

Returns current status of a submitted task including attempt count and result.

```bash
curl http://localhost:31144/v1/tasks/abc-123/status \
  -H "Authorization: Bearer sk-your-key"
```

---

### Queue Stats

```
GET /v1/queue/stats
```

Returns current queue depths by priority level.

```bash
curl http://localhost:31144/v1/queue/stats \
  -H "Authorization: Bearer sk-your-key"
```

---

## Internal Endpoints

These endpoints are used for monitoring and debugging. They do not require authentication.

### Pattern Cache Stats

```
GET /internal/cache/stats
```

Returns Pattern Cache statistics: size, hit rate, tier distribution, and top patterns by STM and LTM.

```bash
curl http://localhost:31144/internal/cache/stats
```

---

### Flush Pattern Cache

```
POST /internal/cache/flush
```

Clears the entire pattern cache and reloads seed patterns.

```bash
curl -X POST http://localhost:31144/internal/cache/flush
```

---

### Trigger Consolidation

```
POST /internal/cache/consolidate
```

Manually triggers STM to LTM pattern consolidation.

```bash
curl -X POST http://localhost:31144/internal/cache/consolidate
```

---

### Router Stats

```
GET /internal/router/stats
```

Returns Confidence Router statistics: Thompson Sampling state (alpha/beta per bin per route), aggregate route distribution, and difficulty histogram.

```bash
curl http://localhost:31144/internal/router/stats
```

---

### Reset Router

```
POST /internal/router/reset
```

Resets all Thompson Sampling state to uniform Beta(1,1) priors.

```bash
curl -X POST http://localhost:31144/internal/router/reset
```

---

### Geometric Lens Stats

```
GET /internal/lens/stats
```

Returns Geometric Lens status: whether models are loaded, parameter counts, and enabled state.

```bash
curl http://localhost:31144/internal/lens/stats
```

---

### Lens Evaluate

```
GET /internal/lens/evaluate?query=your+text
POST /internal/lens/evaluate  (body: {"query": "your text"})
```

Evaluates a query through the full Geometric Lens pipeline (embedding extraction, energy computation, optional correction). Returns energy before/after and normalized energy.

```bash
curl -X POST http://localhost:31144/internal/lens/evaluate \
  -H "Content-Type: application/json" \
  -d '{"query": "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"}'
```

```json
{
  "enabled": true,
  "energy_before": 3.21,
  "energy_after": 3.21,
  "energy_normalized": 0.12,
  "corrected": false
}
```

---

### Lens Score Text

```
POST /internal/lens/score-text
```

Scores a text string through the cost field C(x). Used by the benchmark runner's best-of-k pipeline to rank candidates.

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Text to score (typically "TASK: {prompt}\n\nSOLUTION: {code}") |

```bash
curl -X POST http://localhost:31144/internal/lens/score-text \
  -H "Content-Type: application/json" \
  -d '{"text": "def add(a, b): return a + b"}'
```

```json
{"energy": 4.12, "normalized": 0.08, "enabled": true}
```

---

### Lens Retrain

```
POST /internal/lens/retrain
```

Retrains the cost field C(x) on accumulated pass/fail embeddings. Used by the V2 benchmark runner between epochs for continuous learning.

| Field | Type | Description |
|-------|------|-------------|
| `training_data` | array | List of `{embedding: float[], label: int}` (1=pass, 0=fail) |
| `epochs` | int | Training epochs (default 50) |

```bash
curl -X POST http://localhost:31144/internal/lens/retrain \
  -H "Content-Type: application/json" \
  -d '{"training_data": [{"embedding": [...], "label": 1}], "epochs": 50}'
```

---

### Lens Reload

```
POST /internal/lens/reload
```

Hot-reloads Geometric Lens weights from disk after retraining, without restarting the service.

```bash
curl -X POST http://localhost:31144/internal/lens/reload
```
