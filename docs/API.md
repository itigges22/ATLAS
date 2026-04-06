# ATLAS API Reference

API endpoints for each ATLAS service.

---

## atlas-proxy (Port 8090)

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint. This is what Aider connects to.

When `ATLAS_AGENT_LOOP=1` is set, the proxy runs an internal agent loop with structured tool calls instead of forwarding directly to the LLM.

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
  "stream": true
}
```

**Response (SSE stream):**
```
data: {"id":"atlas-verify","object":"chat.completion.chunk","choices":[{"delta":{"content":"[Turn 1/30] writing hello.py..."}}]}

data: {"id":"atlas-verify","object":"chat.completion.chunk","choices":[{"delta":{"content":"hello.py\n```python\nprint('hello world')\n```"}}]}

data: [DONE]
```

**Example:**
```bash
curl -N http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-9B-Q6_K","messages":[{"role":"user","content":"hi"}],"max_tokens":100,"stream":true}'
```

### GET /health

```bash
curl http://localhost:8090/health
# {"inference":true,"lens":true,"port":"8090","sandbox":true,"status":"ok"}
```

---

## V3 Pipeline Service (Port 8070)

### POST /v3/generate

Run the full V3 pipeline for a file generation task. Streams progress events as SSE.

**Request:**
```json
{
  "file_path": "app/page.tsx",
  "baseline_code": "export default function Page() { ... }",
  "project_context": {"package.json": "{...}", "tsconfig.json": "{...}"},
  "framework": "nextjs",
  "build_command": "npx next build",
  "constraints": ["Must use Tailwind CSS", "Must be a client component"],
  "tier": 2
}
```

**Response (SSE stream):**
```
data: {"stage": "probe", "detail": "Generating probe candidate..."}
data: {"stage": "probe_scored", "detail": "C(x)=0.72 norm=0.68"}
data: {"stage": "plansearch", "detail": "Generating 3 plans..."}
data: {"stage": "sandbox_test", "detail": "Testing 3 candidates..."}
data: {"stage": "sandbox_pass", "detail": "Candidate 1 passed"}

event: result
data: {"code": "...", "passed": true, "phase_solved": "phase1", "candidates_tested": 3}

data: [DONE]
```

### GET /health

```bash
curl http://localhost:8070/health
# {"status": "ok", "service": "v3-pipeline"}
```

---

## Geometric Lens (Port 8099)

### GET /health

```bash
curl http://localhost:8099/health
# {"status": "healthy", "service": "geometric-lens"}
```

### POST /internal/lens/gx-score

Score code using C(x)/G(x) energy.

**Request:**
```json
{"text": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    ..."}
```

**Response:**
```json
{
  "cx_energy": 5.2,
  "gx_score": 0.85,
  "verdict": "likely_correct",
  "enabled": true,
  "latency_ms": 26.4
}
```

**Example:**
```bash
curl http://localhost:8099/internal/lens/gx-score \
  -H "Content-Type: application/json" \
  -d '{"text": "print(\"hello world\")"}'
```

---

## Sandbox (Port 8020)

### POST /execute

Execute code in an isolated environment.

**Request:**
```json
{
  "code": "print('hello from sandbox')",
  "language": "python",
  "timeout": 10
}
```

**Response:**
```json
{
  "success": true,
  "stdout": "hello from sandbox\n",
  "stderr": "",
  "execution_ms": 45.2
}
```

Supported languages: `python`, `javascript`, `typescript`, `go`, `rust`, `c`, `cpp`, `bash`

### GET /health

```bash
curl http://localhost:8020/health
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

### GET /health

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```
