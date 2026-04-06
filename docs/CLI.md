# ATLAS CLI Guide

The ATLAS CLI launches all required services, connects to the local LLM, and drops you into an interactive coding session powered by the V3 pipeline.

---

## Launching

```bash
cd /path/to/your/project
atlas
```

The `atlas` command automatically detects the deployment mode:

- **Docker Compose**: If a running Docker Compose stack is detected, ATLAS connects to the containerized services
- **Bare metal**: If no Docker stack is found, ATLAS starts llama-server, Geometric Lens, V3 Pipeline, and Proxy v2 as local processes

Both paths end the same way: Aider launches connected to the ATLAS proxy, with grammar-constrained tool calls and V3 pipeline integration.

### Usage Modes

```bash
atlas                          # Interactive session
atlas somefile.py              # Add file to chat on launch
atlas --message "fix the bug"  # Non-interactive (runs and exits)
echo "solve this" | atlas      # Pipe mode (stdin as problem)
```

Any arguments after `atlas` are passed through to Aider.

### Startup Sequence

```
    _  _____ _      _   ___
   /_\|_   _| |    /_\ / __|
  / _ \ | | | |__ / _ \\__ \
 /_/ \_\|_| |____/_/ \_\___/

  ✓ llama-server (port 8080)
  ✓ Geometric Lens (port 8099)
  ✓ V3 Pipeline (port 8070)
  ✓ Proxy v2 (port 8090)

[atlas] Stack ready. Launching aider...
  llama-server → V3 Pipeline → Proxy v2 → Aider
  Grammar: response_format:json_object | V3 on T2+ files
  Context: 32K | GPU: RTX 5060 Ti | ~51 tok/s
```

Each service is health-checked before proceeding:

| Service | Port | Health Timeout | Endpoint |
|---------|------|---------------|----------|
| llama-server | 8080 | 120s | GET /health |
| Geometric Lens | 8099 | 30s | GET /health |
| V3 Pipeline | 8070 | 15s | GET /health |
| Proxy v2 | 8090 | 30s | GET /health |

If a service is already running, ATLAS skips it and shows "(already running)". Logs for each service are written to `logs/` in the ATLAS directory.

---

## Streaming Output

Every tool call, V3 pipeline stage, and build verification is streamed in real-time:

```
[Turn 1/30] 📋 planning subtasks...
[Turn 2/30] ✎ writing package.json (T1, direct)
  ✓ wrote successfully (1.2ms)
[Turn 3/30] ✎ writing app.py (T2, V3 pipeline)
  ┌─ V3 Pipeline ─────────────────────────────
  │ Baseline: 134 lines, scoring...
  │ [probe] Generating probe candidate...
  │ [probe_scored] C(x)=0.72
  │ [plansearch] Generating 3 plans...
  │ [sandbox_test] Testing candidates...
  └──── V3 complete: phase1, 3 candidates
  ✓ wrote successfully
[Turn 4/30] 🔧 running: python -m py_compile app.py
  ✓ exit code 0 (0.3s)
[Turn 5/30] 📖 reading requirements.txt
  └─ 12 lines loaded

═══════════════════════════════════════════
✓ Complete (5 turns, 47s)
  Files created:  3 (package.json, app.py, requirements.txt)
  Commands run:   1
  V3 pipeline:    1 file enhanced
  Tokens:         8432
═══════════════════════════════════════════
```

### Status Icons

| Icon | Tool | Example |
|------|------|---------|
| ✎ | `write_file` | `[Turn 2/30] ✎ writing app.py (T1, direct)` |
| ✏️ | `edit_file` | `[Turn 3/30] ✏️ editing auth.py` |
| 🔧 | `run_command` | `[Turn 4/30] 🔧 running: npm test` |
| 📖 | `read_file` | `[Turn 5/30] 📖 reading config.json` |
| 🔍 | `search_files` | `[Turn 6/30] 🔍 searching "handleAuth"` |
| 📁 | `list_directory` | `[Turn 7/30] 📁 listing src/` |
| 📋 | `plan_tasks` | `[Turn 1/30] 📋 planning subtasks...` |

### Result Indicators

| Symbol | Meaning | Example |
|--------|---------|---------|
| ✓ | Success | `✓ wrote successfully (1.2ms)` |
| ✗ | Failure | `✗ failed: SyntaxError on line 12 (0.4s)` |
| └─ | Read result | `└─ 42 lines loaded` |

### Completion Summary

After the agent finishes, a summary box shows:
- **Files created/edited/deleted** with names (max 5 shown, then "+N more")
- **Commands run** count
- **V3 pipeline** count (only shown if V3 was used)
- **Tokens** total consumed

---

## Aider Commands

All standard Aider commands work through ATLAS:

| Command | Description |
|---------|-------------|
| `/add <file>` | Add a file to the chat context |
| `/drop <file>` | Remove a file from context |
| `/clear` | Clear chat history |
| `/tokens` | Show token usage |
| `/undo` | Undo last change |
| `/run <command>` | Run a shell command |
| `/help` | Show all commands |

---

## Python REPL (Alternative)

ATLAS also includes a standalone Python REPL that talks directly to services without Aider:

```bash
pip install -e .
atlas  # If no Docker Compose stack is detected, falls back to Python REPL
```

### REPL Commands

| Command | Description |
|---------|-------------|
| `/solve <file>` | Solve a coding problem from a file |
| `/bench [--tasks N] [--dataset NAME] [--strategy TYPE]` | Run benchmarks |
| `/status` | Check service health |
| `/help` | Show available commands |
| `/quit`, `/exit`, `/q` | Exit |

Plain text input (no `/` prefix) is treated as a coding problem and solved directly.

### REPL Health Checks

On startup, the REPL checks:
- **llama-server** at `ATLAS_INFERENCE_URL` (default: localhost:8080) — required, exits if unavailable
- **Geometric Lens** at `ATLAS_RAG_URL` (default: localhost:8099) — optional, warns "Lens unavailable — verification disabled"
- **Sandbox** at `ATLAS_SANDBOX_URL` (default: localhost:30820) — optional, warns "Sandbox unavailable — code testing disabled"

### Solve Pipeline

When you type a problem or use `/solve`:
1. Generate code from llama-server (streaming if interactive, batch if piped)
2. Extract code (handles `<think>` blocks, markdown fences, raw code)
3. Score via Geometric Lens (C(x)/G(x) energy + verdict)
4. Test via sandbox (if test cases available)
5. Display results with token count and elapsed time

Generation parameters: `max_tokens=8192`, `temperature=0.6`, `top_k=20`, `top_p=0.95`, `stop=["<|im_end|>"]`

---

## What ATLAS Does Well

- **Single-file creation**: Python scripts, Rust CLIs, Go servers, C programs, shell scripts — first-shot, compiles and runs
- **Multi-file project scaffolding**: Next.js, Flask, Express — correct dependency order, config files included
- **Bug fixes**: Reads existing files, identifies issues, applies targeted edits via `edit_file`
- **Feature additions**: Reads project context, adds features using surgical `old_str`/`new_str` changes
- **Code analysis**: Reads entire codebases and explains implementation details
- **V3-enhanced quality**: Files with complex logic (T2) get diverse candidates, build verification, and energy-based selection — producing measurably better code

## What ATLAS Is Not Good At (Yet)

- **Very large existing codebases** (50+ files): The 32K context window limits how much project context the model can process at once
- **Visual output verification**: CSS styling, layout issues, and design quality cannot be verified by the sandbox
- **Real-time interactive applications**: The model cannot run a browser or test interactive UIs
- **Adding features to existing projects**: ~67% reliability (L6 test) — the 9B model sometimes over-explores instead of writing code

## Tips for Best Results

1. **Be specific**: "Create a Flask API with /users GET and POST endpoints, SQLite backend, input validation with Pydantic" works better than "Create a web app"
2. **Provide file context**: When modifying existing code, `/add` files to the Aider chat so ATLAS can read them
3. **Complex tasks take longer**: V3 pipeline fires on feature files (50+ lines with logic), adding 2-5 minutes but producing better code
4. **Watch the terminal**: Streaming shows every tool call, V3 step, and build verification in real-time
5. **Use edit_file hints**: For large existing files, ask for specific changes rather than full rewrites — the proxy rejects `write_file` for existing files over 100 lines

---

## Environment Variables

All ports and URLs are configurable via environment variables:

### Service URLs

| Variable | Default | Used By | Purpose |
|----------|---------|---------|---------|
| `ATLAS_INFERENCE_URL` | `http://localhost:8080` | proxy, v3-service, Python CLI | llama-server endpoint |
| `ATLAS_RAG_URL` | `http://localhost:8099` | Python CLI | Geometric Lens endpoint |
| `ATLAS_LENS_URL` | `http://localhost:8099` | proxy, v3-service | Geometric Lens endpoint |
| `ATLAS_SANDBOX_URL` | `http://localhost:30820` | proxy, v3-service, Python CLI | Sandbox endpoint |
| `ATLAS_V3_URL` | `http://localhost:8070` | proxy | V3 Pipeline endpoint |

### Service Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `ATLAS_MODEL_NAME` | `Qwen3.5-9B-Q6_K` | Model identifier for API responses |
| `ATLAS_MODEL_FILE` | `Qwen3.5-9B-Q6_K.gguf` | GGUF filename in models directory |
| `ATLAS_MODELS_DIR` | `./models` | Host path to model weights |
| `ATLAS_CTX_SIZE` | `32768` | Context window size (tokens) |
| `ATLAS_AGENT_LOOP` | `1` | Enable agent loop in proxy (`1` = on) |
| `ATLAS_PROXY_PORT` | `8090` | Proxy listening port |
| `ATLAS_V3_PORT` | `8070` | V3 service listening port |
| `ATLAS_LLAMA_PORT` | `8080` | llama-server listening port |
| `ATLAS_LENS_PORT` | `8099` | Geometric Lens listening port |
| `ATLAS_SANDBOX_PORT` | `30820` | Sandbox host port |
| `GEOMETRIC_LENS_ENABLED` | `true` | Enable/disable Lens scoring |

### Bare Metal Only

| Variable | Default | Purpose |
|----------|---------|---------|
| `ATLAS_LLAMA_BIN` | `~/llama-cpp-mtp/build/bin/llama-server` | Path to llama-server binary |
| `ATLAS_MODEL_PATH` | `~/models/Qwen3.5-9B-Q6_K.gguf` | Full path to model file |

---

## Configuration Files

### `.aider.model.settings.yml`

Aider model configuration for the ATLAS proxy:

```yaml
- name: openai/atlas
  edit_format: whole
  weak_model_name: openai/atlas
  use_repo_map: true
  send_undo_reply: true
  examples_as_sys_msg: true
  extra_params:
    max_tokens: 32768
    temperature: 0.3
  cache_control: false
  caches_by_default: false
  streaming: true
  reminder: sys
```

### `.aider.model.metadata.json`

Model metadata telling Aider the token limits and cost (local = free):

```json
{
  "openai/atlas": {
    "max_tokens": 32768,
    "max_input_tokens": 32768,
    "max_output_tokens": 32768,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
    "litellm_provider": "openai",
    "mode": "chat"
  }
}
```

Both files must exist in the ATLAS project root for the launcher to work. The launcher passes them to Aider via `--model-settings-file` and `--model-metadata-file`.
