# ATLAS Troubleshooting Guide

Common issues and solutions for ATLAS V3.0.1, organized by service.

---

## Quick Diagnostics

**Always start with `atlas doctor`.** It runs **21 checks** across the host (Docker / Compose / NVIDIA), the running services (containers, health endpoints, image skew), the runtime contract (kernel `vm.overcommit_memory`, model file presence, Lens weights, e2e smoke), and the host-vs-tier match (configured model fits this hardware, CPU/RAM/disk meet the tier minimums). It supersedes the manual `curl` ritual that used to lead this section.

```bash
atlas doctor              # full check (~5–10 s)
atlas doctor --quick      # skip the e2e smoke (~2 s)
atlas doctor -v           # verbose: full detail per check
atlas doctor --json       # machine output (for bootstrap / CI)
```

See [CLI.md → Diagnostic Commands](CLI.md#diagnostic-commands) for the full check list, flag table, and exit codes. Doctor exits `0` on pass-or-warn, `1` on any fail — bootstrap and CI gate on the exit code. If a check fails, run with `-v` to get the underlying command output and remediation hints.

If `atlas doctor` is unavailable (e.g., you haven't `pip install -e .` yet), the manual fallbacks are:

```bash
# Docker Compose — check all services at once
docker compose ps

# Individual health checks
curl -s http://localhost:8080/health | python3 -m json.tool   # llama-server
curl -s http://localhost:8099/health | python3 -m json.tool   # geometric-lens
curl -s http://localhost:8070/health | python3 -m json.tool   # v3-service
curl -s http://localhost:30820/health | python3 -m json.tool  # sandbox
curl -s http://localhost:8090/health | python3 -m json.tool   # atlas-proxy (shows all service statuses)

# GPU status
nvidia-smi

# Docker Compose logs (last 50 lines per service)
docker compose logs --tail 50
```

The atlas-proxy health endpoint reports the status of all upstream services:
```json
{
  "status": "ok",
  "inference": true,
  "lens": true,
  "sandbox": true,
  "port": "8090",
  "stats": { "requests": 0, "repairs": 0, "sandbox_passes": 0, "sandbox_fails": 0 }
}
```

If any field is `false`, that service is the problem.

### Hardware classification

If you're unsure whether your host can run ATLAS at the configured tier — or you just swapped GPUs — run `atlas tier`. It probes VRAM / RAM / CPU / disk, classifies the host into one of 5 tiers (`cpu` / `small` / `medium` / `large` / `xlarge`), and prints the recommended model + runtime knobs (context length, parallel slots, KV-cache quantization). The output also serves as the source-of-truth for `.env` values; see [CLI.md → `atlas tier`](CLI.md#atlas-tier) for the band table and flags.

`atlas doctor` already runs the tier check (`tier_match` + `tier_constraints`) as part of its 21-check sweep, so most users don't need to invoke `atlas tier` directly — only when planning a config change or hardware swap.

---

## Docker / Podman Issues

### GPU Not Detected in Container

**Symptom:** llama-server container starts but model loads on CPU (very slow, ~2 tok/s). `nvidia-smi` shows the GPU from the host but the container can't see it.

**Fix:** Install NVIDIA Container Toolkit:

```bash
# RHEL/Fedora
sudo dnf install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=podman
sudo systemctl restart podman

# Ubuntu/Debian
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is visible inside containers:
```bash
# Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Podman
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.0-base nvidia-smi
```

### First Build Fails (CUDA Not Found)

**Symptom:** `docker compose build` fails with CUDA-related errors during llama-server compilation.

**Fix:** The llama-server Dockerfile builds llama.cpp inside a `nvidia/cuda:12.8.0-devel` base image, so CUDA headers are available during build without host GPU access. Common causes of build failure:
1. Insufficient disk space (~5GB needed for build artifacts)
2. Network issues downloading the CUDA base image or cloning llama.cpp
3. Podman rootless builds may fail with permission issues — try `podman-compose build` with `--podman-build-args="--format docker"`

### llama.cpp Clone Times Out

**Symptom:** Build hangs in the `llama-server builder 3/3` stage and eventually fails with:

```
error: RPC failed; curl 56 OpenSSL SSL_read: Connection timed out, errno 110
fatal: early EOF
fatal: fetch-pack: invalid index-pack output
```

**Cause:** The full llama.cpp git history is large (~1 GB) and the clone is sensitive to flaky/slow connections. A momentary stall causes the SSL read to time out and the whole transfer to abort.

**Fix (already applied in `inference/Dockerfile.v31`):** the Dockerfile uses `git clone --depth 1 --single-branch` with `http.postBuffer=524288000` and `http.lowSpeedLimit/Time` to fail-fast on dead connections instead of hanging for 11 minutes. If you have an older Dockerfile or the issue recurs:

1. Retry the build — transient network blips happen, especially on residential connections.
2. If retries keep failing, pre-pull the repo on the host and bind-mount it into the build context. Quick recipe:
   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
   # then edit Dockerfile.v31 to COPY from /tmp/llama.cpp instead of cloning
   ```
3. Long term: prebuilt llama-server images on GHCR will skip this step entirely (Phase 0 roadmap item).

### SELinux Blocking Container Access (Fedora/RHEL)

**Symptom:** Containers can't read mounted volumes, permission denied on model files.

**Fix:**
```bash
# Allow container access to model directory
chcon -Rt svirt_sandbox_file_t ~/models/

# Or add :Z flag to volume mounts (Docker Compose handles this)
```

### Sandbox Unreachable

**Symptom:** Proxy health shows `"sandbox": false`. V3 build verification fails.

**Fix:** Ensure all services are on the same Docker network. Docker Compose creates the `atlas` network automatically. If running containers manually:
```bash
docker network create atlas
# Start all containers with --network atlas
```

### Port Conflicts

**Symptom:** `docker compose up` fails with "address already in use" on a port.

**Fix:** Check what's using the port and either stop it or change ATLAS ports in `.env`:
```bash
# Find what's using port 8080
lsof -i :8080

# Change port in .env
ATLAS_LLAMA_PORT=8081    # Different port for llama-server
```

All ports are configurable via `.env`. See [CONFIGURATION.md](CONFIGURATION.md).

---

## llama-server Issues

### Model Loading on CPU Instead of GPU

**Symptom:** Generation at ~2 tok/s instead of ~50 tok/s. `nvidia-smi` doesn't show llama-server using the GPU.

**Fix:** Ensure `--n-gpu-layers 99` is set (offloads all layers to GPU). In Docker Compose this is the default. For bare metal, check the command:
```bash
ps aux | grep llama-server | grep 'n-gpu-layers'
```

If using Docker, ensure the NVIDIA container runtime is configured (see GPU section above).

### Model File Not Found

**Symptom:** llama-server exits immediately with "failed to load model" or similar.

**Diagnose first:** `atlas model list --installed` — shows which registry entries are present in `ATLAS_MODELS_DIR`. If your configured model isn't listed, the file is missing.

**Fix path 1 — install via the registry:**
```bash
atlas model recommend                          # what should I have for this hardware?
atlas model install Qwen3.5-9B-Q6_K            # download from HuggingFace
```

**Fix path 2 — manual placement.** Check the model path:
```bash
# Docker Compose — model must be in ATLAS_MODELS_DIR (default: ./models/)
ls -la models/Qwen3.5-9B-Q6_K.gguf

# Bare metal — check ATLAS_MODEL_PATH
ls -la ~/models/Qwen3.5-9B-Q6_K.gguf
```

The filename must match `ATLAS_MODEL_FILE` in `.env` (default: `Qwen3.5-9B-Q6_K.gguf`).

### G(x) verification always returns "unavailable" or `gx_score: 0.5`

**Symptom:** Generations come back from llama-server fine, but every Lens response shows `gx_score: 0.5, verdict: "unavailable"`. The C(x)/G(x) verification half of ATLAS isn't working.

**Cause:** You're running a model that has no Lens artifacts (no trained metric tensor + embeddings for that specific model). The Lens silently no-ops on unknown models. Only `Qwen3.5-9B-Q6_K` ships with artifacts in the repo today; 7B / 14B / 32B were either never trained or had their artifacts removed.

**Diagnose:**
```bash
atlas doctor                                   # tier_match check (#20) flags this directly
atlas model list                               # look for Lens status column
```

**Fix:**
- **Recommended:** switch to a `supported` model. `atlas model recommend` will surface the fallback (`Qwen3.5-9B-Q6_K`) regardless of which tier your hardware lands in.
- **If you must run a `no-artifacts` model:** train Lens artifacts for it locally via `atlas lens build` (PC-058 — not yet shipped). Until that's available, you have to accept G(x) won't score generations.

See [PC-058 roadmap](https://github.com/itigges22/ATLAS/issues/100) for the Lens training pipeline and [SETUP.md → Model Management](SETUP.md#model-management) for the full registry story.

### Out of VRAM

**Symptom:** llama-server crashes or gets OOMKilled shortly after starting. `nvidia-smi` shows VRAM near 100%.

**Diagnose first:** run `atlas doctor` — the `tier_match` check (#20) flags configured models that overshoot the host's tier (e.g., a 14B model on a 12 GB GPU). If it warns, you're running a tier above what the hardware supports; downgrade the model or upgrade the GPU. Run `atlas tier` for the recommended model + runtime knobs for your specific hardware.

**Fix:** The default 9B Q6_K model needs ~8.2 GB VRAM (model + KV cache). Ensure:
1. No other GPU processes are running (`nvidia-smi` — check for other CUDA processes)
2. You have at least 12 GB VRAM (medium tier minimum). For 8 GB cards, switch to the small-tier 7B Q4_K_M model.
3. Context size isn't set too high (default 32K is fine, don't increase without checking VRAM). `atlas tier` shows the recommended `ATLAS_CTX_SIZE` for your tier.

```bash
# Kill other GPU processes if needed
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill {}
```

### Grammar Not Enforced (Model Outputs Thinking Blocks)

**Symptom:** Model outputs `<think>` tags or raw text instead of JSON tool calls.

**Fix:** The proxy sets `response_format: {"type": "json_object"}` automatically when `ATLAS_AGENT_LOOP=1`. If using llama-server directly, include it in your request:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q6_K",
    "messages": [{"role":"user","content":"Say hi"}],
    "max_tokens": 50,
    "response_format": {"type": "json_object"}
  }'
```

If this returns raw text instead of JSON, your llama.cpp build doesn't support `response_format`. Rebuild from the latest source.

### Context Window Too Small

**Symptom:** Tool call arguments get truncated. `write_file` fails with "unexpected end of JSON" or proxy logs show "truncation detected".

**Fix:** Context size should be 32768 (default in Docker Compose). Check:
```bash
# Docker Compose
grep CTX_SIZE .env

# Bare metal
ps aux | grep llama-server | grep ctx-size
```

---

## Proxy Issues

### Agent Loop Not Activating

**Symptom:** Requests go directly to llama-server. No tool calls, no streaming status icons, no V3 pipeline.

**Fix:** Set `ATLAS_AGENT_LOOP=1`. The `atlas` launcher does this automatically. If running the proxy manually:
```bash
ATLAS_AGENT_LOOP=1 atlas-proxy-v2
```

In Docker Compose, this is set in `docker-compose.yml` and doesn't need manual configuration.

### V3 Pipeline Not Firing on Feature Files

**Symptom:** All `write_file` *or* `edit_file` calls are T1 (direct write). No V3 pipeline stages in output.

V3 fires when **all conditions** are met:
1. File has **50+ lines** of content
2. File has **3+ logic indicators** (function defs, control flow, API patterns)
3. V3 service is reachable at `ATLAS_V3_URL`
4. **Request tier ≥ T2** (classifier output, after any agent override) **AND** the file's own tier ≥ T2 (PC-042)

**Both** `write_file` and `edit_file` route through V3 since PC-042. Before that, only `write_file` did — and since the system prompt steers the model toward `edit_file` for all changes to existing files, V3 effectively never ran on real edits. If you're on a build that predates PC-042, that's why.

**Diagnose:**
```bash
# Check V3 service health
curl -s http://localhost:8070/health

# Check proxy logs for tier classification + V3 activation
docker compose logs atlas-proxy | grep -E "write_file|edit_file|tier="
# Look for:
#   "tier=T2:medium" or higher in classifier output
#   "[edit_file] V3 pipeline activating for X (req_tier=2, file_tier=2)"
#   "[write_file] V3 pipeline activating for X"
# T1 means direct write — no V3.
```

If V3 is unreachable, the proxy logs `V3 failed: ...` and falls back to direct write without breaking the edit.

### Truncation Errors (write_file Fails Repeatedly)

**Symptom:** Repeated errors like "Your output was truncated — the content is too long for a single tool call."

**Cause:** The model is trying to write too much content in one call. The proxy detects truncated JSON and rejects the tool call.

**What happens automatically:**
- For existing files > 100 lines: proxy rejects `write_file` and tells the model to use `edit_file` instead
- After 3 consecutive failures: error loop breaker stops the agent and returns a summary

**What you can do:** Rephrase your request to ask for targeted changes rather than full file rewrites. For example, "Add input validation to the login function" instead of "Rewrite auth.py".

**False positives, pre-PC-040.** Before PC-040, *any*
`unexpected end of JSON` from a tool's input parser was
relabeled "tool call truncated." The most common trigger
was the model emitting a tool call with **no `args` field
at all** — e.g. `{"type":"tool_call","name":"read_file"}`
— which is malformed input, not truncated output. The old
remap then steered the model toward `edit_file` of a file
it had never read, looping until the 3-error breaker
fired. PC-040 fixes this in two ways:

1. Empty/missing `args` is caught **before** the tool's
   `Execute` runs, and the proxy returns a per-tool hint
   like `read_file: no arguments provided. Call with
   {"path":"<file>"}. Use list_directory {"path":"."}
   first if you need to discover what files exist.`
2. The "truncated" remap now only fires when the args
   payload is over 200 bytes (real truncation territory).
   Short or empty args fall through to the actual parser
   error.

If you still see "tool call truncated" after PC-040 ships,
it's a real truncation — the model was actually trying to
write a payload too long for the context window. The
`edit_file` advice still applies in that case.

**PC-041 alt-shape lifting.** Some models emit tool calls
in OpenAI-style (`arguments` instead of `args`),
Anthropic-style (`parameters`), or with arguments inlined
at the top level (`{"name":"read_file","path":"x.py"}`).
The proxy now normalizes all three shapes into the
canonical `args` envelope automatically. If a tool call
still arrives with empty args after normalization, the
proxy logs `[agent] turn=N EMPTY ARGS — raw model output:
"..."` so you can see the exact shape it sent and either
add it to the lift list or rephrase the prompt.

### Long Pause Between Tool Result and Next Action

**Symptom:** A tool succeeds, then the agent loop sits
idle for ~30 seconds before the next turn fires. No
errors, no output — eventually the next tool call appears.

**Cause (PC-043).** Qwen3.5-9B with `/nothink` +
`response_format: json_object` occasionally emits zero
tokens after a tool result. The grammar requires the
response to start with `{`, but the model's natural
continuation after a tool result is a brief whitespace /
acknowledgment, which the grammar rejects. The model
emits EOS as its first token, returning empty content,
which the parse-error retry path then has to recover
from with a fresh user message — that's the lost ~30
seconds.

PC-043 catches this inside `callLLMConstrained` and
retries inline once with `temperature=0.7` and a
transient continuation nudge appended to the messages.
The agent loop never sees the empty turn.

**Diagnose:**
```bash
docker compose logs atlas-proxy | grep -E "PC-043|empty LLM|raw_len=0"
```
- `[agent] empty LLM response (PC-043), retrying with
  temp=0.7 + continuation nudge` — the retry fired; if
  the next log line is a normal `turn=N type=tool_call ...`
  the recovery worked.
- `parse error: ... raw_len=0 | raw: ""` — both the
  initial call AND the PC-043 retry returned empty. The
  outer parse-error retry will handle it, but you'll see
  the long pause. If this happens consistently, model is
  in a worse state than PC-043 anticipates — file a
  follow-up ticket with the full proxy log.

**Workaround if PC-043 isn't enough:** Restart the proxy
to clear llama.cpp's slot cache:
```bash
docker compose restart atlas-proxy llama-server
```

### Model Keeps Editing After V3 Already Confirmed the Fix

**Symptom:** The agent makes a successful V3-verified
edit (Aider TUI shows V3 progress events ending in
`Probe passed`), then re-reads the same file and starts
editing other unrelated functions. Each follow-on edit
triggers another full V3 cycle (~110s), and the new edits
sometimes touch code that has nothing to do with the
original bug.

**Cause (PC-044).** The 9B model has trouble
self-assessing "is the user's original problem solved?"
After a tool result with `v3_used=true,
phase_solved=probe`, it has no strong signal to stop, so
it just continues planning more work.

**What PC-044 does.** Immediately after a V3-verified
write_file or edit_file, the agent loop appends a strong
user-role nudge: *"V3 verified this edit passed its
{phase} pipeline. The fix is on disk and build-checked.
If this resolves the user's original request, respond
NOW with {"type":"done","summary":"..."}. Only continue
if you have a specific, concrete additional change to
make — do not re-read the file to double-check, and do
not edit unrelated code."*

**Diagnose:**
```bash
docker compose logs atlas-proxy | grep "PC-044"
```
- `[agent] PC-044: V3-verified edit_file on ... — nudging
  toward done` — the nudge fired. The next agent turn
  should be `type=done`. If it isn't, the model ignored
  the nudge — file a follow-up ticket noting the
  prompt and the next-turn tool call.

**If the model still won't stop after PC-044:** The
follow-up options (hard-stop after re-read, per-file
edit cap, or auto-done from the proxy) are listed in
ISSUES.md PC-044 under "Caveat — promote to a harder
option if the soft nudge doesn't stick."

### Model Hallucinates Filenames From Previous Sessions

**Symptom:** Brand-new session, fresh prompt about a file
in the current directory, and the model's first tool call
is a `read_file` on a filename that doesn't exist
anywhere in this workspace — usually a filename that
*does* exist somewhere else you've worked recently.

**Cause (PC-045).** llama.cpp's KV slot persists between
chat completions to keep the cache warm (PC-035). Across
*sessions*, that means residual attention bias from the
previous session's tokens leaks into the new session.
Most prompts dominate this bias, but model-fabricated
filenames and other low-entropy outputs can pick it up.

**What PC-045 does.** Every `runAgentLoop` invocation
(one per Aider message) starts by POSTing
`/slots/0?action=erase` to llama-server. The KV cache is
reset; the next chat completion re-encodes the system
prompt from scratch (~1-2s on warm GPU). Within the
session, subsequent turns share the now-fresh cache as
normal.

**Diagnose:**
```bash
docker compose logs atlas-proxy | grep "PC-045"
```
- `[PC-045] erased llama slot 0 — fresh KV cache for
  this session` on every Aider message — working as
  intended.
- `[PC-045] erase slot: ...` followed by an error — the
  HTTP call to llama-server failed. Slot may still hold
  stale state, but next chat completion will partially
  overwrite it. Worst case: pre-PC-045 behavior.

**Disable** if you measure the per-message ~1-2s blip
and decide it's worse than occasional cross-session
leakage:
```bash
# .env
ATLAS_FRESH_SLOT_PER_SESSION=0
```
Restart the proxy after changing.

**Workaround if PC-045 is somehow disabled and you see
hallucinations:** Restart `llama-server` to fully clear
all slots:
```bash
docker compose restart llama-server
```

### Multi-File Project: Sandbox `ModuleNotFoundError`

**Symptom:** Edit on a file that imports another module
in the same project. V3 reports verification failure
with `ModuleNotFoundError: No module named 'utils'` (or
similar) even though the import works fine on your
machine.

**Cause (PC-046).** Pre-PC-046 the sandbox wrote *only*
the candidate file as `solution.py` to its workspace.
Any `from utils import …` failed because `utils.py`
didn't exist in the sandbox's tmpdir.

**What PC-046 does.** Sandbox `/execute` accepts a
`files: Dict[str, str]` map; V3's `SandboxAdapter`
ships every file the agent has read (the same
`ProjectContext` dict V3 already feeds to the LLM
prompt) into the sandbox workspace alongside
`solution.py`. Multi-file imports resolve.

**Diagnose:** if you still see `ModuleNotFoundError`
in V3 progress events, the file is probably not in
`ctx.FilesRead` (the proxy's read-tracking set). Read
the missing file via `read_file` so it lands in the
project context that V3 ships to the sandbox.

**If you're using the sandbox `/execute` API directly**
(scripts, tests), pass the supporting files in the
request body:
```bash
curl -X POST http://localhost:30820/execute -d '{
  "code": "from utils import greet\nprint(greet(\"x\"))",
  "language": "python",
  "files": {"utils.py": "def greet(n): return f\"hi {n}\""}
}'
```

### Curses Bottom-Row `addwstr() returned ERR`

**Symptom:** Your curses program (snake game, TUI menu,
status bar, etc.) crashes at runtime with:
```
_curses.error: addwstr() returned ERR
```
…but ATLAS reported the edit passed V3 verification.

**Cause.** Writing to the last cell of a curses window
(any row=LINES-1, or column=COLS-1) is documented as
returning ERR. This is decades-old curses behavior. The
idiomatic fix:
```python
try:
    stdscr.addstr(curses.LINES - 1, 0, border)
except curses.error:
    pass  # writing the bottom-right cell errors; benign
```

**What PC-047 does.** `interactive_lint` now AST-walks
for `addstr/addnstr/addch(curses.LINES - N, ...)` (and
the bare `LINES - N` form after `from curses import LINES`)
that aren't inside a `try/except curses.error` block.
Such candidates are rejected at the lint gate — V3 has
to find a wrapped variant before certifying.

**Diagnose:**
```bash
docker compose logs v3-service | grep "interactive_lint"
```
- `[interactive_lint] OK` — candidate passed all checks.
- `[interactive_lint] FAIL: curses bottom-row write
  without try/except curses.error wrap — line N: ...` —
  PC-047 fired. V3 will either find a wrapped variant
  or surface the failure to the model so it can produce
  one.

**If V3 can't find a wrapped variant**, the model is in
the structural-reasoning gap (Issue B): it knows the
file uses `curses.LINES - 1` but can't reliably
synthesize the try/except wrap. Workaround: tell the
model explicitly in your Aider prompt: *"wrap the
addstr call at line N in `try: ... except curses.error:
pass`."*

### V3 Hangs for Several Minutes on Non-Python Files

**Symptom:** Asking ATLAS to write an HTML/CSS/JSON file
causes a long pause (~5 minutes) with progress events
showing PR-CoT repair attempts and LLM timeouts. The
file eventually gets written via the direct-write
fallback, but the V3 cycle was wasted.

**Cause (PC-048).** Pre-PC-048 the V3 smoke check
hardcoded `compile(_src, '<smoke>', 'exec')` (Python AST
parse) for **every** interactive-task candidate — HTML,
CSS, JSON, anything. Any non-Python file failed the
smoke check with `SYNTAX_ERROR`, which kicked V3 into
PR-CoT repair, which made LLM calls that timed out, then
fell back to direct write.

**What PC-048 does.** `smoke_compile_check` is now
language-aware. The V3 pipeline derives language from
the target file's extension (`pipeline.run(file_path=…)`)
and routes:
- `.py` → AST/compile smoke (existing behavior)
- `.html` / `.htm` / `.xml` → `html.parser` strict mode
- `.json` → `json.loads`
- `.yaml` / `.yml` → `yaml.safe_load` (or skip if PyYAML
  unavailable)
- everything else (CSS, JS, MD, plain text, TOML, …) →
  pass-through with `SMOKE_SKIP (non-Python)`

**Diagnose:**
```bash
docker compose logs v3-service | grep "smoke_check"
```
- `[smoke_check] compile=OK (html)` — PC-048 routed
  correctly.
- `[smoke_check] compile=OK (python)` on a `.html` file —
  the proxy didn't pass `file_path` through. Check
  `atlas-proxy/v3_bridge.go` and the
  `V3GenerateRequest`.
- `[smoke_check] compile=FAIL` followed by
  `[phase3] All candidates failed — entering repair
  phase` followed by `[LLM] Attempt N failed: timed
  out` — the cascade PC-048 was supposed to prevent
  is happening anyway. File a follow-up ticket with
  the failing file extension.

**If you're hitting this on a file extension PC-048
doesn't recognize**, the smoke check defaults to Python
and you get the same cascade. Workaround: set
`ATLAS_AGENT_LOOP=1` and rely on the proxy's direct
write path, or add the extension to `_ext_to_lang` in
`v3-service/main.py:613`.

### V3 Pipeline Doesn't Fire on "Fix It Again" Prompts

**Symptom:** First request creates a file, V3 pipeline
runs (you see V3 progress events). Follow-up "still
doesn't work, try again"-style prompts complete in
microseconds with no V3 events visible. The model just
edits and exits without verification.

**Cause (PC-049).** Pre-PC-049 the agent-loop tier
classifier checked a narrow vocabulary (`fix`,
`broken`, `doesn't work`, `bug`, …) and required at
least one explicit file extension in the prompt. Real
iterative-fix prompts use natural phrases ("still does
not", "isn't working", "try again") with no `.py` in
sight, so the classifier returned T1, V3 never fires.

**What PC-049 does.** Vocabulary expanded to cover
natural fix language (`doesn't`, `is not`, `aren't`,
`failed`, `wrong`, etc.), plus a separate
"continuation marker" detector (`still`, `again`,
`retry`, `another`). Continuation markers substitute
for explicit file names — if you say "still doesn't
work" we now know you mean "the existing file isn't
working" even if you don't name it.

**Diagnose:**
```bash
docker compose logs atlas-proxy | grep "agent tier override"
```
- `agent tier override: T2:medium` — PC-049 promoted
  correctly. V3 should fire on the next edit_file.
- `agent tier override: T1:simple` on a clearly-iterative
  prompt — PC-049's vocabulary missed it. File a
  follow-up ticket with the exact prompt; the
  vocabulary is finite.

**Workaround if classifier still misses your prompt:**
Mention the file by name in the prompt — `app.py` is
enough. The original `fileIndicators >= 1` gate still
works for explicit file mentions.

### "Hallucinated" Filenames From Aider Chat History

**Symptom:** Brand-new prompt about `snake_game.py`,
but the agent's first read is on `show_greeting.py` (or
some other file you've never mentioned in this
session). PC-045 fires correctly (you see
`[PC-045] erased llama slot 0` in the proxy log), so
it's not LLM cache pollution.

**Cause.** Aider stores chat history per-project
directory in `.aider.chat.history.md` and feeds it to
the LLM as conversation context on **every** new
prompt. Filenames mentioned in *prior sessions in the
same directory* leak into the current session's
context. PC-045 erases the LLM KV cache but cannot
clear Aider's history file — that's outside the proxy's
scope.

**Fix:**
```
# In Aider:
/clear

# Or from a shell:
rm .aider.chat.history.md
```

The recovery path works (read fails on the
non-existent file → model pivots to list_directory →
finds the real file), so this is just a one-turn
waste, not a session-blocker.

### File Not Read Before Editing

**Symptom:** `edit_file` fails with "file not read yet — use read_file first before editing."

**Cause:** The proxy tracks which files the agent has read. If the model tries to edit a file it hasn't read in this session, the edit is rejected as a staleness protection.

**Fix:** This is normal behavior — the model should read the file first. If it keeps failing, the model may be confused about which files it has seen. Try `/clear` in Aider and rephrase.

### File Modified Externally

**Symptom:** `edit_file` fails with "file modified since last read — read it again before editing."

**Cause:** The file was changed on disk (by you or another process) after the model read it. The proxy compares modification timestamps.

**Fix:** The model needs to re-read the file. This usually resolves automatically on the next turn.

### Exploration Budget Warning

**Symptom:** Output shows "You have full project context in the system prompt. Do not read more files." or reads are being skipped.

**Cause:** The model has made 4+ consecutive read-only calls (read_file, search_files, list_directory) without writing anything. After 4 reads, the proxy warns. After 5+, it skips reads entirely and tells the model to write.

**Fix:** This is protective behavior. If the model is genuinely stuck exploring, try being more specific about what you want changed.

---

## Geometric Lens Issues

### Lens Not Loaded / Unavailable

**Symptom:** Proxy health shows `"lens": false`. Or startup shows "Lens unavailable — verification disabled."

**Impact:** ATLAS still works but without C(x)/G(x) scoring. V3 candidate selection falls back to sandbox-only verification.

**Fix:** Check Lens health and logs:
```bash
curl -s http://localhost:8099/health
docker compose logs geometric-lens
```

Common causes:
- Lens can't connect to llama-server (check `LLAMA_URL` env var)
- Model weight files missing (service degrades gracefully — this is expected if you haven't trained custom models)

### All Scores Near 0.5

**Symptom:** Every candidate gets `cx_energy: 0.0` and `gx_score: 0.5` regardless of code quality.

**Cause:** Model weights are not loaded. The service returns neutral defaults when models are absent.

**Verify:**
```bash
curl -s http://localhost:8099/internal/lens/gx-score \
  -H "Content-Type: application/json" \
  -d '{"text": "print(1)"}' | python3 -m json.tool
```

If `enabled: false` or `cx_energy: 0.0`, the models aren't loaded. This is expected for a fresh install — model weights are not included in the repository and must be trained or downloaded from [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS).

### Embedding Extraction Fails

**Symptom:** Lens logs show errors like "embedding extraction failed" or timeouts.

**Cause:** Lens calls llama-server's `/v1/embeddings` endpoint. If llama-server is overloaded or the endpoint isn't enabled, this fails.

**Fix:**
```bash
# Test embedding endpoint directly
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test"}' | python3 -m json.tool
```

The `/v1/embeddings` endpoint is available in llama.cpp without special flags for self-embeddings from generation models. In K3s, the `--embeddings` flag is set explicitly in the entrypoint for full embedding support.

---

## Sandbox Issues

### Sandbox Unreachable

**Symptom:** Code is never tested. Proxy health shows `"sandbox": false`.

**Fix:** Check sandbox health:
```bash
# Docker Compose (host port 30820 maps to container port 8020)
curl -s http://localhost:30820/health

# Bare metal (direct port 8020)
curl -s http://localhost:8020/health
```

If the sandbox container is running but unhealthy, check logs:
```bash
docker compose logs sandbox
```

### Code Execution Timeout

**Symptom:** Sandbox returns `"error_type": "Timeout"`. Code takes too long to execute.

**Default timeout:** 30 seconds per request, max 60 seconds (configurable via `MAX_EXECUTION_TIME` env var).

**Fix:** If your code legitimately needs more time, set a higher timeout in the request. If the code has an infinite loop, this is expected behavior.

### Language Not Supported

**Symptom:** Sandbox returns an error for a specific language.

**Supported languages:** Python, JavaScript, TypeScript, Go, Rust, C, C++, Bash.

Check available runtimes:
```bash
curl -s http://localhost:30820/languages | python3 -m json.tool
```

---

## Aider Issues

### `atlas` Shows REPL Instead of Aider (No File Read/Write)

**Symptom:** Running `atlas` shows the built-in REPL with a `Model`, `Speed`, `Lens`, `Sandbox` status block and a `◆` prompt. Typing requests works but no files are created or modified. `--message` flag is ignored.

**Cause:** The `atlas` command auto-detects the proxy and Aider. If either is missing, it falls back to the built-in REPL which supports `/solve` and `/bench` but not file operations.

**Fix:**
1. Ensure the proxy is running: `curl -s http://localhost:8090/health`
2. Ensure Aider is installed: `pip install aider-chat`
3. Ensure services are up: `docker compose ps` (all should show "healthy")

If the proxy is healthy and Aider is installed, `atlas` will automatically launch Aider with the full agent loop (tool calls, file read/write, V3 pipeline).

If Go 1.24+ is installed, `atlas` can also build and launch the proxy automatically — you don't need to start it manually.

### Proxy Lists Wrong Directory or `/tmp`

**Symptom:** The model lists files from `/tmp` or the ATLAS repo instead of your project. `write_file` creates files in the wrong location.

**Cause:** The Docker Compose proxy runs inside a container and can only see the directory mounted at startup. If you're working in a different directory, the proxy can't see it.

**Fix (recommended):** Install Go 1.24+ ([https://go.dev/dl/](https://go.dev/dl/)). The `atlas` CLI will automatically build and launch the proxy locally in your current directory with full file access. No Docker mount needed.

**Fix (without Go):** Set `ATLAS_PROJECT_DIR` in your `.env` to your project path, then restart the proxy:
```bash
# In .env:
ATLAS_PROJECT_DIR=/path/to/your/project

# Restart proxy to pick up new mount:
docker compose up -d atlas-proxy
```

You must update this and restart each time you switch project directories. This is a limitation of running the proxy inside Docker.

### `.env.example` Missing After Clone

**Symptom:** `cp .env.example .env` fails with "No such file or directory".

**Fix:** This was fixed in V3.0.1. If you cloned before the fix, pull the latest:
```bash
git pull
cp .env.example .env
```

### Aider Disconnects on Long Tasks

**Symptom:** Aider times out or disconnects before the agent loop completes, especially during V3 pipeline phases.

**Fix:** Aider's HTTP request timeout needs to be long enough for V3 pipeline execution (which can take minutes). The `.aider.model.settings.yml` in the repo configures streaming mode which keeps the connection alive. If you're still seeing timeouts:

1. Ensure you're using the repo's config files (`.aider.model.settings.yml` and `.aider.model.metadata.json`)
2. Check that `streaming: true` is set in the settings file

### "context deadline exceeded" mid-session

**Symptom:** Aider shows
`litellm.APIError ... context deadline exceeded` partway through
a session, sometimes during a busy turn, sometimes after a quiet
gap.

**Two distinct causes — each fixed differently:**

1. **In-session prompt-cache pressure (PC-029).** After ~4-5 turns
   the llama-server prompt cache fills up and prompt-eval rate
   drops from ~700 tok/s to ~100 tok/s. Tight 5s deadlines on the
   classifier and lens-score paths blew out. **Fixed in atlas-proxy
   v3.0.1+** — classifier ctx 5s→60s, lens client 5s→30s, background
   stream-score 5s→30s. No user-side action needed beyond keeping
   atlas-proxy current.

2. **Cold-start after idle (PC-035).** After 1-2 minutes of
   inactivity, the next request blows the 120s `forwardToLLM`
   client timeout because the prompt cache is evicted and the
   conversation has to be reprocessed cold. The error wording
   distinguishes this case: it includes `Client.Timeout exceeded
   while awaiting headers`, which is specific to Go's HTTP client
   timeout (vs the context-cancel "deadline exceeded" wording for
   case 1). **Fixed in atlas-proxy v3.0.1+** with two layers:
   - Client timeout bumped 120s → 240s (cold-start budget).
   - A 45s-interval keep-warm goroutine that pings llama-server
     with a 1-token completion. Avoids the cold path entirely.
     Disable with `ATLAS_KEEP_LLAMA_WARM=0` if running CPU-only
     or under tight power constraints.

If you still see this after upgrading, check
`docker compose logs atlas-proxy` for `keep-warm: pinging …`
on startup — its absence means the goroutine isn't running and
the cold-start path is still in play.

### Model says "command not found" / "Python is not installed"

**Symptom:** Mid-session, the model issues `run_command` calls
like `python -m py_compile foo.py` that fail in microseconds with
no useful output, then concludes the environment lacks Python.
You know your host has Python.

**Cause (PC-032).** `run_command` runs inside the atlas-proxy
container, which used to be alpine-bare (curl + bash only). Any
`python`, `node`, `gcc`, `make` invocation hit "command not
found" with no stderr surfaced to the UI (PC-033 compounded the
diagnosis problem).

**Fix:** atlas-proxy v3.0.1+ ships with python3, py3-pip,
nodejs/npm, gcc, g++, make, and git baked into the runtime image.
Targeted rebuild after pulling: `docker compose up -d --build
atlas-proxy`. Verify:

```bash
docker compose exec atlas-proxy sh -c "which python3 node gcc make"
```

If anything is missing, you're on an older image; rebuild.

The proper architectural fix (route `run_command` through the
sandbox container's `/execute` endpoint, which is properly
isolated) is tracked as a Phase 1+ follow-up under PC-032.

### I closed Aider but my GPU stayed busy

**Symptom:** You hit Ctrl-C in Aider, close the terminal, or your
SSH session drops mid-generation — but `nvidia-smi` shows
llama-server still pegged at high utilization for the next 30 sec
to several minutes.

**Cause (PC-036).** The proxy was not propagating Aider's
cancellation down to llama-server. Aider closing its TCP
connection cancelled `r.Context()` at the proxy boundary, but
the agent loop and its LLM calls used `context.Background()`
instead, so they kept running until the model finished
generating to `max_tokens` or hit a stop sequence.

**Fix:** atlas-proxy v3.0.1+ wires the request context all the
way through: `handleStreamingChat` → `runInternalAgentLoop` →
`AgentContext.Ctx` → `callLLMConstrained`'s HTTP request. The
agent loop also checks `ctx.Done()` at each turn boundary. Now
when Aider disconnects:
- The current LLM call's TCP connection to llama-server closes
  (within ~1 token tick).
- llama.cpp detects the disconnect on the slot and aborts
  generation, releasing the GPU.
- The agent loop bails on the next turn and logs
  `[agent] cancelled at turn N: context canceled`.

Verify after rebuild: tail `docker compose logs -f atlas-proxy`,
trigger a long generation, hit Ctrl-C in Aider — you should see
the cancellation log line within a second, and `nvidia-smi`
utilization should fall within 1-2 sec.

### Agent says my file "doesn't exist" but I can see it in `ls`

**Symptom:** You're in a project directory, the file is right
there in `ls`, but ATLAS replies with something like "the file
doesn't exist in the current workspace" and refuses to edit it.

**Cause (PC-038).** The proxy container's `/workspace` is
bind-mounted from `ATLAS_PROJECT_DIR` (default: the directory
you ran `docker compose up` from). If you `cd` into a project
elsewhere on the host and run `atlas`, the proxy used to still
only see the original directory.

**Fix:** the `atlas` CLI now auto-aligns the proxy's
`/workspace` to your CWD on every invocation. When a mismatch is
detected, you'll see:

```
$ atlas
  Aligning proxy workspace → /home/isaac/your/project
  ...
```

The container is recreated in ~5-8s on cached images. Same-CWD
invocations are no-ops. If you've pre-mounted a parent (e.g.
`ATLAS_PROJECT_DIR=$HOME`), no realignment happens because
`$HOME` already covers your CWD.

**Disable** with `ATLAS_AUTO_WORKSPACE=0` if you want manual
control:
```bash
ATLAS_AUTO_WORKSPACE=0 atlas
```

**If the model still bails after the bind is correct** —
e.g. you `cd`'d into the right project but the agent still
exits with "Stopped after 3 tool failures with no successful
changes" — Aider didn't `/add` the file, so the model has no
content to work from and tries to discover it. Either:

- `aider /add snake_game.py` once before your first prompt,
  then ATLAS gets the file content directly from Aider, or
- Just be explicit: "fix snake_game.py at line 95 — the
  curses bounds are wrong." Naming the file in the prompt
  triggers the model to run `find_file` / `list_directory`
  to locate it. (PC-039 ensures empty-path tool calls get
  redirected with a hint instead of silently failing.)

**Manual alternative** (the path before PC-038 landed):
```bash
cd /home/isaac/ATLAS
ATLAS_PROJECT_DIR=/path/to/your/project \
    docker compose up -d --no-build atlas-proxy \
    --no-deps --force-recreate
cd /path/to/your/project
atlas
```

### Model "loses" a file it just created (search_files returns 0)

**Symptom:** The model writes `foo.py`, then on the next turn it
runs `search_files "foo\.py"` and gets 0 matches. It then tries
to recreate `foo.py` from scratch, and Aider prompts "Create new
file? [Yes]:".

**Cause (PC-028).** The old `search_files` description said it
searches "in files," which the model read as "in filenames." The
tool actually greps file *contents* — and file contents don't
contain the literal string "foo.py", so it returned zero matches.

**Fix:** atlas-proxy v3.0.1+ updates the `search_files`
description to "search inside file CONTENTS" and adds a new
`find_file` tool for name-based lookups. The model now uses
`find_file "foo\.py"` (or `list_directory`) to check whether a
file exists.

If you're running an older version: workaround is to be explicit
in the prompt ("the file already exists at foo.py — modify it
in place"). The model can still call `read_file` directly to
confirm a file exists.

### Empty Response

**Symptom:** Aider shows the completion summary but no file content was produced.

**Cause:** The model emitted a `done` signal without making any file changes. This can happen with:
- Very short conversational prompts ("hi", "thanks")
- Ambiguous requests where the model doesn't know what file to create

**Fix:** Be more specific. Tell the model exactly what file to create or edit.

### Wrong Working Directory

**Symptom:** Files created in the wrong location. `list_directory` shows unexpected contents.

**Cause:** The proxy detects the project directory by finding the most recently modified `.aider.chat.history.md` file. If you have multiple Aider sessions open, the newest one wins.

**Fix:** Close other Aider sessions, or `cd` into the correct project directory before running `atlas`.

### "Model not found" Error

**Symptom:** Aider fails to start with a model-related error.

**Fix:** Ensure both Aider config files exist in the ATLAS root:
```bash
ls -la .aider.model.settings.yml .aider.model.metadata.json
```

These are included in the repository. If missing, re-clone or restore from backup. They tell Aider to use the `openai/atlas` model pointing at the proxy.

---

## Performance

### Slow Generation (~2 tok/s)

The model is running on CPU instead of GPU. Check:
1. `nvidia-smi` — is llama-server listed as a GPU process?
2. `--n-gpu-layers 99` — are all layers offloaded?
3. NVIDIA Container Toolkit — is the container runtime configured for GPU access?

**Expected performance:** ~51 tok/s on RTX 5060 Ti 16GB with grammar enforcement.

### V3 Pipeline Takes Several Minutes

This is normal for T2 files. The V3 pipeline makes multiple LLM calls:
- **Probe only (best case):** ~10-15 seconds (1 generation + 1 score + 1 test)
- **Phase 1 generation:** ~1-2 minutes (PlanSearch + DivSampling + scoring)
- **Phase 3 repair:** ~2-5 minutes (PR-CoT + Refinement + Derivation, if needed)

To get faster (but lower quality) results:
- Keep files under 50 lines (stays T1, no V3)
- Reduce logic complexity (fewer functions, control flow)
- V3 only fires when truly needed — simple files are written instantly

### High RAM Usage

**Symptom:** System becomes sluggish or services get OOMKilled.

**Expected RAM usage:**
- llama-server: ~8 GB (model in VRAM, minimal RAM)
- geometric-lens: ~200 MB (PyTorch runtime + models)
- v3-service: ~150 MB (PyTorch runtime)
- sandbox: ~100 MB (base, spikes during compilation)
- atlas-proxy: ~30 MB (Go binary)

**Total:** ~500 MB RAM + 8.2 GB VRAM. If you have less than 14 GB system RAM, other services may compete for memory.

---

## Getting Help

If your issue isn't listed here:
1. Check service logs: `docker compose logs <service-name>`
2. Check the proxy health endpoint: `curl http://localhost:8090/health`
3. See [CONFIGURATION.md](CONFIGURATION.md) for all environment variables
4. Open an issue on [GitHub](https://github.com/itigges22/ATLAS/issues)
