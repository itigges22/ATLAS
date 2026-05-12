# ATLAS Proxy

Local inference proxy that hosts the structured agent endpoint
(`/v1/agent`), the typed event broker (`/events`, PC-061), and the
cancel hook (`/cancel`, PC-062) the TUI drives. Plain OpenAI traffic
on `/v1/chat/completions` and unmatched paths pass through to
llama-server unchanged.

## Architecture

```
TUI ─┐
     ├─→ ATLAS Proxy (:8090) ──┬─→ llama-server  (:8080)
CLI ─┘                          ├─→ V3 service    (:8070)
                                ├─→ Lens / RAG    (:8099)
                                └─→ Sandbox       (:30820)
```

## Agent loop (every turn)

Each user message drives an agent loop that runs until the model emits
`{"type":"done"}` or hits the turn cap:

1. **Pre-flight plan** (PC-179 / PC-206) — `v3-service` `/v3/plan` is
   called to seed an explicit step list. 3 candidates sampled, scored
   heuristically, best plan pinned. The active step is injected into
   the system prompt every turn (`plan_reminder.go`).
2. **Grammar-constrained generation** — `llama-server` produces a JSON
   envelope: `tool_call`, `text`, or `done`. GBNF + `response_format:
   json_object` makes invalid output unrepresentable.
3. **Tool dispatch + validation** — 13 tools (`read_file`,
   `search_files`, `list_directory`, `find_file`, `write_file`,
   `edit_file`, `ast_edit`, `delete_file`, `run_command`, `plan_tasks`,
   `run_background`, `tail_background`, `stop_background`). Per-tool
   guardrails: read-tracking, mtime checks, default-deny patterns,
   suspicious-shrinkage guard (`guardrails.go`).
4. **V3 routing for T2+ writes** — when a file edit qualifies (≥ 50
   lines, ≥ 3 logic indicators, both request- and file-tier ≥ T2 per
   PC-042), the edit is offloaded to `v3-service` and the bridge
   re-emits each pipeline stage onto `/events` as a `v3:<stage>`
   envelope.
5. **Adherence + stuck-pattern gates** — every turn is scored against
   the active plan step (`plan_adherence.go`) and three stuck-pattern
   detectors (`tool_repeat.go`, `reasoning_repeat.go`,
   `claim_check.go`). Auto-revise the plan after a configurable
   threshold; bail if the loop is genuinely stuck.

## Tier Classification

| Tier | Description | Treatment |
|------|-------------|-----------|
| T0 | Conversational (hi, thanks) | No agent loop, passthrough to llama-server |
| T1 | Simple / short edits, low complexity | Agent loop runs; tool calls executed directly. No V3 offload. |
| T2 | Multi-file or non-trivial logic, 50+ lines, 3+ logic indicators | Agent loop runs; `write_file` / `edit_file` may route through V3 (PlanSearch / DivSampling / Budget Forcing / PR-CoT / Refinement / Derivation, S\* candidate selection). |
| T3 | Hard tasks (new app, architecture-scale) | Same as T2 with higher turn budget and verification thresholds. |

## Usage

```bash
# Start all services and launch the TUI
atlas

# Or run the proxy standalone
atlas-proxy-v2                          # listens on :8090
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| ATLAS_INFERENCE_URL | http://localhost:8080 | llama-server inference URL |
| ATLAS_LLAMA_URL | (= ATLAS_INFERENCE_URL) | Override for llama-server target |
| ATLAS_LENS_URL | http://localhost:8099 | Lens / RAG API with C(x)+G(x) |
| ATLAS_SANDBOX_URL | http://localhost:30820 | Code execution sandbox |
| ATLAS_V3_URL | http://localhost:8070 | V3 pipeline service |
| ATLAS_PROXY_PORT | 8090 | Proxy listen port |
| ATLAS_MODEL_NAME | Qwen3.5-9B-Q6_K | Model name for llama-server |
| ATLAS_WORKSPACE_DIR | (cwd) | Workspace root for read/write tools |

## Build

```bash
cd proxy && go build -o ~/.local/bin/atlas-proxy-v2 .
```
