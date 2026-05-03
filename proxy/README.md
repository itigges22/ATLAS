# ATLAS Proxy

Local inference proxy that hosts the structured agent endpoint
(`/v1/agent`), the typed event broker (`/events`), and the cancel hook
(`/cancel`) the TUI drives. Plain OpenAI traffic on
`/v1/chat/completions` and unmatched paths pass through to llama-server
unchanged.

## Architecture

```
TUI ─┐
     ├─→ ATLAS Proxy (:8090) ──┬─→ llama-server  (:8080)
CLI ─┘                          ├─→ V3 service    (:8070)
                                ├─→ Lens / RAG    (:8099)
                                └─→ Sandbox       (:30820)
```

## Pipeline Stages (within `write_file` tool calls for T2+)

1. **Intent Classification** — fast heuristic, returns T0-T3
2. **Spec Generation** — for T2+ tasks, produces an implementation checklist
3. **Code Generation** — streams from llama-server with the spec injected
4. **Sandbox Verification** — runs generated code (PTY wrapper for interactive programs)
5. **Error Analysis** — parses tracebacks, picks recovery strategy
6. **Verify-Repair Loop** — up to 3 repair iterations on sandbox fail
7. **C(x)/G(x) Scoring** — quality gate on every response
8. **Best-of-K** — triggered for T3 tasks or low G(x) scores

## Tier Classification

| Tier | Description | Pipeline |
|------|-------------|----------|
| T0 | Conversational (hi, thanks) | Direct to llama-server, no pipeline |
| T1 | Simple (fix typo, add import) | Direct + G(x) scoring |
| T2 | Medium (refactor, write tests) | Spec + verify + G(x) |
| T3 | Hard (new app, architecture) | Spec + verify + best-of-K + G(x) |

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
| ATLAS_AGENT_LOOP | 1 | Enable internal tool-call agent loop |
| ATLAS_WORKSPACE_DIR | (cwd) | Workspace root for read/write tools |

## Build

```bash
cd proxy && go build -o ~/.local/bin/atlas-proxy-v2 .
```
