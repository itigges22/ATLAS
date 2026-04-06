# ATLAS Proxy

Production inference proxy that sits between Aider and inference-server (Qwen3.5-9B).
Implements the ATLAS pipeline: intent routing, spec generation, sandbox verification, C(x)/G(x) scoring, and verify-repair loops.

## Architecture

```
User → Aider → ATLAS Proxy (:8090) → inference-server (:8080)
                    ↕                      ↕
              RAG API (:8099)        Sandbox (:30820)
              C(x)+G(x) scoring     Code execution + PTY
```

## Pipeline Stages

1. **Intent Classification** — Model-based (few-shot), returns T0-T3
2. **Spec Generation** — For T2+ tasks, generates implementation checklist
3. **Code Generation** — Streams from inference-server with spec injected into prompt
4. **Sandbox Verification** — Runs generated code (PTY wrapper for interactive programs)
5. **Error Analysis** — Parses tracebacks, identifies error type and recovery strategy
6. **Verify-Repair Loop** — Up to 3 repair iterations if sandbox fails
7. **C(x)/G(x) Scoring** — Quality gate on every response
8. **Best-of-K** — Triggered for T3 tasks or low G(x) scores

## Tier Classification

| Tier | Description | Pipeline |
|------|-------------|----------|
| T0 | Conversational (hi, thanks) | Direct to inference-server, no pipeline |
| T1 | Simple (fix typo, add import) | Direct + G(x) scoring |
| T2 | Medium (refactor, write tests) | Spec + verify + G(x) |
| T3 | Hard (new app, architecture) | Spec + verify + best-of-K + G(x) |

## Usage

```bash
# Start all services
atlas

# Or manually
atlas-proxy                          # starts proxy on :8090
OPENAI_API_BASE=http://localhost:8090 aider --model openai/atlas
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| ATLAS_FOX_URL | http://localhost:8080 | inference-server inference server |
| ATLAS_RAG_URL | http://localhost:8099 | RAG API with C(x)+G(x) |
| ATLAS_SANDBOX_URL | http://localhost:30820 | Code execution sandbox |
| ATLAS_PROXY_PORT | 8090 | Proxy listen port |
| ATLAS_MODEL_NAME | Qwen3.5-9B-Q6_K | Model name for inference-server |

## Build

```bash
cd atlas-proxy && go build -o ~/.local/bin/atlas-proxy .
```
