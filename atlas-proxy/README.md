# ATLAS Proxy

Production inference proxy that sits between Aider and vLLM gen instance (Qwen3.5-9B).
Implements the ATLAS pipeline: intent routing, spec generation, sandbox verification, C(x)/G(x) scoring, and verify-repair loops.

## Architecture

```
User → Aider → ATLAS Proxy (:8090) → vLLM gen   (:8000)
                    ↕                  vLLM embed (:8001)
              Lens (:31144)        Sandbox (:30820)
              C(x)+G(x) scoring     Code execution + PTY
```

## Pipeline Stages

1. **Intent Classification** — Model-based (few-shot), returns T0-T3
2. **Spec Generation** — For T2+ tasks, generates implementation checklist
3. **Code Generation** — Streams from vLLM gen instance with spec injected into prompt
4. **Sandbox Verification** — Runs generated code (PTY wrapper for interactive programs)
5. **Error Analysis** — Parses tracebacks, identifies error type and recovery strategy
6. **Verify-Repair Loop** — Up to 3 repair iterations if sandbox fails
7. **C(x)/G(x) Scoring** — Quality gate on every response
8. **Best-of-K** — Triggered for T3 tasks or low G(x) scores

## Tier Classification

| Tier | Description | Pipeline |
|------|-------------|----------|
| T0 | Conversational (hi, thanks) | Direct to vLLM gen instance, no pipeline |
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
| LLAMA_GEN_URL | http://localhost:8000 | vLLM gen instance (chat/completions) |
| ATLAS_LENS_URL | http://localhost:31144 | Geometric Lens (C(x) + G(x) scoring) |
| ATLAS_SANDBOX_URL | http://localhost:30820 | Code execution sandbox |
| ATLAS_V3_URL | http://localhost:8070 | V3 pipeline service (PlanSearch, etc.) |
| ATLAS_PROXY_PORT | 8090 | Proxy listen port |
| LLAMA_GEN_MODEL | qwen3.5-9b | vLLM `--served-model-name` for the gen instance |
| ATLAS_AGENT_LOOP | 1 | When `1`, run the internal tool-call agent loop instead of forwarding to vLLM directly |

## Build

```bash
cd atlas-proxy && go build -o ~/.local/bin/atlas-proxy .
```
