# ATLAS Architecture

System architecture for ATLAS V3.0.1. Describes the two-layer design, service components, and data flow.

---

## 1. System Overview

ATLAS is a locally-hosted AI coding assistant. It wraps a frozen 9B-parameter LLM in a structured tool-call agent loop with a V3 pipeline that generates diverse code candidates, build-verifies each one, and selects the best using energy-based scoring.

### Two-Layer Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        ATLAS Stack                            │
│                                                               │
│  ┌─────────┐    ┌──────────────────────────────────────────┐ │
│  │  Aider   │───►│         atlas-proxy (Go)                 │ │
│  │  (TUI)   │◄───│                                          │ │
│  └─────────┘    │  OUTER: Agent Loop                        │ │
│                  │  ┌────────────────────────────────────┐   │ │
│                  │  │ 1. Model emits JSON tool call       │   │ │
│                  │  │ 2. Grammar enforcement (json_object)│   │ │
│                  │  │ 3. Per-file tier classification      │   │ │
│                  │  │ 4. Execute tool                      │   │ │
│                  │  │ 5. Return result, loop until done    │   │ │
│                  │  └────────────┬───────────────────────┘   │ │
│                  │               │ T2 write_file/edit_file    │ │
│                  │  INNER: V3 Pipeline                        │ │
│                  │  ┌────────────▼───────────────────────┐   │ │
│                  │  │ Step 1:  Probe baseline             │   │ │
│                  │  │ Step 2:  Constraint Extraction       │   │ │
│                  │  │ Step 3:  PlanSearch (3 plans)        │   │ │
│                  │  │ Step 4:  DivSampling (K candidates)  │   │ │
│                  │  │ Step 5:  Budget Forcing              │   │ │
│                  │  │ Step 6:  Build Verification          │   │ │
│                  │  │ Step 7:  C(x)/G(x) Scoring           │   │ │
│                  │  │ Step 8:  Best-of-K Selection          │   │ │
│                  │  │ Step 9:  S*/Blend-ASC Tiebreaking     │   │ │
│                  │  │ Step 10: Failure Analysis             │   │ │
│                  │  │ Step 11: PR-CoT Repair                │   │ │
│                  │  │ Step 12: Refinement Loop              │   │ │
│                  │  │ Step 13: Metacognitive Evaluation     │   │ │
│                  │  │ Step 14: Write Winning Candidate      │   │ │
│                  │  └────────────────────────────────────┘   │ │
│                  └──────────────────────────────────────────┘ │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ llama-server  │  │geometric-lens│  │   sandbox     │       │
│  │ CUDA, 32K ctx │  │ C(x)/G(x)   │  │ Code exec     │       │
│  │ ~51 tok/s     │  │ AUC 0.9467   │  │ 8 languages   │       │
│  │ Port 8080     │  │ Port 8099    │  │ Port 8020     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

## 2. Services

| Service | Port | Language | Purpose |
|---------|------|----------|---------|
| **llama-server** | 8080 | C++ (llama.cpp) | LLM inference with CUDA + grammar support |
| **geometric-lens** | 8099 | Python (FastAPI) | C(x)/G(x) energy scoring, embedding extraction |
| **v3-service** | 8070 | Python | V3 pipeline HTTP wrapper (PlanSearch, DivSampling, PR-CoT, etc.) |
| **atlas-proxy** | 8090 | Go | Agent loop, tool-call routing, Aider format translation |
| **sandbox** | 8020 | Python (FastAPI) | Isolated code execution (Python, JS, Go, Rust, C, Shell) |

## 3. Agent Loop (Outer Layer)

The proxy receives chat completion requests from Aider and runs an internal agent loop:

1. **Build system prompt** with tool descriptions, project context, and rules
2. **Call llama-server** with `response_format:json_object` — guarantees valid JSON
3. **Parse response** — one of: `tool_call`, `text`, or `done`
4. **Execute tool** (read_file, write_file, edit_file, delete_file, run_command, search_files, list_directory, plan_tasks)
5. **Return result** to model, loop until `done` or max turns

### Grammar Enforcement

llama-server's `response_format: {"type": "json_object"}` forces every model output to be valid JSON. This eliminates format parsing failures that plagued the V3.0 single-shot architecture.

### Per-File Tier Classification

Each `write_file` call is classified independently:

| Tier | Criteria | Action |
|------|----------|--------|
| T1 | Config, data, CSS, markdown, shell, <50 lines | Direct write (instant) |
| T2 | 50+ lines with 3+ logic indicators (functions, control flow, handlers) | V3 pipeline fires |

Logic indicators: `def`, `func`, `function`, `async`, `if`, `else`, `switch`, `for`, `while`, `try`, `catch`, `export default`, `return (`, `className=`, `.map(`, `import {`, etc.

### Exploration Budget

To prevent the model from spending too many turns reading files:
- After 4 consecutive read-only calls: inject "write your changes now"
- After 5: skip the read, return "you already have this information"

### Truncation Prevention

- 32K context window + 32K max_tokens
- Reject `write_file` for existing files >100 lines (forces `edit_file`)
- Detect truncated JSON args before execution
- Error loop breaker: 3 consecutive failures → stop

## 4. V3 Pipeline (Inner Layer)

Activates inside `write_file`/`edit_file` executors for T2 files. The 14-step pipeline:

### Phase 1: Constraint-Driven Generation
1. **Probe** — Score baseline candidate with C(x)/G(x)
2. **Constraint Extraction** — Extract requirements from project context
3. **PlanSearch** — Generate 3 structurally different implementation plans
4. **DivSampling** — Apply perturbation diversity (role/instruction/style)
5. **Budget Forcing** — Control thinking token budget per candidate

### Verification
6. **Build Verification** — Per-file-type: `tsc --noEmit`, `python -m py_compile`, `gcc -fsyntax-only`, `cargo check`, `bash -n`
7. **C(x)/G(x) Scoring** — Energy-based quality scoring via Geometric Lens
8. **Best-of-K Selection** — Winner by: execution pass > C(x) > G(x)
9. **S*/Blend-ASC** — Tiebreaking between passing candidates

### Phase 3: Repair (if 0/K pass)
10. **Failure Analysis** — Categorize why each candidate failed
11. **PR-CoT Repair** — Progressive repair: root cause → fix → verify (2-6 LLM calls)
12. **Refinement Loop** — Iterative cycle with constraint refinement + derivation chains (3-15 LLM calls)
13. **Metacognitive Evaluation** — Confidence assessment, known pitfall injection
14. **Write Winner** — Best candidate written to disk

## 5. Geometric Lens

Neural scoring system that evaluates code without executing it.

### C(x) Cost Field
- Architecture: 4096→512→128→1 MLP (SiLU + Softplus)
- Input: 4096-dim embeddings from llama-server
- Output: Scalar energy (low = likely correct)
- Training: Contrastive ranking loss on 597 LCB embeddings (504 PASS, 93 FAIL)
- Performance: Val AUC 0.9467, separation ratio 2.04x

### G(x) Metric Tensor
- Architecture: 4096→512→4096 with unit-mean normalization
- Purpose: Geometric correction field for solution space curvature
- Training: Contrastive geometric loss with Mahalanobis distance

## 6. VRAM Budget

Running on RTX 5060 Ti 16GB:

| Component | VRAM |
|-----------|------|
| Qwen3.5-9B-Q6_K model | ~6.9 GB |
| KV cache (32K context) | ~1.3 GB |
| **Total llama-server** | **~8.2 GB** |
| Geometric Lens (CPU) | 128 MB (PyTorch) |
| **Free** | **~7.9 GB** |

All V3 pipeline computation runs on CPU. Only llama-server uses the GPU.

## 7. Deployment

### Docker Compose (Recommended)
```yaml
# docker-compose.yml provides:
# - llama-server (CUDA GPU)
# - geometric-lens
# - v3-service
# - atlas-proxy
# - sandbox (isolated)
```

### Bare Metal
The `atlas` command starts all services as local processes and launches Aider.

### K3s
Manifests generated by `scripts/generate-manifests.sh` from `atlas.conf`. Services deploy as pods in the `atlas` namespace.

## 8. Data Flow

### Creating a New File (T1)
```
User prompt → Aider → proxy → model emits write_file → direct write → Aider applies
```

### Creating a Feature File (T2)
```
User prompt → Aider → proxy → model emits write_file → tier=T2 →
  V3: PlanSearch → DivSampling → Build Verify → Score → Select → Write →
Aider applies
```

### Editing Existing Code
```
User prompt → Aider → proxy → model emits read_file → edit_file →
  proxy applies old_str/new_str replacement → Aider applies
```

### Deleting a File
```
User prompt → proxy detects delete request (fast-path) →
  delete from disk → empty SSE response → Aider sees file gone
```
