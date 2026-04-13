# ATLAS Architecture

System architecture for ATLAS V3.0.1. Two-layer design: an outer agent loop handles tool-call orchestration, and an inner V3 pipeline generates diverse code candidates with build verification and energy-based selection.

---

## 1. System Overview

```mermaid
graph LR
    User["User"] --> Aider["Aider"] --> Proxy["atlas-proxy\n:8090"]

    subgraph outer["Outer Layer"]
        Proxy -->|"grammar JSON"| LLM["llama-server\n:8080"]
        Proxy -->|"T2 files"| V3Service["v3-service\n:8070"]
    end

    subgraph inner["Inner Layer"]
        V3Service --> LLM
        V3Service --> Lens["geometric-lens\n:8099"]
        V3Service --> Sandbox["sandbox\n:30820"]
        Lens --> LLM
    end

    style User fill:#333,color:#fff
    style Aider fill:#1a3a5c,color:#fff
    style Proxy fill:#1a3a5c,color:#fff
    style LLM fill:#5c1a1a,color:#fff
    style V3Service fill:#2d5016,color:#fff
    style Lens fill:#2d5016,color:#fff
    style Sandbox fill:#2d5016,color:#fff
```

Services run as containers via Docker Compose (recommended) or as local processes via the `atlas` launcher. Only llama-server uses the GPU. Everything else runs on CPU.

---

## 2. Services

| Service | Port | Language | Purpose |
|---------|------|----------|---------|
| **llama-server** | 8080 | C++ (llama.cpp) | LLM inference with CUDA, grammar-constrained JSON, self-embeddings |
| **atlas-proxy** | 8090 | Go | Agent loop, tool-call routing, tier classification, Aider format translation |
| **v3-service** | 8070 | Python | V3 pipeline HTTP wrapper (PlanSearch, DivSampling, PR-CoT, etc.) |
| **geometric-lens** | 8099 | Python (FastAPI) | C(x) energy scoring, G(x) XGBoost quality prediction, RAG/project indexing |
| **sandbox** | 30820 (host) / 8020 (container) | Python (FastAPI) | Isolated code execution, compilation, linting, test running |

---

## 3. atlas-proxy (Outer Layer)

The proxy receives chat completion requests from Aider and runs an internal agent loop.

```mermaid
graph LR
    subgraph core["Core Loop"]
        Grammar["Grammar"] --> AgentLoop["Agent Loop"] --> TierClass["Tier Classifier"]
    end
    subgraph tools["Tools"]
        ReadF["read_file"] ~~~ WriteF["write_file"] ~~~ EditF["edit_file"] ~~~ RunCmd["run_command"]
    end
    subgraph pipeline["Verify-Repair"]
        VR["Verify-Repair"] --> BOK["Best-of-K"] --> BV["Build Verifier"]
    end
    subgraph format["I/O"]
        AiderFmt["Aider Fmt"] --> V3Bridge["V3 Bridge"] --> ProjDet["Project Detector"]
    end

    core --> tools --> pipeline --> format

    style core fill:#1a3a5c,color:#fff
    style tools fill:#333,color:#fff
    style pipeline fill:#2d5016,color:#fff
    style format fill:#555,color:#fff
```

### Agent Loop Flow

```mermaid
flowchart LR
    Start["Aider msg"] --> Build["Build prompt"] --> Call["llama-server"] --> Parse["Parse JSON"]
    Parse --> Route{Type?}

    Route -->|"tool_call"| Tier{"T2?"}
    Tier -->|"Yes"| V3["V3 Pipeline"] --> Result["Append result"]
    Tier -->|"No"| Exec["Execute tool"] --> Result
    Result --> Budget{"Budget?"}
    Budget -->|"< 4"| Call
    Budget -->|"4"| Warn["Nudge: write now"] --> Call
    Budget -->|"5+"| Skip["Skip read"] --> Call

    Route -->|"text"| Stream["Stream"] --> Call
    Route -->|"done"| Done["End"]

    style Start fill:#1a3a5c,color:#fff
    style Done fill:#333,color:#fff
    style V3 fill:#2d5016,color:#fff
```

### Grammar Enforcement

llama-server's `response_format: {"type": "json_object"}` forces every model output to be exactly one of three valid JSON shapes:

```json
{"type": "tool_call", "name": "<tool_name>", "args": {...}}
{"type": "text", "content": "<message>"}
{"type": "done", "summary": "<summary>"}
```

The JSON schema uses `oneOf` with `additionalProperties: false` and enumerates tool names from the registry. The model cannot produce invalid JSON — token generation is grammar-constrained at the llama-server level.

### Tools

8 tools available to the model:

| Tool | Purpose | Read-only |
|------|---------|-----------|
| `read_file` | Read file contents (with optional offset/limit) | Yes |
| `write_file` | Create new file or overwrite (routes to V3 for T2 files) | No |
| `edit_file` | Replace exact string in file (old_str/new_str) | No |
| `delete_file` | Delete file or empty directory (forces loop exit after) | No |
| `run_command` | Execute shell command (5 min timeout cap) | No |
| `search_files` | Regex search across files (max 200 matches, skips .git/node_modules) | Yes |
| `list_directory` | List directory contents with type and size | Yes |
| `plan_tasks` | Decompose work into parallel tasks with dependencies | No |

### Per-File Tier Classification

Each `write_file`/`edit_file` call is classified independently:

| Tier | Max Turns | Action |
|------|-----------|--------|
| T0 (Conversational) | 5 | Text response only |
| T1 (Simple) | 30 | Direct write — no V3 overhead |
| T2 (Feature) | 30 | V3 pipeline fires |
| T3 (Hard) | 60 | V3 pipeline fires |

**Always T1 (direct write):**
- Config files by name: package.json, tsconfig.json, Dockerfile, Makefile, pyproject.toml, requirements.txt, .gitignore, and ~30 more
- Data files by extension: .json, .yaml, .yml, .toml, .csv, .xml, .env
- Style files: .css, .scss, .less
- Documentation: .md, .txt, .rst
- Shell scripts: .sh, .bash
- Short files: under 50 lines (V3 overhead exceeds quality gain)

**T2 (V3 pipeline):** Files with 50+ lines AND 3+ logic indicators. Logic indicators include function definitions (`def`, `func`, `function`, `async`), control flow (`if`, `else`, `switch`, `for`, `while`, `try`), API patterns (`export default`, `app.get`, `router.`, `NextResponse`), state management (`useState`, `useEffect`, `dispatch`), and JSX patterns (`return (`, `className=`, `.map(`).

### Safety Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| Conversation trim | Keep 12 messages max (system + first user + last 8) | Prevent context overflow |
| write_file for existing files | Reject if file > 100 lines | Forces edit_file for targeted changes |
| Truncation detection | JSON parse check on tool args | Catches truncated model output |
| Error loop breaker | 3 consecutive failures | Stops runaway failure cycles |
| Exploration budget warning | 4 consecutive read-only calls | Inject "write your changes now" |
| Exploration budget skip | 5+ consecutive read-only calls | Skip the read, return warning |
| Command stdout | 8,000 chars max | Prevent context flooding |
| Command stderr | 4,000 chars max | Prevent context flooding |
| Search results | 200 matches max | Prevent context flooding |
| File search | Skip files > 1 MB | Performance |

---

## 4. V3 Pipeline (Inner Layer)

Activates inside `write_file`/`edit_file` executors for T2+ files. The pipeline has four phases with early exits at every stage.

### Pipeline Flow

```mermaid
flowchart LR
    Entry["T2 detected"] --> Probe["Probe"] --> Score1["C(x)/G(x)"] --> SB1["Sandbox"]
    SB1 --> Pass1{"Pass?"}
    Pass1 -->|"Yes"| Done["Done"]

    Pass1 -->|"No"| PS["PlanSearch"] --> DS["DivSampling"] --> BF["BudgetForcing"] --> Build["Build Check"] --> Score2["Score K"] --> SB2["Test K"]

    SB2 --> AnyPass{"Passed?"}
    AnyPass -->|"2+"| SStar["S* Tiebreak"] --> Done
    AnyPass -->|"1"| Select["Lens Select"] --> Done

    AnyPass -->|"0"| FA["Failure Analysis"] --> PRCOT["PR-CoT"]
    PRCOT --> PRPass{"Pass?"}
    PRPass -->|"Yes"| Done
    PRPass -->|"No"| Refine["Refinement"]
    Refine --> RefPass{"Pass?"}
    RefPass -->|"Yes"| Done
    RefPass -->|"No"| Derive["Derivation"] --> Done

    style Entry fill:#1a3a5c,color:#fff
    style Done fill:#333,color:#fff
    style Probe fill:#1a3a5c,color:#fff
    style PS fill:#1a3a5c,color:#fff
    style DS fill:#1a3a5c,color:#fff
    style BF fill:#1a3a5c,color:#fff
    style SStar fill:#2d5016,color:#fff
    style Select fill:#2d5016,color:#fff
    style Score1 fill:#2d5016,color:#fff
    style Score2 fill:#2d5016,color:#fff
    style SB1 fill:#2d5016,color:#fff
    style SB2 fill:#2d5016,color:#fff
    style Build fill:#2d5016,color:#fff
    style PRCOT fill:#5c3a1a,color:#fff
    style Refine fill:#5c3a1a,color:#fff
    style Derive fill:#5c3a1a,color:#fff
    style FA fill:#5c3a1a,color:#fff
```

Legend: blue = generation, green = verification/selection, brown = repair.

### Phase Details

**Phase 0: Probe** generates a single baseline candidate with progressive retry (light → standard → /nothink). Scored with C(x)/G(x) and tested in sandbox. If it passes, pipeline exits immediately.

**Phase 1: Constraint-Driven Generation**

- **PlanSearch** generates 3 structurally different implementation plans by extracting distinct constraint sets
- **DivSampling** applies perturbation diversity: 4 roles (competitive_programmer, systems_engineer, mathematician, pragmatist) + 4 instructions (step_by_step, edge_case_first, complexity_aware, constraint_driven) + 4 styles (functional, pythonic, optimize_iteratively, structured)
- **Budget Forcing** controls thinking token allocation:

| Tier | Thinking Tokens | Wait Injection |
|------|----------------|----------------|
| nothink | 0 | /nothink prompt |
| light | 1,024 | None |
| standard | 2,048 | If thinking ends < 512 tokens |
| hard | 4,096 | If thinking ends < 1,024 tokens |
| extreme | 8,192 | If thinking ends < 2,048 tokens |

Wait injection appends "Wait, let me reconsider.\n" to force longer thinking. Tier selection driven by C(x) energy.

**Phase 2: Verification and Selection**

- **Build Verification**: Python (`py_compile`), TypeScript (`tsc --noEmit`), JavaScript (`node --check`), Go (`go build`), Rust (`cargo check`), C/C++ (`gcc/g++ -fsyntax-only`), Shell (`bash -n`). Framework overrides for Next.js, React, Flask, Django, Express.
- **S* Tiebreaking** (2+ passing): generates edge-case inputs, runs both candidates, majority wins
- **Lens Selection** (1 passing or fallback): sort by C(x) energy, lowest wins

**Phase 3: Repair** (if 0/K pass) — three strategies, sequential with early exit:

- **Failure Analysis**: categorize failures (wrong_algorithm, implementation_bug, edge_case_miss, time_limit, format_error, partial_correct)
- **Metacognitive Evaluation**: inject compensating constraints from known Qwen3.5 failure patterns
- **PR-CoT**: 4 perspectives (logical_consistency, information_completeness, biases, alternative_solutions) x (analysis + repair) = ~8 LLM calls, up to 3 rounds
- **Refinement Loop**: Failure Analysis → Constraint Refinement → Code Gen → Test → Learn. 2 iterations, 120s budget, ~5+ LLM calls each. Cosine distance filtering (>= 0.15) prevents hypothesis repetition
- **Derivation Chains**: decompose into up to 5 sub-problems, sandbox-verify each, compose final. ~7+ LLM calls

### Module Map

19 Python modules in `benchmark/v3/` orchestrated by `v3-service/main.py`:

```mermaid
graph LR
    Main["main.py"] --> PS["PlanSearch 1A"]
    Main --> DS["DivSampling 1B"]
    Main --> BF["BudgetForcing 1C"]
    Main --> BASC["BlendASC 2A"]
    Main --> REASC["ReASC 2B"]
    Main --> SSTAR["S* 2C"]
    Main --> CS["CandidateSelection"]
    Main --> FA["FailureAnalysis 3A"]
    Main --> CR["ConstraintRefiner 3B"]
    Main --> PRCOT["PR-CoT 3C"]
    Main --> DC["DerivationChains 3D"]
    Main --> RL["RefinementLoop 3E"]
    Main --> MC["Metacognitive 3F"]
    Main --> ACE["ACE 3G"]
    Main --> STG["SelfTestGen"]
    Main --> LF["LensFeedback"]
    Main --> ES["EmbeddingStore"]

    RL --> FA
    RL --> CR
    RL --> DC
    BASC --> BF
    REASC --> BF
    LF --> BASC
    LF --> BF

    style Main fill:#333,color:#fff
    style PS fill:#1a3a5c,color:#fff
    style DS fill:#1a3a5c,color:#fff
    style BF fill:#1a3a5c,color:#fff
    style BASC fill:#2d5016,color:#fff
    style REASC fill:#2d5016,color:#fff
    style SSTAR fill:#2d5016,color:#fff
    style CS fill:#2d5016,color:#fff
    style FA fill:#5c3a1a,color:#fff
    style CR fill:#5c3a1a,color:#fff
    style PRCOT fill:#5c3a1a,color:#fff
    style DC fill:#5c3a1a,color:#fff
    style RL fill:#5c3a1a,color:#fff
    style MC fill:#5c3a1a,color:#fff
    style ACE fill:#5c3a1a,color:#fff
    style STG fill:#333,color:#fff
    style LF fill:#333,color:#fff
    style ES fill:#333,color:#fff
```

Legend: blue = Phase 1 (generation), green = Phase 2 (selection), brown = Phase 3 (repair), gray = utilities.

---

## 5. Geometric Lens

Neural scoring system that evaluates code quality without executing it by analyzing the geometric structure of model embeddings. Runs entirely on CPU. Also serves as the RAG API for project indexing, retrieval, confidence routing, and pattern caching.

#### Why "Geometric Lens"?

The core idea behind the Geometric Lens comes from a simple premise: stop scaling models and start wrapping them in intelligent infrastructure. Jose Crespo's ["Everyone's Wrong About AI Programming"](https://www.josecrespophd.org/p/everyones-wrong-about-ai-programming) argues that AI-generated code drifts toward errors because current LLMs operate in flat embedding spaces where correct and incorrect code paths cost the same. The solution is to build an energy landscape around the model where correct code is "downhill" and incorrect code is "uphill."

Anthropic's [Manipulating Manifolds](https://transformer-circuits.pub/2025/linebreaks/index.html) research provides evidence that transformers already create manipulable geometric structures in their embedding space - the raw material is already there. Bar et al.'s [Geometric Unification of Generative AI](https://arxiv.org/html/2510.00666v1) formalizes how distance functions on data manifolds can be learned and used for scoring.

ATLAS implements this with two complementary models. C(x) is a learned energy function (4096-to-512-to-128-to-1 MLP) over the model's own embeddings. Each code candidate gets embedded by llama-server, and C(x) scores where it sits in that geometry. Low energy means the candidate clusters with known-correct code. High energy means it clusters with known-incorrect code. No external oracle, no execution required - just the geometry of the model's own representations.

G(x) is the metric tensor - a diagonal tensor in PCA-reduced embedding space that captures how the energy landscape curves in different directions. Where C(x) answers "how good is this candidate?", G(x) answers "which direction should we move to improve it?" The correction engine uses G(x) to compute geometry-aware gradient steps (`-α · G⁻¹ · ∇C`), steering candidates downhill toward correctness along the natural curvature of the manifold rather than taking naive gradient steps. G(x) is implemented and deployed in V3.0.1.

### Scoring Models

```mermaid
graph LR
    EE["Embedding Extractor\nllama-server /embedding\n4096-dim"] --> CX["C(x) Cost Field\n4096→512→128→1\nSiLU + Softplus"]
    EE --> GX["G(x) XGBoost\nPCA(128) + classifier"]
    CX --> SVC["Service Layer\nevaluate_combined()"]
    GX --> SVC
    SVC --> V{"Verdict"}
    V -->|">= 0.7"| LC["likely_correct"]
    V -->|">= 0.3"| UN["uncertain"]
    V -->|"< 0.3"| LI["likely_incorrect"]

    TR["Training Pipeline\ncontrastive ranking loss"] --> CX
    EWC["EWC\nFisher information\nprevents catastrophic forgetting"] --> TR
    RB["Replay Buffer\ndomain-stratified\n30% old / 70% new"] --> TR

    MT["Metric Tensor\ndiagonal G(x) in PCA space\n(code exists, not deployed)"] -.-> CORR["Correction Engine\n-α · G⁻¹ · ∇C"]

    style EE fill:#333,color:#fff
    style CX fill:#2d5016,color:#fff
    style GX fill:#2d5016,color:#fff
    style SVC fill:#333,color:#fff
    style TR fill:#1a3a5c,color:#fff
    style EWC fill:#1a3a5c,color:#fff
    style RB fill:#1a3a5c,color:#fff
    style MT fill:#555,color:#ccc
    style CORR fill:#555,color:#ccc
```

| Model | Architecture | Training Data | Performance |
|-------|-------------|---------------|-------------|
| **C(x)** | 4096→512→128→1 MLP (SiLU, Softplus) | 597 LCB embeddings (504 PASS, 93 FAIL) | Val AUC 0.9467, sep 2.04x |
| **G(x)** | PCA(4096→128) + XGBoost | 13,398 embeddings (4,835 PASS, 8,563 FAIL) | PCA 80.8% variance |

C(x) normalization: `1 / (1 + exp(-(energy - 19.0) / 2.0))` → [0, 1]. Parameters: 2,163,457 (~8.7 MB).

> **Note:** Model weights (.pt, .pkl files) are not committed to the repository — they are built during training and baked into the container image or mounted at runtime. When model files are absent, the service degrades gracefully: C(x) returns neutral energy, G(x) returns `gx_score: 0.5` and `verdict: "unavailable"`. Training data and weights are available on [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS).

### RAG / PageIndex V2

```mermaid
graph LR
    subgraph indexing["Indexing Pipeline"]
        AST["AST Parser\ntree-sitter Python"] --> TB["Tree Builder\nhierarchical index"]
        TB --> BM25I["BM25 Index\ninverted index, k1=1.5"]
        TB --> SUM["Summarizer\nLLM-generated summaries"]
        BM25I --> PERS["Persistence\nJSON to disk"]
        SUM --> PERS
    end

    subgraph retrieval["Retrieval"]
        BM25S["BM25 Searcher\nmin_score=0.1, top_k=20"]
        TreeS["Tree Searcher\nLLM-guided traversal\nmax_depth=6, max_calls=40"]
        HYB["Hybrid Retriever\nroutes: bm25_first / tree_only / both"]
        BM25S --> HYB
        TreeS --> HYB
    end

    style indexing fill:#1a3a5c,color:#fff
    style retrieval fill:#2d5016,color:#fff
```

### Confidence Router & Pattern Cache

```mermaid
graph LR
    subgraph router["Confidence Router"]
        SIG["Signal Collector\npattern_cache, retrieval_confidence\nquery_complexity, geometric_energy"]
        DIFF["Difficulty Estimator\nweighted fusion → D(x)"]
        TS["Thompson Sampling\nBeta(α,β) posteriors\nper-route cost weighting"]
        FB["Feedback Recorder\nRedis-backed"]
        FC["Fallback Chain\nCACHE_HIT → FAST_PATH\n→ STANDARD → HARD_PATH"]
        SIG --> DIFF --> TS --> FC
        FB --> TS
    end

    subgraph cache["Pattern Cache"]
        PS["Pattern Store\nRedis: STM (100) / LTM / PERSISTENT"]
        PM["Pattern Matcher\nBM25 over summaries"]
        PE["Pattern Extractor\nLLM-driven"]
        PSC["Pattern Scorer\nEbbinghaus decay"]
        COO["Co-occurrence Graph\nlinked pattern retrieval"]
        PE --> PS
        PS --> PM
        PM --> PSC
        PS --> COO
    end

    style router fill:#5c3a1a,color:#fff
    style cache fill:#5c3a1a,color:#fff
```

4 routes with cost-weighted Thompson Sampling: CACHE_HIT (cost=1, k=0) → FAST_PATH (cost=50, k=1) → STANDARD (cost=300, k=5) → HARD_PATH (cost=1500, k=20).

---

## 6. Sandbox

Isolated code execution with compilation, testing, and linting.

```mermaid
graph LR
    subgraph executors["Language Executors"]
        Py["Python\npylint (0-10) + pytest"]
        JS["JavaScript\nNode.js 20"]
        TS["TypeScript\ntsc --noEmit + tsx"]
        Go["Go 1.22\ngo build + run"]
        Rust["Rust stable\nrustc + run"]
        C["C / C++\ngcc/g++ -Wall"]
        Bash["Bash\nbash -n + run"]
    end

    subgraph support["Support"]
        Syn["Syntax Checker\nper-language AST validation"]
        Err["Error Classifier\n15 types: SyntaxError, NameError\nTypeError, CompileError, Timeout..."]
        Trunc["Output Truncation\nstdout: 4000 chars\nstderr: 2000 chars"]
    end

    style executors fill:#2d5016,color:#fff
    style support fill:#333,color:#fff
```

Language aliases accepted: `py`/`python3` (Python), `js`/`node` (JavaScript), `ts` (TypeScript), `golang` (Go), `rs` (Rust), `c++` (C++), `sh`/`shell` (Bash). Max execution time: 60s. Max memory: 512 MB. Workspace: `/tmp/sandbox` (tmpfs).

---

## 7. VRAM Budget

Running on RTX 5060 Ti 16GB with Docker Compose defaults (32K context):

| Component | VRAM |
|-----------|------|
| Qwen3.5-9B-Q6_K model weights | ~6.9 GB |
| KV cache (32K context) | ~1.3 GB |
| **Total llama-server** | **~8.2 GB** |
| Geometric Lens | 0 (CPU-only, ~12 MB RAM for models, ~128 MB for PyTorch runtime) |
| v3-service | 0 (CPU-only) |
| sandbox | 0 (CPU-only) |
| atlas-proxy | 0 (Go binary, ~30 MB RAM) |
| **Free VRAM** | **~7.8 GB** |

All computation outside of llama-server runs on CPU. The GPU is used exclusively for LLM inference and embedding extraction.

---

## 8. Deployment

### Docker Compose (Recommended)

```mermaid
graph LR
    LLM["llama-server"] -->|"healthy"| GL["geometric-lens"] -->|"healthy"| AP["atlas-proxy"]
    LLM -->|"healthy"| V3["v3-service"] -->|"healthy"| AP
    SB["sandbox"] -->|"healthy"| AP

    style LLM fill:#5c1a1a,color:#fff
    style GL fill:#2d5016,color:#fff
    style V3 fill:#2d5016,color:#fff
    style SB fill:#2d5016,color:#fff
    style AP fill:#1a3a5c,color:#fff
```

`llama-server` and `sandbox` start independently (no dependencies). `geometric-lens` and `v3-service` wait for `llama-server` to be healthy. `atlas-proxy` waits for all four services. First run builds container images (several minutes); subsequent starts are fast.

### Bare Metal

The `atlas` CLI (`pip install -e .`) talks directly to services on their default ports. The bash launcher script can start all services as local processes and launch Aider, or detect a running Docker Compose stack and connect to it.

### K3s

Manifests in `templates/` are processed by `scripts/generate-manifests.sh` from `atlas.conf`. Services deploy as pods in the `atlas` namespace with NodePort exposure. K3s deployment uses the entrypoint scripts in `inference/` which support extended context (160K), KV cache quantization (q8_0/q4_0), flash attention, and mlock.

---

## 9. Data Flow

### T1: Simple File Write

```mermaid
sequenceDiagram
    participant U as User
    participant A as Aider
    participant P as atlas-proxy :8090
    participant L as llama-server :8080

    U->>A: "Create a config file"
    A->>P: POST /v1/chat/completions (SSE)
    P->>L: POST /v1/chat/completions<br/>response_format: json_object
    L-->>P: {"type":"tool_call","name":"write_file","args":{...}}
    Note over P: Tier = T1 (config file)<br/>Direct write, no V3
    P-->>P: Write file to disk
    P-->>A: SSE stream: file content
    A-->>U: File created
```

One LLM call. No V3 overhead.

### T2: Feature File Write

```mermaid
sequenceDiagram
    participant U as User
    participant A as Aider
    participant P as atlas-proxy :8090
    participant L as llama-server :8080
    participant V as v3-service :8070
    participant G as geometric-lens :8099
    participant S as sandbox :30820

    U->>A: "Create a REST API handler"
    A->>P: POST /v1/chat/completions (SSE)
    P->>L: POST /v1/chat/completions<br/>response_format: json_object
    L-->>P: {"type":"tool_call","name":"write_file","args":{...}}
    Note over P: Tier = T2 (50+ lines, logic)<br/>Route to V3

    P->>V: POST /v3/generate (SSE)
    Note over V: Phase 0: Probe
    V->>L: POST /v1/chat/completions (generate code)
    L-->>V: probe candidate
    V->>L: POST /v1/embeddings (4096-dim)
    L-->>V: embedding vector
    V->>G: POST /internal/lens/gx-score
    G-->>V: {cx_energy, gx_score, verdict}
    V->>S: POST /execute (test probe)
    S-->>V: {success: false}

    Note over V: Phase 1: PlanSearch + DivSampling
    V->>L: POST /v1/chat/completions (x K candidates)
    L-->>V: K candidates
    V->>S: POST /execute (test each)
    S-->>V: {success: true} for candidate 2

    Note over V: Phase 2: Lens select winner
    V->>G: POST /internal/lens/gx-score
    G-->>V: scores

    V-->>P: SSE result: winning code
    P-->>P: Write file to disk
    P-->>A: SSE stream: file content
    A-->>U: File created
```

Minimum 3 llama-server calls (1 probe generation + 1 self-test generation + 1 embedding extraction). Maximum 30+ if Phase 3 repair engages all strategies.

### Edit Existing Code

```mermaid
sequenceDiagram
    participant U as User
    participant A as Aider
    participant P as atlas-proxy :8090
    participant L as llama-server :8080

    U->>A: "Fix the bug in auth.py"
    A->>P: POST /v1/chat/completions (SSE)
    P->>L: POST /v1/chat/completions<br/>response_format: json_object
    L-->>P: {"type":"tool_call","name":"read_file","args":{"path":"auth.py"}}
    P-->>P: Read file from disk
    P->>L: POST /v1/chat/completions (with file content)
    L-->>P: {"type":"tool_call","name":"edit_file","args":{"old_str":"...","new_str":"..."}}
    P-->>P: Apply old_str→new_str replacement
    P->>L: POST /v1/chat/completions (with edit result)
    L-->>P: {"type":"done","summary":"Fixed auth bug"}
    P-->>A: SSE stream: edited content
    A-->>U: File updated
```

Existing files over 100 lines are rejected for `write_file` — the model must use `edit_file` with targeted changes.
