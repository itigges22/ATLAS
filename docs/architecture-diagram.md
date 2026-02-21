# ATLAS V2 Architecture Diagram

<div align="center">

```mermaid
flowchart TB
  subgraph MaaS["MaaS Layer (API + Auth)"]
    AP[api-portal<br/>User registration, JWT auth<br/>API key mgmt, /v1/models<br/>Port 3000 / NodePort 30000]
    LP[llm-proxy<br/>Reverse proxy to llama-server<br/>API key validation + rate limiting<br/>Port 8000 / NodePort 30080]
  end

  subgraph Input
    Problem[Coding Problem]
  end

  subgraph Routing["Confidence Router"]
    SC[Signal Collector<br/>4 signals: cache, retrieval,<br/>complexity, geometric energy]
    DE[Difficulty Estimator<br/>Weights: 0.30 / 0.25 / 0.20 / 0.25]
    TS[Thompson Sampling<br/>Beta posteriors per bin x route<br/>Cost-weighted selection]
    AK[Adaptive-k Selection<br/>CACHE_HIT k=0 / FAST k=1<br/>STANDARD k=5 / HARD k=20]
  end

  subgraph Generation["Best-of-K Pipeline"]
    LS[llama-server Server A<br/>Qwen3-14B-Q4_K_M<br/>+ Qwen3-0.6B-Q8_0 Draft<br/>Spec decode ON, embeddings OFF<br/>Port 8000 / NodePort 32735]
    EM[llama-server Server B<br/>nomic-embed-text-v1.5 Q8_0<br/>Embeddings ON, 768-dim<br/>Port 8001 / NodePort 32736]
    PC[Pattern Cache<br/>Redis + Ebbinghaus Decay<br/>STM / LTM tiers]
  end

  subgraph Evaluation["Candidate Selection"]
    GL[Geometric Lens<br/>Cost Field C x 2.7M params<br/>Metric Tensor G x 5.2M params]
    SB[Sandbox<br/>K8s service, port 8020<br/>Isolated code execution + testing<br/>Energy-sorted early exit]
  end

  subgraph Knowledge["Context Retrieval"]
    PI[PageIndex RAG<br/>AST Tree Index + BM25<br/>LLM-guided Tree Search]
  end

  subgraph Workers["Async Processing"]
    TW[task-worker<br/>Polls Redis queues<br/>Runs ralph-loop<br/>Calls sandbox over HTTP]
  end

  subgraph Feedback["Continuous Learning"]
    FR[Feedback Recorder<br/>Thompson state updates]
    RT[Lens Retrain<br/>BCE on pass/fail embeddings<br/>Hot-reload weights]
  end

  subgraph Storage
    RD[(Redis<br/>Pattern cache, Thompson state<br/>Task queue, rate limits, metrics)]
  end

  %% MaaS layer flows
  LP -->|validate key| AP
  AP -->|read usage metrics| RD
  AP -->|model discovery| LS
  LP -->|proxy requests| LS

  %% Interactive RAG flow: rag-api validates keys via api-portal
  PI -.->|validate key| AP

  %% Routing decision flow
  Problem --> SC
  PC -.->|pattern cache score<br/>interactive RAG flow only| SC
  PI -.->|retrieval confidence| SC
  GL -.->|geometric energy| SC
  SC --> DE
  DE --> TS
  TS --> AK
  AK -->|k candidates + temperature| LS
  PC -.->|strategy hints<br/>interactive RAG flow only| LS
  PI -.->|relevant context| LS

  %% Evaluation flow
  LS -->|k candidates| GL
  GL -->|extract embedding| EM
  GL -->|sorted by energy| SB
  SB -->|result + feedback| FR
  SB -->|pass/fail embeddings| RT
  FR -.->|update Beta posteriors| TS
  RT -.->|retrained C x| GL
  SB -->|pattern write| PC

  %% Task worker flow
  TW -->|poll tasks| RD
  TW -->|generate code| LS
  TW -->|execute code| SB

  style AP fill:#5c1a3a,color:#fff
  style LP fill:#5c1a3a,color:#fff
  style GL fill:#2d5016,color:#fff
  style LS fill:#1a3a5c,color:#fff
  style DE fill:#5c3a1a,color:#fff
  style SC fill:#5c3a1a,color:#fff
  style TS fill:#5c3a1a,color:#fff
  style AK fill:#5c3a1a,color:#fff
  style PC fill:#1a3a5c,color:#fff
  style PI fill:#1a3a5c,color:#fff
  style SB fill:#2d5016,color:#fff
  style FR fill:#4a1a5c,color:#fff
  style RT fill:#4a1a5c,color:#fff
  style TW fill:#3a3a3a,color:#fff
  style EM fill:#1a3a5c,color:#fff
  style RD fill:#8b0000,color:#fff
```

</div>

## Service Summary

| Layer           | Service                | K8s Service Name       | Port                   | Technology              | Purpose                                                                 |
|-----------------|------------------------|------------------------|------------------------|-------------------------|-------------------------------------------------------------------------|
| **MaaS**        | api-portal             | api-portal             | 3000 (NodePort 30000)  | FastAPI                 | User registration/login (JWT), API key mgmt (sk-llm-*), /v1/models      |
|                 | llm-proxy              | llm-proxy              | 8000 (NodePort 30080)  | FastAPI                 | Reverse proxy to llama-server with API key validation + rate limiting    |
| **Core**        | rag-api                | rag-api                | 8001 (NodePort 31144)  | FastAPI                 | Orchestration: routing, RAG, cache, lens, key validation via api-portal  |
|                 | llama-server (Server A)| llama-service          | 8000 (NodePort 32735)  | llama.cpp + CUDA        | GPU inference (Qwen3-14B + 0.6B draft, spec decode ON, embeddings OFF)  |
|                 | llama-embed (Server B) | llama-embed-service    | 8001 (NodePort 32736)  | llama.cpp + CUDA        | Embedding sidecar (nomic-embed-text-v1.5, 768-dim, embeddings ON)       |
| **Execution**   | sandbox                | sandbox                | 8020 (NodePort 30820)  | K8s service             | Isolated code execution and testing (HTTP API)                           |
|                 | task-worker            | task-worker             | 8080 (ClusterIP)       | Python                  | Async task processor: polls Redis queues, runs ralph-loop, calls sandbox |
| **Storage**     | Redis                  | redis                  | 6379                   | Redis                   | Pattern cache, Thompson state, task queue, rate limits, usage metrics    |
| **Intelligence**| Confidence Router      | (in rag-api)           | --                     | Thompson Sampling       | 4-signal difficulty estimation, adaptive-k                               |
|                 | Geometric Lens         | (in rag-api)           | --                     | PyTorch (CPU)           | Energy-based candidate scoring, 7.9M params                             |
|                 | Pattern Cache          | (in rag-api)           | --                     | Redis-backed            | Ebbinghaus-decay STM/LTM pattern memory                                  |
|                 | PageIndex              | (in rag-api)           | --                     | tree-sitter + BM25      | AST-aware code retrieval with LLM tree search                            |
| **Dashboard**   | atlas-dashboard        | atlas-dashboard        | 3001 (NodePort 30001)  | Web UI                  | Monitoring dashboard (queue stats, daily metrics, weekly trend)          |
| **Training**    | atlas-nightly-training | (CronJob, suspended)   | --                     | CronJob, suspended       | LoRA fine-tuning (V1 artifact, suspended — V2 uses frozen model)         |

## Data Flows

### Routing Decision (interactive RAG flow)
```
Query
 -> Signal Collector (pattern_cache_score, retrieval_confidence, query_complexity, geometric_energy)
 -> Difficulty Estimator (weighted sum -> D(x) in [0,1])
 -> Thompson Sampling (Beta posteriors, cost-weighted efficiency)
 -> Route Selection (CACHE_HIT / FAST_PATH / STANDARD / HARD_PATH)
 -> Adaptive-k (k=0 / k=1 / k=5 / k=20)
```

### Best-of-K Generation
```
Task + k value
 -> llama-server Server A (k candidates, temperature varies by mode:
      k=1: 0.0, mcq/ifbench: 0.3, code k<=5: 0.6, code k>5: 0.8)
 -> Geometric Lens (extract 768-dim embedding via Server B /embedding,
      score each candidate through C(x), sort by energy)
 -> Sandbox (try in energy order, early exit on first PASS)
 -> Result + feedback
```
Note: In benchmark mode, best-of-k calls llama-server directly (_call_llm).
Pattern Cache strategy hints and pattern_cache_score are only used in the
interactive RAG flow, not the benchmark pipeline.

### Task Worker (production async flow)
```
User request via api-portal
 -> Task queued in Redis
 -> task-worker polls Redis, picks up task
 -> (optional) Retrieve RAG context from rag-api
 -> Generate code via llama-server (direct, bypasses llm-proxy auth)
 -> Execute + test via sandbox service (HTTP POST to sandbox:8020)
 -> Store result in Redis, publish completion
 -> Store result in Redis, publish completion
```

### MaaS Authentication
```
External user
 -> llm-proxy (Bearer sk-llm-* key in Authorization header)
 -> llm-proxy validates key with api-portal /api/validate-key
 -> Rate limit check via Redis sliding window
 -> Proxy request to llama-server

Internal services (rag-api, task-worker)
 -> Call llama-server directly (bypass llm-proxy, no auth needed)
```

### Continuous Learning
```
Sandbox results (pass/fail + code embeddings)
 -> Lens Retrain (BCE loss, epoch-based)
 -> Hot-reload C(x) weights into rag-api
 -> Thompson Sampling feedback (update Beta posteriors)
 -> Pattern Cache write (extract + store successful patterns)
```

## Color Legend

| Color | Meaning |
|-------|---------|
| Dark green | Evaluation (Lens + Sandbox) |
| Dark blue | Generation and retrieval (llama-server, PageIndex, Pattern Cache) |
| Dark brown | Routing (Signal Collector, Difficulty Estimator, Thompson, Adaptive-k) |
| Dark purple | Feedback and learning (Feedback Recorder, Lens Retrain) |
| Dark rose | MaaS layer (API Portal, LLM Proxy) |
| Dark grey | Async workers (Task Worker) |
| Dark red | Storage (Redis) |
