# ATLAS V2 Architecture Diagram

<div align="center">

```mermaid
flowchart TB
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
    LS[llama-server<br/>Qwen3-14B-Q4_K_M<br/>+ Qwen3-0.6B Draft<br/>2 parallel slots]
    PC[Pattern Cache<br/>Redis + Ebbinghaus Decay<br/>STM / LTM tiers]
  end

  subgraph Evaluation["Candidate Selection"]
    GL[Geometric Lens<br/>Cost Field C x 2.7M params<br/>Metric Tensor G x 5.2M params<br/>Val AUC: 0.968]
    SB[Sandbox<br/>Code Execution + Testing<br/>Energy-sorted early exit]
  end

  subgraph Knowledge["Context Retrieval"]
    PI[PageIndex RAG<br/>AST Tree Index + BM25<br/>LLM-guided Tree Search]
  end

  subgraph Feedback["Continuous Learning"]
    FR[Feedback Recorder<br/>Thompson state updates]
    RT[Lens Retrain<br/>BCE on pass/fail embeddings<br/>Hot-reload weights]
  end

  Problem --> SC
  PC -.->|pattern cache score| SC
  PI -.->|retrieval confidence| SC
  GL -.->|geometric energy| SC
  SC --> DE
  DE --> TS
  TS --> AK
  AK -->|k candidates + temperature| LS
  PC -.->|strategy hints| LS
  PI -.->|relevant context| LS
  LS -->|k candidates| GL
  GL -->|sorted by energy| SB
  SB -->|result + feedback| FR
  SB -->|pass/fail embeddings| RT
  FR -.->|update Beta posteriors| TS
  RT -.->|retrained C x| GL
  SB -->|pattern write| PC

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
```

</div>

## Service Summary

| Layer | Service | Port | Technology | Purpose |
|-------|---------|------|------------|---------|
| **Core** | rag-api | 8001 (NodePort 31144) | FastAPI | Orchestration: routing, RAG, cache, lens |
| | llama-server | 8000 (NodePort 32735) | llama.cpp + CUDA | GPU inference (Qwen3-14B-Q4_K_M + 0.6B draft) |
| **Storage** | Redis | 6379 | Redis | Pattern cache, Thompson state, task queue |
| **Intelligence** | Confidence Router | (in rag-api) | Thompson Sampling | 4-signal difficulty estimation, adaptive-k |
| | Geometric Lens | (in rag-api) | PyTorch (CPU) | Energy-based candidate scoring, 7.9M params |
| | Pattern Cache | (in rag-api) | Redis-backed | Ebbinghaus-decay STM/LTM pattern memory |
| | PageIndex | (in rag-api) | tree-sitter + BM25 | AST-aware code retrieval with LLM tree search |
| **Execution** | Sandbox | (in benchmark) | subprocess | Isolated code execution and testing |

## Data Flows

### Routing Decision
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
 -> llama-server (k candidates at temperature 0.6-0.8, pipelined via ThreadPool)
 -> Geometric Lens (score each candidate, sort by energy)
 -> Sandbox (try in energy order, early exit on first PASS)
 -> Result + feedback
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
