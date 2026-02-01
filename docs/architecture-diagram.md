# ATLAS Architecture Diagram

This document contains the full architecture diagram for ATLAS in Mermaid format.

```mermaid
flowchart TB
    subgraph client["Client Layer"]
        opencode([OpenCode / API Client])
    end

    subgraph gateway["API Gateway Layer"]
        direction LR
        proxy["LLM Proxy<br/>:8000<br/>━━━━━━━━━━<br/>Auth & Rate Limiting"]
        portal["API Portal<br/>:3000<br/>━━━━━━━━━━<br/>Users, Keys, Usage"]
    end

    subgraph inference["Inference Layer — GPU Accelerated"]
        direction LR
        llama["llama-server<br/>:8000<br/>━━━━━━━━━━<br/>Qwen3-14B (Q4_K_M)<br/>16384 ctx, 99 layers<br/>Flash Attention"]
        embed["Embedding Service<br/>:8080<br/>━━━━━━━━━━<br/>all-MiniLM-L6-v2<br/>384 dimensions"]
    end

    subgraph orchestration["RAG Orchestration Layer"]
        rag["RAG API<br/>:8001<br/>━━━━━━━━━━<br/>Project Sync<br/>Chunking (100 lines)<br/>Context Assembly"]
    end

    subgraph storage["Persistent Storage"]
        direction LR
        qdrant[("Qdrant<br/>:6333/:6334<br/>━━━━━━━━━━<br/>Vector DB<br/>100GB PVC<br/>HNSW Index")]
        redis[("Redis<br/>:6379<br/>━━━━━━━━━━<br/>Task Queues<br/>Metrics Cache<br/>Training Data")]
    end

    subgraph atlas["ATLAS Task Processing Layer"]
        direction TB

        subgraph queues["Priority Queues"]
            p0["P0: Interactive"]
            p1["P1: Fire & Forget"]
            p2["P2: Batch"]
        end

        worker["Task Worker<br/>━━━━━━━━━━<br/>Ralph Loop Engine<br/>5 retries = 99.5%<br/>Temp Escalation"]

        sandbox["Sandbox<br/>:8020<br/>━━━━━━━━━━<br/>Isolated Execution<br/>pytest + pylint<br/>60s timeout"]

        dash["Dashboard<br/>:3001<br/>━━━━━━━━━━<br/>Real-time Monitor<br/>Queue Depth<br/>Success Rates"]
    end

    subgraph learning["Continuous Learning Layer"]
        trainer["Nightly Trainer<br/>CronJob @ 2am<br/>━━━━━━━━━━<br/>Export → Train → Validate<br/>LoRA r=8, α=16<br/>66% validation gate"]
        lora[("LoRA Adapters<br/>/models/lora/latest<br/>━━━━━━━━━━<br/>Hot-swap via symlink")]
    end

    %% Client connections
    opencode --> proxy

    %% Gateway layer
    proxy <-. "validate key" .-> portal
    proxy --> rag

    %% Orchestration to inference
    rag --> llama
    rag --> embed
    embed --> qdrant
    rag <--> redis

    %% Task processing
    redis --> queues
    queues --> worker
    worker --> sandbox
    worker <-. "generate code" .-> llama
    worker <-. "fetch context" .-> rag
    redis --> dash

    %% Learning loop
    redis -. "successful completions" .-> trainer
    trainer --> lora
    lora -. "hot-swap adapter" .-> llama

    %% Styling
    classDef gpu fill:#76B900,stroke:#333,color:#fff
    classDef storage fill:#4A90D9,stroke:#333,color:#fff
    classDef worker fill:#E67E22,stroke:#333,color:#fff
    classDef learning fill:#9B59B6,stroke:#333,color:#fff

    class llama,embed gpu
    class qdrant,redis storage
    class worker,sandbox worker
    class trainer,lora learning
```

## Component Summary

| Layer | Components | Purpose |
|-------|------------|---------|
| **Client** | OpenCode, API Clients | External interface |
| **Gateway** | LLM Proxy, API Portal | Auth, rate limiting, user management |
| **Inference** | llama-server, Embedding Service | GPU inference, vectorization |
| **Orchestration** | RAG API | Context retrieval, request routing |
| **Storage** | Qdrant, Redis | Vectors, queues, metrics |
| **Processing** | Task Worker, Sandbox, Dashboard | Ralph Loop, isolated execution, monitoring |
| **Learning** | Trainer, LoRA Adapters | Continuous improvement |

## Port Reference

| Service | Port | Protocol |
|---------|------|----------|
| LLM Proxy | 8000 | HTTP |
| API Portal | 3000 | HTTP |
| llama-server | 8000 | HTTP |
| Embedding Service | 8080 | HTTP |
| RAG API | 8001 | HTTP |
| Qdrant | 6333/6334 | HTTP/gRPC |
| Redis | 6379 | Redis |
| Sandbox | 8020 | HTTP |
| Dashboard | 3001 | HTTP |
