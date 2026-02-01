# ATLAS Architecture Diagram

<div align="center">

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#2196F3', 'primaryTextColor': '#212121', 'primaryBorderColor': '#1565C0', 'lineColor': '#455A64', 'secondaryColor': '#E3F2FD', 'tertiaryColor': '#ECEFF1', 'edgeLabelBackground': '#ECEFF1'}}}%%
flowchart TB
    subgraph external[" "]
        client(["Client<br/>OpenCode / API"])
    end
    subgraph gateway["Gateway"]
        proxy["LLM Proxy :8000<br/>Auth • Rate Limit"]
        portal["API Portal :3000<br/>Users • Keys"]
    end
    subgraph core["Core Services"]
        rag["RAG API :8001<br/>Orchestration"]
        embed["Embeddings :8080<br/>MiniLM-L6-v2"]
    end

    %% Central inference engine - outside subgraphs for central positioning
    llama["llama-server :8000<br/>Qwen3-14B • GPU"]

    subgraph data["Storage"]
        qdrant[("Qdrant<br/>100GB Vectors")]
        redis[("Redis<br/>Queues • Metrics")]
    end
    subgraph atlas["Task Processing"]
        worker["Task Worker<br/>Ralph Loop<br/>99.5% Success"]
        sandbox["Sandbox :8020<br/>pytest • pylint"]
        dash["Dashboard :3001<br/>Monitoring"]
    end
    subgraph learn["Learning"]
        trainer["Nightly Trainer<br/>LoRA Fine-tune"]
        lora[("Adapters<br/>Hot-swap")]
    end
    %% Gateway flow
    client -->|"request"| proxy
    proxy -.->|"validate key"| portal
    proxy -->|"chat/completions"| rag

    %% RAG API calls llama-server for inference
    rag -->|"inference"| llama
    rag -->|"embed query"| embed
    embed -->|"search vectors"| qdrant

    %% Task submission to Redis
    rag -->|"submit task"| redis
    redis -->|"poll result"| rag

    %% Ralph Loop: Task Worker flow
    redis -->|"pull task"| worker
    worker -->|"generate code"| llama
    worker -->|"test code"| sandbox
    worker -->|"result + training"| redis

    %% Monitoring
    redis -->|"metrics"| dash

    %% Learning pipeline
    redis -.->|"training data"| trainer
    trainer -->|"fine-tune"| lora
    lora -.->|"load LoRA"| llama
    classDef client fill:#37474F,stroke:#263238,color:#fff
    classDef gateway fill:#607D8B,stroke:#455A64,color:#fff
    classDef core fill:#2196F3,stroke:#1565C0,color:#fff
    classDef gpu fill:#4CAF50,stroke:#2E7D32,color:#fff
    classDef storage fill:#00BCD4,stroke:#00838F,color:#fff
    classDef process fill:#FF9800,stroke:#E65100,color:#fff
    classDef learn fill:#9C27B0,stroke:#6A1B9A,color:#fff
    class client client
    class proxy,portal gateway
    class rag core
    class llama,embed gpu
    class qdrant,redis storage
    class worker,sandbox,dash process
    class trainer,lora learn
```

</div>

## Service Summary

| Layer | Service | Port | Technology | Purpose |
|-------|---------|------|------------|---------|
| **Gateway** | LLM Proxy | 8000 | FastAPI | Auth, rate limiting, routing |
| | API Portal | 3000 | FastAPI + SQLite | User/key management, usage tracking |
| **Core** | RAG API | 8001 | FastAPI | Project sync, chunking, orchestration |
| | llama-server | 8000 | llama.cpp + CUDA | GPU inference (Qwen3-14B) |
| | Embeddings | 8080 | sentence-transformers | Text → 384-dim vectors |
| **Storage** | Qdrant | 6333/6334 | Qdrant | Vector DB, HNSW indexing |
| | Redis | 6379 | Redis | Task queues, metrics, cache |
| **Processing** | Task Worker | — | Python | Ralph Loop execution |
| | Sandbox | 8020 | FastAPI | Isolated pytest/pylint |
| | Dashboard | 3001 | FastAPI + Jinja2 | Real-time monitoring |
| **Learning** | Trainer | — | PyTorch + PEFT | Nightly LoRA fine-tuning |

## Data Flows

### Chat Completion
```
Client → LLM Proxy → RAG API → Embeddings → Qdrant
                         ↓
                    llama-server → Response
```

### Task Processing (Ralph Loop)
```
Client → RAG API → Redis Queue
                       ↓
               Task Worker ⟷ Sandbox
                   ↓
              llama-server
                   ↓
              Success → Training Data
```

### Continuous Learning
```
Redis (completions) → Trainer → LoRA Adapter → llama-server
                                    ↑
                            Hot-swap via symlink
```

## Color Legend

| Color | Meaning |
|-------|---------|
| 🟢 Green | GPU-accelerated services |
| 🔵 Blue | Persistent storage |
| 🟠 Orange | Task processing |
| 🟣 Purple | Learning pipeline |
