# ATLAS Architecture

This document describes the architecture of ATLAS (Autonomous Test-Loop Agent System).

## System Overview

ATLAS is a self-hosted AI coding assistant infrastructure that combines:
- LLM inference with GPU acceleration
- RAG (Retrieval-Augmented Generation) for codebase awareness
- Test-time compute scaling via the Ralph Loop algorithm
- Continuous learning through nightly LoRA fine-tuning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ATLAS System                                   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Core Inference Layer                                                 │  │
│  │                                                                      │  │
│  │  llama-server ◄──► RAG API ◄──► Qdrant ◄──► Embedding Service       │  │
│  │     (GPU)          (orchestration)  (vectors)    (CPU)               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Task Processing Layer (ATLAS)                                        │  │
│  │                                                                      │  │
│  │  Redis ──► Task Worker ──► Sandbox ──► Dashboard                     │
│  │ (queue)   (Ralph Loop)   (isolation)  (monitoring)                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Learning Layer                                                       │  │
│  │                                                                      │  │
│  │  Training Data ──► Nightly Trainer ──► LoRA Adapters                 │  │
│  │    (Redis)           (CronJob)          (hot-swap)                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Management Layer                                                     │  │
│  │                                                                      │  │
│  │  API Portal ◄──► LLM Proxy                                          │  │
│  │  (users/keys)    (auth/routing)                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Core Inference Services

#### llama-server
GPU-accelerated LLM inference using llama.cpp.

- **Port**: 8000
- **Resources**: GPU (all VRAM), 2GB RAM
- **Features**:
  - OpenAI-compatible API
  - Flash attention enabled
  - Speculative decoding (optional)
  - Custom chat templates

#### RAG API
Central orchestration service for retrieval-augmented generation.

- **Port**: 8001
- **Resources**: CPU, 1GB RAM
- **Responsibilities**:
  - Project sync and management
  - Code chunking and indexing
  - Query routing and context assembly
  - API key validation
  - Rate limiting

#### Qdrant
High-performance vector database for semantic search.

- **Port**: 6333 (HTTP), 6334 (gRPC)
- **Storage**: 100GB PVC
- **Features**:
  - Collections per project
  - HNSW indexing
  - Metadata filtering

#### Embedding Service
Text-to-vector conversion using sentence transformers.

- **Port**: 8080
- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Resources**: CPU only (2 cores, 2GB RAM)

### Task Processing Services

#### Redis
Message queue and metrics store.

- **Port**: 6379
- **Storage**: 5GB PVC
- **Data structures**:
  - Priority queues: `tasks:p0`, `tasks:p1`, `tasks:p2`
  - Metrics: `metrics:daily:{date}`, `metrics:recent_tasks`
  - Training data: `training:*` keys

#### Task Worker
Processes tasks using the Ralph Loop algorithm.

- **Resources**: CPU, 2GB RAM
- **Responsibilities**:
  - Queue polling with priority
  - Ralph Loop execution
  - Error accumulation and retry
  - Training data collection

#### Sandbox
Isolated Python execution environment.

- **Port**: 8020
- **Resources**: CPU, 512MB RAM
- **Features**:
  - Syntax checking
  - pytest execution
  - pylint scoring
  - 60-second timeout
  - 512MB memory limit

#### Dashboard
Real-time task monitoring web interface.

- **Port**: 3001
- **Features**:
  - Queue depth visualization
  - Success rate metrics
  - Recent task history
  - 7-day trend graphs

### Learning Services

#### Trainer (CronJob)
Nightly LoRA fine-tuning pipeline.

- **Schedule**: 2am daily
- **Timeout**: 12 hours
- **Resources**: CPU, 8-12GB RAM
- **Pipeline**:
  1. Export successful completions from Redis
  2. Train LoRA adapter (r=8, alpha=16)
  3. Validate adapter (66%+ pass rate)
  4. Update `latest` symlink for hot-swap

### Management Services

#### API Portal
Web-based multi-tenant user management.

- **Port**: 3000
- **Features**:
  - User registration/authentication
  - API key CRUD operations
  - Usage statistics
  - Model discovery
  - Admin panel

#### LLM Proxy
API gateway with authentication and metrics.

- **Port**: 8000 (external)
- **Features**:
  - API key validation
  - Rate limiting
  - Request routing
  - Usage metrics

## Ralph Loop Algorithm

The Ralph Loop is the core retry-until-success algorithm for code generation.

### Mathematical Basis

For a single attempt success probability `p` and `k` maximum attempts:

```
P(success within k attempts) = 1 - (1 - p)^k
```

With conservative estimates:
- Base success rate: p = 0.65
- Maximum attempts: k = 5
- **Cumulative success rate: 99.5%**

### Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ Ralph Loop Iteration                                                │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ Generate │───►│  Syntax  │───►│ Execute  │───►│  Tests   │     │
│  │   Code   │    │  Check   │    │    in    │    │ & Lint   │     │
│  │          │    │          │    │ Sandbox  │    │          │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│       │              │                │               │            │
│       │              ▼                ▼               ▼            │
│       │         ┌─────────┐     ┌──────────┐    ┌──────────┐      │
│       │         │ Syntax  │     │ Runtime  │    │  Test    │      │
│       │         │ Error   │     │  Error   │    │ Failure  │      │
│       │         └────┬────┘     └────┬─────┘    └────┬─────┘      │
│       │              │               │               │             │
│       │              └───────────────┴───────────────┘             │
│       │                              │                             │
│       │                              ▼                             │
│       │                    ┌──────────────────┐                    │
│       │                    │ Accumulate Error │                    │
│       │                    │ Increase Temp    │                    │
│       │                    └────────┬─────────┘                    │
│       │                             │                              │
│       │◄────────────────────────────┘                              │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                      │
│  │ SUCCESS  │                                                      │
│  └──────────┘                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Temperature Escalation

Each retry increases temperature to encourage alternative approaches:

| Attempt | Temperature | Strategy |
|---------|-------------|----------|
| 1 | 0.7 | Conservative, follow patterns |
| 2 | 0.8 | Slight variation |
| 3 | 0.9 | More creative alternatives |
| 4 | 1.0 | Maximum creativity |
| 5 | 1.0 | Final attempt with all errors |

### Error Feedback

Accumulated errors are fed back to the LLM:

```
Previous attempts failed with the following errors:

Attempt 1 (temp=0.7):
- SyntaxError: unexpected indent at line 15

Attempt 2 (temp=0.8):
- RuntimeError: division by zero in calculate_average()

Please generate a corrected solution that addresses these issues.
```

### Early Termination

The loop terminates early for unrecoverable errors:
- Missing dependencies that cannot be installed
- Invalid task specification
- Sandbox timeout exceeded (60s)

## RAG Pipeline

### Code Chunking Strategy

Code is split into semantic chunks for vector storage.

**Line-based chunking (default)**:
- Chunk size: 100 lines
- Overlap: 20 lines
- Preserves file boundaries

**Metadata per chunk**:
```json
{
  "chunk_id": "chunk_abc123",
  "project_id": "proj_xyz789",
  "file_path": "src/auth/jwt.py",
  "start_line": 45,
  "end_line": 144,
  "language": "python"
}
```

### Retrieval Strategy

1. **Query embedding**: Convert user query to vector
2. **Semantic search**: Find top-50 similar chunks
3. **Filtering**: Apply metadata constraints
4. **Ranking**: Score by relevance and recency
5. **Selection**: Return top-20 within token budget

### Context Assembly

Retrieved chunks are formatted for the LLM:

```
## Retrieved Context

### File: src/auth/jwt.py (lines 45-144)
```python
def validate_token(token: str) -> User:
    ...
```

### File: src/models/user.py (lines 1-50)
```python
class User:
    ...
```
```

## Continuous Learning Pipeline

### Data Collection

Successful task completions are stored in Redis:
- Task prompt
- Generated code
- Test results
- User rating (1-5)

### Training Selection

Only high-quality completions (rating >= 4) are used for training.

### LoRA Fine-Tuning

Configuration:
- Rank (r): 8
- Alpha: 16
- Target modules: q_proj, k_proj, v_proj, o_proj
- Learning rate: 2e-5
- Epochs: 3

### Hot-Swap Deployment

New adapters are deployed via symlink update:
```
/models/lora/
├── adapter_20260131_020000/
├── adapter_20260130_020000/
└── latest -> adapter_20260131_020000/
```

llama-server loads from `latest` on each inference request.

## Data Flow Diagrams

### Inference Request Flow

```
Client
  │
  ▼
LLM Proxy ──► API Portal (validate key)
  │
  ▼
RAG API ──► Embedding Service (embed query)
  │      └─► Qdrant (search vectors)
  │
  ▼
llama-server (generate response)
  │
  ▼
Client
```

### Task Processing Flow

```
Client
  │
  ▼
RAG API ──► Redis (enqueue task)

Task Worker ◄── Redis (dequeue)
  │
  ▼
Ralph Loop ──► llama-server (generate)
  │         └─► Sandbox (execute)
  │
  ├─► Success: Redis (store for training)
  └─► Failure: Retry or abort

Dashboard ◄── Redis (read metrics)
```

## Security Model

### Authentication

- API keys with configurable rate limits
- JWT for web UI sessions
- Key validation on every request

### Isolation

- Sandbox runs untrusted code in isolated container
- Resource limits prevent DoS
- No network access from sandbox

### Data Protection

- Per-project data isolation
- Configurable TTL for project data
- No code content logging
