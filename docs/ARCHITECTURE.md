# ATLAS Architecture

This document provides a comprehensive technical deep dive into ATLAS (Adaptive Test-time Learning and Autonomous Specialization).

## Table of Contents

- [System Overview](#system-overview)
- [Design Philosophy](#design-philosophy)
- [Component Deep Dive](#component-deep-dive)
- [Data Flows](#data-flows)
- [Ralph Loop Algorithm](#ralph-loop-algorithm)
- [RAG Pipeline](#rag-pipeline)
- [Continuous Learning Pipeline](#continuous-learning-pipeline)
- [API Reference](#api-reference)
- [Security Model](#security-model)

---

## System Overview

ATLAS is a self-hosted Kubernetes-based AI coding agent infrastructure designed to run entirely on consumer hardware. It demonstrates that sophisticated AI infrastructure—including RAG, test-time compute scaling, and continuous learning—can operate effectively on a single 16GB consumer GPU.

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Runtime** | Python 3.11, FastAPI, uvicorn |
| **Orchestration** | K3s (lightweight Kubernetes) |
| **Inference** | llama.cpp (CUDA 12.8), GGUF models |
| **Vector Storage** | Qdrant (HNSW indexing) |
| **Queue/Cache** | Redis (AOF persistence) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Training** | PyTorch, PEFT (LoRA), Hugging Face transformers |

### Hardware Requirements

| Component | Minimum | Tested Configuration |
|-----------|---------|---------------------|
| GPU | 8GB VRAM | RTX 5060 Ti 16GB |
| RAM | 12GB | 12GB DDR4 |
| Storage | 50GB SSD | 150GB SSD |
| CPU | 4 cores | Ryzen 5 2600 |

---

## Design Philosophy

### Consumer Hardware First

ATLAS prioritizes running on accessible hardware. Every design decision considers memory constraints, single-GPU limitations, and cost-effectiveness.

### Test-Time Compute Scaling

Rather than relying solely on model capability, ATLAS uses the Ralph Loop algorithm to achieve high success rates through intelligent retry with error feedback.

### Continuous Improvement

Successful task completions feed back into nightly training, allowing the system to improve over time on domain-specific patterns.

---

## Component Deep Dive

### 1. LLM Proxy

**Location**: `/home/nobase/k8s/llm-proxy/`
**Port**: 8000 (external-facing)
**Purpose**: API gateway with authentication and rate limiting

The LLM Proxy is the external entry point for all API requests. It validates API keys against the API Portal, enforces rate limits, and proxies requests to downstream services.

**Key Responsibilities**:
- Extract and validate API keys from `Authorization: Bearer` header
- Cache validation results (60-second TTL) to reduce database load
- Enforce per-key rate limits using Redis sliding window
- Log request metrics to Redis for dashboard visualization
- Proxy OpenAI-compatible endpoints to RAG API or llama-server

**Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion with optional streaming |
| `/v1/models` | GET | List available models |
| `/*` | ALL | Catch-all proxy to backend |

**Rate Limiting Implementation**:
```
Key: ratelimit:{key_hash}:count
Window: 60 seconds (sliding)
Algorithm: INCR + EXPIRE
```

---

### 2. API Portal (MaaS)

**Location**: `/home/nobase/k8s/api-portal/`
**Port**: 3000
**Purpose**: Multi-tenant user management and API key administration

The API Portal provides a web-based interface for user registration, API key management, and usage tracking. It follows a "Model as a Service" (MaaS) pattern.

**Key Features**:
- JWT-based authentication (first user auto-promoted to admin)
- API key CRUD with configurable rate limits and expiration
- Usage statistics aggregated from Redis metrics
- Model discovery from llama-server `/v1/models` endpoint
- Admin panel for model management

**Database Schema** (SQLite via SQLAlchemy):

```
User
├── id: Integer (PK)
├── email: String (unique)
├── username: String (unique)
├── hashed_password: String
├── is_admin: Boolean
├── is_active: Boolean
└── created_at: DateTime

APIKey
├── id: Integer (PK)
├── key_hash: String (SHA256, unique)
├── key_prefix: String (first 16 chars for display)
├── user_id: Integer (FK → User)
├── name: String
├── rate_limit: Integer (requests per minute)
├── is_active: Boolean
├── expires_at: DateTime (nullable)
├── created_at: DateTime
└── last_used_at: DateTime

UsageLog
├── id: Integer (PK)
├── api_key_id: Integer (FK → APIKey)
├── tokens_input: Integer
├── tokens_output: Integer
└── created_at: DateTime

LLMModel
├── id: Integer (PK)
├── model_id: String (unique)
├── name: String
├── context_length: Integer
├── max_output: Integer
├── is_active: Boolean
└── is_auto_discovered: Boolean
```

**Quick Start Connection String**:
The dashboard displays a connection string format for OpenCode: `SERVER_URL|API_KEY`

---

### 3. llama-server (LLM Inference)

**Location**: `/home/nobase/k8s/llama-server/`
**Port**: 8000 (internal)
**Purpose**: GPU-accelerated LLM inference

llama-server provides OpenAI-compatible API endpoints for LLM inference using llama.cpp with CUDA acceleration.

**Configuration**:
| Setting | Value | Description |
|---------|-------|-------------|
| Model | Qwen3-14B-Q4_K_M.gguf | Quantized 14B parameter model |
| Context | 16384 tokens | Maximum context window |
| GPU Layers | 99 | All layers offloaded to GPU |
| Parallel Slots | 1 | Single concurrent inference |
| Flash Attention | Enabled | Memory-efficient attention |

**Speculative Decoding** (optional):
- Draft model: Qwen3-0.6B-Q8_0.gguf
- Max draft tokens: 16
- Min draft tokens: 5
- P-min: 0.9

**Container Build**:
Multi-stage Dockerfile:
1. **Builder stage**: Compiles llama.cpp from source with CUDA support
2. **Runtime stage**: Minimal image with compiled binary and Python bindings

**Volume Mounts**:
| Path | Source | Purpose |
|------|--------|---------|
| `/models` | `/home/nobase/models` | Model files |
| `/models/lora` | `/home/nobase/models/lora` | LoRA adapters |

---

### 4. RAG API

**Location**: `/home/nobase/k8s/rag-api/`
**Port**: 8001
**Purpose**: Central orchestration for retrieval-augmented generation

The RAG API is the heart of the system, coordinating project indexing, semantic search, context assembly, and task queue management.

**Core Modules**:

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI application and route definitions |
| `rag.py` | RAG pipeline: retrieval, context assembly, augmentation |
| `chunker.py` | Code chunking with language awareness |
| `vector_store.py` | Qdrant integration for semantic search |
| `storage.py` | Project file management and hashing |
| `config.py` | Configuration management |
| `provenance.py` | Git history analysis for quality scoring |

**Project Limits**:
| Limit | Value |
|-------|-------|
| Max files | 10,000 |
| Max LOC | 1,000,000 |
| Max size | 100MB |
| Default TTL | 7 days |

**Supported Languages**:
Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, C#, Swift, Kotlin, Scala, R, SQL, Bash, YAML, JSON, XML, HTML, CSS, Markdown

---

### 5. Embedding Service

**Location**: `/home/nobase/k8s/embedding-service/`
**Port**: 8080
**Purpose**: Text-to-vector conversion for semantic search

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Output dimensions: 384
- Model size: ~22MB
- Optimized for: Semantic similarity

**Resource Usage**: CPU-only (no GPU required)

**Endpoint**:
```
POST /v1/embeddings
{
  "input": "text to embed" | ["text1", "text2"],
  "model": "all-MiniLM-L6-v2"
}
```

---

### 6. Qdrant (Vector Database)

**Ports**: 6333 (HTTP), 6334 (gRPC)
**Storage**: 100GB PVC
**Purpose**: Semantic search over code embeddings

**Configuration**:
- Indexing: HNSW (Hierarchical Navigable Small World)
- Collections: One per project (isolated data)
- Distance metric: Cosine similarity

**Collection Schema**:
```json
{
  "vectors": {
    "size": 384,
    "distance": "Cosine"
  },
  "payload": {
    "project_id": "string",
    "file_path": "string",
    "start_line": "integer",
    "end_line": "integer",
    "language": "string",
    "chunk_id": "string"
  }
}
```

---

### 7. Redis

**Port**: 6379
**Storage**: 5GB PVC (AOF persistence)
**Purpose**: Task queues, metrics, and caching

**Data Structures**:

| Key Pattern | Type | Purpose |
|-------------|------|---------|
| `tasks:p0` | List | Interactive priority queue |
| `tasks:p1` | List | Default priority queue |
| `tasks:p2` | List | Batch priority queue |
| `task:{id}` | Hash | Task data and status |
| `metrics:daily:{date}` | Hash | Daily aggregates |
| `metrics:recent_tasks` | List | Last 100 completions |
| `training:examples` | List | Successful completions for training |
| `ratelimit:{hash}:count` | String | Rate limit counter (60s TTL) |

**Memory Policy**: LRU eviction at 4GB

---

### 8. Task Worker (Ralph Loop)

**Location**: `/home/nobase/k8s/atlas/task-worker/`
**Resources**: CPU-only, 2GB RAM
**Purpose**: Process code generation tasks using Ralph Loop algorithm

**Modules**:
| Module | Purpose |
|--------|---------|
| `worker.py` | Main loop, task orchestration |
| `ralph_loop.py` | Retry algorithm implementation |
| `executor.py` | Sandbox client |
| `task_queue.py` | Redis queue operations |
| `metrics.py` | Metrics collection |

**Processing Flow**:
1. Poll Redis queues (P0 → P1 → P2 priority)
2. Fetch RAG context if `project_id` specified
3. Execute Ralph Loop until success or exhaustion
4. Store successful completions for training
5. Record metrics and publish completion

---

### 9. Sandbox

**Location**: `/home/nobase/k8s/atlas/sandbox/`
**Port**: 8020
**Purpose**: Isolated code execution with validation

**Capabilities**:
| Feature | Details |
|---------|---------|
| Syntax check | Python `compile()` before execution |
| Test runner | pytest with pass/fail counting |
| Linting | pylint score (0-10 scale) |
| Timeout | 60 seconds maximum |
| Memory | 512MB limit |

**Execution Result Schema**:
```json
{
  "success": true,
  "compile_success": true,
  "tests_run": 5,
  "tests_passed": 5,
  "lint_score": 8.5,
  "stdout": "...",
  "stderr": "...",
  "error_type": null,
  "error_message": null
}
```

---

### 10. Dashboard

**Location**: `/home/nobase/k8s/atlas/dashboard/`
**Port**: 3001
**Purpose**: Real-time task monitoring

**Displays**:
- Queue depth by priority (P0/P1/P2)
- Daily metrics (tasks, success rate, tokens)
- Recent task history (last 20)
- 7-day trend visualization

**Data Source**: Redis polling via `/api/stats` (5-second refresh)

---

### 11. Trainer (Nightly CronJob)

**Location**: `/home/nobase/k8s/atlas/trainer/`
**Schedule**: 2:00 AM UTC daily
**Timeout**: 12 hours
**Purpose**: LoRA fine-tuning from successful completions

**Pipeline Scripts**:
| Script | Purpose |
|--------|---------|
| `nightly_train.sh` | Orchestrates 4-step pipeline |
| `export_training_data.py` | Extract from Redis, filter by quality |
| `train_lora.py` | LoRA fine-tuning with PEFT |
| `validate_adapter.py` | Test prompts, require 66% pass |

**LoRA Configuration**:
| Setting | Value |
|---------|-------|
| Rank (r) | 8 |
| Alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Dropout | 0.05 |
| Learning rate | 2e-4 |
| Epochs | 1 |
| Batch size | 1 (4x gradient accumulation) |

---

## Data Flows

### 1. Chat Completion with RAG

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Chat Completion Request Flow                                                  │
└──────────────────────────────────────────────────────────────────────────────┘

Client
  │
  │ POST /v1/chat/completions
  │ Authorization: Bearer sk_xxx
  │ Body: {model, messages, project_id?, stream?}
  │
  ▼
┌─────────────────────────────────────────────┐
│ LLM Proxy (:8000)                           │
│                                             │
│ 1. Extract API key from header              │
│ 2. Check validation cache (60s TTL)         │
│    └─ Cache miss: POST /api/validate-key    │
│       to API Portal                         │
│ 3. Check rate limit (Redis counter)         │
│ 4. Forward request to RAG API               │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│ RAG API (:8001)                             │
│                                             │
│ If project_id provided:                     │
│ 1. Embed user query                         │
│    └─ POST to Embedding Service (:8080)     │
│ 2. Search Qdrant for similar chunks         │
│    └─ Top-k=20, filter by project_id        │
│ 3. Assemble context from chunks             │
│    └─ Format with file paths, line numbers  │
│ 4. Inject context into system prompt        │
│                                             │
│ Forward to llama-server:                    │
│ POST /v1/chat/completions                   │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│ llama-server (:8000)                        │
│                                             │
│ 1. Load model (Qwen3-14B-Q4_K_M.gguf)       │
│ 2. Load LoRA adapter (/models/lora/latest)  │
│ 3. Generate response (flash attention)     │
│ 4. Return OpenAI-compatible response        │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│ RAG API                                     │
│                                             │
│ 1. Log metrics to Redis                     │
│    └─ metrics:daily:{date}                  │
│    └─ metrics:recent_tasks                  │
│ 2. Return response to client                │
└─────────────────────────────────────────────┘
  │
  ▼
Client receives response

Latency: 500ms - 5s (depending on context size and token count)
```

### 2. Code Generation Task (Ralph Loop)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Task Submission and Processing Flow                                          │
└──────────────────────────────────────────────────────────────────────────────┘

Client
  │
  │ POST /v1/tasks/submit
  │ Body: {prompt, priority, project_id?, max_attempts?, test_code?}
  │
  ▼
┌─────────────────────────────────────────────┐
│ RAG API (:8001)                             │
│                                             │
│ 1. Validate API key                         │
│ 2. Generate task ID (UUID)                  │
│ 3. Store task in Redis: task:{id}           │
│    └─ status: PENDING                       │
│ 4. Enqueue to priority queue                │
│    └─ RPUSH tasks:p{0|1|2}                  │
│ 5. Return {task_id, status: pending}        │
└─────────────────────────────────────────────┘

                    ═══════════════════════════════════════

Task Worker (continuous polling loop)
  │
  │ Poll queues: P0 → P1 → P2
  │ LPOP tasks:p0 || LPOP tasks:p1 || LPOP tasks:p2
  │
  ▼
┌─────────────────────────────────────────────┐
│ Task Processing                             │
│                                             │
│ 1. Update status to RUNNING                 │
│ 2. If project_id: fetch RAG context         │
│    └─ GET /v1/context/retrieve              │
│ 3. Initialize Ralph Loop                    │
│    └─ max_attempts: 5                       │
│    └─ base_temperature: 0.3                 │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Ralph Loop Execution                                                         │
│                                                                              │
│ for attempt in 1..max_attempts:                                              │
│   │                                                                          │
│   │ 1. Check timeout (300s total)                                            │
│   │                                                                          │
│   │ 2. Calculate temperature                                                 │
│   │    └─ temp = 0.3 + (attempt-1) * 0.1                                     │
│   │                                                                          │
│   │ 3. Build enhanced prompt                                                 │
│   │    └─ Include RAG context                                                │
│   │    └─ Include accumulated errors                                         │
│   │                                                                          │
│   │ 4. Generate code                                                         │
│   │    └─ POST llama-server /v1/chat/completions                             │
│   │    └─ Extract code from response                                         │
│   │                                                                          │
│   │ 5. Syntax check                                                          │
│   │    └─ compile(code, '<string>', 'exec')                                  │
│   │    └─ If fail: accumulate error, continue                                │
│   │                                                                          │
│   │ 6. Execute in Sandbox                                                    │
│   │    └─ POST Sandbox:8020/execute                                          │
│   │    └─ Run pytest, pylint                                                 │
│   │                                                                          │
│   │ 7. Check success criteria                                                │
│   │    └─ compile_success == true                                            │
│   │    └─ tests_passed == tests_run (if required)                            │
│   │    └─ lint_score >= 6.0 (if required)                                    │
│   │                                                                          │
│   │ 8. If success: BREAK                                                     │
│   │    Else: accumulate error, increment temperature                         │
│   │                                                                          │
│   │ 9. Check for unrecoverable errors                                        │
│   │    └─ Missing dependencies, resource limits                              │
│   │    └─ If found: early termination                                        │
│   │                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│ Post-Processing                             │
│                                             │
│ 1. Update task in Redis                     │
│    └─ status: COMPLETED or FAILED           │
│    └─ result: {success, code, attempts...}  │
│                                             │
│ 2. If success: store training example       │
│    └─ RPUSH training:examples               │
│                                             │
│ 3. Log metrics                              │
│    └─ metrics:daily:{date}                  │
│    └─ metrics:recent_tasks                  │
│                                             │
│ 4. Publish completion                       │
│    └─ PUBLISH task:complete:{id}            │
└─────────────────────────────────────────────┘

                    ═══════════════════════════════════════

Client
  │
  │ Poll GET /v1/tasks/{id}/status
  │ (or subscribe to Redis pub/sub)
  │
  ▼
Returns: {id, status, attempts, result, completed_at}

Total Time: 2-30 seconds (depending on complexity and attempt count)
```

### 3. Nightly Training Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Nightly Training Pipeline (CronJob @ 2:00 AM UTC)                            │
└──────────────────────────────────────────────────────────────────────────────┘

K8s CronJob triggers Trainer pod
  │
  ▼
┌─────────────────────────────────────────────┐
│ Step 1: Export Training Data                │
│ (export_training_data.py)                   │
│                                             │
│ 1. Query Redis                              │
│    └─ LRANGE training:examples 0 -1         │
│                                             │
│ 2. Filter by quality                        │
│    └─ quality_score >= 0.6                  │
│    └─ rating >= 4                           │
│                                             │
│ 3. Format as JSONL                          │
│    {instruction, input, output}             │
│                                             │
│ 4. Save to /data/training/{DATE}.jsonl      │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│ Step 2: Minimum Examples Check              │
│                                             │
│ Require >= 10 examples                      │
│ If insufficient: exit (skip training)       │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│ Step 3: LoRA Fine-Tuning                    │
│ (train_lora.py)                             │
│                                             │
│ 1. Load base model                          │
│    └─ Qwen2.5-1.5B (CPU-trainable)          │
│                                             │
│ 2. Configure LoRA                           │
│    └─ r=8, alpha=16                         │
│    └─ targets: q_proj, k_proj, v_proj, o_proj│
│    └─ dropout: 0.05                         │
│                                             │
│ 3. Tokenize dataset                         │
│                                             │
│ 4. Train                                    │
│    └─ learning_rate: 2e-4                   │
│    └─ epochs: 1                             │
│    └─ batch_size: 1                         │
│    └─ gradient_accumulation: 4              │
│                                             │
│ 5. Save adapter                             │
│    └─ /models/lora/{DATE}/                  │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│ Step 4: Validation                          │
│ (validate_adapter.py)                       │
│                                             │
│ 1. Load adapter                             │
│ 2. Run test prompts                         │
│ 3. Check pass rate                          │
│    └─ Require >= 66%                        │
│                                             │
│ If validation fails: exit (keep old adapter)│
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│ Step 5: Deployment                          │
│                                             │
│ 1. Update symlink                           │
│    ln -sfn /models/lora/{DATE} \            │
│            /models/lora/latest              │
│                                             │
│ 2. Archive training data                    │
│    gzip /data/training/{DATE}.jsonl         │
│                                             │
│ 3. Clean old adapters (keep last 7)         │
└─────────────────────────────────────────────┘
  │
  ▼
llama-server loads new adapter on next request
(Hot-swap via symlink - no restart needed)

Duration: 2-8 hours on CPU (depends on example count)
```

---

## Ralph Loop Algorithm

The Ralph Loop is the core retry-until-success algorithm that enables ATLAS to achieve 99.5% code generation success rates.

### Mathematical Foundation

For a single attempt success probability `p` and maximum attempts `k`:

```
P(success within k attempts) = 1 - (1 - p)^k
```

**Empirical estimates**:
- Base success rate: p ≈ 0.65
- Maximum attempts: k = 5

**Cumulative success rates**:
| Attempts | Success Rate |
|----------|--------------|
| 1 | 65.0% |
| 2 | 87.8% |
| 3 | 95.7% |
| 4 | 98.5% |
| 5 | **99.5%** |

### Pseudocode

```python
def ralph_loop(task, max_attempts=5, base_temp=0.3, temp_increment=0.1):
    accumulated_errors = []

    for attempt in range(1, max_attempts + 1):
        # Check timeout
        if elapsed_time() > TIMEOUT:
            return failure("timeout")

        # Calculate temperature (increases with each retry)
        temperature = min(base_temp + (attempt - 1) * temp_increment, 1.0)

        # Build enhanced prompt with error context
        prompt = build_prompt(
            task=task,
            context=fetch_rag_context(task.project_id),
            errors=accumulated_errors
        )

        # Generate code
        code = generate_code(prompt, temperature=temperature)

        # Syntax check (fast fail)
        syntax_result = check_syntax(code)
        if not syntax_result.success:
            accumulated_errors.append(format_error(attempt, syntax_result))
            continue

        # Execute in sandbox
        execution_result = execute_in_sandbox(
            code=code,
            test_code=task.test_code,
            timeout=60
        )

        # Check success criteria
        if meets_criteria(execution_result, task.requirements):
            return success(code, attempt, execution_result)

        # Check for unrecoverable errors
        if is_unrecoverable(execution_result.error):
            return failure("unrecoverable", accumulated_errors)

        # Accumulate error for next iteration
        accumulated_errors.append(format_error(attempt, execution_result))

    # Exhausted all attempts
    return failure("max_attempts", accumulated_errors)
```

### Temperature Escalation Strategy

Temperature increases with each retry to encourage the model to explore alternative approaches:

| Attempt | Temperature | Strategy |
|---------|-------------|----------|
| 1 | 0.3 | Conservative: follow established patterns |
| 2 | 0.4 | Minor variations |
| 3 | 0.5 | Moderate creativity |
| 4 | 0.6 | Explore alternatives |
| 5 | 0.7 | Maximum creativity, try novel approaches |

### Error Feedback Format

Accumulated errors are fed back to the LLM to inform subsequent attempts:

```
## Previous Attempts (DO NOT repeat these mistakes):

### Attempt 1 (temperature=0.3)
Error: SyntaxError
Message: unexpected indent at line 15
Code snippet:
    def calculate():
        result = x + y
         return result  # <-- unexpected indent

### Attempt 2 (temperature=0.4)
Error: RuntimeError
Message: division by zero in calculate_average()
Traceback:
  File "solution.py", line 23, in calculate_average
    return total / count
ZeroDivisionError: division by zero

Please generate a corrected solution that addresses these issues.
```

### Early Termination Conditions

The loop terminates early for errors that cannot be resolved through retry:

| Error Type | Example | Action |
|------------|---------|--------|
| Missing dependency | `ModuleNotFoundError: No module named 'nonexistent'` | Terminate |
| Permission denied | `PermissionError: [Errno 13]` | Terminate |
| Resource exhaustion | `MemoryError`, `TimeoutError` | Terminate |
| Invalid specification | Malformed task input | Terminate |

### Success Criteria

A task is considered successful when:

1. **Compilation succeeds**: Code parses without syntax errors
2. **Tests pass** (if `require_tests_pass`): All pytest tests pass
3. **Lint passes** (if `require_lint_pass`): pylint score >= 6.0

---

## RAG Pipeline

### Code Chunking Strategy

The chunker (`rag-api/chunker.py`) splits code into semantic chunks for vector storage.

**Line-Based Chunking** (default):
| Setting | Value | Description |
|---------|-------|-------------|
| Chunk size | 100 lines | Target chunk length |
| Overlap | 20 lines | Context preservation between chunks |
| Boundary | File | Never split across files |

**Chunk Metadata**:
```json
{
  "chunk_id": "chunk_a1b2c3d4",
  "project_id": "proj_xyz789",
  "file_path": "src/auth/jwt.py",
  "start_line": 45,
  "end_line": 144,
  "language": "python",
  "content_hash": "sha256:abc123..."
}
```

### Retrieval Strategy

1. **Query Embedding**
   - Convert user query to 384-dimensional vector
   - Use same embedding model as indexing (consistency)

2. **Semantic Search**
   - Search Qdrant with query vector
   - Retrieve top-50 candidates
   - Filter by project_id

3. **Ranking**
   - Primary: Vector similarity (cosine distance)
   - Secondary: Recency weight
   - Tertiary: File size penalty (prefer smaller, focused files)

4. **Selection**
   - Select top-20 chunks
   - Respect token budget (8000 tokens)
   - Prefer diverse file coverage

### Context Assembly

Retrieved chunks are formatted for the LLM system prompt:

```
## Retrieved Context

The following code snippets are relevant to your query:

### File: src/auth/jwt.py (lines 45-144)
```python
def validate_token(token: str) -> User:
    """Validate JWT and return user."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        return get_user(user_id)
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
```

### File: src/models/user.py (lines 1-50)
```python
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    hashed_password = Column(String)
```

Use this context to inform your response.
```

### Provenance Tracking

The provenance system (`rag-api/provenance.py`) tracks content source via git history:

**AI-Generated Detection**:
Identifies AI-generated content by commit message markers:
- `[AI]`, `[AUTO]`, `[GENERATED]`, `[BOT]`
- Author patterns: `dependabot`, `renovate`, `github-actions`

**Quality Scoring**:
```
Quality = (lint_score/10 × 0.4) + (test_coverage × 0.4) + (complexity_score × 0.2)
```

---

## Continuous Learning Pipeline

### Training Data Collection

Successful task completions are stored in Redis for training:

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "prompt": "Implement a binary search function",
  "context": "## Retrieved Context\n...",
  "completion": "def binary_search(arr, target):\n    ...",
  "quality_score": 0.85,
  "attempts": 2,
  "total_tokens": 1250,
  "rating": 5,
  "timestamp": "2026-01-31T15:30:00Z"
}
```

### Training Selection Criteria

Only high-quality completions are used:

| Criterion | Threshold |
|-----------|-----------|
| Quality score | >= 0.6 |
| User rating | >= 4 (if available) |
| Minimum examples | >= 10 per batch |

### LoRA Fine-Tuning

**Why LoRA?**
- Full fine-tuning of 14B model requires >100GB VRAM
- LoRA trains only ~0.1% of parameters
- Adapters are small (~50MB) and hot-swappable

**Configuration**:
```python
LoraConfig(
    r=8,                    # Low-rank dimension
    lora_alpha=16,          # Scaling factor
    target_modules=[        # Which layers to adapt
        "q_proj", "k_proj",
        "v_proj", "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Training Parameters**:
```python
TrainingArguments(
    learning_rate=2e-4,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    weight_decay=0.01,
    fp16=False,  # CPU training
    logging_steps=10,
    save_strategy="epoch"
)
```

### Hot-Swap Deployment

New adapters are deployed without restarting llama-server:

```
/models/lora/
├── adapter_20260131_020000/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── README.md
├── adapter_20260130_020000/
├── adapter_20260129_020000/
└── latest -> adapter_20260131_020000/
```

llama-server loads from `latest` symlink on each inference request, enabling zero-downtime adapter updates.

---

## API Reference

### RAG API Endpoints

#### Project Sync
```http
POST /v1/projects/sync
Content-Type: application/json
Authorization: Bearer sk_xxx

{
  "project_name": "my-repo",
  "project_hash": "abc123...",
  "files": [
    {
      "path": "src/main.py",
      "content": "...",
      "hash": "def456..."
    }
  ],
  "metadata": {
    "repo_url": "https://github.com/..."
  }
}

Response:
{
  "project_id": "proj_xyz789",
  "status": "synced",
  "stats": {
    "files_indexed": 50,
    "chunks_created": 250,
    "loc_indexed": 5000
  },
  "sync_time_ms": 2500
}
```

#### Chat Completion
```http
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer sk_xxx

{
  "model": "Qwen3-14B",
  "messages": [
    {"role": "user", "content": "Explain this function"}
  ],
  "project_id": "proj_xyz789",
  "stream": false
}

Response:
{
  "id": "chatcmpl-123",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "..."
    }
  }],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 200,
    "total_tokens": 350
  }
}
```

#### Task Submit
```http
POST /v1/tasks/submit
Content-Type: application/json
Authorization: Bearer sk_xxx

{
  "prompt": "Implement binary search",
  "type": "code_generation",
  "priority": "p1",
  "project_id": "proj_xyz789",
  "max_attempts": 5,
  "require_tests_pass": true,
  "test_code": "def test_search():\n    assert binary_search([1,2,3], 2) == 1"
}

Response:
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

#### Task Status
```http
GET /v1/tasks/{task_id}/status
Authorization: Bearer sk_xxx

Response:
{
  "id": "550e8400...",
  "status": "completed",
  "attempts": 2,
  "result": {
    "success": true,
    "stop_reason": "success",
    "final_code": "def binary_search(arr, target):\n    ...",
    "attempts_count": 2,
    "total_duration_ms": 5500,
    "total_tokens": 1200
  },
  "completed_at": "2026-01-31T15:30:00Z"
}
```

### API Portal Endpoints

#### Key Validation (Internal)
```http
POST /api/validate-key
Content-Type: application/json

{
  "api_key": "sk_xxx"
}

Response:
{
  "valid": true,
  "user": "john_doe",
  "rate_limit": 1000,
  "key_name": "My API Key"
}
```

---

## Security Model

### Authentication

| Mechanism | Usage |
|-----------|-------|
| API Keys | Service-to-service, external API access |
| JWT | Web UI session management |
| Key hashing | SHA256 before storage |
| Cache TTL | 60 seconds for validation cache |

### Rate Limiting

```
Algorithm: Sliding window (Redis)
Window: 60 seconds
Key: ratelimit:{key_hash}:count
Enforcement: LLM Proxy
```

### Code Isolation

The Sandbox provides multiple isolation layers:

| Layer | Protection |
|-------|------------|
| Container | Network namespace isolation |
| Timeout | 60 second maximum execution |
| Memory | 512MB limit |
| Filesystem | Temporary workspace, auto-cleanup |

### Data Protection

| Concern | Mitigation |
|---------|------------|
| Project isolation | Per-project Qdrant collections |
| Data retention | Configurable TTL (default 7 days) |
| Secrets | No code content in logs |
| Key storage | Only hashes stored, never plaintext |

---

## Port Reference

| Service | Internal Port | NodePort | Purpose |
|---------|---------------|----------|---------|
| llama-server | 8000 | 32735 | LLM inference |
| llm-proxy | 8000 | 30080 | External API gateway |
| RAG API | 8001 | 31144 | RAG orchestration |
| API Portal | 3000 | 30000 | Web UI + key management |
| Embedding | 8080 | 30808 | Text embeddings |
| Qdrant HTTP | 6333 | 30633 | Vector DB REST API |
| Qdrant gRPC | 6334 | 30634 | Vector DB gRPC |
| Sandbox | 8020 | 30820 | Code execution |
| Dashboard | 3001 | 30001 | Monitoring UI |
| Redis | 6379 | — | Internal only |
