import hashlib
import logging
import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum

import redis
import httpx
from config import config
from storage import project_store, ProjectMetadata
from rag import (
    rag_enhanced_completion, simple_completion, forward_to_llama_stream,
    invalidate_cache,
    write_pattern_async, record_pattern_outcome,
    record_route_feedback, is_routing_enabled,
)
from indexer.tree_builder import build_tree_from_files
from indexer.bm25_index import BM25Index
from indexer.summarizer import summarize_tree, collect_summaries
from indexer.persistence import save_index, load_index, delete_index

# Redis for task queue
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception:
    redis_client = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("RAG API starting up")
    logger.info(f"Llama server: {config.llama.base_url}")

    # Cleanup expired projects on startup
    project_store.cleanup_expired()

    # Load seed persistent patterns into Pattern Cache
    try:
        from cache.seed_patterns import load_seed_patterns
        await load_seed_patterns()
    except Exception as e:
        logger.warning(f"Failed to load seed patterns: {e}")

    yield

    logger.info("RAG API shutting down")


app = FastAPI(
    title="RAG API",
    description="RAG-enhanced API for code-aware LLM interactions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — configurable via CORS_ORIGINS env var (comma-separated)
_cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
_cors_origins = [origin.strip() for origin in _cors_origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Key validation cache (in-memory, short-lived)
_key_cache: Dict[str, dict] = {}
_key_cache_ttl = 60  # seconds


async def validate_key_with_portal(api_key: str) -> Optional[dict]:
    """Validate API key with the API portal service."""
    import time

    # Check cache first
    cached = _key_cache.get(api_key)
    if cached and time.time() - cached["timestamp"] < _key_cache_ttl:
        return cached["data"]

    # Call portal validation endpoint
    portal_url = config.api_portal_url
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{portal_url}/api/validate-key",
                json={"api_key": api_key}
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    # Cache the result
                    _key_cache[api_key] = {
                        "timestamp": time.time(),
                        "data": data
                    }
                    return data
    except Exception as e:
        logger.warning(f"Failed to validate key with portal: {e}")
        # Fall through to check if it's a legacy key

    return None


# Auth dependency
async def verify_api_key(authorization: str = Header(None)) -> str:
    """Verify API key from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Extract key from "Bearer sk-xxx" format
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization format")

    key = parts[1]

    # Validate with API portal
    validation = await validate_key_with_portal(key)
    if validation:
        logger.info(f"API key validated for user: {validation.get('user')}")
        return key

    raise HTTPException(status_code=401, detail="Invalid API key")


# Request/Response models
class FileInfo(BaseModel):
    path: str
    content: str
    hash: Optional[str] = None


class SyncRequest(BaseModel):
    project_name: str
    project_hash: str
    files: List[FileInfo]
    metadata: Optional[Dict[str, Any]] = None


class SyncResponse(BaseModel):
    project_id: str
    status: str
    stats: Optional[Dict[str, int]] = None
    sync_time_ms: Optional[int] = None
    message: Optional[str] = None


class ProjectStatus(BaseModel):
    project_id: str
    project_name: str
    status: str
    stats: Dict[str, Any]
    last_sync: str
    expires_at: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    project_id: Optional[str] = None
    tools: Optional[List[Dict]] = None
    max_tokens: int = 16384
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False


# Endpoints
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rag-api"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "RAG API",
        "version": "1.0.0",
        "endpoints": {
            "sync": "POST /v1/projects/sync",
            "chat": "POST /v1/chat/completions",
            "projects": "GET /v1/projects",
            "models": "GET /v1/models"
        }
    }


@app.post("/v1/projects/sync", response_model=SyncResponse)
async def sync_project(
    request: SyncRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Sync a project's codebase for RAG indexing.
    """
    import time
    start_time = time.time()

    # Validate limits
    files = [{"path": f.path, "content": f.content} for f in request.files]
    total_files = len(files)
    total_loc = sum(f["content"].count("\n") + 1 for f in files)
    total_size = sum(len(f["content"].encode()) for f in files)

    if total_files > config.limits.max_files:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {total_files} > {config.limits.max_files}"
        )

    if total_loc > config.limits.max_loc:
        raise HTTPException(
            status_code=400,
            detail=f"Too many lines: {total_loc} > {config.limits.max_loc}"
        )

    if total_size > config.limits.max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Total size too large: {total_size / 1024 / 1024:.1f}MB > {config.limits.max_size_mb}MB"
        )

    # Generate project ID
    project_id = project_store.generate_project_id(request.project_name, api_key)

    # Check if project already exists with same hash
    existing = project_store.get_metadata(project_id)
    if existing and existing.project_hash == request.project_hash:
        return SyncResponse(
            project_id=project_id,
            status="already_synced",
            message="Project hash matches, no sync needed"
        )

    indexed = 0

    # PageIndex tree building
    try:
        # Load existing index for incremental re-summarization
        old_file_hashes = {}
        existing_summaries = {}
        existing = load_index(project_id)
        if existing:
            old_tree, _ = existing
            old_file_hashes = old_tree.file_hashes
            existing_summaries = collect_summaries(old_tree.root)

        # Build tree from files
        tree_index = build_tree_from_files(
            project_id=project_id,
            files=files,
            project_name=request.project_name,
        )

        # Generate LLM summaries (bottom-up)
        await summarize_tree(
            root=tree_index.root,
            llama_url=config.llama.base_url,
            existing_summaries=existing_summaries,
            file_hashes=tree_index.file_hashes,
            old_file_hashes=old_file_hashes,
        )

        # Build BM25 index
        bm25_index = BM25Index()
        bm25_index.build_from_tree(tree_index)

        # Persist to disk
        save_index(project_id, tree_index, bm25_index)

        # Invalidate in-memory cache
        invalidate_cache(project_id)

        indexed = tree_index.root.node_count()
        logger.info(
            f"PageIndex built for {project_id}: {indexed} nodes, "
            f"{bm25_index.num_docs} BM25 docs"
        )
    except Exception as e:
        logger.error(f"Failed to build PageIndex: {e}")
        raise HTTPException(status_code=500, detail=f"PageIndex build failed: {str(e)}")

    # Save project metadata
    project_store.create_project(
        project_id=project_id,
        project_name=request.project_name,
        project_hash=request.project_hash,
        files=files,
        chunks_created=indexed,
        ttl_hours=config.limits.project_ttl_hours
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    return SyncResponse(
        project_id=project_id,
        status="synced",
        stats={
            "files_indexed": total_files,
            "chunks_created": indexed,
            "loc_indexed": total_loc
        },
        sync_time_ms=elapsed_ms
    )


@app.get("/v1/projects/{project_id}/status", response_model=ProjectStatus)
async def get_project_status(
    project_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get project status and statistics."""
    meta = project_store.get_metadata(project_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectStatus(
        project_id=meta.project_id,
        project_name=meta.project_name,
        status=meta.status,
        stats={
            "files_indexed": meta.files_indexed,
            "chunks_created": meta.chunks_created,
            "loc_indexed": meta.loc_indexed,
            "size_bytes": meta.size_bytes
        },
        last_sync=meta.created_at,
        expires_at=meta.expires_at
    )


@app.get("/v1/projects")
async def list_projects(api_key: str = Depends(verify_api_key)):
    """List all projects."""
    projects = project_store.list_projects()
    return {
        "projects": [
            {
                "project_id": p.project_id,
                "project_name": p.project_name,
                "status": p.status,
                "last_sync": p.created_at
            }
            for p in projects
        ]
    }


@app.delete("/v1/projects/{project_id}")
async def delete_project(
    project_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete a project."""
    # Delete PageIndex data
    delete_index(project_id)
    invalidate_cache(project_id)

    # Delete from file store
    deleted = project_store.delete_project(project_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"deleted": True, "project_id": project_id}


def log_request_metrics(request_type: str, success: bool, tokens: int = 0, model: str = ""):
    """Log request metrics to Redis for dashboard."""
    if not redis_client:
        return
    try:
        from datetime import date
        today = date.today().isoformat()

        # Increment daily counters
        redis_client.hincrby(f"metrics:daily:{today}", "tasks_total", 1)
        if success:
            redis_client.hincrby(f"metrics:daily:{today}", "tasks_success", 1)
        redis_client.hincrby(f"metrics:daily:{today}", "tokens_total", tokens)

        # Add to recent tasks list
        task_record = json.dumps({
            "type": request_type,
            "model": model,
            "tokens": tokens,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        redis_client.lpush("metrics:recent_tasks", task_record)
        redis_client.ltrim("metrics:recent_tasks", 0, 99)  # Keep last 100
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    OpenAI-compatible chat completions endpoint with optional RAG enhancement.
    Supports both streaming and non-streaming responses.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Build kwargs for optional params
    kwargs = {}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p

    request_type = "rag_completion" if request.project_id else "chat_completion"

    if request.project_id:
        # Verify project exists
        if not project_store.project_exists(request.project_id):
            raise HTTPException(status_code=404, detail="Project not found")

        # RAG-enhanced completion
        if request.stream:
            log_request_metrics(request_type, True, 0, request.model)  # Log at start for streaming
            generator = await rag_enhanced_completion(
                project_id=request.project_id,
                messages=messages,
                model=request.model,
                tools=request.tools,
                max_tokens=request.max_tokens,
                stream=True,
                **kwargs
            )
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        result = await rag_enhanced_completion(
            project_id=request.project_id,
            messages=messages,
            model=request.model,
            tools=request.tools,
            max_tokens=request.max_tokens,
            stream=False,
            **kwargs
        )
        tokens = result.get("usage", {}).get("total_tokens", 0) if isinstance(result, dict) else 0
        log_request_metrics(request_type, True, tokens, request.model)
        return result
    else:
        # Simple pass-through
        if request.stream:
            log_request_metrics(request_type, True, 0, request.model)  # Log at start for streaming
            generator = forward_to_llama_stream(
                messages=messages,
                model=request.model,
                tools=request.tools,
                max_tokens=request.max_tokens,
                **kwargs
            )
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        result = await simple_completion(
            messages=messages,
            model=request.model,
            tools=request.tools,
            max_tokens=request.max_tokens,
            stream=False,
            **kwargs
        )
        tokens = result.get("usage", {}).get("total_tokens", 0) if isinstance(result, dict) else 0
        log_request_metrics(request_type, True, tokens, request.model)
        return result


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models (proxy to llama-server)."""
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{config.llama.base_url}/v1/models")
        return response.json()


# Task Queue Models and Endpoints
class Priority(str, Enum):
    INTERACTIVE = "p0"
    FIRE_FORGET = "p1"
    BATCH = "p2"

class TaskSubmitRequest(BaseModel):
    prompt: str
    type: str = "code_generation"
    priority: str = "p1"
    project_id: Optional[str] = None
    max_attempts: int = 5
    require_tests_pass: bool = True
    test_code: Optional[str] = None

class TaskSubmitResponse(BaseModel):
    task_id: str
    status: str

@app.post("/v1/tasks/submit", response_model=TaskSubmitResponse)
async def submit_task(
    request: TaskSubmitRequest,
    api_key: str = Depends(verify_api_key)
):
    """Submit a task for async processing."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Task queue not available")

    task_id = str(uuid.uuid4())
    task_data = {
        "id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "priority": request.priority,
        "status": "pending",
        "type": request.type,
        "prompt": request.prompt,
        "project_id": request.project_id,
        "max_attempts": request.max_attempts,
        "timeout_seconds": 300,
        "require_tests_pass": request.require_tests_pass,
        "require_lint_pass": False,
        "test_code": request.test_code,
        "attempts": [],
        "result": None,
        "completed_at": None,
        "metrics": {}
    }

    # Store task
    redis_client.hset(f"task:{task_id}", mapping={"data": json.dumps(task_data)})
    # Add to priority queue
    redis_client.rpush(f"tasks:{request.priority}", task_id)

    return TaskSubmitResponse(task_id=task_id, status="pending")

@app.get("/v1/tasks/{task_id}/status")
async def get_task_status(
    task_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get current status of a submitted task."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Task queue not available")

    data = redis_client.hget(f"task:{task_id}", "data")
    if not data:
        raise HTTPException(status_code=404, detail="Task not found")

    task = json.loads(data)
    return {
        "id": task["id"],
        "status": task["status"],
        "attempts": len(task.get("attempts", [])),
        "result": task.get("result"),
        "completed_at": task.get("completed_at")
    }

@app.get("/v1/queue/stats")
async def get_queue_stats(api_key: str = Depends(verify_api_key)):
    """Get current queue statistics."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Task queue not available")

    return {
        "p0_waiting": redis_client.llen("tasks:p0"),
        "p1_waiting": redis_client.llen("tasks:p1"),
        "p2_waiting": redis_client.llen("tasks:p2"),
        "total_waiting": sum([
            redis_client.llen("tasks:p0"),
            redis_client.llen("tasks:p1"),
            redis_client.llen("tasks:p2")
        ])
    }


# ──────────────────────────────────────────────────────────────
# Pattern Cache: Write Path + Monitoring Endpoints
# ──────────────────────────────────────────────────────────────

class PatternWriteRequest(BaseModel):
    query: str
    solution: str
    retry_count: int = 1
    max_retries: int = 5
    error_context: Optional[str] = None
    source_files: List[str] = []
    active_pattern_ids: List[str] = []
    success: bool = True


@app.post("/v1/patterns/write")
async def write_pattern(
    request: PatternWriteRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Write path: Extract and store a pattern from a successful task completion.
    This is called after a task passes tests.
    Runs async — returns immediately, extraction happens in background.
    """
    import asyncio

    if not request.success:
        # Record failure outcome for any active patterns
        if request.active_pattern_ids:
            asyncio.create_task(
                record_pattern_outcome(request.active_pattern_ids, success=False)
            )
        return {"status": "recorded_failure"}

    # Fire-and-forget: extract pattern in background
    asyncio.create_task(
        write_pattern_async(
            query=request.query,
            solution=request.solution,
            retry_count=request.retry_count,
            max_retries=request.max_retries,
            error_context=request.error_context,
            source_files=request.source_files,
            active_pattern_ids=request.active_pattern_ids,
        )
    )

    # Record success outcome for active patterns
    if request.active_pattern_ids:
        asyncio.create_task(
            record_pattern_outcome(request.active_pattern_ids, success=True)
        )

    return {"status": "accepted", "message": "Pattern extraction started in background"}


@app.get("/internal/cache/stats")
async def cache_stats():
    """Get Pattern Cache statistics — size, hit rate, tier distribution, top patterns."""
    from cache.pattern_store import get_pattern_store

    store = get_pattern_store()
    stats = store.get_stats()

    # Add top patterns by score
    if stats.get("available"):
        top_stm = store.get_stm_patterns(limit=5)
        top_ltm = store.get_ltm_patterns(limit=5)

        stats["top_stm"] = [
            {"id": p.id, "type": p.type.value, "summary": p.summary[:80],
             "access_count": p.access_count, "surprise": p.surprise_score}
            for p in top_stm
        ]
        stats["top_ltm"] = [
            {"id": p.id, "type": p.type.value, "summary": p.summary[:80],
             "access_count": p.access_count, "surprise": p.surprise_score}
            for p in top_ltm
        ]

    return stats


@app.post("/internal/cache/flush")
async def flush_cache():
    """Clear the entire pattern cache (for testing/reset)."""
    from cache.pattern_store import get_pattern_store

    store = get_pattern_store()
    store.flush()

    # Reload seed patterns
    try:
        from cache.seed_patterns import load_seed_patterns
        await load_seed_patterns()
    except Exception as e:
        logger.warning(f"Failed to reload seed patterns after flush: {e}")

    return {"status": "flushed"}


@app.post("/internal/cache/consolidate")
async def trigger_consolidation():
    """Manually trigger STM → LTM consolidation."""
    from cache.consolidator import run_consolidation

    await run_consolidation()
    return {"status": "consolidation_complete"}


# ──────────────────────────────────────────────────────────────
# Confidence Router: Internal Monitoring Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/internal/router/stats")
async def router_stats():
    """Get Confidence Router statistics — Thompson state, route distribution, difficulty histogram."""
    if not is_routing_enabled():
        return {"enabled": False, "message": "Routing is disabled (ROUTING_ENABLED=false)"}

    try:
        import redis as redis_lib
        from router.route_selector import get_all_thompson_states
        from router.feedback_recorder import get_routing_stats

        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        r = redis_lib.from_url(redis_url, decode_responses=True)

        thompson = get_all_thompson_states(r)
        stats = get_routing_stats(r)

        return {
            "enabled": True,
            "thompson_state": thompson,
            "aggregate_stats": stats,
        }
    except Exception as e:
        logger.error(f"Failed to get router stats: {e}")
        return {"enabled": True, "error": str(e)}


@app.post("/internal/router/reset")
async def router_reset():
    """Reset Thompson Sampling state for recalibration."""
    if not is_routing_enabled():
        return {"status": "skipped", "message": "Routing is disabled"}

    try:
        import redis as redis_lib
        from router.route_selector import reset_thompson_state
        from router.feedback_recorder import reset_stats

        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        r = redis_lib.from_url(redis_url, decode_responses=True)

        reset_thompson_state(r)
        reset_stats(r)

        return {"status": "reset", "message": "Thompson state and stats reset to uniform priors"}
    except Exception as e:
        logger.error(f"Failed to reset router: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/internal/router/feedback")
async def router_feedback(
    route: str,
    difficulty_bin: str,
    success: bool,
):
    """Manually record a routing outcome for Thompson Sampling."""
    try:
        record_route_feedback(route, difficulty_bin, success)
        return {"status": "recorded"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ──────────────────────────────────────────────────────────────
# Geometric Lens: Internal Monitoring Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/internal/lens/stats")
async def lens_stats():
    """Get Geometric Lens status — model info, enabled state."""
    try:
        from geometric_lens.service import get_model_info
        return get_model_info()
    except Exception as e:
        return {"loaded": False, "enabled": False, "error": str(e)}


@app.api_route("/internal/lens/evaluate", methods=["GET", "POST"])
async def lens_evaluate(request: Request, query: str = None):
    """Evaluate a query through the Geometric Lens (for testing).

    Accepts GET with ?query= param or POST with JSON {"query": "..."}.
    """
    if request.method == "POST":
        body = await request.json()
        query = body.get("query", body.get("text", ""))
    if not query:
        raise HTTPException(status_code=422, detail="Missing 'query' parameter")
    try:
        from geometric_lens.service import evaluate_and_correct, get_geometric_energy, is_enabled
        if not is_enabled():
            return {"enabled": False, "message": "Geometric Lens disabled"}

        energy_before, energy_after, corrected = evaluate_and_correct(query)
        normalized = get_geometric_energy(query)

        return {
            "enabled": True,
            "energy_before": energy_before,
            "energy_after": energy_after,
            "energy_normalized": normalized,
            "corrected": corrected is not None,
        }
    except Exception as e:
        return {"error": str(e)}


class LensScoreTextRequest(BaseModel):
    text: str


@app.post("/internal/lens/score-text")
async def lens_score_text(request: LensScoreTextRequest):
    """Score a text string through the Geometric Lens. Returns raw and normalized energy."""
    try:
        import geometric_lens.service as lens_service
        from geometric_lens.embedding_extractor import extract_embedding

        if not lens_service.is_enabled():
            return {"energy": 0.0, "normalized": 0.5, "enabled": False}

        if not lens_service._ensure_models_loaded():
            return {"energy": 0.0, "normalized": 0.5, "error": "models_not_loaded"}

        import torch

        emb = extract_embedding(request.text)
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            energy = lens_service._cost_field(x).item()

        # Normalize to [0,1]: PASS ~5.00 -> ~0.1, FAIL ~14.04 -> ~0.9
        # Training targets were 2.0/25.0; measured outputs converged to 5.00/14.04
        normalized = 1.0 / (1.0 + 2.718 ** (-(energy - 9.5) / 3.0))
        normalized = min(1.0, max(0.0, normalized))

        return {"energy": energy, "normalized": normalized, "enabled": True}
    except Exception as e:
        logger.error(f"Lens score-text failed: {e}")
        return {"energy": 0.0, "normalized": 0.5, "error": str(e)}


class LensRetrainRequest(BaseModel):
    training_data: List[Dict]
    epochs: int = 50


@app.post("/internal/lens/retrain")
async def lens_retrain(request: LensRetrainRequest):
    """Retrain C(x) on accumulated pass/fail embeddings from benchmark execution."""
    try:
        from geometric_lens.training import retrain_cost_field_bce
        from geometric_lens.service import reload_weights
        import os

        embeddings = [d["embedding"] for d in request.training_data]
        labels = [d["label"] for d in request.training_data]

        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "geometric_lens", "models", "cost_field.pt"
        )

        metrics = retrain_cost_field_bce(
            embeddings=embeddings,
            labels=labels,
            epochs=request.epochs,
            save_path=save_path,
        )

        # Remove non-serializable 'model' key from metrics
        metrics.pop("model", None)

        # Hot-reload if retrain succeeded and wasn't skipped
        if not metrics.get("skipped", False):
            reload_result = reload_weights()
            metrics["reload_status"] = reload_result.get("status", "unknown")

        return {"status": "ok", "metrics": metrics}
    except Exception as e:
        logger.error(f"Lens retrain failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/internal/lens/reload")
async def lens_reload():
    """Reload Geometric Lens weights from disk after retraining."""
    try:
        from geometric_lens.service import reload_weights
        result = reload_weights()
        return {"status": result.get("status", "unknown"), **result}
    except Exception as e:
        logger.error(f"Lens reload failed: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port
    )
