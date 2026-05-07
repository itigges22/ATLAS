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
from config import config, api_keys
from storage import project_store, ProjectMetadata
from pipeline import (
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


# Boot-time self-test cache. Populated in lifespan(); read by /health and /ready.
# Keys: lens_enabled, lens_cost_field_loaded, lens_cost_field_dim, lens_gx_loaded,
#       lens_gx_type, embed_dim, self_test_pass, self_test_error.
_BOOT_STATE: Dict[str, Any] = {
    "lens_enabled": False,
    "lens_cost_field_loaded": False,
    "lens_cost_field_dim": None,
    "lens_gx_loaded": False,
    "lens_gx_type": "none",
    "embed_dim": None,
    "self_test_pass": False,
    "self_test_error": None,
}


def _run_lens_self_test() -> None:
    """Boot-time C(x)/G(x) self-test.

    Loads weights, fetches a dummy embedding from llama-server, checks the
    cost-field input dim matches the embedding dim (the silent killer
    behind PC-018), and runs a single C(x) evaluation. Populates
    _BOOT_STATE so /health and /ready can report what actually works.
    Never raises — failures are recorded and surfaced via /ready 503.
    """
    from geometric_lens import service as lens_service

    _BOOT_STATE["lens_enabled"] = lens_service.is_enabled()
    if not lens_service.is_enabled():
        _BOOT_STATE["self_test_error"] = "GEOMETRIC_LENS_ENABLED is false"
        return

    try:
        loaded = lens_service._ensure_models_loaded()
        info = lens_service.get_model_info()
        _BOOT_STATE["lens_cost_field_loaded"] = bool(info.get("loaded"))
        _BOOT_STATE["lens_gx_loaded"] = bool(info.get("gx_loaded"))
        _BOOT_STATE["lens_gx_type"] = info.get("gx_type", "none")
        if not loaded:
            _BOOT_STATE["self_test_error"] = (
                "lens model files missing — run scripts/download-models.sh --lens"
            )
            return

        cf = lens_service._cost_field
        if cf is not None:
            cf_dim = next(cf.parameters()).shape[1] if hasattr(cf, "parameters") else None
            _BOOT_STATE["lens_cost_field_dim"] = cf_dim

        from geometric_lens.embedding_extractor import extract_embedding
        emb = extract_embedding("def add(a, b): return a + b")
        _BOOT_STATE["embed_dim"] = len(emb)

        cf_dim = _BOOT_STATE["lens_cost_field_dim"]
        if cf_dim is not None and cf_dim != len(emb):
            _BOOT_STATE["self_test_error"] = (
                f"lens/embedding dim mismatch: cost_field expects {cf_dim}, "
                f"llama-server returned {len(emb)} (likely wrong model file — see PC-018)"
            )
            return

        raw, norm = lens_service.evaluate_energy("def add(a, b): return a + b")
        if raw == 0.0 and norm == 0.0:
            _BOOT_STATE["self_test_error"] = "C(x) evaluation returned zeros"
            return

        _BOOT_STATE["self_test_pass"] = True
        logger.info(
            "Lens self-test OK: cf_dim=%s embed_dim=%s C(x)_raw=%.2f norm=%.3f gx=%s",
            cf_dim, len(emb), raw, norm, _BOOT_STATE["lens_gx_type"],
        )
    except Exception as e:
        _BOOT_STATE["self_test_error"] = f"{type(e).__name__}: {e}"
        logger.error("Lens self-test failed: %s", _BOOT_STATE["self_test_error"])


def _redis_state() -> Dict[str, Any]:
    if redis_client is None:
        return {"connected": False, "error": "client not initialised"}
    try:
        redis_client.ping()
        return {"connected": True}
    except Exception as e:
        return {"connected": False, "error": f"{type(e).__name__}: {e}"}


def _llama_state() -> Dict[str, Any]:
    url = config.llama.base_url.rstrip("/") + "/health"
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(url)
        return {"reachable": r.status_code == 200, "status_code": r.status_code}
    except Exception as e:
        return {"reachable": False, "error": f"{type(e).__name__}: {e}"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Geometric Lens API starting up")
    logger.info(f"Llama server: {config.llama.base_url}")

    # Cleanup expired projects on startup
    project_store.cleanup_expired()

    # Load seed persistent patterns into Pattern Cache
    try:
        from cache.seed_patterns import load_seed_patterns
        await load_seed_patterns()
    except Exception as e:
        logger.warning(f"Failed to load seed patterns: {e}")

    # Boot-time C(x)/G(x) self-test. Records state; never raises.
    _run_lens_self_test()
    if _BOOT_STATE["lens_enabled"] and not _BOOT_STATE["self_test_pass"]:
        logger.error(
            "Geometric Lens enabled but self-test FAILED: %s. /ready will return 503.",
            _BOOT_STATE["self_test_error"],
        )

    yield

    logger.info("Geometric Lens API shutting down")


app = FastAPI(
    title="Geometric Lens API",
    description="Geometric Lens API for code-aware LLM interactions with RAG, Pattern Cache, and Confidence Router",
    version="3.0.1",
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


# Auth dependency — local key file lookup, no remote portal.
async def verify_api_key(authorization: str = Header(None)) -> str:
    """Verify the bearer token against the locally-loaded api-keys.json."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization format")

    key = parts[1]
    metadata = api_keys.get(key)
    if metadata is not None:
        logger.info(f"API key validated for user: {metadata.get('user', 'unknown')}")
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
    """Structured per-subsystem health.

    Always returns 200 — this endpoint is for *information*, not gating.
    Use /ready for liveness/scoring-functional gating.
    """
    redis_st = _redis_state()
    llama_st = _llama_state()
    lens_ok = (
        not _BOOT_STATE["lens_enabled"] or _BOOT_STATE["self_test_pass"]
    )
    overall = (
        redis_st["connected"]
        and llama_st["reachable"]
        and lens_ok
    )
    return {
        "service": "geometric-lens",
        "status": "healthy" if overall else "degraded",
        "subsystems": {
            "redis": redis_st,
            "llama_server": llama_st,
            "lens": {
                "enabled": _BOOT_STATE["lens_enabled"],
                "cost_field_loaded": _BOOT_STATE["lens_cost_field_loaded"],
                "cost_field_dim": _BOOT_STATE["lens_cost_field_dim"],
                "embed_dim": _BOOT_STATE["embed_dim"],
                "gx_loaded": _BOOT_STATE["lens_gx_loaded"],
                "gx_type": _BOOT_STATE["lens_gx_type"],
                "self_test_pass": _BOOT_STATE["self_test_pass"],
                "self_test_error": _BOOT_STATE["self_test_error"],
            },
        },
    }


@app.get("/ready")
async def ready():
    """Readiness gate. 200 only when scoring is functional, 503 otherwise.

    Use this for orchestrator probes that should pull traffic away when
    lens scoring degrades (the silent-failure mode PC-019 was filed for).
    """
    redis_st = _redis_state()
    llama_st = _llama_state()
    lens_required = _BOOT_STATE["lens_enabled"]
    lens_ok = (not lens_required) or _BOOT_STATE["self_test_pass"]

    ok = redis_st["connected"] and llama_st["reachable"] and lens_ok
    payload = {
        "ready": ok,
        "redis": redis_st["connected"],
        "llama_server": llama_st["reachable"],
        "lens_self_test": _BOOT_STATE["self_test_pass"],
        "lens_required": lens_required,
        "reason": _BOOT_STATE["self_test_error"] if not lens_ok else None,
    }
    if not ok:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Geometric Lens API",
        "version": "3.0.1",
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


def _dispatch_pattern_write(request: PatternWriteRequest) -> dict:
    """Schedule pattern-write + outcome recording. Shared by /v1 and /internal handlers."""
    import asyncio

    if not request.success:
        if request.active_pattern_ids:
            asyncio.create_task(
                record_pattern_outcome(request.active_pattern_ids, success=False)
            )
        return {"status": "recorded_failure"}

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

    if request.active_pattern_ids:
        asyncio.create_task(
            record_pattern_outcome(request.active_pattern_ids, success=True)
        )

    return {"status": "accepted", "message": "Pattern extraction started in background"}


@app.post("/v1/patterns/write")
async def write_pattern(
    request: PatternWriteRequest,
    api_key: str = Depends(verify_api_key),
):
    """Auth-gated write path for external clients. See `_dispatch_pattern_write`."""
    return _dispatch_pattern_write(request)


@app.post("/internal/patterns/write")
async def write_pattern_internal(request: PatternWriteRequest):
    """Unauthenticated write path for in-stack service-to-service calls (v3-service).

    Mirrors `/v1/patterns/write` exactly but skips the bearer-token check, in
    line with the rest of the `/internal/*` surface (lens, sandbox, cache stats).
    Only reachable from inside the docker network in normal deployments.
    """
    return _dispatch_pattern_write(request)


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
        from geometric_lens.service import evaluate_energy, is_enabled
        if not is_enabled():
            return {"enabled": False, "message": "Geometric Lens disabled"}

        raw_energy, normalized = evaluate_energy(query)

        return {
            "enabled": True,
            "energy": raw_energy,
            "energy_normalized": normalized,
        }
    except Exception as e:
        return {"error": str(e)}


class LensScoreTextRequest(BaseModel):
    text: str


class LensScorePerStepRequest(BaseModel):
    text: str
    # Optional transformer-block index. None => last-layer (vanilla /embedding,
    # no PC-202 patch needed). Set to use the PC-202 layers extension and score
    # at the residual stream of a specific intermediate layer (PC-204 fusion).
    layer: Optional[int] = None


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

        # Qwen3.5-9B C(x) retrained: PASS ~13.2, FAIL ~24.9, midpoint ~19.0
        normalized = 1.0 / (1.0 + 2.718 ** (-(energy - 19.0) / 2.0))
        normalized = min(1.0, max(0.0, normalized))

        return {"energy": energy, "normalized": normalized, "enabled": True}
    except Exception as e:
        logger.error(f"Lens score-text failed: {e}")
        return {"energy": 0.0, "normalized": 0.5, "error": str(e)}


class LensRetrainRequest(BaseModel):
    training_data: List[Dict]
    epochs: int = 50
    domain: str = "LCB"
    use_replay: bool = True
    use_ewc: bool = True
    lambda_ewc: float = 1000.0


@app.post("/internal/lens/retrain")
async def lens_retrain(request: LensRetrainRequest):
    """Retrain C(x) on accumulated pass/fail embeddings from benchmark execution."""
    try:
        from geometric_lens.training import retrain_cost_field_bce
        from geometric_lens.service import reload_weights
        import os

        embeddings = [d["embedding"] for d in request.training_data]
        labels = [d["label"] for d in request.training_data]

        models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "geometric_lens", "models"
        )
        save_path = os.path.join(models_dir, "cost_field.pt")

        # Phase 4: Load replay buffer if enabled (4A-CL)
        replay_buffer = None
        if request.use_replay:
            from geometric_lens.replay_buffer import ReplayBuffer
            replay_buffer = ReplayBuffer(max_size=5000)
            replay_path = os.path.join(models_dir, "replay_buffer.json")
            replay_buffer.load(replay_path)  # OK if file doesn't exist yet

        # Phase 4: Load EWC state if enabled (4A-EWC)
        ewc = None
        if request.use_ewc:
            from geometric_lens.ewc import ElasticWeightConsolidation
            ewc = ElasticWeightConsolidation(lambda_ewc=request.lambda_ewc)
            ewc_path = os.path.join(models_dir, "ewc_state.pt")
            ewc.load(ewc_path)  # OK if file doesn't exist yet

        metrics = retrain_cost_field_bce(
            embeddings=embeddings,
            labels=labels,
            epochs=request.epochs,
            save_path=save_path,
            replay_buffer=replay_buffer,
            ewc=ewc,
            domain=request.domain,
        )

        # Remove non-serializable 'model' key from metrics
        metrics.pop("model", None)

        # Hot-reload if retrain succeeded and wasn't skipped
        if not metrics.get("skipped", False):
            reload_result = reload_weights()
            metrics["reload_status"] = reload_result.get("status", "unknown")

            # Phase 4: Save replay buffer and EWC state
            if replay_buffer is not None:
                replay_path = os.path.join(models_dir, "replay_buffer.json")
                replay_buffer.save(replay_path)
                metrics["replay_buffer_size"] = len(replay_buffer)

            if ewc is not None:
                ewc_path = os.path.join(models_dir, "ewc_state.pt")
                ewc.save(ewc_path)
                metrics["ewc_initialized"] = ewc.is_initialized

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


@app.post("/internal/lens/gx-score")
async def lens_gx_score(request: LensScoreTextRequest):
    """Combined C(x) + G(x) scoring in a single call.

    Returns C(x) energy, normalized energy, G(x) XGBoost quality prediction,
    and a human-readable verdict. Uses one embedding extraction for both models.
    """
    try:
        from geometric_lens.service import evaluate_combined, is_enabled

        if not is_enabled():
            return {
                "cx_energy": 0.0, "cx_normalized": 0.5,
                "gx_score": 0.5, "verdict": "unavailable",
                "enabled": False, "gx_available": False,
            }

        return evaluate_combined(request.text)
    except Exception as e:
        logger.error(f"Lens gx-score failed: {e}")
        return {
            "cx_energy": 0.0, "cx_normalized": 0.5,
            "gx_score": 0.5, "verdict": "error",
            "error": str(e),
        }


@app.post("/internal/lens/score-per-step")
async def lens_score_per_step(request: LensScorePerStepRequest):
    """PC-207 lens-as-PRM: score every token in the text instead of pooling.

    Returns C(x) and (when XGBoost is loaded) G(x) per generation step,
    plus aggregates across the whole sequence. Used by V3 candidate
    generation to abort off-rails candidates early instead of paying the
    full decode cost — the lens stops being ORM-by-timing (scores
    completed text) and becomes PRM-by-timing.

    Set `layer` to use the PC-202 hidden-states extension and score the
    residual stream at a specific intermediate layer (PC-204). Leave
    `layer` null to use the model's last-layer hidden state via vanilla
    /embedding (works on unpatched llama-server).
    """
    try:
        from geometric_lens.service import evaluate_per_step, is_enabled

        if not is_enabled():
            return {
                "enabled": False, "gx_available": False,
                "per_step": [], "aggregate": {}, "n_tokens": 0,
            }

        return evaluate_per_step(request.text, layer=request.layer)
    except Exception as e:
        logger.error(f"Lens score-per-step failed: {e}")
        return {
            "enabled": True, "gx_available": False,
            "per_step": [], "aggregate": {}, "n_tokens": 0,
            "error": str(e),
        }


@app.post("/internal/lens/correctability")
async def lens_correctability(request: LensScoreTextRequest):
    """Compute correctability + energy for a text string.

    Correctability measures how traversable the cost landscape is at this
    embedding — higher means repair is more likely to succeed.

    Requires G(x) metric tensor to be loaded.
    """
    try:
        from geometric_lens.service import evaluate_correctability, is_enabled

        if not is_enabled():
            return {"correctability": 0.0, "energy": 0.0, "normalized": 0.5,
                    "enabled": False}

        corr, raw_energy, norm_energy = evaluate_correctability(request.text)
        return {
            "correctability": corr,
            "energy": raw_energy,
            "normalized": norm_energy,
            "gx_available": corr > 0.0,
            "enabled": True,
        }
    except Exception as e:
        logger.error(f"Lens correctability failed: {e}")
        return {"correctability": 0.0, "energy": 0.0, "normalized": 0.5,
                "error": str(e)}


# ──────────────────────────────────────────────────────────────
# Sandbox Analysis Endpoint
# ──────────────────────────────────────────────────────────────

class SandboxAnalyzeRequest(BaseModel):
    code: str
    stdout: str = ""
    stderr: str = ""
    passed: bool = False
    expected_output: Optional[str] = None
    include_gx: bool = True


@app.post("/internal/sandbox/analyze")
async def sandbox_analyze(request: SandboxAnalyzeRequest):
    """Analyze sandbox output with structured error classification and G(x) scoring.

    Combines sandbox error parsing with G(x) quality prediction
    to produce actionable repair instructions.
    """
    try:
        from sandbox_analysis import analyze_sandbox_output, build_repair_prompt

        gx_score = None
        gx_result = {}

        # Optionally score with G(x)
        if request.include_gx and request.code:
            try:
                from geometric_lens.service import evaluate_combined, is_enabled
                if is_enabled():
                    gx_result = evaluate_combined(f"SOLUTION: {request.code}")
                    gx_score = gx_result.get("gx_score")
            except Exception as e:
                logger.warning(f"G(x) scoring in sandbox/analyze failed: {e}")

        analysis = analyze_sandbox_output(
            passed=request.passed,
            stdout=request.stdout,
            stderr=request.stderr,
            expected_output=request.expected_output,
            gx_score=gx_score,
        )

        repair_prompt = ""
        if not analysis.passed:
            repair_prompt = build_repair_prompt(
                analysis=analysis,
                original_code=request.code,
                gx_score=gx_score,
            )

        result = analysis.to_dict()
        result["repair_prompt"] = repair_prompt
        if gx_result:
            result["gx_score"] = gx_result.get("gx_score", 0.5)
            result["cx_energy"] = gx_result.get("cx_energy", 0.0)
            result["cx_normalized"] = gx_result.get("cx_normalized", 0.5)
            result["verdict"] = gx_result.get("verdict", "unavailable")

        return result
    except Exception as e:
        logger.error(f"Sandbox analysis failed: {e}")
        return {"error": str(e), "passed": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port
    )
