import hashlib
import logging
import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Header, Depends
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
from chunker import chunk_project_files, count_lines
from vector_store import vector_store
from rag import rag_enhanced_completion, simple_completion, forward_to_llama_stream

# Redis for task queue
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except:
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
    logger.info(f"Qdrant: {config.qdrant.host}:{config.qdrant.port}")
    logger.info(f"Embedding: {config.embedding.base_url}")

    # Cleanup expired projects on startup
    project_store.cleanup_expired()

    yield

    logger.info("RAG API shutting down")


app = FastAPI(
    title="RAG API",
    description="RAG-enhanced API for code-aware LLM interactions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    total_loc = count_lines(files)
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

    # Chunk the files
    chunks = chunk_project_files(
        project_id=project_id,
        files=files,
        max_chunk_lines=config.chunking.max_chunk_lines,
        overlap_lines=config.chunking.overlap_lines,
        use_ast_chunking=True
    )

    # Index chunks
    try:
        indexed = await vector_store.index_chunks(project_id, chunks)
    except Exception as e:
        logger.error(f"Failed to index chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

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
    # Delete from vector store
    vector_store.delete_collection(project_id)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port
    )
