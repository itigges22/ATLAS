import logging
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from enum import Enum

import httpx
from config import config, api_keys
from sqlite_store import get_db_pool
from storage import project_store
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
from geometric_lens.auth_token import auth_headers as _svc_auth_headers


# ---------------------------------------------------------------------------
# Logging + HTTP-response sanitization helpers
# ---------------------------------------------------------------------------
#
# Untrusted strings (request bodies, file content, exception messages
# that wrap user data) can contain CR/LF and other control chars that
# fake additional log entries when written verbatim. _safe_log() strips
# those and bounds length so a single log line stays one line.
#
# For HTTP responses, _safe_detail() returns a short generic message
# while logging the real exception internally with a correlation ID.
# Useful for endpoints where leaking exception text would expose
# filesystem paths or internal types to a remote caller.
def _safe_log(value: object, maxlen: int = 200) -> str:
    """Render a value for inclusion in a log line. Strips CR/LF and
    other ASCII control chars, masks credential-shaped values,
    truncates to maxlen."""
    from geometric_lens.private_values import filter_private_values
    s = str(value)
    s = "".join(c for c in s if c == "\t" or 0x20 <= ord(c) < 0x7f or ord(c) > 0x9f)
    s = filter_private_values(s)
    if len(s) > maxlen:
        s = s[:maxlen] + "…"
    return s


def _safe_detail(e: Exception, op: str = "operation") -> str:
    """Log the real exception with a correlation ID; return a generic
    detail string safe to send in an HTTP response. Use for endpoints
    where exposing str(e) would leak internal paths / types."""
    err_id = uuid.uuid4().hex[:12]
    logger.error(f"[err {err_id}] {op} failed: {type(e).__name__}: {_safe_log(e)}",
                 exc_info=True)
    return f"{op} failed (error_id={err_id})"


# Configure logging. The private-value filter sits on the root handler
# so every logger in the process (pipeline, cache, router) is covered
# before serialization.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
from geometric_lens.structured_log import (install as _install_logging,
                                            set_request_id as _set_rid)
_install_logging("geometric-lens")
logger = logging.getLogger(__name__)

# Initialize the SQLite state store (patterns, router state, task queue,
# metrics) so the schema exists before the first request. A failure here
# leaves the store degraded: pattern cache/router fall back to neutral
# behavior and the task-queue endpoints return 503 (see ADR 0002).
try:
    get_db_pool()
except Exception as e:
    logger.error(f"Failed to initialize SQLite state store: {e}")


# Boot-time self-test cache. Populated in lifespan() and re-populated after a
# successful reload/retrain; read by /health and /ready.
# Keys: lens_enabled, lens_cost_field_loaded, lens_cost_field_dim, lens_gx_loaded,
#       lens_gx_type, lens_cx_calibrated, lens_gx_calibrated, lens_artifact_model,
#       embed_dim,
#       self_test_pass, self_test_error.
_BOOT_STATE_DEFAULTS: Dict[str, Any] = {
    "lens_enabled": False,
    "lens_cost_field_loaded": False,
    "lens_cost_field_dim": None,
    "lens_gx_loaded": False,
    "lens_gx_type": "none",
    "lens_cx_calibrated": False,
    "lens_gx_calibrated": False,
    "lens_artifact_model": None,
    "embed_dim": None,
    "self_test_pass": False,
    "self_test_error": None,
    # Drift fingerprint (drift_fingerprint.json next to the artifacts):
    # present=False → nothing to enforce; ok=None until checked.
    "fingerprint_present": False,
    "fingerprint_ok": None,
    "fingerprint_error": None,
}
_BOOT_STATE: Dict[str, Any] = dict(_BOOT_STATE_DEFAULTS)

# Serializes /internal/lens/retrain and /internal/lens/reload against each
# other — both mutate the geometric_lens.service module globals and (for
# retrain) the on-disk artifacts.
_lens_weights_lock = threading.Lock()


def _lens_drifted() -> bool:
    """True when the drift fingerprint check failed — scoring responses
    must not claim calibration in this state."""
    return _BOOT_STATE.get("fingerprint_ok") is False


def _apply_drift_flags(result: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp a scoring response with the drift state. On drift, calibration
    claims are withdrawn so a caller that ignores /ready still can't read
    the numbers as trustworthy."""
    drifted = _lens_drifted()
    result["drifted"] = drifted
    if drifted:
        for key in ("calibrated", "cx_calibrated", "gx_calibrated"):
            if key in result:
                result[key] = False
    return result


def _run_lens_self_test() -> None:
    """C(x)/G(x) self-test — run at boot and after a successful reload/retrain.

    Loads weights, fetches a dummy embedding from llama-server, checks the
    cost-field input dim matches the embedding dim (the silent killer
    behind PC-018), and runs a single C(x) evaluation. Populates
    _BOOT_STATE so /health and /ready can report what actually works.
    Never raises — failures are recorded and surfaced via /ready 503.
    """
    from geometric_lens import service as lens_service

    _BOOT_STATE.update(_BOOT_STATE_DEFAULTS)

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
        _BOOT_STATE["lens_cx_calibrated"] = bool(info.get("cx_calibrated"))
        _BOOT_STATE["lens_gx_calibrated"] = bool(info.get("gx_calibrated"))
        _BOOT_STATE["lens_artifact_model"] = info.get("artifact_model")
        if not loaded:
            _BOOT_STATE["self_test_error"] = info.get("error") or (
                "lens model files missing — run `atlas lens build`"
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

        # Drift fingerprint: re-score the reference texts written at
        # training time. A deviation means the serving stack no longer
        # matches what the artifacts were trained on (wrong pooling /
        # normalization / model) even though every request "works" —
        # the failure mode of the 2026-07-15 bench incident.
        from geometric_lens.drift import check_fingerprint
        fp_dir = lens_service.active_models_dir()
        if fp_dir:
            present, fp_ok, fp_detail = check_fingerprint(
                fp_dir, lambda t: lens_service.evaluate_energy(t)[0])
            _BOOT_STATE["fingerprint_present"] = present
            _BOOT_STATE["fingerprint_ok"] = fp_ok if present else None
            _BOOT_STATE["fingerprint_error"] = fp_detail or None
            if present and not fp_ok:
                _BOOT_STATE["self_test_error"] = fp_detail
                return

        _BOOT_STATE["self_test_pass"] = True
        logger.info(
            "Lens self-test OK: cf_dim=%s embed_dim=%s C(x)_raw=%.2f norm=%.3f gx=%s",
            cf_dim, len(emb), raw, norm, _BOOT_STATE["lens_gx_type"],
        )
    except Exception as e:
        # _safe_detail logs the full exception (with correlation ID); the
        # cached value that /health and /ready expose stays generic.
        _BOOT_STATE["self_test_error"] = (
            f"{type(e).__name__}: {_safe_detail(e, 'lens self-test')}"
        )


def _db_state() -> Dict[str, Any]:
    """State of the SQLite store backing patterns, router state, and the
    task queue. Probes a real table (not SELECT 1) so a schema-less or
    broken file/volume shows up as connected=False rather than only
    failing on first write."""
    from sqlite_store import DB_PATH
    try:
        pool = get_db_pool()
        with pool.get_connection() as conn:
            conn.execute("SELECT COUNT(*) FROM store_metadata")
        return {"connected": True, "path": DB_PATH}
    except Exception as e:
        # Full exception goes to the service log via _safe_detail; the
        # response keeps the connected/path/error keys (atlas doctor keys
        # on `connected`) with a generic error value.
        return {"connected": False, "path": DB_PATH,
                "error": f"{type(e).__name__}: {_safe_detail(e, 'sqlite state probe')}"}


def _llama_state() -> Dict[str, Any]:
    url = config.llama.base_url.rstrip("/") + "/health"
    try:
        with httpx.Client(timeout=2.0, headers=_svc_auth_headers()) as client:
            r = client.get(url)
        return {"reachable": r.status_code == 200, "status_code": r.status_code}
    except Exception as e:
        # Routine while llama-server is down — log at warning without a
        # traceback, and keep exception text out of the response.
        logger.warning("llama-server health probe failed: %s: %s",
                       type(e).__name__, _safe_log(e))
        return {"reachable": False,
                "error": f"{type(e).__name__}: llama-server unreachable"}


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

# --- Internal service auth (per-installation token) ---
# /v1/* keeps its api-keys.json Bearer scheme; the service token is
# injected into that dict below so one credential works everywhere.
# /internal/* (previously unauthenticated by design) is enforced by
# this middleware when a token is configured. /health, /ready and /
# stay open (compose/K8s probes are headerless).
from geometric_lens.auth_token import (SERVICE_TOKEN as _SERVICE_TOKEN,
                                       install_urllib_opener as
                                       _install_urllib_opener)
import hmac as _hmac

_install_urllib_opener()  # outbound: embedding extractor, identity probe

if _SERVICE_TOKEN:
    api_keys[_SERVICE_TOKEN] = {"user": "atlas-internal"}

@app.middleware("http")
async def _require_service_token(request, call_next):
    # Enforce only on /internal/* — /v1/* has its own Bearer check
    # (verify_api_key Depends, which now also accepts the service
    # token), and /health, /ready, / stay open for probes.
    if _SERVICE_TOKEN and request.url.path.startswith("/internal/"):
        got = request.headers.get("authorization", "")
        if not _hmac.compare_digest(got, f"Bearer {_SERVICE_TOKEN}"):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={
                "error": "unauthorized",
                "detail": "internal service auth is enabled; send "
                          "Authorization: Bearer <service-token> "
                          "(secrets/service-token)"})
    return await call_next(request)


# Registered AFTER the token middleware: Starlette wraps in reverse
# registration order (last = outermost), and the correlation ID must be
# set/echoed even on requests the auth middleware rejects with 401.
@app.middleware("http")
async def _correlation_id(request, call_next):
    # Adopt the caller's correlation ID (or none); echo it back so the
    # whole turn shares one id across services.
    rid = request.headers.get("x-atlas-request-id", "")
    _set_rid(rid)
    response = await call_next(request)
    if rid:
        response.headers["X-ATLAS-Request-ID"] = rid
    return response


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
# Note: probe/scoring endpoints below are deliberately plain `def` — they do
# synchronous work (sqlite query, httpx sync client, urlopen to llama-server,
# torch), so FastAPI runs them in its threadpool instead of blocking the
# event loop.
@app.get("/health")
def health():
    """Structured per-subsystem health.

    Always returns 200 — this endpoint is for *information*, not gating.
    Use /ready for liveness/scoring-functional gating.
    """
    db_st = _db_state()
    llama_st = _llama_state()
    lens_ok = (
        not _BOOT_STATE["lens_enabled"] or _BOOT_STATE["self_test_pass"]
    )
    overall = (
        db_st["connected"]
        and llama_st["reachable"]
        and lens_ok
    )
    return {
        "service": "geometric-lens",
        "status": "healthy" if overall else "degraded",
        "subsystems": {
            "sqlite": db_st,
            "llama_server": llama_st,
            "lens": {
                "enabled": _BOOT_STATE["lens_enabled"],
                "cost_field_loaded": _BOOT_STATE["lens_cost_field_loaded"],
                "cost_field_dim": _BOOT_STATE["lens_cost_field_dim"],
                "embed_dim": _BOOT_STATE["embed_dim"],
                "gx_loaded": _BOOT_STATE["lens_gx_loaded"],
                "gx_type": _BOOT_STATE["lens_gx_type"],
                "cx_calibrated": _BOOT_STATE["lens_cx_calibrated"],
                "gx_calibrated": _BOOT_STATE["lens_gx_calibrated"],
                "artifact_model": _BOOT_STATE["lens_artifact_model"],
                "self_test_pass": _BOOT_STATE["self_test_pass"],
                "self_test_error": _BOOT_STATE["self_test_error"],
                "fingerprint_present": _BOOT_STATE["fingerprint_present"],
                "fingerprint_ok": _BOOT_STATE["fingerprint_ok"],
                "fingerprint_error": _BOOT_STATE["fingerprint_error"],
            },
        },
    }


@app.get("/ready")
def ready():
    """Readiness gate. 200 only when scoring is functional, 503 otherwise.

    Use this for orchestrator probes that should pull traffic away when
    lens scoring degrades (the silent-failure mode PC-019 was filed for).
    """
    db_st = _db_state()
    llama_st = _llama_state()
    lens_required = _BOOT_STATE["lens_enabled"]
    lens_ok = (not lens_required) or _BOOT_STATE["self_test_pass"]

    ok = db_st["connected"] and llama_st["reachable"] and lens_ok
    payload = {
        "ready": ok,
        "sqlite": db_st["connected"],
        "llama_server": llama_st["reachable"],
        "lens_self_test": _BOOT_STATE["self_test_pass"],
        "lens_required": lens_required,
        "fingerprint_ok": _BOOT_STATE["fingerprint_ok"],
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
        raise HTTPException(status_code=500, detail=_safe_detail(e, "PageIndex build"))

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
    """Log request metrics to SQLite for dashboard."""
    try:
        from datetime import date
        today = date.today().isoformat()

        pool = get_db_pool()
        with pool.get_connection() as conn:
            # Increment daily counters
            counters = [("tasks_total", 1), ("tokens_total", tokens)]
            if success:
                counters.append(("tasks_success", 1))
            for key, delta in counters:
                conn.execute("""
                    INSERT INTO metrics_daily (date, key, value) VALUES (?, ?, ?)
                    ON CONFLICT(date, key) DO UPDATE SET value = value + excluded.value
                """, (today, key, delta))

            # Add to recent tasks list
            task_record = json.dumps({
                "type": request_type,
                "model": model,
                "tokens": tokens,
                "success": success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            conn.execute(
                "INSERT INTO metrics_recent_tasks (task_record) VALUES (?)",
                (task_record,)
            )
            # Keep the newest 100 records
            conn.execute("""
                DELETE FROM metrics_recent_tasks
                WHERE id NOT IN (
                    SELECT id FROM metrics_recent_tasks ORDER BY id DESC LIMIT 100
                )
            """)
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
    async with httpx.AsyncClient(headers=_svc_auth_headers()) as client:
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

    # Store + enqueue in one insert: a pending row in `tasks` is the queue
    # entry (workers pull by status + priority, FIFO within a priority via
    # created_at/rowid). A store failure returns a clean 503 instead of a
    # raw database error.
    try:
        pool = get_db_pool()
        with pool.get_connection() as conn:
            conn.execute(
                "INSERT INTO tasks (id, priority, status, data) VALUES (?, ?, ?, ?)",
                (task_id, request.priority, "pending", json.dumps(task_data))
            )
    except Exception as e:
        logger.warning("task queue unavailable: %s", _safe_log(e))
        raise HTTPException(status_code=503,
                            detail="Task queue not available")

    return TaskSubmitResponse(task_id=task_id, status="pending")

@app.get("/v1/tasks/{task_id}/status")
async def get_task_status(
    task_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get current status of a submitted task."""
    try:
        pool = get_db_pool()
        with pool.get_connection() as conn:
            cur = conn.execute("SELECT data FROM tasks WHERE id = ?", (task_id,))
            row = cur.fetchone()
    except Exception as e:
        logger.warning("task queue unavailable: %s", _safe_log(e))
        raise HTTPException(status_code=503, detail="Task queue not available")

    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    task = json.loads(row["data"])
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
    try:
        pool = get_db_pool()
        with pool.get_connection() as conn:
            cur = conn.execute(
                "SELECT priority, COUNT(*) AS count FROM tasks "
                "WHERE status = 'pending' GROUP BY priority"
            )
            counts = {row["priority"]: row["count"] for row in cur.fetchall()}
    except Exception as e:
        logger.warning("task queue unavailable: %s", _safe_log(e))
        raise HTTPException(status_code=503, detail="Task queue not available")

    return {
        "p0_waiting": counts.get("p0", 0),
        "p1_waiting": counts.get("p1", 0),
        "p2_waiting": counts.get("p2", 0),
        "total_waiting": sum(counts.values())
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


# Strong references to in-flight pattern-write tasks. asyncio only keeps a
# weak reference to tasks, so without this a pattern write could be
# garbage-collected mid-flight. Tasks discard themselves on completion.
_pattern_write_tasks: set = set()


def _spawn_pattern_task(coro) -> None:
    """create_task with a strong reference held until the task completes."""
    import asyncio

    task = asyncio.create_task(coro)
    _pattern_write_tasks.add(task)
    task.add_done_callback(_pattern_write_tasks.discard)


def _dispatch_pattern_write(request: PatternWriteRequest) -> dict:
    """Schedule pattern-write + outcome recording. Shared by /v1 and /internal handlers."""
    if not request.success:
        if request.active_pattern_ids:
            _spawn_pattern_task(
                record_pattern_outcome(request.active_pattern_ids, success=False)
            )
        return {"status": "recorded_failure"}

    _spawn_pattern_task(
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
        _spawn_pattern_task(
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
        from router.route_selector import get_all_thompson_states
        from router.feedback_recorder import get_routing_stats

        thompson = get_all_thompson_states()
        raw_stats = get_routing_stats()

        # get_routing_stats() returns {"error": str(e)} on failure (it logs
        # the full detail itself). Rebuild the payload from known keys so
        # raw exception text never reaches the HTTP response.
        if "error" in raw_stats:
            stats: Dict[str, Any] = {
                "error": "routing stats unavailable (see service log)",
            }
        else:
            stats = {
                "total_decisions": raw_stats.get("total_decisions"),
                "total_successes": raw_stats.get("total_successes"),
                "success_rate": raw_stats.get("success_rate"),
                "route_distribution": raw_stats.get("route_distribution"),
                "difficulty_distribution": raw_stats.get("difficulty_distribution"),
            }

        return {
            "enabled": True,
            "thompson_state": thompson,
            "aggregate_stats": stats,
        }
    except Exception as e:
        return {"enabled": True, "error": _safe_detail(e, "router stats")}


@app.post("/internal/router/reset")
async def router_reset():
    """Reset Thompson Sampling state for recalibration."""
    if not is_routing_enabled():
        return {"status": "skipped", "message": "Routing is disabled"}

    try:
        from router.route_selector import reset_thompson_state
        from router.feedback_recorder import reset_stats

        reset_thompson_state()
        reset_stats()

        return {"status": "reset", "message": "Thompson state and stats reset to uniform priors"}
    except Exception as e:
        return {"status": "error", "error": _safe_detail(e, "router reset")}


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
        return {"status": "error", "error": _safe_detail(e, "router feedback")}


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
        return {
            "loaded": False,
            "enabled": False,
            "error": _safe_detail(e, "lens stats"),
        }


class LensEvaluateBody(BaseModel):
    query: Optional[str] = None
    text: Optional[str] = None


@app.api_route("/internal/lens/evaluate", methods=["GET", "POST"])
def lens_evaluate(request: Request, query: str = None,
                  body: Optional[LensEvaluateBody] = None):
    """Evaluate a query through the Geometric Lens (for testing).

    Accepts GET with ?query= param or POST with JSON {"query": "..."}.
    Malformed POST bodies fail pydantic validation and return a structured
    422 instead of an unhandled 500.
    """
    if request.method == "POST" and body is not None:
        query = body.query if body.query is not None else (body.text or "")
    if not query:
        raise HTTPException(status_code=422, detail="Missing 'query' parameter")
    try:
        from geometric_lens.service import evaluate_energy, is_enabled
        if not is_enabled():
            return {"enabled": False, "message": "Geometric Lens disabled"}

        raw_energy, normalized = evaluate_energy(query)

        return _apply_drift_flags({
            "enabled": True,
            "energy": raw_energy,
            "energy_normalized": normalized,
        })
    except Exception as e:
        return {"error": _safe_detail(e, "lens evaluate")}


class LensScoreTextRequest(BaseModel):
    text: str


class LensScorePerStepRequest(BaseModel):
    text: str
    # Optional transformer-block index. None => last-layer (vanilla /embedding,
    # no PC-202 patch needed). Set to use the PC-202 layers extension and score
    # at the residual stream of a specific intermediate layer (PC-204 fusion).
    layer: Optional[int] = None


@app.post("/internal/lens/score-text")
def lens_score_text(request: LensScoreTextRequest):
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

        normalized = lens_service._normalize_cx_energy(energy)

        return _apply_drift_flags({
            "energy": energy,
            "normalized": normalized,
            "calibrated": lens_service._cx_normalization is not None,
            "enabled": True,
        })
    except Exception as e:
        return {
            "energy": 0.0,
            "normalized": 0.5,
            "error": _safe_detail(e, "lens score-text"),
        }


class LensRetrainRequest(BaseModel):
    training_data: List[Dict]
    epochs: int = 50
    domain: str = "LCB"
    use_replay: bool = True
    use_ewc: bool = True
    lambda_ewc: float = 1000.0


def _models_dir_writable(models_dir: str) -> bool:
    """Probe whether the models dir accepts writes.

    docker-compose mounts the models dir read-only (:ro); os.access alone
    can misreport on such mounts, so back it with a tempfile probe.
    """
    if not os.access(models_dir, os.W_OK):
        return False
    try:
        fd, probe = tempfile.mkstemp(dir=models_dir, prefix=".write_probe_")
        os.close(fd)
        os.remove(probe)
        return True
    except OSError:
        return False


@app.post("/internal/lens/retrain")
def lens_retrain(request: LensRetrainRequest):
    """Retrain C(x) on accumulated pass/fail embeddings from benchmark execution."""
    models_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "geometric_lens", "models"
    )
    # Fail before burning a training run: in the standard compose deployment
    # the models dir is mounted read-only into this container.
    if not _models_dir_writable(models_dir):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "reason": ("models directory is mounted read-only; "
                           "run host-side retrain via `atlas lens retrain`"),
            },
        )

    with _lens_weights_lock:
        try:
            from geometric_lens.training import retrain_cost_field_bce
            from geometric_lens.service import reload_weights

            embeddings = [d["embedding"] for d in request.training_data]
            labels = [d["label"] for d in request.training_data]

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

            if not metrics.get("skipped", False):
                from geometric_lens.calibration import (
                    derive_cx_normalization, save_cx_normalization,
                )
                calibration = derive_cx_normalization(
                    metrics["pass_energy_mean"], metrics["fail_energy_mean"])
                save_cx_normalization(models_dir, calibration)

                # The load path hard-requires model_identity.json (the
                # cross-model artifact guard). A retrain produces a new
                # bundle for the model llama-server is serving RIGHT NOW,
                # so stamp/refresh the identity here — without this, a
                # retrained bundle fails the identity check on the next
                # container restart and the whole lens stays disabled.
                from geometric_lens.identity import save_model_identity
                from geometric_lens.service import _probe_served_model
                served = _probe_served_model() or os.environ.get(
                    "ATLAS_MODEL_NAME", "").strip()
                if served and embeddings:
                    # Record the embedding convention the training data was
                    # extracted under — the caller embedded via this same
                    # server, so the live convention IS the trained one.
                    from geometric_lens.embedding_extractor import (
                        observe_embedding_convention,
                    )
                    try:
                        contract = observe_embedding_convention()
                    except Exception as exc:
                        logger.warning(
                            "retrain: embedding-convention probe failed "
                            "(%s) — identity written without a contract",
                            exc)
                        contract = None
                    save_model_identity(models_dir, served,
                                        len(embeddings[0]),
                                        embedding_contract=contract)
                    metrics["model_identity"] = served
                else:
                    logger.warning(
                        "retrain: could not resolve the served model — "
                        "model_identity.json not written; the reloaded "
                        "bundle will fail the identity check on restart")

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

                # Refresh the boot-state cache so /ready reflects the
                # freshly-retrained weights instead of the boot snapshot.
                if reload_result.get("status") == "reloaded":
                    # Write the fingerprint BEFORE the self-test re-runs:
                    # the retrain moved the energies, so the previous
                    # fingerprint would (correctly) flag the new weights
                    # as drifted and wedge /ready.
                    try:
                        import geometric_lens.service as _svc
                        from geometric_lens.drift import write_fingerprint
                        write_fingerprint(
                            models_dir,
                            lambda t: _svc.evaluate_energy(t)[0],
                            note=f"/internal/lens/retrain for "
                                 f"{served or 'unknown model'}")
                        metrics["fingerprint_written"] = True
                    except Exception as exc:
                        logger.warning(
                            "drift fingerprint write failed after "
                            "retrain: %s", exc)
                        metrics["fingerprint_written"] = False
                    _run_lens_self_test()

            return {"status": "ok", "metrics": metrics}
        except Exception as e:
            return {"status": "error", "error": _safe_detail(e, "lens retrain")}


@app.post("/internal/lens/reload")
def lens_reload():
    """Reload Geometric Lens weights from disk after retraining."""
    try:
        from geometric_lens.service import reload_weights, get_model_info
        with _lens_weights_lock:
            result = reload_weights()
            # Refresh the boot-state cache so /ready reflects the new state.
            if result.get("status") == "reloaded":
                _run_lens_self_test()
                # Report the authoritative post-self-test state, not the
                # transient summary from reload_weights() — that summary's
                # gx_loaded/cx fields are captured mid-swap and can read
                # stale, which the host-side retrain script prints verbatim
                # and misled operators into thinking G(x) failed to load.
                info = get_model_info()
                return {
                    "status": "reloaded",
                    "gx_loaded": bool(info.get("gx_loaded")),
                    "cx_calibrated": bool(info.get("cx_calibrated")),
                    "gx_calibrated": bool(info.get("gx_calibrated")),
                    "artifact_model": info.get("artifact_model"),
                }
        return {"status": result.get("status", "unknown"), **result}
    except Exception as e:
        return {"status": "error", "error": _safe_detail(e, "lens reload")}


@app.post("/internal/lens/gx-score")
def lens_gx_score(request: LensScoreTextRequest):
    """Combined C(x) + G(x) scoring in a single call.

    Returns C(x) energy, normalized energy, G(x) XGBoost quality prediction,
    and a human-readable verdict. Uses one embedding extraction for both models.
    """
    try:
        from geometric_lens.service import evaluate_combined, is_enabled

        if not is_enabled():
            return {
                "cx_energy": 0.0, "cx_normalized": 0.5,
                "cx_calibrated": False,
                "gx_score": 0.5, "verdict": "unavailable",
                "enabled": False, "gx_available": False,
            }

        result = evaluate_combined(request.text)
        if isinstance(result, dict):
            result = _apply_drift_flags(result)
        return result
    except Exception as e:
        return {
            "cx_energy": 0.0, "cx_normalized": 0.5,
            "cx_calibrated": False,
            "gx_score": 0.5, "verdict": "error",
            "error": _safe_detail(e, "lens gx-score"),
        }


@app.post("/internal/lens/score-per-step")
def lens_score_per_step(request: LensScorePerStepRequest):
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

        result = evaluate_per_step(request.text, layer=request.layer)
        agg = result.get("aggregate") or {}
        # _safe_log on the request.layer value strips CRLF + truncates
        # so user input can't fake a separate log entry. The other args
        # are floats/ints from result — structurally safe.
        logger.info(
            "lens score-per-step: in_chars=%d n_tok=%d gx_min=%.3f gx_mean=%.3f off_rails=%d layer=%s lat=%.0fms",
            len(request.text or ""),
            int(result.get("n_tokens", 0)),
            float(agg.get("gx_score_min", 0.0)),
            float(agg.get("gx_score_mean", 0.0)),
            int(agg.get("first_off_rails_idx", -1)),
            _safe_log(request.layer) if request.layer is not None else "last",
            float(result.get("latency_ms", 0.0)),
        )
        return result
    except Exception as e:
        return {
            "enabled": True, "gx_available": False,
            "per_step": [], "aggregate": {}, "n_tokens": 0,
            "error": _safe_detail(e, "lens score-per-step"),
        }


# ---------------------------------------------------------------------------
# Sandbox Analysis Endpoint
# ---------------------------------------------------------------------------

class SandboxAnalyzeRequest(BaseModel):
    code: str
    stdout: str = ""
    stderr: str = ""
    passed: bool = False
    expected_output: Optional[str] = None
    include_gx: bool = True


@app.post("/internal/sandbox/analyze")
def sandbox_analyze(request: SandboxAnalyzeRequest):
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
        return {
            "error": _safe_detail(e, "sandbox analysis"),
            "passed": False,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port
    )
