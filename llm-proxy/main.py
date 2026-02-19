"""
LLM Proxy - Forwards requests to llama-server and logs metrics to Redis.
Validates API keys against the API Portal.
"""

import os
import json
import time
import httpx
import redis
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Response, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Optional, Tuple

LLAMA_URL = os.getenv("LLAMA_URL", "http://llama-service:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
API_PORTAL_URL = os.getenv("API_PORTAL_URL", "http://api-portal:3000")

# API key validation cache
_key_cache = {}
_key_cache_ttl = 60  # seconds

# Redis client
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
except Exception:
    redis_client = None
    print("Warning: Redis not available, metrics will not be logged")


async def validate_api_key(api_key: str) -> Optional[dict]:
    """Validate API key against the API Portal."""
    # Check cache first
    cached = _key_cache.get(api_key)
    if cached and time.time() - cached["timestamp"] < _key_cache_ttl:
        return cached["data"]

    # Validate with portal
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{API_PORTAL_URL}/api/validate-key",
                json={"api_key": api_key}
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    _key_cache[api_key] = {
                        "timestamp": time.time(),
                        "data": data
                    }
                    return data
    except Exception as e:
        print(f"Key validation error: {e}")

    return None


def check_rate_limit(api_key: str, limit: int) -> Tuple[bool, int, int]:
    """Check and increment rate limit counter using sliding window.

    Args:
        api_key: The API key string (used to create unique Redis key)
        limit: Maximum requests allowed per window

    Returns:
        Tuple of (allowed, remaining, ttl_seconds)
    """
    if not redis_client:
        # If Redis unavailable, allow all requests
        return (True, limit - 1, 60)

    try:
        # Use a hash of the key to avoid storing the full key in Redis
        import hashlib
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        redis_key = f"ratelimit:{key_hash}:count"

        # Increment counter
        current = redis_client.incr(redis_key)

        # Set expiry on first request (60 second sliding window)
        if current == 1:
            redis_client.expire(redis_key, 60)

        # Get TTL for reset time
        ttl = redis_client.ttl(redis_key)
        if ttl < 0:
            ttl = 60

        remaining = max(0, limit - current)
        allowed = current <= limit

        return (allowed, remaining, ttl)
    except Exception as e:
        print(f"Rate limit check error: {e}")
        # On error, allow the request but log it
        return (True, limit - 1, 60)


async def require_api_key(authorization: str = Header(None)) -> dict:
    """Dependency to require valid API key. Returns validation data with rate limit info."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Extract key from "Bearer sk-xxx" format
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        key = parts[1]
    else:
        # Also accept raw key
        key = authorization

    validation = await validate_api_key(key)
    if not validation:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Return full validation data including rate limit
    return {"key": key, "validation": validation}


def get_rate_limit_headers(limit: int, remaining: int, reset_seconds: int) -> dict:
    """Generate rate limit headers with real values."""
    return {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(int(time.time()) + reset_seconds)
    }


def log_metrics(request_type: str, model: str, tokens: int, success: bool, duration_ms: int):
    """Log request metrics to Redis."""
    if not redis_client:
        return
    try:
        today = datetime.now(timezone.utc).date().isoformat()

        # Increment daily counters
        redis_client.hincrby(f"metrics:daily:{today}", "tasks_total", 1)
        redis_client.hincrby(f"metrics:daily:{today}", "requests_total", 1)  # Track requests
        if success:
            redis_client.hincrby(f"metrics:daily:{today}", "tasks_success", 1)
        else:
            redis_client.hincrby(f"metrics:daily:{today}", "tasks_failed", 1)
        redis_client.hincrby(f"metrics:daily:{today}", "total_tokens", tokens)
        redis_client.hincrby(f"metrics:daily:{today}", "tokens_total", tokens)  # Alternative key
        redis_client.hincrby(f"metrics:daily:{today}", "total_duration_ms", duration_ms)
        redis_client.hincrby(f"metrics:daily:{today}", "total_attempts", 1)

        # Add to recent tasks
        task_record = json.dumps({
            "task_id": f"chat-{datetime.now(timezone.utc).strftime('%H%M%S')}",
            "type": request_type,
            "model": model,
            "success": success,
            "attempts": 1,
            "duration_ms": duration_ms,
            "tokens": tokens,
            "completed_at": datetime.now(timezone.utc).isoformat()
        })
        redis_client.lpush("metrics:recent_tasks", task_record)
        redis_client.ltrim("metrics:recent_tasks", 0, 99)
    except Exception as e:
        print(f"Failed to log metrics: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"LLM Proxy starting - forwarding to {LLAMA_URL}")
    yield
    print("LLM Proxy shutting down")


app = FastAPI(title="LLM Proxy", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "healthy", "backend": LLAMA_URL}


@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    """Proxy models endpoint - requires valid API key."""
    auth_data = await require_api_key(authorization)
    validation = auth_data["validation"]
    api_key = auth_data["key"]

    # Check rate limit
    rate_limit = validation.get("rate_limit", 1000)
    allowed, remaining, reset = check_rate_limit(api_key, rate_limit)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
            headers={
                "X-RateLimit-Limit": str(rate_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + reset),
                "Retry-After": str(reset)
            }
        )

    rate_headers = get_rate_limit_headers(rate_limit, remaining, reset)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{LLAMA_URL}/v1/models")
        return Response(content=resp.content, media_type="application/json", headers=rate_headers)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str = Header(None)):
    """Proxy chat completions with metrics logging - requires valid API key."""
    auth_data = await require_api_key(authorization)
    validation = auth_data["validation"]
    api_key = auth_data["key"]

    # Check rate limit
    rate_limit = validation.get("rate_limit", 1000)
    allowed, remaining, reset = check_rate_limit(api_key, rate_limit)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
            headers={
                "X-RateLimit-Limit": str(rate_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + reset),
                "Retry-After": str(reset)
            }
        )

    rate_headers = get_rate_limit_headers(rate_limit, remaining, reset)

    start = time.time()

    # Validate JSON body
    try:
        body_bytes = await request.body()
        body = json.loads(body_bytes)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    # Validate required fields
    if "messages" not in body:
        raise HTTPException(status_code=400, detail="Missing required field: messages")

    if not isinstance(body.get("messages"), list):
        raise HTTPException(status_code=400, detail="messages must be an array")

    model = body.get("model", "unknown")
    stream = body.get("stream", False)

    if stream:
        # Streaming response - client must be created inside generator to stay open
        async def stream_with_metrics():
            tokens = 0
            client = httpx.AsyncClient(timeout=300.0)
            try:
                async with client.stream(
                    "POST",
                    f"{LLAMA_URL}/v1/chat/completions",
                    json=body,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                        # Count tokens from SSE data
                        if b'"completion_tokens"' in chunk:
                            try:
                                # Extract token count from chunk
                                text = chunk.decode()
                                for line in text.split('\n'):
                                    if line.startswith('data: ') and 'usage' in line:
                                        data = json.loads(line[6:])
                                        tokens = data.get('usage', {}).get('total_tokens', 0)
                            except Exception:
                                pass
                duration_ms = int((time.time() - start) * 1000)
                log_metrics("chat_stream", model, tokens, True, duration_ms)
            except Exception as e:
                duration_ms = int((time.time() - start) * 1000)
                log_metrics("chat_stream", model, 0, False, duration_ms)
                raise
            finally:
                await client.aclose()

        stream_headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        stream_headers.update(rate_headers)
        return StreamingResponse(
            stream_with_metrics(),
            media_type="text/event-stream",
            headers=stream_headers
        )
    else:
        # Non-streaming
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                resp = await client.post(
                    f"{LLAMA_URL}/v1/chat/completions",
                    json=body,
                    headers={"Content-Type": "application/json"}
                )
                duration_ms = int((time.time() - start) * 1000)

                result = resp.json()
                tokens = result.get("usage", {}).get("total_tokens", 0)
                log_metrics("chat_completion", model, tokens, True, duration_ms)

                return Response(content=resp.content, media_type="application/json", headers=rate_headers)
            except Exception as e:
                duration_ms = int((time.time() - start) * 1000)
                log_metrics("chat_completion", model, 0, False, duration_ms)
                raise


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_all(request: Request, path: str, authorization: str = Header(None)):
    """Catch-all proxy for other endpoints - requires valid API key."""
    auth_data = await require_api_key(authorization)
    validation = auth_data["validation"]
    api_key = auth_data["key"]

    # Check rate limit
    rate_limit = validation.get("rate_limit", 1000)
    allowed, remaining, reset = check_rate_limit(api_key, rate_limit)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
            headers={
                "X-RateLimit-Limit": str(rate_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + reset),
                "Retry-After": str(reset)
            }
        )

    rate_headers = get_rate_limit_headers(rate_limit, remaining, reset)
    async with httpx.AsyncClient(timeout=60.0) as client:
        url = f"{LLAMA_URL}/{path}"

        if request.method == "GET":
            resp = await client.get(url, params=request.query_params)
        else:
            body = await request.body()
            resp = await client.request(
                request.method,
                url,
                content=body,
                headers={"Content-Type": request.headers.get("Content-Type", "application/json")}
            )

        response_headers = dict(rate_headers)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
            headers=response_headers
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
