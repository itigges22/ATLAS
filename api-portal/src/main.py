import os
import httpx
import asyncio
import redis
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session
from sqlalchemy import func

from .config import settings
from .database import init_db, get_db, User, APIKey, UsageLog, LLMModel, ServerConfig
from .auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_user, generate_api_key, hash_api_key
)
from .schemas import (
    UserCreate, UserLogin, UserResponse, TokenResponse,
    APIKeyCreate, APIKeyResponse, APIKeyCreated, APIKeyList,
    UsageStats, MessageResponse,
    LLMModelInfo, LLMModelList, LLMModelCreate, LLMModelResponse, LLMModelListAdmin
)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="API Key Management Portal for Self-Hosted LLM",
    version="1.0.0"
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert Pydantic validation errors from 422 to 400."""
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors(), "body": exc.body}
    )


# Redis client for metrics
try:
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.ping()
except Exception:
    redis_client = None
    print("Warning: Redis not available for metrics")

# Mount static files and templates
os.makedirs("public", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="public"), name="static")
templates = Jinja2Templates(directory="templates")


async def discover_models_from_server(db: Session, server_url: str = None) -> dict:
    """
    Discover models from the LLM server (llama.cpp, vLLM, etc.)
    Returns dict with success status and discovered models
    """
    url = server_url or settings.llm_api_url
    result = {"success": False, "models": [], "error": None}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # First try to get the actual context size from /props endpoint
            actual_context = None
            try:
                props_response = await client.get(f"{url}/props")
                if props_response.status_code == 200:
                    props_data = props_response.json()
                    gen_settings = props_data.get("default_generation_settings", {})
                    actual_context = gen_settings.get("n_ctx")
            except Exception:
                pass  # /props may not be available on all servers

            response = await client.get(f"{url}/v1/models")

            if response.status_code == 200:
                data = response.json()
                models_data = data.get("data", [])

                for model_info in models_data:
                    model_id = model_info.get("id", "unknown")

                    # Determine context length: prefer actual server context, then model meta, then default
                    meta = model_info.get("meta", {})
                    context_length = actual_context or meta.get("n_ctx_train") or model_info.get("context_length") or 8192

                    # Check if model already exists
                    existing = db.query(LLMModel).filter(LLMModel.model_id == model_id).first()

                    if existing:
                        # Update if auto-discovered (don't override admin settings)
                        if existing.is_auto_discovered:
                            existing.context_length = context_length
                            existing.max_output = min(context_length // 2, 4096)
                            existing.source_server = url
                            existing.updated_at = datetime.utcnow()
                    else:
                        # Create new model entry
                        # Generate display name from model_id
                        # Remove common file extensions and clean up the name
                        display_name = model_id
                        for ext in ['.gguf', '.bin', '.safetensors', '.pt', '.onnx']:
                            display_name = display_name.replace(ext, '')
                        # Clean up quantization suffixes for display
                        display_name = display_name.replace('.Q', ' Q').replace('_K', 'K')
                        display_name = display_name.replace("-", " ").replace("_", " ")
                        # Capitalize properly
                        display_name = display_name.strip().title()

                        new_model = LLMModel(
                            model_id=model_id,
                            name=display_name,
                            context_length=context_length,
                            max_output=min(context_length // 2, 4096),
                            is_auto_discovered=True,
                            source_server=url
                        )
                        db.add(new_model)

                    result["models"].append(model_id)

                db.commit()
                result["success"] = True

                # Store the server URL in config
                server_config = db.query(ServerConfig).filter(ServerConfig.key == "llm_server_url").first()
                if server_config:
                    server_config.value = url
                else:
                    db.add(ServerConfig(key="llm_server_url", value=url))
                db.commit()

    except httpx.RequestError as e:
        result["error"] = f"Failed to connect to LLM server: {str(e)}"
    except Exception as e:
        result["error"] = f"Error discovering models: {str(e)}"

    return result


@app.on_event("startup")
async def startup():
    """Initialize database on startup and discover models"""
    init_db()

    # Try to discover models from LLM server
    db = next(get_db())
    try:
        result = await discover_models_from_server(db)
        if result["success"]:
            print(f"Auto-discovered models: {result['models']}")
        else:
            print(f"Model discovery failed: {result.get('error', 'Unknown error')}")
    finally:
        db.close()


# ============ Web UI Routes ============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Landing page"""
    return templates.TemplateResponse("index.html", {"request": request, "settings": settings})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    return templates.TemplateResponse("login.html", {"request": request, "settings": settings})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Registration page"""
    return templates.TemplateResponse("register.html", {"request": request, "settings": settings})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Dashboard page (requires JS-side auth)"""
    return templates.TemplateResponse("dashboard.html", {"request": request, "settings": settings})


# ============ Auth API Routes ============

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if email exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Check if username exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    # First user becomes admin
    is_first_user = db.query(User).count() == 0

    # Create user
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password),
        is_admin=is_first_user
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Generate token
    access_token = create_access_token(data={"sub": user.id})

    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user)
    )


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login and get access token"""
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == credentials.username) | (User.email == credentials.username)
    ).first()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Account is disabled")

    # Generate token
    access_token = create_access_token(data={"sub": user.id})

    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user)
    )


@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return UserResponse.model_validate(current_user)


# ============ API Key Routes ============

@app.post("/api/keys", response_model=APIKeyCreated)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new API key"""
    # Generate the key
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)

    # Calculate expiration
    expires_at = None
    if key_data.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=key_data.expires_days)

    # Create API key record
    api_key = APIKey(
        user_id=current_user.id,
        key_hash=key_hash,
        key_prefix=raw_key[:16] + "...",
        name=key_data.name,
        rate_limit=key_data.rate_limit,
        expires_at=expires_at
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return APIKeyCreated(
        id=api_key.id,
        name=api_key.name,
        key=raw_key,  # Only returned on creation!
        rate_limit=api_key.rate_limit,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at
    )


@app.get("/api/keys", response_model=APIKeyList)
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all API keys for current user"""
    keys = db.query(APIKey).filter(APIKey.user_id == current_user.id).all()
    return APIKeyList(
        keys=[APIKeyResponse.model_validate(k) for k in keys],
        total=len(keys)
    )


@app.delete("/api/keys/{key_id}", response_model=MessageResponse)
async def delete_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an API key"""
    api_key = db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.user_id == current_user.id
    ).first()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    db.delete(api_key)
    db.commit()

    return MessageResponse(message="API key deleted successfully")


@app.patch("/api/keys/{key_id}/toggle", response_model=APIKeyResponse)
async def toggle_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggle API key active status"""
    api_key = db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.user_id == current_user.id
    ).first()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key.is_active = not api_key.is_active
    db.commit()
    db.refresh(api_key)

    return APIKeyResponse.model_validate(api_key)


# ============ Usage Stats ============

@app.get("/api/usage", response_model=UsageStats)
async def get_usage_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get usage statistics from Redis metrics"""
    # Read from Redis if available
    if redis_client:
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            daily_metrics = redis_client.hgetall(f"metrics:daily:{today}")

            total_requests = int(daily_metrics.get("tasks_total", 0))
            total_tokens = int(daily_metrics.get("total_tokens", 0))

            return UsageStats(
                total_requests=total_requests,
                total_tokens_input=total_tokens // 2,  # Approximate split
                total_tokens_output=total_tokens // 2,
                requests_today=total_requests,
                tokens_today=total_tokens
            )
        except Exception as e:
            print(f"Redis read error: {e}")

    # Fallback to database
    key_ids = [k.id for k in db.query(APIKey).filter(APIKey.user_id == current_user.id).all()]

    if not key_ids:
        return UsageStats(
            total_requests=0,
            total_tokens_input=0,
            total_tokens_output=0,
            requests_today=0,
            tokens_today=0
        )

    total_stats = db.query(
        func.count(UsageLog.id).label("total_requests"),
        func.sum(UsageLog.tokens_input).label("total_tokens_input"),
        func.sum(UsageLog.tokens_output).label("total_tokens_output")
    ).filter(UsageLog.api_key_id.in_(key_ids)).first()

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_stats = db.query(
        func.count(UsageLog.id).label("requests_today"),
        func.sum(UsageLog.tokens_input + UsageLog.tokens_output).label("tokens_today")
    ).filter(
        UsageLog.api_key_id.in_(key_ids),
        UsageLog.created_at >= today_start
    ).first()

    return UsageStats(
        total_requests=total_stats.total_requests or 0,
        total_tokens_input=total_stats.total_tokens_input or 0,
        total_tokens_output=total_stats.total_tokens_output or 0,
        requests_today=today_stats.requests_today or 0,
        tokens_today=today_stats.tokens_today or 0
    )


# ============ API Key Validation Endpoint (for RAG API) ============

@app.post("/api/validate-key")
async def validate_key(request: Request, db: Session = Depends(get_db)):
    """Validate an API key - used by the RAG API"""
    body = await request.json()
    api_key = body.get("api_key")

    if not api_key:
        return JSONResponse(
            status_code=400,
            content={"valid": False, "error": "No API key provided"}
        )

    key_hash = hash_api_key(api_key)
    api_key_record = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == True
    ).first()

    if not api_key_record:
        return JSONResponse(
            status_code=401,
            content={"valid": False, "error": "Invalid API key"}
        )

    # Check expiration
    if api_key_record.expires_at and api_key_record.expires_at < datetime.utcnow():
        return JSONResponse(
            status_code=401,
            content={"valid": False, "error": "API key expired"}
        )

    # Update last used
    api_key_record.last_used_at = datetime.utcnow()
    db.commit()

    # Get user info
    user = db.query(User).filter(User.id == api_key_record.user_id).first()

    return JSONResponse(content={
        "valid": True,
        "user": user.username,
        "rate_limit": api_key_record.rate_limit,
        "key_name": api_key_record.name
    })


# ============ OpenAI-Compatible Model Endpoint ============

@app.get("/v1/models", response_model=LLMModelList)
async def list_models(db: Session = Depends(get_db)):
    """
    OpenAI-compatible endpoint to list available models.
    Used by OpenCode to discover available models.
    """
    models = db.query(LLMModel).filter(LLMModel.is_active == True).all()

    return LLMModelList(
        data=[
            LLMModelInfo(
                id=m.model_id,
                name=m.name,
                context_length=m.context_length,
                max_output=m.max_output,
                created=int(m.created_at.timestamp()) if m.created_at else 0
            )
            for m in models
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get info for a specific model"""
    model = db.query(LLMModel).filter(
        LLMModel.model_id == model_id,
        LLMModel.is_active == True
    ).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return LLMModelInfo(
        id=model.model_id,
        name=model.name,
        context_length=model.context_length,
        max_output=model.max_output,
        created=int(model.created_at.timestamp()) if model.created_at else 0
    )


# ============ Admin Model Management ============

def require_admin(current_user: User = Depends(get_current_user)):
    """Dependency to require admin privileges"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@app.get("/api/admin/models", response_model=LLMModelListAdmin)
async def admin_list_models(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin: List all models with full details"""
    models = db.query(LLMModel).all()
    server_config = db.query(ServerConfig).filter(ServerConfig.key == "llm_server_url").first()

    return LLMModelListAdmin(
        models=[LLMModelResponse.model_validate(m) for m in models],
        total=len(models),
        server_url=server_config.value if server_config else settings.llm_api_url
    )


@app.post("/api/admin/models", response_model=LLMModelResponse)
async def admin_create_model(
    model_data: LLMModelCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin: Create or override a model"""
    existing = db.query(LLMModel).filter(LLMModel.model_id == model_data.model_id).first()

    if existing:
        # Update existing model (override auto-discovered settings)
        existing.name = model_data.name
        existing.context_length = model_data.context_length
        existing.max_output = model_data.max_output
        existing.is_active = model_data.is_active
        existing.is_auto_discovered = False  # Mark as admin-managed
        db.commit()
        db.refresh(existing)
        return LLMModelResponse.model_validate(existing)
    else:
        # Create new model
        new_model = LLMModel(
            model_id=model_data.model_id,
            name=model_data.name,
            context_length=model_data.context_length,
            max_output=model_data.max_output,
            is_active=model_data.is_active,
            is_auto_discovered=False
        )
        db.add(new_model)
        db.commit()
        db.refresh(new_model)
        return LLMModelResponse.model_validate(new_model)


@app.patch("/api/admin/models/{model_id}", response_model=LLMModelResponse)
async def admin_update_model(
    model_id: str,
    model_data: LLMModelCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin: Update a model"""
    model = db.query(LLMModel).filter(LLMModel.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    model.name = model_data.name
    model.context_length = model_data.context_length
    model.max_output = model_data.max_output
    model.is_active = model_data.is_active
    model.is_auto_discovered = False
    db.commit()
    db.refresh(model)

    return LLMModelResponse.model_validate(model)


@app.delete("/api/admin/models/{model_id}", response_model=MessageResponse)
async def admin_delete_model(
    model_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin: Delete a model"""
    model = db.query(LLMModel).filter(LLMModel.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    db.delete(model)
    db.commit()

    return MessageResponse(message=f"Model {model_id} deleted")


@app.post("/api/admin/models/discover", response_model=MessageResponse)
async def admin_discover_models(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
    server_url: Optional[str] = None
):
    """Admin: Trigger model discovery from LLM server"""
    result = await discover_models_from_server(db, server_url)

    if result["success"]:
        return MessageResponse(message=f"Discovered models: {', '.join(result['models'])}")
    else:
        raise HTTPException(status_code=500, detail=result.get("error", "Discovery failed"))


# ============ Health Check ============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "api-portal"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
