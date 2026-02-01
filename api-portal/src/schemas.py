from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# User schemas
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    is_active: bool
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# API Key schemas
class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    rate_limit: Optional[int] = Field(default=100, ge=1, le=10000)
    expires_days: Optional[int] = Field(default=None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    id: int
    name: str
    key_prefix: str
    rate_limit: int
    is_active: bool
    last_used_at: Optional[datetime]
    created_at: datetime
    expires_at: Optional[datetime]

    class Config:
        from_attributes = True


class APIKeyCreated(BaseModel):
    """Response when creating a new API key - includes the full key (only shown once)"""
    id: int
    name: str
    key: str  # Full API key - only returned on creation
    rate_limit: int
    created_at: datetime
    expires_at: Optional[datetime]


class APIKeyList(BaseModel):
    keys: List[APIKeyResponse]
    total: int


# Usage schemas
class UsageStats(BaseModel):
    total_requests: int
    total_tokens_input: int
    total_tokens_output: int
    requests_today: int
    tokens_today: int


# General response schemas
class MessageResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    detail: str


# LLM Model schemas (OpenAI-compatible format)
class LLMModelInfo(BaseModel):
    """Model info in OpenAI-compatible format"""
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "organization"
    # Extended info
    name: Optional[str] = None
    context_length: Optional[int] = None
    max_output: Optional[int] = None

    class Config:
        from_attributes = True


class LLMModelList(BaseModel):
    """OpenAI-compatible model list response"""
    object: str = "list"
    data: List[LLMModelInfo]


class LLMModelCreate(BaseModel):
    """Admin schema to manually add/override a model"""
    model_id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    context_length: int = Field(default=8192, ge=1)
    max_output: int = Field(default=4096, ge=1)
    is_active: bool = True


class LLMModelResponse(BaseModel):
    """Full model info for admin"""
    id: int
    model_id: str
    name: str
    context_length: int
    max_output: int
    is_active: bool
    is_auto_discovered: bool
    source_server: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LLMModelListAdmin(BaseModel):
    """Admin model list response"""
    models: List[LLMModelResponse]
    total: int
    server_url: Optional[str] = None
