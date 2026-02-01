import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import aiosqlite
from databases import Database

from .config import settings

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Async database for queries
database = Database(settings.database_url)

# SQLAlchemy setup for models
engine = create_engine(settings.database_url.replace("sqlite:///", "sqlite:///"), connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key_hash = Column(String(255), nullable=False, index=True)  # Store hash of the key
    key_prefix = Column(String(20), nullable=False)  # Store prefix for display (e.g., sk-llm-abc...)
    name = Column(String(100), nullable=False)  # User-friendly name
    rate_limit = Column(Integer, default=100)  # Requests per minute
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="api_keys")


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, index=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False)
    endpoint = Column(String(255), nullable=False)
    tokens_input = Column(Integer, default=0)
    tokens_output = Column(Integer, default=0)
    model = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class LLMModel(Base):
    """Cached model information from the LLM server"""
    __tablename__ = "llm_models"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(100), unique=True, nullable=False)  # e.g., "qwen3-14b"
    name = Column(String(200), nullable=False)  # Display name
    context_length = Column(Integer, default=8192)
    max_output = Column(Integer, default=4096)
    is_active = Column(Boolean, default=True)
    is_auto_discovered = Column(Boolean, default=True)  # False if admin override
    source_server = Column(String(255), nullable=True)  # Where it was discovered from
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ServerConfig(Base):
    """LLM Server configuration"""
    __tablename__ = "server_config"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
