import os
import yaml
import json
from pydantic import BaseModel
from typing import Dict, Optional


class ServerConfig(BaseModel):
    port: int = 8001
    host: str = "0.0.0.0"


class LlamaConfig(BaseModel):
    base_url: str = "http://llama-service:8000"


class QdrantConfig(BaseModel):
    host: str = "qdrant"
    port: int = 6333


class EmbeddingConfig(BaseModel):
    base_url: str = "http://embedding-service:8080"


class LimitsConfig(BaseModel):
    max_files: int = 10000
    max_loc: int = 500000
    max_size_mb: int = 100
    project_ttl_hours: int = 24


class ChunkingConfig(BaseModel):
    max_chunk_lines: int = 100
    overlap_lines: int = 20


class RetrievalConfig(BaseModel):
    top_k: int = 20
    context_budget_tokens: int = 8000


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    llama: LlamaConfig = LlamaConfig()
    qdrant: QdrantConfig = QdrantConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    limits: LimitsConfig = LimitsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    api_portal_url: str = os.environ.get("API_PORTAL_URL", "http://api-portal:3000")


def load_config() -> Config:
    config_path = os.environ.get("CONFIG_PATH", "/app/config/config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f)
            return Config(**data)
    return Config()


def load_api_keys() -> Dict[str, dict]:
    keys_path = os.environ.get("API_KEYS_PATH", "/app/secrets/api-keys.json")
    if os.path.exists(keys_path):
        with open(keys_path) as f:
            return json.load(f)
    # Default key for development
    return {"sk-dev-key": {"user": "dev", "rate_limit": 1000}}


config = load_config()
api_keys = load_api_keys()
