import os
import yaml
import json
from pydantic import BaseModel
from typing import Dict, Optional


class ServerConfig(BaseModel):
    # Default Lens port across the stack:
    #   - Dockerfile: EXPOSE 31144
    #   - docker-compose: ATLAS_LENS_PORT=31144 default
    #   - benchmarks/h200/entrypoint.sh: LENS_PORT=31144
    #   - every consumer URL hardcodes :31144 or reads $ATLAS_LENS_PORT
    # The earlier 8099 default left local `python main.py` runs binding to
    # a port nothing else in the stack could reach — silent unreachability,
    # not a crash. Override via env (LENS_PORT) when intentionally moving.
    port: int = int(os.environ.get("LENS_PORT", "31144"))
    host: str = "0.0.0.0"


class LlamaConfig(BaseModel):
    """vLLM gen instance URL for chat completions calls (named LlamaConfig
    for backwards compat with existing imports)."""
    base_url: str = os.environ.get(
        "LLAMA_GEN_URL",
        os.environ.get("LLAMA_URL", "http://vllm-gen:8000"),
    )
    embed_url: str = os.environ.get("LLAMA_EMBED_URL", "http://vllm-embed:8001")
    gen_model: str = os.environ.get("LLAMA_GEN_MODEL", "qwen3.5-9b")
    embed_model: str = os.environ.get("LLAMA_EMBED_MODEL", "qwen3.5-9b-embed")


class LimitsConfig(BaseModel):
    max_files: int = 10000
    max_loc: int = 500000
    max_size_mb: int = 100
    project_ttl_hours: int = 24


class RetrievalConfig(BaseModel):
    top_k: int = 20
    context_budget_tokens: int = 8000


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    llama: LlamaConfig = LlamaConfig()
    limits: LimitsConfig = LimitsConfig()
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
    import logging
    logging.getLogger(__name__).warning(
        "No API key file found. Set ATLAS_API_KEY_FILE or create /app/config/api_keys.json"
    )
    return {}


config = load_config()
api_keys = load_api_keys()
