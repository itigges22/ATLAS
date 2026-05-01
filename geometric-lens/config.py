import os
import yaml
import json
from pydantic import BaseModel
from typing import Dict, Optional


class ServerConfig(BaseModel):
    port: int = 8099
    host: str = "0.0.0.0"


class LlamaConfig(BaseModel):
    base_url: str = os.environ.get("LLAMA_URL", "http://llama-server:8080")


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


def load_config() -> Config:
    config_path = os.environ.get("CONFIG_PATH", "/app/config/config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f)
            return Config(**data)
    return Config()


def load_api_keys() -> Dict[str, dict]:
    """Load local API keys.

    File format: a JSON object mapping bearer token -> metadata dict, e.g.:
        {"sk-abc123...": {"user": "alice"}, "sk-def456...": {"user": "bob"}}

    Auth is local-only; there is no remote portal validation. Authenticated
    endpoints will return 401 until a key file is provided.
    """
    keys_path = os.environ.get("API_KEYS_PATH", "/app/secrets/api-keys.json")
    if os.path.exists(keys_path):
        with open(keys_path) as f:
            return json.load(f)
    import logging
    logging.getLogger(__name__).warning(
        f"No API key file found at {keys_path}. "
        f"Set API_KEYS_PATH or mount a JSON file at that location. "
        f"Authenticated endpoints will return 401 until a key file is provided."
    )
    return {}


config = load_config()
api_keys = load_api_keys()
