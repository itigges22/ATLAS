import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App settings
    app_name: str = "LLM API Portal"
    debug: bool = False

    # Database
    database_url: str = "sqlite:///./data/portal.db"

    # Security
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production-use-openssl-rand-hex-32")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    # API Key settings
    api_key_prefix: str = "sk-llm-"

    # RAG API connection
    rag_api_url: str = os.getenv("RAG_API_URL", "http://rag-api:8001")
    llm_api_url: str = os.getenv("LLM_API_URL", "http://llama-service:8000")
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379")

    # Rate limiting
    default_rate_limit: int = 100  # requests per minute

    class Config:
        env_file = ".env"

settings = Settings()
