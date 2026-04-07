"""Data models for Pattern Cache (Ebbinghaus Memory)."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class PatternType(str, Enum):
    BUG_FIX = "bug_fix"
    API_PATTERN = "api_pattern"
    ARCHITECTURAL = "architectural"
    IDIOM = "idiom"
    ERROR_FIX = "error_fix"


class PatternTier(str, Enum):
    STM = "stm"
    LTM = "ltm"
    PERSISTENT = "persistent"


# Tiered half-lives in days, per Ebbinghaus/Moltbook consensus
HALF_LIVES: Dict[PatternType, float] = {
    PatternType.API_PATTERN: 7.0,
    PatternType.BUG_FIX: 14.0,
    PatternType.ERROR_FIX: 14.0,
    PatternType.ARCHITECTURAL: 30.0,
    PatternType.IDIOM: 90.0,
}


class Pattern(BaseModel):
    """A cached code pattern with Ebbinghaus decay metadata."""

    id: str
    type: PatternType
    tier: PatternTier = PatternTier.STM
    content: str  # The actual code pattern
    summary: str  # LLM-generated natural language description
    context_query: str  # The original query that produced this pattern
    error_context: Optional[str] = None  # Error message for error_fix type

    # Scoring metadata
    surprise_score: float = 0.0  # 0.0-1.0, from retry count
    access_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Temporal metadata
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_accessed: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_success: Optional[str] = None
    half_life_days: float = 14.0

    # Relational metadata
    source_files: List[str] = Field(default_factory=list)

    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    def age_days(self) -> float:
        created = datetime.fromisoformat(self.created_at)
        now = datetime.now(timezone.utc)
        return (now - created).total_seconds() / 86400

    def days_since_access(self) -> float:
        accessed = datetime.fromisoformat(self.last_accessed)
        now = datetime.now(timezone.utc)
        return (now - accessed).total_seconds() / 86400


class PatternScore(BaseModel):
    """A scored pattern result from cache lookup."""

    pattern: Pattern
    similarity: float = 0.0  # BM25 similarity (0.0-1.0 normalized)
    decay_factor: float = 1.0  # Ebbinghaus decay
    frequency_boost: float = 0.0  # log(1 + access_count)
    composite_score: float = 0.0  # Final score
