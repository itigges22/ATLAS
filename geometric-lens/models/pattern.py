"""Data models for the pattern cache (STM/LTM/Persistent tiers)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PatternType(str, Enum):
    ERROR_FIX = "error_fix"
    BUG_FIX = "bug_fix"
    API_PATTERN = "api_pattern"
    ARCHITECTURAL = "architectural"
    IDIOM = "idiom"


class PatternTier(str, Enum):
    STM = "stm"
    LTM = "ltm"
    PERSISTENT = "persistent"


# Ebbinghaus half-life in days per pattern type
HALF_LIVES: Dict[PatternType, float] = {
    PatternType.ERROR_FIX: 7.0,
    PatternType.BUG_FIX: 14.0,
    PatternType.API_PATTERN: 21.0,
    PatternType.ARCHITECTURAL: 30.0,
    PatternType.IDIOM: 14.0,
}


class Pattern(BaseModel):
    id: str
    type: PatternType
    tier: PatternTier
    content: str
    summary: str
    context_query: str
    error_context: Optional[str] = None
    surprise_score: float = 0.0
    access_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    half_life_days: float = 14.0
    source_files: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def days_since_access(self) -> float:
        """Days elapsed since last access."""
        now = datetime.now(timezone.utc)
        last = self.last_accessed
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (now - last).total_seconds() / 86400.0

    def age_days(self) -> float:
        """Days elapsed since pattern was created."""
        now = datetime.now(timezone.utc)
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (now - created).total_seconds() / 86400.0

    def success_rate(self) -> float:
        """Fraction of accesses that resulted in success."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class PatternScore(BaseModel):
    pattern: Pattern
    similarity: float
    decay_factor: float
    frequency_boost: float
    composite_score: float
