"""Data models for Confidence Router."""

from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field


class Route(str, Enum):
    CACHE_HIT = "cache_hit"
    FAST_PATH = "fast_path"
    STANDARD = "standard"
    HARD_PATH = "hard_path"


class DifficultyBin(str, Enum):
    LOW = "LOW"          # 0.00 - 0.25
    MEDIUM = "MEDIUM"    # 0.25 - 0.50
    HIGH = "HIGH"        # 0.50 - 0.75
    EXTREME = "EXTREME"  # 0.75 - 1.00


# Cost model (relative to CACHE_HIT baseline)
ROUTE_COSTS: Dict[Route, int] = {
    Route.CACHE_HIT: 1,
    Route.FAST_PATH: 50,
    Route.STANDARD: 300,
    Route.HARD_PATH: 1500,
}

# Retry budget per route
ROUTE_RETRY_BUDGET: Dict[Route, int] = {
    Route.CACHE_HIT: 0,
    Route.FAST_PATH: 1,
    Route.STANDARD: 5,
    Route.HARD_PATH: 20,
}

# Fallback chain: route -> next route on failure
FALLBACK_CHAIN: Dict[Route, Optional[Route]] = {
    Route.CACHE_HIT: Route.FAST_PATH,
    Route.FAST_PATH: Route.STANDARD,
    Route.STANDARD: Route.HARD_PATH,
    Route.HARD_PATH: None,  # Terminal failure
}


class SignalBundle(BaseModel):
    """Collected signals for routing decision."""

    pattern_cache_score: float = 0.0      # s_p: [0,1], high = likely easy
    retrieval_confidence: float = 0.0     # PageIndex score: [0,1], high = well-understood
    query_complexity: float = 0.0         # q_c: [0,1], high = likely complex
    geometric_energy: float = 0.0         # C(x) normalized energy from Geometric Lens
    gx_score: float = 0.0                # G(x) XGBoost quality prediction [0,1]


class RouteDecision(BaseModel):
    """Result of a routing decision."""

    route: Route
    difficulty_score: float        # D(x) in [0,1]
    difficulty_bin: DifficultyBin
    retry_budget: int
    signals: SignalBundle
    thompson_samples: Dict[str, float] = Field(default_factory=dict)
    cache_hit_available: bool = False


def difficulty_to_bin(difficulty: float) -> DifficultyBin:
    """Bin a difficulty score into a category."""
    if difficulty < 0.25:
        return DifficultyBin.LOW
    elif difficulty < 0.50:
        return DifficultyBin.MEDIUM
    elif difficulty < 0.75:
        return DifficultyBin.HIGH
    else:
        return DifficultyBin.EXTREME
