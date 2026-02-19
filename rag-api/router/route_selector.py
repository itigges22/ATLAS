"""Thompson Sampling route selection with Redis-backed state."""

import logging
import random
from typing import Dict, Optional, Tuple

import redis as redis_lib

from models.route import (
    Route, DifficultyBin, ROUTE_COSTS, ROUTE_RETRY_BUDGET,
    RouteDecision, SignalBundle, difficulty_to_bin,
)

logger = logging.getLogger(__name__)

# Redis key prefix for Thompson state
KEY_PREFIX = "confidence_router:thompson"


def _key(difficulty_bin: DifficultyBin, route: Route, param: str) -> str:
    """Build Redis key for Thompson state."""
    return f"{KEY_PREFIX}:{difficulty_bin.value}:{route.value}:{param}"


def _load_thompson_state(
    r: redis_lib.Redis,
    difficulty_bin: DifficultyBin,
) -> Dict[Route, Tuple[float, float]]:
    """Load (alpha, beta) for all routes in a difficulty bin from Redis.

    Redis stores raw outcome counts (0-based via incrbyfloat).
    We add 1.0 as the Beta prior: alpha = successes + 1, beta = failures + 1.
    """
    states = {}
    for route in Route:
        alpha_key = _key(difficulty_bin, route, "alpha")
        beta_key = _key(difficulty_bin, route, "beta")
        try:
            raw_alpha = float(r.get(alpha_key) or 0.0)
            raw_beta = float(r.get(beta_key) or 0.0)
        except (TypeError, ValueError):
            raw_alpha, raw_beta = 0.0, 0.0
        # Add Beta(1,1) uniform prior
        states[route] = (raw_alpha + 1.0, raw_beta + 1.0)
    return states


def _save_thompson_param(
    r: redis_lib.Redis,
    difficulty_bin: DifficultyBin,
    route: Route,
    alpha: float,
    beta: float,
):
    """Persist Thompson (alpha, beta) to Redis."""
    pipe = r.pipeline()
    pipe.set(_key(difficulty_bin, route, "alpha"), str(alpha))
    pipe.set(_key(difficulty_bin, route, "beta"), str(beta))
    pipe.execute()


def select_route(
    r: redis_lib.Redis,
    signals: SignalBundle,
    difficulty: float,
    cache_hit_available: bool = False,
) -> RouteDecision:
    """
    Select the best route via Thompson Sampling weighted by cost efficiency.

    Steps:
      1. Bin the difficulty score
      2. Sample from Beta(alpha, beta) posterior for each candidate route
      3. Weight by cost efficiency: sample / route_cost
      4. Apply difficulty-based constraints
      5. Select route with highest cost-weighted sample
    """
    d_bin = difficulty_to_bin(difficulty)

    # Load Thompson state from Redis
    try:
        states = _load_thompson_state(r, d_bin)
    except Exception as e:
        logger.warning(f"Failed to load Thompson state: {e}, defaulting to STANDARD")
        return RouteDecision(
            route=Route.STANDARD,
            difficulty_score=difficulty,
            difficulty_bin=d_bin,
            retry_budget=ROUTE_RETRY_BUDGET[Route.STANDARD],
            signals=signals,
            cache_hit_available=cache_hit_available,
        )

    # Sample from each route's Beta posterior
    samples: Dict[str, float] = {}
    for route in Route:
        # CACHE_HIT only considered when available AND difficulty is low
        if route == Route.CACHE_HIT:
            if not cache_hit_available or difficulty >= 0.3:
                continue

        alpha, beta = states[route]

        # Sample success probability from Beta distribution
        try:
            p_success = random.betavariate(alpha, beta)
        except ValueError:
            p_success = 0.5

        # Cost-weighted efficiency: higher is better
        efficiency = p_success / ROUTE_COSTS[route]

        # Difficulty-based constraints
        if difficulty > 0.6 and route == Route.FAST_PATH:
            efficiency *= 0.3  # Penalize fast path for hard tasks
        elif difficulty < 0.3 and route == Route.HARD_PATH:
            efficiency *= 0.3  # Penalize expensive routes for easy tasks

        samples[route.value] = efficiency

    if not samples:
        selected = Route.STANDARD
    else:
        selected_key = max(samples, key=samples.get)
        selected = Route(selected_key)

    logger.info(
        f"Route selected: {selected.value} (difficulty={difficulty:.3f}, "
        f"bin={d_bin.value}, budget=k{ROUTE_RETRY_BUDGET[selected]})"
    )

    return RouteDecision(
        route=selected,
        difficulty_score=difficulty,
        difficulty_bin=d_bin,
        retry_budget=ROUTE_RETRY_BUDGET[selected],
        signals=signals,
        thompson_samples=samples,
        cache_hit_available=cache_hit_available,
    )


def get_all_thompson_states(r: redis_lib.Redis) -> Dict[str, Dict[str, dict]]:
    """Get all Thompson states for monitoring. Returns {bin: {route: {alpha, beta, mean, samples}}}."""
    result = {}
    for d_bin in DifficultyBin:
        bin_states = {}
        for route in Route:
            alpha_key = _key(d_bin, route, "alpha")
            beta_key = _key(d_bin, route, "beta")
            try:
                raw_alpha = float(r.get(alpha_key) or 0.0)
                raw_beta = float(r.get(beta_key) or 0.0)
            except (TypeError, ValueError):
                raw_alpha, raw_beta = 0.0, 0.0

            # Add Beta(1,1) prior
            alpha = raw_alpha + 1.0
            beta = raw_beta + 1.0
            total_outcomes = raw_alpha + raw_beta
            mean = alpha / (alpha + beta)
            bin_states[route.value] = {
                "alpha": alpha,
                "beta": beta,
                "mean_success_rate": round(mean, 4),
                "total_outcomes": total_outcomes,
            }
        result[d_bin.value] = bin_states
    return result


def reset_thompson_state(r: redis_lib.Redis):
    """Reset all Thompson state to uniform priors (alpha=1, beta=1).

    Deletes raw count keys so they read as 0.0, and the +1.0 prior offset
    restores uniform Beta(1,1).
    """
    pipe = r.pipeline()
    for d_bin in DifficultyBin:
        for route in Route:
            pipe.delete(_key(d_bin, route, "alpha"))
            pipe.delete(_key(d_bin, route, "beta"))
    pipe.execute()
    logger.info("Thompson Sampling state reset to uniform priors")
