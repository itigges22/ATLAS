"""Thompson Sampling route selection with SQLite-backed state."""

import logging
import random
from typing import Dict, Tuple

from sqlite_store import get_db_pool

from models.route import (
    Route, DifficultyBin, ROUTE_COSTS, ROUTE_RETRY_BUDGET,
    RouteDecision, SignalBundle, difficulty_to_bin,
)

logger = logging.getLogger(__name__)

def _load_thompson_state(
    difficulty_bin: DifficultyBin,
) -> Dict[Route, Tuple[float, float]]:
    """Load (alpha, beta) for all routes in a difficulty bin from SQLite.

    We add 1.0 as the Beta prior: alpha = successes + 1, beta = failures + 1.
    """
    states = {}
    pool = get_db_pool()
    with pool.get_connection() as conn:
        cur = conn.execute(
            "SELECT route, alpha, beta FROM thompson_state WHERE difficulty_bin = ?", 
            (difficulty_bin.value,)
        )
        rows = cur.fetchall()
        db_states = {row["route"]: (row["alpha"], row["beta"]) for row in rows}

    for route in Route:
        raw_alpha, raw_beta = db_states.get(route.value, (0.0, 0.0))
        # Add Beta(1,1) uniform prior
        states[route] = (raw_alpha + 1.0, raw_beta + 1.0)
    return states


def select_route(
    signals: SignalBundle,
    difficulty: float,
    cache_hit_available: bool = False,
) -> RouteDecision:
    """
    Select the best route via Thompson Sampling weighted by cost efficiency.
    """
    d_bin = difficulty_to_bin(difficulty)

    # Load Thompson state from SQLite
    try:
        states = _load_thompson_state(d_bin)
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


def get_all_thompson_states() -> Dict[str, Dict[str, dict]]:
    """Get all Thompson states for monitoring. Returns {bin: {route: {alpha, beta, mean, samples}}}."""
    result = {}
    try:
        pool = get_db_pool()
        with pool.get_connection() as conn:
            cur = conn.execute("SELECT difficulty_bin, route, alpha, beta FROM thompson_state")
            rows = cur.fetchall()
            
        db_data = {}
        for row in rows:
            if row["difficulty_bin"] not in db_data:
                db_data[row["difficulty_bin"]] = {}
            db_data[row["difficulty_bin"]][row["route"]] = (row["alpha"], row["beta"])
    except Exception as e:
        logger.error(f"Failed to fetch thompson states: {e}")
        db_data = {}

    for d_bin in DifficultyBin:
        bin_states = {}
        for route in Route:
            raw_alpha, raw_beta = db_data.get(d_bin.value, {}).get(route.value, (0.0, 0.0))

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


def reset_thompson_state():
    """Reset all Thompson state to uniform priors (alpha=1, beta=1).
    Deletes raw count keys so they read as 0.0, and the +1.0 prior offset
    restores uniform Beta(1,1).
    """
    try:
        pool = get_db_pool()
        with pool.get_connection() as conn:
            conn.execute("DELETE FROM thompson_state")
        logger.info("Thompson Sampling state reset to uniform priors")
    except Exception as e:
        logger.error(f"Failed to reset thompson state: {e}")

