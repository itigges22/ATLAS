"""Record task outcomes to update Thompson Sampling state in Redis."""

import logging
from typing import Optional

import redis as redis_lib

from models.route import Route, DifficultyBin

logger = logging.getLogger(__name__)

KEY_PREFIX = "confidence_router:thompson"
STATS_PREFIX = "confidence_router:stats"


def record_outcome(
    r: redis_lib.Redis,
    difficulty_bin: DifficultyBin,
    route: Route,
    success: bool,
):
    """
    Record a task outcome to update Thompson alpha/beta.

    On success: alpha += 1
    On failure: beta += 1
    """
    alpha_key = f"{KEY_PREFIX}:{difficulty_bin.value}:{route.value}:alpha"
    beta_key = f"{KEY_PREFIX}:{difficulty_bin.value}:{route.value}:beta"

    try:
        pipe = r.pipeline()
        if success:
            pipe.incrbyfloat(alpha_key, 1.0)
        else:
            pipe.incrbyfloat(beta_key, 1.0)

        # Track aggregate stats
        pipe.incr(f"{STATS_PREFIX}:total_decisions")
        if success:
            pipe.incr(f"{STATS_PREFIX}:total_successes")
        pipe.incr(f"{STATS_PREFIX}:route:{route.value}")
        pipe.incr(f"{STATS_PREFIX}:bin:{difficulty_bin.value}")

        pipe.execute()

        logger.info(
            f"Outcome recorded: bin={difficulty_bin.value} route={route.value} "
            f"success={success}"
        )
    except Exception as e:
        logger.error(f"Failed to record outcome: {e}")


def get_routing_stats(r: redis_lib.Redis) -> dict:
    """Get aggregate routing statistics from Redis."""
    try:
        total = int(r.get(f"{STATS_PREFIX}:total_decisions") or 0)
        successes = int(r.get(f"{STATS_PREFIX}:total_successes") or 0)

        route_dist = {}
        for route in Route:
            count = int(r.get(f"{STATS_PREFIX}:route:{route.value}") or 0)
            route_dist[route.value] = count

        bin_dist = {}
        for d_bin in DifficultyBin:
            count = int(r.get(f"{STATS_PREFIX}:bin:{d_bin.value}") or 0)
            bin_dist[d_bin.value] = count

        return {
            "total_decisions": total,
            "total_successes": successes,
            "success_rate": round(successes / total, 4) if total > 0 else 0.0,
            "route_distribution": route_dist,
            "difficulty_distribution": bin_dist,
        }
    except Exception as e:
        logger.error(f"Failed to get routing stats: {e}")
        return {"error": str(e)}


def reset_stats(r: redis_lib.Redis):
    """Reset aggregate routing statistics."""
    try:
        keys = []
        keys.append(f"{STATS_PREFIX}:total_decisions")
        keys.append(f"{STATS_PREFIX}:total_successes")
        for route in Route:
            keys.append(f"{STATS_PREFIX}:route:{route.value}")
        for d_bin in DifficultyBin:
            keys.append(f"{STATS_PREFIX}:bin:{d_bin.value}")
        if keys:
            r.delete(*keys)
        logger.info("Routing stats reset")
    except Exception as e:
        logger.error(f"Failed to reset stats: {e}")
