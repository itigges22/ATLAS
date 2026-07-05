"""Record task outcomes to update Thompson Sampling state in SQLite."""

import logging
from models.route import Route, DifficultyBin
from sqlite_store import get_db_pool

logger = logging.getLogger(__name__)

def record_outcome(
    difficulty_bin: DifficultyBin,
    route: Route,
    success: bool,
):
    """
    Record a task outcome to update Thompson alpha/beta.
    On success: alpha += 1
    On failure: beta += 1
    """
    try:
        pool = get_db_pool()
        with pool.get_connection() as conn:
            # Update Thompson State
            if success:
                conn.execute("""
                    INSERT INTO thompson_state (difficulty_bin, route, alpha, beta)
                    VALUES (?, ?, 1.0, 0.0)
                    ON CONFLICT(difficulty_bin, route) DO UPDATE SET alpha = alpha + 1.0
                """, (difficulty_bin.value, route.value))
            else:
                conn.execute("""
                    INSERT INTO thompson_state (difficulty_bin, route, alpha, beta)
                    VALUES (?, ?, 0.0, 1.0)
                    ON CONFLICT(difficulty_bin, route) DO UPDATE SET beta = beta + 1.0
                """, (difficulty_bin.value, route.value))

            # Track aggregate stats
            stats_keys = [
                "total_decisions",
                f"route:{route.value}",
                f"bin:{difficulty_bin.value}"
            ]
            if success:
                stats_keys.append("total_successes")
                
            for key in stats_keys:
                conn.execute("""
                    INSERT INTO routing_stats (key, value)
                    VALUES (?, 1)
                    ON CONFLICT(key) DO UPDATE SET value = value + 1
                """, (key,))

        logger.info(
            f"Outcome recorded: bin={difficulty_bin.value!r} "
            f"route={route.value!r} success={success}"
        )
    except Exception as e:
        logger.error(f"Failed to record outcome: {e}")


def get_routing_stats() -> dict:
    """Get aggregate routing statistics from SQLite."""
    try:
        pool = get_db_pool()
        with pool.get_connection() as conn:
            cur = conn.execute("SELECT key, value FROM routing_stats")
            rows = cur.fetchall()
            stats_map = {row["key"]: row["value"] for row in rows}
            
        total = stats_map.get("total_decisions", 0)
        successes = stats_map.get("total_successes", 0)

        route_dist = {}
        for route in Route:
            route_dist[route.value] = stats_map.get(f"route:{route.value}", 0)

        bin_dist = {}
        for d_bin in DifficultyBin:
            bin_dist[d_bin.value] = stats_map.get(f"bin:{d_bin.value}", 0)

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


def reset_stats():
    """Reset aggregate routing statistics."""
    try:
        pool = get_db_pool()
        with pool.get_connection() as conn:
            conn.execute("DELETE FROM routing_stats")
        logger.info("Routing stats reset")
    except Exception as e:
        logger.error(f"Failed to reset stats: {e}")

