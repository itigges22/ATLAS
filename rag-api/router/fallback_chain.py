"""Fallback/escalation logic when a route fails."""

import logging
from typing import Optional

from models.route import Route, FALLBACK_CHAIN, ROUTE_RETRY_BUDGET

logger = logging.getLogger(__name__)


def get_fallback_route(current_route: Route) -> Optional[Route]:
    """
    Get the next route in the fallback chain.

    Chain: CACHE_HIT -> FAST_PATH -> STANDARD -> HARD_PATH -> None (terminal)

    Returns None if no further escalation is possible.
    """
    next_route = FALLBACK_CHAIN.get(current_route)
    if next_route:
        logger.info(
            f"Escalating from {current_route.value} to {next_route.value} "
            f"(new budget: k={ROUTE_RETRY_BUDGET[next_route]})"
        )
    else:
        logger.info(f"No fallback available from {current_route.value} — terminal failure")
    return next_route


def get_escalation_path(starting_route: Route) -> list:
    """
    Get the full escalation path from a starting route.

    Returns list of (route, retry_budget) tuples.
    """
    path = []
    current = starting_route
    while current is not None:
        path.append((current, ROUTE_RETRY_BUDGET[current]))
        current = FALLBACK_CHAIN.get(current)
    return path
