"""Hebbian co-retrieval graph — directed weighted edges in Redis sorted sets."""

import logging
from typing import List, Tuple, Optional, Set

import redis

from cache.pattern_store import get_pattern_store

logger = logging.getLogger(__name__)

# Redis key prefix for co-occurrence edges
# pcache:cooccur:{pattern_id} -> sorted set of {linked_pattern_id: weight}
COOCCUR_PREFIX = "pcache:cooccur:"

# Max edges per pattern (prune below this to prevent hairball)
MAX_EDGES_PER_PATTERN = 10

# Min co-occurrence count before creating an edge
MIN_COOCCUR_COUNT = 1


class CoOccurrenceGraph:
    """Directed weighted graph of pattern co-occurrence."""

    def __init__(self):
        store = get_pattern_store()
        self._redis = store._redis
        self._available = store.available

    def record_co_occurrence(self, pattern_ids: List[str]):
        """
        Record that these patterns were all active in a successful task.
        Increments edge weights for all pairs (Hebbian: fire together, wire together).
        """
        if not self._available or len(pattern_ids) < 2:
            return

        try:
            pipe = self._redis.pipeline()
            for i, pid_a in enumerate(pattern_ids):
                for j, pid_b in enumerate(pattern_ids):
                    if i == j:
                        continue
                    # Increment edge A->B
                    pipe.zincrby(f"{COOCCUR_PREFIX}{pid_a}", 1, pid_b)
                    # Also increment self-reference count for normalization
                    pipe.zincrby(f"{COOCCUR_PREFIX}{pid_a}", 1, pid_a)
            pipe.execute()
        except Exception as e:
            logger.error(f"Failed to record co-occurrence: {e}")

    def get_linked_patterns(
        self,
        pattern_id: str,
        top_k: int = 5,
        max_depth: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Get linked patterns via DFS traversal of co-occurrence graph.

        Returns list of (pattern_id, edge_weight) tuples.
        Edge weight = Count(i,j) / Count(i,i) per Memoria formulation.
        """
        if not self._available:
            return []

        visited: Set[str] = {pattern_id}
        results: List[Tuple[str, float]] = []

        self._dfs(pattern_id, 0, max_depth, visited, results, 1.0)

        # Sort by weight descending, take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _dfs(
        self,
        current_id: str,
        depth: int,
        max_depth: int,
        visited: Set[str],
        results: List[Tuple[str, float]],
        parent_weight: float,
    ):
        """DFS traversal of co-occurrence graph."""
        if depth >= max_depth:
            return

        try:
            key = f"{COOCCUR_PREFIX}{current_id}"

            # Get self-count for normalization
            self_count = self._redis.zscore(key, current_id) or 1.0

            # Get all edges sorted by weight
            edges = self._redis.zrevrange(key, 0, MAX_EDGES_PER_PATTERN - 1, withscores=True)

            for linked_id, count in edges:
                if linked_id == current_id:
                    continue
                if linked_id in visited:
                    continue

                # E(i->j) = Count(i,j) / Count(i,i)
                weight = (count / self_count) * parent_weight

                if weight < 0.05:  # Skip weak edges
                    continue

                visited.add(linked_id)
                results.append((linked_id, weight))

                # Recurse deeper
                self._dfs(linked_id, depth + 1, max_depth, visited, results, weight)
        except Exception as e:
            logger.error(f"Co-occurrence DFS failed at {current_id}: {e}")

    def cleanup_pattern(self, pattern_id: str):
        """Remove all co-occurrence edges for a deleted pattern."""
        if not self._available:
            return

        try:
            # Delete outgoing edges
            self._redis.delete(f"{COOCCUR_PREFIX}{pattern_id}")

            # Remove from other patterns' edge lists (expensive but necessary)
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(
                    cursor, match=f"{COOCCUR_PREFIX}*", count=50
                )
                pipe = self._redis.pipeline()
                for k in keys:
                    pipe.zrem(k, pattern_id)
                pipe.execute()
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Failed to cleanup co-occurrence for {pattern_id}: {e}")
