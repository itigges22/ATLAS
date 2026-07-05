"""SQLite-backed Hebbian co-retrieval graph."""

import logging
from typing import List, Tuple, Set

from cache.pattern_store import get_pattern_store
from sqlite_store import get_db_pool

logger = logging.getLogger(__name__)

# Max edges per pattern (prune below this to prevent hairball)
MAX_EDGES_PER_PATTERN = 10

class CoOccurrenceGraph:
    """Directed weighted graph of pattern co-occurrence."""

    def __init__(self):
        self._store = get_pattern_store()
        self._pool = get_db_pool()
        self._available = self._store.available

    def record_co_occurrence(self, pattern_ids: List[str]):
        unique_ids = list(dict.fromkeys(pattern_ids))
        if not self._available or len(unique_ids) < 2:
            return

        try:
            with self._pool.get_connection() as conn:
                for pid_a in unique_ids:
                    # Self-count
                    conn.execute("""
                        INSERT INTO co_occurrence (source_id, target_id, count)
                        VALUES (?, ?, 1)
                        ON CONFLICT(source_id, target_id) DO UPDATE SET count = count + 1
                    """, (pid_a, pid_a))
                    
                    for pid_b in unique_ids:
                        if pid_a == pid_b:
                            continue
                        conn.execute("""
                            INSERT INTO co_occurrence (source_id, target_id, count)
                            VALUES (?, ?, 1)
                            ON CONFLICT(source_id, target_id) DO UPDATE SET count = count + 1
                        """, (pid_a, pid_b))
        except Exception as e:
            logger.error(f"Failed to record co-occurrence: {e}")

    def get_linked_patterns(
        self,
        pattern_id: str,
        top_k: int = 5,
        max_depth: int = 1,
    ) -> List[Tuple[str, float]]:
        if not self._available:
            return []

        visited: Set[str] = {pattern_id}
        results: List[Tuple[str, float]] = []

        self._dfs(pattern_id, 0, max_depth, visited, results, 1.0)

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
        if depth >= max_depth:
            return

        try:
            with self._pool.get_connection() as conn:
                # Get self-count
                cur = conn.execute("SELECT count FROM co_occurrence WHERE source_id = ? AND target_id = ?", (current_id, current_id))
                row = cur.fetchone()
                self_count = float(row["count"]) if row else 1.0

                # Get edges
                cur = conn.execute("""
                    SELECT target_id, count FROM co_occurrence 
                    WHERE source_id = ? AND target_id != ?
                    ORDER BY count DESC LIMIT ?
                """, (current_id, current_id, MAX_EDGES_PER_PATTERN))
                edges = cur.fetchall()

            for row in edges:
                linked_id = row["target_id"]
                count = row["count"]

                if linked_id in visited:
                    continue

                weight = (float(count) / self_count) * parent_weight

                if weight < 0.05:
                    continue

                visited.add(linked_id)
                results.append((linked_id, weight))

                self._dfs(linked_id, depth + 1, max_depth, visited, results, weight)
        except Exception as e:
            logger.error(f"Co-occurrence DFS failed at {current_id}: {e}")

    def cleanup_pattern(self, pattern_id: str):
        if not self._available:
            return

        try:
            with self._pool.get_connection() as conn:
                conn.execute("DELETE FROM co_occurrence WHERE source_id = ? OR target_id = ?", (pattern_id, pattern_id))
        except Exception as e:
            logger.error(f"Failed to cleanup co-occurrence for {pattern_id}: {e}")
