"""SQLite-backed pattern storage with STM/LTM scoring."""

import logging
from typing import List, Optional, Dict

from models.pattern import Pattern, PatternTier
from sqlite_store import get_db_pool

logger = logging.getLogger(__name__)

# Capacity limits
STM_CAPACITY = 100


class PatternStore:
    """SQLite client wrapper for pattern CRUD and sorted set management."""

    def __init__(self):
        self._pool = get_db_pool()
        self._available = True
        logger.info("Pattern store connected to SQLite")

    @property
    def available(self) -> bool:
        return self._available

    def get_version(self) -> int:
        """Monotonic counter bumped on every mutation. Used to invalidate caches."""
        if not self._available:
            return 0
        try:
            with self._pool.get_connection() as conn:
                cur = conn.execute("SELECT value FROM store_metadata WHERE key = 'version'")
                row = cur.fetchone()
                return row["value"] if row else 0
        except Exception:
            return 0

    def _bump_version(self):
        if self._available:
            try:
                with self._pool.get_connection() as conn:
                    conn.execute("""
                        INSERT INTO store_metadata (key, value)
                        VALUES ('version', 1)
                        ON CONFLICT(key) DO UPDATE SET value = value + 1
                    """)
            except Exception:
                pass

    def store_pattern(self, pattern: Pattern, score: float = 0.0) -> bool:
        if not self._available:
            return False
        try:
            with self._pool.get_connection() as conn:
                conn.execute("""
                    INSERT INTO patterns (id, data, tier, score)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        data = excluded.data,
                        tier = excluded.tier,
                        score = excluded.score
                """, (pattern.id, pattern.model_dump_json(), pattern.tier.value, score))
            if pattern.tier == PatternTier.STM:
                self._enforce_stm_capacity()
            self._bump_version()
            return True
        except Exception as e:
            logger.error(f"Failed to store pattern {pattern.id}: {e}")
            return False

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        if not self._available:
            return None
        try:
            with self._pool.get_connection() as conn:
                cur = conn.execute("SELECT data FROM patterns WHERE id = ?", (pattern_id,))
                row = cur.fetchone()
                if row:
                    return Pattern.model_validate_json(row["data"])
            return None
        except Exception as e:
            logger.error(f"Failed to get pattern {pattern_id!r}: {e}")
            return None

    def update_pattern(self, pattern: Pattern, score: Optional[float] = None) -> bool:
        if not self._available:
            return False
        try:
            with self._pool.get_connection() as conn:
                if score is not None:
                    conn.execute("""
                        UPDATE patterns SET data = ?, tier = ?, score = ?
                        WHERE id = ?
                    """, (pattern.model_dump_json(), pattern.tier.value, score, pattern.id))
                else:
                    conn.execute("""
                        UPDATE patterns SET data = ?, tier = ?
                        WHERE id = ?
                    """, (pattern.model_dump_json(), pattern.tier.value, pattern.id))
            self._bump_version()
            return True
        except Exception as e:
            logger.error(f"Failed to update pattern {pattern.id}: {e}")
            return False

    def delete_pattern(self, pattern_id: str) -> bool:
        if not self._available:
            return False
        try:
            with self._pool.get_connection() as conn:
                conn.execute("DELETE FROM patterns WHERE id = ?", (pattern_id,))
            self._bump_version()
            return True
        except Exception as e:
            logger.error(f"Failed to delete pattern {pattern_id}: {e}")
            return False

    def get_stm_patterns(self, limit: int = 50) -> List[Pattern]:
        return self._get_sorted_set_patterns(PatternTier.STM.value, limit)

    def get_ltm_patterns(self, limit: int = 50) -> List[Pattern]:
        return self._get_sorted_set_patterns(PatternTier.LTM.value, limit)

    def get_persistent_patterns(self) -> List[Pattern]:
        if not self._available:
            return []
        try:
            with self._pool.get_connection() as conn:
                cur = conn.execute("SELECT data FROM patterns WHERE tier = ?", (PatternTier.PERSISTENT.value,))
                return [Pattern.model_validate_json(row["data"]) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get persistent patterns: {e}")
            return []

    def get_all_patterns(self) -> List[Pattern]:
        patterns = []
        patterns.extend(self.get_stm_patterns(limit=STM_CAPACITY))
        patterns.extend(self.get_ltm_patterns(limit=500))
        patterns.extend(self.get_persistent_patterns())
        return patterns

    def promote_to_ltm(self, pattern_id: str, score: float) -> bool:
        if not self._available:
            return False
        try:
            pattern = self.get_pattern(pattern_id)
            if not pattern:
                return False
            pattern.tier = PatternTier.LTM
            with self._pool.get_connection() as conn:
                conn.execute("""
                    UPDATE patterns SET data = ?, tier = ?, score = ?
                    WHERE id = ?
                """, (pattern.model_dump_json(), PatternTier.LTM.value, score, pattern_id))
            self._bump_version()
            logger.info(f"Promoted pattern {pattern_id} to LTM (score={score:.3f})")
            return True
        except Exception as e:
            logger.error(f"Failed to promote pattern {pattern_id}: {e}")
            return False

    def _get_count(self, tier: str) -> int:
        if not self._available:
            return 0
        try:
            with self._pool.get_connection() as conn:
                cur = conn.execute("SELECT COUNT(*) as c FROM patterns WHERE tier = ?", (tier,))
                row = cur.fetchone()
                return row["c"] if row else 0
        except Exception:
            return 0

    def stm_size(self) -> int:
        return self._get_count(PatternTier.STM.value)

    def ltm_size(self) -> int:
        return self._get_count(PatternTier.LTM.value)

    def persistent_size(self) -> int:
        return self._get_count(PatternTier.PERSISTENT.value)

    def get_stats(self) -> Dict:
        if not self._available:
            return {"available": False}
        try:
            with self._pool.get_connection() as conn:
                cur = conn.execute("SELECT key, value FROM store_metadata WHERE key IN ('hits', 'misses', 'writes')")
                stats_db = {row["key"]: row["value"] for row in cur.fetchall()}
            hits = stats_db.get("hits", 0)
            misses = stats_db.get("misses", 0)
            writes = stats_db.get("writes", 0)
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0

            return {
                "available": True,
                "stm_size": self.stm_size(),
                "ltm_size": self.ltm_size(),
                "persistent_size": self.persistent_size(),
                "total_patterns": self.stm_size() + self.ltm_size() + self.persistent_size(),
                "hits": hits,
                "misses": misses,
                "writes": writes,
                "hit_rate": round(hit_rate, 4),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"available": True, "error": str(e)}

    def _incr_stat(self, key: str):
        if self._available:
            try:
                with self._pool.get_connection() as conn:
                    conn.execute("""
                        INSERT INTO store_metadata (key, value)
                        VALUES (?, 1)
                        ON CONFLICT(key) DO UPDATE SET value = value + 1
                    """, (key,))
            except Exception:
                pass

    def record_hit(self):
        self._incr_stat("hits")

    def record_miss(self):
        self._incr_stat("misses")

    def record_write(self):
        self._incr_stat("writes")

    def flush(self):
        if not self._available:
            return
        try:
            with self._pool.get_connection() as conn:
                conn.execute("DELETE FROM patterns")
                conn.execute("DELETE FROM store_metadata")
                conn.execute("DELETE FROM co_occurrence")
            self._bump_version()
            logger.info("Pattern cache flushed")
        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")

    def _get_sorted_set_patterns(self, tier: str, limit: int) -> List[Pattern]:
        if not self._available:
            return []
        try:
            with self._pool.get_connection() as conn:
                cur = conn.execute("SELECT data FROM patterns WHERE tier = ? ORDER BY score DESC LIMIT ?", (tier, limit))
                return [Pattern.model_validate_json(row["data"]) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get patterns from {tier}: {e}")
            return []

    def _enforce_stm_capacity(self):
        try:
            size = self.stm_size()
            if size > STM_CAPACITY:
                excess = size - STM_CAPACITY
                with self._pool.get_connection() as conn:
                    conn.execute("""
                        DELETE FROM patterns
                        WHERE id IN (
                            SELECT id FROM patterns
                            WHERE tier = ?
                            ORDER BY score ASC
                            LIMIT ?
                        )
                    """, (PatternTier.STM.value, excess))
                self._bump_version()
                logger.info(f"Evicted {excess} patterns from STM (capacity={STM_CAPACITY})")
        except Exception as e:
            logger.error(f"Failed to enforce STM capacity: {e}")


# Module-level singleton
_store: Optional[PatternStore] = None


def get_pattern_store() -> PatternStore:
    """Get or create the pattern store singleton."""
    global _store
    if _store is None:
        _store = PatternStore()
    return _store
