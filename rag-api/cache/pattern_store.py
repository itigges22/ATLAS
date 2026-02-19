"""Redis-backed pattern storage with STM/LTM sorted sets."""

import json
import logging
import os
from typing import List, Optional, Dict
from datetime import datetime, timezone

import redis

from models.pattern import Pattern, PatternTier

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# Redis key prefixes
STM_SORTED_SET = "pcache:stm"  # Sorted set: pattern_id -> composite_score
LTM_SORTED_SET = "pcache:ltm"  # Sorted set: pattern_id -> composite_score
PERSISTENT_SET = "pcache:persistent"  # Set: pattern_ids
PATTERN_HASH = "pcache:pattern:"  # Hash per pattern: pcache:pattern:{id}

# Capacity limits
STM_CAPACITY = 100


class PatternStore:
    """Redis client wrapper for pattern CRUD and sorted set management."""

    def __init__(self, redis_url: str = REDIS_URL):
        try:
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._available = True
            logger.info("Pattern store connected to Redis")
        except Exception as e:
            logger.warning(f"Pattern store Redis unavailable: {e}")
            self._redis = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def store_pattern(self, pattern: Pattern, score: float = 0.0) -> bool:
        """Store a pattern in Redis. STM patterns go to sorted set + hash."""
        if not self._available:
            return False

        try:
            pipe = self._redis.pipeline()

            # Store pattern data as hash
            key = f"{PATTERN_HASH}{pattern.id}"
            pipe.set(key, pattern.model_dump_json())

            # Add to appropriate sorted set
            if pattern.tier == PatternTier.STM:
                pipe.zadd(STM_SORTED_SET, {pattern.id: score})
                # Enforce capacity — remove lowest-scoring if over limit
                pipe.execute()
                self._enforce_stm_capacity()
            elif pattern.tier == PatternTier.LTM:
                pipe.zadd(LTM_SORTED_SET, {pattern.id: score})
                pipe.execute()
            elif pattern.tier == PatternTier.PERSISTENT:
                pipe.sadd(PERSISTENT_SET, pattern.id)
                pipe.execute()
            else:
                pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Failed to store pattern {pattern.id}: {e}")
            return False

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get a pattern by ID."""
        if not self._available:
            return None

        try:
            data = self._redis.get(f"{PATTERN_HASH}{pattern_id}")
            if data:
                return Pattern.model_validate_json(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get pattern {pattern_id}: {e}")
            return None

    def update_pattern(self, pattern: Pattern, score: Optional[float] = None) -> bool:
        """Update pattern data and optionally its score in the sorted set."""
        if not self._available:
            return False

        try:
            key = f"{PATTERN_HASH}{pattern.id}"
            self._redis.set(key, pattern.model_dump_json())

            if score is not None:
                if pattern.tier == PatternTier.STM:
                    self._redis.zadd(STM_SORTED_SET, {pattern.id: score})
                elif pattern.tier == PatternTier.LTM:
                    self._redis.zadd(LTM_SORTED_SET, {pattern.id: score})

            return True
        except Exception as e:
            logger.error(f"Failed to update pattern {pattern.id}: {e}")
            return False

    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern from Redis (hash + sorted set)."""
        if not self._available:
            return False

        try:
            pipe = self._redis.pipeline()
            pipe.delete(f"{PATTERN_HASH}{pattern_id}")
            pipe.zrem(STM_SORTED_SET, pattern_id)
            pipe.zrem(LTM_SORTED_SET, pattern_id)
            pipe.srem(PERSISTENT_SET, pattern_id)
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Failed to delete pattern {pattern_id}: {e}")
            return False

    def get_stm_patterns(self, limit: int = 50) -> List[Pattern]:
        """Get STM patterns sorted by score descending."""
        return self._get_sorted_set_patterns(STM_SORTED_SET, limit)

    def get_ltm_patterns(self, limit: int = 50) -> List[Pattern]:
        """Get LTM patterns sorted by score descending."""
        return self._get_sorted_set_patterns(LTM_SORTED_SET, limit)

    def get_persistent_patterns(self) -> List[Pattern]:
        """Get all persistent patterns."""
        if not self._available:
            return []

        try:
            ids = self._redis.smembers(PERSISTENT_SET)
            patterns = []
            for pid in ids:
                p = self.get_pattern(pid)
                if p:
                    patterns.append(p)
            return patterns
        except Exception as e:
            logger.error(f"Failed to get persistent patterns: {e}")
            return []

    def get_all_patterns(self) -> List[Pattern]:
        """Get all patterns across tiers."""
        patterns = []
        patterns.extend(self.get_stm_patterns(limit=STM_CAPACITY))
        patterns.extend(self.get_ltm_patterns(limit=500))
        patterns.extend(self.get_persistent_patterns())
        return patterns

    def promote_to_ltm(self, pattern_id: str, score: float) -> bool:
        """Promote a pattern from STM to LTM."""
        if not self._available:
            return False

        try:
            pattern = self.get_pattern(pattern_id)
            if not pattern:
                return False

            pattern.tier = PatternTier.LTM

            pipe = self._redis.pipeline()
            pipe.zrem(STM_SORTED_SET, pattern_id)
            pipe.zadd(LTM_SORTED_SET, {pattern_id: score})
            pipe.set(f"{PATTERN_HASH}{pattern_id}", pattern.model_dump_json())
            pipe.execute()

            logger.info(f"Promoted pattern {pattern_id} to LTM (score={score:.3f})")
            return True
        except Exception as e:
            logger.error(f"Failed to promote pattern {pattern_id}: {e}")
            return False

    def stm_size(self) -> int:
        """Get current STM size."""
        if not self._available:
            return 0
        try:
            return self._redis.zcard(STM_SORTED_SET)
        except Exception:
            return 0

    def ltm_size(self) -> int:
        """Get current LTM size."""
        if not self._available:
            return 0
        try:
            return self._redis.zcard(LTM_SORTED_SET)
        except Exception:
            return 0

    def persistent_size(self) -> int:
        """Get current persistent tier size."""
        if not self._available:
            return 0
        try:
            return self._redis.scard(PERSISTENT_SET)
        except Exception:
            return 0

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        if not self._available:
            return {"available": False}

        try:
            # Get hit/miss counters
            hits = int(self._redis.get("pcache:stats:hits") or 0)
            misses = int(self._redis.get("pcache:stats:misses") or 0)
            writes = int(self._redis.get("pcache:stats:writes") or 0)

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

    def record_hit(self):
        """Increment cache hit counter."""
        if self._available:
            try:
                self._redis.incr("pcache:stats:hits")
            except Exception:
                pass

    def record_miss(self):
        """Increment cache miss counter."""
        if self._available:
            try:
                self._redis.incr("pcache:stats:misses")
            except Exception:
                pass

    def record_write(self):
        """Increment cache write counter."""
        if self._available:
            try:
                self._redis.incr("pcache:stats:writes")
            except Exception:
                pass

    def flush(self):
        """Clear all pattern cache data (for testing/reset)."""
        if not self._available:
            return

        try:
            # Delete all pcache: keys
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match="pcache:*", count=100)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
            logger.info("Pattern cache flushed")
        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")

    def _get_sorted_set_patterns(self, key: str, limit: int) -> List[Pattern]:
        """Get patterns from a sorted set, highest score first."""
        if not self._available:
            return []

        try:
            # ZREVRANGE: highest score first
            ids = self._redis.zrevrange(key, 0, limit - 1)
            patterns = []
            for pid in ids:
                p = self.get_pattern(pid)
                if p:
                    patterns.append(p)
            return patterns
        except Exception as e:
            logger.error(f"Failed to get patterns from {key}: {e}")
            return []

    def _enforce_stm_capacity(self):
        """Remove lowest-scoring patterns if STM exceeds capacity."""
        try:
            size = self._redis.zcard(STM_SORTED_SET)
            if size > STM_CAPACITY:
                # Remove the excess lowest-scoring patterns
                excess = size - STM_CAPACITY
                evicted = self._redis.zpopmin(STM_SORTED_SET, excess)
                for pid, _ in evicted:
                    self._redis.delete(f"{PATTERN_HASH}{pid}")
                logger.info(f"Evicted {len(evicted)} patterns from STM (capacity={STM_CAPACITY})")
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
