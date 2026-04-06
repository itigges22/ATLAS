"""Seed persistent patterns — universal coding patterns with infinite TTL."""

import logging
from typing import List

from models.pattern import Pattern, PatternType, PatternTier

logger = logging.getLogger(__name__)

# 15 universal patterns loaded at startup, never decay
SEED_PATTERNS: List[Pattern] = [
    Pattern(
        id="seed-null-check",
        type=PatternType.BUG_FIX,
        tier=PatternTier.PERSISTENT,
        content='if value is None:\n    raise ValueError("Expected non-None value")\n# or\nresult = value if value is not None else default',
        summary="Check for None/null before accessing attributes or using values",
        context_query="null check pattern",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-try-except",
        type=PatternType.IDIOM,
        tier=PatternTier.PERSISTENT,
        content='try:\n    result = risky_operation()\nexcept SpecificError as e:\n    logger.error(f"Operation failed: {e}")\n    result = fallback_value',
        summary="Wrap risky operations in try/except with specific exception types and logging",
        context_query="error handling try except",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-context-manager",
        type=PatternType.IDIOM,
        tier=PatternTier.PERSISTENT,
        content='with open(filepath, "r") as f:\n    data = f.read()\n# or for custom resources:\nclass ManagedResource:\n    def __enter__(self):\n        self.acquire()\n        return self\n    def __exit__(self, *args):\n        self.release()',
        summary="Use context managers (with statement) for automatic resource cleanup",
        context_query="resource cleanup context manager",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-input-validation",
        type=PatternType.BUG_FIX,
        tier=PatternTier.PERSISTENT,
        content='def process(data: str, count: int) -> list:\n    if not isinstance(data, str):\n        raise TypeError(f"Expected str, got {type(data).__name__}")\n    if count < 0:\n        raise ValueError(f"count must be non-negative, got {count}")',
        summary="Validate function inputs at system boundaries with clear error messages",
        context_query="input validation parameter checking",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-list-comprehension",
        type=PatternType.IDIOM,
        tier=PatternTier.PERSISTENT,
        content='# Instead of:\nresult = []\nfor item in items:\n    if condition(item):\n        result.append(transform(item))\n# Use:\nresult = [transform(item) for item in items if condition(item)]',
        summary="Use list comprehensions for concise filtering and transformation",
        context_query="list comprehension filter transform",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-dict-get",
        type=PatternType.BUG_FIX,
        tier=PatternTier.PERSISTENT,
        content='# Instead of:\nif key in d:\n    value = d[key]\nelse:\n    value = default\n# Use:\nvalue = d.get(key, default)',
        summary="Use dict.get() with default value to avoid KeyError",
        context_query="dictionary key access default value",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-f-string",
        type=PatternType.IDIOM,
        tier=PatternTier.PERSISTENT,
        content='name = "world"\n# Instead of: "Hello " + name + "!"\n# Use:\nmessage = f"Hello {name}!"',
        summary="Use f-strings for readable string formatting",
        context_query="string formatting f-string",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-enumerate",
        type=PatternType.IDIOM,
        tier=PatternTier.PERSISTENT,
        content='# Instead of:\ni = 0\nfor item in items:\n    print(i, item)\n    i += 1\n# Use:\nfor i, item in enumerate(items):\n    print(i, item)',
        summary="Use enumerate() instead of manual index tracking in loops",
        context_query="loop index enumerate",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-pathlib",
        type=PatternType.API_PATTERN,
        tier=PatternTier.PERSISTENT,
        content='from pathlib import Path\n\npath = Path("dir") / "subdir" / "file.txt"\nif path.exists():\n    content = path.read_text()\npath.parent.mkdir(parents=True, exist_ok=True)',
        summary="Use pathlib.Path for cross-platform file path operations",
        context_query="file path operations pathlib",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-sql-parameterize",
        type=PatternType.BUG_FIX,
        tier=PatternTier.PERSISTENT,
        content='# NEVER: f"SELECT * FROM users WHERE id = {user_id}"\n# ALWAYS:\ncursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
        summary="Always use parameterized SQL queries to prevent SQL injection",
        context_query="sql injection parameterized query",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-generator",
        type=PatternType.IDIOM,
        tier=PatternTier.PERSISTENT,
        content='def read_large_file(path):\n    with open(path) as f:\n        for line in f:\n            yield line.strip()\n# Memory-efficient: only one line in memory at a time',
        summary="Use generators for memory-efficient processing of large sequences",
        context_query="generator yield memory efficient",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-dataclass",
        type=PatternType.ARCHITECTURAL,
        tier=PatternTier.PERSISTENT,
        content='from dataclasses import dataclass, field\n\n@dataclass\nclass Config:\n    host: str = "localhost"\n    port: int = 8080\n    tags: list = field(default_factory=list)',
        summary="Use dataclasses for structured data containers with sensible defaults",
        context_query="dataclass structured data model",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-retry-backoff",
        type=PatternType.ERROR_FIX,
        tier=PatternTier.PERSISTENT,
        content='import time\n\ndef retry_with_backoff(func, max_retries=3, base_delay=1.0):\n    for attempt in range(max_retries):\n        try:\n            return func()\n        except Exception as e:\n            if attempt == max_retries - 1:\n                raise\n            delay = base_delay * (2 ** attempt)\n            time.sleep(delay)',
        summary="Retry failed operations with exponential backoff to handle transient errors",
        context_query="retry exponential backoff transient error",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-logging-setup",
        type=PatternType.ARCHITECTURAL,
        tier=PatternTier.PERSISTENT,
        content='import logging\n\nlogger = logging.getLogger(__name__)\n\n# Use appropriate levels:\nlogger.debug("Detailed trace info")\nlogger.info("Normal operation")\nlogger.warning("Something unexpected")\nlogger.error("Operation failed", exc_info=True)',
        summary="Use module-level logger with __name__ and appropriate log levels",
        context_query="logging setup module logger levels",
        half_life_days=9999,
    ),
    Pattern(
        id="seed-async-gather",
        type=PatternType.API_PATTERN,
        tier=PatternTier.PERSISTENT,
        content='import asyncio\n\nasync def fetch_all(urls):\n    tasks = [fetch_one(url) for url in urls]\n    results = await asyncio.gather(*tasks, return_exceptions=True)\n    return [r for r in results if not isinstance(r, Exception)]',
        summary="Use asyncio.gather for concurrent async operations with error handling",
        context_query="asyncio gather concurrent parallel async",
        half_life_days=9999,
    ),
]


async def load_seed_patterns():
    """Load seed persistent patterns into Redis at startup."""
    from cache.pattern_store import get_pattern_store

    store = get_pattern_store()
    if not store.available:
        logger.warning("Cannot load seed patterns: Redis unavailable")
        return

    loaded = 0
    for pattern in SEED_PATTERNS:
        # Only store if not already present
        existing = store.get_pattern(pattern.id)
        if existing is None:
            store.store_pattern(pattern, score=100.0)  # High score, never evicted
            loaded += 1

    if loaded > 0:
        logger.info(f"Loaded {loaded} seed persistent patterns")
    else:
        logger.info(f"All {len(SEED_PATTERNS)} seed patterns already present")
