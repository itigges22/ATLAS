"""RAG pipeline — retrieval orchestration with PageIndex, Pattern Cache, and Confidence Router."""

import asyncio
import os
import httpx
import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime, timezone

from config import config

logger = logging.getLogger(__name__)

# In-memory cache of loaded PageIndex data per project
_pageindex_cache: Dict[str, Any] = {}

# Context budget constants (tokens, estimated at 4 chars/token)
PAGEINDEX_BUDGET = 6000
CACHE_BUDGET = 2000
FULL_BUDGET = 8000


def is_routing_enabled() -> bool:
    """Check if confidence routing is enabled (ROUTING_ENABLED env var)."""
    return os.environ.get("ROUTING_ENABLED", "true").lower() in ("true", "1", "yes")


def _get_router_redis():
    """Get Redis client for the router (reuses existing connection)."""
    try:
        import redis
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        r = redis.from_url(redis_url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


def build_context_prompt(chunks: List[Dict[str, Any]], max_tokens: int = 8000) -> str:
    """
    Build a context prompt from retrieved chunks.
    Groups chunks by file and maintains line order.
    """
    if not chunks:
        return ""

    # Group by file
    by_file: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        file_path = chunk.get("file_path", "unknown")
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(chunk)

    # Sort chunks within each file by start_line
    for file_path in by_file:
        by_file[file_path].sort(key=lambda c: c.get("start_line", 0))

    # Build context string
    context_parts = []
    total_chars = 0
    max_chars = max_tokens * 4  # Rough estimate

    for file_path, file_chunks in by_file.items():
        for chunk in file_chunks:
            language = chunk.get("language", "")
            start = chunk.get("start_line", 0)
            end = chunk.get("end_line", 0)
            content = chunk.get("content", "")

            chunk_text = f"""### File: {file_path} (lines {start}-{end})
```{language}
{content}
```
"""
            # Check if we have room
            if total_chars + len(chunk_text) > max_chars:
                context_parts.append(
                    f"\n... (additional context truncated due to length limit)"
                )
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

    return "\n".join(context_parts)


def build_cache_context(scored_patterns) -> str:
    """Build context string from cached pattern matches."""
    if not scored_patterns:
        return ""

    parts = ["[Cached Patterns — Previously successful solutions for similar tasks]"]
    total_chars = 0
    max_chars = CACHE_BUDGET * 4

    for i, ps in enumerate(scored_patterns[:3], 1):
        p = ps.pattern
        age_str = f"{p.days_since_access():.0f}" if p.days_since_access() < 1 else f"{p.days_since_access():.0f}"
        entry = (
            f"\nPattern {i} ({p.type.value}, used {p.access_count} times, "
            f"last used {age_str} days ago):\n"
            f"  Description: {p.summary}\n"
            f"  Code:\n```\n{p.content[:800]}\n```"
        )
        if total_chars + len(entry) > max_chars:
            break
        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


def build_system_prompt(context: str, cache_context: str = "") -> str:
    """Build the system prompt with RAG context and cached patterns."""
    if not context and not cache_context:
        return (
            "You are a coding assistant. The user has not synced their codebase yet, "
            "so you don't have access to their project files. "
            "You can still help with general coding questions."
        )

    parts = [
        "You are a coding assistant with full awareness of the user's codebase."
    ]

    if context:
        parts.append(f"""
## Retrieved Codebase Context

The following code sections are relevant to the user's query:

{context}""")

    if cache_context:
        parts.append(f"""
## Learned Patterns

The following patterns from previously successful solutions may be relevant:

{cache_context}""")

    parts.append("""
## Instructions

- You have access to the above code context from the user's project
- Use this context to provide accurate, project-specific assistance
- When suggesting changes, reference the actual file paths and line numbers
- If cached patterns are shown, consider them as proven solutions but adapt to the specific codebase
- Be precise and reference the actual code shown above
""")

    return "\n".join(parts)


async def retrieve_chunks_pageindex(
    project_id: str,
    query: str,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Retrieve chunks using PageIndex (tree search + BM25)."""
    from indexer.persistence import load_index
    from indexer.bm25_index import BM25Index
    from retriever.hybrid import HybridRetriever

    # Check in-memory cache
    cache_key = f"pageindex:{project_id}"
    cached = _pageindex_cache.get(cache_key)

    if cached is None:
        loaded = load_index(project_id)
        if loaded is None:
            logger.warning(f"No PageIndex found for {project_id}, returning empty")
            return []
        tree_index, bm25_index = loaded
        _pageindex_cache[cache_key] = (tree_index, bm25_index)
        cached = (tree_index, bm25_index)

    tree_index, bm25_index = cached
    retriever = HybridRetriever(
        tree_index=tree_index,
        bm25_index=bm25_index,
        llama_url=config.llama.base_url,
    )

    chunks = await retriever.search(query, top_k=top_k)
    return chunks


async def retrieve_chunks(
    project_id: str,
    query: str,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Retrieve chunks using PageIndex (tree search + BM25)."""
    return await retrieve_chunks_pageindex(project_id, query, top_k)


def invalidate_cache(project_id: str):
    """Invalidate the in-memory PageIndex cache for a project."""
    cache_key = f"pageindex:{project_id}"
    _pageindex_cache.pop(cache_key, None)


# ──────────────────────────────────────────────────────────────
# Pattern Cache: Read Path
# ──────────────────────────────────────────────────────────────

async def retrieve_cached_patterns(query: str, top_k: int = 3):
    """
    Read path: query Pattern Cache for matching patterns.

    Flow:
    1. BM25 match against STM pattern summaries
    2. Co-occurrence expansion to find linked LTM patterns
    3. Direct LTM search if STM matches are weak
    4. Score all candidates with Ebbinghaus decay
    5. Return top-k scored patterns
    """
    from cache.pattern_store import get_pattern_store
    from cache.pattern_matcher import PatternMatcher
    from cache.pattern_scorer import compute_score
    from cache.co_occurrence import CoOccurrenceGraph

    store = get_pattern_store()
    if not store.available:
        return []

    # Step 1: Get all patterns and build BM25 matcher
    all_patterns = store.get_all_patterns()
    if not all_patterns:
        store.record_miss()
        return []

    matcher = PatternMatcher()
    matcher.build(all_patterns)

    # Step 2: BM25 search
    bm25_matches = matcher.search(query, top_k=10)

    if not bm25_matches:
        store.record_miss()
        return []

    # Step 3: Co-occurrence expansion for top matches
    cooccur = CoOccurrenceGraph()
    candidate_patterns = {}  # pattern_id -> (Pattern, similarity)

    for pattern, similarity in bm25_matches:
        candidate_patterns[pattern.id] = (pattern, similarity)

        # Follow co-occurrence edges (depth 1-2)
        linked = cooccur.get_linked_patterns(pattern.id, top_k=3, max_depth=2)
        for linked_id, edge_weight in linked:
            if linked_id not in candidate_patterns:
                linked_pattern = store.get_pattern(linked_id)
                if linked_pattern:
                    # Use edge_weight * parent similarity as linked similarity
                    linked_sim = similarity * edge_weight
                    candidate_patterns[linked_id] = (linked_pattern, linked_sim)

    # Step 4: Direct LTM search if STM matches are weak
    top_sim = bm25_matches[0][1] if bm25_matches else 0
    if top_sim < 0.3:
        ltm_patterns = store.get_ltm_patterns(limit=50)
        if ltm_patterns:
            ltm_matcher = PatternMatcher()
            ltm_matcher.build(ltm_patterns)
            ltm_matches = ltm_matcher.search(query, top_k=5)
            for pattern, similarity in ltm_matches:
                if pattern.id not in candidate_patterns:
                    candidate_patterns[pattern.id] = (pattern, similarity)

    # Step 5: Score with Ebbinghaus decay
    scored = []
    for pattern, similarity in candidate_patterns.values():
        ps = compute_score(pattern, similarity)
        scored.append(ps)

    scored.sort(key=lambda ps: ps.composite_score, reverse=True)

    # Step 6: Record hit/miss and return
    result = scored[:top_k]
    if result:
        store.record_hit()
        logger.info(
            f"Pattern cache HIT: {len(result)} patterns for query '{query[:50]}...' "
            f"(top score={result[0].composite_score:.3f})"
        )
    else:
        store.record_miss()

    return result


async def record_pattern_access(scored_patterns):
    """Update last_accessed and access_count for retrieved patterns."""
    from cache.pattern_store import get_pattern_store
    from cache.pattern_scorer import compute_storage_score

    store = get_pattern_store()
    if not store.available:
        return

    now = datetime.now(timezone.utc).isoformat()
    for ps in scored_patterns:
        p = ps.pattern
        p.last_accessed = now
        p.access_count += 1
        score = compute_storage_score(p)
        store.update_pattern(p, score=score)


# ──────────────────────────────────────────────────────────────
# Pattern Cache: Write Path
# ──────────────────────────────────────────────────────────────

async def write_pattern_async(
    query: str,
    solution: str,
    retry_count: int,
    max_retries: int,
    error_context: Optional[str],
    source_files: List[str],
    active_pattern_ids: Optional[List[str]] = None,
):
    """
    Write path: extract and store a pattern from a successful task completion.
    Runs ASYNC — does not block the response pipeline.
    """
    from cache.pattern_store import get_pattern_store
    from cache.pattern_extractor import extract_pattern
    from cache.pattern_scorer import compute_storage_score
    from cache.co_occurrence import CoOccurrenceGraph
    from cache.consolidator import update_category_surprise

    store = get_pattern_store()
    if not store.available:
        return

    try:
        # Extract pattern via LLM
        pattern = await extract_pattern(
            query=query,
            solution=solution,
            retry_count=retry_count,
            max_retries=max_retries,
            error_context=error_context,
            source_files=source_files,
            llama_url=config.llama.base_url,
        )

        if not pattern:
            logger.warning("Pattern extraction returned None, skipping write")
            return

        # Compute storage score and store
        score = compute_storage_score(pattern)
        store.store_pattern(pattern, score=score)
        store.record_write()

        logger.info(
            f"Pattern written: {pattern.id} type={pattern.type.value} "
            f"surprise={pattern.surprise_score:.2f} score={score:.3f}"
        )

        # Update co-occurrence graph
        pattern_ids = [pattern.id]
        if active_pattern_ids:
            pattern_ids.extend(active_pattern_ids)

        if len(pattern_ids) >= 2:
            cooccur = CoOccurrenceGraph()
            cooccur.record_co_occurrence(pattern_ids)

        # Update category surprise
        update_category_surprise(pattern.type, pattern.surprise_score)

    except Exception as e:
        logger.error(f"Pattern write failed: {e}")


async def record_pattern_outcome(
    pattern_ids: List[str],
    success: bool,
):
    """Record whether injected patterns led to task success or failure."""
    from cache.pattern_store import get_pattern_store
    from cache.pattern_scorer import compute_storage_score

    store = get_pattern_store()
    if not store.available:
        return

    for pid in pattern_ids:
        pattern = store.get_pattern(pid)
        if pattern:
            if success:
                pattern.success_count += 1
                pattern.last_success = datetime.now(timezone.utc).isoformat()
            else:
                pattern.failure_count += 1
            score = compute_storage_score(pattern)
            store.update_pattern(pattern, score=score)


# ──────────────────────────────────────────────────────────────
# Main completion pipeline
# ──────────────────────────────────────────────────────────────

async def rag_enhanced_completion(
    project_id: str,
    messages: List[Dict[str, str]],
    model: str,
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 16384,
    stream: bool = False,
    **kwargs,
):
    """
    Perform RAG-enhanced chat completion with Pattern Cache and Confidence Router.

    1. Extract query from messages
    2. Search for relevant chunks (via configured retrieval mode)
    3. Query Pattern Cache for matching patterns (READ PATH)
    4. ROUTER: Collect signals, estimate difficulty, select route, set retry budget
    5. Build enhanced system prompt with context + cached patterns
    6. Forward to llama-server
    7. Return response (with route metadata in _route_decision for feedback recording)
    """
    # Extract query from last user message
    query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            query = msg.get("content", "")
            break

    if not query:
        logger.warning("No user message found in request")
        if stream:
            return forward_to_llama_stream(messages, model, tools, max_tokens, **kwargs)
        return await forward_to_llama(messages, model, tools, max_tokens, **kwargs)

    # Search for relevant chunks (PageIndex)
    try:
        chunks = await retrieve_chunks(
            project_id=project_id,
            query=query,
            top_k=config.retrieval.top_k,
        )
        logger.info(f"Retrieved {len(chunks)} chunks for query")
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        chunks = []

    # READ PATH: Query Pattern Cache
    cache_context = ""
    scored_patterns = []
    try:
        scored_patterns = await retrieve_cached_patterns(query, top_k=3)
        if scored_patterns:
            cache_context = build_cache_context(scored_patterns)
            # Record access (async, non-blocking)
            asyncio.create_task(record_pattern_access(scored_patterns))
    except Exception as e:
        logger.error(f"Pattern cache read failed: {e}")
        scored_patterns = []

    # ── Confidence Router ──────────────────────────────────────
    route_decision = None
    if is_routing_enabled():
        try:
            from router.signal_collector import collect_signals
            from router.difficulty_estimator import estimate_difficulty
            from router.route_selector import select_route as thompson_select
            from models.route import Route

            # Collect signals
            signals = collect_signals(
                query=query,
                scored_patterns=scored_patterns,
                chunks=chunks,
            )

            # Estimate difficulty
            difficulty = estimate_difficulty(signals)

            # Determine if cache hit is viable
            cache_hit_available = (
                bool(scored_patterns)
                and scored_patterns[0].composite_score > 0.7
            )

            # Select route via Thompson Sampling
            r = _get_router_redis()
            if r:
                route_decision = thompson_select(
                    r=r,
                    signals=signals,
                    difficulty=difficulty,
                    cache_hit_available=cache_hit_available,
                )
                logger.info(
                    f"Router decision: route={route_decision.route.value} "
                    f"D(x)={route_decision.difficulty_score:.3f} "
                    f"bin={route_decision.difficulty_bin.value} "
                    f"k={route_decision.retry_budget}"
                )
        except Exception as e:
            logger.error(f"Confidence Router failed, defaulting to STANDARD: {e}")
            route_decision = None

    # Context budget splitting:
    # If cache has hits: PageIndex gets 6K, cache gets 2K
    # If no cache hits: PageIndex gets full 8K
    if cache_context:
        context = build_context_prompt(chunks, PAGEINDEX_BUDGET)
    else:
        context = build_context_prompt(chunks, FULL_BUDGET)

    system_prompt = build_system_prompt(context, cache_context)

    # Prepare messages with system prompt
    enhanced_messages = [{"role": "system", "content": system_prompt}]

    # Add original messages (skip any existing system messages)
    for msg in messages:
        if msg.get("role") != "system":
            enhanced_messages.append(msg)

    # Forward to llama
    if stream:
        return forward_to_llama_stream(
            enhanced_messages, model, tools, max_tokens, **kwargs
        )
    result = await forward_to_llama(enhanced_messages, model, tools, max_tokens, **kwargs)

    # Attach route decision to result for feedback recording
    if route_decision and isinstance(result, dict):
        result["_route_decision"] = {
            "route": route_decision.route.value,
            "difficulty_score": route_decision.difficulty_score,
            "difficulty_bin": route_decision.difficulty_bin.value,
            "retry_budget": route_decision.retry_budget,
            "signals": route_decision.signals.model_dump(),
            "thompson_samples": route_decision.thompson_samples,
        }

    return result


# ──────────────────────────────────────────────────────────────
# Confidence Router: Feedback Recording
# ──────────────────────────────────────────────────────────────

def record_route_feedback(
    route_value: str,
    difficulty_bin_value: str,
    success: bool,
):
    """Record a routing outcome to update Thompson Sampling state."""
    if not is_routing_enabled():
        return

    try:
        from router.feedback_recorder import record_outcome
        from models.route import Route, DifficultyBin

        r = _get_router_redis()
        if not r:
            return

        route = Route(route_value)
        d_bin = DifficultyBin(difficulty_bin_value)
        record_outcome(r, d_bin, route, success)
    except Exception as e:
        logger.error(f"Failed to record route feedback: {e}")


async def forward_to_llama(
    messages: List[Dict[str, str]],
    model: str,
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 16384,
    **kwargs,
) -> Dict[str, Any]:
    """Forward request to llama-server."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **kwargs,
    }

    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{config.llama.base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        # Handle reasoning_content: ensure content is always populated
        if "choices" in result:
            for choice in result["choices"]:
                msg = choice.get("message", {})
                content = msg.get("content", "")
                reasoning = msg.get("reasoning_content", "")

                if not content and reasoning:
                    msg["content"] = reasoning
                elif content and reasoning:
                    pass
                if "content" not in msg:
                    msg["content"] = ""

        return result


async def simple_completion(
    messages: List[Dict[str, str]],
    model: str,
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 16384,
    stream: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Simple completion without RAG (for requests without project_id)."""
    return await forward_to_llama(
        messages, model, tools, max_tokens, stream=stream, **kwargs
    )


async def forward_to_llama_stream(
    messages: List[Dict[str, str]],
    model: str,
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 16384,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """Forward streaming request to llama-server."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        **kwargs,
    }

    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{config.llama.base_url}/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break

                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk:
                            for choice in chunk["choices"]:
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                reasoning = delta.get("reasoning_content", "")

                                if not content and reasoning:
                                    delta["content"] = reasoning

                        yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        yield f"data: {data}\n\n"
