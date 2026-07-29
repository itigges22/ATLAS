"""Geometric Lens pipeline — retrieval orchestration with PageIndex and the Pattern Cache."""

import asyncio
import httpx
import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timezone

from config import config
from geometric_lens.auth_token import auth_headers as _svc_auth_headers

logger = logging.getLogger(__name__)

# In-memory cache of loaded PageIndex data per project
_pageindex_cache: Dict[str, Any] = {}

# Versioned cache of the Pattern Cache BM25 matcher.
# Invalidated by checking PatternStore.get_version() — if the version has
# changed since we last built, we rebuild from get_all_patterns().
_pattern_matcher_cache: Dict[str, Any] = {"version": -1, "matcher": None}

# Context budget constants (tokens, estimated at 4 chars/token)
PAGEINDEX_BUDGET = 6000
CACHE_BUDGET = 2000
FULL_BUDGET = 8000


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
                    "\n... (additional context truncated due to length limit)"
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
    from retriever.hybrid import HybridRetriever

    # Check in-memory cache
    cache_key = f"pageindex:{project_id}"
    cached = _pageindex_cache.get(cache_key)

    if cached is None:
        loaded = load_index(project_id)
        if loaded is None:
            # !r escapes control chars so a malicious project_id can't
            # forge log entries (py/log-injection). The validator in
            # persistence.load_index already rejects anything not in
            # ^[A-Za-z0-9_-]{1,128}$, but the safe-render here protects
            # against future refactors that bypass the validator.
            logger.warning(f"No PageIndex found for {project_id!r}, returning empty")
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

def _get_pattern_matcher(store):
    """Get a PatternMatcher built over all patterns, rebuilding only when the store mutates."""
    from cache.pattern_matcher import PatternMatcher

    current_version = store.get_version()
    cached_matcher = _pattern_matcher_cache.get("matcher")
    cached_version = _pattern_matcher_cache.get("version")

    if cached_matcher is not None and cached_version == current_version:
        return cached_matcher

    all_patterns = store.get_all_patterns()
    if not all_patterns:
        _pattern_matcher_cache["matcher"] = None
        _pattern_matcher_cache["version"] = current_version
        return None

    matcher = PatternMatcher()
    matcher.build(all_patterns)
    _pattern_matcher_cache["matcher"] = matcher
    _pattern_matcher_cache["version"] = current_version
    return matcher


async def retrieve_cached_patterns(query: str, top_k: int = 3):
    """
    Read path: query Pattern Cache for matching patterns.

    Flow:
    1. BM25 match across all patterns (STM + LTM + persistent)
    2. Co-occurrence expansion of the top BM25 hits
    3. Score all candidates with Ebbinghaus decay
    4. Return top-k by composite score
    """
    from cache.pattern_store import get_pattern_store
    from cache.pattern_scorer import compute_score
    from cache.co_occurrence import CoOccurrenceGraph

    store = get_pattern_store()
    if not store.available:
        return []

    matcher = _get_pattern_matcher(store)
    if matcher is None:
        store.record_miss()
        return []

    bm25_matches = matcher.search(query, top_k=10)
    if not bm25_matches:
        store.record_miss()
        return []

    cooccur = CoOccurrenceGraph()
    candidate_patterns = {}  # pattern_id -> (Pattern, similarity)

    for pattern, similarity in bm25_matches:
        candidate_patterns[pattern.id] = (pattern, similarity)

        linked = cooccur.get_linked_patterns(pattern.id, top_k=3, max_depth=2)
        for linked_id, edge_weight in linked:
            if linked_id not in candidate_patterns:
                linked_pattern = store.get_pattern(linked_id)
                if linked_pattern:
                    candidate_patterns[linked_id] = (linked_pattern, similarity * edge_weight)

    scored = [
        compute_score(pattern, similarity)
        for pattern, similarity in candidate_patterns.values()
    ]
    scored.sort(key=lambda ps: ps.composite_score, reverse=True)

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
    verify: bool = False,
    test_code: str = "",
    stdin: str = "",
    expected_output: Optional[str] = None,
    **kwargs,
):
    """
    Perform RAG-enhanced chat completion with the Pattern Cache.

    1. Extract query from messages
    2. Search for relevant chunks (via configured retrieval mode)
    3. Query Pattern Cache for matching patterns (READ PATH)
    4. Build enhanced system prompt with context + cached patterns
    5. Forward to llama-server
    6. Return response
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

    verify_result = None

    # ── Pattern Cache: Write Path ──────────────────────────────
    # Drive writes off the verify result so we only persist patterns whose
    # solutions actually passed validation. Without verify there's no
    # ground-truth signal, so we leave the cache untouched.
    if verify_result is not None:
        active_pattern_ids = [ps.pattern.id for ps in scored_patterns]

        if active_pattern_ids:
            asyncio.create_task(
                record_pattern_outcome(active_pattern_ids, success=verify_result.passed)
            )

        if verify_result.passed:
            solution_text = verify_result.final_code or ""
            if solution_text:
                source_files = list({
                    c.get("file_path", "") for c in chunks if c.get("file_path")
                })
                last_error = ""
                for attempt in reversed(verify_result.attempt_history or []):
                    if not attempt.get("passed", True):
                        last_error = attempt.get("stderr", "") or attempt.get("error", "")
                        break

                asyncio.create_task(
                    write_pattern_async(
                        query=query,
                        solution=solution_text,
                        retry_count=verify_result.attempts,
                        max_retries=verify_result.max_attempts,
                        error_context=last_error or None,
                        source_files=source_files,
                        active_pattern_ids=active_pattern_ids,
                    )
                )

    return result


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

    async with httpx.AsyncClient(timeout=300.0, headers=_svc_auth_headers()) as client:
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

    async with httpx.AsyncClient(timeout=300.0, headers=_svc_auth_headers()) as client:
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
