import httpx
import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator

from config import config
from vector_store import vector_store

logger = logging.getLogger(__name__)


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
        file_path = chunk["file_path"]
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(chunk)

    # Sort chunks within each file by start_line
    for file_path in by_file:
        by_file[file_path].sort(key=lambda c: c["start_line"])

    # Build context string
    context_parts = []
    total_chars = 0
    max_chars = max_tokens * 4  # Rough estimate

    for file_path, file_chunks in by_file.items():
        for chunk in file_chunks:
            # Format chunk
            chunk_text = f"""### File: {file_path} (lines {chunk['start_line']}-{chunk['end_line']})
```{chunk['language']}
{chunk['content']}
```
"""
            # Check if we have room
            if total_chars + len(chunk_text) > max_chars:
                # Add truncation notice
                context_parts.append(f"\n... (additional context truncated due to length limit)")
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

    return "\n".join(context_parts)


def build_system_prompt(context: str) -> str:
    """Build the system prompt with RAG context."""
    if not context:
        return """You are a coding assistant. The user has not synced their codebase yet, so you don't have access to their project files. You can still help with general coding questions."""

    return f"""You are a coding assistant with full awareness of the user's codebase.

## Retrieved Codebase Context

The following code sections are relevant to the user's query:

{context}

## Instructions

- You have access to the above code context from the user's project
- Use this context to provide accurate, project-specific assistance
- When suggesting changes, reference the actual file paths and line numbers
- If you need more context about a specific file or function, ask the user
- Be precise and reference the actual code shown above
"""


async def rag_enhanced_completion(
    project_id: str,
    messages: List[Dict[str, str]],
    model: str,
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 16384,
    stream: bool = False,
    **kwargs
):
    """
    Perform RAG-enhanced chat completion.

    1. Extract query from messages
    2. Search vector store for relevant chunks
    3. Build enhanced system prompt with context
    4. Forward to llama-server
    5. Return response
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

    # Search for relevant chunks
    try:
        chunks = await vector_store.search(
            project_id=project_id,
            query=query,
            top_k=config.retrieval.top_k
        )
        logger.info(f"Retrieved {len(chunks)} chunks for query")
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        chunks = []

    # Build context and system prompt
    context = build_context_prompt(chunks, config.retrieval.context_budget_tokens)
    system_prompt = build_system_prompt(context)

    # Prepare messages with system prompt
    enhanced_messages = [{"role": "system", "content": system_prompt}]

    # Add original messages (skip any existing system messages)
    for msg in messages:
        if msg.get("role") != "system":
            enhanced_messages.append(msg)

    # Forward to llama
    if stream:
        return forward_to_llama_stream(enhanced_messages, model, tools, max_tokens, **kwargs)
    return await forward_to_llama(enhanced_messages, model, tools, max_tokens, **kwargs)


async def forward_to_llama(
    messages: List[Dict[str, str]],
    model: str,
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 16384,
    **kwargs
) -> Dict[str, Any]:
    """Forward request to llama-server."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **kwargs
    }

    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{config.llama.base_url}/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        result = response.json()

        # Handle reasoning_content: ensure content is always populated
        # Some models (like Qwen3) put responses in reasoning_content instead of content
        if "choices" in result:
            for choice in result["choices"]:
                msg = choice.get("message", {})
                content = msg.get("content", "")
                reasoning = msg.get("reasoning_content", "")

                # If content is empty but reasoning_content exists, use reasoning
                if not content and reasoning:
                    msg["content"] = reasoning
                # If both exist, combine them (content first, then reasoning as context)
                elif content and reasoning:
                    # Keep content as-is, reasoning is available separately
                    pass
                # Ensure content key exists
                if "content" not in msg:
                    msg["content"] = ""

        return result


async def simple_completion(
    messages: List[Dict[str, str]],
    model: str,
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 16384,
    stream: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Simple completion without RAG (for requests without project_id)."""
    return await forward_to_llama(messages, model, tools, max_tokens, stream=stream, **kwargs)


async def forward_to_llama_stream(
    messages: List[Dict[str, str]],
    model: str,
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 16384,
    **kwargs
) -> AsyncGenerator[str, None]:
    """Forward streaming request to llama-server."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        **kwargs
    }

    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{config.llama.base_url}/v1/chat/completions",
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break

                    # Parse and potentially fix reasoning_content issue
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk:
                            for choice in chunk["choices"]:
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                reasoning = delta.get("reasoning_content", "")

                                # If content is empty but reasoning exists, use it
                                if not content and reasoning:
                                    delta["content"] = reasoning

                        yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        yield f"data: {data}\n\n"
