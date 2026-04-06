"""LLM-generated node summaries via llama-server."""

import logging
import hashlib
import re
from typing import Dict, Optional

import httpx

from models.tree_node import TreeNode, NodeType

logger = logging.getLogger(__name__)

# Default llama-server URL (overridable)
LLAMA_URL = "http://llama-service:8000"


async def summarize_tree(
    root: TreeNode,
    llama_url: str = LLAMA_URL,
    existing_summaries: Optional[Dict[str, str]] = None,
    file_hashes: Optional[Dict[str, str]] = None,
    old_file_hashes: Optional[Dict[str, str]] = None,
):
    """
    Generate summaries for all nodes in the tree, bottom-up.

    Leaf nodes get summaries from LLM based on code content.
    Non-leaf nodes get summaries rolled up from children.

    Supports incremental re-summarization: if file_hashes match
    old_file_hashes, the existing summary is reused.
    """
    existing_summaries = existing_summaries or {}
    file_hashes = file_hashes or {}
    old_file_hashes = old_file_hashes or {}

    await _summarize_node(
        root, llama_url, existing_summaries, file_hashes, old_file_hashes
    )


async def _summarize_node(
    node: TreeNode,
    llama_url: str,
    existing_summaries: Dict[str, str],
    file_hashes: Dict[str, str],
    old_file_hashes: Dict[str, str],
):
    """Recursively summarize a node (bottom-up)."""
    # Summarize children first
    for child in node.children:
        await _summarize_node(
            child, llama_url, existing_summaries, file_hashes, old_file_hashes
        )

    # Check if we can reuse an existing summary
    if node.node_id in existing_summaries:
        if _can_reuse_summary(node, file_hashes, old_file_hashes):
            node.summary = existing_summaries[node.node_id]
            return

    # Generate summary based on node type
    if node.node_type == NodeType.REPOSITORY:
        node.summary = _rollup_summary(node, "repository")
    elif node.node_type == NodeType.DIRECTORY:
        node.summary = _rollup_summary(node, "directory")
    elif node.node_type in (NodeType.FILE, NodeType.CLASS):
        if node.children:
            node.summary = _rollup_summary(node, node.node_type.value)
        elif node.content:
            node.summary = await _llm_summarize(
                node.content, node.node_type.value, node.name, llama_url
            )
        else:
            node.summary = f"{node.node_type.value}: {node.name}"
    elif node.node_type in (NodeType.FUNCTION, NodeType.BLOCK):
        if node.content:
            node.summary = await _llm_summarize(
                node.content, node.node_type.value, node.name, llama_url
            )
        else:
            node.summary = f"{node.node_type.value}: {node.name}"


def _can_reuse_summary(
    node: TreeNode,
    file_hashes: Dict[str, str],
    old_file_hashes: Dict[str, str],
) -> bool:
    """Check if a node's summary can be reused (file hasn't changed)."""
    # For file-level nodes, check if the file hash is the same
    if node.node_type == NodeType.FILE:
        path = node.path
        new_hash = file_hashes.get(path)
        old_hash = old_file_hashes.get(path)
        return new_hash is not None and new_hash == old_hash

    # For code-level nodes (class, function, block), check via content hash
    if node.content and node.metadata.file_hash:
        content_hash = hashlib.sha256(node.content.encode()).hexdigest()
        return content_hash == node.metadata.file_hash

    # For directories and repository, reuse if all children were reused
    # (This is implicit — if children didn't change, rollup won't change)
    return False


def _rollup_summary(node: TreeNode, node_type: str) -> str:
    """Create a summary by rolling up child summaries."""
    if not node.children:
        return f"{node_type}: {node.name}"

    child_summaries = []
    for child in node.children:
        if child.summary:
            child_summaries.append(f"- {child.name}: {child.summary}")

    if not child_summaries:
        return f"{node_type}: {node.name}"

    # For directories with many children, truncate
    if len(child_summaries) > 10:
        shown = child_summaries[:8]
        remaining = len(child_summaries) - 8
        shown.append(f"- ... and {remaining} more items")
        child_summaries = shown

    parts = "\n".join(child_summaries)
    return f"{node_type} '{node.name}' contains:\n{parts}"


async def _llm_summarize(
    code: str,
    node_type: str,
    name: str,
    llama_url: str,
) -> str:
    """Call llama-server to generate a summary for a code node."""
    # Truncate very long code to keep prompt manageable
    max_chars = 2000
    if len(code) > max_chars:
        code = code[:max_chars] + "\n... (truncated)"

    prompt = (
        f"Describe what this Python {node_type} '{name}' does in 1-2 sentences. "
        f"Focus on: purpose, inputs/outputs, and key dependencies.\n\n"
        f"```python\n{code}\n```\n\n"
        f"Summary:"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{llama_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a code documentation assistant. Provide concise, accurate summaries of code. Respond with only the summary, no preamble.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 100,
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
            data = response.json()

            content = ""
            if "choices" in data and data["choices"]:
                msg = data["choices"][0].get("message", {})
                content = msg.get("content", "")
                # Fallback to reasoning_content if content is empty
                if not content:
                    content = msg.get("reasoning_content", "")

            if content:
                content = _clean_reasoning_preamble(content.strip())
            return content if content else f"{node_type}: {name}"

    except Exception as e:
        logger.warning(f"LLM summarization failed for {name}: {e}")
        return f"{node_type}: {name}"


def _clean_reasoning_preamble(text: str) -> str:
    """Strip LLM reasoning preamble from Qwen3 reasoning_content responses."""
    # Remove common reasoning starters that Qwen3 produces
    # Pattern: "Okay, let's see. The user wants me to..." up to the actual content
    # The actual summary usually starts after a sentence about "The function/class..."
    lines = text.split("\n")
    cleaned = []
    skip_preamble = True
    for line in lines:
        stripped = line.strip()
        if skip_preamble:
            # Skip lines that are clearly reasoning, not summary
            if re.match(
                r"^(Okay|Let me|Let's|Hmm|So,|Alright|I need to|The user|They)",
                stripped,
                re.IGNORECASE,
            ):
                continue
            # Skip empty lines in preamble
            if not stripped:
                continue
            skip_preamble = False
        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # If everything was stripped, extract the best sentence from original
    if not result:
        # Find sentences that describe functionality
        sentences = re.split(r'(?<=[.!])\s+', text)
        for s in sentences:
            s = s.strip()
            if any(kw in s.lower() for kw in [
                "function", "method", "class", "returns", "takes",
                "prints", "creates", "initializes", "defines",
            ]):
                return s
        # Last resort: return last sentence
        if sentences:
            return sentences[-1].strip()

    return result


def collect_summaries(root: TreeNode) -> Dict[str, str]:
    """Collect all node summaries into a flat dict for caching."""
    summaries: Dict[str, str] = {}
    _collect(root, summaries)
    return summaries


def _collect(node: TreeNode, summaries: Dict[str, str]):
    """Recursively collect summaries."""
    if node.summary:
        summaries[node.node_id] = node.summary
    for child in node.children:
        _collect(child, summaries)
