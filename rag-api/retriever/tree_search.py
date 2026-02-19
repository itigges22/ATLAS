"""LLM-guided tree traversal for semantic code retrieval."""

import logging
import json
import re
from typing import List, Dict, Any, Optional

import httpx

from models.tree_node import TreeNode, NodeType

logger = logging.getLogger(__name__)

# Relevance score threshold for expanding a child node
EXPAND_THRESHOLD = 6
# High-confidence threshold — stop exploring siblings
HIGH_CONFIDENCE_THRESHOLD = 8
# Maximum depth of tree traversal
MAX_DEPTH = 6
# Maximum total LLM reasoning calls per retrieval
MAX_REASONING_CALLS = 40


class TreeSearcher:
    """LLM-guided tree traversal retriever."""

    def __init__(self, root: TreeNode, llama_url: str = "http://llama-service:8000"):
        self.root = root
        self.llama_url = llama_url
        self._call_count = 0

    async def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search the tree for code relevant to the query.

        Starts at root, evaluates child summaries with LLM reasoning,
        expands high-scoring nodes, recurses to leaf level.
        """
        self._call_count = 0
        collected: List[Dict[str, Any]] = []

        await self._traverse(self.root, query, collected, depth=0)

        # Sort by score descending and return top_k
        collected.sort(key=lambda x: x.get("score", 0), reverse=True)
        return collected[:top_k]

    async def _traverse(
        self,
        node: TreeNode,
        query: str,
        collected: List[Dict[str, Any]],
        depth: int,
    ):
        """Recursively traverse the tree guided by LLM scoring."""
        if depth > MAX_DEPTH:
            return
        if self._call_count >= MAX_REASONING_CALLS:
            return

        # If this is a leaf node with content, collect it
        if not node.children and node.content:
            file_path = node.path.split("::")[0] if "::" in node.path else node.path
            collected.append({
                "file_path": file_path,
                "start_line": node.metadata.start_line or 0,
                "end_line": node.metadata.end_line or 0,
                "content": node.content,
                "language": node.metadata.language or "python",
                "score": 10.0,  # Leaf reached by traversal gets high score
                "source": "tree",
                "node_name": node.name,
            })
            return

        # If no children, nothing to explore
        if not node.children:
            return

        # Score children using LLM reasoning
        scores = await self._score_children(node, query)

        # Sort children by score descending
        scored_children = sorted(
            zip(node.children, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Expand children above threshold
        for child, score in scored_children:
            if score < EXPAND_THRESHOLD:
                continue

            # If child is a leaf with content, collect directly
            if not child.children and child.content:
                file_path = (
                    child.path.split("::")[0] if "::" in child.path else child.path
                )
                collected.append({
                    "file_path": file_path,
                    "start_line": child.metadata.start_line or 0,
                    "end_line": child.metadata.end_line or 0,
                    "content": child.content,
                    "language": child.metadata.language or "python",
                    "score": float(score),
                    "source": "tree",
                    "node_name": child.name,
                })
            else:
                # Recurse deeper
                await self._traverse(child, query, collected, depth + 1)

            # Early termination: if we found a high-confidence match, skip siblings
            if score >= HIGH_CONFIDENCE_THRESHOLD:
                break

    async def _score_children(
        self,
        parent: TreeNode,
        query: str,
    ) -> List[int]:
        """
        Use LLM to score relevance of each child node to the query.

        Returns a list of scores (0-10) parallel to parent.children.
        """
        if not parent.children:
            return []

        # Build the child summaries for the prompt
        child_descriptions = []
        for i, child in enumerate(parent.children):
            summary = child.summary or f"{child.node_type.value}: {child.name}"
            # Truncate very long summaries
            if len(summary) > 200:
                summary = summary[:200] + "..."
            child_descriptions.append(f"{i}. [{child.node_type.value}] {child.name}: {summary}")

        children_text = "\n".join(child_descriptions)

        prompt = (
            f"Given the query: \"{query}\"\n\n"
            f"Score each item's relevance (0-10) to the query. "
            f"Return ONLY a JSON array of integers.\n\n"
            f"Items:\n{children_text}\n\n"
            f"Scores:"
        )

        self._call_count += 1

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.llama_url}/v1/chat/completions",
                    json={
                        "model": "default",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a code relevance scorer. "
                                    "Given a query and a list of code components, "
                                    "score each on relevance 0-10. "
                                    "Respond with ONLY a JSON array of integers, e.g. [3, 8, 1, 5]."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 50,
                        "temperature": 0.0,
                    },
                )
                response.raise_for_status()
                data = response.json()

                content = ""
                if "choices" in data and data["choices"]:
                    msg = data["choices"][0].get("message", {})
                    content = msg.get("content", "")
                    if not content:
                        content = msg.get("reasoning_content", "")

                return _parse_scores(content, len(parent.children))

        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}")
            # Fallback: give all children a score above expand threshold
            # so traversal still works when LLM is unavailable
            return [7] * len(parent.children)


def _parse_scores(response_text: str, expected_count: int) -> List[int]:
    """Parse LLM response into a list of integer scores."""
    response_text = response_text.strip()

    # Try to extract JSON array
    match = re.search(r'\[[\d\s,]+\]', response_text)
    if match:
        try:
            scores = json.loads(match.group())
            scores = [max(0, min(10, int(s))) for s in scores]
            # Pad or truncate to expected count
            while len(scores) < expected_count:
                scores.append(0)
            return scores[:expected_count]
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to extract individual numbers
    numbers = re.findall(r'\b(\d+)\b', response_text)
    if numbers:
        scores = [max(0, min(10, int(n))) for n in numbers[:expected_count]]
        while len(scores) < expected_count:
            scores.append(0)
        return scores

    # Complete fallback
    return [5] * expected_count
