"""Hybrid retriever: routes between BM25 and tree search, merges results."""

import logging
import re
from typing import List, Dict, Any, Optional

from models.tree_node import TreeIndex
from indexer.bm25_index import BM25Index
from retriever.bm25_search import BM25Searcher
from retriever.tree_search import TreeSearcher

logger = logging.getLogger(__name__)

# BM25 score threshold for skipping tree search entirely
BM25_SKIP_TREE_THRESHOLD = 3.0


class HybridRetriever:
    """
    Hybrid retriever combining BM25 keyword search and LLM-guided tree search.

    Routing logic:
    1. If query contains specific identifiers → BM25 first
    2. If BM25 returns high-confidence results → skip tree search (fast path)
    3. Otherwise → tree traversal (or both for mixed queries)
    """

    def __init__(
        self,
        tree_index: TreeIndex,
        bm25_index: BM25Index,
        llama_url: str = "http://llama-service:8000",
    ):
        self.tree_index = tree_index
        self.bm25_index = bm25_index
        self.bm25_searcher = BM25Searcher(bm25_index)
        self.tree_searcher = TreeSearcher(tree_index.root, llama_url)
        self.llama_url = llama_url

    async def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search for code relevant to the query using the best strategy.

        Returns results in the standard chunk format:
        [{file_path, start_line, end_line, content, language, score, source}]
        """
        route = self._decide_route(query)
        logger.info(f"Hybrid retrieval route: {route} for query: {query[:80]}...")

        if route == "bm25_only":
            results = self.bm25_searcher.search(query, top_k=top_k)
            return results

        elif route == "tree_only":
            results = await self.tree_searcher.search(query, top_k=top_k)
            return results

        elif route == "bm25_first":
            # Try BM25 first
            bm25_results = self.bm25_searcher.search(query, top_k=top_k)

            # If BM25 found strong matches, return them
            if bm25_results and bm25_results[0].get("score", 0) >= BM25_SKIP_TREE_THRESHOLD:
                logger.info(
                    f"BM25 fast path: {len(bm25_results)} results, "
                    f"top score {bm25_results[0]['score']:.2f}"
                )
                return bm25_results

            # Otherwise, also run tree search and merge
            tree_results = await self.tree_searcher.search(query, top_k=top_k)
            return self._merge_results(bm25_results, tree_results, top_k)

        else:  # "both"
            bm25_results = self.bm25_searcher.search(query, top_k=top_k)
            tree_results = await self.tree_searcher.search(query, top_k=top_k)
            return self._merge_results(bm25_results, tree_results, top_k)

    def _decide_route(self, query: str) -> str:
        """
        Decide which retrieval strategy to use.

        Returns one of: "bm25_only", "tree_only", "bm25_first", "both"
        """
        has_identifiers = _query_has_identifiers(query)
        has_semantic = _query_is_semantic(query)
        # Also check if BM25 has any term matches (catches lowercase identifiers
        # like "hello" that don't match CamelCase/snake_case patterns)
        has_bm25_hits = self.bm25_index.has_exact_match(query)

        if has_identifiers and not has_semantic:
            return "bm25_first"
        elif has_semantic and not has_identifiers and not has_bm25_hits:
            return "tree_only"
        elif has_identifiers and has_semantic:
            return "both"
        elif has_bm25_hits:
            # Query terms match BM25 index even without identifier patterns
            return "both" if has_semantic else "bm25_first"
        else:
            return "tree_only"

    def _merge_results(
        self,
        bm25_results: List[Dict[str, Any]],
        tree_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from BM25 and tree search."""
        seen_keys = set()
        merged = []

        # Normalize scores: BM25 scores are unbounded, tree scores are 0-10
        # Scale BM25 scores to 0-10 range
        max_bm25 = max((r.get("score", 0) for r in bm25_results), default=1.0) or 1.0

        for r in bm25_results:
            r["score"] = (r.get("score", 0) / max_bm25) * 10.0

        # Interleave: prioritize by score
        all_results = bm25_results + tree_results

        for r in sorted(all_results, key=lambda x: x.get("score", 0), reverse=True):
            # Dedup key: file_path + start_line
            key = (r.get("file_path", ""), r.get("start_line", 0))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(r)

            if len(merged) >= top_k:
                break

        return merged


def _query_has_identifiers(query: str) -> bool:
    """Check if the query contains likely code identifiers."""
    # CamelCase pattern
    if re.search(r'[A-Z][a-z]+[A-Z]', query):
        return True
    # snake_case pattern
    if re.search(r'[a-z]+_[a-z]+', query):
        return True
    # Dot-separated identifiers
    if re.search(r'[a-zA-Z]+\.[a-zA-Z]+', query):
        return True
    # Error class names
    if re.search(r'[A-Z]\w+Error\b', query):
        return True
    # Function call pattern
    if re.search(r'\w+\(\)', query):
        return True
    return False


def _query_is_semantic(query: str) -> bool:
    """Check if the query is a semantic/natural language request."""
    semantic_signals = [
        "fix", "bug", "issue", "error", "problem", "broken",
        "how", "why", "what", "where", "find",
        "implement", "add", "create", "build", "make",
        "refactor", "optimize", "improve", "clean",
        "explain", "understand", "describe",
        "the", "in", "of", "to", "a", "an",
    ]
    words = query.lower().split()
    semantic_count = sum(1 for w in words if w in semantic_signals)
    return semantic_count >= 2
