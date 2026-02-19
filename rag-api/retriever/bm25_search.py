"""BM25-based exact identifier search — fast path for specific lookups."""

import logging
from typing import List, Dict, Any, Optional

from indexer.bm25_index import BM25Index, BM25Result

logger = logging.getLogger(__name__)


class BM25Searcher:
    """Fast-path retriever for exact identifier matches."""

    def __init__(self, bm25_index: BM25Index):
        self.index = bm25_index

    def search(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Search for code matching the query using BM25.

        Returns a list of result dicts compatible with the RAG context format:
        [{file_path, start_line, end_line, content, language, score}]
        """
        results = self.index.search(query, top_k=top_k)

        formatted = []
        for r in results:
            if r.score < min_score:
                continue

            # Extract file path from the node path (strip ::symbol suffix)
            file_path = r.path.split("::")[0] if "::" in r.path else r.path

            formatted.append({
                "file_path": file_path,
                "start_line": r.start_line or 0,
                "end_line": r.end_line or 0,
                "content": r.content or "",
                "language": _detect_language_from_path(file_path),
                "score": r.score,
                "source": "bm25",
                "node_name": r.name,
            })

        return formatted

    def has_strong_match(self, query: str, threshold: float = 3.0) -> bool:
        """
        Check if BM25 has a high-confidence match for this query.

        Used by the hybrid router to decide if tree traversal can be skipped.
        """
        results = self.index.search(query, top_k=1)
        if results and results[0].score >= threshold:
            return True
        return False


def _detect_language_from_path(path: str) -> str:
    """Detect language from file path extension."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
    }
    for ext, lang in ext_map.items():
        if path.endswith(ext):
            return lang
    return "text"
