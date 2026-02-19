"""BM25 matching against pattern summaries for fast lookup."""

import math
import re
import logging
from typing import List, Tuple

from models.pattern import Pattern

logger = logging.getLogger(__name__)


class PatternMatcher:
    """In-memory BM25 index over pattern summaries for fast matching."""

    def __init__(self):
        self._patterns: List[Pattern] = []
        # term -> list of (pattern_index, term_frequency)
        self._inverted: dict[str, list[tuple[int, int]]] = {}
        self._doc_lengths: dict[int, int] = {}
        self._avg_length: float = 0.0
        self._k1: float = 1.5
        self._b: float = 0.75

    def build(self, patterns: List[Pattern]):
        """Build BM25 index from pattern summaries + context_queries."""
        self._patterns = patterns
        self._inverted.clear()
        self._doc_lengths.clear()

        for idx, pattern in enumerate(patterns):
            # Combine summary + context_query + error_context for richer matching
            text = f"{pattern.summary} {pattern.context_query}"
            if pattern.error_context:
                text += f" {pattern.error_context}"

            tokens = _tokenize(text)
            self._doc_lengths[idx] = len(tokens)

            # Count term frequencies
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            for term, count in tf.items():
                if term not in self._inverted:
                    self._inverted[term] = []
                self._inverted[term].append((idx, count))

        n = len(patterns)
        self._avg_length = (
            sum(self._doc_lengths.values()) / n if n > 0 else 0.0
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Pattern, float]]:
        """
        Search patterns by BM25 similarity.

        Returns list of (Pattern, normalized_score) tuples.
        Scores are normalized to 0.0-1.0 range.
        """
        query_terms = _tokenize(query)
        if not query_terms or not self._patterns:
            return []

        n = len(self._patterns)
        scores: dict[int, float] = {}

        for term in query_terms:
            if term not in self._inverted:
                continue

            postings = self._inverted[term]
            df = len(postings)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            for idx, tf in postings:
                doc_len = self._doc_lengths.get(idx, 0)
                avg = max(self._avg_length, 1.0)
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * doc_len / avg)
                score = idf * numerator / denominator

                scores[idx] = scores.get(idx, 0.0) + score

        if not scores:
            return []

        # Normalize scores to 0.0-1.0
        max_score = max(scores.values())
        if max_score <= 0:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            (self._patterns[idx], score / max_score)
            for idx, score in ranked
        ]


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 matching."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ").replace(".", " ")
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if len(t) > 1]
