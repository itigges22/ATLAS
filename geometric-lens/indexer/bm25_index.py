"""BM25 inverted index for exact identifier lookup."""

import json
import math
import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict

from models.tree_node import TreeNode, TreeIndex

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """A single BM25 search result."""
    node_id: str
    path: str
    name: str
    score: float
    content: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None


@dataclass
class BM25Index:
    """Inverted index with BM25 scoring for identifier lookup."""

    # term -> list of (node_id, term_frequency)
    inverted_index: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    # node_id -> total terms in document
    doc_lengths: Dict[str, int] = field(default_factory=dict)
    # node_id -> (path, name, content, start_line, end_line)
    doc_metadata: Dict[str, Tuple[str, str, str, Optional[int], Optional[int]]] = field(
        default_factory=dict
    )
    # Total number of documents
    num_docs: int = 0
    # Average document length
    avg_doc_length: float = 0.0

    # BM25 parameters
    k1: float = 1.5
    b: float = 0.75

    def build_from_tree(self, tree_index: TreeIndex):
        """Build the BM25 index from a tree index."""
        self.inverted_index.clear()
        self.doc_lengths.clear()
        self.doc_metadata.clear()

        self._index_node(tree_index.root)

        self.num_docs = len(self.doc_lengths)
        if self.num_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.num_docs
        else:
            self.avg_doc_length = 0.0

        logger.info(
            f"BM25 index built: {self.num_docs} docs, "
            f"{len(self.inverted_index)} unique terms"
        )

    def _index_node(self, node: TreeNode):
        """Recursively index a node and its children."""
        # Index leaf-level and code-level nodes (functions, classes, blocks)
        if node.node_type.value in ("function", "class", "block", "file"):
            terms = _tokenize(node.name)

            # Also tokenize content for richer matching
            if node.content:
                terms.extend(_tokenize_code(node.content))

            if terms:
                # Count term frequencies
                tf: Dict[str, int] = {}
                for term in terms:
                    tf[term] = tf.get(term, 0) + 1

                # Store in inverted index
                for term, count in tf.items():
                    if term not in self.inverted_index:
                        self.inverted_index[term] = []
                    self.inverted_index[term].append((node.node_id, count))

                self.doc_lengths[node.node_id] = len(terms)
                self.doc_metadata[node.node_id] = (
                    node.path,
                    node.name,
                    node.content or "",
                    node.metadata.start_line,
                    node.metadata.end_line,
                )

        # Recurse into children
        for child in node.children:
            self._index_node(child)

    def search(self, query: str, top_k: int = 20) -> List[BM25Result]:
        """Search the BM25 index with a query string."""
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        # Score each document
        scores: Dict[str, float] = {}

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            postings = self.inverted_index[term]
            # IDF: log((N - n + 0.5) / (n + 0.5) + 1)
            df = len(postings)
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf in postings:
                doc_len = self.doc_lengths.get(doc_id, 0)
                # BM25 score
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1)
                )
                score = idf * numerator / denominator

                scores[doc_id] = scores.get(doc_id, 0.0) + score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in ranked:
            meta = self.doc_metadata.get(doc_id)
            if meta:
                path, name, content, start_line, end_line = meta
                results.append(BM25Result(
                    node_id=doc_id,
                    path=path,
                    name=name,
                    score=score,
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                ))

        return results

    def has_exact_match(self, query: str) -> bool:
        """Check if any query term has an exact match in the index."""
        terms = _tokenize(query)
        for term in terms:
            if term in self.inverted_index:
                return True
        return False

    def to_dict(self) -> dict:
        """Serialize the index to a dictionary."""
        return {
            "inverted_index": {
                k: v for k, v in self.inverted_index.items()
            },
            "doc_lengths": self.doc_lengths,
            "doc_metadata": self.doc_metadata,
            "num_docs": self.num_docs,
            "avg_doc_length": self.avg_doc_length,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        """Deserialize the index from a dictionary."""
        idx = cls()
        idx.inverted_index = {
            k: [tuple(item) for item in v]
            for k, v in data.get("inverted_index", {}).items()
        }
        idx.doc_lengths = data.get("doc_lengths", {})
        # Convert metadata lists back to tuples
        idx.doc_metadata = {
            k: tuple(v) for k, v in data.get("doc_metadata", {}).items()
        }
        idx.num_docs = data.get("num_docs", 0)
        idx.avg_doc_length = data.get("avg_doc_length", 0.0)
        return idx


def _tokenize(text: str) -> List[str]:
    """Tokenize text into terms suitable for BM25 matching."""
    # Split camelCase and snake_case
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ").replace(".", " ")
    text = text.lower()

    # Split on non-alphanumeric
    tokens = re.findall(r"[a-z0-9]+", text)

    # Filter very short tokens
    return [t for t in tokens if len(t) > 1]


def _tokenize_code(code: str) -> List[str]:
    """Extract identifiers from code for BM25 indexing."""
    # Find Python-style identifiers
    identifiers = re.findall(r'\b([a-zA-Z_]\w{2,})\b', code)

    tokens = []
    keywords = {
        "and", "as", "assert", "async", "await", "break", "class", "continue",
        "def", "del", "elif", "else", "except", "finally", "for", "from",
        "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
        "or", "pass", "raise", "return", "try", "while", "with", "yield",
        "True", "False", "None", "self", "cls",
    }

    for ident in identifiers:
        if ident in keywords:
            continue
        # Split camelCase
        parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", ident)
        parts = parts.replace("_", " ").lower()
        subtokens = re.findall(r"[a-z0-9]+", parts)
        tokens.extend(t for t in subtokens if len(t) > 1)
        # Also add the full identifier as a token
        tokens.append(ident.lower())

    return tokens
