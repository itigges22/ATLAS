"""Persist and load tree index + BM25 index to/from disk."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

from models.tree_node import TreeIndex
from indexer.bm25_index import BM25Index

logger = logging.getLogger(__name__)

# Default base path (same as storage.py)
INDEX_BASE_PATH = "/data/projects"


def save_index(
    project_id: str,
    tree_index: TreeIndex,
    bm25_index: BM25Index,
    base_path: str = INDEX_BASE_PATH,
):
    """Save tree index and BM25 index to disk."""
    project_dir = os.path.join(base_path, project_id)
    os.makedirs(project_dir, exist_ok=True)

    tree_index.created_at = datetime.now(timezone.utc).isoformat()

    # Save tree index
    tree_path = os.path.join(project_dir, "tree_index.json")
    with open(tree_path, "w") as f:
        json.dump(tree_index.model_dump(), f)

    # Save BM25 index
    bm25_path = os.path.join(project_dir, "bm25_index.json")
    with open(bm25_path, "w") as f:
        json.dump(bm25_index.to_dict(), f)

    logger.info(
        f"Saved indexes for {project_id}: "
        f"tree={os.path.getsize(tree_path)} bytes, "
        f"bm25={os.path.getsize(bm25_path)} bytes"
    )


def load_index(
    project_id: str,
    base_path: str = INDEX_BASE_PATH,
) -> Optional[Tuple[TreeIndex, BM25Index]]:
    """Load tree index and BM25 index from disk."""
    project_dir = os.path.join(base_path, project_id)

    tree_path = os.path.join(project_dir, "tree_index.json")
    bm25_path = os.path.join(project_dir, "bm25_index.json")

    if not os.path.exists(tree_path) or not os.path.exists(bm25_path):
        return None

    try:
        with open(tree_path) as f:
            tree_data = json.load(f)
        tree_index = TreeIndex.model_validate(tree_data)

        with open(bm25_path) as f:
            bm25_data = json.load(f)
        bm25_index = BM25Index.from_dict(bm25_data)

        logger.info(
            f"Loaded indexes for {project_id}: "
            f"{tree_index.root.node_count()} nodes, "
            f"{bm25_index.num_docs} BM25 docs"
        )
        return tree_index, bm25_index

    except Exception as e:
        logger.error(f"Failed to load indexes for {project_id}: {e}")
        return None


def index_exists(
    project_id: str,
    base_path: str = INDEX_BASE_PATH,
) -> bool:
    """Check if a PageIndex exists for this project."""
    project_dir = os.path.join(base_path, project_id)
    tree_path = os.path.join(project_dir, "tree_index.json")
    return os.path.exists(tree_path)


def delete_index(
    project_id: str,
    base_path: str = INDEX_BASE_PATH,
):
    """Delete persisted indexes for a project."""
    project_dir = os.path.join(base_path, project_id)
    for fname in ("tree_index.json", "bm25_index.json"):
        fpath = os.path.join(project_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            logger.info(f"Deleted {fpath}")
