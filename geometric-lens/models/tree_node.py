"""Data models for the unified code navigation tree."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    REPOSITORY = "repository"
    DIRECTORY = "directory"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    BLOCK = "block"


class NodeMetadata(BaseModel):
    line_count: Optional[int] = None
    language: Optional[str] = None
    file_hash: Optional[str] = None
    import_count: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class TreeNode(BaseModel):
    node_id: str
    node_type: NodeType
    name: str
    path: str
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)
    children: List[TreeNode] = Field(default_factory=list)
    content: Optional[str] = None
    summary: Optional[str] = None

    def node_count(self) -> int:
        """Return total number of nodes in this subtree, including self."""
        return 1 + sum(child.node_count() for child in self.children)


# Required for self-referential model
TreeNode.model_rebuild()


class TreeIndex(BaseModel):
    project_id: str
    root: TreeNode
    file_hashes: Dict[str, str] = Field(default_factory=dict)
    created_at: Optional[str] = None
