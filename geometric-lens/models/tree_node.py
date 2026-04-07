"""Data models for the PageIndex tree index."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    REPOSITORY = "repository"
    DIRECTORY = "directory"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    BLOCK = "block"


class NodeMetadata(BaseModel):
    line_count: int = 0
    import_count: int = 0
    last_modified: Optional[float] = None
    language: str = "unknown"
    file_hash: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    retrieval_count: int = 0
    last_retrieved: Optional[float] = None


class TreeNode(BaseModel):
    node_id: str
    node_type: NodeType
    name: str
    path: str
    summary: str = ""
    children: List["TreeNode"] = Field(default_factory=list)
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)
    content: Optional[str] = None  # Raw code for leaf nodes

    def leaf_nodes(self) -> List["TreeNode"]:
        """Get all leaf nodes (functions, classes, blocks with no children)."""
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.leaf_nodes())
        return leaves

    def find_by_path(self, path: str) -> Optional["TreeNode"]:
        """Find a node by its path."""
        if self.path == path:
            return self
        for child in self.children:
            found = child.find_by_path(path)
            if found:
                return found
        return None

    def node_count(self) -> int:
        """Total nodes in this subtree."""
        return 1 + sum(c.node_count() for c in self.children)

    def depth(self) -> int:
        """Max depth of this subtree."""
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)


class TreeIndex(BaseModel):
    """The full persisted tree index for a project."""
    project_id: str
    root: TreeNode
    version: int = 1
    created_at: Optional[str] = None
    file_hashes: Dict[str, str] = Field(default_factory=dict)
