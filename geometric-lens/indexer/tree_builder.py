"""Build a unified navigation tree from filesystem + AST structure."""

import hashlib
import logging
import os
from typing import List, Dict, Optional

from models.tree_node import TreeNode, TreeIndex, NodeType, NodeMetadata
from indexer.ast_parser import parse_python_file, ASTNode

logger = logging.getLogger(__name__)

# File extensions we index
INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs",
    ".java", ".c", ".cpp", ".h", ".hpp", ".rb",
}

# Directories to skip
SKIP_DIRS = {
    "__pycache__", ".git", ".svn", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "*.egg-info",
}

# Files to skip
SKIP_FILES = {
    ".gitignore", ".dockerignore", "LICENSE", "Makefile",
}


def build_tree_from_files(
    project_id: str,
    files: List[Dict[str, str]],
    project_name: str = "",
) -> TreeIndex:
    """
    Build a unified navigation tree from a list of project files.

    Args:
        project_id: Unique project identifier
        files: List of {"path": str, "content": str}
        project_name: Human-readable project name

    Returns:
        TreeIndex with complete tree structure
    """
    # Root node
    root = TreeNode(
        node_id=_node_id(project_id, "/"),
        node_type=NodeType.REPOSITORY,
        name=project_name or project_id,
        path="/",
        metadata=NodeMetadata(
            line_count=sum(len(f["content"].split("\n")) for f in files),
        ),
    )

    file_hashes: Dict[str, str] = {}

    # Group files by directory
    dir_map: Dict[str, List[Dict[str, str]]] = {}
    for f in files:
        path = f["path"]
        dir_path = os.path.dirname(path) or "/"
        if dir_path not in dir_map:
            dir_map[dir_path] = []
        dir_map[dir_path].append(f)
        file_hashes[path] = hashlib.sha256(f["content"].encode()).hexdigest()

    # Build directory tree structure
    dir_nodes: Dict[str, TreeNode] = {"/": root}

    # Sort directories to ensure parents are created before children
    all_dirs = sorted(dir_map.keys())

    for dir_path in all_dirs:
        _ensure_dir_node(dir_path, dir_nodes, project_id)

    # Process each file
    for f in files:
        path = f["path"]
        content = f["content"]
        dir_path = os.path.dirname(path) or "/"

        ext = os.path.splitext(path)[1]
        language = _detect_language(ext)

        # Create file node
        file_node = TreeNode(
            node_id=_node_id(project_id, path),
            node_type=NodeType.FILE,
            name=os.path.basename(path),
            path=path,
            metadata=NodeMetadata(
                line_count=len(content.split("\n")),
                language=language,
                file_hash=file_hashes.get(path),
            ),
            content=content,
        )

        # Parse AST for supported languages
        if ext == ".py":
            ast_nodes = parse_python_file(content, path)
            _attach_ast_children(file_node, ast_nodes, project_id, path, content)
        # TODO: Add JS/TS parsing when tree-sitter-javascript is added

        # Count imports for metadata
        if ext == ".py":
            import_count = sum(
                1 for line in content.split("\n")
                if line.strip().startswith(("import ", "from "))
            )
            file_node.metadata.import_count = import_count

        # Add to parent directory
        parent = dir_nodes.get(dir_path, root)
        parent.children.append(file_node)

    return TreeIndex(
        project_id=project_id,
        root=root,
        file_hashes=file_hashes,
    )


def _ensure_dir_node(
    dir_path: str,
    dir_nodes: Dict[str, TreeNode],
    project_id: str,
) -> TreeNode:
    """Ensure a directory node exists, creating parent directories as needed."""
    if dir_path in dir_nodes:
        return dir_nodes[dir_path]

    # Create parent first
    parent_path = os.path.dirname(dir_path) or "/"
    if parent_path == dir_path:
        parent_path = "/"
    parent = _ensure_dir_node(parent_path, dir_nodes, project_id)

    # Create this directory node
    node = TreeNode(
        node_id=_node_id(project_id, dir_path),
        node_type=NodeType.DIRECTORY,
        name=os.path.basename(dir_path) or dir_path,
        path=dir_path,
    )
    dir_nodes[dir_path] = node
    parent.children.append(node)
    return node


def _attach_ast_children(
    file_node: TreeNode,
    ast_nodes: List[ASTNode],
    project_id: str,
    file_path: str,
    full_content: str,
):
    """Convert AST nodes to TreeNode children and attach to file node."""
    for ast_node in ast_nodes:
        # Skip imports — they're captured in metadata
        if ast_node.node_type == "import":
            continue

        child = _ast_to_tree_node(ast_node, project_id, file_path, full_content)
        if child:
            file_node.children.append(child)

    # If file has no AST children (or only imports), keep the file as a leaf
    # with its full content available


def _ast_to_tree_node(
    ast_node: ASTNode,
    project_id: str,
    file_path: str,
    full_content: str,
) -> Optional[TreeNode]:
    """Convert an ASTNode to a TreeNode."""
    node_type_map = {
        "class": NodeType.CLASS,
        "function": NodeType.FUNCTION,
        "block": NodeType.BLOCK,
    }
    nt = node_type_map.get(ast_node.node_type)
    if nt is None:
        return None

    ast_path = f"{file_path}::{ast_node.name}"

    node = TreeNode(
        node_id=_node_id(project_id, ast_path),
        node_type=nt,
        name=ast_node.name,
        path=ast_path,
        metadata=NodeMetadata(
            line_count=ast_node.end_line - ast_node.start_line + 1,
            start_line=ast_node.start_line,
            end_line=ast_node.end_line,
        ),
        content=ast_node.content,
    )

    # Add method children for classes
    for child_ast in ast_node.children:
        child = _ast_to_tree_node(child_ast, project_id, file_path, full_content)
        if child:
            node.children.append(child)

    return node


def _node_id(project_id: str, path: str) -> str:
    """Generate a deterministic node ID."""
    return hashlib.sha256(f"{project_id}:{path}".encode()).hexdigest()[:16]


def _detect_language(ext: str) -> str:
    """Detect language from file extension."""
    lang_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
    }
    return lang_map.get(ext, "unknown")
