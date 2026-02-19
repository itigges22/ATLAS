"""AST parsing using tree-sitter for Python source files."""

import logging
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    PY_LANGUAGE = Language(tspython.language())
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    _TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter not available, AST parsing disabled")


@dataclass
class ASTNode:
    """A parsed AST node with type, name, line range, and parent reference."""
    node_type: str  # "class", "function", "block", "import"
    name: str
    start_line: int  # 1-indexed
    end_line: int    # 1-indexed, inclusive
    content: str
    children: List["ASTNode"] = field(default_factory=list)
    parent_name: Optional[str] = None
    decorators: List[str] = field(default_factory=list)


def parse_python_file(source: str, file_path: str = "") -> List[ASTNode]:
    """
    Parse a Python source file into a list of top-level AST nodes.

    Returns a flat list of top-level definitions (classes, functions, blocks).
    Classes contain their methods as children.
    """
    if not _TREE_SITTER_AVAILABLE:
        return _fallback_parse(source, file_path)

    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))
    root = tree.root_node

    nodes = []
    lines = source.split("\n")

    for child in root.children:
        node = _extract_node(child, lines, parent_name=None)
        if node:
            nodes.append(node)

    return nodes


def _extract_node(
    ts_node,
    lines: List[str],
    parent_name: Optional[str] = None
) -> Optional[ASTNode]:
    """Extract an ASTNode from a tree-sitter node."""
    node_type_str = ts_node.type

    if node_type_str == "class_definition":
        return _extract_class(ts_node, lines, parent_name)
    elif node_type_str in ("function_definition", "decorated_definition"):
        return _extract_function(ts_node, lines, parent_name)
    elif node_type_str in ("import_statement", "import_from_statement"):
        return _extract_import(ts_node, lines)
    elif node_type_str == "decorated_definition":
        # Get the inner definition
        for child in ts_node.children:
            if child.type in ("function_definition", "class_definition"):
                return _extract_node(child, lines, parent_name)
    elif node_type_str in ("expression_statement", "assignment"):
        # Top-level assignments / constants
        start = ts_node.start_point[0] + 1
        end = ts_node.end_point[0] + 1
        content = "\n".join(lines[start - 1:end])
        name = _extract_assignment_name(ts_node)
        if name:
            return ASTNode(
                node_type="block",
                name=name,
                start_line=start,
                end_line=end,
                content=content,
                parent_name=parent_name,
            )

    return None


def _extract_class(ts_node, lines: List[str], parent_name: Optional[str]) -> ASTNode:
    """Extract a class definition with its methods."""
    start = ts_node.start_point[0] + 1
    end = ts_node.end_point[0] + 1
    content = "\n".join(lines[start - 1:end])

    name = ""
    for child in ts_node.children:
        if child.type == "identifier":
            name = child.text.decode("utf-8")
            break

    # Extract methods
    methods = []
    body = None
    for child in ts_node.children:
        if child.type == "block":
            body = child
            break

    if body:
        for child in body.children:
            if child.type in ("function_definition", "decorated_definition"):
                method = _extract_function(child, lines, parent_name=name)
                if method:
                    methods.append(method)

    decorators = _extract_decorators(ts_node)

    return ASTNode(
        node_type="class",
        name=name,
        start_line=start,
        end_line=end,
        content=content,
        children=methods,
        parent_name=parent_name,
        decorators=decorators,
    )


def _extract_function(
    ts_node, lines: List[str], parent_name: Optional[str]
) -> Optional[ASTNode]:
    """Extract a function definition."""
    # Handle decorated definitions
    actual_node = ts_node
    decorators = []
    if ts_node.type == "decorated_definition":
        decorators = _extract_decorators(ts_node)
        for child in ts_node.children:
            if child.type in ("function_definition", "class_definition"):
                actual_node = child
                break
        # If inner node is a class, delegate
        if actual_node.type == "class_definition":
            cls = _extract_class(ts_node, lines, parent_name)
            cls.decorators = decorators
            return cls

    start = ts_node.start_point[0] + 1
    end = ts_node.end_point[0] + 1
    content = "\n".join(lines[start - 1:end])

    name = ""
    for child in actual_node.children:
        if child.type == "identifier":
            name = child.text.decode("utf-8")
            break

    return ASTNode(
        node_type="function",
        name=name,
        start_line=start,
        end_line=end,
        content=content,
        parent_name=parent_name,
        decorators=decorators,
    )


def _extract_import(ts_node, lines: List[str]) -> ASTNode:
    """Extract an import statement."""
    start = ts_node.start_point[0] + 1
    end = ts_node.end_point[0] + 1
    content = "\n".join(lines[start - 1:end])

    # Get module name
    name = content.strip()

    return ASTNode(
        node_type="import",
        name=name,
        start_line=start,
        end_line=end,
        content=content,
    )


def _extract_decorators(ts_node) -> List[str]:
    """Extract decorator names from a decorated definition or class/function."""
    decorators = []
    for child in ts_node.children:
        if child.type == "decorator":
            text = child.text.decode("utf-8").strip()
            decorators.append(text)
    return decorators


def _extract_assignment_name(ts_node) -> Optional[str]:
    """Extract the name from an assignment node."""
    if ts_node.type == "expression_statement":
        for child in ts_node.children:
            if child.type == "assignment":
                return _extract_assignment_name(child)
    elif ts_node.type == "assignment":
        for child in ts_node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
            elif child.type == "pattern_list":
                # Tuple unpacking: a, b = ...
                names = []
                for sub in child.children:
                    if sub.type == "identifier":
                        names.append(sub.text.decode("utf-8"))
                return ", ".join(names) if names else None
    return None


def extract_identifiers(source: str) -> List[str]:
    """
    Extract all meaningful identifiers from Python source code.
    Used for BM25 index construction.
    """
    if not _TREE_SITTER_AVAILABLE:
        return _fallback_extract_identifiers(source)

    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))

    identifiers = set()
    _walk_for_identifiers(tree.root_node, identifiers)
    return list(identifiers)


def _walk_for_identifiers(node, identifiers: set):
    """Recursively walk the tree collecting identifiers."""
    if node.type == "identifier":
        name = node.text.decode("utf-8")
        # Skip single-character and common builtins
        if len(name) > 1 and name not in (
            "self", "cls", "True", "False", "None",
            "print", "len", "range", "str", "int", "float",
            "list", "dict", "set", "tuple", "type", "super",
            "isinstance", "issubclass", "hasattr", "getattr",
            "setattr", "delattr", "property", "staticmethod",
            "classmethod", "abstractmethod",
        ):
            identifiers.add(name)
    elif node.type == "string":
        # Skip string contents
        return

    for child in node.children:
        _walk_for_identifiers(child, identifiers)


# Fallback parsers when tree-sitter is not available

def _fallback_parse(source: str, file_path: str) -> List[ASTNode]:
    """Regex-based fallback parser for when tree-sitter is unavailable."""
    import re

    nodes = []
    lines = source.split("\n")

    class_pattern = re.compile(r"^class\s+(\w+)")
    func_pattern = re.compile(r"^(?:async\s+)?def\s+(\w+)")
    import_pattern = re.compile(r"^(?:import|from)\s+")

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        match = class_pattern.match(stripped)
        if match:
            name = match.group(1)
            end = _find_block_end(lines, i)
            content = "\n".join(lines[i:end + 1])
            nodes.append(ASTNode(
                node_type="class",
                name=name,
                start_line=i + 1,
                end_line=end + 1,
                content=content,
            ))
            i = end + 1
            continue

        match = func_pattern.match(stripped)
        if match:
            name = match.group(1)
            end = _find_block_end(lines, i)
            content = "\n".join(lines[i:end + 1])
            nodes.append(ASTNode(
                node_type="function",
                name=name,
                start_line=i + 1,
                end_line=end + 1,
                content=content,
            ))
            i = end + 1
            continue

        if import_pattern.match(stripped):
            nodes.append(ASTNode(
                node_type="import",
                name=stripped,
                start_line=i + 1,
                end_line=i + 1,
                content=stripped,
            ))

        i += 1

    return nodes


def _find_block_end(lines: List[str], start: int) -> int:
    """Find the end of a Python block by indentation."""
    if start >= len(lines):
        return start

    # Get the indentation of the definition line
    first_line = lines[start]
    base_indent = len(first_line) - len(first_line.lstrip())

    end = start
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip():  # blank line
            end = i
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent:
            break
        end = i

    return end


def _fallback_extract_identifiers(source: str) -> List[str]:
    """Regex-based identifier extraction fallback."""
    import re
    # Match Python identifiers
    matches = re.findall(r'\b([a-zA-Z_]\w{2,})\b', source)
    # Filter out keywords
    keywords = {
        "and", "as", "assert", "async", "await", "break", "class", "continue",
        "def", "del", "elif", "else", "except", "finally", "for", "from",
        "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
        "or", "pass", "raise", "return", "try", "while", "with", "yield",
        "True", "False", "None", "self", "cls",
    }
    return list(set(m for m in matches if m not in keywords))
