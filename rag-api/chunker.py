import re
from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib


@dataclass
class Chunk:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str
    chunk_type: str = "lines"
    symbol_name: str = ""


# Language detection based on file extension
LANGUAGE_MAP = {
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
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".md": "markdown",
    ".txt": "text",
}


def detect_language(file_path: str) -> str:
    """Detect language from file extension."""
    for ext, lang in LANGUAGE_MAP.items():
        if file_path.endswith(ext):
            return lang
    return "text"


def generate_chunk_id(project_id: str, file_path: str, start_line: int) -> str:
    """Generate a unique chunk ID."""
    content = f"{project_id}:{file_path}:{start_line}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def chunk_file_by_lines(
    project_id: str,
    file_path: str,
    content: str,
    max_lines: int = 100,
    overlap: int = 20
) -> List[Chunk]:
    """
    Chunk a file by lines with overlap.
    This is a simple but effective chunking strategy.
    """
    lines = content.split("\n")
    chunks = []
    language = detect_language(file_path)

    if len(lines) <= max_lines:
        # File is small enough to be a single chunk
        chunk_id = generate_chunk_id(project_id, file_path, 1)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            content=content,
            language=language,
            chunk_type="file"
        ))
    else:
        # Split into overlapping chunks
        start = 0
        while start < len(lines):
            end = min(start + max_lines, len(lines))
            chunk_content = "\n".join(lines[start:end])
            chunk_id = generate_chunk_id(project_id, file_path, start + 1)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                content=chunk_content,
                language=language,
                chunk_type="lines"
            ))

            # Move to next chunk with overlap
            start = end - overlap
            if start >= len(lines) - overlap:
                break

    return chunks


def chunk_code_by_functions(
    project_id: str,
    file_path: str,
    content: str,
    max_lines: int = 100
) -> List[Chunk]:
    """
    Attempt to chunk code by functions/classes.
    Falls back to line-based chunking if no functions found.
    """
    language = detect_language(file_path)
    chunks = []

    # Function/class patterns by language
    patterns = {
        "python": [
            r"^(class\s+\w+.*?:)",
            r"^(def\s+\w+.*?:)",
            r"^(async\s+def\s+\w+.*?:)",
        ],
        "javascript": [
            r"^(function\s+\w+\s*\()",
            r"^(const\s+\w+\s*=\s*(?:async\s*)?\()",
            r"^(class\s+\w+)",
            r"^(export\s+(?:default\s+)?(?:async\s+)?function)",
        ],
        "typescript": [
            r"^(function\s+\w+\s*\()",
            r"^(const\s+\w+\s*=\s*(?:async\s*)?\()",
            r"^(class\s+\w+)",
            r"^(export\s+(?:default\s+)?(?:async\s+)?function)",
            r"^(interface\s+\w+)",
            r"^(type\s+\w+\s*=)",
        ],
        "go": [
            r"^(func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+\s*\()",
            r"^(type\s+\w+\s+struct)",
            r"^(type\s+\w+\s+interface)",
        ],
    }

    if language not in patterns:
        # Fall back to line-based chunking
        return chunk_file_by_lines(project_id, file_path, content, max_lines)

    lines = content.split("\n")

    # Find function/class boundaries
    boundaries = []
    for i, line in enumerate(lines):
        for pattern in patterns[language]:
            if re.match(pattern, line.strip()):
                boundaries.append(i)
                break

    if not boundaries:
        # No functions found, fall back to line-based
        return chunk_file_by_lines(project_id, file_path, content, max_lines)

    # Add file end as final boundary
    boundaries.append(len(lines))

    # Create chunks from boundaries
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # If chunk is too large, split it
        if end - start > max_lines:
            sub_chunks = chunk_file_by_lines(
                project_id, file_path,
                "\n".join(lines[start:end]),
                max_lines, overlap=20
            )
            # Adjust line numbers
            for sc in sub_chunks:
                sc.start_line += start
                sc.end_line += start
            chunks.extend(sub_chunks)
        else:
            chunk_content = "\n".join(lines[start:end])
            chunk_id = generate_chunk_id(project_id, file_path, start + 1)

            # Try to extract symbol name
            symbol_name = ""
            first_line = lines[start].strip()
            match = re.search(r"(?:class|def|func|function|const|type|interface)\s+(\w+)", first_line)
            if match:
                symbol_name = match.group(1)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                content=chunk_content,
                language=language,
                chunk_type="function" if "def" in first_line or "func" in first_line else "class",
                symbol_name=symbol_name
            ))

    return chunks if chunks else chunk_file_by_lines(project_id, file_path, content, max_lines)


def chunk_project_files(
    project_id: str,
    files: List[Dict[str, str]],
    max_chunk_lines: int = 100,
    overlap_lines: int = 20,
    use_ast_chunking: bool = True
) -> List[Chunk]:
    """
    Chunk all files in a project.

    Args:
        project_id: Unique project identifier
        files: List of {"path": str, "content": str}
        max_chunk_lines: Maximum lines per chunk
        overlap_lines: Overlap between chunks
        use_ast_chunking: Try to chunk by functions/classes

    Returns:
        List of Chunk objects
    """
    all_chunks = []

    for file_info in files:
        file_path = file_info["path"]
        content = file_info["content"]

        # Skip empty files
        if not content.strip():
            continue

        if use_ast_chunking:
            chunks = chunk_code_by_functions(
                project_id, file_path, content, max_chunk_lines
            )
        else:
            chunks = chunk_file_by_lines(
                project_id, file_path, content, max_chunk_lines, overlap_lines
            )

        all_chunks.extend(chunks)

    return all_chunks


def count_lines(files: List[Dict[str, str]]) -> int:
    """Count total lines of code across all files."""
    return sum(len(f["content"].split("\n")) for f in files)


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough estimate: 4 chars per token)."""
    return len(text) // 4
