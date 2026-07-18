"""Project-wide structural decomposition.

Faithful port of wavescope-mcp `src/project.ts` discovery + project-level
important-position aggregation. Honors a root `.gitignore` (last-match-wins,
including `!` negation), skips well-known build/cache dirs, caps file count and
size, sniffs for binary files, and follows symlinks once without escaping root.

Synchronous (no concurrency pool) — the upstream parallelism is an I/O
optimization, not a behavioral property.
"""

from __future__ import annotations

import os
import re
import stat as statmod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .context import FileContext, ImportantPosition
from .language import configs

_CODE_EXTENSIONS = {ext for c in configs for ext in c.extensions}
_CODE_FILENAMES = {fn for c in configs for fn in c.filenames}

_SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build",
    "target", ".next", ".turbo", "coverage", ".pytest_cache", ".cache",
    ".idea", ".vscode", "vendor", "out", "obj", ".gradle", ".tox",
    ".mypy_cache", ".ruff_cache", "bower_components", ".serverless",
    ".terraform", ".eggs", "site-packages", ".yarn", ".parcel-cache",
    "__snapshots__",
}

MAX_FILE_BYTES = 2_000_000
MAX_FILES = 5_000
_BINARY_SNIFF_BYTES = 4096


# ─── .gitignore parser (minimal) ────────────────────────────

@dataclass
class _GitignoreRule:
    regex: "re.Pattern[str]"
    dir_only: bool
    negate: bool


def _compile_glob(pat: str) -> "re.Pattern[str]":
    # Leading `/` means root-relative; otherwise match at any depth.
    anchored = pat.startswith("/")
    body = pat[1:] if anchored else pat

    regex = ""
    i = 0
    while i < len(body):
        c = body[i]
        if c == "*" and i + 1 < len(body) and body[i + 1] == "*":
            has_trailing_slash = i + 2 < len(body) and body[i + 2] == "/"
            i += 3 if has_trailing_slash else 2
            regex += "(?:[^/]+/)*" if has_trailing_slash else ".*"
        elif c == "*":
            regex += "[^/]*"
            i += 1
        elif c == "?":
            regex += "[^/]"
            i += 1
        elif c == "[":
            # Character class: pass through to the regex (gitignore classes
            # are regex-compatible modulo `!` negation). Escaping the
            # brackets instead made `[ab].py` silently never match, so files
            # git ignores were indexed. Unterminated `[` falls back to a
            # literal bracket.
            close = body.find("]", i + 2)  # +2: "[]" would be empty
            if close != -1:
                cls = body[i + 1:close]
                if cls.startswith("!"):
                    cls = "^" + cls[1:]
                regex += "[" + cls + "]"
                i = close + 1
            else:
                regex += "\\["
                i += 1
        elif c in ".+(){}^$|\\" or c == "]":
            regex += "\\" + c
            i += 1
        else:
            regex += c
            i += 1

    prefix = "^" if anchored else "(^|/)"
    try:
        return re.compile(f"{prefix}{regex}(/|$)")
    except re.error:
        # A class body that isn't valid regex (e.g. `[z-a]`) degrades to a
        # fully-escaped literal match rather than crashing the walk.
        return re.compile(f"{prefix}{re.escape(body)}(/|$)")


def _load_gitignore(root: str) -> List[_GitignoreRule]:
    path = os.path.join(root, ".gitignore")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return []
    rules: List[_GitignoreRule] = []
    for line_raw in raw.split("\n"):
        line = line_raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        dir_only = line.endswith("/")
        pat_raw = line[1:] if negate else line
        pat = pat_raw[:-1] if dir_only else pat_raw
        rules.append(_GitignoreRule(_compile_glob(pat), dir_only, negate))
    return rules


def _is_ignored(rel_path: str, is_dir: bool, rules: List[_GitignoreRule]) -> bool:
    normalized = rel_path.replace(os.sep, "/")
    ignored = False
    for rule in rules:  # last match wins (negation re-includes)
        if rule.dir_only and not is_dir:
            continue
        matches = bool(rule.regex.search(normalized)) or (
            rule.dir_only and bool(rule.regex.search(normalized + "/"))
        )
        if matches:
            ignored = not rule.negate
    return ignored


# ─── File discovery ─────────────────────────────────────────

def _is_binary(full_path: str) -> bool:
    """Standard NUL-byte sniff over the first _BINARY_SNIFF_BYTES."""
    try:
        with open(full_path, "rb") as f:
            chunk = f.read(_BINARY_SNIFF_BYTES)
        return b"\x00" in chunk
    except OSError:
        return True


@dataclass
class ProjectFile:
    filename: str
    path: str
    context: FileContext


def _discover_files(root: str) -> Tuple[List[ProjectFile], bool]:
    results: List[ProjectFile] = []
    visited_real: set = set()
    try:
        root_real = os.path.realpath(root)
    except OSError as e:
        raise OSError(f"Cannot read directory: {root}") from e
    visited_real.add(root_real)

    gitignore = _load_gitignore(root)
    truncated = False
    pending: List[str] = []

    def walk(directory: str, depth: int = 0) -> None:
        nonlocal truncated
        if truncated:
            return
        # Depth guard: recursion is one Python frame per directory level, so
        # a pathologically deep tree (>1000 levels) would raise an uncaught
        # RecursionError out of decompose_project. No real project nests
        # this deep; treat it like the file-count cap.
        if depth > 64:
            truncated = True
            return
        try:
            entries = os.listdir(directory)
        except OSError:
            return
        for entry in entries:
            if truncated:
                return
            full_path = os.path.join(directory, entry)
            rel_path = os.path.relpath(full_path, root)
            try:
                st = os.lstat(full_path)
            except OSError:
                continue

            # Resolve symlinks: follow, but refuse ones escaping root. After
            # this, `st` reflects the resolved target for symlinks.
            if statmod.S_ISLNK(st.st_mode):
                try:
                    target = os.path.realpath(full_path)
                except OSError:
                    continue
                if not (target == root_real or target.startswith(root_real + os.sep)):
                    continue
                try:
                    st = os.stat(target)
                except OSError:
                    continue

            if statmod.S_ISDIR(st.st_mode):
                # Register pruned dirs' realpaths too: a symlink alias to a
                # skipped/ignored dir must not resurrect its contents under
                # the alias path.
                if entry in _SKIP_DIRS or _is_ignored(rel_path, True, gitignore):
                    try:
                        visited_real.add(os.path.realpath(full_path))
                    except OSError:
                        pass
                    continue
                try:
                    dir_real = os.path.realpath(full_path)
                except OSError:
                    continue
                if dir_real in visited_real:
                    continue
                visited_real.add(dir_real)
                walk(full_path, depth + 1)
            elif statmod.S_ISREG(st.st_mode):
                if _is_ignored(rel_path, False, gitignore):
                    continue
                _, ext = os.path.splitext(entry)
                ext = ext.lower()
                if ext not in _CODE_EXTENSIONS and entry not in _CODE_FILENAMES:
                    continue
                if st.st_size > MAX_FILE_BYTES:
                    continue
                try:
                    file_real = os.path.realpath(full_path)
                except OSError:
                    file_real = full_path
                if file_real in visited_real:
                    continue
                visited_real.add(file_real)
                if len(results) + len(pending) >= MAX_FILES:
                    truncated = True
                    return
                pending.append(full_path)

    walk(root)

    for full_path in pending:
        if _is_binary(full_path):
            continue
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        name = os.path.basename(full_path)
        results.append(ProjectFile(filename=name, path=full_path, context=FileContext(name, content)))

    return results, truncated


class ProjectIndex:
    """A loaded set of FileContexts under a root, with project-wide queries."""

    def __init__(self, root: str, files: List[ProjectFile], truncated: bool):
        self.root = root
        self.files = files
        self.truncated = truncated
        self._file_map: Dict[str, FileContext] = {
            os.path.relpath(f.path, root): f.context for f in files
        }

    @staticmethod
    def load(root: str) -> "ProjectIndex":
        resolved = os.path.abspath(root)
        files, truncated = _discover_files(resolved)
        return ProjectIndex(resolved, files, truncated)

    def get_file(self, rel_path: str) -> Optional[FileContext]:
        return self._file_map.get(rel_path)

    def list_files(self) -> List[str]:
        return [os.path.relpath(f.path, self.root) for f in self.files]

    def get_important_positions(
        self, min_coefficient: float = 0.3, limit: int = 20
    ) -> List[ImportantPosition]:
        """Project-wide important positions across all files, top `limit` by
        |coefficient|. Labels are suffixed with the relative file path."""
        top: List[ImportantPosition] = []
        for f in self.files:
            peaks = f.context.get_important_positions(min_coefficient, max(limit, 30))
            if not peaks:
                continue
            rel = os.path.relpath(f.path, self.root)
            for p in peaks:
                top.append(
                    ImportantPosition(
                        position=p.position,
                        coefficient=p.coefficient,
                        scale=p.scale,
                        label=f"{p.label} ({rel})",
                        filename=rel,
                    )
                )
            top.sort(key=lambda x: abs(x.coefficient), reverse=True)
            if len(top) > limit:
                del top[limit:]
        return top


def decompose_project(
    root: str, min_coefficient: float = 0.3, limit: int = 20
) -> List[ImportantPosition]:
    """Convenience entry point: load the project and return its top structural
    positions (the coarse "which files / regions matter" map for planning)."""
    return ProjectIndex.load(root).get_important_positions(min_coefficient, limit)


def decompose_file_map(
    file_map: Dict[str, str], min_coefficient: float = 0.3, limit: int = 20
) -> List[ImportantPosition]:
    """Like decompose_project but over in-memory {rel_path: content} rather than
    the filesystem. Used when the caller already has the project files in hand
    (e.g. the proxy sends them to v3-service, which has no project volume mount),
    so the coarse band can be computed without disk access. Mirrors
    ProjectIndex.get_important_positions: top `limit` positions by |coefficient|,
    labels suffixed with the relative path."""
    top: List[ImportantPosition] = []
    for i, (rel, content) in enumerate(file_map.items()):
        # Same bounds as the disk path: the pure-Python CWT is ~0.5s per
        # 1,000 lines, so an uncapped in-memory map would let one huge
        # supplied file (or thousands of entries) stall the planner for
        # minutes. MAX_FILE_BYTES/MAX_FILES mirror _discover_files.
        if i >= MAX_FILES:
            break
        if not content or len(content) > MAX_FILE_BYTES:
            continue
        ctx = FileContext(os.path.basename(rel), content)
        peaks = ctx.get_important_positions(min_coefficient, max(limit, 30))
        for p in peaks:
            top.append(
                ImportantPosition(
                    position=p.position,
                    coefficient=p.coefficient,
                    scale=p.scale,
                    label=f"{p.label} ({rel})",
                    filename=rel,
                )
            )
        top.sort(key=lambda x: abs(x.coefficient), reverse=True)
        if len(top) > limit:
            del top[limit:]
    return top
