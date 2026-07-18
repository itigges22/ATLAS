"""Relative-link and anchor integrity for every Markdown file.

Every `[text](target)` link with a relative-file target must point at a
file that exists, and every `#fragment` (same-file or cross-file) must
match a heading anchor — computed with GitHub's slugger (lowercase,
punctuation stripped, spaces to hyphens, `-1`/`-2` suffixes for
duplicate headings) — or an explicit `<a id="..."></a>` anchor.
External http(s)/mailto links are never checked; this stays hermetic.
"""

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO = Path(__file__).resolve().parents[2]

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__",
             ".pytest_cache"}

# Known-unfixable links, as "relative/path.md:target". Each entry needs
# a comment explaining why the link cannot be fixed.
SKIP: set = set()

LINK_RE = re.compile(r"\[[^\]]*\]\(\s*(<[^>]*>|[^()\s]+(?:\([^()\s]*\))?)\s*(?:\"[^\"]*\")?\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
EXPLICIT_ANCHOR_RE = re.compile(r'<a\s+(?:id|name)="([^"]+)"')


def _git_tracked_md():
    """Git-tracked markdown only: local-only files (gitignored notes,
    audit scratch) don't exist in CI checkouts and aren't part of the
    published docs contract."""
    out = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO,
        capture_output=True, text=True, check=True)
    return {REPO / line for line in out.stdout.splitlines() if line}


def _markdown_files():
    tracked = _git_tracked_md()
    for path in sorted(tracked):
        if SKIP_DIRS.intersection(path.parts):
            continue
        if path.is_file():
            yield path


def _strip_fences(text: str) -> str:
    """Blank out fenced code blocks, preserving line numbers."""
    out, in_fence, fence = [], False, ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence, fence = True, stripped[:3]
            out.append("")
        elif in_fence:
            if stripped.startswith(fence):
                in_fence = False
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def _slug(heading: str) -> str:
    """GitHub's heading slugger."""
    text = heading.strip()
    # Rendered text only: drop link targets and image/emphasis/code markers.
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.lower()
    text = re.sub(r"[^\w\- ]", "", text)  # keep word chars (incl. CJK), - _ and space
    return text.replace(" ", "-")  # no collapsing of consecutive hyphens


def _anchors(path: Path) -> set:
    raw = path.read_text(errors="replace")
    text = _strip_fences(raw)
    anchors, seen = set(), {}
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if not m:
            continue
        base = _slug(m.group(2))
        n = seen.get(base, 0)
        seen[base] = n + 1
        anchors.add(base if n == 0 else f"{base}-{n}")
    anchors.update(EXPLICIT_ANCHOR_RE.findall(raw))
    return anchors


def _links(path: Path):
    """Yield (line_number, target) for every inline link."""
    for lineno, line in enumerate(_strip_fences(path.read_text(errors="replace")).splitlines(), 1):
        line = re.sub(r"`[^`]*`", "", line)  # links inside inline code don't render
        for m in LINK_RE.finditer(line):
            target = m.group(1).strip("<>").strip()
            if target:
                yield lineno, target


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "ftp://"))


def _check_file(path: Path, anchor_cache: dict) -> list:
    failures = []
    rel = path.relative_to(REPO)
    for lineno, target in _links(path):
        if _is_external(target) or f"{rel}:{target}" in SKIP:
            continue
        split = urlsplit(target)
        if split.scheme or split.netloc:
            continue
        file_part, fragment = unquote(split.path), unquote(split.fragment)
        dest = path if not file_part else (path.parent / file_part).resolve()
        if not dest.exists():
            failures.append(f"{rel}:{lineno} → {target} (file not found)")
            continue
        if fragment and dest.suffix == ".md":
            if dest not in anchor_cache:
                anchor_cache[dest] = _anchors(dest)
            anchors = anchor_cache[dest]
            if fragment not in anchors and fragment.lower() not in anchors:
                failures.append(f"{rel}:{lineno} → {target} (no such anchor)")
    return failures


def test_relative_links_and_anchors_resolve():
    anchor_cache, failures = {}, []
    for path in _markdown_files():
        failures.extend(_check_file(path, anchor_cache))
    assert not failures, (
        "broken Markdown links/anchors:\n  " + "\n  ".join(failures))
