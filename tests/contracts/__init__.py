"""Contract-test helpers.

Go functions and constants move between files of a package during
reorganizations, so contract tests locate sources by a content marker,
never by filename.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def go_source(component: str, marker: str) -> str:
    """Source of the non-test .go file under REPO/component containing marker."""
    for go in sorted((REPO / component).glob("*.go")):
        if go.name.endswith("_test.go"):
            continue
        src = go.read_text()
        if marker in src:
            return src
    raise AssertionError(f"{marker!r} not found in any {component}/*.go")
