"""Version-string parity across every surface that states a release.

pyproject.toml is the single source of truth. Each documentation or
code surface that repeats the version (package __version__, README
badges, ARCHITECTURE header, CHANGELOG top entry, SUPPORT_MATRIX
applies-to line, translated README badges) must agree, so a release
bump can't leave a stale number behind. Parsing only.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _truth() -> str:
    text = (REPO / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', text, re.M)
    assert m, "pyproject.toml must declare version = \"X.Y.Z\""
    return m.group(1)


def test_package_dunder_version_matches():
    version = _truth()
    text = (REPO / "atlas" / "__init__.py").read_text()
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    assert m, "atlas/__init__.py must declare __version__"
    assert m.group(1) == version, (
        f"atlas/__init__.py __version__ is {m.group(1)!r}, "
        f"pyproject.toml says {version!r}")


def test_readme_badge_matches():
    version = _truth()
    text = (REPO / "README.md").read_text()
    assert f"version-V{version}-" in text, (
        f"README.md version badge must contain 'version-V{version}-' "
        f"(pyproject.toml says {version})")


def test_architecture_header_matches():
    version = _truth()
    text = (REPO / "docs" / "ARCHITECTURE.md").read_text()
    assert f"ATLAS V{version}" in text, (
        f"docs/ARCHITECTURE.md header must say 'ATLAS V{version}' "
        f"(pyproject.toml says {version})")


def test_changelog_top_entry_matches():
    version = _truth()
    text = (REPO / "CHANGELOG.md").read_text()
    # An optional leading [Unreleased] section (Keep-a-Changelog
    # convention for merged-but-untagged work) sits above the release
    # entries; the version sync-check applies to the first RELEASE
    # heading below it.
    headings = re.findall(r"^## \[([^\]]+)\]", text, re.M)
    assert headings, "CHANGELOG.md must contain a '## [X.Y.Z]' release heading"
    releases = [h for h in headings if h.lower() != "unreleased"]
    assert releases, "CHANGELOG.md must contain a release heading besides [Unreleased]"
    assert releases[0] == version, (
        f"CHANGELOG.md top release entry is [{releases[0]}], "
        f"pyproject.toml says {version} — add the release entry")


def test_support_matrix_applies_to_matches():
    version = _truth()
    text = (REPO / "SUPPORT_MATRIX.md").read_text()
    m = re.search(r"Applies to:\s*\*\*V(\d+\.\d+\.\d+)", text)
    assert m, "SUPPORT_MATRIX.md must contain an 'Applies to: **VX.Y.Z' line"
    assert m.group(1) == version, (
        f"SUPPORT_MATRIX.md applies to V{m.group(1)}, "
        f"pyproject.toml says {version}")


@pytest.mark.parametrize("lang", ["zh-CN", "ja", "ko"])
def test_translated_readme_badge_matches(lang):
    version = _truth()
    path = REPO / "docs" / "lang" / lang / "README.md"
    assert f"version-V{version}-" in path.read_text(), (
        f"docs/lang/{lang}/README.md version badge must contain "
        f"'version-V{version}-' (pyproject.toml says {version})")
