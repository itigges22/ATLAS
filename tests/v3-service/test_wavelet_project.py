"""Conformance suite for project-wide decomposition + discovery.

Adapted from wavescope-mcp `src/project.test.ts`. Exercises discovery
(extension filtering, SKIP_DIRS, .gitignore, binary sniff) and project-wide
important-position aggregation against a temp tree.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

from wavelet.project import decompose_project, decompose_file_map, ProjectIndex  # noqa: E402

PY_A = """class Alpha:
    def run(self):
        return 1


def helper_a():
    return 2
"""

PY_B = """import os


class Beta:
    def go(self):
        return os.getcwd()
"""


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


class TestDiscovery:
    def test_indexes_code_files(self, tmp_path):
        _write(tmp_path, "a.py", PY_A)
        _write(tmp_path, "pkg/b.py", PY_B)
        idx = ProjectIndex.load(str(tmp_path))
        files = set(idx.list_files())
        assert "a.py" in files
        assert str(Path("pkg/b.py")) in files

    def test_skips_non_code_extensions(self, tmp_path):
        _write(tmp_path, "a.py", PY_A)
        _write(tmp_path, "notes.md", "# notes")
        _write(tmp_path, "data.json", "{}")
        idx = ProjectIndex.load(str(tmp_path))
        files = idx.list_files()
        assert "a.py" in files
        assert "notes.md" not in files
        assert "data.json" not in files

    def test_skips_skip_dirs(self, tmp_path):
        _write(tmp_path, "a.py", PY_A)
        _write(tmp_path, "node_modules/dep.py", PY_B)
        _write(tmp_path, "__pycache__/x.py", PY_B)
        files = ProjectIndex.load(str(tmp_path)).list_files()
        assert "a.py" in files
        assert all("node_modules" not in f for f in files)
        assert all("__pycache__" not in f for f in files)

    def test_honors_gitignore(self, tmp_path):
        _write(tmp_path, "a.py", PY_A)
        _write(tmp_path, "ignored.py", PY_B)
        _write(tmp_path, "sub/keep.py", PY_A)
        _write(tmp_path, ".gitignore", "ignored.py\n")
        files = ProjectIndex.load(str(tmp_path)).list_files()
        assert "a.py" in files
        assert "ignored.py" not in files
        assert str(Path("sub/keep.py")) in files

    def test_gitignore_negation(self, tmp_path):
        # Last-match-wins: `!keep.py` re-includes after `*.py` excludes it.
        _write(tmp_path, "keep.py", PY_A)
        _write(tmp_path, "skip.py", PY_B)
        _write(tmp_path, ".gitignore", "*.py\n!keep.py\n")
        files = ProjectIndex.load(str(tmp_path)).list_files()
        assert "keep.py" in files
        assert "skip.py" not in files

    def test_gitignore_dir_prune_blocks_negation(self, tmp_path):
        # Faithful to git (and wavescope): a file under an ignored *directory*
        # cannot be re-included by negation — the directory is pruned during
        # the walk, so descent never happens.
        _write(tmp_path, "build/keep.py", PY_A)
        _write(tmp_path, ".gitignore", "build/\n!build/keep.py\n")
        files = ProjectIndex.load(str(tmp_path)).list_files()
        assert str(Path("build/keep.py")) not in files

    def test_skips_binary(self, tmp_path):
        _write(tmp_path, "a.py", PY_A)
        (tmp_path / "weird.py").write_bytes(b"def x():\n    return 0\x00\x00binary")
        files = ProjectIndex.load(str(tmp_path)).list_files()
        assert "a.py" in files
        assert "weird.py" not in files


class TestProjectImportantPositions:
    def test_aggregates_across_files_with_path_labels(self, tmp_path):
        _write(tmp_path, "a.py", PY_A)
        _write(tmp_path, "b.py", PY_B)
        positions = decompose_project(str(tmp_path), min_coefficient=0.3, limit=20)
        assert len(positions) > 0
        # Labels carry the relative filename suffix; every entry has a filename.
        assert all(p.filename for p in positions)
        labels = " ".join(p.label for p in positions)
        assert "(a.py)" in labels or "(b.py)" in labels
        # Sorted by magnitude descending and capped at the limit.
        for i in range(1, len(positions)):
            assert abs(positions[i - 1].coefficient) >= abs(positions[i].coefficient)

    def test_limit_caps_results(self, tmp_path):
        _write(tmp_path, "a.py", PY_A)
        _write(tmp_path, "b.py", PY_B)
        assert len(decompose_project(str(tmp_path), 0.0, 2)) <= 2

    def test_empty_project(self, tmp_path):
        assert decompose_project(str(tmp_path)) == []


class TestDecomposeFileMap:
    def test_in_memory_matches_disk(self, tmp_path):
        # The in-memory entry point (used when v3-service has no project mount)
        # should produce the same labels/filenames as a disk scan of the same
        # files.
        _write(tmp_path, "a.py", PY_A)
        _write(tmp_path, "b.py", PY_B)
        disk = decompose_project(str(tmp_path), limit=20)
        mem = decompose_file_map({"a.py": PY_A, "b.py": PY_B}, limit=20)
        assert {(p.filename, p.label) for p in mem} == {(p.filename, p.label) for p in disk}

    def test_labels_carry_path_and_sorted(self):
        positions = decompose_file_map({"a.py": PY_A, "b.py": PY_B}, limit=20)
        assert positions
        assert all(p.filename in ("a.py", "b.py") for p in positions)
        for i in range(1, len(positions)):
            assert abs(positions[i - 1].coefficient) >= abs(positions[i].coefficient)

    def test_empty_and_blank(self):
        assert decompose_file_map({}) == []
        assert decompose_file_map({"a.py": ""}) == []

    def test_limit_caps(self):
        assert len(decompose_file_map({"a.py": PY_A, "b.py": PY_B}, limit=2)) <= 2
