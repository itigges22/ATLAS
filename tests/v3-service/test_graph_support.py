"""Tests for graph support pieces: import resolution, Prolog facts, cache,
flag, and the build_graph entry point (issue #39, Phase 0)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

import graph  # noqa: E402
from graph.types import (  # noqa: E402
    CodeGraph, DefinesFact, CallsFact, ImportsFact, ExportsFact, ContainsFact, FileNode,
)
from graph.facts import escape_atom, graph_to_prolog, BUILTIN_RULES  # noqa: E402
from graph.resolve import resolve_imports, _module_name  # noqa: E402
from graph.cache import FileGraphCache, file_hash  # noqa: E402
from graph.flags import call_graph_enabled, ENV_VAR  # noqa: E402

_HAS_TS = graph.extraction_available()


class TestEscapeAtom:
    def test_bare_atom(self):
        assert escape_atom("main") == "main"
        assert escape_atom("foo_bar2") == "foo_bar2"

    def test_quoted(self):
        assert escape_atom("Foo") == "'Foo'"
        assert escape_atom("a/b.py") == "'a/b.py'"

    def test_escapes_quote_and_backslash(self):
        assert escape_atom("a'b") == "'a\\'b'"
        assert escape_atom("a\\b") == "'a\\\\b'"


class TestGraphToProlog:
    def test_emits_facts_and_rules(self):
        g = CodeGraph(
            defines=[DefinesFact("a.py", "main", "function", 1)],
            calls=[CallsFact("main", "helper")],
            imports=[ImportsFact("a.py", "os", "os")],
            exports=[ExportsFact("a.py", "main")],
            contains=[ContainsFact("Svc", "run")],
            files=[FileNode("a.py", "python", 10)],
        )
        prog = graph_to_prolog(g)
        assert "defines('a.py', main, function, 1)." in prog
        assert "calls(main, helper)." in prog
        assert "imports('a.py', os, os)." in prog
        assert "exports('a.py', main)." in prog
        assert "contains('Svc', run)." in prog
        assert "entry_point(main)." in prog
        assert BUILTIN_RULES in prog  # the reaches/path/dead rules are appended

    def test_resolved_imports_emitted(self):
        g = CodeGraph(imports=[ImportsFact("a.py", "h", "pkg.util", resolved="pkg/util.py")])
        prog = graph_to_prolog(g)
        assert "imports_resolved('a.py', h, 'pkg/util.py')." in prog

    def test_explicit_entry_points(self):
        g = CodeGraph(exports=[ExportsFact("a.py", "main"), ExportsFact("a.py", "other")])
        prog = graph_to_prolog(g, entry_points=["main"])
        assert "entry_point(main)." in prog
        assert "entry_point(other)." not in prog


class TestModuleName:
    def test_paths_to_modules(self):
        assert _module_name("pkg/mod.py") == "pkg.mod"
        assert _module_name("pkg/__init__.py") == "pkg"
        assert _module_name("a/b/c.pyi") == "a.b.c"


class TestResolveImports:
    def test_exact_module(self):
        # `from pkg.util import thing` -> source "pkg.util", which is a file.
        g = CodeGraph(imports=[ImportsFact("app.py", "thing", "pkg.util")])
        resolve_imports(g, ["app.py", "pkg/util.py"])
        assert g.imports[0].resolved == "pkg/util.py"

    def test_from_pkg_import_module(self):
        # from pkg import mod  -> source "pkg", name "mod", pkg/mod.py exists
        g = CodeGraph(imports=[ImportsFact("app.py", "mod", "pkg")])
        resolve_imports(g, ["app.py", "pkg/mod.py", "pkg/__init__.py"])
        assert g.imports[0].resolved == "pkg/mod.py"

    def test_unresolved_external(self):
        g = CodeGraph(imports=[ImportsFact("app.py", "os", "os")])
        resolve_imports(g, ["app.py"])
        assert g.imports[0].resolved is None

    def test_ambiguous_suffix_not_resolved(self):
        g = CodeGraph(imports=[ImportsFact("app.py", "x", "mod")])
        resolve_imports(g, ["app.py", "a/mod.py", "b/mod.py"])
        assert g.imports[0].resolved is None  # two candidates -> leave unresolved


class TestCache:
    def test_hit_on_same_content(self):
        if not _HAS_TS:
            pytest.skip("tree-sitter not installed")
        c = FileGraphCache()
        g1 = c.get_or_extract("a.py", "def f():\n    pass\n")
        assert len(c) == 1
        g2 = c.get_or_extract("a.py", "def f():\n    pass\n")
        # Same content -> not re-parsed (cache stays size 1), but a distinct copy
        # is returned so callers can't corrupt the cached object.
        assert len(c) == 1
        assert g1 is not g2
        assert [d.name for d in g1.defines] == [d.name for d in g2.defines]

    def test_resolve_does_not_corrupt_cache_across_batches(self):
        if not _HAS_TS:
            pytest.skip("tree-sitter not installed")
        c = graph.FileGraphCache()
        # Batch A includes the util module, so a.py's import resolves.
        files_a = {"a.py": "from pkg.util import helper\n\ndef main():\n    return helper()\n",
                   "pkg/util.py": "def helper():\n    return 1\n"}
        ga = graph.build_graph(files_a, cache=c)
        assert next(i for i in ga.imports if i.name == "helper").resolved == "pkg/util.py"
        # Batch B is a.py alone. Its import must NOT still show the resolution
        # from batch A (the bug: shared mutable ImportsFact on the cached graph).
        gb = graph.build_graph({"a.py": files_a["a.py"]}, cache=c)
        assert next(i for i in gb.imports if i.name == "helper").resolved is None

    def test_miss_on_changed_content(self):
        if not _HAS_TS:
            pytest.skip("tree-sitter not installed")
        c = FileGraphCache()
        c.get_or_extract("a.py", "def f(): pass")
        c.get_or_extract("a.py", "def g(): pass")
        assert len(c) == 2  # different content -> separate entries

    def test_hash_includes_path(self):
        assert file_hash("a.py", "x") != file_hash("b.py", "x")
        assert file_hash("a.py", "x") == file_hash("a.py", "x")

    def test_lru_eviction(self):
        if not _HAS_TS:
            pytest.skip("tree-sitter not installed")
        c = FileGraphCache(max_entries=2)
        c.get_or_extract("a.py", "def a(): pass")
        c.get_or_extract("b.py", "def b(): pass")
        c.get_or_extract("c.py", "def c(): pass")
        assert len(c) == 2


class TestFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert call_graph_enabled() is False

    def test_truthy(self, monkeypatch):
        for v in ("1", "true", "On", "yes"):
            monkeypatch.setenv(ENV_VAR, v)
            assert call_graph_enabled() is True


class TestBuildGraph:
    def test_end_to_end_cross_file(self):
        if not _HAS_TS:
            pytest.skip("tree-sitter not installed")
        files = {
            "app.py": "from pkg.util import helper\n\ndef main():\n    return helper()\n",
            "pkg/util.py": "def helper():\n    return 1\n",
        }
        g = graph.build_graph(files)
        names = {d.name for d in g.defines}
        assert {"main", "helper"} <= names
        # cross-file import resolved to the defining file
        imp = next(i for i in g.imports if i.name == "helper")
        assert imp.resolved == "pkg/util.py"
        # reachability across the project graph
        assert graph.reachability(g, "main", "helper") is True

    def test_ignores_non_python(self):
        g = graph.build_graph({"a.md": "# not code", "b.json": "{}"})
        assert g.defines == []
