"""Graph-backed call resolution for the structural veto (issue #39, Phase 1).

The shipped veto (v3-service `structural_score`) accepts a bare call name if ANY
project file defines a top-level symbol with that name, even when the candidate
never imports it. That lets through genuine broken cross-file references (a
NameError the sandbox can miss on an unexecuted path). This module resolves
direct-identifier calls precisely against the import graph instead:

- a call resolves if its name is defined locally, is a builtin, is an imported
  name, or is supplied by a wildcard import whose module's *actual* exports
  include it (resolved via the call graph, not blanket-accepted);
- when a wildcard import can't be resolved to an in-batch file (stdlib /
  third-party), resolution is treated as uncertain and nothing is flagged
  (conservative, matching today's leniency);
- attribute / method calls (`obj.foo()`) are out of scope, exactly as the
  shipped veto, because the receiver type isn't statically known.

`strict=False` restores the old behavior (accept any project symbol) so the veto
can be tightened gradually.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from . import extract as _extract
from .resolve import resolve_imports
from .types import CodeGraph

# Compact builtin set; the caller (v3-service) can pass its fuller PY_BUILTINS.
_DEFAULT_BUILTINS = frozenset({
    "print", "len", "range", "int", "str", "float", "bool", "list", "dict",
    "tuple", "set", "frozenset", "type", "isinstance", "issubclass", "getattr",
    "setattr", "hasattr", "super", "open", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "sum", "min", "max", "abs", "round", "any", "all",
    "repr", "format", "iter", "next", "id", "hash", "ord", "chr", "bytes",
    "bytearray", "object", "property", "staticmethod", "classmethod", "vars",
    "dir", "callable", "exec", "eval", "globals", "locals", "input",
})


def direct_call_names(code: str) -> List[str]:
    """Direct-identifier call targets in `code` (e.g. `foo()`), in source order.
    Skips attribute/subscript/chained calls — those need receiver-type info the
    static graph doesn't have. Mirrors v3-service `_extract_python_call_targets`."""
    if not _extract.available():
        return []
    try:
        parser = _extract._ts.Parser(_extract._PY_LANG)
        tree = parser.parse(bytes(code, "utf-8"))
    except Exception:
        return []
    out: List[str] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "call":
            for child in node.children:
                if child.type == "identifier":
                    out.append(_extract._text(child))
                    break
                if child.type != "(":
                    break  # attribute / subscript → skip
        stack.extend(node.children)
    return out


def _collect_identifiers(node, out: Set[str]) -> None:
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "identifier":
            out.add(_extract._text(n))
        for i in range(n.child_count):
            stack.append(n.child(i))


def bound_names(code: str) -> Set[str]:
    """Every name bound anywhere in the file: def/class names, function and
    lambda parameters, assignment / walrus / augmented-assignment targets, loop
    variables, `with`/`except as` aliases, and global/nonlocal declarations.

    A call to a name in this set must NOT be flagged unresolved — it could be a
    local, a parameter, or an assigned callable. Over-collecting (e.g. picking up
    identifiers inside parameter type annotations) is deliberately safe: it can
    only cause the veto to MISS a bug, never to reject valid code. Without this,
    `def run(cb): return cb()` and `x = lambda: 1; x()` are false positives.
    """
    if not _extract.available():
        return set()
    try:
        parser = _extract._ts.Parser(_extract._PY_LANG)
        tree = parser.parse(bytes(code, "utf-8"))
    except Exception:
        return set()

    names: Set[str] = set()
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        t = n.type
        if t in ("function_definition", "class_definition"):
            nm = n.child_by_field_name("name")
            if nm is not None:
                names.add(_extract._text(nm))
        elif t in ("parameters", "lambda_parameters"):
            _collect_identifiers(n, names)
        elif t in ("assignment", "augmented_assignment"):
            left = n.child_by_field_name("left")
            if left is not None:
                _collect_identifiers(left, names)
        elif t == "named_expression":
            # walrus `(g := ...)` — tree-sitter names the target field `name`.
            target = n.child_by_field_name("name")
            if target is not None:
                _collect_identifiers(target, names)
        elif t in ("for_statement", "for_in_clause"):
            left = n.child_by_field_name("left")
            if left is not None:
                _collect_identifiers(left, names)
        elif t == "as_pattern":
            alias = n.child_by_field_name("alias")
            if alias is not None:
                _collect_identifiers(alias, names)
        elif t in ("global_statement", "nonlocal_statement"):
            _collect_identifiers(n, names)
        for i in range(n.child_count):
            stack.append(n.child(i))
    return names


def _defs_by_file(graph: CodeGraph) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for d in graph.defines:
        out.setdefault(d.file, set()).add(d.name)
    return out


def unresolved_calls(
    candidate_path: str,
    candidate_code: str,
    project_files: Optional[Dict[str, str]] = None,
    builtins: Optional[Set[str]] = None,
    strict: bool = True,
) -> dict:
    """Resolve the candidate's direct calls against the import graph.

    Returns {"ok", "unresolved": [...], "n_calls_total", "lenient": bool}.
    `lenient` is True when an unresolvable wildcard import means nothing can be
    confidently flagged. `ok=False` (with "error") when extraction is unavailable.
    """
    if not _extract.available():
        return {"ok": False, "error": "tree-sitter not installed"}

    project_files = project_files or {}
    builtins = builtins or set(_DEFAULT_BUILTINS)

    cand = _extract.extract_file(candidate_path, candidate_code)
    # All names bound anywhere in the file (params, locals, assignments, defs),
    # not just def/class names — otherwise callbacks and assigned callables
    # false-positive. See bound_names.
    local = bound_names(candidate_code) | {d.name for d in cand.defines}
    import_names: Set[str] = set()
    for i in cand.imports:
        if i.name == "*":
            continue  # handled below via wildcard resolution
        import_names.add(i.name)
        # `import a.b.c` binds only the top package `a` (you'd call a.b.c.x()
        # as an attribute, never bare `c`). Bind the first segment, not the last.
        import_names.add(i.name.split(".")[0])

    # Build the project graph (cached) to resolve wildcard modules to their
    # actual exported names, and for the non-strict project-symbol fallback.
    all_files = dict(project_files)
    all_files[candidate_path] = candidate_code
    from . import build_graph  # local import avoids a cycle at module load
    proj = build_graph(all_files)
    defs_by_file = _defs_by_file(proj)
    project_symbols = {d.name for d in proj.defines}

    # Resolve the candidate's wildcard imports to in-batch files.
    resolve_imports(cand, list(all_files.keys()))
    wildcard_names: Set[str] = set()
    lenient = False
    for i in cand.imports:
        if i.name != "*":
            continue
        if i.resolved and i.resolved in defs_by_file:
            wildcard_names |= defs_by_file[i.resolved]
        else:
            # Wildcard from an unresolved module (stdlib / third-party): we can't
            # know what it supplies, so don't flag anything.
            lenient = True

    resolved = local | set(builtins) | import_names | wildcard_names
    if not strict:
        resolved |= project_symbols

    calls = direct_call_names(candidate_code)
    unresolved: List[str] = []
    seen: Set[str] = set()
    if not lenient:
        for name in calls:
            if name in resolved or name in seen:
                continue
            seen.add(name)
            unresolved.append(name)

    return {
        "ok": True,
        "unresolved": unresolved[:10],
        "n_unresolved": len(unresolved),
        "n_calls_total": len(calls),
        "lenient": lenient,
    }
