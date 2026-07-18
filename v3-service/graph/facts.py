"""CodeGraph → Prolog facts + rules.

Faithful port of chiasmus `src/graph/facts.ts` (`escapeAtom`, `BUILTIN_RULES`,
`graphToProlog`). Emitted early per the plan so the optional Phase 5 solver layer
can be added without re-plumbing; nothing consumes it yet beyond the
`facts` analysis output. The insight facts (communities / hubs / bridges) from
the original are intentionally omitted.
"""

from __future__ import annotations

import re

from .types import CodeGraph

_BARE_ATOM = re.compile(r"^[a-z][a-z0-9_]*$")


def escape_atom(s: str) -> str:
    """Escape a string as a Prolog atom (single-quoted if needed)."""
    # fullmatch, not match: `$` in re.match also matches just before a trailing
    # newline, so "foo\n" would slip through unquoted and break the fact.
    if _BARE_ATOM.fullmatch(s):
        return s
    escaped = (
        s.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\0", "\\0")
    )
    return f"'{escaped}'"


MEMBER_RULES = """member(X, [X|_]).
member(X, [_|T]) :- member(X, T)."""

BUILTIN_RULES = (
    MEMBER_RULES
    + """

% Cycle-safe reachability via visited list
reaches(A, B) :- reaches(A, B, [A]).
reaches(A, B, _) :- calls(A, B).
reaches(A, B, Visited) :- calls(A, Mid), \\+ member(Mid, Visited), reaches(Mid, B, [Mid|Visited]).

% Path finding (returns the call chain)
path(A, B, Path) :- path(A, B, [A], Path).
path(A, B, _, [A, B]) :- calls(A, B).
path(A, B, Visited, [A|Rest]) :- calls(A, Mid), \\+ member(Mid, Visited), path(Mid, B, [Mid|Visited], Rest).

% Function-only reachability (for cycle detection). Methods excluded — unqualified
% method names collide across classes, producing phantom self-loops.
func_calls(A, B) :- calls(A, B), \\+ defines(_, A, method, _), \\+ defines(_, B, method, _).
func_reaches(A, B) :- func_reaches(A, B, [A]).
func_reaches(A, B, _) :- func_calls(A, B).
func_reaches(A, B, Visited) :- func_calls(A, Mid), \\+ member(Mid, Visited), func_reaches(Mid, B, [Mid|Visited]).

% Dead code: defined function not called by anyone and not an entry point.
dead(Name) :- defines(_, Name, function, _), \\+ calls(_, Name), \\+ entry_point(Name).

% Convenience predicates
caller_of(Target, Caller) :- calls(Caller, Target).
callee_of(Source, Callee) :- calls(Source, Callee)."""
)


def graph_to_prolog(graph: CodeGraph, entry_points=None) -> str:
    """Render a CodeGraph as a Prolog program (facts + built-in rules)."""
    lines = [
        ":- dynamic(defines/4).",
        ":- dynamic(calls/2).",
        ":- dynamic(imports/3).",
        ":- dynamic(exports/2).",
        ":- dynamic(contains/2).",
        ":- dynamic(file/2).",
        ":- dynamic(entry_point/1).",
        "",
    ]

    for f in graph.files:
        lines.append(f"file({escape_atom(f.path)}, {escape_atom(f.language)}).")
    if graph.files:
        lines.append("")

    for d in graph.defines:
        lines.append(
            f"defines({escape_atom(d.file)}, {escape_atom(d.name)}, "
            f"{escape_atom(d.kind)}, {d.line})."
        )
    if graph.defines:
        lines.append("")

    for c in graph.calls:
        lines.append(f"calls({escape_atom(c.caller)}, {escape_atom(c.callee)}).")
    if graph.calls:
        lines.append("")

    for i in graph.imports:
        lines.append(
            f"imports({escape_atom(i.file)}, {escape_atom(i.name)}, {escape_atom(i.source)})."
        )
    if graph.imports:
        lines.append("")

    resolved_rows = [i for i in graph.imports if i.resolved]
    if resolved_rows:
        lines.append(":- dynamic(imports_resolved/3).")
        for i in resolved_rows:
            lines.append(
                f"imports_resolved({escape_atom(i.file)}, {escape_atom(i.name)}, "
                f"{escape_atom(i.resolved)})."
            )
        lines.append("")

    for e in graph.exports:
        lines.append(f"exports({escape_atom(e.file)}, {escape_atom(e.name)}).")
    if graph.exports:
        lines.append("")

    for c in graph.contains:
        lines.append(f"contains({escape_atom(c.parent)}, {escape_atom(c.child)}).")
    if graph.contains:
        lines.append("")

    if entry_points:
        for ep in entry_points:
            lines.append(f"entry_point({escape_atom(ep)}).")
    else:
        for name in dict.fromkeys(e.name for e in graph.exports):
            lines.append(f"entry_point({escape_atom(name)}).")
    lines.append("")

    lines.append(BUILTIN_RULES)
    return "\n".join(lines)
