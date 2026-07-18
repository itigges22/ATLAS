"""Minimal in-process Datalog evaluator over call-graph facts (issue #39,
Phase 5 — the optional "+ solver" layer).

The native analyses answer the everyday queries; this adds rule-based querying
over the same facts for the cases native traversal can't express cleanly
(arbitrary derived relations, transitive closure as a relation). It is a small,
dependency-free, bounded semi-naive-style fixpoint evaluator — NOT a full
SWI-Prolog. The Prolog facts emitted by facts.py remain the bridge to a real
external solver (chiasmus_verify / SWI-Prolog) when arbitrary rules are needed;
this engine covers the in-process, dependency-free case.

Correctness is anchored to the native layer: the built-in `reaches` closure is
cross-checked against analyses.reachability in the test suite.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .types import CodeGraph


class Var:
    """A Datalog variable. Distinct from string constants (which may be
    capitalized class names), so we never confuse a name with a variable."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"?{self.name}"


Term = object  # Var | str
Literal = Tuple[str, List[Term]]


class Datalog:
    def __init__(self):
        self.facts: Dict[str, Set[tuple]] = {}
        self.rules: List[Tuple[Literal, List[Literal]]] = []

    def add_fact(self, pred: str, *args: str) -> None:
        self.facts.setdefault(pred, set()).add(tuple(args))

    def add_rule(self, head: Literal, body: List[Literal]) -> None:
        self.rules.append((head, body))

    def run(self, max_iter: int = 10000) -> "Datalog":
        """Naive fixpoint: re-derive until no new tuple appears or the iteration
        cap (runaway guard) is hit."""
        it = 0
        changed = True
        while changed and it < max_iter:
            changed = False
            it += 1
            for head, body in self.rules:
                hpred, hterms = head
                rel = self.facts.setdefault(hpred, set())
                for binding in self._join(body, {}):
                    # Skip if a head variable is unbound by the body (an unsafe
                    # rule) rather than raising — keeps the public engine robust
                    # against caller-supplied rules. The built-in rules are safe.
                    if any(isinstance(t, Var) and t.name not in binding for t in hterms):
                        continue
                    htuple = tuple(self._subst(t, binding) for t in hterms)
                    if htuple not in rel:
                        rel.add(htuple)
                        changed = True
        return self

    def _join(self, body: List[Literal], binding: dict):
        if not body:
            yield dict(binding)
            return
        (pred, terms) = body[0]
        rest = body[1:]
        for fact in tuple(self.facts.get(pred, ())):
            b2 = self._unify(terms, fact, binding)
            if b2 is not None:
                yield from self._join(rest, b2)

    @staticmethod
    def _unify(terms: List[Term], fact: tuple, binding: dict):
        if len(terms) != len(fact):
            return None
        b = dict(binding)
        for t, val in zip(terms, fact):
            if isinstance(t, Var):
                if t.name in b:
                    if b[t.name] != val:
                        return None
                else:
                    b[t.name] = val
            elif t != val:
                return None
        return b

    @staticmethod
    def _subst(t: Term, binding: dict):
        return binding[t.name] if isinstance(t, Var) else t

    def query(self, pred: str, *pattern: Term) -> List[dict]:
        """Return all variable bindings for which `pred(pattern)` holds."""
        out: List[dict] = []
        for fact in self.facts.get(pred, ()):
            b = self._unify(list(pattern), fact, {})
            if b is not None:
                out.append(b)
        return out


def _load_calls(graph: CodeGraph, dl: Datalog) -> None:
    for c in graph.calls:
        dl.add_fact("calls", c.caller, c.callee)


def reaches_engine(graph: CodeGraph) -> Datalog:
    """A Datalog DB with the calls facts and the built-in transitive-reachability
    rules evaluated to fixpoint."""
    dl = Datalog()
    _load_calls(graph, dl)
    a, b, m = Var("A"), Var("B"), Var("M")
    dl.add_rule(("reaches", [a, b]), [("calls", [a, b])])
    dl.add_rule(("reaches", [a, b]), [("calls", [a, m]), ("reaches", [m, b])])
    return dl.run()


def reachable_pairs(graph: CodeGraph) -> List[Tuple[str, str]]:
    """The full transitive-closure relation `reaches/2` as (from, to) pairs —
    a relation the native single-pair API doesn't expose directly."""
    dl = reaches_engine(graph)
    return sorted(dl.facts.get("reaches", set()))


def solver_reaches(graph: CodeGraph, frm: str, to: str) -> bool:
    """Reachability via the Datalog engine. Semantically matches
    analyses.reachability (a self-pair holds only through a real call step)."""
    dl = reaches_engine(graph)
    return (frm, to) in dl.facts.get("reaches", set())
