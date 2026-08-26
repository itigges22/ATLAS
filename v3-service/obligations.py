"""What a task obliges, as a closed typed vocabulary.

`contract.py` keeps four concepts apart and says the adapter never decides
what the user required. Until now nothing supplied the other half: the
requirements an adapter was measured against were the adapter's own, so a
verifier read its own reach and called the answer the task's obligation.

This module is the missing half. An obligation is derived from the validated
REQUEST -- the client's declared outputs, its declared verification commands,
the artifact classes the proxy's own syntax gate already governs, and the
baseline a replacement would overwrite. Nothing here reads a prompt, a
filename convention, a benchmark, or a model's opinion.

Three things are deliberately separate and must not be collapsed:

  KIND      what sort of thing is owed (a closed set)
  STRENGTH  how strong the evidence closing it must be
  SUBJECT   the exact thing it is owed about, carried as a hash

The subject is a hash and never the text. A declared verification command is
a subject, and a command string in an operator log is a content leak; a
uniform rule that never carries text cannot leak one by exception.

The same vocabulary exists in the Go proxy (proxy/obligation_kinds.go).
Two copies of a closed set is a divergence waiting to happen, so a contract
test parses both and fails when they disagree.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

import contract

# --- kinds -----------------------------------------------------------------

# The artifact the client named must exist at its canonical path.
KIND_ARTIFACT_EXISTS = "artifact_exists"
# Its bytes must satisfy the structural check the proxy already owns for that
# artifact class -- the same gate live completion policy requires, not a
# second extension table.
KIND_SYNTACTIC_VALIDITY = "syntactic_validity"
# One exact command the client required, run as the client wrote it.
KIND_DECLARED_COMMAND = "declared_command"
# One example or oracle case the client stated, with its own expected answer.
KIND_DECLARED_EXAMPLE = "declared_example"
# An artifact that already carries a current passing verdict must not be
# replaced by something demonstrated less well.
KIND_BASELINE_PRESERVED = "baseline_preserved"
# Something is owed and nothing here can name what. Never satisfiable, never
# vacuously complete.
KIND_UNSUPPORTED = "unsupported"

KINDS = (
    KIND_ARTIFACT_EXISTS,
    KIND_SYNTACTIC_VALIDITY,
    KIND_DECLARED_COMMAND,
    KIND_DECLARED_EXAMPLE,
    KIND_BASELINE_PRESERVED,
    KIND_UNSUPPORTED,
)

# How strong the evidence closing each kind must be.
#
# A declared command is BEHAVIORAL, not ORACLE: exit zero says the command the
# client asked for ran and succeeded against these bytes. It does not say the
# answer was checked against a reference, and calling an arbitrary exit-zero
# command an oracle is how "it ran" became "it is right".
#
# KIND_BASELINE_PRESERVED is absent because its floor is not a constant: it is
# whatever the evidence currently describing that baseline already reached.
# KIND_UNSUPPORTED is absent because no strength closes it.
_KIND_REQUIRED_STRENGTH = {
    KIND_ARTIFACT_EXISTS: contract.SYNTAX,
    KIND_SYNTACTIC_VALIDITY: contract.SYNTAX,
    KIND_DECLARED_COMMAND: contract.BEHAVIORAL,
    KIND_DECLARED_EXAMPLE: contract.ORACLE,
}

# Kinds whose floor is supplied per obligation rather than by the kind.
_DYNAMIC_STRENGTH_KINDS = (KIND_BASELINE_PRESERVED,)

# Kinds nothing can close.
_UNSATISFIABLE_KINDS = (KIND_UNSUPPORTED,)


class ObligationError(ValueError):
    """An obligation that cannot be established. Raised at construction: a
    malformed obligation must never exist to be measured against."""


def obligation_id(kind: str, subject: str) -> str:
    """The canonical name of one obligation.

    Deterministic and content-free: the subject is hashed, so the id can be
    logged, compared and carried on the wire without ever holding a path's
    contents or a command's text. The Go proxy computes the same string.
    """
    if kind not in KINDS:
        raise ObligationError(f"unknown obligation kind {kind!r}")
    if not isinstance(subject, str) or not subject.strip():
        raise ObligationError("obligation subject must be a non-empty string")
    digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()[:32]
    return f"{kind}:{digest}"


def required_strength(kind: str, baseline_strength: Optional[str] = None) -> str:
    """The floor evidence must reach to close this kind.

    A baseline obligation is at least as strong as the evidence already
    describing that baseline: replacing a file whose oracle passed on the
    strength of a compile is a regression dressed as a delivery.
    """
    if kind not in KINDS:
        raise ObligationError(f"unknown obligation kind {kind!r}")
    if kind in _UNSATISFIABLE_KINDS:
        raise ObligationError(f"{kind} is never satisfiable and has no strength")
    if kind in _DYNAMIC_STRENGTH_KINDS:
        if baseline_strength not in contract.STRENGTH_ORDER:
            raise ObligationError(
                f"{kind} needs the baseline's own strength, got "
                f"{baseline_strength!r}")
        return baseline_strength
    if baseline_strength is not None:
        raise ObligationError(
            f"{kind} has a fixed strength; a baseline strength cannot raise it")
    return _KIND_REQUIRED_STRENGTH[kind]


def obligation(*, kind: str, subject: str, required: bool = True,
               baseline_strength: Optional[str] = None,
               weight: float = 1.0) -> Dict[str, Any]:
    """One thing the task owes, named by hash and floored by kind.

    An unsupported obligation is representable on purpose: a task that owes
    something this build cannot name must say so, so coverage reports it
    unmeasurable rather than reporting nothing at all.
    """
    if kind not in KINDS:
        raise ObligationError(f"unknown obligation kind {kind!r}")
    oid = obligation_id(kind, subject)
    if kind in _UNSATISFIABLE_KINDS:
        strength = ""
    else:
        strength = required_strength(kind, baseline_strength)
    return {
        "id": oid,
        "kind": kind,
        "subject_hash": oid.split(":", 1)[1],
        "required": bool(required),
        "required_strength": strength,
        "weight": float(weight),
    }


def validate(obs: Sequence[Dict[str, Any]]) -> None:
    """Every obligation must be well-formed and uniquely named.

    Fails closed on an unknown kind or an unknown strength: an obligation
    that survives validation with either is one some later reader has to
    remember to reject.
    """
    seen = set()
    for o in obs:
        kind = o.get("kind")
        if kind not in KINDS:
            raise ObligationError(f"unknown obligation kind {kind!r}")
        oid = o.get("id")
        if not isinstance(oid, str) or not oid.startswith(f"{kind}:"):
            raise ObligationError(f"obligation id {oid!r} does not name kind {kind!r}")
        if oid in seen:
            raise ObligationError(f"duplicate obligation {oid!r}")
        seen.add(oid)
        strength = o.get("required_strength")
        if kind in _UNSATISFIABLE_KINDS:
            if strength:
                raise ObligationError(f"{kind} cannot carry a strength")
            continue
        if strength not in contract.STRENGTH_ORDER:
            raise ObligationError(
                f"unknown required strength {strength!r} for {oid!r}")


def to_requirements(obs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The obligations as contract requirements, keyed by obligation id.

    contract.py stays generic: it sees opaque criterion ids and never learns
    what a kind means. The id carries the kind so a reader can tell what an
    unmet requirement was about without this module having to explain.
    """
    validate(obs)
    return [contract.requirement(o["id"], required=bool(o["required"]),
                                 weight=float(o.get("weight", 1.0)))
            for o in obs]


def closure_floor(obs: Sequence[Dict[str, Any]]) -> str:
    """The strongest floor any REQUIRED obligation demands.

    An unsupported required obligation makes the floor unreachable rather
    than absent -- it returns the top of the order, so a record cannot close
    a task that owes something nothing measured.
    """
    validate(obs)
    order = contract.STRENGTH_ORDER
    floor = order[0]
    for o in obs:
        if not o.get("required"):
            continue
        if o["kind"] in _UNSATISFIABLE_KINDS:
            return order[-1]
        if order.index(o["required_strength"]) > order.index(floor):
            floor = o["required_strength"]
    return floor


def kinds_of(obs: Sequence[Dict[str, Any]]) -> List[str]:
    """The distinct kinds present, in declaration order."""
    out = []
    for o in obs:
        if o.get("kind") not in out:
            out.append(o.get("kind"))
    return out
