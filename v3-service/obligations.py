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
# KIND_DECLARED_COMMAND is absent because its floor is not a constant: it is
# the kind the CLIENT typed for that exact command, and RUNTIME for a command
# declared without a type. No declared command reaches ORACLE -- exit zero says
# the command ran and succeeded against these bytes, never that an answer was
# compared with a reference.
#
# KIND_BASELINE_PRESERVED is absent because its floor is not a constant: it is
# whatever the evidence currently describing that baseline already reached.
# KIND_UNSUPPORTED is absent because no strength closes it.
_KIND_REQUIRED_STRENGTH = {
    KIND_ARTIFACT_EXISTS: contract.SYNTAX,
    KIND_SYNTACTIC_VALIDITY: contract.SYNTAX,
    KIND_DECLARED_EXAMPLE: contract.ORACLE,
}

# The strengths a client may declare for a command. Oracle is absent: a
# comparison against a reference is not something an exit status can be.
_DECLARABLE_VERIFICATION_KINDS = (contract.SYNTAX, contract.RUNTIME,
                                  contract.BEHAVIORAL)

# --- when an obligation can be answered -------------------------------------
#
# The first structured task could never close, and the reason was circular.
# artifact_exists was a required obligation with a syntax floor; nothing can
# evidence a file's existence before the candidate lands; delivery needs
# authorization; authorization needed the obligation met.
#
# Three roles, and every kind has exactly one:
#
#   target_identity              names WHICH artifact a delivery may replace.
#                                Never evidence about bytes.
#   authorization_prerequisite   must be met by evidence bound to the exact
#                                candidate BEFORE those bytes may land.
#   post_delivery_settlement     answerable only once the bytes are on disk
#                                and the ledger agrees they are there.
#
# artifact_exists carries the first and the third and none of the second.
ROLE_TARGET_IDENTITY = "target_identity"
ROLE_AUTHORIZATION_PREREQUISITE = "authorization_prerequisite"
ROLE_POST_DELIVERY_SETTLEMENT = "post_delivery_settlement"

# Total over KINDS. A kind with no role is one nothing knows when to ask
# about, so the lookup fails closed.
_KIND_ROLE = {
    KIND_ARTIFACT_EXISTS: ROLE_POST_DELIVERY_SETTLEMENT,
    KIND_SYNTACTIC_VALIDITY: ROLE_AUTHORIZATION_PREREQUISITE,
    KIND_DECLARED_COMMAND: ROLE_AUTHORIZATION_PREREQUISITE,
    KIND_DECLARED_EXAMPLE: ROLE_AUTHORIZATION_PREREQUISITE,
    KIND_BASELINE_PRESERVED: ROLE_AUTHORIZATION_PREREQUISITE,
    # Owed and unnameable: a prerequisite, so it blocks authorization, and
    # unsatisfiable, so it blocks it forever.
    KIND_UNSUPPORTED: ROLE_AUTHORIZATION_PREREQUISITE,
}

# The separate question: does this kind identify an artifact a delivery may
# replace? Only the declared output does, and it says nothing about quality.
_KIND_NAMES_TARGET = {KIND_ARTIFACT_EXISTS}

# Kinds whose floor is supplied per obligation rather than by the kind.
_DYNAMIC_STRENGTH_KINDS = (KIND_BASELINE_PRESERVED, KIND_DECLARED_COMMAND)

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
        if (kind == KIND_DECLARED_COMMAND
                and baseline_strength not in _DECLARABLE_VERIFICATION_KINDS):
            raise ObligationError(
                f"{kind} cannot carry {baseline_strength!r}: a command's exit "
                f"status is not a comparison against a reference")
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


def role(kind: str) -> str:
    """When this kind can be answered. Raises on a kind with no role."""
    if kind not in _KIND_ROLE:
        raise ObligationError(f"{kind!r} has no role; nothing knows when to ask it")
    return _KIND_ROLE[kind]


def authorization_prerequisites(obs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What must be met by candidate-bound evidence before bytes may land.

    artifact_exists is deliberately absent: it cannot be evidenced before the
    candidate is on disk, and treating it as a prerequisite is the circle this
    split removes.
    """
    validate(obs)
    return [o for o in obs if role(o["kind"]) == ROLE_AUTHORIZATION_PREREQUISITE]


def post_delivery_settlement(obs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validate(obs)
    return [o for o in obs if role(o["kind"]) == ROLE_POST_DELIVERY_SETTLEMENT]


def names_target(kind: str) -> bool:
    """Whether this kind identifies an artifact a delivery may replace."""
    if kind not in KINDS:
        raise ObligationError(f"unknown obligation kind {kind!r}")
    return kind in _KIND_NAMES_TARGET


def authorization_floor(obs: Sequence[Dict[str, Any]]) -> str:
    """The strongest floor the PREREQUISITES demand, or "" when there is none.

    "" is a real answer and not a permissive one: a declared document with no
    declared verification owes nothing measurable here, which means there is
    nothing to satisfy rather than nothing to do.
    """
    validate(obs)
    order = contract.STRENGTH_ORDER
    floor = ""
    for o in authorization_prerequisites(obs):
        if not o.get("required"):
            continue
        if o["kind"] in _UNSATISFIABLE_KINDS:
            return order[-1]
        if not floor or order.index(o["required_strength"]) > order.index(floor):
            floor = o["required_strength"]
    return floor


def kinds_of(obs: Sequence[Dict[str, Any]]) -> List[str]:
    """The distinct kinds present, in declaration order."""
    out = []
    for o in obs:
        if o.get("kind") not in out:
            out.append(o.get("kind"))
    return out
