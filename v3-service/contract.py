"""Generic, versioned evidence contract.

Four concepts kept separate, because collapsing any pair is what produced the
defects so far:

  1. TASK REQUIREMENTS   what this user asked for (owned by the task)
  2. ADAPTER CAPABILITIES what a verifier can observe at all (owned by the adapter)
  3. OBSERVATIONS        what this execution actually demonstrated
  4. DECISION POLICY     how 1+3 give coverage, ranking and closure (generic)

The adapter declares what it CAN measure; it does not decide what the user
REQUIRED. Moving INTERACTIVE_REQUIRED wholesale into the browser adapter
would have relocated the coupling rather than removed it — every canvas
artifact would still inherit collision and scoring criteria, including
animations, drawing tools, visualisations and simulations.

Criterion ids are opaque strings to everything in this module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0.0"

# Observation status
DEMONSTRATED = "demonstrated"
REFUTED = "refuted"
UNOBSERVED = "unobserved"
INCONCLUSIVE = "inconclusive"
NOT_APPLICABLE = "not_applicable"

# Evidence strength: how strong was the VERIFIER. Independent of coverage.
SYNTAX = "syntax"
RUNTIME = "runtime"
BEHAVIORAL = "behavioral"
ORACLE = "oracle"
STRENGTH_ORDER = [SYNTAX, RUNTIME, BEHAVIORAL, ORACLE]


def requirement(criterion_id: str, required: bool = True, weight: float = 1.0) -> Dict:
    return {"id": criterion_id, "required": bool(required), "weight": float(weight)}


def observation(status: str, detail: str = "", confidence: float = 1.0) -> Dict:
    return {"status": status, "detail": detail, "confidence": float(confidence)}


def build(contract_id: str, contract_version: str, adapter_id: str,
          adapter_version: str, requirements: List[Dict],
          observations: Dict[str, Dict], capabilities: List[str],
          evidence_strength: str, execution_status: str = "ok",
          supported: bool = True, artifact_scope: str = "",
          project_snapshot_hash: str = "") -> Dict:
    """Assemble a contract record and derive the policy fields.

    A required criterion the adapter CANNOT measure can never be complete —
    inability to observe is not evidence of absence.
    """
    caps = set(capabilities)
    missing: List[str] = []
    got = 0.0
    total = 0.0
    for req in requirements:
        cid = req["id"]
        total += req["weight"]
        obs = observations.get(cid) or observation(
            UNOBSERVED if cid in caps else NOT_APPLICABLE)
        if obs["status"] == DEMONSTRATED:
            got += req["weight"]
        elif req["required"]:
            missing.append(cid)

    # An adapter must not claim a criterion outside its declared capability.
    overclaimed = [c for c, o in observations.items()
                   if c not in caps and o.get("status") == DEMONSTRATED]

    requirements_complete = not missing and not overclaimed and supported
    quality = (got / total) if total else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": contract_id,
        "contract_version": contract_version,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "artifact_scope": artifact_scope,
        "project_snapshot_hash": project_snapshot_hash,
        "requirements": list(requirements),
        "capabilities": sorted(caps),
        "observations": dict(observations),
        "execution_status": execution_status,
        "supported": bool(supported),
        "evidence_strength": evidence_strength,
        "requirements_complete": requirements_complete,
        "quality_score": round(quality, 4),
        "missing_required": missing,
        "overclaimed": overclaimed,
        "closure_eligible": closure_eligible(evidence_strength, requirements_complete,
                                             quality, supported, execution_status),
    }


def closure_eligible(strength: str, requirements_complete: bool, quality: float,
                     supported: bool = True, execution_status: str = "ok",
                     quality_threshold: float = 1.0) -> bool:
    """Closure is a POLICY decision, not an evidence level.

    All required behaviour can be demonstrated while optional quality
    dimensions remain improvable; refusing to close there is a choice about
    ambition, not a claim that the evidence was weak.
    """
    if not supported or execution_status != "ok":
        return False
    if strength not in (BEHAVIORAL, ORACLE):
        return False
    return requirements_complete and quality >= quality_threshold


def comparable(a: Dict, b: Dict) -> bool:
    """Two records may be score-compared only under the same rubric.

    A 0.75 canvas score and a 0.75 API score do not share a scale.
    """
    return (a.get("contract_id") == b.get("contract_id")
            and a.get("contract_version") == b.get("contract_version")
            and a.get("artifact_scope") == b.get("artifact_scope")
            and a.get("project_snapshot_hash") == b.get("project_snapshot_hash"))


def rank_key(rec: Dict) -> Tuple:
    """Within one contract: requirements first, then quality, then strength."""
    return (
        1 if rec.get("requirements_complete") else 0,
        float(rec.get("quality_score", 0.0)),
        STRENGTH_ORDER.index(rec["evidence_strength"])
        if rec.get("evidence_strength") in STRENGTH_ORDER else -1,
        1 if rec.get("supported") else 0,
    )


def select(records: List[Dict]) -> Tuple[Optional[Dict], List[Dict]]:
    """Best record among the largest mutually comparable group.

    Records under another rubric are returned as incomparable rather than
    ranked numerically against the winner.
    """
    usable = [r for r in records if r.get("supported")]
    if not usable:
        return None, list(records)
    groups: Dict[Tuple, List[Dict]] = {}
    for r in usable:
        key = (r.get("contract_id"), r.get("contract_version"),
               r.get("artifact_scope"), r.get("project_snapshot_hash"))
        groups.setdefault(key, []).append(r)
    best_group = max(groups.values(), key=len)
    winner = max(best_group, key=rank_key)
    incomparable = [r for r in records if r not in best_group]
    return winner, incomparable
