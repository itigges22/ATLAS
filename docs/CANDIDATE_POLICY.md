# Candidate policy: strict, advisory, confirm

**ATLAS is an interactive coding agent, not a formal verifier.** For an ordinary
coding task there is no oracle to consult and no proof to be had. A machine that
waited for one would never replace the model's own work with a better candidate,
which is exactly what the delivery chain did: it required trusted behavioral
evidence, and the only producer of that evidence ran *after* the gate that
needed it.

What can be decided honestly is narrower and still useful: whether a candidate
is **preferable to the model's original proposal** on evidence that is bounded,
typed, and bound to the exact bytes — with the user in the loop through the
terminal and the diff.

Three modes, because there are three genuinely different situations.

| Mode | When | What lands |
| --- | --- | --- |
| `strict` | a client declared verification the proxy can run | the candidate, automatically, once that verification passes at the declared strength against these exact bytes |
| `advisory` | no oracle exists | a candidate may be *preferred* on bounded evidence. A measured quality policy, never a proof. |
| `confirm` | the decision belongs to the user | the candidate, after a one-time approval bound to the exact bytes |

`strict` is the shipped default. **`advisory` computes its answer and delivers
nothing** until [Candidate Confidence Policy
Validation](#candidate-confidence-policy-validation) has measured it; see
"Calibration status" below for why.

## Who selects the mode

The **validated client request** (`task_contract.candidate_policy`) or **trusted
operator configuration** (`ATLAS_CANDIDATE_POLICY`), in that order, defaulting to
`strict`. The model cannot select it and neither can the V3 service:
`candidatePolicyOf` reads the validated contract and the process environment and
nothing else, and a structural test asserts it names no service or model type.

An unrecognised value in a request is refused at the boundary. An unrecognised
value in the environment falls back to `strict`, because refusing every request
in a deployment over a typo in a variable is worse than running the behaviour
every client already has.

## Proposal is not authorization

The service **proposes**; the proxy **authorizes**. These were one decision, and
merging them is what made the trusted producer unreachable.

- `proposedV3Candidate` answers one question: are these bytes materially
  different from the caller's own? It reads no verdict — a structural test
  asserts it touches neither `Passed`, `Evidence`, `ClosureEligible` nor
  `Selection`.
- `closure_eligible`, selection status, selector score, consensus, Lens and ASA
  are **advisory metadata**. They rank. They do not authorize, and no field of
  them mints a grant.
- The proxy stages the proposal, produces its own evidence about those exact
  bytes, and decides.
- A rejected candidate leaves the model's own proposal exactly as it was.

The one place a service verdict still decides anything is a request that
declared **no output knowledge**: it states no target and no obligation, so
there is nothing to authorize against and its delivery rule is the one it has
always had. That rule is named `serviceCertifiedCandidate` rather than left
looking like an authorization.

Hard proposal requirements survive unchanged, because they are about bytes being
usable rather than proven: materially different, valid identity, correct file
class, no language swap, no edit-boundary violation, nothing malformed, and no
target or workspace mutation during staging.

## The decisions

Closed vocabulary. Every value is a statement about what happened, never about
how likely the candidate is to be correct.

| Decision | Meaning |
| --- | --- |
| `baseline_retained` | the model's own proposal is what lands |
| `candidate_preferred_advisory` | bounded evidence prefers the candidate; not proof, and not a delivery in this build |
| `candidate_authorized_strict` | trusted declared verification passed at the declared strength on these exact bytes |
| `human_confirmation_required` | the candidate is presentable and the decision is the user's |
| `candidate_rejected_hard_veto` | something disqualifying was observed |
| `insufficient_confidence` | nothing disqualified it and nothing supported it |

`insufficient_confidence` is a real answer and not a fallback: "we had no
evidence" and "we had evidence against" are different, and only one of them is a
fact about the candidate.

## Hard vetoes

A veto is a fact with an owner outside the model. One is enough, and no signal
outweighs one — a veto outranks even a strict authorization.

`syntax_or_structural_failure`, `execution_evidence_unavailable`,
`candidate_mutated_protected_assets`, `language_or_target_mismatch`,
`stale_candidate_or_workspace_identity`, `declared_verification_failed`,
`unauthorized_path_expansion`, `weaker_than_baseline_on_a_trusted_check`,
`cancelled_or_timed_out`, `incomplete_evidence`,
`destructive_operation_without_permission`.

Advisory mode lowers the evidence required to **prefer** a candidate. It lowers
nothing about identity, permission, path containment, mutation or delivery:
one route identity, one candidate identity, one exact-byte grant, one
consumption, one write, disk and ledger agreement, validation, settlement, and a
complete disposition on both the route and the grant. Destructive operations stay
with the permission owner; advisory confidence is not a permission.

## Advisory signals, and their calibration status

Recorded, never thresholded. Every one is either the same model grading its own
output, a service ranking that output, or a scorer whose normalisation carries
its own calibration flag.

| Signal | Owner | Status |
| --- | --- | --- |
| Lens `gx_score_mean` severe veto (0.52) | geometric-lens | **calibrated**, on 188 live scores — as a *degeneracy veto*, not a correctness predictor |
| Lens `cx_normalized` / `energy_norm` | geometric-lens | carries a `calibrated` flag; uncalibrated when the normalisation artifact is absent. Ranking only |
| `closure_quality_threshold` (1.0) | v3-service | means "every required criterion complete", not "likely correct" |
| CodeT consensus / cluster agreement | v3-service | same model on both sides, so agreement is not independence |
| Best-of-K margin | v3-service | never measured against outcomes |
| proxy gate pass | proxy | trusted, and a syntax fact rather than a quality one |

**No defensible correctness threshold exists.** So advisory is implemented in
shadow form: the policy computes its decision, records it with its signals, and
delivers nothing on it. The records are what a calibration will be computed
from.

### Candidate Confidence Policy Validation

The measurement that has to exist before advisory may change what lands:
planner-only against full V3 with advisory acting, reported separately for
new-file versus edit, new-source versus historical-mechanism, trusted
verification versus no oracle, and per model family and size. It must show a
credible positive net improvement, a regression rate under a pre-registered
ceiling, no increase in unbounded or dishonest terminals, no permission, path,
identity or delivery regression, reproduced on Gemma and then evaluated
independently on Qwen.

## What the user sees

Delivered bytes name their origin, from a closed vocabulary the terminal can
render: `model_proposal`, `strict_trusted_candidate`, `advisory_candidate`,
`human_approved_candidate`. Only a decision that actually delivers may claim a
candidate origin; everything else is the model's own work and says so.

The provenance is a server-side fact. It is not part of `modelFacingResult`: the
user needs to know what they are reading, and the model does not get to argue
with the answer. No internal confidence vocabulary is presented as a correctness
guarantee, because none of it is one.

## Where it lives

| File | What it owns |
| --- | --- |
| `proxy/candidate_policy.go` | modes, sources, the decision vocabulary, the telemetry record |
| `proxy/advisory_policy.go` | the veto vocabulary, the signal set, and the policy owner |
| `proxy/candidate_provenance.go` | what the terminal is told about delivered bytes |
| `proxy/verification_requirements.go` | typed verification requirements and asset authority |
| `proxy/tools.go` | the new-file route: proposal, staging, policy, delivery |
| `proxy/edit_route_delivery.go` | the edit route, through the same owner |
