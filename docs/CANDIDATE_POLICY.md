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

**There is no service-certification path at all.** A request that declared no
output knowledge used to deliver on the service's own closure verdict — the
producer of a candidate certifying that candidate. It read as a compatibility
rule and it was a self-certification bypass: no target the client named, no
obligation, no floor, and an authority that came from the side being checked.
It is gone, and nothing replaced it. Such a request retains the model's own
bytes under strict.

## Structured mutation scope

Removing that bypass would make the pipeline useless for the interactive case
if nothing else changed: the normal TUI knows neither an output path nor a
command. Inferring a contract from the prose is the wrong answer — a
seventy-character window before a filename is how an input file once became a
deliverable.

The model's own tool call is the third option, and it is structured rather than
inferred. `write_file`, `edit_file`, `insert_after`, `replace_lines` and
`structural_edit` each name a canonical target and bound a mutation in fields.
`deriveMutationScope` reads that off the call: the tool, the canonical target,
the pre-call bytes and the caller's own result, plus the workspace and target
generations.

What a scope is not is evidence. It says WHERE a candidate may act and nothing
about whether it is any good:

- it cannot expand a path, change a target, authorize a deletion or weaken a
  permission;
- it mints nothing — `mintAuthorizationGrant` refuses without one, which is the
  only direction it acts in, and every other condition still has to hold;
- a candidate outside the boundary its own call defined fires the
  `outside_structured_mutation_scope` veto.

It fails closed on an unknown tool, a path that does not resolve inside the
workspace, a spelling the two resolvers disagree about, a missing identity, a
deletion, a moved target, a moved workspace, and a request that has ended.
Every grant now carries the `MutationScopeID` of the call it came from.

A calibrated advisory decision may one day mint a one-time grant for candidate
bytes constrained to exactly that scope; confirm may do the same after an
exact-byte approval. Neither delivers today.

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

### Candidate Confidence Calibration

The measurement that has to exist before advisory may change what lands, frozen
and awaiting authorization under `redteam/runs/candidate-confidence-calibration`.
Planner-only against full V3 with the advisory decision still shadow-only, so
candidate bytes never reach a live task workspace. One feature, one direction,
one threshold, derived on a calibration subset that shares no task family with
the holdout, frozen, and evaluated on that holdout exactly once. Gemma first;
Qwen later as an independent replication that is never pooled and never used to
select the threshold.

Its pre-registered sample size is thirty candidate-eligible pairs per subset,
and the derivation refuses rather than fitting when the corpus is short. It is
short: the existing frozen corpus is sixteen paired cases and produced zero
candidate-eligible proposals, so a corpus has to be built before a threshold
can honestly be derived.

## Pre-registered product policy

Recorded before any calibration is run, so the result cannot be read backwards
into the decision:

- The intended eventual interactive TUI default is **advisory, and only after
  calibration and independent validation**.
- Until then **strict remains the default**.
- **Confirm is an explicit user-selected fallback**, not a prompt on every
  normal edit.
- **Destructive operations keep their existing explicit permission flow.**
  Advisory confidence is not a permission.
- **Human review of the final diff remains part of the product.** ATLAS does
  not claim a universal correctness oracle.
- No advisory confidence value may be described as a probability of
  correctness unless a calibration supports that interpretation.

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
