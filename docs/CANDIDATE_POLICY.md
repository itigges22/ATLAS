# Candidate policy: strict, advisory, automatic_v3

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
| `automatic_v3` | no oracle exists, and the pipeline picked a winner | the exact candidate V3 selected, once every hard safety requirement holds |

`strict` is the shipped default. **`advisory` computes its answer and delivers
nothing** until [Candidate Confidence Policy
Validation](#candidate-confidence-policy-validation) has measured it; see
"Calibration status" below for why. `automatic_v3` is selectable by a trusted
client or operator and is not the default until live validation passes.

## Automatic delivery, and why it is not a lowered bar

Strict asks whether trusted evidence meets a floor the client declared. A
request that declared none has no floor, so the honest strict answer is to keep
the model's own bytes — and a pipeline that generates K candidates, ranks them
and selects a winner never reaches the artifact at all. That is an evidence
question being answered as if it were a safety question.

`automatic_v3` separates them. The evidence question keeps its honest answer:
nothing about the candidate is claimed to be correct, and no score, consensus,
Lens value, service verdict or hidden evaluator is consulted. What is checked
is everything that was never about evidence:

- V3 generation completed, and the candidate is the **exact selected winner**,
  identified by content hash rather than reconstructed from a score or an
  array position;
- request, invocation, route-entry and candidate identities are complete and
  attributable, and the candidate instance id separates duplicate bytes and
  hash-prefix collisions;
- the bytes are non-blank and materially different from the baseline;
- the canonical target is valid, inside the workspace, and grounded: either
  the client declared it as an expected output, or -- when the client declared
  no outputs at all -- it is the exact canonical target of the model's own
  structured mutation call (see "The structured mutation target" below);
- the structured mutation scope admits these exact bytes;
- no undeclared path is added or altered, and no protected asset is mutated;
- workspace, target, route, baseline and candidate identity are all fresh;
- language and artifact class match the target;
- applicable syntax and structural checks pass;
- **no declared verification failed**, and none was left unobservable;
- the candidate is not weaker than the baseline on an applicable trusted check;
- nothing timed out, was cancelled, or exhausted memory, processes or output;
- a destructive operation still goes through its existing permission flow;
- the one-time exact-byte grant, revalidation, write, ledger, validation,
  settlement and provenance all succeed.

**Absence of an oracle is not failure.** No behavioral oracle, no declared
command, an unsupported adapter, no closure certificate, no independent critic:
each is *unavailable* evidence, recorded as such, and none of them rejects an
otherwise safe selected candidate. A requirement the client **did** declare is
different — a failed check is a hard veto, and one that timed out, was
cancelled, exhausted a resource or could not be observed cannot authorize a
delivery. Applicable syntax or structural failure remains a hard veto with or
without a declared command.

**The user's involvement does not change.** The existing permission prompts
still gate dangerous tools, deletion keeps its exact-object approval flow, and
what lands is reviewed as an ordinary workspace diff that can be revised or
undone. There is no candidate approval prompt, no allow/deny request, no
session approval and no candidate confirmation UI: the competition between
candidates is internal, and asking a person to adjudicate it would be asking
them to review work they cannot see.

### The structured mutation target

The ordinary interactive request declares no outputs. A person typing into the
TUI has told the client nothing structured about which files the task requires,
and the TUI sends none rather than guess them from the prose. Until now that
left `automatic_v3` with nothing to bind a delivery to, so the selected
candidate never reached the artifact for exactly the traffic the pipeline
exists to serve.

What such a request does have, once the model acts, is a structured tool call:
`write_file` or one of the edit tools, with a canonical path in its parsed
arguments. That path is the **structured mutation target**. Under
`automatic_v3`, for a `work` request that declared no outputs, it grounds the
delivery of the selected candidate to that one path. It comes from the parsed
call and from nothing else: not the user's prose, not the model's prose, not a
filename in a message, not a plan or a summary or a Lens output.

It is a narrow thing on purpose:

- it is not an obligation. It never becomes an expected output, never enters
  completion, and never says a file the user asked for exists;
- it authorizes no other path, no additional file, and no deletion, move,
  rename or command; the existing permission flow for dangerous tools is
  untouched and cannot consult it;
- declared outputs are never widened by it: a request that named outputs is
  bound to those outputs on every basis;
- strict and advisory never use it, and a `question` request can create no
  mutation authority at all;
- every hard veto, the one-time grant, the exact-byte comparison, the disk
  re-read and the settlement apply exactly as they do for a declared target.

The grant records which grounding it used (`declared_output` or
`structured_mutation_target`), and structural tests pin that only the
authorization owners read it.

### The withdrawn `confirm` mode

An earlier draft carried a fourth mode, `confirm`, which would have presented a
candidate for a one-time exact-byte approval. It never shipped: it had no
approval surface, no wire consumer, no TUI consumer, it never set `Delivers`,
and it appears in no sealed evidence. It is removed rather than quarantined,
and no replacement human-confirmation mechanism takes its place — the product
decision is that candidate competition is internal.

## Who selects the mode

The **validated client request** (`task_contract.candidate_policy`) or **trusted
operator configuration** (`ATLAS_CANDIDATE_POLICY`), in that order, defaulting to
`strict`. The model cannot select it and neither can the V3 service:
`candidatePolicyOf` reads the validated contract and the process environment and
nothing else, and a structural test asserts it names no service or model type.

In the TUI the user selects it with `/candidate-policy strict|advisory|automatic`,
a session-wide control shown in the header and reset by a new session; the
default sends nothing and is read as `strict`. An unrecognised value in a request
is refused at the boundary. An unrecognised
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

## When the producer is not consulted at all

Before any of the above runs, both byte-producing routes decide whether to ask
the pipeline for candidates. That decision is a **cost** rule, not a safety one,
and it is made per mutation:

| Reason | Predicate |
| --- | --- |
| `file_tier_below_threshold` | the file classifies below T2 (under 10 lines, or a config/data/style/doc class) |
| `edit_below_complexity_floor` | an edit whose result is under 80 lines with cyclomatic complexity under 8 |
| `producer_not_configured` | the session has no `V3URL` |
| `generation_disabled` | the session's V3 mode is not full |
| `active_debug_iteration` | the session wrote this file and just watched it fail a run |
| `proposal_failed_syntax_guard` | a syntax or structural guard answered before the producer could be asked |
| `internal_unclassified` | fail-closed: a skip nobody taught the vocabulary about |

`writeGenerationBypass` owns the new-file answer and `editGenerationBypass` owns
the answer for the four edit tools; a structural test keeps the conditions out
of the routes themselves. Each skip writes one `candidate_generation_bypass`
capture record carrying the request, the tool, the reason and the two predicate
inputs that decided it — path-free and content-free, like every capture record.

This matters for measurement as much as for cost. A skipped mutation mints no
route entry, so before these records existed it produced no telemetry of any
kind: an outcome-blind pilot could see that a family yielded no candidate and
could not see whether the pipeline had declined, failed, or never been asked.
Nine of twenty-four families in the corrected eligibility pilot ended exactly
there, and the two cost rules account for seven of them.

The rules themselves are unchanged. A cost rule that skips generation is not the
same statement as a policy rule that refuses delivery, and the two thresholds
above were set for latency on small files, not for confidence in candidates.

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
bytes constrained to exactly that scope. It does not deliver today.

Hard proposal requirements survive unchanged, because they are about bytes being
usable rather than proven: materially different, valid identity, correct file
class, no language swap, no edit-boundary violation, nothing malformed, and no
target or workspace mutation during staging.

## How a route entry ends

Separate from the policy decision above, and answered at a different moment:
the policy says what was decided about a candidate, the routing disposition
says how the route entry that carried it ended. `skipped_infeasible`,
`producer_unavailable`, `producer_timed_out`, `cancelled`,
`no_candidate_produced`, `candidate_not_closure_eligible`,
`candidate_revoked_by_gate`, `baseline_retained`, `authorization_refused`,
`candidate_authorized`, and the fail-closed `internal_unclassified`.

`baseline_retained` means the producer offered nothing materially different, so
there was never a candidate; the record names no candidate hash, because the
only bytes in play are the caller's own. A candidate that WAS offered and then
withdrawn by a gate ends as `candidate_revoked_by_gate` and names the hash of
the bytes that were withdrawn. Reporting both as a retained baseline, over the
hash of whatever ended up on disk, is how the model's own content came to be
recorded under `candidate_hash`.

## The decisions

Closed vocabulary. Every value is a statement about what happened, never about
how likely the candidate is to be correct.

| Decision | Meaning |
| --- | --- |
| `baseline_retained` | the model's own proposal is what lands |
| `candidate_preferred_advisory` | bounded evidence prefers the candidate; not proof, and not a delivery in this build |
| `candidate_authorized_strict` | trusted declared verification passed at the declared strength on these exact bytes |
| `candidate_automatic_v3` | the V3 selection path chose these exact bytes and every hard safety requirement held |
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
- **`automatic_v3` is an explicit trusted-configuration choice**, not a prompt
  on every normal edit and not something a request's prose can select.
- **Destructive operations keep their existing explicit permission flow.**
  Advisory confidence is not a permission.
- **Human review of the final diff remains part of the product.** ATLAS does
  not claim a universal correctness oracle.
- No advisory confidence value may be described as a probability of
  correctness unless a calibration supports that interpretation.

## The acquisition control

An outcome-blind acquisition — an eligibility pilot, a calibration run — has one
invariant that outranks what it is measuring: **candidate bytes must never enter
the active task workspace.** A delivered candidate changes what the model sees
next, how many routes the task takes, which terminal it reaches and what
evidence exists at the end, whether or not anyone runs an evaluator afterwards.

`ATLAS_CANDIDATE_CAPTURE_ONLY` is that control, and it is **not a fourth policy
mode**: strict, advisory and automatic_v3 are what a client may ask for and they are
unchanged. It is operator configuration on a private experimental process,
default off, unreachable from any task contract, model output, service response
or header, and failing closed to ordinary behaviour on any value it does not
recognise.

It sits at the one place a candidate grant is created — inside
`authorizeCandidateDelivery`, after the authorization decision and immediately
before `mintAuthorizationGrant` — and a structural test pins that there is
exactly one minting caller and that the control is consulted before it.

What it suppresses is the licence, not the answer:

- the policy runs, the declared commands run against the exact staged candidate
  bytes, and the hard vetoes fire, all unchanged;
- no grant is minted, none is consumed, and the baseline stays on disk;
- the decision is recorded as what it was — `candidate_authorized_strict` stays
  `candidate_authorized_strict` rather than being flattened into
  `baseline_retained`;
- two private records carry it: the suppression, and the would-have disposition
  from the closed set `would_authorize_strict`, `would_prefer_advisory`,
  `would_deliver_automatic_v3`, `rejected_hard_veto`,
  `insufficient_confidence`, `baseline_retained`,
  `capture_only_suppressed_delivery`;
- nothing model-facing mentions it, and no extra model turn results.

With the control off, a trusted strict candidate still earns its grant and
lands, byte for byte as before.

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
| `proxy/automatic_delivery.go` | whether the exact selected candidate may land, and the grant basis |
| `proxy/verification_requirements.go` | typed verification requirements and asset authority |
| `proxy/tools.go` | the new-file route: proposal, staging, policy, delivery |
| `proxy/edit_route_delivery.go` | the edit route, through the same owner |
| `proxy/candidate_reachability.go` | whether the producer is consulted at all, and why not |
| `proxy/automatic_delivery.go` | whether the exact selected candidate may land automatically, and the grant basis |
