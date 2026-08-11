# Invariant 2 design: mutation outcome vs validation outcome

Investigation only. Nothing implemented.

## 1. Write-path matrix (measured, not inferred)

Black-box through `runAgentLoop`, proxy/write_path_matrix_test.go:

| tier | V3 | new/overwrite | validation | bytes land | prior bytes |
|---|---|---|---|---|---|
| T1 | absent | overwrite | **none run** | yes | **DESTROYED** |
| T1 | configured | overwrite | **none run** | yes | **DESTROYED** |
| T2Medium | configured | overwrite | regression gate | no | preserved |
| T2Medium | configured | new, invalid | run, failed | **yes (warned)** | n/a |
| T2Medium | configured | new, valid | run, passed | yes | n/a |
| T2Medium | absent | any | none run | yes | untested |

Gate condition, tools.go:855:

    if fileTier >= Tier2Medium && ctx.V3URL != "" && !ctx.BypassV3 && !iterating {

It is an AND. A small file misses protection even with V3 configured — proven
by the T1/with-V3 cell failing identically to T1/no-V3.

Second gated path, tools.go:901 (`iterating`, via isActiveDebugIteration):
runs the same healthy->broken rule without needing V3. Small overwrites
outside an active debug iteration reach neither.

## 2. Intent, assessed separately per the four questions

* **Invalid NEW file applied with a warning: INTENTIONAL.** tools.go:857-866
  documents it — "a NEW file lands with a warning so the model can run it and
  read the real traceback" — and `writeNewFileWithWarning` exists for it.
* **Repairing an already-broken file: INTENTIONAL.** tools.go:869-878 records
  two measured sessions where refusing an imperfect fix left the broken
  version on disk.
* **Skipping validation on small overwrites: INTENTIONAL AS A TIER POLICY,
  its consequence NOT considered.** The comment at tools.go:912-919 says the
  T0/T1 direct path "carries no syntax gate" deliberately, to avoid
  hard-blocking content a strict checker rejects both before and after. That
  reasoning is about avoiding false rejections; nothing anywhere states that
  destroying known-good bytes is acceptable.
* **Destroying known-good bytes: NOT INTENTIONAL.** Every protective comment
  frames the goal as protecting WORKING code ("The strictness on existing
  files protects WORKING code"). The T1 path defeats that goal for exactly
  the files it was meant to cover.
* **`Success` was designed as tool-invocation success.** It predates the
  distinction and is now overloaded — see below.

## 3. `Success` consumer classification (20 production sites)

What each site actually needs:

| site | needs |
|---|---|
| tools.go:895, 978, 1195 | mutation applied (records SessionWrites) |
| tools.go:1160 | tool invocation completed (early return) |
| tools.go:2269 | transport success of an AST sub-call |
| agent.go:1627, 1909 | tool invocation completed (error routing) |
| agent.go:1656, 1669 | transport/tool success (serialized to the client) |
| agent.go:1698 | mutation applied (write bookkeeping) |
| agent.go:1728 | mutation applied (literal-contract check) |
| **agent.go:1764** | **completion authorization** — sets madeProductiveChange |
| agent.go:1791, 1809, 2055 | tool invocation completed (run_command) |
| agent.go:1870 | task progress (plan adherence) |
| agent.go:1887 | tool invocation completed (edit retry) |
| agent.go:1999 | task progress (work-vs-conversation signal) |
| **agent.go:2113** | **completion authorization** — mutation for done gate |
| agent.go:2151 | validation passed + provenance (V3 verified phase) |

Two sites conflate mutation with completion authorization: **agent.go:1764**
and **agent.go:2113**. Both include `delete_file`, which is how a successful
delete authorized `done` over an empty workspace in the archived pair-1 trace.

Compatibility constraint: agent.go:1656 and 1669 serialize `"success"` into
the client event stream. That wire field must keep its present meaning
(tool invocation completed) or every consumer of the SSE contract breaks.

## 4. Proposed contract (for review, not implemented)

Keep `Success` as-is for transport. Add orthogonal fields:

    Mutation   MutationOutcome  // none | applied | refused
    Validation ValidationResult // notRun | notApplicable | passed | failed
    ValidationDetail string     // the rejection/warning text
    Deliverable bool            // may this result authorize completion

Rules the fields must express, none collapsible to one boolean:

* invalid new file: Mutation=applied, Validation=failed, Deliverable=false
  (preserves the intentional debugging policy without letting it authorize done)
* invalid overwrite: Mutation=refused, Validation=failed, prior bytes intact
* validation skipped: Validation=notRun — never `passed`
* non-code file: Validation=notApplicable
* delete: Mutation=applied, Validation=notApplicable, Deliverable=false

Splitting `Success` alone is insufficient: the T1 overwrite destroys good
bytes BEFORE any field is set. Invariant 2 must also close the validation
gap so the T1 overwrite path is refused, not merely labelled.

## 5. Acceptance tests (production entry point only)

1. valid new code file: applied + passed + deliverable
2. invalid new code file: applied + failed + NOT deliverable, bytes on disk
3. valid overwrite: applied + passed, new bytes on disk
4. invalid overwrite preserves prior bytes — T1/no-V3, T1/V3, T2/V3
5. validation unavailable: notRun, never passed
6. non-code file: notApplicable
7. warned/unvalidated content cannot authorize done
8. delete of a session-created deliverable cannot authorize done
9. existing suites stay green (SSE `"success"` unchanged)
