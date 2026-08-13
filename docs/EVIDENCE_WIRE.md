# Evidence wire: the versioned envelope between v3-service and the proxy

The V3 service answers `/v3/generate` with a `passed` boolean, a phase name and
a score. None of those says what was actually demonstrated: `passed` covers a
compile smoke, a partial oracle score and a complete one indistinguishably, and
the proxy had no way to tell them apart. The envelope carries the evidence
itself, versioned, so a consumer can read what happened instead of guessing
from a boolean.

This phase is **transport only**. No decision changed: candidate selection,
delivery authorization, `passed`, early return and every gate behave exactly as
before. The proxy decodes, validates and records the envelope; nothing reads it
to authorize anything yet.

## Ownership

| Layer | Owns | Must not |
| --- | --- | --- |
| `v3-service/contract.py` | The domain: vocabulary, coverage arithmetic, comparability, ranking, closure policy, and the wire envelope that renders them | — |
| `v3-service/pipeline.py` | Orchestration: runs verifiers, supplies the selected record | Decide wire shape |
| `v3-service/adapters.py` | Adapter knowledge: which criteria an adapter can observe, what its grading means in contract terms, and the live-record → contract-record bridge | Decide closure, ranking or selection |
| `v3-service/main.py` | Transport: calls `adapters.evidence_envelope`, writes its output | Hold any policy — pinned by `test_main_serialises_but_decides_nothing` |
| `v3-service/evidence.py` | Compatibility only, pending retirement | Gain any new behaviour — pinned by `test_no_new_behaviour_was_added_to_the_retiring_prototype` |
| `proxy/types.go` | The one Go wire representation, beside `V3GenerateResponse` | — |
| `proxy/v3_bridge.go` | The response boundary: decode plus strict validation and availability | Re-derive domain policy, or infer strength from `passed`/`phase_solved`/`winning_score`/`verification_evidence` |

No separate evidence module exists on either side. One contract, one serialiser,
one wire type, one validator — pinned by
`test_one_canonical_contract_one_serialiser_no_duplicate_policy`.

Local `ToolResult` validation (syntax, structural) stays completely separate:
it says what **the proxy** checked about the bytes it wrote. The envelope says
what **the service** demonstrated about a candidate. Merging them would let a
local syntax pass read as behavioural evidence.

## Envelope

```
wire_version            transport shape (major-compatible; unknown major = unavailable)
record_schema_version   contract.SCHEMA_VERSION, versioned independently
identity                contract id+version, adapter id+version or calibration id,
                        artifact scope, evaluation-context hash, candidate-content hash
evaluation              execution status, supported, evidence strength,
                        requirements complete, closure eligible, quality scores
coverage                required / demonstrated / missing / unmeasurable, optional observations
selection               status, reason, tied / incomparable / ineligible counts
delivery                delivered-content hash, describes_delivered_candidate
```

Criterion ids are opaque strings end to end. No task vocabulary appears in the
schema or in the policy; `test_generic_contract_stays_prompt_agnostic` enforces it.

Candidate evidence (`evaluation`, `coverage`) and selection evidence
(`selection`) are separate objects, because collapsing them is how "the best of
a bad pool" becomes "verified". A best record that is not closure-eligible is
`selection.status = best_not_closure_eligible` with
`evaluation.closure_eligible = false` — and `passed` is untouched.

## Absent, unavailable, available

Three states, never two:

- **absent** — no envelope was sent: a legacy service, or a run that measured
  nothing. `evidence` is omitted entirely.
- **unavailable** — an envelope arrived and cannot be trusted: unknown wire
  major, incomplete identity, unknown enum value, or an internal contradiction
  (closure claimed over a non-`ok` execution, over incomplete requirements, for
  an unsupported artifact, or a verified winner without closure eligibility).
  Never "failed": nothing about the candidate was demonstrated either way.
- **available** — structurally valid and internally consistent. Says nothing
  yet about whether it describes the bytes being delivered.

The service never emits a malformed envelope: a record that cannot be
serialised is sent as no evidence plus `evidence_unavailable_reason`. The Go
validation exists for buggy or future producers, and the golden fixtures
include three damaged envelopes so that path is exercised.

## Hashes before provenance

`EvidenceSupportsProvenanceFor` is the rule a later slice will use: the
envelope must be available **and** its `candidate_content_hash` must equal the
sha256 of the exact bytes about to be written. The producer's own
`describes_delivered_candidate` flag is never trusted on its own — the consumer
hashes what it is delivering.

## Golden fixtures

`v3-service/testdata/evidence_wire_cases.json` is one document holding all 13
cases: id, description, the exact response body written by the real serialiser,
and what both sides must conclude (availability, strength, selection status,
whether the evidence describes the delivered bytes, and the reason an
unavailable envelope must give). It is built and verified byte-for-byte by
`tests/v3-service/test_contract_genericity.py`; regenerate with
`ATLAS_WRITE_EVIDENCE_FIXTURES=1 pytest tests/v3-service/test_contract_genericity.py`.
The Go tests (`proxy/contract_gate_test.go`) read the same document, decode
those exact bytes and check the declared expectations independently — so the
two languages agree with the contract rather than with each other, and adding a
case on the Python side automatically binds the Go side to it.

## `evidence.py` retirement inventory

`evidence.py` is the prototype that predates `contract.py`. It is **not deleted
in this phase**. Its live importer set is pinned by
`test_evidence_py_importers_are_inventoried`, so a new one cannot appear
unnoticed. It is currently `{pipeline.py, adapters.py}` — the second is the
bridge below, which exists only until step 1 of the cutover.

| Live use | Belongs in | Cutover |
| --- | --- | --- |
| `select_adapter`, `js_is_instrumentable`, `extract_inline_script`, `js_probe_source_inline`, `parse_probe_output`, `combine_runs` | adapter-specific probing, beside its adapter | Move with the browser probe adapter; no policy involved |
| `result`, `result_from_adapter`, `grade_interactive` | adapter → contract record construction | Replace with `contract.build` calls emitted by each adapter; `adapters.contract_record` is the interim bridge and disappears with it |
| `INTERACTIVE_REQUIRED`, `INTERACTIVE_OPTIONAL` | the adapter's declared capabilities | Become the adapter's requirement/capability declaration; the ids stay opaque above it |
| `STRENGTH_ORDER`, `at_least`, `rank_key` | `contract.py` (already has `STRENGTH_ORDER`, `rank_key`, `select`) | Delete on cutover — duplicated policy, and the two scales differ (`behavioral_partial`/`behavioral_complete` vs `behavioral`/`oracle`) |
| `may_return_early`, `may_return_early_result`, `selection_mode`, `selection_enabled`, `probing_enabled` | orchestration policy → `pipeline.py`, with the floor read from the contract | Requires the early-return decision to move onto contract records; **this is the first behaviour-changing slice and is out of scope here** |

Cutover order, each with its own evidence:

1. Adapters emit `contract.build` records directly; `adapters.contract_record`
   becomes a pass-through and is deleted, taking `adapters.py`'s `import
   evidence` with it.
2. Early-return and shadow-selection read contract records; `evidence.py`'s
   ranking and strength scale are removed.
3. Probing helpers move beside the browser adapter.
4. `evidence.py` is deleted once the importer sentinel is empty of production
   modules.

No deletion happens until that sentinel proves no production path imports it.
