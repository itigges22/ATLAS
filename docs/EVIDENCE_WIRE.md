# Evidence wire: the versioned envelope between v3-service and the proxy

The V3 service answers `/v3/generate` with a `passed` boolean, a phase name and
a score. None of those says what was actually demonstrated: `passed` covers a
compile smoke, a partial oracle score and a complete one indistinguishably, and
the proxy had no way to tell them apart. The envelope carries the evidence
itself, versioned, so a consumer can read what happened instead of guessing
from a boolean.

The envelope is now **authoritative for delivery**. A generated candidate may
replace the caller's content, and carry V3 provenance, only when the envelope
is available and self-consistent, its selection concluded a `verified_winner`,
its record is closure-eligible, and its hash names the exact bytes Go would
write. `passed`, `phase_solved`, `winning_score` and the verification-evidence
strings authorize nothing; `passed` stays on the wire as a compatibility and
telemetry field.

Because sanitisation rewrites a candidate after the service earned its
evidence, authorization is re-asked of the final bytes: a hash that no longer
matches revokes to the caller's baseline and withdraws provenance, and the
revocation continues through the same final-byte validation the other gates
use.

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

`EvidenceSupportsProvenanceFor` is the rule, and `v3DeliveryAuthorized` is the
only place it is applied:

| Condition | Why |
| --- | --- |
| non-empty candidate code | nothing to deliver otherwise |
| availability `available` | present, same-major, self-consistent |
| `selection.status == verified_winner` | a winner, not merely a best record |
| `evaluation.closure_eligible` | that winner met its own contract's floor |
| `candidate_content_hash` equals sha256 of the final bytes | the evidence is about what will be written |

The producer's own `describes_delivered_candidate` flag is never trusted — the
consumer hashes what it is delivering. Every other outcome (best-not-eligible,
tied, incomparable, ineligible, no winner, hash mismatch, absent, unknown
version, malformed, contradictory) delivers the caller's baseline with no
provenance.

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

## `evidence.py`: retired

The prototype that predated `contract.py` is **deleted**. Every symbol moved to
the layer that owns it, with no compatibility module, alias or shim left behind:

| Was in `evidence.py` | Now |
| --- | --- |
| `select_adapter`, `js_is_instrumentable`, `extract_inline_script`, `js_probe_source_inline`, `js_probe_source`, `parse_probe_output`, `combine_runs`, the JS harness and its regexes | `adapters.py` — adapter routing and probe mechanics |
| `INTERACTIVE_REQUIRED` / `INTERACTIVE_OPTIONAL`, adapter id constants | `adapters.BROWSER_REQUIRED` / `BROWSER_OPTIONAL`, `adapters.ADAPTER_*` |
| `result`, `result_from_adapter`, `grade_interactive` | `adapters.contract_record`, which builds contract records from raw observations |
| `selection_mode`, `probing_enabled`, `selection_enabled`, `OFF`/`SHADOW`/`ENFORCE` | `pipeline._selection_mode`, `_probing_enabled`, `_selection_enabled`, `MODE_*` — same environment variable, same semantics |
| `may_return_early`, `may_return_early_result`, `at_least`, `rank_key`, `STRENGTH_ORDER` and the prototype strength scale | **deleted** — superseded by `contract.select`, `contract.rank_key` and the contract's own strength ordering |

Sentinels in `tests/v3-service/test_contract_genericity.py` prove the file is
gone, that no Python file imports it, that each moved symbol has exactly one
definition in exactly one owner, that the superseded policy has no definition
anywhere, that mode parsing exists only in `pipeline.py`, and that browser
vocabulary never reaches the generic contract or the pipeline.
