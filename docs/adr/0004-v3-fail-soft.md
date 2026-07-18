# ADR 0004: V3 failures fall back to the model's own content

Status: accepted (V3.1.x behavior, E2E-pinned 2026-07)

## Context
The V3 pipeline (candidates, scoring, selection) sits between the
model's proposed write and the disk. It can be unavailable, slow, or
return garbage.

## Decision
V3 errors, timeouts (ATLAS_V3_TIMEOUT, default 180s), and malformed
responses fall back to writing the model's own content directly, with
the fallback visible (logged, v3_used unset in the tool result) — never
a silent skip, never a hard turn failure. Rationale: the model's
content already passed the syntax/guardrail gates, so the fallback is
safe, and an unavailable enhancement layer must not brick the product.

## Consequences
Users on a degraded stack silently lose the quality uplift but keep a
working agent; the tool result and events make the degradation
observable. Pinned by tests/e2e/test_v3_lens_acceptance.py failure
modes.
