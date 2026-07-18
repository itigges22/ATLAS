# ADR 0003: Direct-mode model-agnostic; V3/Lens/ASA per-model bundles

Status: accepted (encoded in code since V3.1.2; resolves GH #66's core)

## Context
"Model-agnostic" was ambiguous: the agent loop genuinely is (GGUF
--jinja templating, no name-keyed behavior, probed dims), but Lens
scoring and ASA steering are trained against a specific model's
residual stream and are wrong — not just suboptimal — on another model.

## Decision
The contract is: any llama.cpp-loadable GGUF runs the direct agent
path; the V3 scoring/steering stack requires the model's own bundle,
enforced at load (model_identity.json + embedding-dim checks; ASA
sidecar marker at llama-server boot). Mismatched bundles are rejected
and the lens reports itself disabled — never silently misapplied.
Registry entries carry per-file hashes and honest lens/asa status.

## Consequences
"Any model, full stack" is not claimed (SUPPORT_MATRIX.md § Model
contract). Onboarding a new model = the documented bench → lens build →
asa build loop. Remaining #66 scope (LLMBackend abstraction, MoE
hidden-state extraction) is roadmap, not a correctness gap.
