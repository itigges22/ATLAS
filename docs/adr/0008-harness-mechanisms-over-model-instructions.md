# ADR 0008: Harness mechanisms over model instructions

Status: accepted 2026-08

## Context
Nine consecutive dogfooding runs of one task against
`gemma-4-12b-it-Q4_K_M` produced zero complete, correct results. The
harness defects those runs exposed are fixed (see CHANGELOG, the
2026-08-01/02 gate work). What remained was model behaviour, and the
runs separate cleanly into two kinds of intervention:

- Every **mechanism** added — syntax gates, the embedded-script gate,
  the stopped-render-loop and duplicate-binding checks, the node-size
  precondition — fired correctly and prevented the failure it targeted.
- Every **instruction** improved — better rejection wording, a more
  precise verification message, tool guidance naming the right tool —
  was ignored at least once. Run 9 re-sent a byte-identical tool call
  against a rejection that named the file, the line, the cause and two
  concrete fixes.

Two 2026 results frame the same split. arXiv:2605.00334 (AgentFloor, 16
open-weight models 0.27B-32B, 16k+ runs) found no prompt-side lever that
transferred across models, and one structured-decomposition prompt that
regressed every model tried. arXiv:2605.22166 (Life-Harness, training-free,
evolved on Qwen3-4B and frozen) improved 116/126 model-environment settings
across 18 backbones, and its own ablation attributes the largest drops to
its two non-prompt layers — action validation and trajectory regulation.

Citations verified against arXiv 2026-08-02. Note that arXiv:2606.01522,
cited elsewhere as the calibration for a retry cap, is about error-message
detail and repair success and says nothing about retry limits; the
byte-identical rule below rests on our own run data and on determinism,
not on that paper.

## Decision
Interventions that change what the model **can** do, or what it is
**shown**, are shipped unconditionally and are expected to be
model-agnostic. Interventions that ask the model to **behave**
differently are treated as per-model configuration, and are never the
sole mechanism protecting a correctness property.

Three changes follow from it directly, all model-agnostic:

1. **A byte-identical re-send of a rejected tool call is refused before
   it executes** (`identicalRetryRefusal`). The harness is
   deterministic, so the same call against the same workspace produces
   the same rejection. Scoped to calls that failed — re-reading a file
   after editing it is byte-identical and correct — and cleared when the
   same call later succeeds. This replaces nothing: the existing
   repetition detector needs three occurrences and steers the following
   turn, which an identical pair never reaches.

2. **A completion claim the run cannot support is replaced by a
   harness-authored summary** (`unverifiedSummary`). The verification
   gate bounces `done` three times and then lets it through; the user
   saw the model's claim. The harness now states what was written and
   that nothing verified it, keeping the model's account labelled as
   unverified.

3. **`outline_file` reports embedded-language regions**
   (`embedded_region_outline`), reusing the block extraction the
   embedded-script gate already performs. The host grammar cannot see
   into a string literal, so the outline of a Flask app whose UI is one
   template named `function:index` and nothing else. It now names the
   `<script>` region, its line range, and the functions inside it,
   together with the fact that no selector reaches them.

Making `done` ungrammatical was considered and rejected: it requires
strict schema-GBNF, and Gemma-family models require
`ATLAS_GRAMMAR_MODE=loose` or they emit `done` instead of calling tools
at all. Rewriting the summary achieves the same user-facing property
without depending on the grammar mode.

## Consequences
Per-model configuration is now an explicit category rather than an
accident. Grammar strictness, sampler profile, reasoning mode and quant
choice already live in `.env`; the direction of travel is a per-model
harness profile alongside the existing per-model Lens/ASA bundles (ADR
0003), populated by a probe run at model-registration time. That is not
built here, and should not be until a probe has demonstrated it predicts
something — the probe-to-policy mapping is an engineering bet, not a
validated design, and Life-Harness got its transfer result by evolving a
harness against traces rather than by running a fixed battery.

None of this is expected to make a weak model complete tasks it
otherwise fails. It reduces wasted turns and stops unsupported success
claims reaching the user. Task completion remains bounded by the model,
which is what the quant and control-vector work addresses.
