# V3.2 — Architecture-First Planning (RPG-style plan-then-fill over wavelet bands)

> **Removed.** This design shipped behind `ATLAS_RPG_PLANNING` and was deleted
> after measurement. An A/B against the flat planner on the reference local
> model gave **0 improvements, 2 regressions, and roughly 10x planning
> latency** — the outcome bottleneck is the model writing correct code, not the
> plan it writes against. The RPG planner, the wavelet decomposition that fed
> it, and the signature veto and drift-regeneration loops built on top were all
> removed rather than kept alive behind a default-off flag.
>
> The document is kept as the design record and the reason it did not pay off.
> `ATLAS_RPG_PLANNING` remains in the config schema marked deprecated so an
> existing `.env` that sets it gets a clear warning instead of "unknown key".

Tracks **[#120](https://github.com/itigges22/ATLAS/issues/120)**. Builds on the wavelet
decomposition angle from **[#39](https://github.com/itigges22/ATLAS/issues/39)**.

### Post-review hardening

A review pass found two deployment problems and several correctness gaps, all
fixed. The flag is now forwarded to the v3-service container in
`docker-compose.yml` (it was unreachable before, making the feature dead in the
default deployment), and the proposal-stage coarse band is built from the
in-memory `project_context` via `decompose_file_map` rather than a disk scan,
because the v3-service container has no project volume mount. The `/v3/plan`
SSE reader buffer was raised so the larger RPG payload cannot silently drop a
plan, signature parsing now handles Go-style receiver methods and never treats
a bare declaration keyword as a function name, the edit path now runs the same
drift retry and reporting as the write path, `planConstraintsForTarget` prefers
the most-specific matching step, the corrective regeneration accepts a retry
that resolves the targeted signatures, and the topological sort dedups file ids.

## 1. What we're building

Replace ATLAS's flat, free-form planning artifact with a **structured, two-stage,
architecture-first plan** and execute it **plan-coarse / implement-fine**:

1. **Proposal-level planning** — *what* to build: a capability tree (modules →
   components → leaf capabilities).
2. **Implementation-level planning** — *how*: expand leaves into files, classes,
   function signatures, and the data-flow / ordering edges between them. This is
   the **Repository Planning Graph (RPG)** artifact.
3. **Graph-guided generation** — traverse the RPG in topological order, generating
   per node, verifying as we go, threading existing V3 phases (DivSampling, S\*,
   PR-CoT repair) **per node** instead of per whole-file.

The **wavelet decomposition** (from wavescope) supplies resolution bands for free:
the planner reasons at the **coarse band** (module/class structure, "which files
even matter"), the implementer zooms to the **fine band** (function bodies). On L6
tasks (editing existing repos), the coarse band grounds the proposal stage in the
*actual* structure of the repo rather than the model's guess.

### Source grounding

- **RPG / ZeroRepo** — *A Repository Planning Graph for Unified and Scalable
  Codebase Generation*, [arXiv:2509.16198](https://arxiv.org/abs/2509.16198)
  (paper text in repo root `2509.16198v6.md`). Reports +35.8 pts test accuracy,
  +27.3 pts coverage, near-linear scaling vs. a Claude Code baseline on RepoCraft.
  Supplies the **repository / architecture** axis (this work).
- **PlanSearch** — *Planning In Natural Language Improves LLM Search for Code
  Generation*, Wang et al. ICLR 2025
  ([arXiv:2409.03733](https://arxiv.org/abs/2409.03733), paper text in repo root
  `2409.03733v2.md`). Already implemented in ATLAS as `benchmark/v3/plan_search.py`
  (Feature 1A). Searches natural-language plans for a **single problem** to raise
  idea-space diversity. Supplies the **algorithmic / problem** axis — it fills in
  each RPG node, it does not replace RPG. Per the #120 scope clarification: RPG
  plans the repo up front (coarse / module band), then PlanSearch + Derivation
  Chains fill each node (fine / function band).
- **wavescope-mcp** — `~/src/wavescope-mcp`. The wavelet engine we reuse:
  - `src/signal.ts` — per-line structural importance signal (indent + structural
    keywords + decorators, comment/string aware), 14 languages.
  - `src/wavelet.ts` — Ricker (Mexican-hat) CWT at scales `[1,2,4,8,16,32,64,128]`,
    peak detection with cross-scale ridge collapse.
  - `src/context.ts` — `FileContext`: assembles **fine / medium / coarse** bands and
    `getImportantPositions()`.
  - `src/project.ts` — project-wide indexer (gitignore-aware, file caps, LRU cache).
  - `src/diff.ts` — `diff_wavelet_context`: which structural boundaries were
    added / removed / shifted between two git revisions — our **drift detector**.

## 2. How it maps onto today's pipeline

(Reconnaissance reference for every path below: the current pipeline.)

| RPG concept | ATLAS today | What changes |
|---|---|---|
| Proposal-level plan (what) | `Plan{Steps[]}` flat list, `PLAN_PROMPT_TEMPLATE` (`v3-service/main.py:1707`) | New `capability tree` stage feeding a graph |
| Implementation-level plan (how) | — (none; steps are tool calls) | New `RPG` artifact: files + signatures + edges |
| Graph-guided generation | `/v3/generate` whole-file pipeline | Topological per-node traversal |
| Guided localization | `symbol_index` (tree-sitter, ad-hoc) + symbol-name match | Coarse-band "which files matter" + RPG node→file map |
| Structural verification veto (#39 pt 1) | import/class-survival shape check (`v3-service`) | Extend: does generated code realize the **planned** signatures/edges? |
| Plan adherence / revision | `proxy/plan_adherence.go` (off-plan streak → re-plan) | Node-level: drift via `diff_wavelet_context` → re-plan touched nodes |

**Services involved:** Go `proxy` (8090, agent loop + plan orchestration),
Python `v3-service` (8070, plan generation + AST + symbol index), Python
`geometric-lens` (8099, scoring), `sandbox` (test exec), `llama-server` (8080).
Plan generation lives in `v3-service`; the new RPG stages live there too.

## 3. How we reuse the wavelet engine — port to Python (decided)

The wavelet engine is TypeScript; ATLAS is Go + Python. **Decision: port the core
engine into v3-service as a Python module** (`v3-service/wavelet/`), in-process —
no new runtime, no extra container, tightest integration with the planning stages
that consume it.

Port scope (the four files that carry the algorithm, ~1.45 KLOC TS → Python):

| wavescope source | Python target | What it provides |
|---|---|---|
| `src/signal.ts` | `wavelet/signal.py` | per-line structural-importance signal (indent + structural keywords + decorators; comment/string aware) |
| `src/wavelet.ts` | `wavelet/cwt.py` | Ricker CWT at scales `[1,2,4,8,16,32,64,128]`, peak detection w/ ridge collapse |
| `src/context.ts` | `wavelet/context.py` | `FileContext`: fine/medium/coarse band assembly, `get_important_positions` |
| `src/language.ts` | `wavelet/language.py` | per-language structural-keyword tables (14 langs) |
| `src/diff.ts` | `wavelet/diff.py` | peak-profile diff between two git revisions (drift detector) |

**Library vs. hand-roll (surveyed — §6):** no drop-in library preserves
wavescope's calibrated behavior, so we port faithfully and use **`numpy` only**
(already in the stack via `geometric-lens`) to vectorize the convolution. Do *not*
pull in PyWavelets/SciPy for the transform — see §6.

Reuse-fidelity rules so we don't silently fork behavior:
- **Port verbatim, behavior-for-behavior** — same scales, same band radii
  (`fine ±radius/5`, `medium ±radius/2`, `coarse ±radius`), same ridge-collapse
  and dedup semantics. The CWT is plain numerics (`numpy` makes the convolution a
  few lines), so divergence risk is low.
- **Bring the tests across too** — translate wavescope's unit tests (e.g.
  `signal`, `wavelet`, `context`, `diff`) into `pytest` and treat them as the
  port's conformance suite. A handful of golden-output fixtures captured from the
  upstream TS (run wavescope once, snapshot peaks/bands for a few sample files)
  guards against drift.
- **Document provenance** — each ported file headers the upstream path + commit
  so future syncs are traceable.

Everything downstream of Phase 0 is agnostic to this choice — it consumes the same
in-process `FileContext` API regardless.

## 4. Phased plan

Each phase is independently shippable and **behind a feature flag, default off**
(`ATLAS_RPG_PLANNING=0`), mirroring the `PlanSearchConfig.enabled` pattern. No
behavior change until a phase is explicitly enabled and benchmarked.

### Phase 0 — Wavelet substrate (foundation) — ✅ SHIPPED
**Goal:** v3-service can decompose a project / file into resolution bands in-process.

- Ported the engine into `v3-service/wavelet/` (`language.py`, `signal.py`,
  `cwt.py`, `context.py`, `project.py`, `flags.py`) — pure stdlib, no new deps.
- Internal Python API consumed by the planning stages:
  - `decompose_project(root)` → important structural positions across the repo
    (port of `get_important_positions(directory)`), gitignore-aware, file-capped.
  - `FileContext(path).query_wavelet_context(center, radius)` → `{fine, medium,
    coarse}` bands (port of `query_wavelet_context`).
- Env contract `ATLAS_RPG_PLANNING` established (`wavelet.flags.rpg_planning_enabled`,
  default off); the planner gates on it in Phase 1.
- Conformance suite in `tests/v3-service/test_wavelet_*.py` (translated from the
  upstream vitest specs) **plus a golden-fixture test** asserting bit-for-bit
  numeric parity (8 dp) with the actual upstream `wavelet.ts` (captured via
  `npx tsx`). Added `tests/v3-service` to the CI pytest matrix.
- **Deferred to Phase 3:** `diff_context` (port of `diff_wavelet_context`, the
  drift detector) — it's only consumed by the Phase-3 re-plan loop, so it ships
  with its consumer rather than as unused substrate.
- **Exit (met):** v3-service produces a coarse structural map of any repo
  in-process, no pipeline change. In-process port keeps this off the network path
  (issue #39's latency budget).

### Phase 1 — RPG artifact + two-stage planning — ✅ SHIPPED
**Goal:** produce the graph instead of (alongside) the flat plan.

- RPG schema + construction in `v3-service/rpg.py`: `Capability` / `FileSpec` /
  `FunctionSpec` / `Edge` / `RPG` dataclasses, nodes carrying dual semantics
  (capability + file/function), edges carrying `data_flow` / `order`.
- **Stage A (proposal):** `build_proposal_prompt` → capability tree. On L6 it's
  seeded with the Phase-0 coarse band (`decompose_project` labels) so the tree
  maps onto real modules.
- **Stage B (implementation):** `build_implementation_prompt` → files, function
  signatures, and edges → the RPG.
- Tolerant JSON extraction (`extract_json_object`, mirroring `_parse_plan_json`)
  + `validate_rpg` (acyclic via Kahn topo sort, leaf→file coverage, edge
  resolution, parent resolution) + `score_rpg` graph-shape heuristic.
- `flatten_to_plan` projects the RPG onto the existing flat `Plan` (topological
  file order, producers before consumers, verify step last) so the proxy agent
  loop and `plan_adherence.go` are unchanged. Full RPG artifact is attached to
  the `/v3/plan` response under `rpg` for observability.
- Wired into `main.generate_plan`, gated by `ATLAS_RPG_PLANNING`, strictly
  additive — any failure (flag off, modules absent, model output unusable) falls
  through to the flat planner. Dockerfile ships the new modules.
- `complete_fn`-injected construction (LLM-agnostic) → fully unit-tested with a
  fake model: `tests/v3-service/test_rpg.py` (27) + `test_rpg_integration.py`
  (monkeypatched `LLMAdapter`, flag on/off).
- **Exit (met):** with the flag on, planning emits a validated RPG and the agent
  loop runs off the flattened projection; off, behavior is byte-identical.

### Phase 2 — Graph-guided generation (plan-coarse, implement-fine) — ✅ SHIPPED
**Goal:** generate by traversing the RPG, not free-form.

- **Topological traversal comes free from Phase 1's projection:** `flatten_to_plan`
  already emits write_file steps in producer→consumer order, and the proxy agent
  loop already executes them in order (accumulating each written file into the
  context for downstream nodes). No separate driver needed.
- **Per-node fill = the existing problem-level pipeline, not a new one.** Each
  write_file step routes (T2+) to `/v3/generate`, which already runs PlanSearch
  (`benchmark/v3/plan_search.py`, arXiv:2409.03733) + DivSampling / S\* /
  Derivation Chains. Phase 2's job was to make that call *RPG-aware*:
  - `rpg.node_constraints(rpg, file_id)` derives each node's planned interface
    (capability, function signatures, incoming/outgoing data-flow edges).
  - `flatten_to_plan` attaches `node_id` + `constraints` to each write step.
  - Proxy: `PlanStep` gained `NodeID` + `Constraints`; the full graph is parsed
    into `Plan.RPG` (`proxy/types.go`). `planConstraintsForTarget` (`proxy/rpg.go`)
    maps a file path to its node's constraints, threaded into the
    `V3GenerateRequest.Constraints` in `writeFileWithV3` / `improveContentWithV3`.
  This realizes the #120 framing literally: RPG = repo/architecture axis (coarse
  band), PlanSearch = algorithmic axis. PlanSearch's diversity search now runs
  *within* a node whose interface is pinned, not over the whole file.
- **Strictly additive:** flat-planner steps carry no constraints, so
  `planConstraintsForTarget` returns nil and generation is byte-identical when
  `ATLAS_RPG_PLANNING` is off.
- Tests: `test_rpg.py` (node_constraints + step enrichment), `proxy/rpg_test.go`
  (graph + step parsing, path-suffix constraint lookup, copy-safety, nil-plan).
  Full proxy `go test` + 119 python tests green.
- **Deferred:** explicit `structural_edit`-primitive biasing / ASA steering from the
  fixed node target → Phase 3 (rides with the structural-verification work).
- **Exit (met):** end-to-end, an RPG plan generates files in topological order
  with each node's `/v3/generate` call constrained by its planned interface.

### Phase 3 — Graph-guided verification, localization & drift re-planning — ✅ SHIPPED (core)
**Goal:** close the loop with structure.

- **Structural verification veto (extend #39 pt 1):** `rpg.verify_node_realization`
  / `missing_planned_signatures` — reject a sandbox-passing candidate whose
  generated code doesn't define its planned function signatures, not just
  "import survives." Python uses stdlib `ast` (precise, methods included);
  other languages use a keyword+name regex. Wired into the V3 pipeline `run`
  as an **RPG signature veto** right after the existing structural veto:
  flag-gated, recovers planned signatures from the request constraints
  (`planned_signatures_from_constraints`), and is conservative — never empties
  the candidate set, never fires when code is opaque or the flag is off.
- **Localization:** `rpg.localize(rpg, query)` ranks RPG nodes by token overlap
  of capability name + path + function names/summaries — the graph-aware
  replacement for symbol-name-only matching (#39 pt 4).
- **Drift detection:** `wavelet/diff.py` ported faithfully from `diff.ts`
  (`diff_peaks` / `diff_contents`, golden-parity test vs upstream `npx tsx`).
  `rpg.node_drift` compares a node's planned functions against generated code
  and flags `should_replan` when a planned boundary is missing (conservative on
  opaque code). `diff_contents` gives the write-time before/after structural
  diff for the #39 "re-run touched files on write" signal.
- Tests: `test_wavelet_diff.py` (port + golden) and Phase-3 additions to
  `test_rpg.py` (defined_names, signature extraction/veto, verify, drift,
  localize). 145 python tests green; proxy `go test` green.
- **Re-plan loop (drift detection + impact surfacing) — ✅ wired:** the
  `/v3/generate` response now carries `rpg_signature_missing` (planned signatures
  the *winning* code failed to realize — the veto rejects failing candidates
  mid-pipeline, but a winner can still drift when all fell short). Proxy:
  `V3GenerateResponse.RPGSignatureMissing` → after a V3 write, `reportRPGDrift`
  (`proxy/rpg.go`) computes the affected downstream subgraph (`affectedDownstream`,
  BFS over RPG edges, cycle-safe) and emits an `rpg_drift` event naming the
  drifted file + the downstream nodes that may need regeneration. Go-tested
  (path→node lookup, downstream BFS, cycle safety, event emission).
- **Automatic node-local regeneration — ✅ wired:** `regenerateOnDrift`
  (`proxy/rpg.go`) runs one bounded corrective retry when a write drifts — it
  re-calls `/v3/generate` with the missing signatures injected as a hard
  constraint and keeps the retry only if it realizes strictly more of the plan.
  Bounded to a single retry (the V3 pipeline is expensive), no-op when RPG is off
  or there was no drift. Tested against a fake generate server (retry-succeeds,
  retry-no-better, and no-op cases). Wired into `writeFileWithV3`, the RPG
  node-creation path; if drift survives the retry, `reportRPGDrift` still
  surfaces the downstream subgraph.
- **Exit (met):** candidates are vetoed against the planned interface; on write,
  drift triggers a bounded corrective regeneration, and any surviving drift is
  surfaced with its affected subgraph.

### Phase 4 — Evaluation & rollout — ◑ tooling + rollout shipped; live benchmark pending hardware
**Goal:** evidence before default-on.

- **Offline metrics harness — ✅ shipped:** `v3-service/rpg_eval.py` scores RPG
  artifacts on graph quality (parse/valid/acyclic rates, leaf-coverage,
  signature density, score, file/edge/function scale) per-graph and aggregated,
  with a CLI (`python rpg_eval.py artifacts/*.json` / `--jsonl`). Tested in
  `tests/v3-service/test_rpg_eval.py`. This summarizes a benchmark run's plans
  and catches plan-shape regressions in CI without a model.
- **Live RepoCraft comparison — runbook, pending hardware:** the head-to-head
  (RPG-on vs flat planner: functional coverage, test accuracy, code scale, L6
  localization turns) needs a GPU + model, so it runs on the benchmark stack,
  not in this environment. Runbook:
  1. Pick a multi-file fixture set (RepoCraft-style; start with 2–3 small repos).
  2. Run each task twice — `ATLAS_RPG_PLANNING=0` then `=1` — capturing the
     `/v3/plan` artifacts and final pass/coverage.
  3. Feed captured RPG artifacts through `rpg_eval.py` for graph metrics; compare
     end-to-end pass/coverage/scale across the two arms (ablation in the style of
     `docs/reports/V3_ABLATION_STUDY.md`).
  4. Watch latency: two planning stages + per-node constraints add LLM calls —
     gate by tier if the T2 path regresses.
- **Default stays OFF pending that evidence.** `ATLAS_RPG_PLANNING` ships as an
  experimental opt-in (documented in `.env.example` + `CHANGELOG.md`). Flip the
  default only if the live wins replicate — that's the whole point of "evidence
  before default-on." Credit Dmitri (@yogthos) per the issue.
- **Default-flip criteria (proposed):** RPG-on ≥ flat on end-to-end pass rate on
  the fixture set, no T2 latency regression beyond budget, and ≥90% valid/acyclic
  RPGs from `rpg_eval.py` on the captured artifacts.

## 5. Risks / open questions

- **9B model capacity.** RPG/ZeroRepo numbers are from frontier models; a local
  9B may struggle to emit a valid graph. Mitigations: grammar-constrain the RPG
  JSON (GBNF, like existing plans), keep nodes small, lean on the coarse band so
  the model *recognizes* structure rather than *inventing* it.
- **Graph vs. lighter skeleton** (issue open question). Start with the minimal
  graph that still carries edges; don't over-model in Phase 1.
- **Latency.** Two planning stages + per-node generation is more LLM calls. Gate
  by tier — only T2+/L6 tasks get the full RPG; keep the flat planner for small
  edits.
- **Port drift** — the Python port could silently diverge from upstream wavescope.
  Mitigated by the conformance suite + golden fixtures (§3). No drop-in library
  preserves wavescope's calibrated behavior, so we port rather than wrap (see §6).
- **Language coverage.** wavescope covers 14 languages for bands; tree-sitter
  `structural_edit` is Python/HTML today. RPG file/signature fidelity is best where both
  overlap (Python first), generic-fallback elsewhere.

## 6. Library survey — why we port instead of `pip install`

We checked for an existing Python library before committing to a port.

- **`scipy.signal.cwt` / `scipy.signal.ricker`** — *removed in SciPy 1.15*
  (deprecated 1.12). Not an option.
- **PyWavelets (`pywt.cwt(sig, scales, "mexh")`)** — the official, maintained
  replacement; the `mexh` wavelet *is* the Ricker / Mexican-hat. Provides the
  generic transform.
- **Why it's still a port:** wavescope's CWT is a *custom* convolution —
  `1/√a` normalization, **reflect** boundary, `±5a` kernel truncation, integer
  `t/a` sampling — and its peak step is a bespoke magnitude-sorted **cross-scale
  ridge collapse** (`ridgeWindow`). `pywt`'s `mexh` differs in normalization,
  sampling, and boundary, so coefficient magnitudes would shift; every calibrated
  threshold downstream (`min_coefficient` default `0.3`, band behavior,
  `ridgeWindow=2`) is tuned to wavescope's exact scale. Swapping `pywt` in is a
  silent behavior fork, not a faithful port.
- **Decision:** port the transform faithfully from the TS; use **`numpy` only**
  (vectorize the per-scale convolution). `numpy>=1.26` is already a stack
  dependency (`geometric-lens/requirements.txt`), so no new heavyweight import.
  `find_peaks` / `find_peaks_cwt` are *not* used — the ridge-collapse logic is
  bespoke and ports directly.
- **Not needed here:** wavescope's Haar-DWT entropy/complexity-heatmap path
  (`haar.ts`, `wce.ts`) powers `get_complexity_heatmap`, which is out of scope for
  planning bands — skip it (and its `libwce` dependency).

_Sources: SciPy 1.15 release notes (wavelet functions removed); PyWavelets `cwt`
docs / migration guidance._

## 7. First concrete step

Phase 0, the wavelet substrate: port `signal.ts` + `wavelet.ts` + `context.ts` +
`language.ts` into `v3-service/wavelet/` (numpy-backed), stand up the translated
pytest conformance suite + golden fixtures, and confirm v3-service can pull a
coarse structural map of a sample repo in-process within the latency budget — all
behind `ATLAS_RPG_PLANNING`.
