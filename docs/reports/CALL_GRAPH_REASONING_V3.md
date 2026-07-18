# Structural code reasoning — precomputed call graph + reachability

Tracks **[#39](https://github.com/itigges22/ATLAS/issues/39)**. Reference
implementation: **chiasmus** (`~/src/chiasmus`), Dmitri Sotnikov's tree-sitter +
solver engine, which the issue already credits as the inspiration.

## 1. What we're building

ATLAS shipped "v1 plus four follow-up points" of #39 in V3.1, but the
architectural ask in the title, a call graph backed by tree-sitter that can
answer transitive reachability, never landed. The maintainer's own honest read
of the issue lists what is missing:

- No precomputed call graph stored anywhere; tree-sitter walks happen ad-hoc per
  request.
- No transitive reachability. The "reachability slice" is symbol-name match, not
  graph traversal across call edges.
- The structural veto is a shape check ("does the import exist, does the class
  def survive"), not "do the called functions resolve to reachable definitions."
- No Prolog / Datalog / Z3 anywhere.

This work builds the real layer: a **precomputed, cached call graph** with
**native reachability queries**, and then rewires the four shipped integration
points to use it instead of their shallow stand-ins. A logic solver is an
optional final layer, not the foundation (see §3).

## 2. The chiasmus lesson that reframes the issue

The issue title says "tree-sitter + solver", and the help-wanted ask is for
Prolog/Datalog/Z3 contributors. But chiasmus, the cited reference, does **not**
use its solver for call-graph reachability. It computes callers, callees,
reachability, paths, impact, dead code, and cycles **natively in JS in O(V+E)**
(`src/graph/native-analyses.ts`: `reachability` is BFS, `cycles` is Tarjan SCC,
`callers`/`callees` are reverse/forward BFS). It emits Prolog facts
(`src/graph/facts.ts` `graphToProlog`) only so an optional `chiasmus_verify`
step can run custom rule queries; the everyday structural questions never touch
a solver.

That is the right shape for ATLAS too. Native graph traversal is deterministic,
fast, dependency-free, and trivially correct, which matters for a local tool on
the proxy's latency budget. The solver earns its place only for queries native
traversal can't express cleanly (custom rule-based reachability under
constraints, "which calls satisfy property X"). So we lead with the graph and
treat the solver as an additive layer that reuses a facts artifact we emit
early. This also de-risks the project: the user-visible wins (points 1-4) land
without ever depending on a Prolog/Z3 runtime.

## 3. Source grounding

- **Issue #39** — the five integration points and the "what's missing" table.
- **chiasmus** `~/src/chiasmus/src/`, the design we model on:
  - `graph/types.ts` — the `CodeGraph` data model: flat `defines` /
    `calls` / `imports` / `exports` / `contains` fact tuples. Language-agnostic,
    translates directly to Datalog/Prolog.
  - `graph/parser.ts` — tree-sitter invocation, per-language grammar loading.
  - `graph/extractor.ts` — AST → graph facts, per-language walks, two-pass
    (collect calls, then resolve).
  - `graph/resolve-calls.ts` + `type-env.ts` + `tsconfig-aliases.ts` +
    `suffix-index.ts` — cross-file import resolution and qualified-name
    resolution (receiver chains → `Class.method`).
  - `graph/native-analyses.ts` — the BFS/DFS/SCC analyses (the part we port).
  - `graph/facts.ts` — `graphToProlog` and the fact/rule schema (for the
    optional solver layer; quoted verbatim in §5 Phase 5).
  - `graph/cache.ts` + `diff.ts` — per-file-hash caching and snapshot diff.
  - `graph/adapter-registry.ts` — how a new language plugs in.
  - `solvers/prolog-solver.ts` — SWI-Prolog WASM, the reference for the solver.

## 4. How it maps onto ATLAS today

| #39 point | ATLAS today | What changes |
|---|---|---|
| precomputed call graph | none; ad-hoc tree-sitter walks | new cached `CodeGraph` substrate (Phase 0) |
| 1. candidate scoring veto | `structural_score` direct-identifier name check (`v3-service/main.py`) | resolve calls against graph; reject calls with no reachable definition (Phase 1) |
| 3. repair call-chain context | `call_chain_context` 1-hop callers/callees (`v3-service/main.py`) | multi-hop reachability slice + path witnesses (Phase 2) |
| 4. context auto-injection | `symbol_index` name-match snippets (`proxy/symbol_index.go` + v3-service) | inject the symbol's graph neighborhood / impact set (Phase 3) |
| 2. tier classification | cyclomatic complexity, shipped (`/internal/cyclomatic_complexity`) | optional graph fan-in/out enrichment (Phase 4, low priority) |
| ast_edit | friendly selectors, Python + HTML (shipped) | unchanged; raw-query routing is a separate small follow-up |

Tree-sitter is already a v3-service dependency (`tree_sitter`,
`tree_sitter_python`, `tree_sitter_html`), and v3-service already extracts
Python defs/calls in `symbol_index`, `structural_score`, and
`call_chain_context`. So the graph substrate extends code that already exists
rather than greenfield. Internal endpoints already live in v3-service
(`/internal/symbol_index`, `/internal/ast_edit`, `/internal/cyclomatic_complexity`),
so a `/internal/call_graph` endpoint is the natural home.

## 5. Decisions

Two decisions were confirmed up front (same shape as the wavelet port for #120).

**Reuse strategy: port the graph engine to Python.** Model `v3-service/graph/`
on chiasmus's `src/graph/`: reuse the `CodeGraph` fact model, the native
analyses (BFS/SCC/dead-code/entry-points), import resolution via a suffix index,
and the per-file-hash cache, and reuse the Prolog fact schema verbatim.
Reimplement the tree-sitter extraction in Python, extending what v3-service
already does. This keeps the graph in-process, reuses the existing tree-sitter
setup, adds no new runtime, and matches the #120 precedent. The cost is that
this is a bigger port than the wavelet engine, since the extractor and resolver
are substantial, so Phase 0 is scoped tightly to the analyses the four
integration points need rather than all 16 chiasmus analyses, and to Python
first. The alternatives considered and set aside were a Node sidecar running
chiasmus directly (maximal reuse but a new runtime and a network hop on the
latency budget) and a from-scratch minimal reimplementation (less code but
reinvents resolution and caching chiasmus already solved). Everything downstream
of Phase 0 is agnostic to this choice, since it consumes the same in-process
graph-query contract.

**Solver scope: native first, solver as an optional Phase 5.** Points 1-4 are
delivered with native graph traversal. Prolog facts are emitted early so the
solver can be added later without re-plumbing, and the solver itself lands only
as an additive final layer for custom rule queries the native layer can't
express. This keeps the core dependency-free and matches how chiasmus itself
works (§2).

## 6. Phased plan

Each phase is independently shippable and behind a feature flag, default off
(`ATLAS_CALL_GRAPH=0`), strictly additive. No behavior change until enabled.

### Phase 0 — Call-graph substrate — ✅ SHIPPED
**Goal:** v3-service can build and cache a project call graph and answer native
queries.

- Ported into `v3-service/graph/`: `types.py` (CodeGraph fact model),
  `extract.py` (tree-sitter Python extraction, modeled on chiasmus `walkPython`),
  `analyses.py` (callers / callees / reachability / path / impact / cycles /
  dead-code / entry-points, native O(V+E) with iterative Tarjan SCC), `resolve.py`
  (Python import resolution, the suffix-index analogue), `facts.py`
  (`graph_to_prolog` for the Phase 5 solver, emitted early), `cache.py`
  (per-file-hash LRU for incremental recompute), `flags.py` (`ATLAS_CALL_GRAPH`).
- `build_graph(file_map)` builds and caches the project graph; `run_analysis`
  dispatches the native analyses.
- Internal contract shipped: `POST /internal/call_graph` taking
  `{file_map, analysis, target/from/to/entry_points}`, gated by
  `ATLAS_CALL_GRAPH` (returns `ok=false` when off). Dockerfile copies the
  package; CI installs the tree-sitter grammars and runs the suite.
- **Golden parity** against chiasmus's own `native-analyses.ts` (captured via
  `npx tsx`): callers, callees, reachability both directions, path sequence,
  impact in BFS order, dead-code, and a self-recursion cycle all match. One
  port fix surfaced from it — `impact` now returns BFS discovery order to match
  chiasmus (Python sets aren't insertion-ordered).
- Tests: `tests/v3-service/test_graph_{analyses,extract,support}.py`, 39 tests.
- **Exit (met):** v3-service answers callers / reachability / path over a real
  project, cross-file imports resolved, behind the flag. No pipeline change yet.

### Phase 1 — Deepen the structural veto (#39 point 1) — ✅ SHIPPED
**Goal:** reject candidates whose calls don't resolve to reachable definitions,
not just whose imports survive.

- `graph/resolve_calls.py`: `unresolved_calls` resolves the candidate's
  **direct-identifier** calls (attribute/method calls stay out of scope, as in
  the shipped veto) against the import graph. A call resolves if it is defined
  locally, a builtin, an imported name, or supplied by a wildcard import whose
  module's **actual exports** are resolved via the graph. The deepening
  (strict policy, chosen): a bare call to a name that merely exists in some
  *unimported* project file is now flagged, where the shipped veto accepted it.
- Conservative guards: an unresolvable wildcard (stdlib / third-party) makes the
  result `lenient` and flags nothing; attribute calls are never flagged.
- Wired into the pipeline `run` after the existing structural veto, gated by
  `ATLAS_CALL_GRAPH`, additive, and it never empties the candidate set (a
  fully-failing set falls through to phase-3 repair). Off → unchanged behavior.
- Tests: `tests/v3-service/test_graph_resolve_calls.py` (8) — local/builtin/
  import resolution, the strict unimported-symbol flag, wildcard-to-exports,
  stdlib-wildcard leniency, attribute calls never flagged.
- **Exit (met):** the veto catches a bare call to an unimported project symbol
  (a real NameError) that the shipped shape check let through, while staying
  lenient on opaque wildcards and attribute calls.

### Phase 2 — Multi-hop repair context (#39 point 3) — ✅ SHIPPED
**Goal:** give the repair model the real call chain, not 1-hop neighbors.

- `graph/context.py` `repair_context`: a reachability slice — the call path from
  an entry point down to the failing function, its transitive callers (impact
  set) beyond the direct ones, and its callees, all bounded for token budget.
- Wired in the pipeline `run` Phase-3 repair block, gated by `ATLAS_CALL_GRAPH`,
  preferring the graph block and falling back to the shipped `call_chain_context`
  on flag-off or any failure.
- Tests in `test_graph_context.py`.
- **Exit (met):** repair context for a deep failure includes the actual entry→
  function call path and the transitive impact set, not just direct neighbors.

### Phase 3 — Graph-scoped context injection (#39 point 4) — ✅ SHIPPED
**Goal:** when the user names a symbol, inject its graph neighborhood.

- `graph/context.py` `symbol_neighborhood`: a symbol's callers, callees, impact
  set, and defining files. v3-service attaches it to `/internal/symbol_index` as
  an additive `graph` field (the `matched`/`skipped` shape is unchanged), gated
  by `ATLAS_CALL_GRAPH`, building the project graph once per request.
- The proxy consumes it end to end: `symbolIndexResult.Graph` (`proxy/symbol_index.go`)
  parses the field and `formatGraphNeighborhood` folds the callers/callees into
  the injected `[system note]` context in `agent.go`. (A review caught this as
  initially half-wired — produced but discarded — and it's now consumed.)
- Tests: `test_graph_context.py` + Go `symbol_index_test.go` (parse + format).
- **Exit (met):** naming a symbol injects its definitions plus its structural
  neighborhood. The fewer-exploration-turns claim needs the live L6 measurement.

### Phase 4 — Tier enrichment (#39 point 2, low priority) — ◑ signal shipped
**Goal:** optional graph signals in tier classification.

- `analyses.complexity`: per-node fan-in/fan-out plus the graph maxima, exposed
  via the `complexity` analysis on `/internal/call_graph`. Tests in
  `test_graph_solver.py`.
- Wiring into the Go tier classifier is intentionally **not** done: point 2
  already ships via cyclomatic complexity, and the plan gates this on a tier-
  accuracy measurement we can't run here. The signal is available for that
  measurement to consume.

### Phase 5 — Optional solver layer (the literal "+ solver") — ✅ SHIPPED (dependency-free)
**Goal:** custom rule-based queries native traversal can't express.

- Prolog facts already emit from Phase 0 (`facts.py` `graph_to_prolog`,
  exposed via the `facts` analysis) using chiasmus's schema, ready for an
  external SWI-Prolog / `chiasmus_verify`.
- `graph/datalog.py`: a compact, dependency-free in-process Datalog engine
  (facts + structured rules → bounded naive fixpoint → query). Ships the
  built-in transitive `reaches` closure (`reachable_pairs`, exposed as the
  `closure` analysis) and supports arbitrary user rules in-process. Its `reaches`
  is **cross-checked against `analyses.reachability` for all node pairs** in the
  test suite, so the solver layer is provably consistent with the native one.
- A real SWI-Prolog backend (`pyswip` / sidecar) for arbitrary external rules
  remains an option; the facts artifact is the bridge and needs no re-plumbing.
- **Exit (met):** the closure relation and arbitrary in-process rules evaluate
  over the facts, dependency-free; native stays the default path.

### Phase 6 — Multi-language — ✅ SHIPPED (Python + JavaScript)
**Goal:** beyond Python.

- `extract.py` dispatches by extension to a Python or JavaScript walk; JS covers
  defines / calls / imports / contains (function/class/method declarations,
  `const` arrow/function defines, call and member-call expressions, named /
  default / namespace imports). `build_graph` ingests both; analyses run on the
  merged multi-language graph.
- The `tree_sitter_javascript` grammar is added to the Dockerfile and CI.
- **Golden parity** against chiasmus's `extractGraph` on a JS fixture (classes,
  methods, arrow-const defines, member calls, `new` not counted as a call,
  named/default imports) — defines, calls, and imports match exactly.
- Tests in `test_graph_multilang.py`. TS (type nodes) and qualified-name
  resolution are the natural next increment; JS uses bare names like Python.

## 7. Risks / open questions

- **Scope of the port.** chiasmus's extractor + resolver is substantial. Scope
  Phase 0 to the analyses points 1-4 actually need, not all 16, and to Python
  first. Resist porting communities/hubs/bridges until something needs them.
- **Dynamic dispatch / metaprogramming.** Python `getattr`, runtime routes,
  duck typing. Tree-sitter sees syntax, not runtime. Keep the veto conservative:
  unresolved-due-to-dynamism is not a bug to reject on. chiasmus excludes methods
  from reachability for exactly this reason (`native-analyses.ts` filters
  `kind=method`).
- **Latency.** Building a graph per request is too slow; the cache is essential.
  Recompute only touched files on write. Bench before it lands in the proxy's
  critical path (#39 explicitly flags the latency budget).
- **Solver runtime.** A real Prolog/Z3 in the Python stack is a new heavy
  dependency. Deferring it to Phase 5 and gating it keeps the core
  dependency-free; revisit Node-sidecar-vs-Python-binding when it's actually
  needed.
- **Correctness of cross-file resolution in Python.** No tsconfig analogue;
  resolution is import-statement + module-path based. Model on chiasmus's
  suffix-index approach but expect Python-specific edge cases (relative imports,
  `__init__.py` re-exports).

## 8. First concrete step

Phase 0, the substrate: port the `CodeGraph` model and the native analyses into
`v3-service/graph/` (Python), extending the existing tree-sitter extraction,
with a per-file-hash cache and a `/internal/call_graph` endpoint, all behind
`ATLAS_CALL_GRAPH`. Confirm v3-service answers callers / reachability / path over
a real project within the latency budget, with golden parity against
`chiasmus_graph` on shared Python fixtures.
