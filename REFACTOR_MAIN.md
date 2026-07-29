# REFACTOR_MAIN — codebase reorganization tracker

**Goal:** apply "modules that fire together, wire together." Merge accidentally-fragmented
units, split God-files, dedupe scattered logic, remove dead code, keep functionality
identical (the test suites are the invariant). Docs updated LAST, only to match reality.

**Method (non-negotiable, per prior-session lessons):**
1. Work component by component, smallest safe change first.
2. Full relevant test suite green **before and after** every step; commit per step on `dev`.
3. **Do not refactor code pending a keep/cut decision** — see Category B. Polishing code
   that may be deleted is the exact failure this session already reverted once.
4. No force-push. No PR to main. CI monitored on each pushed SHA.
5. Names must be true; note dead code when found but act on it only within its component's step.

**Baseline invariant (must stay green), captured 2026-07-29:**
python `tests/` **1731 passed, 6 skipped**; lens **63 passed**; proxy `go test ./...` **ok**;
tui `go build ./...` **ok**. Every step must reproduce this.

---

## The load-bearing finding

The lens's apparent over-fragmentation is mostly a **dormant subsystem**, not a refactor
target. Traced 2026-07-29:

- `router/` (route_selector, feedback_recorder, difficulty_estimator, signal_collector,
  fallback_chain), the pattern-cache **read** path (`retrieve_cached_patterns`,
  `pattern_matcher`), `rag_enhanced_completion`, and most of `pipeline.py` are reachable
  **only** through the lens endpoint `/v1/chat/completions`.
- The proxy never calls `/v1/chat/completions`. Its only live lens call is
  `/internal/lens/score-per-step`.
- `/v1/chat/completions`, `/v1/projects/*`, `/v1/tasks/*`, `/v1/queue/stats` are the exact
  endpoints removed in `9f330d5` and reverted back on 2026-07-22.

**Consequence:** these live in **Category B** — blocked until the retrieval/routing/pattern-cache
keep-or-cut decision is made. If CUT, they disappear and no merge is needed. If KEEP, they get
merged *as part of wiring them in properly*. Either way, merging them now is premature.

---

## Category A — safe to refactor now (live, decision-independent)

Ordered by value ÷ risk. Each row is one commit, tests green after.

| # | Item | Files | Type | Risk | Status |
|---|---|---|---|---|---|
| A1 | Dedupe ANSI colors + `safe_print` into `display.py` | atlas/cli: display.py + doctor, tier, asa, lens, onboard, init, **model** (7th copy found) | inverse-merge | low | ✅ done |
| A2 | Extract publish/registry/model-resolve helpers out of `lens.py` into a shared module | lens.py → new `atlas/cli/publishing.py` + probe → `client.py` + `atlas_root` → `env.py` | inverse-merge | med | ✅ done |
| A3 | Split `v3-service/main.py` (4018) God-file | → adapters.py, scoring.py, symbols.py, planning.py, pipeline.py, server(main).py | split | med | ✅ done |
| A4 | Split `tui/model.go` (2866): extract `events.go` | model.go 2866→1745; events.go 1135 | split | med | ✅ done |
| A5 | **Consolidate the proxy: 33 → 13 files** (direction reversed by owner: too many files; collapse, never split) | main+5 infra fragments; gates.go ×6; detectors.go ×3; context.go ×4; permissions.go ×3; lens.go ×3; grammar→tools.go | merge | med | ✅ done |
| A5b | Decompose `runAgentLoop` IN PLACE — `runState` + gate dedup + bounce collapse; no new files | agent.go internals only | in-file | high | ✅ stage 1 |

Notes:
- **A1** is textbook accidental fragmentation: `display.py` already owns the colors, yet 6
  command files re-declare `RESET/BOLD/RED/...` verbatim and reimplement `_safe_print` in 5
  divergent variants. One concern, scattered. Lowest risk, highest clarity win — done first.
- **A2**: `asa.py` reaches into `lens.py` for 9 symbols including private `_`-prefixed ones
  (`_atlas_root, LlamaProbe, probe_llama, publish_preflight, _sha256_file, _resolve_model_arg,
  _hf_token, open_registry_pr_via_api, _registry_set_asa`). A publishing concern buried in a
  2178-line command file. Extract to one module all three (lens, asa, publish) import.
- **A3/A4/A5** are God-file **splits** — the opposite disease. Seams identified per file
  below. A5 is highest-risk (it is the agent loop); it goes last and gets its own careful
  sub-plan. The 7,994 lines of proxy Go tests are the safety net.
- **NOT to be merged** (confirmed legitimate): CLI subcommand-per-file separation; tui
  dispatcher/handler (`commands.go`↔`model.go`) and command/transport (`chat.go`↔`model.go`)
  splits; v3-service `graph/` (clean layered DAG); lens hubs `pattern_store.py`,
  `models/pattern.py`, `models/route.py`.

### Split seams (reference)

**v3-service/main.py (4018):** adapters (`LLMAdapter`, `SandboxAdapter`, `EmbedAdapter`),
scoring (`score_candidate*`, `structural_score`, `classify_task_type`, lint/compile checks),
symbols (the ~860-line AST toolkit `_ast_*`/`structural_edit`, 2384-3246), pipeline
(`V3PipelineService`, 1054-1915), planning (`generate_plan` block, 2011-2383), server
(`V3Handler`, 3248-3960). Handlers call free functions by name — no circular-import risk.

**tui/model.go (2866):** extract event-formatting block (`appendChatEvent` + all `format*`,
1246-2565) to `events.go` — takes `json.RawMessage`, returns strings, no `tuiModel` receiver,
moves cleanly. Optionally shed view utils (`scrollPaneAt`, `findPane*`, `stripANSI`) to panes.

**proxy agent.go (4254) / tools.go (3562):** `runAgentLoop` is 1421 lines / 25+ concerns.
Extract the per-turn body to `runTurn`, collect ~12 loose tracking booleans into a `turnState`
struct, move each honesty gate behind one interface, split `tools.go` one tool per file.
Sub-plan authored before starting.

---

## Category B — UNBLOCKED (owner decisions, 2026-08-05)

1. **Retrieval/routing stack: CUT, permanently.** Component-by-component commits,
   owner sign-off before each push.
2. **Pattern cache: WIRE A READER.** Type+recency matching feeding the agent as
   context; the BM25/indexer stack is not needed for it and dies with the cut.
3. **Behavior-affecting collapse (detector unification, word-list vs evidence
   gates): IN SCOPE**, validated by a real dogfood session before merge.

**Phase 3b reader design (commit 4 of the cut sequence):**
- Lens: one endpoint, `POST /internal/patterns/context` {task, top_k} →
  {patterns:[{summary, content, type, age_days}]}. Matching = pattern-type
  match (heuristic classify of the task text) scored through the existing
  `compute_score` with similarity := 1.0 on type match / 0.3 otherwise —
  type+recency+success, zero BM25. 1-hop expansion via
  `co_occurrence.get_linked_patterns`; `record_pattern_access` on serve
  (hit/miss stats become real again).
- Proxy: in the run setup next to the symbol-index injection — call the
  endpoint with the user message, inject top ≤3 patterns as one
  `[system note]: lessons from previous sessions` block, hard char cap,
  fail-soft on any error, no flag (always-on; flags are the disease).
- Keeps alive (per cut-order hazard): `pattern_scorer.compute_score`,
  `co_occurrence.get_linked_patterns`. `pattern_matcher.py` (BM25) dies.

**v3+lens debris ledger (2026-08-05): ~5,736 net lines.** Tier 0 highlights:
dead `/internal/call_graph` + `/v3/run` routes; **v3 Dockerfile ships neither
rpg.py nor wavelet/ — the quarantine is already enforced in the image and
`ATLAS_RPG_PLANNING` is wired to nothing deployed**. Tier 2: graph/ facade
40→8 exports; Datalog+Prolog engines reachable only via the dead route (three
reachability implementations); unshipped JS extraction; three pairs of
graph/-vs-symbols duplicate implementations on the write-blocking path.
Tier 3: ~640 lines of zero-caller machinery (provenance.py module,
evaluate_gx, write-only EMA, legacy cvector builder, per-file inventory in
the audit transcript). Full 6-commit cut sequence + blast radius (≈4,294
lines, lens tree −36%) recorded in the audit report; commits 1-3 dispatched,
held locally for owner sign-off.

**CLI/scripts/infra debris ledger (2026-08-05): 49 findings, ~4,525 lines.**
P0 bugs fixed same-day (4f27031, local): install.sh unbound $SERVICES aborting
the K3s installer's last step; compose not forwarding ATLAS_CONTROL_VECTOR
(documented override silently inert in Docker); config validate flagging keys
init itself writes; ATLAS_GPU_INDEX_LIST deleted (zero readers). Big buckets
queued: five superseded C(x) trainer scripts (~1,560 incl. one that emits
bundles the runtime refuses); the V2/TB2 benchmark subgraph (~1,870); the
unbuilt inference/Dockerfile+entrypoint pair CI still maintains; eleven
atlas-root resolvers and nine .env parsers that A2's pass missed; dead
draft-model/JWT/LoRA knob families; 4× duplicated sandbox smoke CI jobs;
atlas/cli/events.py as shipped test fixture (307). Full ledger in the audit
transcript. Grand total across all three audits: **≈10,600 removable lines**
(+1,974 quarantined RPG/wavelet, unshipped by the v3 image, if cut).

Original decision table kept for reference:

| Item | Files | If KEEP | If CUT |
|---|---|---|---|
| router/ merge → routing.py | route_selector, feedback_recorder, difficulty_estimator, signal_collector | merge while wiring in | delete dir |
| cache consolidation merge | consolidator + co_occurrence → consolidation.py | merge | revisit |
| pattern-cache read path | retrieve_cached_patterns, pattern_matcher | wire a reader in | delete |
| lens main.py route split | /v1/chat/completions, /v1/projects/*, /v1/tasks/*, /v1/queue/stats handlers | split live routes | delete dead routes, then split remainder |
| pipeline.py | rag_enhanced_completion, retrieve_chunks* | keep | shrink to score-only |

**Dead code confirmed in this orbit (delete with the CUT, or during KEEP-wiring):**
- `router/fallback_chain.py` — entire module, zero importers (`get_fallback_route`,
  `get_escalation_path` both zero-caller; escalation never wired into pipeline).
- `router/difficulty_estimator.get_difficulty_bin` — zero callers; callers use
  `models.route.difficulty_to_bin` directly.

---

## Debris backlog (deep audit, 2026-08-05)

**proxy/tui ledger** (35 findings, ~357 removable lines):
- Tier 1 (zero-risk deletions + comment repairs) — dispatched for execution:
  dead `LensScore` type, four never-incremented health counters, the
  `callLLMConstrained` legacy param + `buildToolCallSchemaJSON`, unregistered
  `lint_python` hint, write-only struct fields (`Plan.WinningIndex/Reasons`,
  `symbolGraphNode.DefinedIn/Impact`, `Envelope.ParentID` both sides,
  `pipelineState.doneSummary`), `RealProjectDir` dup field, `sendChat` shim,
  9 comments pointing at deleted files, 3 detached doc comments, `ragOK`
  rename, dead-knob documentation in .env.example.
- Tier 2 (mechanism collapses, queued): `classifyParseFailure`+
  `categorizeParseFailure` identical trees (~55); the two template-existence
  walkers in gates.go (~130); `patternReadTracker` global duplicating
  `ctx.FilesRead` (~40); three truncation helpers → one.
- Deferred decisions: 6 never-emitted ErrorCode constants (API surface);
  `EvtMetric` emitted but unconsumed.
- **Correction from memory**: `ATLAS_GRAMMAR_MODE=loose` flagged dead by the
  audit is load-bearing for Gemma (strict GBNF → done-spam). Documented, kept.
- **rpg.go quarantine is LEAKY**: `planConstraintsForTarget`,
  `regenerateOnDrift`, `reportRPGDrift` are called from tools.go's write
  paths; `types.go` carries RPG fields; v3_bridge.go's 16MB SSE buffer is
  sized for RPG graphs. The RPG keep/cut decision spans four files.

## Category C — leave alone (correctly factored; recorded so they aren't re-litigated)

- `v3-service/graph/` — clean layered DAG, no cycles, each file one concern. Minor: trim
  over-broad `__init__` facade exports (low priority).
- atlas CLI subcommands (doctor/tier/asa/lens/init/onboard) — legitimate command-per-file;
  every co-change pair is a clean consumer→API dependency, not fragmentation.
- tui `commands.go`/`chat.go` vs `model.go` — dispatcher/handler + command/transport
  boundaries. Merging would enlarge the God-file.
- lens `pattern_store.py`, `models/pattern.py`, `models/route.py` — genuine hubs (4-5
  distinct importers each).
- lens `pattern_extractor.py`, `seed_patterns.py` — single-caller but substantial
  single-responsibility modules; single-caller ≠ fragment.

---

## Trivial cleanups (fold into whichever component step touches the file)

- `tui/panes.go:1042` local `max(a,b int)` — redundant with Go 1.21 builtin `max`.
- proxy: (to be catalogued during A5).

---

## Progress log

- 2026-07-29 — Mapped all components (3 parallel Explore passes). Traced lens liveness;
  established Category B. Wrote this plan. Baseline captured: python 1731/6 skipped,
  lens 63, proxy ok, tui build ok.
- 2026-07-29 — **A1 done.** Found a 7th copy of the block in `model.py` during execution.
  Centralized colors, `supports_unicode()`, `UNICODE_OK`, `DASH`, `OK/NO/WARN` glyphs, and
  `safe_print()` in `display.py` (canonical body = doctor's superset rewrite table; on
  unicode terminals output is byte-identical, on ASCII terminals asa/lens/init/onboard gain
  the glyph rewrites instead of bare `?` replacement). Deleted 7 divergent `_safe_print`
  defs, 7 const blocks, and 5 declared-but-never-used constants (CYAN×4, DIM×2).
  `fit.py`'s `from tier import _safe_print` kept working via the alias. −209/+95 lines.
  Suite: 1731 passed, 6 skipped — matches baseline exactly.
- 2026-07-29 — **A2 done.** Three moves, all Hebbian-grounded:
  (1) publish/registry machinery → new `atlas/cli/publishing.py` (sha256, HF token,
  `gh api`, registry-file editing, PR flow, preflight, model-arg resolution) — was
  buried in lens.py and reached into by asa/publish via 9+ private `_names`; now
  public API with true names. (2) llama probe (`LlamaProbe`, `probe_llama`,
  `llama_url`, the None-contract JSON helpers) → `client.py`, which already owned
  service reachability. (3) `atlas_root()` → `env.py`; killed FOUR duplicate
  resolvers (lens's canonical walk, asa's delegating wrapper, fit's 8-hop variant,
  bench's Path variant now delegates) and model.py's duplicate `_hf_token` (lens's
  superset variant is canonical — also reads HUGGINGFACE_HUB_TOKEN).
  Test patch points retargeted (test_asa: `lens_module.probe_llama` → `asa.probe_llama`;
  test_lens: moved names → `publishing.*`); `bench._atlas_root`/`asa._atlas_root`
  stay patchable module attrs. lens.py 2178 → ~1600 lines.
  Suite: 1731 passed, 6 skipped — matches baseline exactly.
- 2026-07-29 — **A4 done** (out of order while A3's split ran in background).
  `tui/events.go` extracted from model.go: `appendChatEvent` + all ten `format*`
  event renderers + `summarizeTool*` + the cancelled-event classifiers
  (`envelopeLooksCancelled`/`looksCancelled`) — the full SSE-event → chat-row
  formatting concern, verbatim. model.go 2866 → 1745 (update loop + view),
  events.go 1135. Same Go package, so zero call-site changes. go build/vet/test
  green, gofmt clean.
- 2026-08-05 — Session cut out mid-run; the A3 and A5a split agents died before
  writing anything (tree was clean). Full re-verification caught what the Go
  suites could not: the event-contract tests pinned `appendChatEvent` to
  `tui/model.go` by filename and went red after A4. Root-fixed by scanning the
  package by content (same policy as the proxy side) and un-pinned
  `v3StageToEvent` from tools.go in the same pass, pre-empting the identical
  break the pending tools split would have caused. Suite back to 1731/6.
  A3 + A5a relaunched. Standing constraint recorded: refactors collapse, they
  don't grow — no added lines unless structurally required.
- 2026-08-05 — **Direction reversed on the Go side by the owner**: too many files
  is the disease, not the cure. Killed the tools.go split mid-run (nothing
  written). **Proxy consolidated 33 → 13 files, −155 lines**: server infra
  (correlation, auth, api_version, security, private_values) → main.go; all six
  honesty/plan gates (claim_check, structural_gate, syntax_gate, plan_adherence,
  plan_reminder, asset_lint) → gates.go; stuck detectors (tool_repeat,
  reasoning_repeat, traceback) → detectors.go; context enrichment (symbol_index,
  project, workspace, session_manifest) → context.go; permission_gate +
  trust_mode → permissions.go; lens_score + lens_samples + calibration_status →
  lens.go; grammar → tools.go. Unchanged: agent.go, tools.go, guardrails.go,
  types.go, events.go, v3_bridge.go, rpg.go (quarantined). Fallout, root-fixed:
  five more contract tests pinned Go filenames — added one shared content-based
  `go_source()` in tests/contracts/__init__.py, collapsed the event-contract
  test's local copy into it; dropped the now-stale (and unnecessary)
  `proxy/private_values.go` secret-scan allowlist entry, which strengthens the
  scan. go build/vet/test + gofmt green first try; suite 1731/6 exact.
- 2026-08-05 — **A3 done.** `v3-service/main.py` (4018) split into flat sibling
  modules, code moved verbatim by line range: `adapters.py` 477 (LLM/sandbox/
  embed clients, service token + outbound auth, pattern-cache write hook),
  `scoring.py` 505 (lens scoring, task-type classifier, smoke checks, build
  verification, interactive lint), `symbols.py` 892 (tree-sitter toolkit:
  structural_edit, symbol index, structural_score, call-chain context,
  cyclomatic complexity), `planning.py` 409 (/v3/plan), `pipeline.py` 984
  (V3PipelineService + problem builder), `main.py` 820 (V3Handler, bootstrap).
  +69 lines total, all import blocks/headers. Cross-module refs are
  module-qualified (`adapters.X` / `scoring.X` / `symbols.X`) so module-attr
  monkeypatching keeps working; import DAG is main → pipeline → planning →
  scoring → adapters, symbols (scoring needs adapters' `_service_headers` +
  `LENS_URL`; no cycles). Dockerfile COPYs + compose-override mounts added for
  the five modules. Test patch points retargeted (`LLMAdapter`/`SERVICE_TOKEN`
  → adapters, `_STRUCTURAL_EDIT_AVAILABLE` → symbols, `urllib` → scoring).
  Suite: 1731 passed, 6 skipped — matches baseline exactly.
