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
python `tests/` **1565 passed, 6 skipped** (cuts removed their tests: was 1731);
lens **64**; contracts **60**; proxy+tui go build/vet/test **ok**. Every step must
reproduce this (baseline re-pinned 2026-08-05 after the cut waves).

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

1. **Retrieval/routing stack: CUT, permanently.** Commits 1-3 pushed (c02844b,
   778a444, 5b83142); 4-6 in flight.
2. **Pattern cache: WIRE A READER.** Type+recency matching feeding the agent as
   context; the BM25/indexer stack is not needed for it and dies with the cut.
3. **Behavior-affecting collapse (detector unification, word-list vs evidence
   gates): IN SCOPE**, validated by a real dogfood session before merge.
4. **RPG: CUT EVERYWHERE** ✅ abdf085 (−5,319) — unshipped in the v3 image;
   remove rpg.py/rpg_eval.py/wavelet/ + proxy rpg.go + the tools.go call
   sites + types RPG fields + the 16MB SSE sizing + TUI handlers + knob
   surface (schema keeps a deprecated= entry). Issue #148 stays as record.
5. **V2/TB2 bench subgraph: DELETE** ✅ 60059de (−1,872).
6. **Trainer scripts: DELETED** ✅ 2b6c7bb (−1,559) — coverage gap (provenance
   manifest) root-fixed first: `atlas lens build` now writes provenance.json
   into every activated bundle; memory updated.
7. **LTM tier: DELETE** — no scheduler exists, promotion unreachable in prod;
   consolidator goes with it (its whole job); reader keeps co-occurrence
   expansion only if something still writes the graph at pattern-write time.

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

## Audit coverage matrix (what has been READ, so it is not re-read)

Every row = a full-file read by the 2026-08-05 deep audit. A future pass may
trust these as covered; re-audit only after major rewrites of that area.

| Area | Files read | Ledger | Executed |
|---|---|---|---|
| proxy/ (13 non-test .go) | all, line-by-line | L-A below | Tier 1 done (03bdfe5) |
| tui/ (~15 .go) | all | L-A | Tier 1 done (03bdfe5) |
| v3-service/ incl. graph/ | all (quarantine: isolation only) | L-B | queued |
| geometric-lens/ (all .py) | all | L-B | cut commits 1-3 in flight |
| atlas/ (24 modules) | all | L-C | P0s done (4f27031) |
| scripts/ (22) + lib/ | all | L-C | P0 install.sh done |
| sandbox/ (4), inference/ (8) | all | L-C | queued |
| .github/workflows (7), compose (6), .env.example, atlas.conf.example, pyproject, .gitignore, .dockerignore | all | L-C | partial (schema/compose fixes) |
| benchmark/ | structure + entry points (not per-task data) | L-C | queued |
| docs/ | NOT audited for prose accuracy (deliberate — docs pass is LAST); stale-subject list captured in L-C | — | pending |

## Debris ledgers — full disposition (dedup record)

Status legend: ✅ done · 🔜 queued · ⏸ owner decision · ❌ rejected (do not re-flag).

### L-A: proxy + tui (35 findings)

| # | Item | Action | Status |
|---|---|---|---|
| A1 | `LensScore` type + false comment (main.go/types.go) | DELETE | ✅ 03bdfe5 |
| A2 | 4 never-incremented health counters + stats block | DELETE | ✅ |
| A3 | `maxRepairAttempts` + fictional banner | DELETE+FIX | ✅ |
| A4 | `callLLMConstrained` ignored param + `buildToolCallSchemaJSON` | DELETE | ✅ |
| A5 | `lint_python` hint case | DELETE | ✅ |
| A6 | `verifyCompletionClaims` unread `summary` param | FIX | ✅ |
| A7 | `sendChat` shim (tui) | DELETE | ✅ |
| A8 | `RealProjectDir` — never assigned; dead delete-block removed WITH it (substituting WorkingDir would have resurrected dead behavior) | DELETE | ✅ (deviation documented) |
| A9 | `Plan.WinningIndex`/`.Reasons` write-only | DELETE | ✅ |
| A10 | `symbolGraphNode.DefinedIn`/`.Impact` | DELETE | ✅ |
| A11 | `Envelope.ParentID` (Go×2 + events.py + docs) | DELETE | ✅ |
| A12 | `pipelineState.doneSummary` | DELETE | ✅ |
| A13 | 12 stale-filename comments (9 ledger + 3 found in-flight) | FIX | ✅ |
| A14 | 3+1 detached doc comments (guardrails.go, agent.go) | FIX | ✅ |
| A15 | `ragOK`→`lensOK` | FIX | ✅ |
| A16 | Aider-archaeology comment (main.go) | DELETE | ✅ |
| A17 | events.go cancellation-out-of-scope claim | FIX | ✅ |
| A18 | tui `event:`-frame skip (redundant with data: filter) | DELETE | ✅ |
| A19 | literal `—` escapes (13+, incl. 4 beyond ledger) | FIX | ✅ (Replacer smart-quote escapes kept — functional) |
| A20 | dead-knob docs: GRAMMAR_MODE/LENS_DATA_DIR/MAX_READ_BYTES/CONTROL_VECTOR/TUI block | FIX | ✅ (+schema decl in follow-up) |
| A21 | `classifyParseFailure`+`categorizeParseFailure` twin trees (~55) | COLLAPSE | 🔜 Tier 2 |
| A22 | two template-existence walkers in gates.go (~130) | COLLAPSE | 🔜 Tier 2 |
| A23 | `patternReadTracker` global dup of ctx.FilesRead (~40; its no-import-cycle rationale is false) | COLLAPSE | 🔜 Tier 2 |
| A24 | `truncate` vs `truncateStr` (keep `truncateForCorrective`) | COLLAPSE | 🔜 Tier 2 |
| A25 | `reInlineTemplate` pre-pass for one sentence fragment | judgment | 🔜 with A22 |
| A26 | 15 TB2-era calibration comments; 34 PC-ticket citations | restate/strip | 🔜 docs-adjacent pass |
| A27 | 6 never-emitted ErrorCode consts | DELETE | ✅ pruned: schema enum + canonical set now = the six codes writeError actually emits |
| A28 | `EvtMetric` | **❌ REVERSED**: audit claim stale — today 3 emit sites (agent.go token totals + v3-plan stages) and 2 model.go handler cases. KEEP. | ❌ |
| A29 | **rpg.go quarantine LEAKY**: `planConstraintsForTarget`/`regenerateOnDrift`/`reportRPGDrift` called from tools.go write paths; RPG fields in types.go; 16MB SSE buffer in v3_bridge.go sized for RPG | ⏸ owner (4-file decision) | ⏸ |
| A30 | `ATLAS_GRAMMAR_MODE=loose` flagged dead by audit | **❌ REJECTED** — Gemma requires loose (done-spam on strict); documented instead | ❌ |
| A31 | lens.go/gates.go/detectors.go/tools.go header comments understate contents | FIX headers | 🔜 |
| A32 | `ATLAS_MACOS_PREFIX`/`ATLAS_LLAMA_HOST`/`ATLAS_BACKEND` | **❌ REJECTED 2026-08-01 — NOT dead.** Verified 8 / 3 / 15 referencing files respectively (the macOS hybrid path + entrypoints + compose). The July audit searched Go/py only and missed the shell/compose surface. | ❌ keep |

### L-B: v3-service + geometric-lens (key items; blast radius in audit report)

| # | Item | Action | Status |
|---|---|---|---|
| B1 | `/internal/call_graph` route + handler (69) — zero callers | DELETE | 🔜 (before B4-B6) |
| B2 | `/v3/run` route + `_handle_run` (58) + false events.py docstring | DELETE+FIX | 🔜 |
| B3 | `ATLAS_V3_PORT` read never forwarded; container port hardcoded | FIX or constant | 🔜 |
| B4 | Datalog engine (143) — reachable only via B1 | DELETE | 🔜 |
| B5 | Prolog facts emitter (136) — same; 3rd reachability impl | DELETE | 🔜 |
| B6 | `graph/analyses.complexity` (24) — undocumented analysis value | DELETE | 🔜 |
| B7 | JS extraction (120) — grammar not in image; `_JS_AVAILABLE` always False | DELETE or ship grammar | ⏸ owner |
| B8 | graph/ facade: 40 exports → 8 real (5 post-B1) | SHRINK | 🔜 |
| B9 | `_DEFAULT_BUILTINS` hand-list vs `PY_BUILTINS` (documented bug class) | COLLAPSE | 🔜 |
| B10 | `direct_call_names` vs `_extract_python_call_targets`; `bound_names` vs `_extract_python_bound_names` (write-blocking divergence risk) | COLLAPSE | 🔜 |
| B11 | `repair_context` vs `call_chain_context` (~80) | COLLAPSE (parameterize hops) | 🔜 |
| B12 | **RPG/wavelet not COPY'd into the v3 image** — ATLAS_RPG_PLANNING wired to nothing deployed; 3 lazy-import sites take the except path always | ⏸ owner (pairs with A29) | ⏸ |
| B13 | pipeline: `failure_analyzer`/`constraint_refiner` constructed never used; `constraints=[]` clobbers the param; `last_logprobs`; unused `test_input`; dup threading import; unreachable "fallback" retry key | DELETE/FIX | 🔜 |
| B14 | main.py test-only import shim (6 names) → tests import owning modules | FIX | 🔜 |
| B15 | `_post_pattern_outcome` fires before baseline substitution (cache sees `solution=""`) | FIX (real bug) | 🔜 |
| B16 | scoring.py inline smoke-check generators shadowed by sandbox.syntax_check (70) | DELETE | 🔜 |
| B17 | `_ext_to_lang` advertises 5 languages the checker rejects | FIX | 🔜 |
| B18 | lens: `evaluate_gx` (49), `extract_embeddings_batch`, `get_embedding_contract`, `save/load_models` aliases, Replay/EWC Config dataclasses + stats(), false telemetry/atlas.conf docstrings | DELETE/FIX | 🔜 |
| B19 | `provenance.py` (151) — nothing writes or reads it; docstring claims CLI surfaces it (false) | DELETE or wire | ⏸ owner |
| B20 | `get_category_surprise` + write-only EMA (~15) | DELETE | 🔜 |
| B21 | `/v1/patterns/write` twin of `/internal/patterns/write` | DELETE /v1 twin | ✅ d985885 |
| B22 | `/internal/lens/stats` dup of /health payload | DELETE | 🔜 |
| B23 | cache flush/consolidate routes zero-caller; **no scheduler → LTM tier unreachable in prod** | DELETE (decided) | ✅ 733e6a8 (consolidator + LTM tier gone) |
| B24 | legacy `build_cvector_prompts.py` (141) superseded by build_steering_vector | DELETE | 🔜 |
| B25 | `render_pos`/`render_neg` identical wrappers | COLLAPSE | 🔜 |
| B26 | `ast_edit_steering.gguf` filename kept post-rename | **❌ REJECTED as debris** — SHA-pinned by registry, documented in CHANGELOG | ❌ |
| B27 | retrieval cut (whole-file list, main.py/pipeline.py regions, sqlite tables, Dockerfile/compose/docs collateral; ≈4,294 net) with 6-commit sequence; hazard: reader rewrite (c4) must land WITH pattern_matcher deletion | CUT | ✅ all six commits done (c4 733e6a8 reader, c5 b48eca7, c6 d985885 + cc4e717 reqs) |
| B28 | `sandbox_analysis.py` + `/internal/sandbox/analyze` | **CORRECTED 2026-08-01: the claimed caller does not exist.** atlas/client.py has no such call; a whole-tree grep finds only the route, its own module, and docs. The July audit was wrong and I recorded it as KEEP. Now DELETE with W2. | 🔜 W2 |
| B29 | lens dead knobs: CORS_ORIGINS, ROUTING_ENABLED, CONFIG_PATH, API_KEYS_PATH, SANDBOX_URL prefix mismatch, `_energy_disabled_logged` | DIE-WITH-CUT | 🔜 (c5/c6) |
| B30 | `ATLAS_ALLOW_PICKLE_GX` legit but undeclared in schema | FIX | 🔜 |

### L-C: CLI + scripts + infra (49 findings)

| # | Item | Action | Status |
|---|---|---|---|
| C1 | install.sh unbound `$SERVICES` (installer aborts last step) | FIX | ✅ 4f27031 |
| C2 | schema missing GPU_VENDOR/PROXY_UID/GID (validate flags init's own output) | FIX | ✅ |
| C3 | compose missing ATLAS_CONTROL_VECTOR passthrough | FIX | ✅ |
| C4 | `retrain_cx.py` (458) — emits bundles runtime refuses; reads nonexistent TB2 path | DELETE | 🔜 |
| C5 | `ATLAS_RPG_PLANNING` knob surface (compose/.env/schema) | pairs with A29/B12 | ⏸ |
| C6 | 4 more trainer scripts (retrain_cx_phase0, retrain_lens_from_results, collect_/prepare_lens_training) superseded by `atlas lens build` (~1,100) | DELETE | ⏸ owner — **note: memory pins `retrain_lens_from_results.py` in the onboarding loop; verify `atlas lens build --from-results` fully covers it BEFORE deleting, then update memory** |
| C7 | V2/TB2 bench subgraph: v2_runner, v2_report, run_v2_benchmark.sh, benchmark/cli.py (~1,870) | DELETE | ⏸ owner (user previously said TB2 testing is over) |
| C8 | unbuilt inference/Dockerfile + entrypoint.sh pair; CI pins its LLAMA_CPP_REV | DELETE | 🔜 |
| C9 | client.py 5 zero-caller fns (~115) | DELETE | 🔜 |
| C10 | download-models.sh reimplements `atlas model install`; stale 3-model manifest | COLLAPSE to shim | 🔜 |
| C11 | uninstall.sh deletes resources from 3 dead eras (redis, llm-proxy label, nightly cronjob, phantom manifests dir) | FIX | 🔜 |
| C12 | ELEVEN atlas-root resolvers (model.py copy is verbatim env.py) — A2 missed these | COLLAPSE-INTO env.atlas_root | 🔜 |
| C13 | NINE .env parsers (4 in atlas/) | COLLAPSE-INTO compose.read_env_file | 🔜 |
| C14 | `_canonical_model_identity` ×2 + `_model_marker_value` | COLLAPSE into publishing.py | 🔜 |
| C15 | onboard.py hand-rolled GGUF parser vs gguf.read_gguf_kv | COLLAPSE | 🔜 |
| C16 | display.py 17 unused constants + h() + clear() | DELETE | 🔜 |
| C17 | 4 duplicate container-smoke CI jobs → 1 matrix | COLLAPSE | 🔜 |
| C18 | `check_nvidia` alias, `ATLAS_RAG_URL` fallback, `_read_saved_cost_field_dim` shim, `sign_manifest`, init `_ok/_warn/_err`, `solve.sandbox_test`, `status_block(speed)`, `open_registry_pr_via_api(color)`, `_verify_one(color)`, onboard `_c` no-op | DELETE (10 small) | 🔜 |
| C19 | dead knob families: DRAFT_MODEL/SPECULATIVE (~30), JWT secret chain (~20), LORA/TRAINING dirs (~12) | DELETE | 🔜 |
| C20 | .gitignore 12 phantom paths + stale V2.5 headers; .dockerignore phantoms + 3 Aider rules | DELETE | 🔜 |
| C21 | CI uses deprecated SKIP_NVIDIA alias (its only caller) | FIX then DELETE alias | 🔜 |
| C22 | redact.py docstrings say 3 copies; there are 4 (test enforces 4) | FIX | 🔜 |
| C23 | lens.py docstring advertises nonexistent `atlas lens push` | FIX | 🔜 |
| C24 | entrypoint metal message says "V3.1.2 planned" for shipped feature | FIX | 🔜 |
| C25 | `atlas bench --strategy` 3 of 4 choices inert (always --baseline) | FIX help or drop flag | ⏸ owner (bench semantics) |
| C26 | pyproject torch>=2.13 floor above CI-proven 2.12.1; no 2.13 wheel exists | FIX | 🔜 |
| C27 | events.py = 307-line shipped test fixture; keep EVENT_TYPES as contract anchor | COLLAPSE-INTO tests/ | ⏸ owner (packaging surface) |
| C28 | check_dockerfile_sources.py — KEEP through the cut (it exists to catch exactly this), re-evaluate after | KEEP | ❌ for now |
| C29 | compose retrieval leftovers (PROJECT_DATA_DIR, lens-data volume, api-keys comment; init.py still generates api-keys.json) | DIE-WITH-CUT c5/c6 | 🔜 |
| C30 | rocm compose pulls never-published image (upgrade.py already special-cases) | FIX docs or publish | ⏸ owner |
| C31 | verify-install.sh NVIDIA-only check hard-fails AMD/Metal installs; omits v3-service | FIX | 🔜 |
| C32 | run_v31_ablation.sh hardcodes K3s NodePort health URL | FIX from .env | 🔜 |
| C33 | production-readiness.py PYTEST_PATHS omits 3 suites CI gates on | FIX | 🔜 |
| C34 | model.py docstring omits install-artifacts subcommand | FIX | 🔜 |
| C35 | local junk: patches/*.bak, _agenttest/ (gitignored) | DELETE locally | 🔜 |
| C36 | Docs-with-dead-subjects list (SETUP retrain menu, CONFIGURATION rows, benchmark/README V2 sections, models/README) | captured for docs pass | 🔜 LAST |

## Phase 4 — existence interrogation (owner directive, 2026-08-05: "do they
## both need to exist?" — question components, not just files)

Queued behind the current cut waves. Each row needs an evidence-based
recommendation before any action.

| # | Question | First evidence | Status |
|---|---|---|---|
| E1 | **repl.py vs tui/** — two interactive chat frontends. repl.py:1 calls itself "the main ATLAS interface"; tui.py:3 says it replaced the Aider chat UI; tui.py imports repl for proxy-launch + `_stop_local_proxy`. | **VERDICT (2026-08-05): one chat surface = TUI.** Extract proxy-launch lifecycle to a runtime module (TUI keeps it); delete the REPL chat loop (its features exist as plain commands); bare `atlas` launches the TUI. CLI stays as the non-interactive command surface. | 🔜 queued |
| E8 | **tui/ still confetti**: 15 files / 7,862 lines. | **Merge map:** auth+consumer→chat (transport); plan→state; files→panes; calibration→commands; debug→main. 15 → 8 files. | 🔜 queued |
| E9 | **atlas/cli/ nesting** — package contains ONLY cli/, so `atlas/cli/commands/x.py` carries a dead level. | **Flatten `atlas/cli/*` → `atlas/*`** (`atlas.commands.doctor`); pyproject entry point + ~37 files of import rewrites, mechanical. Package name stays `atlas` (pip). | 🔜 queued |
| E10 | **tui/demo.go = 1,147 lines (15% of the TUI) of self-contained demo mode.** | **KEEP** (owner delegated 2026-08-05): documented `--demo`/`/demo` split-pane base-vs-V3 recording — the proof-of-value tool; cohesive single file. | ❌ keep |
| E11 | **docs/lang/{ja,ko,zh-CN}** — ×4 maintenance on every doc edit. | **FREEZE** (delegated): docs pass adds a may-lag banner; translations no longer block or accompany code changes; revisit post-1.0. Public-facing translations are not deleted. | ✅ decided |
| E2b | **Fold direction for the lens↔v3 service merge** (E2, owner: "if it makes sense, do it" — it does). | Preliminary: fold v3's routes INTO the lens's FastAPI app (kills the hand-rolled stdlib V3Handler server, keeps the framework); one Python service, one image, one URL for the proxy; compose loses a service. Final direction + plan measured AFTER cut c4-c6 lands; capstone of the campaign, after dogfood validation of the reader. | 🔜 capstone |

**Delegated micro-decisions (2026-08-05, "do what you think is best"):**
- A27 error codes: DELETE the six never-emitted constants; shrink AllErrorCodes,
  the JSON schema enum, and the contract canonical set together (never emitted →
  no client ever saw them; not a breaking change in practice).
- A28 EvtMetric: DELETE emission + type across proxy/tui/events.py + schema
  (emitted once, consumed nowhere).
- E7/C27 events.py: ✅ split — spec (EVENT_TYPES, Event, errors, make_event,
  parse_envelope; 214 lines) stays in-package; consumer/assertion harness
  (iter_sse_lines, iter_events, is_terminal, collect, assert_monotonic) moved
  to tests/cli/event_harness.py.
| E2 | **geometric-lens as a separate service** | **VERDICT: KEEP SEPARATE** (measured 2026-08-01). The fold trades a container for a shared failure domain + restart schedule between a stateful torch model-server (147 MB weights, SQLite state, rebuilt on `atlas lens build`, seconds to boot) and a stateless AST/orchestration service (0.72 MB code, sub-second boot) — disjoint rebuild triggers, one GIL, and a lens OOM would newly take out structural_check/pycheck/outline/symbol_index/structural_edit//v3/plan//v3/generate. Measured benefit: **2.8 ms per pipeline run** (8 cross-hops × ~0.35 ms; the hop is 0.03% of the `score-per-step` call it rides on) and ~15 lines of duplicated auth. **Revisit only when BOTH**: the lens stops loading torch/.pt at startup, AND the product decides a degraded lens should fail T2/T3 writes anyway (i.e. proxy handleReady no longer distinguishes lens_ready from v3). | ❌ keep |
| E3 | **graph/ vs symbols.py** — after the B9-B11 collapses, graph/'s unique value is import-resolution + reachability for `unresolved_calls`/`repair_context`. May fold into symbols.py entirely, deleting the package. | B1/B4-B6 first (dead route + engines), then re-measure. | 🔎 after B-wave |
| E4 | **sandbox as separate container** | Isolation IS the feature (untrusted code execution). KEEP — recorded so it isn't re-asked. | ❌ keep |
| E5 | **benchmark/ remainder** (runner, best_of_k, geo_learning, models, v3/) | Load-bearing for `atlas bench` → lens training loop. KEEP. | ❌ keep |
| E6 | **scripts/ remainder** after C-wave — each survivor must justify itself vs an `atlas` command or CI use. | Inventory post-cut. | 🔎 |
| E7 | **atlas/cli/events.py in the shipped package** (C27) | ✅ split: spec stays (214), harness → tests/cli/event_harness.py | ✅ |
| E12 | **benchmark/v3 IS v3-service, misfiled** — pipeline.py imports 11+ modules from benchmark.v3/llm_client/runner; the v3 image COPYs benchmark/ for exactly this reason. Owner spotted it via "geo_learning in v3-service". | Relocation running: stages → `v3-service/stages/`, harness (v3_runner, datasets, geo_learning, models, config, best_of_k) → `atlas/bench/` (pip-installed CLI becomes onboarding-self-contained); results path preserved for --from-results. | 🔜 agent |
| E13 | **metacognitive.py** — doubly inert in production: profile_path="" (no table loads) AND get_warnings([], …) (loop can't run); returns [] on every live call. Name is false. | DELETE (owner rule: not working + not truly metacognitive) | 🔜 agent |
| E14 | **geo_learning.py** — real 378-line embedding-banking module, single caller (v3_runner) | KEEP as module; moves with the harness to atlas/bench/ | 🔜 agent |
| E15 | **sandbox → v3-service fold?** | **NO — security boundary**: generated code must execute outside the orchestrator's container (privileges/blast radius). Reaffirms E4. | ❌ keep |
| E16 | **proxy test-file confetti** — 49 test files, 43 named for deleted subjects (the folder complaint) | ✅ merged along production seams: 61 → 24 .go files | ✅ |
| E17 | **datasets/ breadth** — 8 loaders, only livecodebench ever invoked | ✅ seven deleted | ✅ |

## V3 pipeline efficacy ledger (2026-07-31 review; full report in session transcript)

**Meta-finding: TWO ORCHESTRATORS.** pipeline.py (live) vs atlas/bench/v3_runner.py
(bench) disagree on k policy, S* adapter, probe thinking, ACE/ReASC presence,
vetoes (live-only), constraints plumbing (bench-only). The 74.6% ablation
certifies the BENCH orchestrator; the live cascade has never been benchmarked
as-wired. Bench-learned fixes (k=3 pin, S* stdin adapter) never reached live.

| Stage | Verdict | Evidence |
|---|---|---|
| probe | KEEP | bench-validated early exit |
| plan_search | KEEP (simplify: fake threadpools, dead :400 expr) | **+12.4pp** — the workhorse |
| div_sampling | KEEP | fallback generator, ~free |
| self_test_gen | KEEP (dedupe double-gen = D2) | legitimizes Phase 3 |
| pr_cot | KEEP → verify-as-you-go | **36/42 rescues** (H200 join: 69/70) |
| refinement_loop | KEEP; ✅ budget gate shipped (aada0c4): enters only when one iteration (~3 LLM calls at observed speed) fits the remaining budget — H200: 453/487 entries burned ~6 min at 0 iterations | 6/42 rescues |
| candidate_selection | KEEP | 30 live lines; ablation baselines |
| embedding_store | KEEP (bench) | feeds lens training |
| llm_client | KEEP | the model-agnosticism layer |
| budget_forcing | SIMPLIFY → tokens table (⏸ owner-data) | Wait-injection = 0 callers; live thinking silently off |
| blend_asc dynamic k | ✅ **CUT everywhere, k=3 pinned** (1c64a3d) | +0.0pp; H200: allocation tracked normalization scale, not task; superseded by cxgx_gate below |
| cxgx_gate | ✅ **SHIPPED both paths, default on** | 66.9% gated vs 64.6% fixed-k=3 vs 61.7% randomgate at n=175/arm; k>=3 floor makes worst case = the pin it replaces |
| s_star | ✅ **CUT everywhere** (e2e6b03) | +0.0pp; H200: 118 tiebreaks all 0-0, 110/110 winners = lens min-energy, WITH the fixed stdin adapter |
| derivation_chains | ✅ **CUT** (466a114) | **0/485 H200 rescues** (0/194 local), ≤17 calls, verification fiction; roadmap already prescribed removal |
| ace_pipeline | ✅ **CUT** (6c4ee2b) | learned task-id strings into a playbook discarded at exit |
| reasc | ✅ **CUT** (6c4ee2b) | runner recorded its verdict and ignored it |
| lens_feedback | MEASURE (bench flag A/B) | postdates ablation |

**CxGx allocation gate — shipped (2026-07-31).** The C(x)-only allocator
stayed cut (its k tracked the lens normalization scale, not task
difficulty, and with no floor it gave k=1 to 60% of tasks that had just
failed the probe). Its successor is in the tree as
`v3-service/stages/cxgx_gate.py`, shared by both orchestrators: C(x)
normalized energy picks a base tier, the G(x) XGBoost score escalates it
+1/+2 against the model's calibrated severe boundary, and a hard **k >= 3
floor** means the gate can only add candidates to the old pin, never
starve generation. The four-arm triangulation on the owner's H200 dataset
(n=175/arm): gated **66.9%** · fixed k=3 64.6% · same tier mix shuffled
across tasks 61.7% · all-k=8 67.4% at ~27% more tokens; reproduced on the
v5 task set at 83.4% vs 80.6%. The shuffled arm is the load-bearing
control — beating it by 5.1 pp at matched spend is what says the lens
signal carries information rather than the compute alone.

Two deliberate divergences from the owner's `cxgx-patch/v3_runner.py`
reference: (1) the live path is budget-aware — the proxy's
`ATLAS_V3_TIMEOUT` cap has no bench counterpart, so an escalation that
cannot be generated inside the remaining wall-clock is lowered to what can
be (same reasoning as the aada0c4 refinement gate), reserving one
refinement iteration so the extra candidates cannot starve Phase 3;
(2) the G(x) bands are anchored to the model's `gx_thresholds.json` severe
boundary rather than hardcoded at 0.60/0.45, which are that boundary and
0.75x it for the benched model — hardcoding them would repeat the defect
that made the lens's own thresholds per-model. The `ATLAS_CXGX_GATE` /
`_FORCE_TIER` / `_RANDOM_MAP` env arms and `random_tiers_v6.json` stayed in
the dataset: they are the experiment's ablation scaffolding, and the
experiment is already run.

**Composition defects D1-D8** (full text in review): D1 veto→repair-pool hole
(vetoed stub can ship) · D2 duplicate self-test gen w/ None-downgrade · D3 two
orchestrators · D4 live SandboxAdapter drops test_input (verification = "ran
with empty stdin") · D5 EmbedAdapter→[] disarms geometry filter silently · D6
live constraints plumbing vacuous · D7 dead knobs/code inside stages · D8 live
stage telemetry nonexistent (docstrings advertise bench-only JSONL).

**Execution state:** D1/D2/D4/D7/D8 fixes landed (data-independent). H200
dataset arrived (/home/isaac/atlas-benchmark-data) and the CUT WAVE IS
EXECUTED (2026-07-31, local commits on dev): derivation 466a114 · S*
e2e6b03 · k-pin 1c64a3d · ACE+ReASC 6c4ee2b · refinement budget gate
aada0c4 · proxy max_tokens clamp (owner-D8). Stage tree 17 → 12 modules,
then 13 with the CxGx gate. Worst-case live LLM calls ≈47 → ≈27 (≈21 on
the interactive path when the budget gate forecloses refinement); k is
floored at 3 and escalates only as far as the wall-clock allows (bench:
up to 8, unbounded). Owner root-cause note validated: the TUI-visible
hangs-then-baseline was derivation burning the 180s cap; stub writes = D1.

**H200 ops findings (record-only, out of repo scope except D8/A3):**
- BUGS.md **A3** (embed sidecar `--pooling` unpinned): **already fixed in
  this repo** — the shipped entrypoint (`inference/entrypoint-v3.1.sh`,
  used by Dockerfile.v31/rocm/vulkan) pins `--pooling
  "$ATLAS_EMBED_POOLING"` (default `mean`) and the lens enforces the
  convention via `model_identity.json`'s embedding_contract. The one
  unpinned copy is `inference/entrypoint.sh`, half of the unbuilt
  Dockerfile pair already queued for deletion as ledger item C8.
- BUGS.md **A4/A5/A6**: owner-side ops findings on the H200 serving
  host (out of repo scope) — recorded here so they are not re-derived.
- Owner-**D8** (client-disconnect zombie generations saturating llama
  slots): ✅ proxy-level clamp shipped in this commit —
  `clampGenerationBody` forces an explicit bounded `max_tokens`/
  `n_predict` onto every passthrough generation request
  (ATLAS_MAX_COMPLETION_TOKENS, default 8192); the agent loop's own
  calls already carried `ATLAS_MAX_TOKENS`.

**Dogfood 2026-08-01 (demo2, post-fix stack):** session completed done in 12
turns / 9 tools. VALIDATED live: reader served 3 July-banked idiom patterns
(learning loop closed for the first time); content-loop cut + parse-error
steer both recovered; structural_edit emitted correctly by the model;
verification gate honored (python run + serve before done). NEW FINDINGS:
(F1) JS-inside-template-string blind spot — model introduced a stray-paren
JS syntax error at app.py:121 that python-compile, server-run, and every
gate structurally cannot see; done summary also misdescribed its own edit
(claims nextDirection-buffer check; code checks direction). Candidate fix
direction: lens/lint on embedded <script> blocks or a headless page probe.
(F2) v3 telemetry volume mounted root-owned vs appuser(1001) → fail-soft
disabled with a clear log line (fail-soft verified working); live volume
chowned + Dockerfile pre-creates the path owned by appuser.

**E2E 2026-07-31 (two sessions, post-F1-gate stack).** Session 1 (localStorage
high score): model degenerated into repeated `)**\rVert` LaTeX at 1,215 chars;
content-loop cut fired **correctly** (verified `isLoopingTail` against the real
template — no false positive), truncated output failed the syntax gate 3x, the
path-aware breaker ended the session honestly. Nothing corrupt reached disk.
Session 2 (spacebar pause) exposed **D9**.

**D9 — V3 candidate regressions were charged to the model.** `edit_file` and
`structural_edit` hand their spliced content to `improveContentWithV3` and
write back whatever returns; V3 regenerates the *whole file*, so it can
reintroduce a defect the model had already repaired. The downstream write gate
then rejected the edit quoting a line number that exists only in a candidate
the model never saw, and the model spent its last three turns hunting a defect
in a file verified clean on disk (`edit_file` → "string to replace not found").
`write_file` never had this hole: its V3 path already falls back to the model's
baseline. **Root fix at the V3 boundary, not at the three gate sites:**
`improveContentWithV3` now runs `v3CandidateRegression` (syntax /
embedded-script / unresolved-call, scored against the *caller's* content, not
the file) and drops a regressing candidate, returning the caller's content with
zero V3 metadata. Downstream gates therefore only ever judge model-authored
content, so their existing messages are accurate again and need no change.
Guarded by three tests at the boundary; the drop test reproduces the E2E
failure and fails without the fix.

**D9a — the sanitiser ran after the boundary check (caught reviewing D9's own
fix, never shipped).** Both `improveContentWithV3` callers stripped V3's markdown
fences AFTER the call returned. With the regression check now at that boundary, a
fenced candidate does not parse, so it would be dropped as broken and V3 would go
silently unused for every wrapped response. Sanitising moved INTO the boundary
ahead of the check; both callers lost their duplicate block. Guarded by
`TestImproveContentV3SanitizesBeforeJudgingCandidate`, which fails when the order
is swapped back.

**D10 — failure nudges advertised selectors the target cannot accept.** Session 3
reached for `structural_edit` with `<script>` on a .py file and got "unknown
selector '<script>' for python". It had been told to: the write_file-unavailable
nudge (`proxy/agent.go`) and both edit_file mismatch hints (`proxy/tools.go`)
offered `<tag>` with no HTML qualifier, and they fire exactly when the model is
stuck and most likely to follow them literally. The system prompt
(`proxy/agent.go:2316`) already qualified it correctly; these three did not.
`structuralSelectorHint(ext)` now keys the offer off the target's extension.
`v3-service/symbols.py` additionally answers a tag-on-Python selector with what
DOES work (rewrite the enclosing function, or anchor edit_file on one unique
line) instead of a selector list that misses the intent — markup in a Python
string is one string literal to the Python grammar, so no selector reaches in.

**D11 — one honesty gate could starve the other three.** The four completion
gates shared a single bounce counter and are evaluated in fixed order, so
whichever fired first could spend the whole allowance and silence the rest.
Session 4 put all three bounces on the verification gate; `exitGates` then
returned early at its head on every later exit, so the done-without-action gate
never ran and the exit was completely ungated. Each gate now carries its own
budget and an exhausted gate falls through instead of returning early.
`TestExitGatesOneGateCannotStarveAnother` reproduces the starvation directly:
with the shared counter the exit passes ungated.

> **Correction (same day).** This entry first said session 4's model "exited
> having changed NOTHING while claiming the pause logic is already present,
> which it was not". The model was RIGHT: the fixture recovered from the
> session-2 log already contained a complete spacebar pause (`isPaused`, a
> `key === ' '` branch, and the draw loop honouring it). The check that
> cleared it was `grep -c paused`, which is case-sensitive and does not match
> `isPaused` — so sessions 2-5 were all asked to add a feature that was
> already there, and session 4's "it is already present" was accurate
> reporting, not a hallucination. The starvation defect is unaffected and is
> proven by the test rather than by that session; what is retracted is the
> claim that the model lied. The fixture has since been stripped of the pause
> so the task is genuinely undone.

### Measured-reliability campaign 2026-07-31 (later) — harness + task + quality

`scripts/e2e-reliability.py` runs real sessions against the live stack and
reports three things that must not be conflated: **Harness Integrity** (did
ATLAS's own plumbing do anything wrong — should be 100%), **Task Success**
(did the change land — bounded by the model), and **code quality** of what the
agent wrote (measured, never judged: McCabe complexity, function/file length,
pyflakes defects separated from style nits). Task suite: 4 AoC-style puzzles
with holdout inputs, a 3-file CLI app, 3 code edits, 2 conversational probes.
`scripts/code_quality.py` runs standalone. Detectors H1-H9 are unit-tested
against synthetic streams AND replayed against the captured sessions.

**Why holdout inputs.** The model sees `input.txt`; the check re-runs its
program against an input it never saw. Not theoretical: on `aoc_shoal` the
model recognised the puzzle from training, wrote the canonical textbook
example over the real input without reading it, and solved that instead —
reporting honestly, "created input.txt with sample data".

**Cycle 1 (corrected): task 5/8, harness 7/8.** All four harness defects came
from one session that hit the wall clock. Turn counts fell sharply as the
messages got accurate (sonar 7→3, course 8→5). Quality: worst complexity 10
(limit 15), longest function 66 lines, longest file 71 — structure is fine;
6 real defects (F821 undefined name, F841, 3×F401) is the honest weak spot.

**Product fixes, each traced to an observed session, not a hunch:**

*ATLAS confusing the model with its own output* — the call-graph footer read as
file content (`old_str` anchored on it); `read_file`'s `N<tab>` numbering parsed
as tab-delimited data (a correct grid algorithm printed 0); entity-encoded
content reported as an unrelated syntax error.

*Gates wrong or silent* — D9 V3 candidates blamed on the model; D11 one honesty
gate starving the other three; O4 a semantic no-op accepted as a completed edit;
rejected tool calls emitting no `tool_result`; `write_file` clobbering a file
nobody had read.

*Steering that led nowhere* — `<tag>` selectors offered for `.py`; "rewrite the
function holding the template" when no function holds it; "match byte-for-byte"
for degenerate output carrying `\rVert`; the f-string fix buried in a
parenthetical.

*Environment* — proxy on ~/demo vs sandbox on ~/demo2 (every session edited one
tree and verified another; `atlas doctor` catches it and was never run);
sandbox Python 3.11 rejecting `f"{d["k"]}"`, valid from 3.12 (PEP 701) — base
bumped to 3.13.

**The pattern.** The largest category is ATLAS *misleading* the model, not
failing to catch it. The gates were mostly right; what was broken was what
ATLAS said afterward — which is also why turn counts dropped once the messages
became accurate.

**Harness bugs found in the measurement itself** (recorded because five of them
produced confidently wrong numbers): `working_dir` silently overridden so
sessions ran outside the checked directory; a pause check that passed against
an untouched fixture; a fixture that already implemented the feature; H2
matching a CORRECT rejection; H6 reading the wrong payload key; H7 encoding a
wrong assumption about job lifetime; no workspace isolation between tasks; every
fixture treated as immutable, failing a task for doing what it was told.

**Open for an owner call:** O1-O3 above, plus whether `/syntax-check` (12
languages, the gate behind most blocked writes) should stay in `sandbox/` — it
is not a sandboxing concern, and an upstream runtime like NVIDIA OpenShell
would not provide it (its base image also ships only Python+Node, so the
12-toolchain burden returns as bring-your-own-container).

### E2E campaign 2026-07-31 — five sessions, one fixture

Fixture: the 208-line Flask snake game (`/workspace/app.py`), a Python file whose
game loop lives in JS inside the `HTML_TEMPLATE` string. Task each time: "add a
spacebar pause toggle, then verify the app still starts." Verified clean on both
layers before each run (py_compile + `node --check` on the extracted script).

| # | Outcome | Root cause | Disposition |
|---|---|---|---|
| 1 | breaker | model degenerated into `)**\rVert` LaTeX; content-loop cut fired CORRECTLY (verified `isLoopingTail` against the real template — no false positive) | machinery correct |
| 2 | breaker | **D9** | fixed |
| 3 | breaker | **D10**, plus 7009- and 6318-char `old_str` degenerating into tab-runs | fixed |
| 4 | ungated exit | **D11** | fixed |
| 5 | see below | model gamed the no-op check with a typo | OPEN — owner call |

**Fixture caveat for sessions 2-5.** The fixture used for those runs already
implemented the pause being asked for (see the D11 correction above), so their
TASK outcomes carry no signal — the work was done before the session started.
Their HARNESS outcomes stand: D9, D10 and D11 are all defects in how ATLAS
routed, steered, or gated the work, and none of them depends on what the file
contained. Sessions from the reliability harness onward use a stripped fixture.

**The through-line across 1, 3 and 5 is one thing: the model has no structural
way to edit markup inside a host-language string.** `structural_edit` cannot
target it, so `edit_file` is the only route, so the model tries to reproduce
thousands of characters of embedded JS as `old_str` and degenerates partway
through. This is the twin of F1: F1 was "we cannot SEE breakage in embedded
scripts" (closed by the embedded-script gate); this is "we cannot EDIT them
structurally". D10 softens it by steering to a single-line anchor. Closing it
properly means mapping a `<script>` selector through the enclosing string literal
— the locator already exists for the F1 gate — and is an owner decision, not a
stabilisation-phase bolt-on.

**Open finding O1 — the no-op rejection invites a cosmetic perturbation.**
`structural_edit`'s identical-content message says "emit a replacement that
actually differs". Session 5's model satisfied it by misspelling
`HTML_TEMPLATE` → `HTML_TEMPELATE`, which passes every gate and would `NameError`
on the first request. V3 happened to rewrite the file and silently repair it. The
gate measures difference, not correctness, and cannot do otherwise on its own.

**Open finding O2 — the structural gate tracks unresolved CALLS, not NAMES.**
Measured directly against `/internal/structural_check`: identical output for a
clean source and one whose only defect is `HTML_TEMPELATE`. So an edit that
introduces an undefined variable reference is not caught, which is the #147
failure (a 500 on every request) reached through a different node type. Not
fixed here on purpose: Python name resolution is genuinely hard (globals,
builtins, comprehensions, star-imports), the existing check is deliberately
narrow and fails open, and a false positive BLOCKS a legitimate write. Widening
it is an owner call.

**Open finding O4 — every gate can pass while the claim is false.** The single
most important result of the measured runs. A session ran `structural_edit` on
`function:index`, the edit SUCCEEDED, the file parsed, the app served 200 to
curl, and the model reported adding "a `paused` boolean, a spacebar event
listener, and a check in the `draw` function". None of it was in the template.
Every honesty gate was satisfied on its own terms: a productive change landed
(action gate), a verification command ran and passed (verification gate), the
file is syntactically valid (syntax gate), no unresolved call appeared
(structural gate). What none of them can see is that *the change was made to
the wrong node* — `index()` only renders the module-level `HTML_TEMPLATE`, so
rewriting it cannot affect the JS. **Verification proves the app still runs, not
that the requested feature exists**, and no current gate closes that gap.

Two contributing defects, both fixed: the tag-selector message told the model to
"rewrite the whole function holding the template" when no function holds it
(84d4054), and `structural_edit` accepted a replacement that was mostly the
model's own deliberation as `#` comments ("Wait, the instruction is to update
the JS inside the HTML_TEMPLATE string") because comments are valid Python.
The second is NOT fixed: detecting reasoning-leaked-as-comments risks rejecting
legitimate comments, so it is an owner call. The general problem — a
feature-level post-condition, rather than "it still runs" — is the real gap, and
is what `scripts/e2e-reliability.py` checks per task and the product does not.

**Open finding O3 — V3 rewrites the whole file on a 3-line structural_edit.**
Session 5's `structural_edit` targeted `function:index` with 78 chars of content;
V3 activated post-splice and returned a file with comments stripped throughout,
CSS indentation changed, the trailing newline gone, and a new UI line reading
"Space to Pause" for a feature it did not implement (`grep -c paused` = 0). This
is the documented whole-file design working as specified, but the blast radius on
a single-node edit is worth an owner decision.

## Detector unification spec (Phase 3c — behavior-identical stage 1)

**The six detectors are not redundant — their PLUMBING is.** Each observes
genuinely different evidence (arg identity / reasoning text / lens score trend /
failure paths / read-write ratio / write count), so merging the *signals* would
lose real coverage. What duplicates is everything around them:

| Today | Count |
|---|---|
| State owners (AgentContext fields + loop locals) | 5 + 5 |
| `pendingXCorrective` vars + near-identical inject blocks | 3 |
| Hand-rolled ladders (repeat steer-then-kill, path-aware breaker + output-rescue, two-stage read nudge) | 3 |
| "Caller is responsible for resetting …" contracts (ownership split detector↔loop) | 3 |

**Target: one `loopHealth` struct** owned by `runState` (mirroring the gate
collapse) holding all stuck-state, one `observe*` method per signal, **owning
its own resets** (killing the caller-responsibility split), emitting findings
with a severity:
- `nudge` — inject a `[system note]`, continue (read budget, first repeat)
- `steer` — inject + advance the escalation counter (edit-miss → structural_edit)
- `stop` — end the run with a summary (second repeat, same-path error loop)

One corrective queue drained at ONE site replaces three inject blocks; one
ladder replaces three, preserving today's observable thresholds: steer-before-
kill on repeats, path-aware continue on multi-file grinds, output-rescue before
any stop. **No signal removed, no threshold moved** — plumbing collapse, not
behavior redesign. Merges only after a dogfood session (Category B item 3).
Sequencing: implement after the F1 gate work clears `agent.go` (index-collision
lesson from the 2026-08-05 concurrent-agent incident).

---

## Lens healthcheck: RESOLVED — the split is already correct (2026-08-01)

Investigated and **closed without changing behavior.** `/health` returning
200 unconditionally is not a bug: it is the liveness probe, and its own
docstring states the contract ("this endpoint is for *information*, not
gating. Use /ready for liveness/scoring-functional gating"). Readiness is
gated in the right place — proxy/main.go's handleReady reads the lens's
`/ready`, and lens scoring fail-softs everywhere else, so a degraded lens
costs scoring rather than the stack.

Repointing the compose healthcheck at `/ready` would have been a
regression, not a fix: v3-service and atlas-proxy both `depends_on:
geometric-lens: condition: service_healthy`, and `/ready` 503s until the
lens artifacts exist, so **no fresh install would ever start**. Docker
also restarts unhealthy containers, which cannot conjure missing weights.

Action taken: an 11-line comment in docker-compose.yml explaining why the
probe is `/health` and what breaks if someone "fixes" it — same treatment
as the deliberately-off spec-decode knobs.

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

### W3 — `structured_log.py` ×3 and `private_values.py` ×4: KEEP COPIES (2026-07-31)

Verdict **(a) keep the copies, enforce and advertise the duplication**, not (b) a
shared module. Three facts decide it:

1. **Build contexts.** Only v3-service builds from the repo root
   (`context: .`, `dockerfile: v3-service/Dockerfile`). geometric-lens builds from
   `context: ./geometric-lens` and sandbox from `context: ./sandbox`, so neither can
   `COPY` a repo-root shared path. Making them able to means flipping both contexts
   to `.` (docker-compose.yml + both Dockerfiles + .dockerignore), which also hands
   two images a repo-sized build context and a cache key that busts on any repo file.
   That is a large, cache-hostile change to share 164 lines.
2. **atlas/redact.py can never join.** It is a module of the `atlas` pip package, and
   no service image installs that package (checked all three requirements files). So
   even (b) leaves a copy outside the shared path, and the parity contract has to
   exist regardless — (b) buys a smaller contract, not the removal of one.
3. **The mechanism already works.** Both modules already open with a CANONICAL COPY
   NOTICE naming the sibling copies and the enforcing test, and both contract tests
   already assert byte-identity across the full copy set. Nothing was silently
   drifting; the gap was in what happens to copies the contract does not know about.

**Shipped with W3** (tests only, no service code touched):
- Copy registry centralized in `tests/contracts/__init__.py` — one place to register
  a copy, and registering is what puts it under contract.
- `test_no_unregistered_copies` — discovers copies by the notice marker (not by
  filename, since `atlas/redact.py` is the same module renamed) across every `.py` in
  the repo and fails if one exists outside the registry. Closes the hole where a new
  service vendoring either module escapes byte-identity enforcement entirely.
- `test_each_copy_points_at_this_contract` — the notice must name the enforcing test,
  so it cannot decay into decoration.

**Follow-ups this could not land** (they must change every copy in one commit, and
W3 held to geometric-lens/ + tests/):
- **C22**, already on the ledger: `private_values.py`'s notice says the file exists
  "in three places" and names geometric-lens/, sandbox/, v3-service/. There are four
  — `atlas/redact.py` is the fourth, and its own copy of the notice therefore fails to
  list itself. Correct text for all four copies: name all four paths explicitly.
- Once C22 lands, strengthen `test_each_copy_points_at_this_contract` to assert the
  notice names *every* registered copy path, not just the enforcing test. That closes
  the loop: registry, notice, and enforcement can no longer disagree.

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
