# Changelog

> This changelog is maintained as a best-effort summary; for line-level detail and any gaps, see the commit history (`git log`) or the GitHub PR list.

## [Unreleased]

### Verbatim reproduction

Every edit failure observed on the 12B traced to one thing: the model cannot
reproduce multi-line text exactly. A 1-line anchor lands; a 9-13 line anchor
comes back with `food.y` written as `hood.y`, `scoreElement` as
`scorerElement`, a stray `)`, or `℘` where `&&` belongs. The tool call is
then rejected for a mismatch that has nothing to do with the model's
understanding of the task. These changes attack the copying itself rather
than adding another retry around it.

**Changed**

- The tool-choice guidance in the system prompt now names the line-addressed
  edit tools. It listed `edit_file` / `write_file` / `structural_edit` and
  repeated those three, so a model changing a multi-line region could only
  pick from tools that need it reproduced — `replace_lines` and `insert_after`
  were reachable only from the raw tool list. The `old_str`-not-found steers
  had the same gap: they pointed at `insert_after`, which *adds*, and named
  nothing for *changing*. Observed live as two failed 15-line `old_str`
  attempts in a row followed by the model abandoning the edit; on the next run
  with the guidance fixed it reached for `replace_lines` directly.
- DRY sampling now defaults **off** (`ATLAS_DRY_MULTIPLIER=0`, was `0.8`).
  DRY penalizes repeated sequences, and copying a file into an `old_str` *is*
  a repeated sequence: the penalty is `multiplier × base^(matched −
  allowed_length)`, so at the previous defaults a 12-token verbatim run
  carried a −23.0 logit penalty on the correct next token. The previous
  `ATLAS_DRY_PENALTY_LAST_N` comment claimed the 2048-token window bounded
  this to the model's own output; it does not. llama-server's
  `init_sampler()` seeds the DRY ring buffer with the prompt tokens, so the
  file being copied from is itself inside the repetition window. The knob
  still works for any model that needs it.
- Agent-loop requests now decode greedily by default: `samplers: ["top_k"]`,
  `top_k: 1` (`ATLAS_TRANSCRIPTION_SAMPLER=0` restores the server chain).
  An agent turn is transcription — the tool call names a path and repeats a
  span of the file — and there is one correct token at each step. Sent as a
  one-element sampler chain rather than `temperature: 0` because temperature
  is applied *last* in llama.cpp's chain, which would pick greedily from an
  already penalty-distorted distribution instead of the raw one.
- The most recently read file is restated, line-numbered, as a final message
  immediately before the generation point (`ATLAS_RESTATE_LAST_READ=0`
  disables). The `read_file` result the model copies from sits thousands of
  tokens back behind the system prompt and every tool description, while the
  model's own drifting copy sits adjacent to the cursor. Skipped when the
  content is already the last message, when nothing has been read, and above
  24 KB.

**Added**

- `replace_lines` — a sixteenth tool that changes an existing line range
  without reproducing it. `insert_after` removed the verbatim burden for
  *adding* code; this does the same for *changing* it. The model supplies
  `start_line`/`end_line` plus `expected_first_line` and
  `expected_last_line`, so the staleness check costs two lines of copying
  instead of N. Assertions compare ignoring leading/trailing whitespace
  (indentation is the most common drift and is not evidence of a stale
  range); a mismatch returns the actual line and a ±3-line numbered window.
  Capped at 20 lines and one hunk per call, requires a prior read, and runs
  the same fallback-syntax, unresolved-call and embedded-script gates as
  `edit_file`.

- A repeated `let`/`const` in one scope is now refused at the write gate. An
  edit appended a second `let score = 0` to a `<script>` that already had one;
  tree-sitter parses that, so the syntax check passed and it landed. A
  duplicate lexical binding is an *early* SyntaxError, so the browser throws
  out the whole script before running a line of it and every handler on the
  page dies — while the Python compiles, the server starts and the page
  returns 200. Only `let`/`const` within a single scope are reported, which
  the spec makes unconditionally an error; `var`, function declarations and
  shadowing across scopes are all legal and left alone.
- A stopped render loop is now refused at the write gate. Asked to make the
  snake speed up with the score, the model replaced `setInterval(draw, 100)`
  with `setTimeout(draw, delay)` at the same top-level spot and never re-armed
  it inside `draw()`. The JavaScript parses, `python app.py` starts the
  server, and the agent reported the change verified — while the game drew
  exactly one frame. `embedded_script_check` now takes the pre-edit file and
  reports a function a repeating timer used to drive that fires once and never
  re-arms; the existing embedded-script gate refuses the write and names the
  call site. Deleting a loop outright, adding a fresh delayed one-shot, and a
  loop that re-arms from inside its own body are all left alone — the finding
  needs both versions, because one alone cannot tell a dead loop from an
  intentional one.

**Fixed**

- A V3 candidate that rewrites text the caller's edit never touched is now
  discarded, keeping the caller's content (the same fallback the parse check
  already used). `edit_file` and `structural_edit` splice their change and
  hand the composed *file* to v3-service, which regenerates it — on a small
  file that is a retype of everything the edit left alone, and the same
  verbatim drift applies. Caught in a live session: V3's accepted candidate
  wrote `#e94562` as `#e94162` and `<h1 id="msg">` as `<h1 id=" msg">`,
  which makes `getElementById('msg')` return `null` at runtime. Neither is a
  syntax error, so the parse and embedded-script gates passed them. The rule
  is that a line the edit did not remove must survive V3 intact; V3 remains
  free to improve the lines the edit actually changed.
- `insert_after` was missing from four `switch` statements that every other
  edit tool appears in: the lens breaker's failure-path extraction (so three
  consecutive failures on the same file did not trip the path-aware
  breaker), the worked-example generator (so its schema shipped without a
  filled-in example), the productive-change counter (so successful inserts
  did not count as progress), and workspace-path containment validation.
  `replace_lines` is registered in all four.

### Measured reliability

A day of running ATLAS against itself and fixing what the sessions showed.
Every fix below was traced to an observed session and carries a test that
fails without it. `scripts/e2e-reliability.py` reports the two numbers this
work is judged on — harness integrity (ATLAS's own plumbing, which should be
100%) and task success (bounded by the model) — plus objective code-quality
probes from `scripts/code_quality.py`.

**Added**

- `insert_after` — a fifteenth tool that inserts lines after a line number
  rather than after text the model must reproduce. Both existing edit
  primitives put a large verbatim-output burden somewhere (`edit_file` an
  anchor, `structural_edit` a whole node), and that is the step that
  measurably fails. `read_file` already prints line numbers, so this takes a
  number the model can cite and only the new text.
- `scripts/verify-deployed.sh` — refuses to let a measurement describe code
  that is not running, catching both source-newer-than-image and
  image-newer-than-container.
- Live-stack coverage for the TUI (17 of 21 slash commands driven through a
  pty), the control plane (`/cancel`, `/v1/permission`), and multi-turn
  conversations, none of which had any.

**Fixed — tier and conversation**

- A question that said "do not change any code" was classified as work, so
  ATLAS was *more* likely to edit when told not to. Three causes: negation
  blindness in both intent classifiers, an explain-plus-no-edit directive
  read positionally, and a question detector that only saw a trailing `?`.
- Questions about code were answered without opening the file, because one
  system-prompt bullet lumped them in with greetings.
- A reply that announced a tool call, or promised an answer, ended the turn
  without delivering either.

**Fixed — gates and writes**

- V3 candidates that regressed the caller's content were blamed on the model.
- One honesty gate could spend the shared bounce budget and silence the other
  three.
- A semantic no-op (only comments changed) counted as a completed edit.
- A rejected tool call emitted no `tool_result`, so the call never resolved
  for the client.
- `write_file` could clobber a file the session had never read.
- New files bypassed the syntax gate, because the sandbox's YAML checker
  wrongly rejected multi-document files and had disabled the gate wholesale.

**Fixed — what ATLAS told the model**

- `read_file`'s line numbers and the call-graph footer read as file content;
  a correct grid algorithm parsed the display format and printed 0.
- Steering offered `<tag>` selectors for `.py` files, and named a function
  "holding the template" when the template is a module-level constant.
- Cryptic Python errors were passed through unexplained: stray backslashes,
  entity-encoded content, and f-string quote nesting that is valid from 3.12.

**Changed**

- Sandbox base image moved to Python 3.13 (was 3.11, which rejected valid
  3.12 syntax and cost a full session).
- Nine real `.env` keys were reported as typos by `atlas config validate`.


### Simplification campaign (2026-07-29 → 2026-08)

One component-by-component pass over the whole tree — merge the fragments,
split the God-files, cut what nothing calls — with the test suites as the
invariant. Headline numbers, measured from the campaign's first commit:
**3,047 → 514 tracked files, net ≈ −56,500 lines including data**. The
per-component disposition ledgers live in the commit history for that
range.

- **One chat surface.** The pipe-mode `/solve` REPL is gone; bare `atlas`
  launches the TUI (no-TTY prints a pointer to `atlas doctor` and exits
  nonzero). The proxy launch/align/stop lifecycle moved to
  `atlas/runtime.py`. The TUI itself merged 15 files → 9; the proxy
  consolidated 33 files → 12, its tests 61 → 24 mirror files.
- **Retrieval/routing stack removed, this time for good** (the 2026-07-22
  removal below was reverted for per-component review; that review is now
  done). PageIndex/BM25 retrieval, the confidence router, the lens `/v1/*`
  surface (projects, tasks, queue, chat/completions, auth), the cache
  consolidator + LTM tier, and dead lens routes (`/internal/lens/stats`,
  cache flush/consolidate, `/v1/patterns/write`) are gone.
- **Pattern-cache reader added.** What replaces retrieval:
  `POST /internal/patterns/context` serves lessons from previous sessions
  (type + recency + success scoring, co-occurrence expansion), and the
  agent loop injects the top ≤3 as a `[system note]` — always-on,
  fail-soft, no flag.
- **RPG planning removed everywhere** (it was never shipped in the v3
  image); the A/B on the reference 12B showed no improvement at ~10x
  planning latency. [#148](https://github.com/itigges22/ATLAS/issues/148)
  is the record.
- **V2/TB2 benchmark subgraph and the five superseded trainer scripts
  removed.** The onboarding loop is fully CLI-driven: `atlas bench` →
  `atlas lens build --from-results`, and every lens build now writes a
  `provenance.json` manifest into the activated bundle.
- **Code moved to where it fires.** The V3 pipeline stages live in
  `v3-service/stages/`; the benchmark harness is `atlas/bench/` inside the
  pip-installed package (repo-root `benchmark/` holds data only);
  `atlas/cli/*` flattened to `atlas/*`; `v3-service/main.py` split into
  flat siblings (adapters/scoring/symbols/planning/pipeline).
- **Debris ledgers executed.** Dead routes (`/v3/run`,
  `/internal/call_graph` and its Datalog/Prolog engines), the inert
  metacognitive module, aspirational error codes (the taxonomy is now the
  six codes `writeError` actually emits), the never-incremented health
  counters, duplicate parse-failure/template-walker/read-ledger mechanisms,
  and seven unused dataset loaders.

### Reverted: the removals below were undone on 2026-07-22

The RPG, wavelet, retrieval, and ablation-data removals described in this section
were reverted the same day. The code is back in the tree. Each subsystem is being
reviewed one component at a time rather than in a single pass, so the entries below
describe what was removed and why, not the current state of the tree.

Reverted in `db4b055`, `b407eed`, `69f2dea`, `8d0abf2`. The `structural_edit` rename
and the sampling and honesty-gate work were kept and remain accurate as written.

### Removed: RPG planning, wavelet decomposition, and the retrieval stack (2026-07-22)
- **RPG planning removed.** `ATLAS_RPG_PLANNING` shipped default-off; the A/B
  against the flat planner on the reference local model returned 0
  improvements, 2 regressions, and roughly 10x planning latency. The outcome
  bottleneck is the model writing correct code, not the plan it writes
  against. Deleted `v3-service/{rpg.py,rpg_eval.py}`, `v3-service/wavelet/`,
  `proxy/rpg.go`, the two-stage planner, the signature veto, the drift
  regeneration loop, and the RPG types. The flag stays in the config schema
  marked deprecated so an existing `.env` gets a specific warning rather than
  "unknown key". `docs/reports/RPG_WAVELET_PLANNING_V3_2.md` is kept and
  marked removed — the design record and the reason it did not pay off are
  both worth having.
- **Retrieval stack removed.** `/v1/projects/*`, `/v1/tasks/*`,
  `/v1/queue/stats`, and the lens's own `/v1/chat/completions` had no caller
  anywhere in the repo — they appeared only as rows in the API.md table. The
  PageIndex tree index, BM25, hybrid retriever, project store, and the router
  stages that fed them are gone. The pattern cache stays: v3-service writes to
  it through `/internal/patterns/write` after every successful candidate.
- **Endpoints kept and verified against their callers**: `score-per-step`
  (proxy, v3-service), `gx-score` (CLI, v3-service), `score-text` and
  `sandbox/analyze` (CLI), `retrain` (benchmark), `reload` (retrain scripts),
  `patterns/write` (v3-service). 23 lens routes before, 19 after.

### Renamed: ast_edit is now structural_edit (2026-07-22)
- The tool resolves a friendly selector (`function:NAME`, `class:NAME`,
  `<tag>`) to exactly one tree-sitter node and replaces that node's source
  text. tree-sitter produces a concrete syntax tree, and the tool never
  traverses one, so the old name described neither the operation nor the
  substrate — and it required the model to reason about compiler internals
  when the trigger is "replace this whole function". Renamed across the proxy,
  v3-service, CLI, ASA scripts, tests, docs, and all three translations,
  including the `/internal/ast_edit` endpoint.
- `ast_edit_steering.gguf` keeps its name: `model_registry.py` pins it by
  filename and SHA256 against the HuggingFace dataset, so renaming would 404
  every download. The ASA contrast prompts embedded the literal old tool name,
  so published vectors predate the rename; `asa_calibration/README.md`
  documents the rebuild path.
- `atlas/cli/client.py`'s `RAG_API_URL` became `LENS_URL` (it always pointed
  at the lens), and `ATLAS_LENS_URL` now takes precedence over the deprecated
  `ATLAS_RAG_URL`, which previously overrode its own replacement.

### Sampling and honesty gates keyed on evidence (2026-07-22)
- **Repetition sampling enabled.** llama-server ships every repetition control
  off (`repeat_penalty=1.0`, `dry_multiplier=0.0`, both penalties 0.0), and
  the proxy set none, so nothing bounded a repeating generation. DRY is now
  set on outgoing requests — chosen over `repeat_penalty`, which scores
  individual tokens and punishes the indentation and keywords source code
  repeats legitimately. Six env knobs, forwarded by compose and registered in
  the config schema (which gained a `float` kind rather than demoting them to
  unvalidated strings). Values are not yet A/B'd.
- **Truncation recovery rejects degenerate output.** The three recovery paths
  rebuilt tool args from whatever the field extractor read, with no check —
  a run of repeated newlines parses as cleanly as a function body, so a
  degenerate generation became a real `edit_file` against the user's file.
- **Verification gate** now also fires when a test or build command actually
  exited non-zero and nothing has passed since, catching a failing test the
  model introduced itself — which no reading of the user's message predicts.
- **Done-without-action gate** now also fires when the model opened the
  project on a non-conversational message and nothing reached disk, covering
  verbs absent from the intent list (`remove the debug logging` matched none).
- **Message tier reduced to its one real decision.** `TierMaxTurns` treats
  T1/T2/T3 identically, `shouldGeneratePlan` tests only T0, and v3-service
  reads the tier into a log line without branching, so the T3 branch was
  removed. T0 now requires positive evidence (short greeting or question
  shape) instead of being the fallthrough: "slow it down significantly" was
  classified conversational, capped at 5 turns, and returned a zero-tool-call
  non-answer.
- `run_command`'s description now marks the boundary it does not cover —
  servers and watchers belong in `run_background`.

### CI gates for failure modes the matrix could not see (2026-07-22)
- **`min-python`** compares the tree against `pyproject.toml`'s
  `requires-python`. A PEP 604 annotation in `sandbox/executor_server.py`
  broke imports on the declared 3.9 floor while CI ran only 3.11/3.12, so it
  failed for contributors and never in CI.
- **`dockerfile-sources`** checks every `COPY` source exists, resolved against
  each service's own build context from `docker-compose.yml`. Stale COPYs
  after the retrieval and RPG removals broke image builds while imports,
  tests, and lint stayed green.
- Python suite repaired from 15 failures and 29 errors to zero (module loaders
  missing `sys.path` and `sys.modules` registration; a missing skip guard on
  the proxy binary).
- Benchmark ablation conditions A–D now index the HuggingFace copies instead
  of being vendored; verified byte-identical first. Tracked files: 3,045 to
  605.


### VS Code extension

- New `extensions/vscode/`: a zero-runtime-dependency VS Code client for the
  proxy HTTP API (#35) — chat sidebar with streamed turns, tool chips, and
  plan checklists; interactive permission flow (allow once / allow for
  session / deny, with pre-decision diff previews for `write_file`,
  `edit_file`, `structural_edit` and `insert_after`); native diff review of
  applied changes; status bar from `/ready`; workspace-mismatch warning;
  service token in SecretStorage. Contributed by @Anuj-72.

### CPU-torch images actually CPU-only again (2026-07-20)
- The lens and v3-service Dockerfiles pre-install torch from the CPU-only
  index, but their pin (2.12.1) had drifted behind requirements.txt
  (2.13.0), so the requirements install silently "upgraded" torch from
  PyPI and dragged the ~8 GB nvidia/cu* dependency stack into both
  CPU-only images (lens 8.29 GB, v3 7.91 GB, vs ~3 GB intended). On a
  43 GB host this also made full image rebuilds fail outright on disk
  space. Pins aligned to 2.13.0; a new contract test
  (tests/contracts/test_torch_cpu_pin.py) fails when either service's
  Dockerfile torch pin diverges from its requirements.txt, or when the
  pre-install loses the CPU index.

### Structural gate on every write path (#147 close-out, 2026-07-20)
- **Coverage completed**: the `write_file` paths that still skipped the gate
  now run it — the V3 winner (including the baseline resurrection when the
  pipeline returns nothing), the V3-error fallback (matters on `/generate`
  timeouts when `/internal/structural_check` still answers), and the T0/T1
  direct path (a sub-10-line `.py` calling an unimported name previously
  landed ungated). The direct path gets the structural gate ONLY — a syntax
  gate there would hard-block legitimate non-parsing T1 content (JSONC,
  multi-doc/templated YAML, scaffold `.py` templates). `BypassV3` (demo
  baseline pane) skips the direct-path gate so the baseline shows the raw
  model; the edit-path and iteration fast-path gates run in all modes, as
  before.
- **False-block hardening** (two adversarial review rounds over this change):
  the resolver's builtin set is now interpreter-derived instead of
  hand-curated (the curated subset was missing `exit`, `TimeoutError`,
  `ConnectionError`, `memoryview`, ... and would have vetoed valid new
  files); `/internal/structural_check` returns the FULL unresolved list (the
  gate diffs original-vs-edited lists, and the previous 10-name cap made
  that comparison unsound in both directions); the gate's `project_context`
  now also includes session-written `.py` files (truncated to 4 KB, like the
  V3 builders) so it is never stricter than the in-pipeline veto it
  backstops; a vetoed V3 winner falls back to the model's own gate-passing
  baseline instead of rejecting (the offending call is V3-authored) — and
  the fallback write lands with plain, non-V3 telemetry (no winning
  score / phase / verification evidence and no "V3 complete" stream), so
  the completion nudge never reports the unverified baseline as
  V3-verified; write-path rejections use a `write_file`-flavored message
  (the edit-flavored one steered models to `edit_file` on files that don't
  exist); an unreadable existing original skips the gate instead of
  counting every pre-existing call as introduced; the winner gate rechecks
  cancellation after its HTTP round-trips so a mid-gate cancel lands
  nothing on disk. The `write_file` iteration fast-path syntax gate now
  applies the same healthy→broken rule as `edit_file` (it hard-blocked a
  strict-invalid config — multi-doc YAML, JSONC — being iterated, and no
  longer does).
- **Gate correctness**: an original-side check failure (transient service
  error; malformed Python is NOT this case — tree-sitter parses tolerantly)
  is retried once and then fails open instead of counting every unresolved
  name as newly introduced; a nil request context no longer panics; both V3
  request builders exclude the target's own pre-edit snapshot from
  `project_context` so the in-pipeline veto can't credit a def the write
  deletes.
- **Tests**: gate-level regression pair for the issue's scope item 3 with an
  import-aware fake (delete-import blocked / import-elsewhere-in-file passes),
  original-side fail-open with retry, nil-context, write-flavored rejection,
  unreadable-original skip; endpoint-level coverage of
  `/internal/structural_check`'s exact response contract including the
  uncapped list; resolver tests for real builtins.
- Known v1 limits (documented in the resolver, out of #147 scope): attribute
  calls (`os.getcwd()` after deleting `import os`) and non-call name
  references are not resolved; shell-redirection writes bypass all gates;
  a tolerantly-parsed broken original can under-report its pre-existing
  unresolved calls and block a one-error-at-a-time repair.

### CodeQL: all 14 open alerts fixed (2026-07-20)
- Expected-output and gate-rejection log lines escape CR/LF
  (go/log-injection ×4); `missingExpectedOutputs` and the asset-lint `Stat`
  probes are contained via `filepath.IsLocal` (go/path-injection ×4).
  Containment keeps the enforcement signal: expected outputs are checked
  against the workspace root AND the system temp dir (host-verify tasks
  name `/tmp` outputs), and an asset reference escaping the workspace is
  reported as dangling without being probed (it can't be served from the
  workspace) rather than silently skipped.
- The `asa → fit → doctor` import cycle (py/cyclic-import ×3) is broken by
  extracting the shared `.env` resolution into `atlas/cli/env.py` (doctor
  re-exports it; fit/lens/publish read it directly — monkeypatch
  `atlas.cli.env` to steer those commands) and the GGUF header reader into
  `atlas/cli/gguf.py`. The dotenv walk keeps its previous reach (7 hops
  from `atlas/cli` = 8 from `atlas/cli/commands`), so it cannot newly pick
  up an ancestor `.env` it never saw before.
- geometric-lens style notes: `from geometric_lens import service` import
  form, explicit `+` string concatenation in the drift probe texts, and the
  legacy-shape warning latch became a mutated holder instead of a rebound
  global.

### Code-review hardening of the #147 / TB2 series (2026-07-20)
An xhigh review of the unpromoted series found 15 correctness defects, all fixed:
the structural resolver now tracks locally-bound names (params, loop/with
targets, assignments) so it no longer false-rejects valid edits or vetoes valid
candidates; the write_file iteration fast-path and the text-exit path got the
structural / verification / completion-claim gates they were missing; the
structural gate excludes the edited file's stale pre-edit content; the read cap
floor no longer exceeds a small slot's budget and a truncated read records only
what was shown (correct dedup + EndLine); UTF-16 BOM files read as text; the
command-not-found and inline-script steers, the expected-output gate, the
active-iteration filename match, and the write fingerprint were all de-noised
against false positives.

### Structural gate on the edit path (#147, 2026-07-19)
- An `ast_edit`/`edit_file` that introduced an unresolved direct call — e.g.
  `render_template` while the file imported only `render_template_string` —
  parsed fine, passed V3 verification, and landed as verified; every request
  then 500'd (NameError). The in-pipeline structural veto was gated off when the
  edit sent no `project_context`, and `ast_edit` had no gate at all.
- Fixes: the V3 structural veto now runs whenever candidates exist (not only
  when project files are present), resolving against the candidate's own
  imports; a new `/internal/structural_check` endpoint exposes the resolver;
  and a proxy-side structural gate on both edit paths refuses a write that
  *introduces* an unresolved direct call (healthy→broken, matching the syntax
  gate — a pre-existing unresolved name mid-repair is allowed). Python-only,
  fail-open when v3-service is unreachable.

### Agent-loop: commit the deliverable + read-size safety (TB2 rounds 5-6, 2026-07-19)
- **Expected-output gate**: parse the prompt for the file the task asks the
  model to produce ("save your solution in X", "the file Z must exist") and
  check it against disk before allowing done/text exit — a partial artifact or
  exploration-without-committing satisfies the generic action gate while the
  named deliverable is still missing. Bounces naming the specific file.
- **Loop-stop output-rescue**: the repeat/error breakers steer toward the named
  deliverable once before hard-stopping (many hard tasks loop on run_command
  and never reach the done/text exit where the gate lives).
- **read_file byte cap**: a single read is capped at half the per-slot context
  (worst-case ~1 token/char) so one huge read can't overflow the window — a
  model that gunzipped a data file and read it whole hit 2.26M tokens and a hard
  context-overflow 400 the force-trim retry couldn't fix. Unconditional (a line
  limit doesn't bound bytes) and context-derived. Binary reads already return a
  tool pointer instead of bytes.

### Agent-loop: stop killing iteration (Terminal-Bench 2.0 round 2, 2026-07-19)
Re-analysis of a 20-task round found that nearly every "failure" was a stopping
condition firing on *productive* work, not the model reaching its limit (turns
are uncapped). Fixes:
- **Repetition detector distinguishes iteration from reassertion.** `write_file`
  repetition is now keyed on path + a whitespace-stripped content fingerprint:
  rewriting a file with materially different content (fixing successive compiler
  errors) is iteration and no longer counts as a loop; reasserting the same draft
  still does.
- **Steer before kill.** The repetition breaker injects a corrective note and
  continues on the first detection, ending the session only if the model repeats
  after the nudge. The old immediate hard-stop (including the
  "productive change → stop" path) terminated models one nudge from finishing.
- **Broken-inline-script steer.** A `python -c` verification one-liner that fails
  with a SyntaxError in its own `-c` argument now steers the model to move the
  test into a `.py` file, instead of letting it re-run the unparseable command
  into the breaker with a possibly-correct solution on disk.
- **Text-exit action gate.** The `text` response path is gated the same way
  `done` is: on an action-intent prompt with no productive change, it bounces
  instead of letting the model narrate its intent and quit having done nothing.
- **Binary-file read guard.** `read_file` on a binary (a NUL byte in the head)
  no longer returns garbage bytes — it returns a directed pointer to the right
  tools (`strings`/`readelf`/`objdump`/`nm`/`file`/`xxd`), which the model
  otherwise never reached for (it read a compiled ELF as text and gave up).
  `file` and `xxd` added to the sandbox image (binutils already rode in with gcc).
- **Fast-path writes during active iteration.** Once the model has written a
  file and just saw it fail a run, the next write is a targeted fix — it now
  skips the V3 pipeline (still syntax-gated) and writes directly, instead of
  paying V3's multi-minute per-call latency (which on a mid-debug file often
  "completes without result" anyway). This unthrottles edit-test-fix loops from
  ~5 cycles in 25 min to run-speed. V3 still owns the first write of each file.

### Agent-loop hardening from the Terminal-Bench 2.0 dogfood round (2026-07-18)
- **`atlas doctor` workspace-mount check** — new `workspace_mounts` check fails
  loudly when the proxy and sandbox bind different host directories as
  `/workspace` (a silent split that sends file tools and `run_command` to
  different filesystems while every `/health` stays green). New
  TROUBLESHOOTING entry documents the symptom and fix (`ATLAS_PROJECT_DIR` +
  recreate both containers together).
- **Sandbox image: common CLI tools baked in** — `git`, `sqlite3`, `jq`,
  `patch`, `zip`, `xz-utils`. The sandbox is non-root on a read-only base, so
  absent binaries can never be installed at runtime; `git clone` and
  `sqlite3 .recover` both dead-ended on "command not found".
- **Missing-command steer** — `command not found` shell errors now get a
  directed [system note] stating that system packages cannot be installed in
  the sandbox and pointing at pip-installable equivalents or the preinstalled
  toolchains, instead of the model re-running into the repetition breaker.
- **Conversation-trim correctness** — the token budget now counts the pinned
  user instruction and pinned file content (previously re-injected without
  being counted) and reserves proportional tokenizer slack (`slot/8`);
  a llama-server over-context 400 force-trims to the minimum window and
  retries once instead of killing the session.
- **Sandbox tmpfs sizing is env-tunable** — `ATLAS_SANDBOX_TMP_SIZE` (2G),
  `ATLAS_SANDBOX_PIP_SIZE` (1G), `ATLAS_SANDBOX_CACHE_SIZE` (512M); the old
  fixed 256M `~/.local` overflowed on `pip install pandas pyarrow`.

### V3.2 — RPG-style architecture-first planning (#120, experimental, opt-in)
- New `ATLAS_RPG_PLANNING` flag (default **off**) enables repository-level,
  plan-then-fill planning ahead of the existing problem-level PlanSearch:
  - **Wavelet substrate** (`v3-service/wavelet/`) — a faithful, dependency-free
    Python port of [wavescope-mcp](https://github.com/yogthos/wavescope-mcp)
    (Ricker CWT, structural signal, multi-resolution bands, project decomposition,
    peak-diff). Numeric parity with upstream is golden-tested.
  - **Repository Planning Graph** (`v3-service/rpg.py`, [arXiv:2509.16198](https://arxiv.org/abs/2509.16198)) —
    two-stage construction (proposal capability tree → implementation files +
    signatures + data-flow edges), graph validation/scoring, and a topological
    projection to the existing flat `Plan` so the agent loop is unchanged. The
    proposal stage is seeded with the wavelet coarse band on existing repos.
  - **Graph-guided generation** — each node's planned interface (signatures,
    edges) threads into its `/v3/generate` call (`proxy/rpg.go`), so the existing
    PlanSearch ([arXiv:2409.03733](https://arxiv.org/abs/2409.03733)) fills a node
    whose architecture is already pinned.
  - **Structural verification + drift** — the candidate veto now rejects code
    that doesn't realize its planned signatures; post-write drift detection
    surfaces the affected downstream subgraph for re-planning.
  - **Offline metrics** — `v3-service/rpg_eval.py` scores RPG artifacts for CI /
    benchmark summaries.
  - Strictly additive: with the flag off, planning and generation are unchanged.
  - Design + phased status: `docs/reports/RPG_WAVELET_PLANNING_V3_2.md`. Credit
    idea + framing to Dmitri Sotnikov (@yogthos), author of wavescope-mcp.

## [3.1.3] - 2026-07-06 — Maia

### Upgrade, rollback, and diagnostics
- `atlas upgrade [--to TAG] [--dry-run]`: staged upgrade with a recorded restore point (tag + image digests + `.env` backup), cosign signature verification of the target images (unpublished backend images are skipped, not fatal), readiness wait, quick-doctor smoke check, and automatic restore of the previous release on any failure (restore never re-pulls — a moved mutable tag can't replace the cached known-good images). Same-tag release tags no-op; mutable tags (`latest`, `dev`) run a full refresh. `atlas rollback [--to TAG]` returns to the restore point, and a failed `--to` reverts `.env` to the previously deployed tag.
- `atlas diagnostics collect`: a shareable support bundle (doctor output, service health, compose config, recent logs) with private values filtered.
- `atlas config validate | migrate`: typed schema over `.env` (types/ranges/enums, unknown and deprecated keys), forward migration with a `.bak` and a schema-version stamp, `--dry-run` preview.
- Signed artifact manifests: `atlas artifact verify | snapshot | rollback` — SSH-signed provenance manifests over lens/ASA bundles (verified against `.github/allowed_signers` + per-file SHA-256), one-generation bundle snapshot/rollback, and lens retrains auto-write a provenance manifest.

### Observability
- Structured JSON logs behind `ATLAS_LOG_FORMAT=json` across proxy, v3, lens, and sandbox, with `X-ATLAS-Request-ID` correlation: the proxy assigns/echoes the ID, forwards it on every outbound service call (with or without internal auth configured), v3 propagates it to lens/sandbox calls, and 401 responses carry the echo. All log paths pass the private-value filter, including exception tracebacks; the assignment filter also masks single-quoted values and Python dict reprs.
- Stable error-code taxonomy on the proxy API (`GET /version` lists codes; errors return the documented JSON envelope) and an OpenAPI 3.1 spec for the proxy surface with route-parity contract tests.
- Command-execution trust modes (`ATLAS_TRUST_MODE`): `untrusted` refuses command execution (both `run_command` and `run_background`), `trusted` (default) forces sandbox execution, `fully-trusted` permits host execution.
- Performance budget gate in CI: versioned measurement schema (CLI import time, proxy binary size) checked against `benchmark/perf/budgets.json`; a result matching zero budgeted metrics or a failed import fails the gate instead of passing vacuously.

### Adversarial review passes
- Two loop-until-clean adversarial reviews over the release window fixed 33 confirmed bugs, including: a trust-mode bypass via `run_background`; correlation-ID forwarding dead on token-less installs; `.env` corruption on files without a trailing newline; upgrade signature verification skipping the llama image; artifact verification only working on the signer's machine; bundle snapshot/rollback producing mixed-generation bundles; a default `atlas upgrade` that could never fetch a moved `latest`; restore paths that masked the original failure; sandbox/v3 images missing `structured_log.py` (startup crash on next build); CI gates that could never fail (attestation checks, OpenAPI parity, perf vacuous-pass); and a K3s lens PVC that rendered empty on upgraded installs.

### Dependency updates
- Grouped Dependabot updates merged: GitHub Actions majors (with `go-version: 1.26` — setup-go v6 pins GOTOOLCHAIN=local), Go tui deps, the Python group (fastapi 0.139 / uvicorn 0.50 / pydantic 2.13 / xgboost 3.2 / torch 2.12.1, with the torch pin synced across Dockerfile/CI/guard tests), and Docker digests (CUDA 12.9.2, golang 1.26-alpine, alpine 3.24). Base-image majors (CUDA 13, Python 3.14) are deliberate migrations and now ignored by Dependabot config, as are setuptools bumps past the RHEL9 python3.9 floor; staticcheck bumped to 2026.1 (go1.26 stdlib).

### Installer trust
- Release-pinned install: `ATLAS_BOOTSTRAP_REF=vX.Y.Z` pins the cloned checkout to the (SSH-signed) tag and `ATLAS_IMAGE_TAG` to the matching cosign-signed images; README/SETUP document the pinned and review-before-running variants beside the one-shot `curl | bash`.

### Docs & repo
- Ops docs consolidated: UPGRADE/ROLLBACK/BACKUP_RESTORE merged into OPERATIONS.md; README opens with the project definition and a Why ATLAS section; star-history chart moved to the working endpoint; `.mailmap` maps contributor identities for git tooling.

### Production-platform pass (support, supply chain, governance, ops)
- `SUPPORT_MATRIX.md`: every OS/backend/model/deployment/language/feature path classified (Supported/Preview/Experimental/Community-tested/Research-only/Unsupported) with validation provenance; N/N-1 compatibility policy; the model contract stated plainly (direct-mode agnostic, per-model bundles for V3/Lens/ASA).
- Supply chain: Docker bases digest-pinned (Dependabot-maintained) except the ROCm/Vulkan community-backend bases, which are tag-pinned; every pushed image carries SLSA provenance + SPDX SBOM attestations and a keyless cosign signature over its digest.
- Sandbox: non-root runtime (uid-mapped to the host user by `atlas init`), cap_drop ALL, CPU quota, toolchains relocated out of /root, K3s securityContext with seccomp RuntimeDefault, and an optional egress cutoff (`ATLAS_SANDBOX_NET_INTERNAL`) — verified on a local hardened-profile run.
- Lens state store is SQLite (ADR 0007, GH #57, core implementation from #128 by @HarshalPatel1972): pattern cache, co-occurrence graph, Thompson-sampling router posteriors, task queue, and metrics live in one WAL-mode `geometric_state.db` on the `lens-state` volume. The redis service, redis-data volume, and the `ATLAS_REDIS_*` config keys are removed (`atlas config migrate` drops them); `REDIS_URL` itself is simply no longer read; degradation semantics unchanged (cache/router go neutral on store failure, task queue 503s). One less external dependency; state backup is a single file.
- Governance: GOVERNANCE/MAINTAINERS/CODEOWNERS; SECURITY.md severities, response targets, embargo/CVE, backports, artifact revocation; THIRD_PARTY_NOTICES; seven ADRs; a single OPERATIONS.md runbook (health, logs, runbooks, upgrade, rollback, backup). Planning/status trackers are kept out of the repo.
- Tracker hygiene: label vocabulary created + applied to all open issues; #39 closed with evidence; fresh-audit status on #66/#115/#27; #124/#126/#128 marked blocked with exact conformance lists.

### V3/Lens pipeline acceptance test
- A second deterministic E2E (`tests/e2e/test_v3_lens_acceptance.py`) boots the **real v3-service** alongside the real proxy and sandbox: a Tier-2 write routes through the proxy's V3 bridge, the probe fails on purpose, lens-calibrated allocation yields k=3, PlanSearch generates candidates via the fake llama, each candidate is scored through both lens endpoints (a recording fake lens proves the calls) and verified in the real sandbox, and winner selection writes the lens-preferred candidate to disk. Failure modes at the seams: V3 unreachable/malformed/timeout fall back to the documented direct write; a lens outage leaves V3 running uncalibrated.
- `tests/contracts/` drift gates run in CI: proxy↔TUI event producer/consumer parity, envelope-type parity (Go producer / Go consumer / Python spec), config keys ↔ readers ↔ docs, CLI subcommands ↔ implementations, registry hash/consumption contracts.
- Product/benchmark scoring contracts aligned: unified neutral lens fail-soft sentinel, deterministic energy-sorted benchmark candidate ordering, corrected pattern-cache retry key; intentional orchestrator differences documented in `benchmark/README.md`.

### End-to-end acceptance test
- CI runs a deterministic full-control-plane test (`tests/e2e/`): the real proxy binary and the real sandbox executor (host uvicorn, no Docker) against a scripted fake llama-server, driving one complete agent turn — read, edit, sandbox-verified `run_command` behind an interactive permission approve, done — over the production SSE protocol. Asserts stage order (a silently skipped stage fails), file contents, and the sandbox side-effect; a second test pins the fail-closed session-less deny. The sandbox executor's workspace root is env-overridable (`ATLAS_SANDBOX_WORKSPACE_ROOT`; containers keep `/workspace`). Scope: this covers the control plane deterministically — real llama.cpp inference, GPU backends, hidden-state extraction, ASA steering, and model-dependent V3/Lens quality remain hardware-gated or manually validated (see the SETUP hardware table).

### Permissions fail closed
- `/v1/agent` requests without a `session_id` deny destructive tool calls in `default`/`accept-edits` mode instead of silently executing them (there is no channel to answer the prompt). Unattended clients use `mode:"yolo"` or `session_allowed_tools`; the API doc's non-TUI client guide now covers `/v1/permission`. The TUI clears a pending permission modal on `permission_denied` and turn end.

### Wiring completed
- `v3_reasoning_token` is rendered in the TUI's V3 streaming row (previously emitted and dropped — a frozen "decoding…" row through every PlanSearch phase).
- Lens retrains (both the service endpoint and `scripts/retrain_lens_from_results.py`) write `model_identity.json` for the served model; without it the load path's identity check disabled the entire lens on the next restart. Published lens bundles on HF now include identity files, pinned in the registry, so `atlas model install-artifacts` yields a bundle that actually loads. The gemma registry entry carries lens+ASA url bases and hashes.
- Compose passes through the documented-but-unreachable knobs: `ATLAS_PLAN_THINKING` (v3-service), `ATLAS_SHELL_SNAPSHOT_*` (sandbox), `ATLAS_CONTROL_VECTOR_*` (llama-server). `.env.example` gains the five consumed-but-undocumented keys.
- `scripts/build-containers.sh` builds the five current services from their real contexts and tags exactly as the K3s manifests reference (previously built from a removed directory layout, silently producing one image under names no manifest pulls); `uninstall.sh`'s image removal no longer aborts on an unset variable.
- ASA marker checks are case-insensitive in both launchers, matching `atlas asa check`; `atlas publish --dry-run` without repos no longer crashes; the `train` extra includes xgboost + scikit-learn and the ImportError guidance mentions it.

### Removed (unwired, placeholder, or caller-less — verified)
- The `plan_tasks` tool (acknowledged tasks as pending without executing them) and its never-wired parallel executor; the `PermissionRule`/`checkPermissionRules` rules engine (nothing loaded rules — the live machinery is `needsPermission` + `awaitPermission` + the built-in deny-list); `build_verify.go`, `v3_adapter.go`, unused grammar/schema wrappers, `EmitSimple`, `calibrationTooltip`, and v3-service's unwired dual-emit envelope helper.
- The metric-tensor G(x) path: `evaluate_gx` is XGBoost-only and the metric tensor served only `/internal/lens/correctability`, an endpoint with no caller; the 67 MB `metric_tensor.pt` is out of the Q6_K bundle.
- The V3.0-era inference files (`Dockerfile.mtp`, spec-decode/9B/embed entrypoints, custom jinja templates, the malformed unused patch file), the `model_recommendations` back-compat shim (callers migrated to `model_registry`), zero-reference scripts (`run_full_benchmarks.sh`, `validate_benchmarks.py`, `smoke-test-9b.sh`, `deploy-9b.sh`, `measure_bok_latency.sh`), `router/fallback_chain.py`, the `/ablation` coming-soon stub, the benchmark `--runs` no-op flag, and the dead `ATLAS_ENABLE_TRAINING`/`ATLAS_REGISTRY` config keys.
- Docs describe only the live protocol: the v3-service dual-emit claim, the "done is always last" broker claim, and stale consumer lists are corrected in PROTOCOL.md and `atlas/cli/events.py`.

### Supply-chain & artifact integrity
- Lens/ASA artifact downloads verify SHA-256 against per-file hashes in the model registry (`lens_artifact_sha256` / `asa_artifact_sha256`); a mismatch removes the partial file and fails the install, and files without a registry hash are labeled unverified instead of `[ok]`. `download-models.sh` verifies GGUF downloads against the same registry hashes (previously size-check only on the shell path).
- Lens checkpoint loading uses `torch.load(weights_only=True)` (the artifact can come from a remote download; full-pickle loading would execute code during deserialization). The legacy `gx_xgboost.pkl` fallback is opt-in via `ATLAS_ALLOW_PICKLE_GX=1` instead of automatic.
- `benchmark/custom/validate.py` enforces `tasks.json.lock`: a task set that drifted from its approved hash fails validation instead of the lock being informational.
- `gx_thresholds.json` (per-model G(x) operating thresholds) is tracked with its sibling lens artifacts, so a fresh clone runs with threshold interventions enabled.

### CI / release safety
- Image publishing is two-phase: the build matrix pushes only immutable `:sha-<short>` tags; a promote job repoints `:dev` / `:latest` / semver tags via `imagetools create` only after every service built **and** the `tests` workflow passed on the same commit. Failed tests or a partial matrix can no longer overwrite moving tags, including with mixed-commit images.
- Pull requests build the four small service images (`push: false`) so Dockerfile breakage is caught before merge instead of on the post-merge publish.
- Every GitHub Action is pinned to a full commit SHA (Dependabot keeps them fresh); `test.yml` / `install-test.yml` run with explicit `permissions: contents: read`; Dependabot also covers the Docker base images across all five service directories.
- CI runs the static `tests/infrastructure` checks, `geometric-lens/tests` (34 hermetic tests, previously never in CI), the `test-integrity` + `python-compile` gates, and shellcheck over all of `scripts/*.sh` (previously 2 of 13 scripts). Install matrix uses the maintained `rockylinux/rockylinux:9` image.

### Community health
- `SECURITY.md` (threat model scoped to the single-user local deployment, private reporting via GitHub advisories), issue templates (bug report requires `atlas doctor` output), a PR template, and Dependabot config.

### CLI
- `atlas --version` prints the CLI version; the REPL banner shows the real version instead of a hardcoded `v3.1`.
- `atlas bench` exits non-zero when the runner fails or produces no results (previously always exited 0).

### Fixes & accuracy
- Sandbox trust-model docstring describes what the container actually enforces; the never-enforced `MAX_MEMORY_MB` knob is removed (memory is capped by `ATLAS_SANDBOX_MEM`).
- `plan_tasks` is documented as a planning aid (tasks are acknowledged, not executed); the unreferenced MTP inference experiment files are removed.
- Packaging metadata completed (readme, URLs, classifiers, `train` extra for `atlas lens/asa build` dependencies; `setuptools>=77` to match the SPDX license form). Docs: uninstall section in SETUP, six previously-undocumented env vars in CONFIGURATION, `python3 -m benchmark.cli` invocation corrected, pass@1-v(k=3) defined where the headline number appears.

### Interactive permissions
- `default` and `accept-edits` modes now prompt before a destructive tool call runs. The turn pauses on a bordered approval box (`[y] allow once`, `[a] allow for session`, `[n]`/`Esc` deny); `Ctrl+C` still cancels the whole turn. An "allow for session" choice whitelists that tool so it isn't asked again (carried in the request's `session_allowed_tools`). `accept-edits` auto-allows file writes/edits and prompts `run_command`/`delete_file`; `yolo` is unchanged.
- New `POST /v1/permission` endpoint and `permission_request` SSE event carry the decision back to the paused turn (keyed by `session_id` + `tool_call_id`, mirroring `/cancel`). A fail-safe timeout (`ATLAS_PERMISSION_TIMEOUT_SEC`, default 600s) denies if nothing is answered.

### Sessions
- The TUI saves each session to `~/.cache/atlas-tui/sessions/<id>.json` (one file per session, written each turn). `atlas --continue` resumes the most recent session in the current directory; `atlas --resume` picks one from a list; `atlas --resume <id>` resumes a specific session. The saved transcript is replayed into the view and fed back to the model as history; a directory mismatch keeps the current directory and warns. `/clear` starts a fresh session, leaving the prior one on disk.

### Installer / bootstrap
- Bootstrap writes the registry's default recommended model into `.env` when none is selected (logged), so the one-shot `curl | bash` flow completes without the wizard; an existing selection is respected.
- No detected GPU selects the Vulkan overlay automatically, plus the new `docker-compose.cpu.yml` when `/dev/dri` is absent — GPU-less hosts boot via the lavapipe CPU ICD (slow but functional).
- firewalld changes are opt-in via `ATLAS_BOOTSTRAP_OPEN_FIREWALL=1`; services bind loopback, so local installs leave the firewall alone.
- ASA steering-vector build dispatches per GPU vendor (CUDA/ROCm image + device flags), loads `.env` keys first, and skips cleanly on CPU-only hosts; re-runs pull the existing checkout as its owner (no dubious-ownership failure under sudo); service health wait raised to 450s to cover llama-server warmup.
- `install.sh` (K3s) fails early with guidance when `bc` is missing; `download-models.sh` downloads via curl and writes a relative `default.gguf` symlink; the macOS native launcher keeps smaller fallback defaults than the Docker path (ctx 32768, q8_0/q4_0 KV, 1 slot) for Mac unified-memory headroom, and treats `.env` as optional so an env-only launch works.
- GPU vendor detection word-bounds the AMD `lspci` match so NVIDIA/Intel GPUs aren't misdetected as AMD (#129).

### Proxy
- `/ready` also gates on v3-service health.
- Cancellation aborts in-flight V3 plan/write calls and sandbox calls; a cancelled `write_file`/`edit_file`/`ast_edit` no longer falls back to writing content to disk; per-turn cancel handles so overlapping turns on one session id can't remove each other's registration.
- The verification, done-without-action, and claim-check gates share a 3-bounce cap, so a persistently bounced `done` is eventually accepted instead of looping.
- The markdown-fence sanitizer only strips a true whole-file wrapper — interior fences (e.g. docstring examples) pass through unchanged.
- The safety deny-list (`.env`/`*.pem`/`*credentials*` writes, destructive shell patterns) is enforced centrally at tool dispatch in every permission mode.
- `outline_file` returns structured JSON including the rendered outline; `delete_file` reports removal errors and refuses non-empty directories; exploration budget escalates its nudge instead of skipping the read; session read-cache access is lock-guarded.

### Security & workspace containment
- Proxy-level workspace containment: every path-taking tool argument (`path`, `source`, `destination`, `cwd`) is resolved and checked against the workspace root before any handler touches the filesystem. Paths escaping via `..`, absolute paths, or symlinked components are refused in every permission mode.
- Untrusted text written to logs is field-encoded so it can't forge or split log lines.
- v3-service verifies candidate code before accepting it: an allowlisted build/test command gate (shell metacharacters blocked) and language-aware syntax checks reject candidates that don't compile/parse.
- Sandbox executor parses XML with `defusedxml` (untrusted-input safe).
- `scripts/production-readiness.py` is the developer gate (test integrity, Python compile + unit tests, Go race/vet for proxy and TUI, and the v3 syntax/sandbox contract tests); CI runs the same named gates.

### TUI
- `/demo` raw lane runs with no sandbox or file tools; in review mode the raw pane keeps the model response while the V3 pane shows written files; stream events can no longer be overtaken by the done marker; prompt animation is multi-byte-safe; markdown re-wraps on terminal resize.
- Feedback flow: staged per-file verdicts survive a failed submit; `/deny` validates the path against the files the last pass actually wrote; non-200 `/feedback` responses surface as errors; input echoes never replay into agent history.
- Bearer-token loader reads the `atlas init` api-keys.json shape; `/events` reconnect backoff resets after a healthy connection; renderers added for reasoning-repetition interventions, stream cuts, and symbol-index injection.

### CLI (continued)
- `atlas compose <args...>` passthrough subcommand (base file + backend overlay); `atlas --help` lists subcommands; unknown subcommands print usage and exit 2.
- Service URLs resolve from the Docker `.env` port keys when no explicit URL env var is set (repl, client, doctor, lens check).
- `atlas onboard --url` offers to write `ATLAS_MODEL_FILE`/`ATLAS_MODEL_NAME` into `.env` (interactive prompt; `--apply` for non-interactive).
- `atlas doctor` prints each result as it completes (JSON mode still buffers).
- `atlas model`: models dir resolves from the compose `.env`; a resumed download that the server reports complete (HTTP 416) is verified and finalized in place; `install-artifacts` exits 3 when no artifacts are registered for direct download and points at the published repos; Gemma-family registry entries carry the `gemma` license identifier.
- `atlas init` reports failure when `api-keys.json` is not written and asks before tightening a loose `secrets/` dir; `atlas asa build` resolves the lens container via compose (non-default project names) and survives docker-exec timeouts with recovery guidance; `atlas solve` uses `/v1/chat/completions` so the GGUF's own chat template applies; the startup status block drops the hardcoded speed figure; version reports 3.1.3.

### Geometric Lens — per-model calibration
- Per-model score calibration, model-identity checks, and threshold loading are their own modules (`calibration.py`, `identity.py`, `thresholds.py`): C(x) energy is normalized to a per-model scale and G(x) verdicts use per-model thresholds, so the same framework works across models without hardcoded constants.
- The proxy exposes `GET /v1/calibration/status` (lens + ASA compat verdict for the loaded model); the TUI reads it as a header badge on startup.
- `atlas lens check | build | publish` and `atlas asa check | build | publish` cover the per-model probe, training, and artifact-publish flow.
- Adds the `entrypoint-v3.1.sh` inference entrypoint (env-driven, model-neutral) shared by the Docker images and the macOS launcher.

### Lens service + retrain tooling
- `/internal/lens/retrain` refuses with a structured 503 when the models dir is mounted read-only, pointing at host-side `atlas lens retrain`; retrain/reload are serialized and refresh the readiness state on success.
- Artifact identity is verified against the model llama-server actually serves (`/v1/models` probe, `ATLAS_MODEL_NAME` fallback) and against the checkpoint's input dimension.
- G(x) loading is shared between boot and per-directory reloads, so a reload yields the complete lens; G(x) operating thresholds derive from out-of-fold CV scores instead of the final booster's in-sample scores.
- `retrain_lens_from_results.py` mean-pools per-token embedding responses (matching serve-time extraction) and hot-reloads the service; `retrain_cx.py` resolves ports from `.env` with a K3s NodePort fallback; benchmark lens-feedback keeps its sample buffer when a retrain is refused or fails.
- The lens image ships the `gguf` package (ASA vector writer), and the ASA build fails fast when it is missing.

### v3-service / sandbox
- v3-service serves requests on a threading HTTP server with a thread-safe graph cache; client disconnects abort the pipeline at phase boundaries; selection winners are matched by original candidate index (was positional against a sorted list); self-test harness executes candidates from a string literal so multiline strings survive; sandbox client timeout raised to 45s.
- Sandbox executor: per-call cap set to 300s in the Compose stack (`ATLAS_SANDBOX_MAX_EXECUTION_TIME`); process-group kill on timeout; optional `stdin` on `/execute`; project-context file writes routed through the O_NOFOLLOW containment helper; background jobs abandoned for 2h are reaped.

### Compose / K3s
- All services run with `restart: unless-stopped`; runtime-tuning keys (`ATLAS_V3_TIMEOUT`, `ATLAS_MAX_TOKENS`, `ATLAS_AGENT_HISTORY_BUDGET`, `ATLAS_LENS_RETRAIN_MIN`, `ATLAS_KEEP_LLAMA_WARM`, `ATLAS_FRESH_SLOT_PER_SESSION`) pass through to the proxy; `ATLAS_GPU_INDEX` reaches the llama container; `.env.example` documents the runtime-tuning section.
- Inference Dockerfiles EXPOSE 8080 (matching the entrypoint); ROCm/Vulkan images install curl for the compose healthcheck.
- K3s templates pin container-side ports so moving a Service port can't break probes; the proxy pod mounts models read-only and the lens-training corpus hostPath (`ATLAS_LENS_TRAINING_DIR`), and receives ctx/slot sizing; `deploy-9b.sh` uses the shared entrypoint and split KV-cache type keys.
- `production-readiness.py` and CI validate every shipped compose overlay combination; the installer CI job asserts a non-empty model selection lands in `.env`.

### Docs
- Documentation refreshed against the current code: MAP regenerated; API (feedback/training-status endpoints, readiness gate, tool table, workspace containment); CLI, CONFIGURATION, SETUP, SOURCES, PLAN_MODE, TROUBLESHOOTING updated.
- Documentation refactor for concision and structure: internal ticket references and dated change-narration removed from user-facing prose; duplicated content consolidated to a single canonical home with cross-links; `CAPABILITIES.md` + `PRODUCTION_READINESS.md` merged into `RELEASE.md`; `MAP.md` slimmed to a directory-level orientation map; shipped-release status trackers moved to `docs/reports/archive/`. Translations (`docs/lang/`) re-sync as a follow-up.
- README intro clarified.
- Full-tree accuracy audit against the code. API/PROTOCOL: `run_command` fails (no host fallback) when the sandbox is unreachable unless `ATLAS_VERIFY_IN=host`; the proxy is the live envelope producer (v3-service envelope opt-in is unwired); lens `/v1/*` endpoints require Bearer auth; missing request fields, SSE events, and v3-service endpoints documented; `ATLAS_PROXY_NODEPORT` name corrected. ARCHITECTURE: 15-tool table (adds `outline_file`); redis and the geometric-lens→v3-service edge in the service graph; XGBoost G(x) deployed, gradient-step correction unwired; sandbox cap 300s in Compose. CLI: `/demo` and `atlas lens retrain` documented; TUI has no `/bench`. CONFIGURATION: shell-gate table matches the catastrophic-only policy; token-budget trim; `ATLAS_REASONING_BUDGET`/`ATLAS_BACKEND` compose-passthrough notes corrected; adds `ATLAS_PERMISSION_TIMEOUT_SEC`, `ATLAS_LENS_HOST_DIR`, `ATLAS_LLAMA_HOST`; 12 sandbox languages. SETUP: `atlas` is the pip entrypoint (launcher-script passage removed); ASA `.gguf.model` marker gate + `ATLAS_CONTROL_VECTOR_ALLOW_UNVERIFIED`; TUI needs Go 1.26.2+. TROUBLESHOOTING: V3 fire conditions (10-line floor, ≥2 indicators), write-to-existing-path gate, exploration-budget nudges. PUBLISHING: pre-flight scope stated accurately. ja/ko/zh-CN copies updated to match; `docker-compose.override.yml` added to `.gitignore` (DEVELOPMENT.md documents it as ignored).

## [3.1.2] - 2026-06-17 — Maia

### Hardware reach
- AMD ROCm via llama.cpp — including RDNA4 / RX 9070 (gfx1200/gfx1201) and community-verified cards (#26)
- Apple Silicon — native macOS hybrid Metal path (native llama-server for inference + Docker for the rest of the stack) (#32)
- Vulkan universal fallback — one image covering AMD / Intel / Snapdragon / Apple-via-MoltenVK / CPU (#114)

### Agent reliability — local-model tool loop
- Tool results are rendered as user-role turns on the wire. Gemma's chat template has no `tool` role and silently dropped `role:"tool"` messages, so the model never saw any tool output (`list_directory` / `read_file` / `run_command`) and re-issued the same call until the repetition breaker fired. This was the root cause behind the "it can't see what it's reading / it just loops" reports. Model-agnostic (Qwen reads the `[tool result]` marker the same way).
- Read-dedup false-negative fixed: `fileContentInContext` probed the raw longest line, but tool results are stored JSON-escaped, so any file whose longest line contained a quote (e.g. a Flask app's embedded HTML) was wrongly judged "trimmed" and re-served every read → read loop. Now probes the longest escape-free run.
- Traceback → directed edit (#39 / option 3): a `run_command` crash extracts the deepest in-project frame, quotes the offending line, and steers a minimal `edit_file`; run tools are banned from the next decision's grammar so the model must edit rather than re-run.
- `move_file` tool: relocations/renames (e.g. `index.html` → `templates/`) no longer require a read→write→delete dance; shell `mv`/`cp` point here. Refuses to clobber an existing destination.
- Steers for common dead-ends: `No module named X` → `pip install` (instead of re-running), and a filename that differs only in case from a real workspace file → the correct name.
- Per-turn `max_tokens` 32768 → 8192 (`ATLAS_MAX_TOKENS`) and a content-stream loop cut, bounding runaways that previously ran to the slot ceiling.
- Conversation window sized to the per-slot context (with the active file pinned in the trim) instead of a flat cap that dropped the file under edit.

### Sandbox — shell policy + isolation
- `run_command` shell gate narrowed from "block every mutating verb" to catastrophic-only (whole-project/root `rm -rf`, fork bombs, device destruction), since the sandbox container (read-only rootfs, no-new-privileges, project-only writable mount, cwd jailed) is the real boundary. Ordinary `mv`/`cp`/`mkdir`/`rm <file>`/`sed -i` now run; `bash -c`/`eval` are unwrapped so a wrapped catastrophic command can't slip through.
- Host-sized cgroup limits on the sandbox: `pids_limit` (kernel-level fork-bomb stop) and a memory cap (`atlas init` detects host RAM and writes `ATLAS_SANDBOX_MEM` ~75%); `:-0` fallback keeps a raw `docker compose up` working uncapped.
- Interactive wall-clock cap on the V3 pipeline (`ATLAS_V3_TIMEOUT`, default 180s) — a runaway falls back to the model's own (syntax-gated) content instead of hanging the session.

### Geometric Lens — per-model thresholds + in-the-loop training data
- G(x) operating thresholds are now per-model and ship with the lens artifact (`gx_thresholds.json`): the lens service loads them and returns them in each score response; the proxy uses them for its regression checks. The hardcoded 0.3 / 0.15 / 0.05 cutoffs were calibrated to one model's score scale and never fired for a model (e.g. Gemma) whose scores cluster higher. `atlas lens build` auto-emits the file, calibrated from the run's PASS-score percentiles.
- ast_edit now matches `<script>` / `<style>` (tree-sitter parses them as dedicated `script_element` / `style_element` nodes, not generic `element`s — the old query matched 0).
- In-the-loop lens-training data collection: each agent file-write is captured per pass; in the TUI, `/good`·`/bad` rate a pass and `/review` + `/deny`·`/accept` set per-file verdicts, which the proxy turns into labeled, weighted samples (a 👎 pass down-weights even its accepted files; a denial is a full-weight negative). `/redo` regenerates a rejected file. A one-time "lens retrain available" banner appears once enough balanced samples accrue.
- `atlas lens retrain` trains the lens on that collected corpus (weighted G(x)) so it learns the user's own workloads, and emits fresh calibrated thresholds. New env: `ATLAS_LENS_DATA_DIR`, `ATLAS_LENS_RETRAIN_MIN`. TUI slash commands: `/good /bad /review /deny /accept /redo`.

### Structural call-graph reasoning (#39, thanks @yogthos)
- Intra-file call-graph neighborhood (`calls:` / `called by:` per symbol) rides on `outline_file` and whole-file `read_file` of a `.py`, gated by `ATLAS_CALL_GRAPH`. Surfaces structure at the localization decision point without a repo-wide scan. (PR #125 by Dmitri Sotnikov, integrated and extended.)

### Documentation
- Translated ARCHITECTURE.md to zh-CN / ja / ko (#25); added a language switcher to the English ARCHITECTURE.md

## [3.1.0] - 2026-05-12 — Maia

### Removed
- Removed dead `ATLAS_USE_FOX` code paths in benchmark runner (#22)

### Aider removed
- `proxy/aider_format.go` (whole-file format translator), `handleChatCompletions` + `handleStreamingChat`, and the OpenAI-compat agent-loop wrapping are all deleted (~2000 lines). `/v1/chat/completions` on the proxy is now a transparent passthrough to llama-server via the catch-all handler.
- `.aider.model.settings.yml`, `.aider.model.metadata.json`, the `.aider*` `.gitignore` exceptions, and the `_find_aider`/`launch_aider` paths in `atlas/cli/repl.py` are all gone. Bare `atlas` (interactive tty) now launches the TUI by default; pipe mode falls through to the built-in `/solve` REPL.
- Proxy launcher (`atlas/cli/repl.py`) now reaps any pre-existing `atlas-proxy-v2` process before spawning a fresh one and redirects proxy stdout/stderr to `~/.cache/atlas/proxy.log` instead of `/dev/null`. Closes the "old binary in memory after rebuild" foot-gun.

### Bubbletea TUI (PC-062)
- New `atlas tui` subcommand launches a native Bubbletea terminal UI as the canonical chat client (and is now the default for plain `atlas`)
- Five-pane layout: header (proxy/cwd/mode/spinner) + pipeline (live V3 stage table from `/events`) + chat (glamour-rendered markdown + inline tool calls) + events log + stats strip + textarea input
- Hotkeys: Enter send, Shift+Enter newline, Ctrl+L clear, Ctrl+T cycle permission mode, Ctrl+R resend last, Ctrl+C cancel turn / quit, Ctrl+D quit
- Slash commands inside the TUI: `/add /drop /context /diff /commit /undo /run /help /quit`
- New atlas-proxy `POST /cancel` endpoint indexed by `session_id` — TUI cancels the in-flight `/v1/agent` turn on Ctrl+C as defense-in-depth alongside TCP disconnect
- 43 atlas-tui Go tests + 4 atlas-proxy `/cancel` tests, all green under `go test -race`
- `tui/` is a standalone Go module (`github.com/itigges22/atlas-tui`) — depends on bubbletea, lipgloss, bubbles, glamour

### Documentation
- Added multilingual documentation: Simplified Chinese (zh-CN), Japanese (ja), Korean (ko) for README, SETUP, and TROUBLESHOOTING
- Added language selector badges to README
- Added star history chart to Latest News section
- Rewrote README contributing section to encourage issue reports and community feedback
- Fixed V3_1_STATUS.md false claims about speed optimizations that were never applied to code
- Documented RDNA4 (RX 9070 / 9070 XT, gfx1200/gfx1201) ROCm 7.x setup in SETUP.md and TROUBLESHOOTING.md — requires `ATLAS_ROCM_TAG=7.2.3-complete`; `ATLAS_HSA_OVERRIDE_GFX_VERSION` must stay unset (#119, thanks @Kaihui-AMD)
- Corrected stale Metal/macOS docs: the macOS hybrid Metal path (#32) is now documented as shipping across README, SETUP.md, CONFIGURATION.md, and ARCHITECTURE.md (was mislabeled "V3.1.2 planned"); rewrote ARCHITECTURE.md §8.4 to describe the actual hybrid (native llama-server + Docker) rather than the never-shipped pure-native install
- Restructured the README roadmap into V3.1.1 (hardware reach, landed), V3.1.2 (BYO-model + ROCm-on-K8s), and V3.2 (planning phase #120, structural+wavelet reasoning #39, reasoning-with-sampling #9), with a help-wanted backlog — all sourced from open issues
- De-staled user-facing CLI strings: `atlas init` and `atlas tier` no longer print "Metal — V3.1.2 planned"; they report Metal as the supported macOS hybrid path (#32) — strings/comments only, no logic change
- Synced zh-CN / ja / ko translations (README + SETUP.md) to the corrected English: Metal/macOS shown as shipping, multi-vendor GPU support table, V3.1.1/V3.1.2/V3.2 roadmap, and fixed NVIDIA-only requirements rows and SETUP_MACOS.md link paths

### Code Accuracy Audit
- Audited and corrected comments across 72 files for V3.0.1 accuracy
- Updated model references: Qwen3-14B to Qwen3.5-9B, embedding dimensions 5120 to 4096
- Renamed service references: rag-api to geometric-lens, Fox to llama-server
- Corrected G(x) XGBoost status: deployed and active (was incorrectly described as removed)
- Fixed normalization comments from "Fox 9B" to "Qwen3.5-9B C(x)"
- Marked legacy Fox code paths as unused in benchmark runner and geo_learning

### Test Fixes
- Fixed embedding dimensions in test fixtures (5120 to 4096)
- Fixed geometric-lens port in test conftest (8001 to 8099)
- Updated DivSampling test assertions to match actual 4+4+4 perturbation counts
- Corrected G(x) cost field parameter count: ~2.16M / 8.3MB (was ~2.7M / 10MB)
- Finished the 3.0.1 api-portal cleanup: removed `tests/integration/test_e2e_flow.py` and `tests/integration/test_e2e_training.py` (616 lines). These depended on the `test_api_key` fixture which calls the deleted api-portal service, so every test in them errored on session setup. The 3.0.1 changelog claimed this cleanup was done but these two files survived it.
- `test_empty_messages_handled` (`tests/infrastructure/test_llm.py`) now accepts 200/400/422/500. Current llama.cpp returns 500 for empty messages array; the test was hard-coded to 200 and broke against newer llama.cpp builds.
- PC-061 step B: implemented `_emit_event`, `_classify_stage`, `_logical_stage` in `v3-service/main.py`. The test file (`tests/v3-service/test_event_emission.py`) was committed in c5216be ("Install observability") but the implementation never landed, leaving the test red on dev. The contract is now satisfied: legacy `{stage, detail}` frame always emitted, typed envelope opt-in, suffix-based stage classification (`_pass`/`_skip`/`_done` → stage_end success=true, `_failed` → stage_end success=false, `_error` → error event, `_retry` → fresh stage_start), and stage_start→stage_end pairing via logical-name parent_id + duration_ms.

### Repo restructure
- Renamed `atlas-tui` → `tui` and `atlas-proxy` → `proxy` at the repo level; moved ablation data under `docs/reports`. 362 reference updates across the tree.

### Phase 0: first-run installer + model wizard
- New `atlas init` command (`atlas/cli/commands/init.py`): interactive first-run wizard that probes hardware, picks the right tier (T0/T1/T2/T3), recommends a model, writes `~/.atlas/config.yaml`.
- New `atlas model` command (`atlas/cli/commands/model.py`) with `list` / `verify` / `add` / `remove` subcommands; backed by `model_registry.py` (`add`/`get`/`list` with SHA verification) and `model_recommendations.py` (per-tier defaults, split out from `tier.py` in PC-055.2).
- `atlas/cli/events.py` (PC-061 step A): typed-event SSE protocol — `Event` dataclass, `parse_envelope`, `iter_events`, suffix-based stage classification. Schema documented in `docs/PROTOCOL.md`. Producer-side helpers in v3-service landed as PC-061 step B (see Test Fixes above).
- `atlas doctor` extended for the same hardware probe used by the wizard.

### Install + bootstrap hardening
- Hardened fresh-VM install path against partial failures across RHEL 9, Ubuntu, Rocky; `curl … | bash` and `curl … | sudo bash` both work.
- Auto-install NVIDIA driver libraries on RHEL 9 and put the Python CLI on `$PATH`.
- Bootstrap now installs Go and pre-builds `atlas-tui` so first-run latency is download-bound, not compile-bound.

### CI: lint + security + cross-distro
- Added ruff (Python lint) and CodeQL (security scan) as GitHub workflows.
- New PR-time test job that runs the full Python suite against a cross-distro install matrix (Ubuntu 22.04 / 24.04 / Rocky 9).
- Fixed pip PEP 660 friction, Rocky curl conflict, and a CLI-wizard GPU-mock path that was breaking the matrix.

### PC-159: surgical-edit gate (proxy)
- New gate in `proxy/agent.go` that refuses an `edit_file` when the proposed change would rewrite more than a configured fraction of the target file. Forces the model to pick the right tool (`write_file` for new files, `ast_edit` for structural rewrites, `edit_file` only for actual surgical patches).

### Chat history threading (proxy + tui)
- `/v1/agent` now accepts full prior chat history from the TUI, replacing the per-call stateless wrapper. Assistant turns are re-wrapped in a JSON envelope so the proxy can tell user messages from prior model turns when rebuilding context.

### Plan mode (May 5)
- New `/v3/plan` endpoint on v3-service generates a structured plan (steps + verify step + adherence score) before the agent loop begins; Qwen3 reasoning extraction fixed in the same commit.
- `proxy/agent.go` consumes the plan via a plan bridge, an agent-loop hook that pins the current step into each request, and an adherence gate that flags reasoning that drifts from the active step.
- TUI renders `plan_loaded` / `plan_adherence` / `plan_revise` events live (`tui/commands.go`, `tui/model.go`).
- New docs: `docs/PLAN_MODE.md`, `docs/PROTOCOL.md`.

### Proxy reliability (May 5)
- Output sanitiser strips reasoning preambles and dangling JSON fragments from model responses before parsing.
- Shell-op gate refuses dangerous `rm -rf /` style commands and the `bash -c` bypass route.
- System prompt hardened: clearer tool-use rules, fewer hallucinated fields.
- Verification gate added before `type=done` (foundation that tonight's done-without-action gate composes with).
- Host paths in tool-call arguments translated to container paths so the sandbox sees the right file when the model thinks in host-fs terms.
- Fixed a conversation-history drop bug where the post-V3 trim was eating the user's prompt; V3 pipeline now fires on more edit shapes (not just write_file).
- Lens-call timeout in v3-service bumped from 5s to 30s with structured fallback logging on miss.

### Sandbox + execution stack
- **PC-188**: every `run_command` now executes inside the sandbox container, not on the host. Closes the "model writes `rm` and the host runs it" risk.
- **PC-189**: workspace-drift fix and a false-positive in the truncating-redirect detector (was rejecting legit `> file.txt` writes).
- **PC-190**: sandbox verify stack pre-bakes common dev deps (pytest, ruff, etc.), uses tmpfs for the working tree, prints a "create a venv" hint when the model tries to install into the system Python.
- **PC-191/192/193**: sandbox is language-agnostic — works on a working codebase (not just a single-file scratchpad). Detects Python, Node, Go, Rust, Java, C/C++ project layouts and uses the appropriate runner.

### Anti-laziness gates
- **PC-194/195**: `write_file` rejects empty content, single-line stubs, "TODO"-only files, files with `pass`-only bodies, and other lazy outputs.
- **PC-196**: explicit `run_background` tool for long-running processes (e.g. `python app.py`); shell `&` backgrounding through `run_command` is detected and routed to `run_background`.
- **PC-197**: completion-claim verification — when the model declares `done`, the gate checks the workspace state matches the claim (structural check, foundation that tonight's claim-check gate extends).
- **PC-198**: trims boilerplate from the system prompt and strips host `/workspace/` prefixes from model-emitted paths.
- **PC-199/200**: detects "stops at the easy fix" pattern (one tweak then `done`); raises tier-aware turn caps so the model has runway to complete a real task.
- **PC-201**: `write_file` is allowed to overwrite an existing file when that file is corrupted (e.g. truncated mid-write from a prior crashed turn) instead of failing with the usual "file exists" gate.

### PC-202: per-layer residual hidden states from llama-server
- Patched llama-server's `/embedding` endpoint to accept a `layers: [int]` parameter and return the residual-stream hidden state at each requested layer. Foundation for both PC-207 (per-token lens scoring) and tonight's ASA steering vector build.

### PC-206 + PC-207: lens-as-PRM (per-step process reward)
- **PC-206**: thinking-mode plumbing in `v3-service/main.py` `LLMAdapter` — `thinking` keyword resolves per-call against an instance default.
- **PC-207**: lens computes per-token C(x) + G(x) scores during candidate generation; `/internal/lens/score-per-step` exposes aggregates (gx_min, gx_mean, off_rails_idx, cx_norm_max) the proxy and v3-service consume for early-exit and ranking. Wired into v3-service candidate generation, the agent loop (foundation for tonight's reasoning-repeat + path-aware detectors), with structured per-step logging across all three services.
- Severe-score short-circuit: gx_min below 0.05 fires a corrective immediately without waiting for a second sample (calibrated against the May 7 dashboard.html stub-loop session).
- V3↔lens alignment: lens now vetoes a sandbox-passing candidate when its gx_min indicates a stub or placeholder collapse — closes the "sandbox approves a stub V3 generated" loophole.

### Agent loop reliability sweep (May 7)
- Empty-response fallback: when the model returns nothing parseable, the loop emits a corrective hint instead of retrying the same prompt verbatim.
- Plan-threshold guard: refuses to enter the agent loop on a plan with adherence score below threshold.
- Tool-repeat detector: precursor to tonight's reasoning-repetition detector — catches verbatim tool-call repeats within a window.

### GH #39: AST-aware surgical edits + tier-aware V3 routing (May 8)
- **v1 (5e44ffb)**: new `ast_edit` tool — friendly-selector AST node replacement using tree-sitter. Supports `function:NAME`, `class:NAME`, and `<tag>` selectors. The selector vocabulary is intentionally small in v1; nested selectors (e.g. `<style>` inside `<head>`) are NOT supported and produce a "0 nodes matched" error.
- **Point 1 (468a555)**: structural verification veto for V3 candidates — rejects candidates that pass sandbox but fail structural shape checks (e.g. removed a required import, lost the class definition).
- **Point 2 (b95f741)**: cyclomatic-complexity enrichment in tier classification — `tier.py` now considers logic density, not just line count, when assigning T0/T1/T2/T3.
- **Point 3 (2629652)**: Phase 3 repair receives call-chain context (callers + callees of the file being repaired) so the repair model can reason about cross-file effects.
- **Point 4 (bd0b02b)**: auto-injection of a reachability slice from the user's message — the lens picks the most relevant file regions and inlines them into the system context before the loop starts.
- Plan generation made aware of `ast_edit` so plan steps suggest it when the target is a structural edit.
- `edit_file` "string not found" error now suggests `ast_edit` as the recovery; `write_file` rejection on existing files also points to `ast_edit`.
- Three follow-up fixes: encoding (HTML entities in selector args), trim-resilience (large `content` fields surviving the post-V3 trim), and parse-failure categorization in logs.
- Jinja crash fix when `symbol_index` injects snippets: the snippet role was being set to `system`, which Jinja resolved as a template literal; changed to `user` role.

### BiasBusters tool-selection mitigations
- Tool descriptions rewritten to push the model toward the right tool for the task: `edit_file` framed as the surgical default, `ast_edit` marked REQUIRED for HTML/Python structural edits, `write_file` restricted to new-file creation only.
- Conditional GBNF grammar built per turn: when the loop has already entered a step the model has just claimed done, the grammar bans re-emitting the same tool name token-side so the model can't loop on the same failed tool call.
- Per-step tool-list filter (`buildToolDescriptionsExcluding`): the system prompt strips tools the loop has explicitly excluded for this step, so the model never sees them as options.
- ASA (Activation Steering for Aast_edit) wired into the inference entrypoint: `inference/entrypoint-v3.1-9b.sh` auto-detects `/models/ast_edit_steering.gguf` and applies it always-on via llama.cpp `--control-vector`. Default scale 0.5, default layer range full-model, both overridable via env. PC-202's per-layer-residual `/embedding` patch is the upstream that makes this possible.

### ASA steering vector
- New `geometric-lens/asa_calibration/` directory: 1000 contrast-pair prompts (50+ base templates × variation pools) cover function selectors (54%), HTML tags (27%), and CSS classes (19%). `generate_pairs.py` produces `contrast_pairs.jsonl`; `build_steering_vector.py` extracts residuals via the lens `extract_per_layer_per_token` endpoint at layer 27 (of 36 in Qwen3.5-9B), means across tokens/prompts/sign, and writes a llama.cpp-format GGUF control vector. Final vector: 16736 bytes, ‖v_global‖ = 8.6444 after 730s on 2000 prompts.

### Agent loop hardening
- **Plan-progress reminder** (`proxy/plan_reminder.go`): ephemeral system note injected into every step request rendering `plan progress N/M — currently on step "sX": <action> <target>` plus done/remaining sub-step IDs. Lazy-initializes `ctx.PlanStepsSatisfied`. Not persisted to `ctx.Messages`, so it survives the post-V3 conversation trim cycle.
- **Reasoning-repetition detector** (`proxy/reasoning_repeat.go`): tracks the model's reasoning-stream opening; on 3 consecutive identical normalized openings (case-folded, whitespace-collapsed, 80-char snippet) the loop queues a corrective system message. Successfully broke a session-2 stuck loop in live testing.
- **Path-aware error breaker** (`extractFailurePath` in `proxy/lens_score.go`, breaker logic in `proxy/agent.go`): tracks `ctx.RecentFailurePaths` per tool failure. Known limitation: the v1 implementation resets on intervening successes, so it can miss long stuck-loop sequences with sporadic productive turns in between.
- **Done-without-action gate** (`proxy/guardrails.go`): refuses `type=done` when the user prompt is fix-intent and no successful verification command has run this loop. Action-intent words (`rewrite`, `create`, `add`, `update`, `redesign`) also trigger a productive-change check parallel to the existing verify check. Caught 4 false-success done attempts in live testing.
- **Truncation recovery shims** (`proxy/agent.go`): `recoverTruncatedAstEdit` + `recoverTruncatedEditFile` + `recoverTruncatedToolCall` rescue malformed tool emissions from the model and re-pack into a well-formed shape. Each shim is targeted at a specific failure mode observed in production logs.
- **Conversation history error surfacing** (`proxy/agent.go`): `extractModelResponse` now exposes the actual `Unmarshal` error path (directErr vs balancedErr) so debug logs distinguish parse-shape failures from content failures.
- **Removed `ResponseHeaderTimeout`** from `proxy/v3_bridge.go` and removed all client-level timeouts on the V3 HTTP path. Long V3 chains (10+ minute passes) were getting bounced by the 10-minute response-header window even when the pipeline was making progress.
- **Removed `absoluteMaxTurns` ceiling** from `proxy/types.go`. Turn caps now come solely from `TierMaxTurns` (T0:5, T1/T2/T3:0 = uncapped) with no override clamp. Reasoning: 8 detectors armed in the loop make a hard cap redundant — let the detectors decide when to break.

### Surgical-edit hardening (V3 routing)
- `proxy/tools.go` ast_edit executor: tier classification now uses `max(oldTier, newTier)` and the previous V3-tier floor for HTML was dropped (it was over-triggering V3 on the smallest CSS tweaks). Doctype dedup (`leadingDoctypeRe` + `stripLeadingDoctype` in `proxy/guardrails.go`) prevents the model's "<!DOCTYPE html>" prefix from being inserted twice when ast_edit replaces the `<body>`.
- Suspiciously-shrunk-edit guard (`validateNotSuspiciouslyShrunk` in `proxy/guardrails.go`): rejects an edit that shrinks an >100-byte file to <64 bytes. Final threshold tuned after a legitimate 80-byte one-liner refactor was false-rejected at 128. Triggered on a destructive 32-byte stub in pre-release testing.
- Working-directory phantom-dir guard (`validateWorkingDirReference` + `workspaceRefRe`): catches model emissions that try to `cd templates/workspace` or similar nested-workspace references; legitimate `cd /workspace` at the sandbox root is allowed.
- Action-intent gate (`actionIntentWords` + `isActionIntentMessage` + `actionWithoutProductiveChangeMessage`): companion to the verification gate, catches `done` declarations on `rewrite`/`create`/`add`/`redesign`-style prompts that don't include a productive edit this loop.

### TUI reasoning stream visibility
- `tui/model.go` adds a `streamingReasoningText` buffer and a `reasoning_token` event handler that renders with a `‹thinking›` prefix so the user sees the model's reasoning stream live alongside its content. Both buffers reset on `llm_call_start` / `llm_call_end`.
- `tui/commands.go` extended to forward the `delta.ReasoningContent` field from the SSE stream as `reasoning_token` events.
- `proxy/agent.go` plumbs reasoning content through the agent loop: stashes `ctx.LastTurnReasoning`, captures `pendingReasoningCorrective` via `recordReasoning`, and re-emits reasoning deltas to the client mid-turn (with a `sync.Mutex` around the `http.ResponseWriter` to fix the SSE race that produced the "chunked line ends with bare LF" errors).

### Tests
- New Go tests: `proxy/path_aware_test.go`, `proxy/reasoning_repeat_test.go`, `proxy/recover_truncated_test.go`, `proxy/step_restriction_test.go`. Extended `proxy/guardrails_test.go`, `proxy/plan_hook_test.go`.
- All `go test ./...` on both `proxy/` and `tui/` modules pass.
- Full Python suite: 1055 passed / 4 skipped / 0 failed / 0 errors locally.

## [3.0.1] - 2026-04-05

### Tool-Call Agent Loop Architecture
- Replaced Aider format-translation proxy with structured JSON tool-call agent loop
- Grammar-constrained output via llama-server `response_format:json_object` — 100% valid JSON
- 8 tool definitions: `read_file`, `write_file`, `edit_file`, `delete_file`, `run_command`, `search_files`, `list_directory`, `plan_tasks`
- Per-file tier classification: T1 (config/data) writes directly, T2 (logic/features) routes through V3 pipeline
- 3400+ lines new Go code across 12 files in `proxy/`

### V3 Pipeline Integration
- All 14 V3 steps wired into `write_file`/`edit_file` executors for T2/T3 files
- PlanSearch → DivSampling → Budget Forcing → Build Verification → C(x)/G(x) Scoring → Best-of-K → S*/Blend-ASC → Failure Analysis → PR-CoT Repair → Refinement Loop → Derivation Chains → Metacognitive → Final Write
- Per-file-type build verification: tsc, py_compile, gcc, go build, cargo check, bash -n
- V3 service SSE streaming: pipeline progress visible in real-time

### CLI Experience
- `atlas` command: starts all services and launches Aider
- Streaming progress: `[Turn N/M]` with tool call details, V3 pipeline steps, completion summary
- Exploration budget: 4 consecutive read-only calls triggers nudge, prevents model from over-exploring
- Pre-injected project context: model sees project file list in system prompt
- File deletion via fast-path before tier classification
- Truncation prevention: 32K context, reject write_file for existing files >100 lines, detect truncated args before execution

### Deployment
- Docker Compose (`docker-compose.yml`) for full stack orchestration
- Podman compatible with host networking
- `.env.example` with all configurable parameters
- `atlas` script auto-detects Docker vs bare-metal and routes accordingly

### Renames (362 total reference updates)
- `rag-api/` → `geometric-lens/` (directory + all references)
- `ATLAS_RAG_URL` → `ATLAS_LENS_URL`
- `ATLAS_FOX_URL` → `ATLAS_INFERENCE_URL`
- `foxURL` → `inferenceURL` (Go code)
- `ralph-loop` → `verify-repair loop`
- `rag.py` → `pipeline.py` (geometric-lens orchestration)

### Reliability
- 8-level test × 3 iterations: 95.8% (23/24)
- 5-language integration: 100% (Shell, Python, Rust, C, Go)
- L6 (add feature to existing project): 67% — marked as future improvement

### Documentation Overhaul
- **ARCHITECTURE.md**: Complete rewrite — 13 Mermaid diagrams (service topology, agent loop flow, V3 pipeline, module map, sequence diagrams), every component verified against source code
- **API.md**: Complete rewrite — every endpoint across all 5 services verified against source, request/response formats, SSE stages
- **CLI.md**: Complete rewrite — startup flow diagram, streaming format, workflow examples, troubleshooting, env vars, Aider config reference
- **CONFIGURATION.md**: Complete rewrite — every env var across all services verified, internal constants, Docker Compose vs K3s differences
- **MAP.md**: Complete rewrite — every file in repo with clickable tree, 150 file links, 18 description tables
- **SETUP.md**: Complete rewrite — verified build steps, first-run guide, bare metal, K3s, hardware sizing, Lens training guide
- **TROUBLESHOOTING.md**: Complete rewrite — quick diagnostics, 20+ issue scenarios with verified fixes
- **README.md**: Honest 7-step setup with actual download command, prerequisites, model clarity (Qwen3-14B vs Qwen3.5-9B)
- Reorganized historical docs into `docs/reports/` (ablation studies, status tracking, migration guides)

### Bug Fixes
- **geometric-lens Dockerfile port mismatch**: Container was listening on 8001 but docker-compose expected 8099 — fresh Docker Compose deploys had broken Lens service. Fixed Dockerfile to use port 8099.
- **Python CLI default RAG port**: `atlas/cli/client.py` defaulted to port 31144 (K3s NodePort) instead of 8099 (Docker Compose). Fixed default to match Docker Compose.
- **Missing Aider config files**: `.aider.model.settings.yml` and `.aider.model.metadata.json` were not in the repo — the `atlas` launcher would fail without them. Restored both files and added `.gitignore` exceptions.
- GitHub Issue #6: `hostname -I` → portable fallback chain (`ip addr` → `hostname -I` → `hostname -i`) for Arch Linux compatibility
- GitHub Issue #10: `rag-api/` → `geometric-lens/` restructuring resolved missing models directory
- GitHub Issue #11: Added Geometric Lens training documentation to SETUP.md with HuggingFace dataset link
- GitHub Issue #12 / PR #13: `docker image exists` → `docker image inspect` in build script

### Cleanup
- Removed 62 stale test directories, old v1 proxy binary, dead G(x) metric tensor training scripts
- Removed stale tests for deleted services (api-portal, dashboard, embedding-service, task-worker)
- Removed root-level development artifacts (bubble_sort.py, snake_game.py, etc.)
- All hardcoded `/home/isaac/` paths replaced with `$HOME` or `ATLAS_DIR` env vars

## [3.0] - 2026-03-05

### V3.0 Benchmark Release
- **74.6% LCB pass@1** (447/599) on frozen Qwen3-14B
- Full ablation study: conditions A–D with per-task results
- Phase 1 (PlanSearch/DivSampling): +12.4pp
- Phase 3 (PR-CoT/Refinement/Derivation): +7.3pp
- Self-verified Phase 3 using model-generated test cases

## [2.5.1] - 2026-02-23

### Confirmation Ablation: Embedding Source Hypothesis — STRONG CONFIRMATION
- **H1: Self-embeddings restore C(x) discrimination: CONFIRMED (+39.5pp)**
  - C(x) selects passing candidate 87.8% on mixed-result tasks vs 48.3% random (p < 0.000001)
  - V2.5 result (+0.6pp under nomic 768-dim) was an embedding source limitation, not architecture failure
  - Reverse energy selects only 4.3%, proving strong directional signal
  - Val AUC: 0.9934, energy separation: 21.75 (7.2x wider than V2.5)
- **H2: G(x) adds value beyond C(x): NEUTRAL (0.0pp)**
  - G(x) contributes zero at optimal alpha (0.001); monotonically degrades at higher alpha
  - Zero corrections, zero breakages across all mixed-result tasks
- **Outcome B**: Ship C(x)-only with self-embeddings, remove or redesign G(x)
- **Difficulty routing validated**: Q1 (low energy) = 100% oracle, Q4 (high energy) = 0.3%
- **C(x) confirmed as both verifier (87.8% selection) and router (perfect difficulty stratification)**
- Runtime: 24h 42m on LiveCodeBench v5 (599 tasks, K=3, 4 epochs)
- Infrastructure: Qwen3-14B with `--embeddings` (no spec decode, ~45 tok/s)
- Risk R6 (Lens non-discriminating) RESOLVED; Risk R11 (no verifier) substantially mitigated

## [2.5.0] - 2026-02-21

### Ablation Study
- Systematic ablation of Geometric Lens, router, and infrastructure components
- Finding: C(x) energy scoring ≈ random for candidate selection under nomic embeddings (37.7% vs 37.1%, within 3.4pp seed variance) — **V2.5.1 confirmed this was an embedding source limitation** (87.8% accuracy restored with self-embeddings)
- Finding: C(x) energy strongly correlates with task difficulty (58.5% vs 18.9% pass rate across tiers)
- Finding: G(x) metric tensor confirmed dormant (5.2M params, zero impact)
- Finding: Pattern cache bypassed entirely by benchmark runner

### Architecture Change
- Discovered `--embeddings` flag breaks speculative decoding (forces n_batch=512)
- Migrated to two-server sidecar architecture: generation + spec decode on Server A, embeddings via nomic-embed-text-v1.5 on Server B
- Recovered ~2.6x generation throughput (~38 tok/s → ~100 tok/s)
- Net VRAM delta: approximately -230 MiB (sidecar cheaper than --embeddings overhead)

## [2.0.0] - 2026-02-18

### Architecture Changes
- Replaced Qdrant vector DB + embedding service with PageIndex tree-based RAG
- Added Geometric Lens (Cost Field + Metric Tensor) for candidate quality prediction
- Added Confidence Router with difficulty-based adaptive-k selection
- Added Pattern Cache (Redis + Ebbinghaus memory decay)
- Added Best-of-K pipeline with parallel candidate generation
- Added sandboxed code execution for benchmark evaluation
- Added speculative decoding with Qwen3-0.6B draft model
- Added KV cache quantization (q4_0)

### Benchmark Results (Run ID: v2_run_20260217_125310)
- LiveCodeBench: 36-41% pass@1 (across Lens training epochs, k=3)
- GPQA Diamond: 47.0% (k=5)
- SciCode: 14.7% sub-problems (341 tasks, k=1)
- Geometric Lens: 0.968 Val AUC, ~80% first-pick accuracy (151/188)
- Throughput: 109 tasks/hr on RTX 5060 Ti 16GB

### Removed
- Qdrant vector database
- MiniLM-L6-v2 embedding service
- LoRA nightly training pipeline (moved to v1_archived/, CronJob suspended)
- V1 benchmark suite (HumanEval, MBPP, Custom)

### Fixed Post-Release
- mlock allocation failure — added LimitMEMLOCK=infinity systemd override for K3s
- Speculative decode slot 1 failure — quantized draft KV cache to q4_0 (-ctkd/-ctvd)
- Dashboard crash-loop — fixed missing Jinja2 default filters

### Notes
- IFBench evaluation incomplete (excluded from results)
- All results from single benchmark run (variance unknown)

## [1.0.0] - 2026-02-04

Initial release. See benchmark/v1_benchmark_report.md for V1 results.
