# Best-of-K Lens Selection Design

## Summary

Replace sequential retry-on-failure with generate-K-candidates + lens-guided selection. The Geometric Lens C(x) scores each candidate's energy; lowest energy = most likely correct. User sees one response (effective pass@1), internally it's best-of-K.

## Architecture

### Layer 1: rag-api Endpoint

New `POST /internal/lens/score-text` endpoint in `rag-api/main.py`:
- Input: `{"text": "TASK: ...\n\nSOLUTION: ..."}`
- Internally: extracts embedding via llama-server `/embedding`, evaluates C(x)
- Output: `{"energy": 3.14, "normalized": 0.12}`
- Single HTTP call per candidate (vs 2 with separate embed + score)

### Layer 2: Host-Side Module

New `benchmark/best_of_k.py` (stdlib only, no torch):
- `score_candidate(text, rag_api_url)` — calls `/internal/lens/score-text`, returns energy
- `select_best_candidates(candidates, rag_api_url)` — scores list, returns sorted by energy
- `BestOfKTracker` — telemetry class tracking:
  - Per-task: k generated, energies, selected index, sandbox calls, oracle pass@k
  - Aggregates: selection accuracy, avg sandbox calls, energy distributions
- `get_temperature(k)` — returns 0.0 for k<=1, 0.6 for k<=5, 0.8 for k<=10

### Layer 3: v2_runner.py Integration

New `_run_task_best_of_k()` method in V2BenchmarkRunner:
- For **code tasks** (function/stdio eval_mode): Lazy generation with early exit
  1. Generate candidate at temperature>0
  2. Score with lens (energy)
  3. If energy < threshold, try sandbox immediately
  4. If sandbox passes, stop (early exit)
  5. If not, generate next candidate
  6. After all K generated, try remaining untested candidates in energy order
- For **non-code tasks** (mcq/ifbench): Generate all K, return lowest-energy
- Records all candidates + energies in telemetry

### Epoch Integration

`run_lcb_learning_epochs()` updated:
- Epoch 0: Mode B, temp=0, k=1 (baseline, no best-of-K)
- Epochs 1+: Mode A, temp=0.6, k=5 (best-of-K with lens selection)
- Training embeddings collected from ALL candidates (not just selected one)
- Both PASS and FAIL labels recorded for lens retraining

### Temperature Strategy

| Route | k | Temperature | Strategy |
|-------|---|------------|----------|
| CACHE_HIT | 0 | N/A | Use cached solution |
| FAST_PATH | 1 | 0.0 | Single deterministic generation |
| STANDARD | 5 | 0.6 | Best-of-5 with lens selection |
| HARD_PATH | 10 | 0.8 | Best-of-10 with lens selection |

Mode B always uses temp=0, k=1 (baseline comparison).

## Report Section

Four new tables in v2_report.py:

1. **Selection Accuracy** — did lens pick a passing candidate?
2. **Effective pass@1 vs Oracle pass@k** — actual selection vs upper bound
3. **Sandbox Call Efficiency** — avg calls to find PASS with vs without ordering
4. **Energy Distribution** — selected vs rejected vs PASS vs FAIL means

## Validation Criteria 16-20

16. k=5 generates >= 3 unique code solutions
17. Energy std dev > 0.5 across k candidates per task
18. Selection accuracy > random (> 1/k for tasks with mixed pass/fail)
19. Avg sandbox calls to PASS < k/2
20. Early exit triggers (avg candidates generated < max_k)

## Files to Create/Modify

| File | Action |
|------|--------|
| `rag-api/main.py` | Add `/internal/lens/score-text` endpoint |
| `benchmark/best_of_k.py` | NEW: scoring, selection, telemetry tracker |
| `benchmark/v2_runner.py` | Add `_run_task_best_of_k()`, modify epoch runner |
| `benchmark/v2_report.py` | Add 4 new report tables |
