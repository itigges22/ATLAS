# ATLAS Benchmark Suite

## V3.1 (Qwen3.5-9B) — In Progress

The V3 pipeline runner (`v3_runner.py` + the modules in `v3/`) targets
the V3.1.0 model+topology: Qwen3.5-9B-Q6_K with the PlanSearch /
DivSampling / Budget Forcing / PR-CoT / Refinement / Derivation stack.
Ablation work is tracked under `scripts/run_v31_ablation.sh`; the
6-condition study (A–F) is mid-run on hardware tier `medium`.
Headline 9B numbers are not yet published — `docs/SOURCES.md`
"Known Limitations" lists this as a V3.1.x roadmap item.

Until the 9B numbers land, the **V3 (14B) results in
[`docs/reports/V3_ABLATION_STUDY.md`](../docs/reports/V3_ABLATION_STUDY.md)**
are the canonical published evidence (74.6% LiveCodeBench v5 pass@1,
599 tasks, 4 ablation conditions).

---

## V2 Benchmark (Historical)

The V2 path runs against the older topology (`Fox 9B` / `Qwen3-14B`
with `--method best-of-K` selection). Preserved here for
reproducibility of the V2 results table below and the legacy
`runner.py` + `v2_runner.py` modules.

Run: `./run_v2_benchmark.sh`
Config: See `config.py` for parameters.

## What's Measured

- **LiveCodeBench v5** (primary): Coding problem-solving, 599 tasks, stdin/stdout evaluation
- **GPQA Diamond**: Knowledge reasoning, 198 multiple-choice questions
- **IFBench**: Instruction following, 300 tasks. **Note:** IFBench evaluation is incomplete -- evaluate_ifbench_loose() defaults to True for ~11/15 instruction categories. IFBench is excluded from headline results pending proper implementation.
- **SciCode**: Scientific coding, ~80 multi-step problems
- **Custom**: Real-world coding tasks, 100 problems from `benchmark/custom/tasks.json`

## V2 Results Summary

| Benchmark | Tasks | pass@1 | Conditions |
|-----------|-------|--------|------------|
| LiveCodeBench v5 | 599 | 36-41% | k=3, Geometric Lens selection |
| GPQA Diamond | 198 | 47.0% | k=5, MCQ |
| Custom | 100 | 53-55% | k=1 |
| SciCode | 341 | 14.7% sub-problems | k=1 |

Run ID: `v2_run_20260217_125310`
Hardware: RTX 5060 Ti 16GB VRAM
Throughput: 109 tasks/hr aggregate

All results from a single benchmark run. Not averaged across multiple runs; variance unknown.

## Reproducing Results

1. Ensure cluster is running: `kubectl get pods -n atlas`
2. Lock codebase: `git stash` any changes
3. Run: `./run_v2_benchmark.sh`
4. Results written to: `benchmark/results/`

## Files

| File | Purpose |
|------|---------|
| `runner.py` | Base benchmark runner (function + stdio modes, ChatML formatting, code extraction) |
| `cli.py` | Command-line interface (`atlas benchmark --humaneval --dry-run`, etc.) |
| `config.py` | Benchmark configuration loaded from `atlas.conf` |
| `models.py` | Data models: BenchmarkTask, AttemptResult, TaskResult, BenchmarkRun |
| `datasets/` | Dataset loaders (HumanEval, MBPP, EvalPlus, LiveCodeBench v5, GPQA, IFBench, SciCode) |
| `analysis/` | Cost analysis, hardware info, pass@k metric |
| `custom/` | 100 custom benchmark tasks + validator |
| **V3 pipeline (active)** | |
| `v3_runner.py` | V3 benchmark runner entry point (PlanSearch + DivSampling + PR-CoT, ablation conditions A–F) |
| `v3/` | 19 V3 pipeline modules (plan_search.py, div_sampling.py, budget_forcing.py, pr_cot.py, refinement_loop.py, etc.) |
| **V2 (historical)** | |
| `v2_runner.py` | V2 benchmark orchestrator (phases 0–6, telemetry, Mode A/B) |
| `v2_report.py` | V2 result analysis and reporting |
| `best_of_k.py` | Best-of-K selection with Geometric Lens (V2-era, superseded by V3 S* candidate selection) |
| `geo_learning.py` | Geometric Lens retraining pipeline |
| `run_v2_benchmark.sh` | V2 benchmark launch script with pre-flight checks |

## V1 Benchmark (Archived)

V1 is preserved in the git history under the `v1.0.0` tag — check it
out (`git checkout v1.0.0`) if you need the original runner or report
text. The V1 results file (`v1_benchmark_report.md`) was removed
during the V2 → V2.5 reorg and is no longer reachable from the V3.1.0
working tree.
