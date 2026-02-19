# ATLAS Benchmark Suite

## V2 Benchmark

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
| SciCode | ~80 | 14.7% sub / 5.0% main | k=1 |

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
| `best_of_k.py` | Best-of-K selection with Geometric Lens |
| `v2_runner.py` | V2 benchmark orchestrator |
| `v2_report.py` | V2 result analysis and reporting |
| `geo_learning.py` | Geometric Lens retraining pipeline |
| `runner.py` | Base benchmark runner |
| `cli.py` | Command-line interface |
| `config.py` | Benchmark configuration |
| `run_v2_benchmark.sh` | V2 benchmark launch script with pre-flight checks |
| `datasets/` | Dataset loaders (LiveCodeBench, GPQA, IFBench, SciCode, Custom) |

## V1 Benchmark (Archived)

V1 results are in `v1_benchmark_report.md`. V1 is tagged as `v1.0.0`.
