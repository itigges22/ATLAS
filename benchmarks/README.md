# ATLAS V3.0.1 Benchmark Suite

Benchmark evaluation of ATLAS V3.0.1 pipeline on Qwen3.5-9B-AWQ against
the Qwen3.5-9B model card baselines (26 benchmarks).

## Setup

```bash
pip install -r benchmarks/requirements.txt
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"
```

## Quick Start

```bash
# Check infrastructure status
./benchmarks/check_status.sh

# Run the full suite (IFBench → GPQA → IFEval → C-Eval → MMLU-Redux → MMLU-Pro → SuperGPQA)
cd /home/isaac/ATLAS
nohup benchmarks/run_suite.sh > benchmarks/logs/suite_$(date +%Y%m%d).log 2>&1 &

# Run a single benchmark
python -m benchmarks.v301_runner --benchmark gpqa_diamond
python -m benchmarks.v301_runner --benchmark ifbench
python -m benchmarks.v301_runner --benchmark mmlu_pro --limit 100  # test with 100 questions

# Run LiveCodeBench v6 through V3 pipeline
benchmarks/run_lcb_v6.sh --smoke  # 10-task smoke test
```

## Registered Benchmarks

| Benchmark | Mode | Questions | Baseline | Est. time |
|-----------|------|-----------|----------|-----------|
| gpqa_diamond | mcq (thinking) | 198 | 81.7 | ~22h |
| ifeval | ifeval | 541 | 91.5 | ~9h |
| ifbench | ifbench (nothink) | 300 | 64.5 | ~5h |
| mmlu_pro | mcq_nothink | 12,032 | 82.5 | ~50h |
| supergpqa | mcq_nothink | 26,529 | 58.2 | ~111h |
| mmlu_redux | mcq_nothink | 5,600 | 91.1 | ~23h |
| c_eval | mcq_nothink | 1,346 | 88.2 | ~6h |

## Key Files

- `v301_runner.py` — Main benchmark runner (4 execution modes)
- `run_suite.sh` — Sequential suite launcher
- `run_benchmark.sh` — Individual benchmark launcher
- `run_lcb_v6.sh` — LiveCodeBench v6 via V3 pipeline
- `check_status.sh` — Status checker for all 26 benchmarks
- `eval_libs/` — Google IFEval evaluation library
- `AGGREGATE_REPORT.md` — Results template
- `TIMING_ESTIMATES.md` — Per-benchmark timing
- `REMAINING_BENCHMARKS.md` — Implementation notes for unbuilt benchmarks

## Crash Recovery

All benchmarks support crash recovery. If a run is interrupted, restart the
same command — it will skip completed tasks and continue from where it left off.
