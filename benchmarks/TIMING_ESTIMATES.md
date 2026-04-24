# ATLAS V3.0.1 Benchmark Timing Estimates

**Based on observed throughput: ~10 tok/s for thinking mode, ~48 tok/s for nothink**
**Hardware: RTX 5060 Ti 16GB, Qwen3.5-9B-Q6_K, --parallel 1**

## Per-question timing

| Mode | Tokens/q | Time/q | Notes |
|------|----------|--------|-------|
| MCQ thinking | 3000-8000 | 5-13 min | GPQA observed: 411s for 3549 tok |
| MCQ nothink | 200-600 | 4-12 s | Estimated from earlier tests |
| IF thinking | 500-2000 | 1-3 min | IFEval/IFBench estimated |
| IF nothink | 300-1000 | 6-20 s | IFBench with /completion |

## Total wall-clock estimates

### Wave 1: Instruction Following
| Benchmark | Tasks | Mode | Est. time |
|-----------|-------|------|-----------|
| IFEval | 541 | thinking | 9-27 hrs |
| IFBench | 300 | nothink | 0.5-1.5 hrs |
| MultiChallenge | ~TBD | multi-turn | TBD |

### Wave 1: Reasoning & Coding (via V3 pipeline)
| Benchmark | Tasks | Mode | Est. time |
|-----------|-------|------|-----------|
| LCB v6 | ~600 | V3 pipeline | 10-16 hrs (from plan) |
| OJBench | ~TBD | CLI | TBD |
| HMMT Feb/Nov | ~TBD | CLI | TBD |

### Wave 2: Knowledge & STEM
| Benchmark | Tasks | Mode | Est. time |
|-----------|-------|------|-----------|
| GPQA Diamond | 198 | thinking | 16-43 hrs |
| MMLU-Pro | 12,032 | nothink | 13-40 hrs |
| MMLU-Redux | ~TBD | nothink | TBD |
| C-Eval | ~TBD | nothink | TBD |
| SuperGPQA | ~TBD | nothink | TBD |

### Key constraints
1. Only 1 parallel slot — benchmarks run sequentially
2. Thinking mode = 30x slower but may be needed for accuracy
3. GPU is fully utilized during benchmark runs
4. Total suite estimated: 200-500+ hours (2-4 weeks of 24/7 running)

## Optimization opportunities
1. Run easy benchmarks (IFBench, nothink MCQs) first for quick results
2. Run GPQA Diamond with thinking (critical, hard benchmark, 198q only)
3. Queue long runs (MMLU-Pro, IFEval) for overnight
4. Multi-turn benchmarks (MultiChallenge) need separate implementation
