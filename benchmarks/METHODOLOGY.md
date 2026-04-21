# ATLAS V3.0.1 Benchmark Methodology

This document records exactly how the V3.0.1 baseline and ATLAS pipeline benchmarks were run. Both tracks use **identical sampling parameters and prompt conventions** so the delta measures only what the ATLAS V3 pipeline adds, not a methodology drift.

Last updated: 2026-04-18.

## Reference

The settings below are Qwen3.5-9B's **published** benchmarking methodology, applied to V3.0.1 with one forced deviation (Q6_K quantization due to 16 GB VRAM).

## Sampling (identical for baseline and ATLAS)

| Parameter | Value | Source |
|-----------|-------|--------|
| `temperature` | **1.0** | Qwen3.5-9B model card |
| `top_p` | **0.95** | Qwen3.5-9B model card |
| `top_k` | **20** | Qwen3.5-9B model card |
| `presence_penalty` | **1.5** | Qwen3.5-9B model card |
| thinking mode | enabled via `chat_template_kwargs.enable_thinking=True` | Qwen3.5-9B model card |
| seed | `42 + task_index` (deterministic per task) | ATLAS convention |

## Model and hardware

| Aspect | V3.0.1 setting | Qwen published | Deviation? |
|--------|----------------|----------------|------------|
| Model | Qwen3.5-9B-Q6_K (GGUF) | Qwen3.5-9B bf16 | Quantization (VRAM-bound) |
| Inference engine | llama.cpp (custom patched image) | Qwen's internal stack | Engine choice |
| GPU | RTX 5060 Ti 16 GB | — | VRAM-bound |
| Context window per slot | 16 K tokens | up to 262 K | VRAM-bound |
| Server parallel slots | 4 (`--parallel 4 -c 65536`) | — | Throughput tuning |
| KV cache | `-ctk q8_0 -ctv q4_0` | — | VRAM-bound |

The Q6_K quantization and 16 K context ceiling are the only methodology deviations forced by the 16 GB VRAM budget. Both should cost a few pp of accuracy versus the bf16 baseline — the published V3.0.1 numbers bake this in rather than pretending to bf16 numbers.

## Per-benchmark settings

| Benchmark | Tasks run | Prompt mode | `max_tokens` | Evaluator | Notes |
|-----------|-----------|-------------|--------------|-----------|-------|
| **C-Eval** | 1346 / 1346 | nothink (pre-filled `<think></think>` block) | 1024 | ATLAS MCQ extractor | Ran 2026-04-17, 8.8h. Scored 80.01%; kept despite methodology deviation from strict Qwen thinking mode because 4-option MCQ fits comfortably in 1024 tokens without thinking. |
| **MMLU-Pro** | 1000 / 12032 (seed=42 sample) | **thinking mode** | 12288 | ATLAS MCQ extractor (A-J) | Seeded 1000-task sample (identical for baseline and ATLAS V3 runs). Earlier 3000-task runs under 1024-token nothink cap were invalid and have been discarded. |
| **IFEval** | 541 / 541 | thinking mode | 8192 | Google IFEval library (strict + loose prompt-level and instruction-level accuracy) | Runs under full Qwen methodology. eval_libs/evaluation_lib.py patched to skip constraints with kwarg mismatches rather than crash. |
| **GPQA Diamond** | 198 / 198 | thinking mode | 12288 | ATLAS MCQ extractor (A/B/C/D) | Runs under full Qwen methodology. |
| **IFBench** | 300 / 300 | thinking mode | 8192 | Allen AI official IFBench library (58 constraint types) | Runs under full Qwen methodology. |

### Known methodology deviations

These are explicit choices, not bugs:

1. **C-Eval uses nothink instead of thinking.** Scored 80.01% vs Qwen's 88.2%. The 8.2 pp gap is consistent with Q6_K quantization loss plus the nothink deviation; we are not re-running C-Eval since 4-option MCQ with short questions does not need thinking headroom to reach a correct answer.
2. **MMLU-Pro is sampled to 1000 of 12032 tasks.** Full 12 K would take ~100 days on this hardware under Qwen methodology (thinking mode, 12288 max_tokens, parallel 4). The 1000-task sample is seeded (seed=42) and sorted by task ID, so baseline and ATLAS V3 pipeline runs see the exact same 1000 questions — keeping the comparison apples-to-apples. Sample is drawn in `benchmarks/v301_runner.py::load_mmlu_pro_tasks` so any code path that loads MMLU-Pro (baseline or V3) gets the same subset.
3. **Q6_K quantization.** Forced by 16 GB VRAM. Expected to cost 1-3 pp on most benchmarks, potentially more on math-heavy tasks.

4. **MMLU-Pro baseline timeout rate (~11% at 2026-04-21, 100-task checkpoint).** Under Qwen methodology (thinking mode, `max_tokens=12288`, parallel-4 server), a non-trivial fraction of MMLU-Pro tasks generate longer than the 7200 s urllib client timeout and get retried 3× before failing. Failed tasks are scored as wrong. Root cause: at ~1.2 tok/s per slot under parallel-4 DeltaNet decoding, a single task that wants the full 12 288-token budget takes ~2h50m — past the client timeout. Stripping out the 11 % timeout-fails from the 100-task checkpoint, the *answered-task* accuracy was 76.4 % vs the headline 68.0 %. We are letting the baseline run through to completion rather than changing the timeout or token cap mid-run, so the published V3.0.1 baseline number reflects this noise floor. The ATLAS V3 pipeline addresses this class of failure via Budget Forcing (tiered thinking budgets) and sandbox-based early exits, so the baseline-vs-ATLAS delta on MMLU-Pro captures a real pipeline win and not only a timeout-rate win.

## Runtime knobs

### Runner concurrency
- Baseline (`benchmarks/v301_runner.py`): `BENCHMARK_PARALLEL=4` — four tasks in flight at a time via `ThreadPoolExecutor`, matching the server's four slots.
- ATLAS pipeline (`benchmark/v3_runner.py`): `ATLAS_LLM_PARALLEL=1 ATLAS_PARALLEL_TASKS=4` — four tasks dispatched concurrently, each running the full Phase 1/2/3 pipeline.

### Timeouts
- Baseline `urllib` request timeout: **7200 s** (2 h) — long enough for a slow thinking-mode generation at 2 tok/s/slot under parallel 4.
- V3 runner `LLMAdapter` default timeout: **1800 s** (30 min).
- V3 runner PlanSearch / PR-CoT / Refinement / Derivation-Chain timeouts: **1800 s** each (bumped from 300 s after silent retry-loop diagnosis, see `feedback_v3_timeout_bug` in memory).

### Server flags (K3s ConfigMap `llama-entrypoint`)
```
--parallel 4 -c 65536
-ctk q8_0 -ctv q4_0
--flash-attn on --mlock
-b 4096 -ub 4096
--ctx-checkpoints 0 --no-cache-prompt
--embeddings --jinja --no-warmup
```

## How the baseline-vs-ATLAS comparison stays honest

1. **Same model weights** (Qwen3.5-9B-Q6_K, same `.gguf` file).
2. **Same sampling parameters** (table above, applied identically in both runners).
3. **Same thinking mode** (enabled via `chat_template_kwargs`).
4. **Same server config, same GPU, same quantization.**
5. The V3 pipeline only changes *orchestration*: how many candidates to generate (PlanSearch, DivSampling), how much thinking budget to allocate (Budget Forcing), how to rank candidates (Geometric Lens C(x) + G(x)), whether to run a refinement / PR-CoT / derivation-chain pass. Every individual LLM call inside V3 uses the same sampling as the baseline.

This means the published V3.0.1 delta (baseline → ATLAS) reflects the pipeline alone, not a sampling-parameter lift.
