# V3.1 Benchmark Methodology

Scope: the in-progress V3.1 benchmark run comparing **Qwen3.5-9B-Q6_K** (baseline) against **Qwen3.5-9B-Q6_K + ATLAS V3 pipeline** across five benchmarks. This document records exactly how those runs are configured so the baseline-vs-ATLAS delta reflects only the pipeline's contribution, not a sampling or prompt drift.

V3 (14B) results and ablation live in [V3_ABLATION_STUDY.md](V3_ABLATION_STUDY.md) — this document is scoped to V3.1's 9B work and should not be read as a replacement for that report.

Last updated: 2026-04-21.

---

## 1. Reference point

Sampling and prompting settings below are Qwen3.5-9B's **published** benchmarking methodology, applied to V3.1 runs with forced deviations only where the 16 GB VRAM budget makes the published setting impossible (quantization, context window).

All sampling parameters are applied **identically** to the baseline runner (`benchmarks/v301_runner.py`) and the ATLAS V3 pipeline runner (`benchmark/v3_runner.py`).

## 2. Sampling (identical for baseline and ATLAS)

| Parameter | Value | Source |
|-----------|-------|--------|
| `temperature` | **1.0** | Qwen3.5-9B model card |
| `top_p` | **0.95** | Qwen3.5-9B model card |
| `top_k` | **20** | Qwen3.5-9B model card |
| `presence_penalty` | **1.5** | Qwen3.5-9B model card |
| thinking mode | enabled via `chat_template_kwargs.enable_thinking=True` | Qwen3.5-9B model card |
| seed | `42 + task_index` (deterministic per task) | ATLAS convention |

## 3. Model and hardware

| Aspect | V3.1 setting | Qwen published | Deviation reason |
|--------|--------------|----------------|------------------|
| Model | Qwen3.5-9B-Q6_K (GGUF) | Qwen3.5-9B bf16 | 16 GB VRAM can't fit bf16 |
| Inference engine | llama.cpp (custom patched) | Qwen's internal stack | open-source local stack |
| GPU | RTX 5060 Ti 16 GB | — | target hardware for ATLAS |
| Per-slot context window | 16 K tokens | up to 262 K | VRAM-bound |
| Server parallel slots | 4 (`--parallel 4 -c 65536`) | — | throughput |
| KV cache | `-ctk q8_0 -ctv q4_0` | — | VRAM-bound |

The Q6_K quantization and 16 K per-slot context are forced deviations. Both should cost a few pp of accuracy versus a bf16 run on unconstrained hardware — the published V3.1 numbers reflect that honestly rather than claim bf16 performance.

## 4. Per-benchmark configuration

| Benchmark | Tasks run | Prompt mode | `max_tokens` | Evaluator | Notes |
|-----------|-----------|-------------|--------------|-----------|-------|
| **C-Eval** | 1346 / 1346 | nothink (pre-filled `<think></think>`) | 1024 | ATLAS MCQ extractor (A–D) | Ran 2026-04-17, 8.8 h. Scored 80.01 %. Kept as-is under a documented methodology deviation: 4-option MCQ on short questions does not need thinking headroom to reach a correct answer. See §6 for why we did not re-run. |
| **MMLU-Pro** | 1000 / 12032 (seed=42 sample) | **thinking** | 12288 | ATLAS MCQ extractor (A–J) | Re-started 2026-04-19 under Qwen methodology after earlier attempts under a 1024-token cap produced ~59 % accuracy (100 % of extract-fails and 66 % of wrongs hit the cap mid-reasoning). The 1000-task seeded sample is identical between baseline and ATLAS runs — see §5. |
| **IFEval** | 541 / 541 | thinking | 8192 | Google IFEval library (strict + loose, prompt-level and instruction-level) | `benchmarks/eval_libs/evaluation_lib.py` patched to degrade gracefully on constraint/kwarg mismatches rather than crash. |
| **GPQA Diamond** | 198 / 198 | thinking | 12288 | ATLAS MCQ extractor (A–D) | Runs under full Qwen methodology. |
| **IFBench** | 300 / 300 | thinking | 8192 | Allen AI official IFBench library (58 constraint types) | Runs under full Qwen methodology. |

## 5. Baseline-vs-ATLAS determinism

Because the baseline and ATLAS runs need to be directly comparable:

1. **Same model weights** — same `.gguf` file, same quantization.
2. **Same sampling parameters** — table in §2, applied identically.
3. **Same thinking mode** — enabled via `chat_template_kwargs`.
4. **Same server config** — `--parallel 4 -c 65536 -ctk q8_0 -ctv q4_0 --flash-attn on --mlock --jinja`.
5. **Same MMLU-Pro sample** — the 1000-task seed=42 subset is drawn inside `benchmarks/v301_runner.py::load_mmlu_pro_tasks()`. Any code path that loads MMLU-Pro (baseline or V3) goes through that loader and gets the exact same subset.

What V3 adds is **orchestration**, not per-call parameter changes:
- PlanSearch (constraint extraction + structured planning)
- DivSampling (candidate diversity)
- Budget Forcing (tiered thinking budget allocation per phase)
- Geometric Lens C(x) energy + G(x) XGBoost quality for candidate ranking
- Phase 3 refinement (PR-CoT, derivation chains, metacognitive loop, ACE)

Every individual LLM call inside the V3 pipeline uses the same sampling as the baseline. The baseline→ATLAS delta therefore measures pipeline effects, not a sampling-parameter lift.

## 6. Known methodology deviations

Deliberate choices, documented so reviewers can weight them:

1. **C-Eval uses nothink, not thinking.** Scored 80.01 % vs Qwen's published 88.2 %. The 8.2 pp gap is consistent with Q6_K quantization loss plus the nothink deviation. Re-running under full Qwen methodology would cost ~3 more days of compute for a benchmark whose 4-option short-question format doesn't meaningfully benefit from thinking headroom. Kept as a documented deviation.
2. **MMLU-Pro is sampled to 1000 of 12032 tasks** with a seeded deterministic subset (seed=42, sorted by task ID). Full 12 K would take ~100 days on this hardware under Qwen methodology. 1000 tasks give a 95 % CI of roughly ±3 pp, which is enough to resolve the baseline vs ATLAS delta if it exists.
3. **Q6_K quantization** is forced by 16 GB VRAM. Expected to cost 1–3 pp on most benchmarks, potentially more on math-heavy categories.

## 7. Runtime knobs

### Runner concurrency
- Baseline (`benchmarks/v301_runner.py`): `BENCHMARK_PARALLEL=4` dispatches four tasks in flight via `ThreadPoolExecutor`, matching the four server slots.
- ATLAS pipeline (`benchmark/v3_runner.py`): `ATLAS_LLM_PARALLEL=1 ATLAS_PARALLEL_TASKS=4` dispatches four tasks concurrently, each running the full Phase 1/2/3 pipeline.

### Client-side timeouts
- Baseline `urllib` request timeout: **7200 s** (2 h). Sized for thinking-mode generation at ~2 tok/s per slot under parallel-4.
- V3 runner default timeout: **1800 s** (30 min) per LLM call. Individual V3 phases (PlanSearch, PR-CoT, Refinement, Derivation Chains) also set **1800 s** each, bumped up from an initial 300 s after diagnosing a silent-retry-loop bug (see §9).

### Server flags (K3s ConfigMap `llama-entrypoint`)
```
--parallel 4 -c 65536
-ctk q8_0 -ctv q4_0
--flash-attn on --mlock
-b 4096 -ub 4096
--ctx-checkpoints 0 --no-cache-prompt
--embeddings --jinja --no-warmup
```

## 8. Observed issues during the run

Documented here for reproducibility. None of these are being fixed mid-benchmark; they will be addressed in the V3.2 cycle.

### 8.1 MMLU-Pro baseline timeout rate (~11 % at the 100-task checkpoint, 2026-04-21)

Under Qwen methodology (thinking mode, `max_tokens=12288`, parallel-4), a long-tail of MMLU-Pro tasks generate longer than the 7200 s urllib client timeout. They get retried 3× by the runner and fail. Failed tasks are scored as wrong, which drags the headline accuracy down.

**Root cause.** Per-slot decode throughput under parallel-4 on the hybrid DeltaNet architecture averages ~1.2 tok/s. A single task that wants the full 12 288-token thinking budget takes ~2 h 50 min of wall-clock — past the 7200 s client timeout.

**Impact.** At the 100-task checkpoint: 11 tasks timed out, 68 correct / 21 wrong / nothing extracted. Headline accuracy: 68.0 %. **Answered-task accuracy (stripping timeouts): 76.4 %.**

**Why we did not intervene mid-run.** Changing the timeout or token cap mid-benchmark would make the published number unreproducible. The V3.1 baseline is meant to be a reference point, not a best-effort number. The ATLAS V3 pipeline addresses this failure class via Budget Forcing (tiered thinking allocation) and sandbox-based early exits, so the baseline-vs-ATLAS delta on MMLU-Pro is expected to capture a real pipeline win on top of any quality delta.

### 8.2 MCQ extractor defaulted to A–D for MMLU-Pro (10-option)

An earlier restart showed 0 correct in 2 completed tasks. Root cause: `extract_mcq_answer` defaulted `valid_options="ABCD"`, but MMLU-Pro has 10 options (A–J). A model that correctly wrote "Answer: H" was being scored as extract-fail. Fixed by adding `valid_options` to the registry and dispatch. Task IDs affected are not in the current 1000-sample run.

### 8.3 Silent retry loop in V3 pipeline (fixed)

Earlier smoke tests of the V3 pipeline on LCB v6 saw zero completed tasks for an hour. Server logs showed `srv stop: cancel task` followed by identical re-submissions. Root cause: `LLMAdapter` default timeout was 900 s plus four per-phase 300 s overrides, but 9B under thinking mode can run 14–23 min per call. Bumped default to 1800 s and per-phase overrides to 1800 s. No silent retries observed since.

### 8.4 IFEval evaluator kwarg mismatch

The Google IFEval library's `build_description` method rejected unknown kwargs (e.g. `num_highlights`) with a `TypeError` that crashed the entire evaluation. Patched `benchmarks/eval_libs/evaluation_lib.py` to catch `TypeError`, attempt a no-kwargs rebuild, and gracefully mark the instruction as failed rather than crash the run.

## 9. Timeline

- **2026-04-17** — V3.1 benchmark launch. C-Eval completes in 8.8 h at 80.01 %. MMLU-Pro started under an earlier 1024-token nothink configuration, scoring 59.4 % at the 1258-task checkpoint.
- **2026-04-18** — Identified the 1024-token cap as the cause of MMLU-Pro underperformance. Switched to full Qwen methodology (thinking mode, 12288 max tokens, temp=1.0, top_k=20, presence_penalty=1.5). Restarted MMLU-Pro.
- **2026-04-18** — Discovered the 10-option extractor bug via first-2-tasks inspection showing 0 correct despite model outputs being correct. Fixed and restarted.
- **2026-04-19** — Runtime projection under full Qwen methodology on the full 12 032 MMLU-Pro set was ~100 days. Sampled down to 1000 tasks (seed=42, deterministic) to fit a practical budget while preserving comparability between baseline and ATLAS.
- **2026-04-21** — 100-task checkpoint: 68 correct / 21 wrong / 11 timeouts. Headline accuracy 68.0 %, answered-task accuracy 76.4 %. Timeout issue documented here rather than fixed mid-run.

## 10. Open items

- **MMLU-Pro final numbers.** Publish both the headline score (including timeouts as wrong) and the answered-task score, clearly labeled, so the quality loss and the infrastructure loss are not conflated.
- **ATLAS V3 pipeline runs.** Queued to start when the baseline completes. Same 1000-task MMLU-Pro sample, same sampling parameters, different orchestration (V3 pipeline on top).
- **Final V3.1 report.** Separate document; this methodology doc is the reference it cites.
