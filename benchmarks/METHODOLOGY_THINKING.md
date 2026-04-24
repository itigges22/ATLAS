# Thinking Mode Methodology — V3.0.1 Benchmark Suite

## Background

Qwen3.5-9B uses a hybrid architecture with native reasoning via `<think>` tags.
Per the Qwen3.5 model card and blog, **all published baseline scores use thinking
mode enabled** with an 8192-token thinking budget.

## Our Approach

**We match Qwen's methodology: natural thinking with 8192 budget, no forced caps.**

### Server configuration

```bash
llama-server \
  --jinja \                  # Let the model's native template handle <think> tags
  --flash-attn on \
  -c 16384 -ctk q8_0 -ctv q4_0 \
  --parallel 1 --cont-batching \
  --embeddings
  # NO --reasoning off (prevents planning, hurts count/constraint tasks)
  # NO --reasoning-budget N (cuts mid-thought, produces garbage content)
```

With `--jinja`, the model's built-in template handles `<think>...</think>` blocks.
The server returns clean content (thinking stripped) via the chat completions endpoint.

### Runner configuration

- **Endpoint**: `/v1/chat/completions` (not `/completion`)
- **max_tokens**: 8192 (thinking-heavy benchmarks) or 4096 (MCQ with capped thinking)
- **Temperature**: 0.6 (Qwen's recommended thinking-mode setting)
- **Top-P**: 0.95, **Top-K**: 20 (Qwen's recommended)

## Why this matters

**Without thinking, the model cannot plan constraint satisfaction.**

Empirical test on IFBench task requiring specific keyword counts (1, 2, 3, 5, 7):
- **No thinking**: 0/5 keywords counted correctly — spammed `paradox` 21 times
- **Natural thinking**: 5/5 keywords exact match

This is why IFBench scored 36% without thinking vs 100% on first 2 tasks with thinking.
The 28pp gap to Qwen's 64.5% baseline was a methodology failure, not quantization.

## Thinking budget per benchmark

| Benchmark | Budget | Rationale |
|-----------|--------|-----------|
| IFBench | 8192 (full) | Matches Qwen; constraint satisfaction needs planning |
| IFEval | 8192 (full) | Matches Qwen; instruction following needs planning |
| GPQA Diamond | 8192 (full) | Matches Qwen; graduate-level science needs reasoning |
| LCB v6 | V3 pipeline tiers | Per-task adaptive (nothink/light/standard/hard/extreme) |
| MMLU-Pro 3K | 2048 (capped) | Time budget — MCQ letter answers need less reasoning |
| C-Eval | 2048 (capped) | Time budget — MCQ letter answers need less reasoning |

**Qwen uses 8192 for all benchmarks.** We cap MCQ at 2048 due to 15-day time budget.
This is documented as a methodology deviation. If MCQ scores miss baseline by >15pp,
we escalate to 8192 and re-run.

## Time cost

- **Thinking benchmarks** (8192 budget): ~12 minutes per task
  - IFBench 300 tasks → ~60 hours
  - GPQA 198 tasks → ~40 hours
- **MCQ benchmarks** (2048 cap): ~3 minutes per task average
- **Total suite**: ~13.5 days of GPU time (within 15-18 day budget)

## Fallback policy

If any benchmark scores >15pp below Qwen baseline:
1. Verify prompt format matches Qwen's eval (system prompt, formatting)
2. Try increasing thinking budget if capped
3. Try Qwen's exact sampling (temp=1.0 if currently 0.6)
4. If unable to reach baseline within time budget, disclose the gap

**Working settings are what we report on.** We don't manipulate settings per-task
to hit arbitrary targets — we use consistent config and disclose any settings we
adjusted to match Qwen.

## Baseline vs ATLAS comparison

Both runs use identical thinking settings. The delta between:
- **Baseline run**: Raw model with natural thinking
- **ATLAS run**: Same model + V3 pipeline (PlanSearch, candidate selection, verify-repair, Lens)

...is what the pipeline adds. Apples-to-apples comparison at equal thinking budget.

## Reference
- Qwen3.5 blog: https://qwenlm.github.io/blog/qwen3/
- Qwen3.5-9B model card: https://huggingface.co/Qwen/Qwen3.5-9B
- llama.cpp `--reasoning-budget` doc: https://github.com/ggml-org/llama.cpp/pull/20297
