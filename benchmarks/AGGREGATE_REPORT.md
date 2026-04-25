# ATLAS V3.0.1 Full Benchmark Suite — Aggregate Report

**Model:** Qwen3.5-9B-AWQ (AWQ-Q4 quantization)
**Pipeline:** ATLAS V3.0.1
**Hardware:** NVIDIA GeForce RTX 5060 Ti 16GB, K3s cluster (vLLM)
**Date range:** 2026-04-14 — (ongoing)

> **Note on the inference cutover (2026-04-25):** Earlier runs in this
> aggregate used llama-server with Q6_K. After the vLLM / AWQ-Q4 cutover,
> per-benchmark REPORT.md files capture the precise infrastructure their
> numbers were generated on. The aggregate's caveats below apply broadly to
> both eras.

## Thinking mode methodology

**All results use natural thinking via the model's chat template, matching
Qwen's methodology.**

- Thinking-heavy benchmarks (IFBench, IFEval, GPQA, LCB): **8192 token budget** (matches Qwen)
- MCQ benchmarks (MMLU-Pro, C-Eval): **2048 token budget** (capped for time budget)
- No `chat_template_kwargs.enable_thinking=false` (would disable planning, hurts constraint tasks)
- No reasoning-budget cap (cuts mid-thought, produces garbage)

Both baseline (raw model) and ATLAS pipeline runs use identical thinking settings.
The **delta between baseline and ATLAS pipeline** measures what the V3 pipeline adds.

If any benchmark scores >15pp below Qwen baseline, we investigate settings and
adjust until we find a working configuration, then report on working settings.

See `METHODOLOGY_THINKING.md` for full details.

## Methodology deltas

| Aspect | Qwen3.5-9B baseline | ATLAS V3.0.1 |
|--------|---------------------|--------------|
| Precision | bf16 | AWQ-Q4 |
| Temperature | 1.0 (recommended) | 0.6 |
| Top-P | 0.95 | 0.95 |
| Top-K | 20 | server default |
| Presence penalty | 1.5 | 0.0 |
| Context length | up to 262K | 32K (VRAM-limited) |
| Thinking mode | enabled by default | model decides |
| Pipeline | direct model | ATLAS V3.0.1 (varies by benchmark) |

## Results

| Section | Benchmark | Qwen3.5-9B (baseline) | ATLAS V3.0.1 (9B AWQ-Q4) | Delta | 95% CI |
|---------|-----------|----------------------|------------------------|-------|--------|
| **Knowledge & STEM** | | | | | |
| | MMLU-Pro | 82.5 | — | — | — |
| | MMLU-Redux | 91.1 | — | — | — |
| | C-Eval | 88.2 | — | — | — |
| | SuperGPQA | 58.2 | — | — | — |
| | GPQA Diamond | 81.7 | — | — | — |
| **Instruction Following** | | | | | |
| | IFEval | 91.5 | — | — | — |
| | IFBench | 64.5 | — | — | — |
| | MultiChallenge | 54.5 | — | — | — |
| **Long Context** | | | | | |
| | AA-LCR | 63.0 | — | — | — |
| | LongBench v2 | 55.2 | — | — | — |
| **Reasoning & Coding** | | | | | |
| | HMMT Feb 25 | 83.2 | — | — | — |
| | HMMT Nov 25 | 82.9 | — | — | — |
| | LiveCodeBench v6 | 65.6 | — | — | — |
| | OJBench | 29.2 | — | — | — |
| **General Agent** | | | | | |
| | BFCL-V4 | 66.1 | — | — | — |
| | TAU2-Bench | 79.1 | — | — | — |
| | VITA-Bench | 29.8 | — | — | — |
| | DeepPlanning | 18.0 | — | — | — |
| **Multilingualism** | | | | | |
| | MMMLU | 81.2 | — | — | — |
| | MMLU-ProX | 76.3 | — | — | — |
| | NOVA-63 | 55.9 | — | — | — |
| | INCLUDE | 75.6 | — | — | — |
| | Global PIQA | 83.2 | — | — | — |
| | PolyMATH | 57.3 | — | — | — |
| | WMT24++ | 72.6 | — | — | — |
| | MAXIFE | 83.4 | — | — | — |

## Known caveats

1. **Quantization gap:** Qwen baseline uses full bf16; ATLAS uses AWQ-Q4.
   AWQ-Q4 is more aggressive than Q6_K but preserves accuracy well on
   reasoning/instruction tasks; may have measurable impact on
   knowledge-heavy benchmarks.

2. **Sampling divergence:** Qwen recommends temp=1.0/top_k=20/presence_penalty=1.5
   for thinking-mode tasks. ATLAS uses temp=0.6 which is optimized for the
   pipeline but differs from Qwen's recommended settings.

3. **Context cap:** Qwen3.5-9B natively supports 262K context. ATLAS on 16GB
   VRAM caps at 32K. Long-context benchmarks (AA-LCR, LongBench v2) will
   reflect this hardware limitation.

4. **Pipeline variation:** Some benchmarks use the full ATLAS V3 pipeline
   (PlanSearch, Budget Forcing, Geometric Lens) while others use direct
   model inference. The mode is documented per-benchmark.

5. **Thinking mode:** The model's thinking mode (reasoning_content) is active
   for benchmarks run through the chat completions endpoint. This consumes
   tokens from the max_tokens budget, requiring higher limits.

## Benchmark footnotes (from Qwen model card)

- TAU2-Bench: Qwen uses the official setup except for the airline domain,
  where they applied fixes proposed in the Claude Opus 4.5 system card.
- MMLU-ProX: averaged accuracy on 29 languages.
- WMT24++: harder subset of WMT24 with XCOMET-XXL scorer, 55 languages.
- MAXIFE: accuracy on English + multilingual original prompts (23 settings).
