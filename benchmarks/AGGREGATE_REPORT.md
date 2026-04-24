# ATLAS V3.0.1 Full Benchmark Suite — Aggregate Report

**Model:** Qwen3.5-9B-Q6_K (Q6_K quantization)
**Pipeline:** ATLAS V3.0.1
**Hardware:** NVIDIA GeForce RTX 5060 Ti 16GB, K3s cluster
**Date range:** 2026-04-14 — (ongoing)

## Thinking mode methodology

**All results use natural thinking via `--jinja`, matching Qwen's methodology.**

- Thinking-heavy benchmarks (IFBench, IFEval, GPQA, LCB): **8192 token budget** (matches Qwen)
- MCQ benchmarks (MMLU-Pro, C-Eval): **2048 token budget** (capped for time budget)
- No `--reasoning off`, no `--reasoning-budget N` (both produce garbage output)

Both baseline (raw model) and ATLAS pipeline runs use identical thinking settings.
The **delta between baseline and ATLAS pipeline** measures what the V3 pipeline adds.

If any benchmark scores >15pp below Qwen baseline, we investigate settings and
adjust until we find a working configuration, then report on working settings.

See `METHODOLOGY_THINKING.md` for full details.

## Methodology deltas

| Aspect | Qwen3.5-9B baseline | ATLAS V3.0.1 |
|--------|---------------------|--------------|
| Precision | bf16 | Q6_K |
| Temperature | 1.0 (recommended) | 0.6 |
| Top-P | 0.95 | 0.95 |
| Top-K | 20 | server default |
| Presence penalty | 1.5 | 0.0 |
| Context length | up to 262K | 32K (VRAM-limited) |
| Thinking mode | enabled by default | model decides |
| Pipeline | direct model | ATLAS V3.0.1 (varies by benchmark) |

## Results

| Section | Benchmark | Qwen3.5-9B (baseline) | ATLAS V3.0.1 (9B Q6_K) | Delta | 95% CI |
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

1. **Quantization gap:** Qwen baseline uses full bf16; ATLAS uses Q6_K.
   Q6_K retains ~99.5% of quality but may affect some benchmarks.

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
