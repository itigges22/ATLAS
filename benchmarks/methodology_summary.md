# ATLAS V3.0.1 Benchmark Suite — Common Methodology

## System under test

**ATLAS V3.0.1** running on **Qwen3.5-9B-AWQ** (AWQ-Q4 quantization of the
Qwen3.5-9B model from QuantTrio). The model runs on a single NVIDIA GeForce
RTX 5060 Ti with 16GB VRAM via vLLM in a K3s cluster (or via docker-compose
for local development).

## Execution principles

1. **Production simulation**: Every benchmark that can be run through the
   ATLAS CLI / Aider is run that way, because that tests the product. 
   Raw-pipeline mode is used only for MCQ benchmarks where CLI behavior
   would corrupt the evaluation.

2. **Audit-grade artifacts**: Every benchmark produces config.yaml,
   methodology.md, responses.jsonl, traces/, results.json, REPORT.md,
   and sample_questions.jsonl.

3. **Reproducibility**: seed=42 for all random operations, model SHA256
   recorded, sampling parameters documented.

4. **Crash recovery**: Responses written incrementally via append-to-JSONL.
   Runner detects previously completed tasks and resumes.

## Three execution modes

| Mode | Description | Used for |
|------|-------------|----------|
| **Raw pipeline** | Direct model inference via `/v1/chat/completions` | MCQ, short-answer benchmarks |
| **CLI** | Through ATLAS Go CLI agent (tool calling, file I/O, iteration) | Code, agent, instruction-following tasks |
| **Aider** | Through Aider on ATLAS | Repo-level code editing (future) |

## Quantization disclosure

The Qwen3.5-9B model card baselines use full bf16 precision. ATLAS uses
AWQ-Q4 quantization (`QuantTrio/Qwen3.5-9B-AWQ` from HuggingFace), which
quantizes weights to 4-bit using the activation-aware weight quantization
scheme. AWQ-Q4 is more aggressive than Q6_K (the prior llama.cpp quant) but
preserves accuracy well on instruction-following and reasoning tasks; it
may have a measurable quality impact on knowledge-heavy benchmarks.

## Sampling parameter disclosure

Qwen recommends for thinking-mode tasks:
- temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5

ATLAS V3.0.1 uses:
- temperature=0.6, top_p=0.95 (pipeline-optimized defaults)

This divergence is intentional — the ATLAS pipeline is tuned for its own
sampling parameters. Direct parameter matching would undermine the pipeline.

## Thinking mode

Qwen3.5-9B uses a thinking mode where the model generates `reasoning_content`
(internal chain-of-thought) before the final `content` (user-visible answer).
Both consume the `max_tokens` budget. For MCQ benchmarks, `max_tokens=8192`
is needed to accommodate both thinking (~2000-5000 tokens) and the answer.

## Statistical reporting

- All accuracy metrics reported with 95% confidence intervals
- CIs computed via bootstrap resampling (n=1000, seed=42)
- Bootstrap uses random.Random with deterministic seed for reproducibility

## Caveats included in every report

1. Quantization gap (AWQ-Q4 vs bf16)
2. Sampling parameter divergence
3. Thinking mode handling
4. Pipeline components enabled/disabled
5. Context length limitations (32K vs 262K native)
