# GPQA Diamond — Methodology

## Benchmark description

GPQA (Graduate-Level Google-Proof QA) Diamond is the hardest subset of the
GPQA benchmark, containing 198 graduate-level multiple choice questions in
biology, physics, and chemistry. Questions are designed to be "Google-proof" —
experts can answer them but non-experts struggle even with internet access.

Source: Rein et al., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark" (2023)

## Dataset source

Downloaded from OpenAI's public blob storage:
`https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv`

This is the same source used by the OpenAI Simple Evals framework.

## Evaluation protocol

1. Each question is presented with 4 answer options (A/B/C/D)
2. Options are shuffled deterministically per question (MD5-seeded Fisher-Yates)
3. The model receives a system prompt asking it to reason and end with "Answer: X"
4. Answer extraction uses a 9-pattern regex cascade (last match wins)
5. Accuracy = correct / total
6. 95% CI computed via bootstrap resampling (n=1000, seed=42)

## Prompt format

```
Answer the following multiple choice question. The last line of your response
should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
```

## Differences from Qwen baseline

| Aspect | Qwen baseline | ATLAS V3.0.1 |
|--------|---------------|--------------|
| Precision | bf16 | AWQ-Q4 |
| Temperature | 1.0 | 0.6 |
| Top-P | 0.95 | 0.95 |
| Top-K | 20 | (not set, server default) |
| Presence penalty | 1.5 | 0.0 |
| Thinking mode | enabled by default | model decides |
| Pipeline | direct model | ATLAS pipeline (MCQ bypass) |

## Scoring function

Simple exact match on extracted letter vs. ground truth letter.
Extraction failures (no valid A/B/C/D found) count as incorrect.
