# IFEval — Methodology

## Benchmark description

IFEval (Instruction-Following Evaluation) is a benchmark containing 541 prompts
with verifiable instructions. Each prompt contains one or more constraints that
can be programmatically verified, such as "write in more than 400 words" or
"mention the keyword 'AI' at least 3 times."

The benchmark identified 25 types of verifiable instructions organized into
8 categories: keywords, language, length constraints, detectable content,
detectable format, combination, start/end, change case, and punctuation.

Source: Zhou et al., "Instruction-Following Evaluation for Large Language Models"
(arXiv:2311.07911, 2023)

## Dataset source

Downloaded from HuggingFace: `google/IFEval` (Apache 2.0 license)
541 prompts in the `train` split.

## Evaluation protocol

1. Each prompt is sent to the model with system prompt: "Follow the instructions
   exactly as given."
2. The model's response is evaluated against all instructions in the prompt
3. Two evaluation modes:
   - **Strict**: Response evaluated as-is
   - **Loose**: 8 variations tested (removing first/last lines, stripping asterisks)
4. Two accuracy levels:
   - **Prompt-level**: percentage of prompts where ALL instructions are followed
   - **Instruction-level**: percentage of individual instructions followed
5. Primary metric (reported by Qwen): **strict prompt-level accuracy**
6. 95% CI computed via bootstrap resampling (n=1000, seed=42)

## Evaluation library

Uses the official Google Research IFEval evaluation library from:
`github.com/google-research/google-research/tree/master/instruction_following_eval`

Files: instructions.py, instructions_registry.py, instructions_util.py,
evaluation_lib.py — downloaded and adapted for local import.

## 25 Instruction types

| Category | Instruction ID | Description |
|----------|---------------|-------------|
| keywords | keywords:existence | Check keyword presence |
| keywords | keywords:frequency | Check keyword occurrence count |
| keywords | keywords:forbidden_words | Ensure words are absent |
| keywords | keywords:letter_frequency | Check letter occurrence count |
| language | language:response_language | Validate response language |
| length | length_constraints:number_sentences | Check sentence count |
| length | length_constraints:number_paragraphs | Check paragraph count |
| length | length_constraints:number_words | Check word count |
| length | length_constraints:nth_paragraph_first_word | Check nth paragraph first word |
| content | detectable_content:number_placeholders | Check bracket placeholders |
| content | detectable_content:postscript | Validate postscript exists |
| format | detectable_format:number_bullet_lists | Check bullet count |
| format | detectable_format:constrained_response | Match predefined options |
| format | detectable_format:number_highlighted_sections | Count highlighted sections |
| format | detectable_format:multiple_sections | Validate section count |
| format | detectable_format:json_format | Validate JSON output |
| format | detectable_format:title | Check for title in <<>> |
| combination | combination:two_responses | Validate dual responses |
| combination | combination:repeat_prompt | Check prompt repetition |
| startend | startend:end_checker | Validate ending phrase |
| startend | startend:quotation | Check quotation wrapping |
| case | change_case:capital_word_frequency | Check capitalized words |
| case | change_case:english_capital | Validate all-caps |
| case | change_case:english_lowercase | Validate all-lowercase |
| punctuation | punctuation:no_comma | Ensure no commas |

## Differences from Qwen baseline

| Aspect | Qwen baseline | ATLAS V3.0.1 |
|--------|---------------|--------------|
| Precision | bf16 | AWQ-Q4 |
| Temperature | 1.0 | 0.6 |
| System prompt | unknown | "Follow the instructions exactly as given." |
| Thinking mode | enabled | model decides (temperature-dependent) |
