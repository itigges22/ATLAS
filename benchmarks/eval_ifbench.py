#!/usr/bin/env python3
"""Re-evaluate IFBench responses using the official Allen AI evaluator.

Usage:
    python benchmarks/eval_ifbench.py

Reads responses from benchmarks/section_b_instruction_following/ifbench/responses.jsonl
and evaluates using the official 58-constraint IFBench evaluation library.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.eval_libs.ifbench import evaluation_lib as ifbench_eval

PROMPTS = "benchmark/datasets/.cache/ifbench.jsonl"
RESPONSES = "benchmarks/section_b_instruction_following/ifbench/responses.jsonl"

# Load prompts
inputs = ifbench_eval.read_prompt_list(PROMPTS)
print(f"Loaded {len(inputs)} prompts")

# Load responses
prompt_to_response = {}
with open(RESPONSES) as f:
    for line in f:
        rec = json.loads(line)
        if rec.get('prompt') and 'response' in rec:
            prompt_to_response[rec['prompt']] = rec['response']
print(f"Loaded {len(prompt_to_response)} responses")

# Evaluate
strict_outputs = []
loose_outputs = []
for inp in inputs:
    if inp.prompt not in prompt_to_response:
        continue
    strict_out = ifbench_eval.test_instruction_following_strict(inp, prompt_to_response)
    loose_out = ifbench_eval.test_instruction_following_loose(inp, prompt_to_response)
    strict_outputs.append(strict_out)
    loose_outputs.append(loose_out)

n = len(strict_outputs)
strict_pass = sum(1 for o in strict_outputs if o.follow_all_instructions)
loose_pass = sum(1 for o in loose_outputs if o.follow_all_instructions)

strict_inst_total = sum(len(o.instruction_id_list) for o in strict_outputs)
strict_inst_correct = sum(sum(o.follow_instruction_list) for o in strict_outputs)
loose_inst_total = sum(len(o.instruction_id_list) for o in loose_outputs)
loose_inst_correct = sum(sum(o.follow_instruction_list) for o in loose_outputs)

print(f"\n{'='*50}")
print(f"IFBench Results ({n} evaluated / 300 total)")
print(f"{'='*50}")
print(f"Strict prompt-level: {strict_pass}/{n} = {strict_pass/n*100:.1f}%")
print(f"Strict inst-level:   {strict_inst_correct}/{strict_inst_total} = {strict_inst_correct/strict_inst_total*100:.1f}%")
print(f"Loose prompt-level:  {loose_pass}/{n} = {loose_pass/n*100:.1f}%")
print(f"Loose inst-level:    {loose_inst_correct}/{loose_inst_total} = {loose_inst_correct/loose_inst_total*100:.1f}%")
print(f"Baseline (Qwen):     64.5%")
print(f"{'='*50}")
