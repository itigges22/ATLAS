# Remaining Benchmarks — Implementation Notes

## Section B: Instruction Following

### MultiChallenge (baseline: 54.5)
- **Source**: ScaleAI/MultiChallenge on HuggingFace
- **Format**: Multi-turn conversation (4 challenges: instruction retention, inference memory, versioned editing, self-coherence)
- **Implementation needed**: Multi-turn conversation handler — send multiple messages, maintain conversation state
- **Evaluation**: Per-challenge scoring from the official eval code (github.com/ekwinox117/multi-challenge)
- **Difficulty**: Medium — need to implement multi-turn API calls

## Section C: Long Context

### AA-LCR (baseline: 63.0)
- **Source**: Artificial Analysis Long-Context Reasoning
- **Format**: Long documents + questions about them
- **Implementation needed**: Context length testing — need to reduce --parallel to fit larger contexts in VRAM
- **Config change**: Cap context at 64K-96K (32K may be too short)
- **Difficulty**: Medium — VRAM-constrained, need to experiment with context sizes
- **Expected penalty**: Scores will be lower due to context cap vs 262K native

### LongBench v2 (baseline: 55.2)
- **Source**: THUDM/LongBench on HuggingFace
- **Format**: Mixed long-doc QA, summarization, code
- **Same VRAM constraint** as AA-LCR

## Section D: Reasoning & Coding

### LiveCodeBench v6 (baseline: 65.6)
- **Source**: bzantium/livecodebench, config=release_v6 (1055 tasks)
- **Implementation**: READY — LiveCodeBenchV6Dataset class created, run_lcb_v6.sh script ready
- **Runs through V3 pipeline** (v3_runner.py) with sandbox code execution
- **Difficulty**: Low — just needs to be launched

### OJBench (baseline: 29.2)
- **Source**: Needs research — online judge style competitive programming
- **Format**: Competitive programming with stdin/stdout
- **Implementation needed**: Dataset loader + eval harness (similar to LCB)
- **Difficulty**: Medium

### HMMT Feb 25 / Nov 25 (baselines: 83.2 / 82.9)
- **Source**: Harvard-MIT Math Tournament 2025 problems
- **Format**: Math competition problems, usually open-ended numerical/proof answers
- **Implementation needed**: Dataset sourcing (may be on GitHub/AOPS), answer extraction for numerical answers
- **Difficulty**: Medium-High — math answer verification is non-trivial

## Section E: General Agent

### BFCL-V4 (baseline: 66.1)
- **Source**: berkeley-function-calling-leaderboard on HuggingFace
- **Format**: Function-calling benchmark — test tool use accuracy
- **Implementation needed**: Adapt ATLAS CLI tool-calling to BFCL format
- **Difficulty**: High — needs CLI agent integration

### TAU2-Bench (baseline: 79.1)
- **Source**: Needs research
- **Format**: Retail/airline customer service scenarios
- **Special**: Must apply Claude Opus 4.5 airline domain fix per Qwen model card
- **Difficulty**: High

### VITA-Bench (baseline: 29.8)
- **Format**: Agentic task completion
- **Difficulty**: High — full agent harness needed

### DeepPlanning (baseline: 18.0)
- **Format**: Multi-step planning, long-horizon
- **Difficulty**: High

## Section F: Multilingualism

### MMMLU (baseline: 81.2)
- Translated MMLU, ~14 languages. MCQ format. Similar to MMLU-Redux but multilingual.
- **Difficulty**: Low-Medium

### MMLU-ProX (baseline: 76.3)
- MMLU-Pro extended to 29 languages. MCQ format.
- **Difficulty**: Low-Medium

### NOVA-63 (baseline: 55.9)
- 63 languages. MCQ/short answer.
- **Difficulty**: Medium

### INCLUDE (baseline: 75.6)
- Regional/cultural knowledge. MCQ.
- **Difficulty**: Medium

### Global PIQA (baseline: 83.2)
- Physical commonsense, multilingual. MCQ.
- **Difficulty**: Low-Medium

### PolyMATH (baseline: 57.3)
- Math across multiple languages. CLI mode (write verification code).
- **Difficulty**: Medium

### WMT24++ (baseline: 72.6)
- Translation eval with XCOMET-XXL scorer (55 languages).
- **Special**: Needs XCOMET-XXL model for scoring — separate GPU inference or CPU (slow)
- **Difficulty**: High — external scorer dependency

### MAXIFE (baseline: 83.4)
- Multilingual instruction following, 23 settings.
- **Difficulty**: Medium

## Priority order for next sessions
1. LiveCodeBench v6 (READY, just launch)
2. MultiChallenge (multi-turn, moderate work)
3. MMMLU / MMLU-ProX / Global PIQA (MCQ, similar to existing)
4. HMMT (math, needs dataset sourcing)
5. Long context (AA-LCR, LongBench v2 — needs VRAM config)
6. Agent benchmarks (BFCL-V4, TAU2, VITA, Deep — highest eng cost)
7. WMT24++ (needs external scorer)
