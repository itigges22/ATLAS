# Sources & Research

Papers and research that directly informed the ATLAS architecture. Grouped by the ATLAS component they underpin. Entries without arXiv IDs are either preprints without ID, books, community work, or internal references.

If you're implementing an ATLAS subsystem, the paper(s) under that section's heading are the first reading list.

---

## Test-Time Compute (foundational)

The premise behind ATLAS: scaling inference compute on a frozen model can match or exceed a bigger one.

- **Brown et al., 2024.** *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* arXiv [2407.21787](https://arxiv.org/abs/2407.21787). Weaker models exceed stronger ones with enough samples — the core motivation for every ATLAS version.
- **Wang et al., 2023.** *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* [ICLR 2023](https://openreview.net/forum?id=1PL1NIMMrw). Majority voting over sampled CoTs; direct precursor to DivSampling + PlanSearch candidate pools.
- **Lightman et al., 2023.** *Let's Verify Step by Step.* arXiv [2305.20050](https://arxiv.org/abs/2305.20050). Process reward models — conceptual basis for ThinkPRM and the verification phases in V3.

---

## Code Generation with Feedback

V3's sandbox-test loop and PR-CoT repair are descendants of these.

- **Chen et al., 2022.** *CodeT: Code Generation with Generated Tests.* [ICLR 2023](https://openreview.net/forum?id=ktrw68Cmu9c). Test-driven code generation; blueprint for self-generated test harnesses in PR-CoT.
- **Zhang et al., 2023.** *Self-Edit: Fault-Aware Code Editor for Code Generation.* arXiv [2305.04087](https://arxiv.org/abs/2305.04087). Execution feedback loop; informs the refinement phase.
- **Shinn et al., 2023.** *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023. Verbal self-correction loop — foundational for the metacognitive component of Phase 3.

---

## Inference Efficiency

- **Leviathan et al., 2023.** *Fast Inference from Transformers via Speculative Decoding.* ICML 2023, arXiv [2211.17192](https://arxiv.org/abs/2211.17192). 2-3× speedup via draft models. Used in earlier V3.0 spec-decode configs; retained as a reference while V3.1 runs on 9B without a draft model.

---

## Fine-Tuning & Parameter-Efficient Training

Relevant to how ATLAS trains the Geometric Lens (small MLPs, XGBoost) rather than the base model.

- **Hu et al., 2021.** *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv [2106.09685](https://arxiv.org/abs/2106.09685). The classic.
- **Aghajanyan et al., 2020.** *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* arXiv [2012.13255](https://arxiv.org/abs/2012.13255). ~200 params can recover 90% of full fine-tuning — theoretical backbone for GeLoRA and for our decision to train small auxiliary networks instead of the model itself.
- **Dettmers et al., 2023.** *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv [2305.14314](https://arxiv.org/abs/2305.14314). 4-bit NF4; practical reference for low-VRAM training recipes.
- **Hong et al., 2024.** *GeLoRA: Geometric Adaptive Ranks for Efficient LoRA Fine-Tuning.* arXiv [2412.09250](https://arxiv.org/abs/2412.09250). Intrinsic dimensionality as a compression lower bound — direct inspiration for the geometric scoring approach in the Lens.

---

## Retrieval-Augmented Generation (RAG)

- **Zhang et al., 2024.** *RAFT: Adapting Language Model to Domain Specific RAG.* arXiv [2403.10131](https://arxiv.org/abs/2403.10131). Training for RAG in the presence of distractors. Reference for the confidence router's handling of noisy retrieval.
- **Asai et al., 2023.** *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* arXiv [2310.11511](https://arxiv.org/abs/2310.11511). Teaching models when to retrieve. Influences the proxy's selective retrieval decisions.
- **VectifyAI, 2025.** *MAFIN 2.5 / PageIndex.* Product / reasoning-based retrieval over tree structures. Basis for ATLAS's PageIndex V2 indexer (tree-sitter AST + BM25 + LLM-guided traversal).

---

## Geometric Foundations (Lens Core)

Why the Lens exists and why it works.

- **Anthropic, 2026.** *When Models Manipulate Manifolds* (internal research communication). Claude represents information on curved manifolds — foundational intuition that motivated our energy-based scoring.
- **Crespo, J., 2026.** *Everyone's Wrong About AI Programming — Except Maybe Anthropic.* Medium. Lyapunov energy landscapes on toroidal-solenoid spaces — accessible framing of the geometric perspective ATLAS operationalizes.
- **Blondel et al. (Google DeepMind), 2025.** *Autoregressive Models are Secretly Energy-Based Models.* arXiv [2512.15605](https://arxiv.org/abs/2512.15605). Exact ARM↔EBM bijection. **This is the paper that justifies the Lens existing at all** — the Lens is the EBM correction layer the paper describes. Primary citation for V3.

---

## Pattern Cache & Memory

The cache's decay curve and organic-forgetting behavior come from these.

- **Ebbinghaus, 1885.** *Über das Gedächtnis* (On Memory). The forgetting curve — memory strength decays roughly exponentially with time. Underlies the tiered decay schedule in the pattern cache.
- **ACT-R** (Anderson et al.). Adaptive Control of Thought-Rational cognitive architecture. ~30 day half-life for activation — numeric baseline for cache decay.
- **Luhmann, N.** *Zettelkasten System.* Knowledge management via displacement and organic pruning — conceptual model for how the cache evicts rather than garbage-collects.
- **Behrouz et al. (Google Research), 2025.** *Titans: Learning to Memorize at Test Time.* arXiv [2501.00663](https://arxiv.org/abs/2501.00663). Surprise-based memory; 170M outperforms GPT-4 on BABILong. Direction for future cache upgrades.
- **Park et al., 2025.** *Memoria: Resolving Fateful Forgetting Problem through Human-Inspired Memory Architecture.* arXiv [2310.03052](https://arxiv.org/abs/2310.03052). Hebbian learning + lifespan-based memory — alternative design space for the cache.

---

## V3 Pipeline — Phase 1 (Constraint-Driven Generation)

- **Wang et al., 2025.** *PlanSearch: Planning as Search over Programs.* ICLR 2025 Spotlight, arXiv [2409.03733](https://arxiv.org/abs/2409.03733). pass@200 of 77% on LiveCodeBench with Sonnet via idea-space diversity. **Primary citation for V3's PlanSearch component.**
- **Wang et al., 2025.** *Think Diverse: DivSampling through Perturbation-Based Diversity.* arXiv [2502.11027](https://arxiv.org/abs/2502.11027). 9.5% relative gain via perturbation. **Primary citation for V3's DivSampling component.**
- **Muennighoff et al., 2025.** *s1: Simple Test-Time Scaling (Budget Forcing).* arXiv [2501.19393](https://arxiv.org/abs/2501.19393). Wait-token injection; runaway chain prevention. **Primary citation for V3's Budget Forcing component.**
- **Aytes et al., 2025.** *Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching.* EMNLP 2025, arXiv [2503.05179](https://arxiv.org/abs/2503.05179). 76% token reduction, -0.14% accuracy. Reference for compressed reasoning prompts.

---

## V3 Pipeline — Phase 2 (Intelligent Compute)

- **Liu et al., 2025.** *Compute-Optimal TTS: 0.5B beats GPT-4o.* arXiv [2512.02008](https://arxiv.org/abs/2512.02008). No single TTS strategy dominates — motivates the Confidence Router's difficulty-aware routing.
- **Feng & Odonnat, 2025.** *Blend-ASC: Adaptive Sample Complexity for Efficient Test-Time Compute.* arXiv [2511.12309](https://arxiv.org/abs/2511.12309). 6.8× fewer samples at equal accuracy. **Primary citation for Blend-ASC adaptive-k.**
- **ReASC, 2026.** *Early Stopping for Test-Time Compute.* arXiv [2601.02970](https://arxiv.org/abs/2601.02970). Stop when confidence high enough. **Primary citation for ReASC.**
- **Li et al. (UC Berkeley), 2025.** *S\*: Test Time Scaling for Code Generation via Distinguishing Inputs.* arXiv [2502.14382](https://arxiv.org/abs/2502.14382). 85.7% LCB with 32B via distinguishing tests. **Primary citation for S\* candidate selection.**

---

## V3 Pipeline — Phase 4 (Verification / Repair)

- **FunPRM, 2025.** *Function-Level Process Reward Models.* Meta-learning correction via function decomposition. Reference for fine-grained PRM signals in V3 repair.
- **ThinkPRM, 2025.** *Generative Verification via Reasoning.* Thinking tokens for verification outperform direct generation. Basis for the thinking-first repair posture.

---

## Continual Learning & Failure Modes

Why ATLAS freezes the base model and operates *around* it.

- **Shumailov et al., 2024.** *The Curse of Recursion: Training on Generated Data Makes Models Forget (Model Collapse).* arXiv [2305.17493](https://arxiv.org/abs/2305.17493). Model collapse from synthetic data — the risk that gated all "train on our own traces" proposals in V2/V3.
- **Kirkpatrick et al., 2017.** *Overcoming Catastrophic Forgetting in Neural Networks (EWC).* PNAS, [doi:10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114). Elastic Weight Consolidation — the retraining recipe V3 uses when we do need to update the Lens without wiping prior knowledge.

---

## V3.2 Exploratory (new additions)

Papers that motivated roadmap items filed as open issues.

- **Karan & Chatterji, 2025.** *Reasoning with Sampling: Your Base Model is Smarter Than You Think.* arXiv [2510.14901](https://arxiv.org/abs/2510.14901). MCMC over logits during decoding — tracked as [issue #40](https://github.com/itigges22/ATLAS/issues/40).
- **Sotnikov, D., 2026.** *chiasmus: tree-sitter + solver call graph for code analysis.* GitHub [yogthos/chiasmus](https://github.com/yogthos/chiasmus). Inspiration for V3.1 structural code reasoning — tracked as [issue #39](https://github.com/itigges22/ATLAS/issues/39).

---

## What this list intentionally excludes

- **MoE / expert quantization work** (DynaExq, MoPEQ, MxMoE) — relevant only to the V3 Phase 5/6 model-swap roadmap, which is not active development.
- **Medusa** (parallel decoding heads) — ATLAS uses standard draft-model speculative decoding, not Medusa.
- **Forward-Forward, Direct Feedback Alignment** — speculative alternative learning rules, not in current plans.
- **Community posts without a primary source** — referenced informally in PRs/issues but not cited here.

If a paper you expected to see is missing and it's actually load-bearing for something shipping, open an issue and we'll add it.
