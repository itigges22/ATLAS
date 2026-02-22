# ATLAS V2.5 Ablation Study

Results of the V2.5 ablation study conducted 2026-02-19 to 2026-02-21, evaluating the Geometric Lens, router, and infrastructure components for the V3 go/no-go decision.

---

## 1. Executive Summary

V2.5 was a systematic ablation of ATLAS V2's infrastructure to determine which components provide real value and which are dormant or marginal. The headline finding: **Lens C(x) energy scoring is statistically indistinguishable from random candidate selection** (37.7% vs 37.1% pass@1, within the 3.4pp seed-to-seed variance). The diversity benefit of generating 3 candidates at temperature 0.6 (Best-of-K) accounts for nearly all improvement over single-attempt baselines. A secondary discovery -- that llama.cpp's `--embeddings` flag breaks speculative decoding -- led to a two-server architecture change that recovered ~2.6x generation throughput. For V3, BoK-3 diversity is validated, C(x) energy should be repurposed for difficulty routing rather than candidate ranking, and G(x) should be activated or removed.

---

## 2. Methodology

The study had two phases:

**Phase A -- Code Analysis** (Tasks 1-5): Static analysis of the ATLAS codebase to trace data flows, identify dormant components, and verify what the system actually does vs. what was intended.

- Task 1: G(x) metric tensor dormancy analysis
- Task 2: Component inventory audit (8 subsystems, 43 files, 7,980 LOC)
- Task 3: Scoring path trace (end-to-end from runner to sandbox)
- Task 4: Pattern cache status
- Task 5: Simple baseline comparison (LogReg, RandomForest vs C(x) MLP)

**Phase B -- Telemetry Analysis** (Tasks 7-10): Empirical ablation using modified benchmark runners with different candidate selection strategies, run on LiveCodeBench v5 (499 tasks with Best-of-K enabled).

- Task 7: Failure correlation analysis (PlanSearch addressable market)
- Task 8: Lens vs Random vs Reverse-energy selection
- Task 9: Energy threshold validation (difficulty tiers)
- Task 10: Multi-run variance analysis (3 seeds)

All runs used Qwen3-14B-Q4_K_M with Qwen3-0.6B-Q8_0 draft model on an RTX 5060 Ti 16GB.

---

## 3. Key Findings

### 3.1 G(x) Metric Tensor: Confirmed Dormant

**Status: FUNCTIONALLY DORMANT** (5.2M parameters, zero impact on outcomes)

G(x) is loaded at service startup and technically invoked via `/internal/lens/evaluate`, but its output (the corrected embedding `delta_x = -alpha * G^{-1} * grad_C`) is computed and immediately discarded. The benchmark only reads the scalar energy value from the response.

The primary scoring path (`/internal/lens/score-text`) uses **only C(x)** -- it never imports or calls `metric_tensor`, `compute_correction`, or `evaluate_and_correct`. No benchmark code references G(x) in any form.

**V3 options**: (A) Remove entirely and save ~5.2M params + ~20ms latency, (B) Activate by wiring the corrected embedding into retrieval or ranking, or (C) Keep dormant (current state).

### 3.2 Component Inventory

| Component | Files | LOC | Status |
|-----------|-------|-----|--------|
| geometric_lens | 8 | 1,544 | Active (C(x) only; G(x) dormant) |
| confidence_router | 6 | 536 | Active |
| pattern_cache | 8 | 1,092 | Bypassed by benchmark |
| pageindex_rag | 10 | 1,648 | Active (for interactive use) |
| best_of_k | 1 | 196 | Active |
| benchmark_runner | 3 | 1,993 | Orchestrator |
| embedding_extraction | 1 | 61 | Active |
| lora_training | 6 | 910 | Dormant (CronJob suspended) |
| **Total** | **43** | **7,980** | |

6 actively-wired subsystems (5,077 LOC across 34 files). The lora_training component is architecturally decoupled by design (K8s CronJob, not inline).

### 3.3 Scoring Path: C(x)-Only Confirmed

The effective V2 selection formula:

```
selected = argmin_i C(embed(code_i))
```

Where:
- C: MLP (input_dim -> 512 -> 128 -> 1, SiLU + Softplus)
- embed: mean-pooled embeddings from the embedding server
- Candidates sorted by energy ascending, sandboxed in order, early exit on first pass

No metric tensor, no gradient computation, no correction vector in the selection path. Telemetry signals (route decisions, cache stats, geometric energy) are logged but do not influence candidate ordering.

### 3.4 Pattern Cache: Bypassed Entirely

**Status: BYPASSED_ENTIRELY**

The benchmark runner calls llama-server directly (port 32735), completely bypassing rag-api and its pattern cache. The cache is fully implemented (8 files, 1,092 LOC) with functional read and write paths for interactive RAG use, but the benchmark never:
- Routes through rag-api's `/v1/chat/completions`
- Queries for matching patterns
- Writes new patterns after task completion

The benchmark's only rag-api contact is telemetry (cache stats, lens scoring).

### 3.5 Simple Baseline Comparison

| Model | AUC | Dataset |
|-------|-----|---------|
| C(x) MLP (2.7M params) | 1.000 | In-distribution (60 gate embeddings, 5-fold CV) |
| C(x) MLP | 0.805 | Out-of-distribution (LiveCodeBench) |
| LogReg (L2, StandardScaler) | 1.000 +/- 0.000 | In-distribution (5-fold CV) |
| RandomForest (100 trees) | 1.000 +/- 0.000 | In-distribution (5-fold CV) |

On in-distribution data (60 samples, 30 PASS / 30 FAIL), the data is linearly separable -- even logistic regression achieves perfect AUC. The C(x) MLP's 2.7M parameters are overkill for binary classification on this data. The MLP's justification rests on OOD generalization (0.805 on LCB) and differentiable energy output for geometric corrections, not in-distribution accuracy.

### 3.6 Failure Correlation (PlanSearch Addressable Market)

Analysis of 311 failures from 499 Best-of-K events:

| Diversity Level | Count | Percentage | PlanSearch Benefit |
|-----------------|-------|------------|-------------------|
| Identical solutions (1 unique) | 6 | 1.9% | Will help |
| Low diversity (2 unique) | 19 | 6.1% | Should help |
| Full diversity (3+ unique) | 286 | 92.0% | May NOT help |

**Only 8.0% of failures are addressable by PlanSearch-style intervention.** 92% of failures already have full diversity -- the model generates 3 distinct solutions, but none pass. This suggests the bottleneck is capability, not diversity.

First-pick accuracy: 151/188 (80.3%) -- the lens's lowest-energy candidate was a passer 80% of the time when at least one candidate passed.

### 3.7 Lens vs Random: THE Key Result

| Strategy | Pass@1 | Rate | Avg Sandbox Calls | First-Pick Rate |
|----------|--------|------|-------------------|-----------------|
| Energy (original) | 188/499 | 37.7% | 2.36 | 80.3% |
| Random | 185/499 | 37.1% | 2.33 | 83.2% |
| Reverse energy | 183/499 | 36.7% | 2.38 | 78.1% |

**Verdict: energy ~ random.** The C(x) energy scoring provides ~0.6pp advantage over random selection, which is within the ~1.7% seed-to-seed standard deviation. Sandbox call counts are nearly identical across all strategies. Random selection actually has a higher first-pick rate (83.2% vs 80.3%).

**Interpretation**: The BoK-3 diversity benefit (generating 3 candidates at temp=0.6) accounts for nearly all the pass@1 improvement. C(x) scoring adds negligible value for candidate ranking on LiveCodeBench.

### 3.8 Energy Threshold Validation

C(x) energy does correlate with task difficulty, even though it doesn't help with candidate ranking:

| Tier | Tasks | Percentage | Pass Rate |
|------|-------|------------|-----------|
| nothink (low energy) | 159 | 32% | 58.5% |
| standard | 92 | 18% | 50.0% |
| hard | 52 | 10% | 23.1% |
| extreme (high energy) | 196 | 39% | 18.9% |

Energy range: [0.0, 27.6], mean 9.3. Tasks with low energy pass at 3x the rate of high-energy tasks. This validates repurposing the energy signal for **difficulty-adaptive routing** (deciding how much compute to allocate per task) rather than candidate selection.

### 3.9 Multi-Run Variance

| Seed | Strategy | Pass Rate |
|------|----------|-----------|
| 42 (original) | energy | 37.7% |
| 123 | energy | 35.9% |
| 7 | energy | 34.3% |

**Mean: 36.0% +/- 1.7% (range: 3.4pp)**

The seed-to-seed variance (3.4pp) is **larger** than the energy-vs-random gap (0.6pp). The original 37.7% result may be on the high end of the variance band. True BoK-3 performance estimate: 36.0% +/- 1.7%.

---

## 4. Architecture Discovery

During the ablation study, we discovered that llama.cpp's `--embeddings` flag forces `n_batch = n_ubatch = 512`, which breaks speculative decoding (the draft model keeps its default `n_batch`, causing 0% token acceptance). This explained why V2's spec decode was "partially broken" -- it was completely broken, masked by the `--embeddings` flag.

**Solution**: A two-server sidecar architecture where:
- **Server A** (port 8000): Qwen3-14B + draft, spec decode ON, embeddings OFF (~100 tok/s)
- **Server B** (port 8001): nomic-embed-text-v1.5, embeddings ON, 768-dim output (~26ms per request)

Both containers share a single GPU in the same K3s pod. The embed sidecar adds only ~300 MiB VRAM. See [V2_TO_V2_5_MIGRATION.md](V2_TO_V2_5_MIGRATION.md) for full details.

---

## 5. V3 Go/No-Go Assessment

### Validated (keep for V3)
- **Best-of-K diversity**: Generating 3 candidates at temp=0.6 is the primary performance driver
- **Energy as difficulty signal**: C(x) energy correlates with task difficulty (58.5% vs 18.9% pass rates across tiers) -- repurpose for routing
- **Two-server architecture**: Spec decode + sidecar embeddings is stable and efficient
- **Epoch-based learning**: The lens retrain infrastructure works correctly

### Needs Revision
- **C(x) for candidate ranking**: Statistically indistinguishable from random. Either remove from selection path or fundamentally improve
- **G(x) metric tensor**: 5.2M dormant parameters. Activate (wire corrected embeddings into pipeline) or remove
- **Pattern cache for benchmarks**: Fully implemented but bypassed. Wire into benchmark flow or acknowledge it's interactive-only
- **Router for benchmarks**: All routing signals are logged but don't affect candidate generation flow in the current benchmark

### Checklist (10/10 complete, 0 blocked)

| Task | Status | Key Finding |
|------|--------|-------------|
| T1: G(x) dormancy | DONE | Confirmed dormant |
| T2: Component inventory | DONE | 8 components, 43 files, 7,980 LOC |
| T3: Scoring path trace | DONE | C(x)-only confirmed |
| T4: Pattern cache status | DONE | Bypassed entirely |
| T5: Simple baselines | DONE | Match C(x) in-dist; OOD inconclusive |
| T6: Ablation scripts | DONE | v2_runner_ablation.py written |
| T7: Failure correlation | DONE | 92% full-diversity failures, 8% PlanSearch-addressable |
| T8: Lens vs Random | DONE | 37.7% vs 37.1% -- marginal |
| T9: Thresholds | DONE | Energy predicts difficulty (58.5% vs 18.9%) |
| T10: Variance | DONE | 36.0% +/- 1.7%, seed variance > lens advantage |

---

## 6. Raw Data References

All raw data, scripts, and intermediate findings are in `v2_5_results/` (gitignored, available on request):

```
v2_5_results/
  report/V2_5_Analysis_Report.md    -- Auto-assembled full report
  findings/                          -- Individual finding documents (01-10)
  scripts/                           -- Analysis scripts (Python)
    v2_runner_ablation.py            -- Modified V2 runner with ablation strategies
    component_inventory.py           -- Codebase audit script
    simple_baselines.py              -- sklearn baseline comparison
    failure_correlation.py           -- BoK failure diversity analysis
    lens_vs_random.py                -- Strategy comparison
    validate_thresholds.py           -- Energy-difficulty tier analysis
    variance.py                      -- Multi-run aggregation
    assemble_report.py               -- Report assembly
```
