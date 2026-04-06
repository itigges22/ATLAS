# ATLAS V3 Implementation Status

**Last Updated**: 2026-03-05
**Current Phase**: V3.0 Complete — Preparing V3.1
**V3.0 Result**: **74.6% LCB pass@1** (447/599) on frozen Qwen3-14B (+33pp over V2)
**Next**: V3.1 — Model swap (Qwen3.5-9B), Lens Evolution (Phase 4), Phase 2 redesign

## V3.0 Ablation Study Results (Complete)

| Condition | Phases | Tasks | Pass Rate | Status |
|-----------|--------|-------|-----------|--------|
| A — Baseline | None | 599/599 | **54.9%** (329) | Complete |
| B — Phase 1 | P1 | 599/599 | **67.3%** (403) | Complete |
| C — Phase 1+2 | P1+P2 | 599/599 | **67.3%** (403) | Complete |
| D — Phase 1+3 | P1+P3 | 599/599 | **74.6%** (447) | Complete |
| E — Full V3 | P1+P2+P3 | 92/599 | — | Discontinued |

### Phase Contributions

| Phase | Contribution | Mechanism |
|-------|-------------|-----------|
| Phase 1 (PlanSearch/BF/Div) | +12.4pp (54.9→67.3%) | Diverse candidate generation |
| Phase 2 (Blend-ASC/S*) | +0.0pp (67.3→67.3%) | Non-functional on stdio tasks |
| Phase 3 (Self-verified refinement) | +7.3pp (67.3→74.6%) | PR-CoT=36, Refinement=6, Derivation=0 |

### V3 Revamp (Complete)

Phase 3 was redesigned to use self-generated test cases instead of real LCB tests. The model generates its own I/O pairs from the problem statement, uses them for iterative repair, and only submits to real tests for final scoring. This makes the result legitimate and comparable to other systems' pass@1.

Full ablation report: [V3_ABLATION_STUDY.md](V3_ABLATION_STUDY.md)

---

## Progress (Original Steps)

| Step | Feature | Status | Date | Notes |
|------|---------|--------|------|-------|
| 0 | Dual-Server Validation (PRE-1..5) | COMPLETE | 2026-02-26 | Single server: generation + spec decode + 5120-dim self-embeddings. Patched llama.cpp (draft embedding=false). ~76-104 tok/s. VRAM 15.4/16.3 GB. Nomic sidecar removed. |
| 1 | Budget Forcing (1C) | COMPLETE | 2026-02-24 | 71/71 tests pass. 5 tiers, Wait injection, energy→tier mapping, JSONL telemetry. AC-1C-1..5 validated. |
| 2 | PlanSearch (1A) | COMPLETE | 2026-02-24 | 39/39 tests pass. 3-step pipeline: constraints→plans→code. Budget Forcing integration. JSONL telemetry. AC-1A-1..5 validated. |
| 3 | DivSampling (1B) | COMPLETE | 2026-02-24 | 31/31 tests pass. 12 perturbations (4 role, 4 instruction, 4 style). Extensible library, JSONL telemetry. AC-1B-1..4 validated. |
| 4 | Phase 1 Benchmark + Go/No-Go | NOT STARTED | — | Gate: LCB >= +7% absolute |
| 5 | Blend-ASC + ReASC (2A+2B) | COMPLETE | 2026-02-24 | 87/87 tests pass. Adaptive K table (4 tiers), probe→energy→allocate. ReASC bottom-10% confidence + dual gate. AC-2A-1..4, AC-2B-1..3 validated. |
| 6 | S* Tiebreaking (2C) | COMPLETE | 2026-02-24 | 47/47 tests pass. Distinguishing input gen, sandbox scoring, tiebreak logic. AC-2C-1..4 validated. **BUG**: SandboxAdapter ignores test_case for stdio — fixed in revamp. |
| 7 | Phase 2 Benchmark + Go/No-Go | NOT STARTED | — | Gate: LCB >= +5% over P1 |
| 8 | Failure Analysis (3A) | COMPLETE | 2026-02-24 | 60/60 tests pass. 6 failure categories, constraint violation detection, structured analysis. AC-3A-1..4 validated. |
| 9 | PR-CoT Repair (3C) | COMPLETE | 2026-02-24 | 42/42 tests pass. 4 perspectives, repair generation, max rounds. AC-3C-1..3 validated. |
| 10 | Constraint Refinement (3B) | COMPLETE | 2026-02-24 | 49/49 tests pass. Hypothesis parsing, cosine distance filtering, refined constraint sets. AC-3B-1..4 validated. |
| 11 | Refinement Loop (3E) | COMPLETE | 2026-02-24 | 38/38 tests pass. Full analyze→refine→generate→test cycle, time budget, max iterations. AC-3E-1..5 validated. |
| 12 | Derivation Chains (3D) | COMPLETE | 2026-02-24 | 62/62 tests pass. Problem decomposition, sandbox-verified sub-steps, composition. **BUG**: sub-problem test_case ignored for stdio — fixed in revamp. |
| 13 | Metacognitive Model (3F) | COMPLETE | 2026-02-24 | 50/50 tests pass. Category-specific failure patterns, compensating constraints, LLM pattern extraction. AC-3F-1..4 validated. |
| 14 | ACE Playbooks (3G) | COMPLETE | 2026-02-24 | 83/83 tests pass. Evolving playbook, confidence decay (Ebbinghaus), token-budgeted injection, persistence. AC-3G-1..3 validated. **Revamp adds derivation graph.** |
| 15 | Phase 3 Benchmark + Go/No-Go | COMPLETE | 2026-02-26 | 40/50=80% pass@1 (+43pp over V2). Phase1: 34, PR-CoT: 5, Refinement: 1. Gate: PASS. **NOTE**: Used answer key — result is ceiling, not legitimate. |
| 16 | Replay Buffer (4A-CL) | COMPLETE | 2026-02-26 | 35/35 tests pass. Reservoir sampling, domain-stratified replay, 30%/70% mix, JSON persistence. AC-4A-CL-1..3 validated. |
| 17 | EWC (4A-EWC) | COMPLETE | 2026-02-26 | 24/24 tests pass. Diagonal Fisher, penalty term, save/load state, <5s compute. AC-4A-EWC-1..3 validated. |
| 18 | Enhanced Retrain (4A-RT) | COMPLETE | 2026-02-26 | 16/16 tests pass. Replay+EWC integrated into retrain_cost_field_bce. Post-retrain Fisher recompute + buffer update. Backward compatible. AC-4A-RT-1..3 validated. |
| 19 | Phase 4 Validation | COMPLETE | 2026-02-26 | 5/5 tests pass. 5-domain AUC retention=1.0000 (0 degradation). Energy gaps positive [8.04, 14.26, 13.51]. 3-domain cycle 0.6s. Gate: **GO** (retention > 0.99). |
| 20 | Full V3 Benchmark (14B) | COMPLETE | 2026-03-05 | **74.6% LCB pass@1** (447/599). Self-verified Phase 3. Ablation conditions A-D complete. |
| 21 | MoE Model Swap (Phase 5) | NOT STARTED | — | Only after Step 20 |

## Benchmark Results

| Phase | LCB pass@1 | Delta | Date | Run ID | Notes |
|-------|-----------|-------|------|--------|-------|
| V2 Baseline | 37% | — | 2026-02-17 | v2_run_20260217_125310 | |
| Ablation A (baseline) | **54.9%** | +17.9pp | 2026-03-01 | condition_a_baseline | No V3 features |
| Ablation B (P1) | **67.3%** | +30.3pp | 2026-03-01 | condition_b_phase1 | Phase 1 only |
| Ablation C (P1+P2) | **67.3%** | +30.3pp | 2026-03-03 | condition_c_phase1_2 | Phase 2 adds nothing |
| **Ablation D (P1+P3)** | **74.6%** | **+37.6pp** | 2026-03-05 | condition_d_phase1_3 | **V3.0 release result** |

## Go/No-Go Decisions

| Phase | Gate | Result | Decision | Date |
|-------|------|--------|----------|------|
| Phase 4 (Step 19) | AUC retention > 0.99 | 1.0000 (0 degradation across 5 domains) | **GO** | 2026-02-26 |
| V3.0 (Step 20) | LCB >= 70% | **74.6%** (447/599) | **GO** | 2026-03-05 |

## Key Decisions

- **2026-03-05**: **V3.0 RELEASE** — 74.6% LCB pass@1 confirmed with self-verified Phase 3. All documentation updated. Result is clean and comparable to other systems' pass@1.
- **2026-03-02**: **V3 REVAMP** — Redesigned Phase 3 to use self-generated tests instead of real LCB tests. ChatML bug fixed in self_test_gen.py.
- **2026-02-26**: Spec decode RESTORED. One-line patch to llama.cpp (`params_dft.embedding = false`). Single server does generation + spec decode + self-embeddings. ~76-104 tok/s.
- **2026-02-24**: Server A confirmed 5120-dim self-embeddings operational.

## V3.1 Planned Work

- Model swap to Qwen3.5-9B (native multi-token prediction)
- Lens Evolution (Phase 4) — online C(x) recalibration during benchmarks
- Phase 2 redesign — S* stdio-compatible distinguishing input generation
- Target: 80-90% LCB pass@1
