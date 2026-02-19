# Geometric Lens Continuous Learning Protocol — Design

**Date:** 2026-02-16
**Status:** Approved

## Problem

The Geometric Lens C(x) was trained on hand-labeled "expected difficulty" data (60 examples). A/B validation showed zero correlation with actual failures (Spearman rho = -0.05). The model needs to learn from real pass/fail outcomes.

## Solution

Epoch-based benchmark execution with rolling retraining of C(x) on real failure data collected during the run.

## Architecture

### Files Modified

1. **`benchmark/v2_runner.py`** — Epoch-based LCB execution, embedding collection, retrain orchestration
2. **`rag-api/geometric_lens/service.py`** — `reload_weights()` for hot reload
3. **`rag-api/geometric_lens/training.py`** — `retrain_cost_field_bce()` for BCE-based retraining
4. **`rag-api/main.py`** — `/internal/lens/retrain` and `/internal/lens/reload` endpoints

### New Files

5. **`benchmark/geo_learning.py`** — Spearman rho, retrain payload prep, LearningCurveTracker
6. **`benchmark/v2_report.py`** — Learning curve report section (modification)

### Data Flow

```
Benchmark Runner (host, stdlib)
  |
  |-- llama-server /embedding (get 5120-dim embedding per task)
  |-- llama-server /chat/completions (get LLM response)
  |-- sandbox execution (PASS/FAIL)
  |
  |-- Writes: training_embeddings.jsonl (embedding + label per task)
  |
  |-- Between epochs: POST to rag-api /internal/lens/retrain
  |-- After retrain: POST to rag-api /internal/lens/reload
```

### Epoch Structure

Tasks shuffled randomly (seed=42) before splitting:

| Epoch | Tasks | Training Data | Lens State |
|-------|-------|---------------|------------|
| 0 | 0-99 | 0 examples | OFF (baseline) |
| 1 | 100-299 | ~100 examples | Retrained |
| 2 | 300-499 | ~300 examples | Retrained |
| 3 | 500-699 | ~500 examples | Retrained |
| 4 | 700-N | ~700 examples | Retrained |

### Retraining

- BCE loss: FAIL=high energy (label 1.0), PASS=low energy (label 0.0)
- Class weighting for imbalanced data (pos_weight = n_pass / n_fail)
- 50 epochs, early stopping on validation AUC
- Adaptive LR: 1e-3 for <100 examples, 5e-4 for 100-500, 1e-4 for 500+
- Minimum 20 FAIL examples required before retraining

### Validation Criteria (from spec)

11. Rolling retrain executes in <60 seconds
12. Hot reload loads new weights without pod restart
13. Validation AUC after Epoch 0 > 0.55
14. Epoch 4 pass@1 >= Epoch 0 pass@1 (positive trend)
15. Spearman rho after final retrain > 0.10
