# Geometric Lens Models & Training Data

## Active Models

| File | Size | Purpose |
|------|------|---------|
| `cost_field.pt` | ~8M | C(x) cost field — hidden-dim→512→128→1 MLP, maps embeddings to correctness energy. Dim follows the trained model (bundled: 3840, Gemma 4 12B). |
| `cost_field.safetensors` | ~8M | Pickle-free twin of `cost_field.pt`, written together by `save_cost_field` and shipped by `atlas lens publish`. |
| `metric_tensor.pt` | 65M | Legacy G(x) metric tensor. The loader tries it FIRST; it yields to the XGBoost path only because its checkpoint architecture is `xgboost_importance` (load_metric_tensor returns None for non-`pca_contrastive` checkpoints) — not because the JSON pair exists. An old `pca_contrastive` checkpoint here would win over the JSON pair. |
| `gx_xgboost.json` | 17K | G(x) XGBoost ensemble — native XGBoost JSON dump (preferred loader path, see PC-031). |
| `gx_weights.json` | ~11M | G(x) PCA projection + training stats (hidden-dim→128). |

`gx_xgboost.pkl` (the legacy pickle fallback) is removed on retrain —
`save_gx` deletes it so a previous model's pickle can't shadow the JSON.

## Training Data

**In-repo sample**: `geometric-lens/data/sample/embeddings.json` — 10 embeddings (5 PASS, 5 FAIL) showing the training-data format (`{"embeddings": [...], "labels": [1|0, ...]}`; 3840-dim, from a Gemma 4 12B bench run).

**Full dataset on HuggingFace**: https://huggingface.co/datasets/itigges22/ATLAS

| Dataset | Embeddings | PASS | FAIL | Dimension | Size |
|---------|-----------|------|------|-----------|------|
| Phase 0 (original) | 597 | 504 | 93 | 4096 | 48MB |
| Full training set | 13,398 | 4,835 | 8,563 | 4096 | 1.1GB |
| Fox 9B variant | 800 | 400 | 400 | 4096 | 65MB |
| 5120-dim variant | 520 | — | — | 5120 | 53MB |

Note: The large training files (>2MB) are stored on HuggingFace, not in the git repo. Only the sample and model weights are committed.

## Training Stats

| File | Contents |
|------|---------|
| `phase0_stats.json` | Phase 0 C(x) training: Val AUC 0.9467, Sep 2.04x, 3-fold CV |
| `retrain_stats.json` | C(x) retrain: Val AUC 0.8245, 800 samples |
| `gx_train_stats.json` | G(x) XGBoost: 13,398 samples, PCA-128 + SupCon + LDA |

## Training Scripts

Located in `/scripts/`:
- `retrain_cx.py` — Retrain C(x) cost field from full dataset
- `retrain_cx_phase0.py` — Phase 0 C(x) training (597 samples)
- `retrain_lens_from_results.py` — Retrain from benchmark results
- `collect_lens_training_data.py` — Collect embeddings from benchmark runs
- `prepare_lens_training.py` — Prepare training data

## Reproduction

```bash
# Download full dataset from HuggingFace
# Place in geometric-lens/geometric_lens/models/

# Retrain C(x) from Phase 0 data (597 embeddings, ~2 min)
python scripts/retrain_cx_phase0.py

# Retrain C(x) from full data (13,398 embeddings)
python scripts/retrain_cx.py
```
