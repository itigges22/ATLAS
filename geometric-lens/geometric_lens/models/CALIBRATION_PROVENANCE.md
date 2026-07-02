# Lens Calibration Provenance

Per SUPPORT_MATRIX §9.5. One record per calibration run. The C(x)
cost field and its `cx_normalization.json` are bound together (the
normalization sigmoid is fit to that cost field's energy scale) — a
calibration file is only valid with the cost field it was derived from.

## gemma-4-12b-it-Q4_K_M — 2026-07-02

| Field | Value |
|---|---|
| Backbone | gemma-4-12b-it-Q4_K_M (GGUF) |
| Hidden dim | 3840 |
| Quantization | Q4_K_M |
| Hidden-state layer | per-model `/embedding` extraction (PC-202 patch) |
| Dataset | LiveCodeBench v5 (via `benchmark/results/gemma_lens`) |
| Samples | 287 embedded (196 PASS / 91 FAIL), 314 tasks with code |
| Split | 230 train (157/73) · 57 val (39/18) |
| Training commit | e2d0c4c (dev) |
| llama.cpp rev | 2e97c5f96f9fe2bb26f794a348e05d7a1c74baa1 |
| Trainer | `scripts/retrain_lens_from_results.py`, `retrain_cost_field_bce` |
| Hyperparameters | epochs 100 (early-stopped 60, patience 10), BCE loss |
| Seed | 42 (fixed in the script) |
| Metrics | val AUC **0.732**, train AUC 0.805, val acc 73.7%, Spearman ρ 0.464 |
| Energy separation | PASS mean 9.25 · FAIL mean 11.81 (higher = worse, correct direction) |
| Normalization | midpoint 10.53, steepness 1.57 (`cx_normalization.json`) |
| G(x) thresholds | unchanged (G(x) not retrained this run) |
| Created | 2026-07-02 |

### Reproduce

```bash
ATLAS_MODEL_NAME=gemma-4-12b-it-Q4_K_M \
  python3 scripts/retrain_lens_from_results.py \
  --results-dir benchmark/results/gemma_lens/v3_lcb/per_task --epochs 100
docker compose restart geometric-lens   # or POST /internal/lens/reload
```

### Status

Derived + verified on maintainer hardware (RTX 5060 Ti dev box); the
live gemma lens reports `cx_calibrated: true`. **Not yet published**:
the HF bundle (`itigges22/atlas-lens-gemma4-12b`) still hosts the
uncalibrated cost field, so a fresh `atlas model install-artifacts`
gets the uncalibrated bundle. Re-publishing the calibrated cost field +
`cx_normalization.json` is a maintainer decision — the val AUC is
moderate (0.73) and it overwrites a shared artifact, so it should not
be pushed as canonical without an A/B / quality check. Publish command
when ready:

```bash
atlas lens publish gemma-4-12b-it-Q4_K_M --repo itigges22/atlas-lens-gemma4-12b
# then re-pin lens_artifact_sha256 in model_registry.py and set
# lens_calibrated=True (docs/PUBLISHING.md)
```
