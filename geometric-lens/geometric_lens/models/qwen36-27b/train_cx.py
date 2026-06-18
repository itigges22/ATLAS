#!/usr/bin/env python3
"""Train C(x) cost field from pre-computed 5120-dim embeddings."""
import json, os, sys, time

path = "/app/geometric_lens/models/qwen36-27b/training_embeddings_5120d.json"
with open(path) as f:
    data = json.load(f)

sys.stderr.write(f"Loaded {len(data['embeddings'])} embeddings, dim={data['dim']}\n")
labels = [1 if l == "PASS" else 0 for l in data["labels"]]
data["labels"] = labels
sys.stderr.write(f"Labels: {sum(labels)} PASS, {len(labels) - sum(labels)} FAIL\n")
sys.stderr.flush()

from geometric_lens.training import train_cost_field, save_cost_field

start = time.time()
result = train_cost_field(data, epochs=200, lr=1e-3, margin=1.0)
elapsed = time.time() - start

test_auc = result.get("best_test_auc") or result.get("final_test_auc") or 0.0
train_auc = result.get("final_train_auc") or 0.0
sys.stderr.write(f"Train AUC: {train_auc:.4f}  |  Test AUC: {test_auc:.4f}  |  Time: {elapsed:.0f}s\n")
sys.stderr.flush()

artifact_dir = "/app/geometric_lens/models/qwen36-27b"
cost_path = save_cost_field(result["model"], save_dir=artifact_dir)
sys.stderr.write(f"Saved: {cost_path}\n")
sys.stderr.flush()
print("DONE")
