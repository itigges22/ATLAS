#!/usr/bin/env python3
import os
ATLAS_DIR = os.environ.get("ATLAS_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Retrain C(x) cost field using Fox 9B embeddings.

Collects code + pass/fail labels from ablation results,
embeds each through Fox at localhost:8080, then trains
the cost field to discriminate PASS vs FAIL.
"""

import json
import glob
import os
import sys
import time
import random
from urllib.request import Request, urlopen

# Add geometric-lens to path for imports
sys.path.insert(0, '" + ATLAS_DIR + "/geometric-lens')

FOX_URL = "http://localhost:8080/embedding"
MODELS_DIR = "" + ATLAS_DIR + "/geometric-lens/geometric_lens/models"
DATA_DIR = "" + ATLAS_DIR + "/v3_ablation_results/condition_a"


def get_fox_embedding(text: str, retries=2) -> list:
    """Get 4096-dim embedding from Fox 9B."""
    payload = json.dumps({"content": text}).encode()
    req = Request(FOX_URL, data=payload, headers={"Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            emb = data[0]["embedding"]
            if isinstance(emb[0], list):
                # Per-token: mean pool
                dim = len(emb[0])
                pooled = [0.0] * dim
                for tok in emb:
                    for i, v in enumerate(tok):
                        pooled[i] += v
                for i in range(dim):
                    pooled[i] /= len(emb)
                return pooled
            return emb
        except Exception as e:
            if attempt < retries:
                print(f"  Retry {attempt+1}: {e}")
                time.sleep(1)
            else:
                raise


def collect_samples():
    """Load code + labels from all benchmark results."""
    samples = []
    seen_codes = set()  # dedup by code hash

    def add_sample(code, passed, task_id):
        if not code or len(code) < 20:
            return
        h = hash(code[:500])
        if h in seen_codes:
            return
        seen_codes.add(h)
        samples.append({
            "code": code,
            "label": "PASS" if passed else "FAIL",
            "task_id": task_id
        })

    # Source 1: V3 ablation condition_a (has code directly)
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
        try:
            d = json.load(open(f))
            add_sample(d.get("code", ""), d.get("passed", False),
                       d.get("task_id", os.path.basename(f)))
        except Exception:
            pass

    # Source 2: V2 epoch results (code in attempts[].generated_code)
    v2_dirs = sorted(glob.glob(
        "v2_5_results/benchmarks/v2_original_run/epoch_*/per_task"))
    for d in v2_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                data = json.load(open(f))
                for a in data.get("attempts", []):
                    code = a.get("generated_code", "")
                    add_sample(code, a.get("passed", False),
                               a.get("task_id", os.path.basename(f)))
            except Exception:
                pass

    return samples


def embed_samples(samples):
    """Embed all code samples through Fox."""
    embeddings = []
    labels = []
    total = len(samples)

    for i, s in enumerate(samples):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Embedding {i+1}/{total}...", flush=True)

        try:
            emb = get_fox_embedding(s["code"])
            embeddings.append(emb)
            labels.append(s["label"])
        except Exception as e:
            print(f"  Skip {s['task_id']}: {e}")

    return embeddings, labels


def train_model(embeddings, labels):
    """Train C(x) cost field using BCE-style training."""
    import torch
    from geometric_lens.cost_field import CostField

    dim = len(embeddings[0])
    print(f"\nTraining C(x) on {len(embeddings)} samples (dim={dim})")

    # Convert labels
    n_pass = sum(1 for l in labels if l == "PASS")
    n_fail = sum(1 for l in labels if l == "FAIL")
    print(f"  Distribution: {n_pass} PASS, {n_fail} FAIL")

    # Convert to tensors
    X = torch.tensor(embeddings, dtype=torch.float32)
    # Targets: PASS → 2.0 (low energy), FAIL → 25.0 (high energy)
    PASS_TARGET = 2.0
    FAIL_TARGET = 25.0
    Y = torch.tensor([PASS_TARGET if l == "PASS" else FAIL_TARGET for l in labels],
                      dtype=torch.float32)

    # Train/val split (80/20, stratified)
    pass_idx = [i for i, l in enumerate(labels) if l == "PASS"]
    fail_idx = [i for i, l in enumerate(labels) if l == "FAIL"]
    random.seed(42)
    random.shuffle(pass_idx)
    random.shuffle(fail_idx)

    val_pass = pass_idx[:len(pass_idx) // 5]
    val_fail = fail_idx[:len(fail_idx) // 5]
    train_pass = pass_idx[len(pass_idx) // 5:]
    train_fail = fail_idx[len(fail_idx) // 5:]

    train_idx = train_pass + train_fail
    val_idx = val_pass + val_fail
    random.shuffle(train_idx)
    random.shuffle(val_idx)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    labels_val = [labels[i] for i in val_idx]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Class weights (balance PASS/FAIL)
    weight_pass = len(train_idx) / (2 * sum(1 for i in train_idx if labels[i] == "PASS"))
    weight_fail = len(train_idx) / (2 * sum(1 for i in train_idx if labels[i] == "FAIL"))
    W_train = torch.tensor([weight_pass if labels[i] == "PASS" else weight_fail
                            for i in train_idx], dtype=torch.float32)

    # Model
    model = CostField(input_dim=dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_val_auc = 0
    best_state = None
    patience = 15
    stale = 0

    for epoch in range(200):
        model.train()
        # Mini-batch training
        perm = torch.randperm(len(X_train))
        batch_size = 64
        epoch_loss = 0
        n_batches = 0

        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            idx = perm[start:end]
            xb, yb, wb = X_train[idx], Y_train[idx], W_train[idx]

            pred = model(xb).squeeze(-1)
            loss = (wb * (pred - yb) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val).squeeze(-1)
                val_loss = ((val_pred - Y_val) ** 2).mean().item()

                # Compute AUC
                val_auc = compute_auc(val_pred.numpy(), labels_val)

                # Energy stats
                pass_mask = Y_val < 10
                fail_mask = Y_val > 10
                pass_e = val_pred[pass_mask].mean().item() if pass_mask.any() else 0
                fail_e = val_pred[fail_mask].mean().item() if fail_mask.any() else 0

            print(f"  Epoch {epoch+1:3d}: loss={epoch_loss/n_batches:.4f} "
                  f"val_auc={val_auc:.4f} pass_E={pass_e:.2f} fail_E={fail_e:.2f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    # Load best state
    if best_state:
        model.load_state_dict(best_state)

    # Final eval
    model.eval()
    with torch.no_grad():
        all_pred = model(X).squeeze(-1)
        pass_energies = [all_pred[i].item() for i in range(len(labels)) if labels[i] == "PASS"]
        fail_energies = [all_pred[i].item() for i in range(len(labels)) if labels[i] == "FAIL"]

    pass_mean = sum(pass_energies) / len(pass_energies)
    fail_mean = sum(fail_energies) / len(fail_energies)

    print(f"\n  FINAL: val_auc={best_val_auc:.4f}")
    print(f"  PASS energy: mean={pass_mean:.2f}, range=[{min(pass_energies):.2f}, {max(pass_energies):.2f}]")
    print(f"  FAIL energy: mean={fail_mean:.2f}, range=[{min(fail_energies):.2f}, {max(fail_energies):.2f}]")
    print(f"  Separation: {fail_mean/pass_mean:.2f}x")

    return model, best_val_auc, pass_mean, fail_mean


def compute_auc(scores, labels):
    """Simple AUC: probability that a random FAIL has higher energy than a random PASS."""
    pass_scores = [scores[i] for i in range(len(labels)) if labels[i] == "PASS"]
    fail_scores = [scores[i] for i in range(len(labels)) if labels[i] == "FAIL"]
    if not pass_scores or not fail_scores:
        return 0.5
    concordant = 0
    total = 0
    for f in fail_scores:
        for p in pass_scores:
            total += 1
            if f > p:
                concordant += 1
            elif f == p:
                concordant += 0.5
    return concordant / total


def main():
    print("=" * 60)
    print("C(x) Cost Field Retraining — Fox 9B Embeddings")
    print("=" * 60)

    # Step 1: Collect labeled code
    print("\n[1/4] Collecting labeled samples...")
    samples = collect_samples()
    n_pass = sum(1 for s in samples if s["label"] == "PASS")
    n_fail = sum(1 for s in samples if s["label"] == "FAIL")
    print(f"  Found {len(samples)} samples ({n_pass} PASS, {n_fail} FAIL)")

    # Balance and cap at ~800 total (400 each max) for manageable embedding time
    MAX_PER_CLASS = 400
    pass_samples = [s for s in samples if s["label"] == "PASS"]
    fail_samples = [s for s in samples if s["label"] == "FAIL"]
    random.seed(42)
    random.shuffle(pass_samples)
    random.shuffle(fail_samples)
    samples = pass_samples[:MAX_PER_CLASS] + fail_samples[:MAX_PER_CLASS]
    random.shuffle(samples)
    n_pass = sum(1 for s in samples if s["label"] == "PASS")
    n_fail = sum(1 for s in samples if s["label"] == "FAIL")
    print(f"  After balancing: {len(samples)} samples ({n_pass} PASS, {n_fail} FAIL)")

    # Step 2: Embed through Fox
    print("\n[2/4] Embedding through Fox 9B (this takes a few minutes)...")
    start = time.time()
    embeddings, labels = embed_samples(samples)
    elapsed = time.time() - start
    print(f"  Embedded {len(embeddings)} samples in {elapsed:.1f}s ({elapsed/len(embeddings):.2f}s/sample)")
    print(f"  Embedding dim: {len(embeddings[0])}")

    # Save embeddings for future use
    emb_path = os.path.join(MODELS_DIR, "training_embeddings_fox9b.json")
    print(f"\n[3/4] Saving embeddings to {emb_path}...")
    with open(emb_path, 'w') as f:
        json.dump({
            "embeddings": embeddings,
            "labels": labels,
            "dim": len(embeddings[0]),
            "model": "Qwen3.5-9B (Fox)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_pass": sum(1 for l in labels if l == "PASS"),
            "n_fail": sum(1 for l in labels if l == "FAIL"),
        }, f)
    print(f"  Saved {len(embeddings)} embeddings")

    # Step 3: Train
    print("\n[4/4] Training C(x) cost field...")
    model, val_auc, pass_mean, fail_mean = train_model(embeddings, labels)

    # Save model
    import torch
    import shutil

    # Backup old model
    old_path = os.path.join(MODELS_DIR, "cost_field.pt")
    backup_path = os.path.join(MODELS_DIR, "cost_field_pretrain_backup.pt")
    if os.path.exists(old_path):
        shutil.copy2(old_path, backup_path)
        print(f"\n  Backed up old model to {backup_path}")

    # Save new model
    torch.save(model.state_dict(), old_path)
    print(f"  Saved new model to {old_path}")

    # Save stats
    stats = {
        "val_auc": val_auc,
        "pass_energy_mean": pass_mean,
        "fail_energy_mean": fail_mean,
        "n_samples": len(embeddings),
        "n_pass": sum(1 for l in labels if l == "PASS"),
        "n_fail": sum(1 for l in labels if l == "FAIL"),
        "dim": len(embeddings[0]),
        "model": "Qwen3.5-9B (Fox)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    stats_path = os.path.join(MODELS_DIR, "retrain_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats to {stats_path}")

    # Hot reload
    print("\n  Attempting hot reload via RAG API...")
    try:
        req = Request("http://localhost:31144/internal/lens/reload",
                       method="POST",
                       headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as resp:
            print(f"  Reload response: {resp.read().decode()}")
    except Exception as e:
        print(f"  Hot reload failed (manual restart needed): {e}")

    print(f"\n{'=' * 60}")
    print(f"DONE! C(x) retrained: val_AUC={val_auc:.4f}")
    print(f"  PASS energy: {pass_mean:.2f}")
    print(f"  FAIL energy: {fail_mean:.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
