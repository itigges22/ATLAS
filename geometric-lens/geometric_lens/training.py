"""Training script for C(x) cost field.

Trains C(x) with contrastive ranking loss on PASS/FAIL embedding pairs.

Note: G(x) metric tensor training was removed in V3.1 — V2.5.1 ablation
confirmed zero contribution at any correction strength (5.2M dead params).

Designed to run inside the geometric-lens container where torch is available.
"""

import json
import math
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric_lens.cost_field import CostField


def load_gate_data(path: str = None) -> dict:
    """Load embeddings and labels from gate analysis."""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "gate_embeddings.json"
        )
    with open(path) as f:
        return json.load(f)


def build_pairs(embeddings, labels):
    """Build contrastive pairs: (pass_embedding, fail_embedding)."""
    pass_embs = [e for e, l in zip(embeddings, labels) if l == 1]
    fail_embs = [e for e, l in zip(embeddings, labels) if l == 0]
    pairs = []
    for p in pass_embs:
        for f in fail_embs:
            pairs.append((p, f))
    return pairs


def train_cost_field(
    data: dict,
    epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 1.0,
    weight_decay: float = 1e-4,
    test_fraction: float = 0.3,
    seed: int = 42,
) -> dict:
    """Train C(x) with contrastive ranking loss.

    Loss = max(0, C(x_pass) - C(x_fail) + margin)

    We want C(x_fail) > C(x_pass) + margin.

    Returns dict with model, metrics, train/test AUC.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    embeddings = data["embeddings"]
    labels = data["labels"]
    dim = len(embeddings[0])

    # Stratified train/test split
    pass_idx = [i for i, l in enumerate(labels) if l == 1]
    fail_idx = [i for i, l in enumerate(labels) if l == 0]
    random.shuffle(pass_idx)
    random.shuffle(fail_idx)

    n_pass_test = max(1, int(len(pass_idx) * test_fraction))
    n_fail_test = max(1, int(len(fail_idx) * test_fraction))

    test_idx = set(pass_idx[:n_pass_test] + fail_idx[:n_fail_test])
    train_idx = [i for i in range(len(embeddings)) if i not in test_idx]

    train_embs = [embeddings[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_embs = [embeddings[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"Train: {len(train_embs)} (PASS={sum(train_labels)}, FAIL={len(train_labels)-sum(train_labels)})")
    print(f"Test:  {len(test_embs)} (PASS={sum(test_labels)}, FAIL={len(test_labels)-sum(test_labels)})")

    # Build contrastive pairs
    train_pairs = build_pairs(train_embs, train_labels)
    test_pairs = build_pairs(test_embs, test_labels)
    print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    # Convert to tensors
    device = torch.device("cpu")
    model = CostField(input_dim=dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    loss_history = []
    best_test_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_pairs)
        total_loss = 0.0
        n_batches = 0

        # Mini-batch training (batch_size=32 pairs)
        batch_size = min(32, len(train_pairs))
        for batch_start in range(0, len(train_pairs), batch_size):
            batch = train_pairs[batch_start:batch_start + batch_size]
            pass_batch = torch.tensor([p[0] for p in batch], dtype=torch.float32, device=device)
            fail_batch = torch.tensor([p[1] for p in batch], dtype=torch.float32, device=device)

            c_pass = model(pass_batch)
            c_fail = model(fail_batch)

            # Ranking loss: want C(fail) > C(pass) + margin
            loss = torch.relu(c_pass - c_fail + margin).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        loss_history.append(avg_loss)

        # Report every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()  # Note: model.eval() is the PyTorch eval mode toggle, not code evaluation
            with torch.no_grad():
                # Compute AUC on test set
                test_auc = compute_energy_auc(model, test_embs, test_labels, device)
                train_auc = compute_energy_auc(model, train_embs, train_labels, device)

            print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}")

            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Final assessment
    model.eval()  # Note: model.eval() is the PyTorch eval mode toggle, not code evaluation
    with torch.no_grad():
        final_train_auc = compute_energy_auc(model, train_embs, train_labels, device)
        final_test_auc = compute_energy_auc(model, test_embs, test_labels, device)

        # Compute energy statistics
        all_pass = torch.tensor([e for e, l in zip(embeddings, labels) if l == 1],
                                dtype=torch.float32, device=device)
        all_fail = torch.tensor([e for e, l in zip(embeddings, labels) if l == 0],
                                dtype=torch.float32, device=device)
        pass_energies = model(all_pass).squeeze()
        fail_energies = model(all_fail).squeeze()

    print(f"\n--- Final Results ---")
    print(f"Best test AUC: {best_test_auc:.4f}")
    print(f"Final train AUC: {final_train_auc:.4f}")
    print(f"Final test AUC: {final_test_auc:.4f}")
    print(f"PASS energy: {pass_energies.mean():.4f} +/- {pass_energies.std():.4f}")
    print(f"FAIL energy: {fail_energies.mean():.4f} +/- {fail_energies.std():.4f}")
    print(f"Separation: {fail_energies.mean() - pass_energies.mean():.4f}")

    return {
        "model": model,
        "best_test_auc": best_test_auc,
        "final_train_auc": final_train_auc,
        "final_test_auc": final_test_auc,
        "pass_energy_mean": pass_energies.mean().item(),
        "fail_energy_mean": fail_energies.mean().item(),
        "loss_history": loss_history,
    }


def compute_energy_auc(model, embeddings, labels, device):
    """Compute AUC: does C(x) rank FAIL higher than PASS?

    Higher AUC = better separation (FAIL embeddings get higher energy).
    """
    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    energies = model(X).squeeze().tolist()

    # AUC: probability that a random FAIL has higher energy than a random PASS
    pass_e = [e for e, l in zip(energies, labels) if l == 1]
    fail_e = [e for e, l in zip(energies, labels) if l == 0]

    if not pass_e or not fail_e:
        return 0.5

    concordant = 0
    total = 0
    for fe in fail_e:
        for pe in pass_e:
            total += 1
            if fe > pe:
                concordant += 1
            elif fe == pe:
                concordant += 0.5

    return concordant / total if total > 0 else 0.5


def save_cost_field(cost_field, save_dir=None):
    """Save trained C(x) model weights."""
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(save_dir, exist_ok=True)

    cost_path = os.path.join(save_dir, "cost_field.pt")
    torch.save(cost_field.state_dict(), cost_path)
    print(f"C(x) model saved to {cost_path}")
    return cost_path


def load_cost_field(save_dir=None, dim=None):
    """Load trained C(x) model weights.

    If dim is None, infers from saved weights. Falls back to EMBEDDING_DIM
    env var (default 768).
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    cost_path = os.path.join(save_dir, "cost_field.pt")
    if dim is None:
        if os.path.exists(cost_path):
            sd = torch.load(cost_path, map_location="cpu", weights_only=True)
            dim = sd["net.0.weight"].shape[1]  # first Linear's input dim
        else:
            dim = int(os.environ.get("EMBEDDING_DIM", "768"))

    cost_field = CostField(input_dim=dim)

    if os.path.exists(cost_path):
        cost_field.load_state_dict(torch.load(cost_path, map_location="cpu", weights_only=True))

    cost_field.eval()
    return cost_field


# Backward-compatible aliases
def save_models(cost_field, metric_tensor=None, save_dir=None):
    """Save model weights. metric_tensor param kept for backward compat (ignored)."""
    return save_cost_field(cost_field, save_dir), None


def load_models(save_dir=None, dim=None):
    """Load model weights. Returns (cost_field, None) for backward compat."""
    return load_cost_field(save_dir, dim), None


def retrain_cost_field_bce(
    embeddings: list,
    labels: list,
    epochs: int = 50,
    lr: float = None,
    weight_decay: float = 1e-4,
    test_fraction: float = 0.2,
    seed: int = 42,
    save_path: str = None,
    replay_buffer=None,
    ewc=None,
    domain: str = None,
    epoch_num: int = 0,
) -> dict:
    """Retrain CostField on real pass/fail benchmark data using weighted MSE.

    Uses MSE loss to energy targets (PASS=2.0, FAIL=25.0) with class-weighted
    samples to handle imbalanced data. Includes early stopping on validation
    AUC with patience=10 (checked every 5 epochs).

    Phase 4 additions:
    - replay_buffer: If provided, mixes 30% old data with 70% new data (4A-CL).
    - ewc: If provided, adds EWC penalty to preserve old domain weights (4A-EWC).
    - domain: Domain tag for replay buffer entries (e.g., "LCB", "SciCode").
    - epoch_num: Training epoch number for replay buffer tracking.

    Args:
        embeddings: List of float lists, each 5120-dim.
        labels: List of "PASS" or "FAIL" strings.
        epochs: Maximum training epochs.
        lr: Learning rate. If None, selected adaptively based on dataset size.
        weight_decay: AdamW weight decay.
        test_fraction: Fraction of data reserved for validation.
        seed: Random seed for reproducibility.
        save_path: If provided, save model state_dict to this path.
        replay_buffer: ReplayBuffer instance for continual learning (4A-CL).
        ewc: ElasticWeightConsolidation instance for forgetting prevention (4A-EWC).
        domain: Domain tag string for replay buffer.
        epoch_num: Epoch number for replay buffer entries.

    Returns:
        Dict with val_auc, train_auc, val_accuracy, train_size, val_size,
        fail_ratio, best_test_auc, model, and skipped flag.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    n = len(embeddings)
    n_pass = sum(1 for l in labels if l == "PASS")
    n_fail = sum(1 for l in labels if l == "FAIL")

    # Minimum data check
    if n_fail < 5 or n_pass < 5:
        return {
            "skipped": True,
            "val_auc": 0.5,
            "train_auc": 0.5,
            "val_accuracy": 0.0,
            "train_size": 0,
            "val_size": 0,
            "fail_ratio": n_fail / max(n, 1),
            "best_test_auc": 0.5,
            "model": None,
        }

    # Mix with replay buffer if provided (4A-CL: 30% old / 70% new)
    if replay_buffer is not None and len(replay_buffer) > 0:
        new_data = [{"embedding": e, "label": l} for e, l in zip(embeddings, labels)]
        mixed = replay_buffer.get_training_mix(new_data, replay_ratio=0.30)
        embeddings = [d["embedding"] for d in mixed]
        labels = [d["label"] for d in mixed]
        n = len(embeddings)
        n_pass = sum(1 for l in labels if l == "PASS")
        n_fail = sum(1 for l in labels if l == "FAIL")
        print(f"Retrain BCE | Replay mix: {len(new_data)} new + {n - len(new_data)} replay = {n} total")

    # Adaptive learning rate
    if lr is None:
        if n < 100:
            lr = 1e-3
        elif n < 500:
            lr = 5e-4
        else:
            lr = 1e-4

    # Convert string labels to numeric (PASS=1, FAIL=0) for AUC computation
    numeric_labels = [1 if l == "PASS" else 0 for l in labels]

    # Stratified train/test split
    pass_idx = [i for i, l in enumerate(labels) if l == "PASS"]
    fail_idx = [i for i, l in enumerate(labels) if l == "FAIL"]
    random.shuffle(pass_idx)
    random.shuffle(fail_idx)

    n_pass_test = max(1, int(len(pass_idx) * test_fraction))
    n_fail_test = max(1, int(len(fail_idx) * test_fraction))

    test_idx = set(pass_idx[:n_pass_test] + fail_idx[:n_fail_test])
    train_idx = [i for i in range(n) if i not in test_idx]

    train_embs = [embeddings[i] for i in train_idx]
    train_labels = [numeric_labels[i] for i in train_idx]
    val_embs = [embeddings[i] for i in test_idx]
    val_labels = [numeric_labels[i] for i in test_idx]

    print(f"Retrain BCE | Train: {len(train_embs)} (PASS={sum(train_labels)}, FAIL={len(train_labels)-sum(train_labels)})")
    print(f"Retrain BCE | Val:   {len(val_embs)} (PASS={sum(val_labels)}, FAIL={len(val_labels)-sum(val_labels)})")

    # Energy targets and class weights
    PASS_TARGET = 2.0
    FAIL_TARGET = 25.0
    THRESHOLD = 13.5

    fail_weight = n_pass / max(n_fail, 1)
    pass_weight = 1.0

    # Build per-sample target and weight tensors for training set
    device = torch.device("cpu")
    train_targets = []
    train_weights = []
    for label in train_labels:
        if label == 1:
            train_targets.append(PASS_TARGET)
            train_weights.append(pass_weight)
        else:
            train_targets.append(FAIL_TARGET)
            train_weights.append(fail_weight)

    train_X = torch.tensor(train_embs, dtype=torch.float32, device=device)
    train_T = torch.tensor(train_targets, dtype=torch.float32, device=device).unsqueeze(1)
    train_W = torch.tensor(train_weights, dtype=torch.float32, device=device).unsqueeze(1)

    # Initialize model and optimizer
    dim = len(embeddings[0])
    model = CostField(input_dim=dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop with early stopping
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()

        # Shuffle training data
        perm = torch.randperm(len(train_embs))
        train_X_shuffled = train_X[perm]
        train_T_shuffled = train_T[perm]
        train_W_shuffled = train_W[perm]

        total_loss = 0.0
        n_batches = 0
        batch_size = min(32, len(train_embs))

        for batch_start in range(0, len(train_embs), batch_size):
            batch_end = batch_start + batch_size
            x_batch = train_X_shuffled[batch_start:batch_end]
            t_batch = train_T_shuffled[batch_start:batch_end]
            w_batch = train_W_shuffled[batch_start:batch_end]

            energies = model(x_batch)
            per_sample_mse = (energies - t_batch) ** 2
            weighted_loss = (per_sample_mse * w_batch).mean()

            # Add EWC penalty to prevent catastrophic forgetting (4A-EWC)
            if ewc is not None and ewc.is_initialized:
                weighted_loss = weighted_loss + ewc.penalty(model)

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Check validation AUC every 5 epochs for early stopping
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_auc = compute_energy_auc(model, val_embs, val_labels, device)
                train_auc = compute_energy_auc(model, train_embs, train_labels, device)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_val_auc = compute_energy_auc(model, val_embs, val_labels, device)
        final_train_auc = compute_energy_auc(model, train_embs, train_labels, device)

        # Compute accuracy at threshold 13.5
        val_X = torch.tensor(val_embs, dtype=torch.float32, device=device)
        val_energies = model(val_X).squeeze()
        val_predictions = ["FAIL" if e > THRESHOLD else "PASS" for e in val_energies.tolist()]
        val_true = ["PASS" if l == 1 else "FAIL" for l in val_labels]
        val_correct = sum(1 for p, t in zip(val_predictions, val_true) if p == t)
        val_accuracy = val_correct / max(len(val_labels), 1)

    # Spearman ρ between energy and outcome (FAIL=1, PASS=0)
    # For binary labels, rank-biserial ρ = 2*AUC - 1 (exact, Cureton 1956)
    spearman_rho = 2.0 * final_val_auc - 1.0

    print(f"\n--- Retrain Results ---")
    print(f"Best val AUC:    {best_val_auc:.4f}")
    print(f"Final train AUC: {final_train_auc:.4f}")
    print(f"Final val AUC:   {final_val_auc:.4f}")
    print(f"Val accuracy:    {val_accuracy:.2%} (threshold={THRESHOLD})")
    print(f"Spearman ρ:      {spearman_rho:.4f}")

    # Save model if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # Compute energy statistics for Lens Feedback recalibration
    # Note: model.eval() is the PyTorch eval mode toggle, not code evaluation
    model.eval()
    with torch.no_grad():
        all_X = torch.tensor(embeddings, dtype=torch.float32, device=device)
        all_energies = model(all_X).squeeze().tolist()
        if isinstance(all_energies, float):
            all_energies = [all_energies]

    _pass_e = [e for e, l in zip(all_energies, labels) if l == "PASS"]
    _fail_e = [e for e, l in zip(all_energies, labels) if l == "FAIL"]
    pass_energy_mean = sum(_pass_e) / max(len(_pass_e), 1) if _pass_e else 0.0
    fail_energy_mean = sum(_fail_e) / max(len(_fail_e), 1) if _fail_e else 0.0

    # Post-retrain: update replay buffer with new domain samples (4A-CL)
    if replay_buffer is not None and domain is not None:
        sorted_e = sorted(all_energies)
        q25 = sorted_e[len(sorted_e) // 4] if len(sorted_e) > 3 else sorted_e[0]
        q50 = sorted_e[len(sorted_e) // 2] if len(sorted_e) > 1 else sorted_e[0]
        q75 = sorted_e[3 * len(sorted_e) // 4] if len(sorted_e) > 3 else sorted_e[-1]

        difficulty_quartiles = []
        for e in all_energies:
            if e <= q25:
                difficulty_quartiles.append(1)
            elif e <= q50:
                difficulty_quartiles.append(2)
            elif e <= q75:
                difficulty_quartiles.append(3)
            else:
                difficulty_quartiles.append(4)

        replay_buffer.add_batch(embeddings, labels, domain, epoch_num,
                                difficulty_quartiles)
        print(f"Retrain BCE | Added {len(embeddings)} samples to replay buffer (domain={domain})")

    # Post-retrain: recompute Fisher for EWC (4A-EWC)
    if ewc is not None and domain is not None:
        ewc.compute_fisher(model, embeddings, labels, n_samples=min(500, len(embeddings)))
        print(f"Retrain BCE | Fisher recomputed ({len(embeddings)} samples, domain={domain})")

    return {
        "skipped": False,
        "val_auc": final_val_auc,
        "train_auc": final_train_auc,
        "val_accuracy": val_accuracy,
        "spearman_rho": spearman_rho,
        "train_size": len(train_embs),
        "val_size": len(val_embs),
        "fail_ratio": n_fail / max(n, 1),
        "best_test_auc": best_val_auc,
        "pass_energy_mean": pass_energy_mean,
        "fail_energy_mean": fail_energy_mean,
        "model": model,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("GEOMETRIC LENS TRAINING — C(x) Cost Field")
    print("=" * 60)

    # Load gate data
    data = load_gate_data()
    print(f"Loaded {len(data['embeddings'])} embeddings, dim={len(data['embeddings'][0])}")

    # Train C(x)
    print("\n--- Training C(x) Cost Field ---")
    cx_result = train_cost_field(data, epochs=200, margin=1.0)

    if cx_result["best_test_auc"] < 0.70:
        print(f"\nWARNING: Test AUC {cx_result['best_test_auc']:.4f} < 0.70 threshold")
        print("C(x) may not generalize well. Proceeding with caution...")

    # Save model
    print("\n--- Saving Model ---")
    save_cost_field(cx_result["model"])

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"C(x) test AUC:    {cx_result['best_test_auc']:.4f} (threshold: 0.70)")
    print(f"C(x) PASS energy: {cx_result['pass_energy_mean']:.4f}")
    print(f"C(x) FAIL energy: {cx_result['fail_energy_mean']:.4f}")
    print("=" * 60)
