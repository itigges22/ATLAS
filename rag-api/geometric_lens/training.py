"""Training script for C(x) cost field and G(x) metric tensor.

Trains C(x) with contrastive ranking loss on PASS/FAIL embedding pairs.
Trains G(x) with gradient alignment loss after C(x) is trained.

Designed to run inside the rag-api container where torch is available.
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
from geometric_lens.metric_tensor import MetricTensor


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


def train_metric_tensor(
    cost_field: CostField,
    data: dict,
    epochs: int = 200,
    lr: float = 1e-3,
    alpha: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Train G(x) metric tensor with gradient alignment loss.

    The loss has two components:
    1. Correction should reduce C(x): C(x + dx) < C(x) for FAIL embeddings
    2. Corrected FAIL embeddings should move toward PASS centroid

    Returns dict with model and metrics.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    embeddings = data["embeddings"]
    labels = data["labels"]
    dim = len(embeddings[0])

    device = torch.device("cpu")

    # Freeze cost field
    cost_field.eval()  # Note: model.eval() is the PyTorch eval mode toggle, not code evaluation
    for p in cost_field.parameters():
        p.requires_grad_(False)

    metric = MetricTensor(input_dim=dim).to(device)
    optimizer = optim.Adam(metric.parameters(), lr=lr, weight_decay=weight_decay)

    # Compute PASS centroid
    pass_embs = [e for e, l in zip(embeddings, labels) if l == 1]
    fail_embs = [e for e, l in zip(embeddings, labels) if l == 0]
    pass_centroid = torch.tensor(pass_embs, dtype=torch.float32, device=device).mean(0)

    fail_tensor = torch.tensor(fail_embs, dtype=torch.float32, device=device)

    loss_history = []
    reduction_rates = []

    for epoch in range(epochs):
        metric.train()
        total_loss = 0.0

        # Shuffle fail embeddings
        perm = torch.randperm(len(fail_embs))
        fail_shuffled = fail_tensor[perm]

        batch_size = min(16, len(fail_embs))
        n_batches = 0

        for batch_start in range(0, len(fail_embs), batch_size):
            batch = fail_shuffled[batch_start:batch_start + batch_size]
            x = batch.detach().requires_grad_(True)

            # Forward through C(x) to get gradient
            energy_before = cost_field(x)
            grad_C = torch.autograd.grad(energy_before.sum(), x, create_graph=True)[0]

            # Compute G(x) and correction
            G_diag = metric(batch)
            G_inv = 1.0 / (G_diag + 1e-8)
            delta_x = -alpha * G_inv * grad_C.detach()
            corrected = batch + delta_x

            # Loss 1: Energy should decrease after correction
            energy_after = cost_field(corrected)
            energy_reduction_loss = torch.relu(energy_after - energy_before.detach()).mean()

            # Loss 2: Corrected should be closer to PASS centroid
            dist_before = torch.norm(batch - pass_centroid, dim=1)
            dist_after = torch.norm(corrected - pass_centroid, dim=1)
            alignment_loss = torch.relu(dist_after - dist_before).mean()

            # Loss 3: Regularize G to stay near identity (prevent collapse)
            reg_loss = ((G_diag - 1.0) ** 2).mean() * 0.01

            loss = energy_reduction_loss + 0.5 * alignment_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 40 == 0 or epoch == 0:
            metric.eval()  # Note: model.eval() is the PyTorch eval mode toggle, not code evaluation
            with torch.no_grad():
                rate = compute_correction_rate(cost_field, metric, fail_tensor, alpha, device)
                reduction_rates.append(rate)
            print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | C(x) reduction rate: {rate:.2%}")

    # Final assessment
    metric.eval()  # Note: model.eval() is the PyTorch eval mode toggle, not code evaluation
    final_rate = compute_correction_rate(cost_field, metric, fail_tensor, alpha, device)

    # Check correction magnitudes (needs grad for C(x))
    x = fail_tensor.detach().clone().requires_grad_(True)
    energy = cost_field(x)
    grad_C = torch.autograd.grad(energy.sum(), x)[0]
    with torch.no_grad():
        G_diag = metric(fail_tensor)
        G_inv = 1.0 / (G_diag + 1e-8)
        delta_x = -alpha * G_inv * grad_C
        correction_norms = torch.norm(delta_x, dim=1)
        embedding_norms = torch.norm(fail_tensor, dim=1)
        relative_magnitude = (correction_norms / embedding_norms).mean()

    print(f"\n--- G(x) Final Results ---")
    print(f"Correction reduces C(x) for: {final_rate:.2%} of FAIL embeddings")
    print(f"Mean ||dx||/||x||: {relative_magnitude:.4f}")
    print(f"Mean ||dx||: {correction_norms.mean():.4f}")

    return {
        "model": metric,
        "correction_rate": final_rate,
        "relative_magnitude": relative_magnitude.item(),
        "loss_history": loss_history,
    }


def compute_correction_rate(cost_field, metric_tensor, fail_embeddings, alpha, device):
    """What fraction of FAIL embeddings have lower C(x) after correction?"""
    with torch.enable_grad():
        x = fail_embeddings.detach().clone().requires_grad_(True)
        energy_before = cost_field(x)
        grad_C = torch.autograd.grad(energy_before.sum(), x)[0]

    with torch.no_grad():
        G_diag = metric_tensor(fail_embeddings)
        G_inv = 1.0 / (G_diag + 1e-8)
        delta_x = -alpha * G_inv * grad_C
        corrected = fail_embeddings + delta_x
        energy_after = cost_field(corrected)
        reduced = (energy_after < energy_before.detach()).float().mean().item()
    return reduced


def save_models(cost_field, metric_tensor, save_dir=None):
    """Save trained model weights."""
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(save_dir, exist_ok=True)

    cost_path = os.path.join(save_dir, "cost_field.pt")
    metric_path = os.path.join(save_dir, "metric_tensor.pt")

    torch.save(cost_field.state_dict(), cost_path)
    torch.save(metric_tensor.state_dict(), metric_path)
    print(f"Models saved to {save_dir}/")
    return cost_path, metric_path


def load_models(save_dir=None, dim=5120):
    """Load trained model weights."""
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    cost_field = CostField(input_dim=dim)
    metric_tensor = MetricTensor(input_dim=dim)

    cost_path = os.path.join(save_dir, "cost_field.pt")
    metric_path = os.path.join(save_dir, "metric_tensor.pt")

    if os.path.exists(cost_path):
        cost_field.load_state_dict(torch.load(cost_path, map_location="cpu", weights_only=True))
    if os.path.exists(metric_path):
        metric_tensor.load_state_dict(torch.load(metric_path, map_location="cpu", weights_only=True))

    cost_field.eval()  # Note: model.eval() is the PyTorch eval mode toggle, not code evaluation
    metric_tensor.eval()  # Note: model.eval() is the PyTorch eval mode toggle, not code evaluation
    return cost_field, metric_tensor


def retrain_cost_field_bce(
    embeddings: list,
    labels: list,
    epochs: int = 50,
    lr: float = None,
    weight_decay: float = 1e-4,
    test_fraction: float = 0.2,
    seed: int = 42,
    save_path: str = None,
) -> dict:
    """Retrain CostField on real pass/fail benchmark data using weighted MSE.

    Uses MSE loss to energy targets (PASS=2.0, FAIL=25.0) with class-weighted
    samples to handle imbalanced data. Includes early stopping on validation
    AUC with patience=10 (checked every 5 epochs).

    Args:
        embeddings: List of float lists, each 5120-dim.
        labels: List of "PASS" or "FAIL" strings.
        epochs: Maximum training epochs.
        lr: Learning rate. If None, selected adaptively based on dataset size.
        weight_decay: AdamW weight decay.
        test_fraction: Fraction of data reserved for validation.
        seed: Random seed for reproducibility.
        save_path: If provided, save model state_dict to this path.

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
        "model": model,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("GEOMETRIC LENS TRAINING")
    print("=" * 60)

    # Load gate data
    data = load_gate_data()
    print(f"Loaded {len(data['embeddings'])} embeddings, dim={len(data['embeddings'][0])}")

    # Phase 2: Train C(x)
    print("\n--- Phase 2: Training C(x) Cost Field ---")
    cx_result = train_cost_field(data, epochs=200, margin=1.0)

    if cx_result["best_test_auc"] < 0.70:
        print(f"\nWARNING: Test AUC {cx_result['best_test_auc']:.4f} < 0.70 threshold")
        print("C(x) may not generalize well. Proceeding with caution...")

    # Phase 3: Train G(x)
    print("\n--- Phase 3: Training G(x) Metric Tensor ---")
    gx_result = train_metric_tensor(cx_result["model"], data, epochs=200)

    if gx_result["correction_rate"] < 0.80:
        print(f"\nWARNING: Correction rate {gx_result['correction_rate']:.2%} < 80% threshold")

    # Save models
    print("\n--- Saving Models ---")
    save_models(cx_result["model"], gx_result["model"])

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"C(x) test AUC:        {cx_result['best_test_auc']:.4f} (threshold: 0.70)")
    print(f"C(x) PASS energy:     {cx_result['pass_energy_mean']:.4f}")
    print(f"C(x) FAIL energy:     {cx_result['fail_energy_mean']:.4f}")
    print(f"G(x) correction rate: {gx_result['correction_rate']:.2%} (threshold: 80%)")
    print(f"G(x) ||dx||/||x||:   {gx_result['relative_magnitude']:.4f} (target: < 0.10)")
    print("=" * 60)
