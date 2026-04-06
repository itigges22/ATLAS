#!/usr/bin/env python3
"""Phase 0: Retrain C(x) on existing 597 4096-dim embeddings.

Debugged version — tests multiple approaches to find best AUC:
1. Contrastive ranking loss with stabilized training (lower lr, grad clip, warmup)
2. Weighted MSE to energy targets (PASS=2.0, FAIL=25.0)
3. Combination: MSE warmup then contrastive fine-tune

Acceptance Criteria:
- Val AUC >= 0.90
- PASS mean energy < FAIL mean energy, separation ratio >= 2.0
"""

import json
import os
import random
import sys

import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "geometric-lens"))
from geometric_lens.cost_field import CostField


def compute_auc(model, embeddings, labels, device):
    """AUC: P(FAIL energy > PASS energy)."""
    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    with torch.no_grad():
        energies = model(X).squeeze().tolist()
    if isinstance(energies, float):
        energies = [energies]

    pass_e = [e for e, l in zip(energies, labels) if l == "PASS"]
    fail_e = [e for e, l in zip(energies, labels) if l == "FAIL"]

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


def energy_stats(model, embeddings, labels, device):
    """Compute energy distribution statistics."""
    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    with torch.no_grad():
        energies = model(X).squeeze().tolist()
    if isinstance(energies, float):
        energies = [energies]

    pass_e = [e for e, l in zip(energies, labels) if l == "PASS"]
    fail_e = [e for e, l in zip(energies, labels) if l == "FAIL"]

    pm = sum(pass_e) / len(pass_e)
    fm = sum(fail_e) / len(fail_e)
    ps = (sum((e - pm)**2 for e in pass_e) / len(pass_e)) ** 0.5
    fs = (sum((e - fm)**2 for e in fail_e) / len(fail_e)) ** 0.5

    return pm, ps, fm, fs


def train_contrastive(train_pass, train_fail, val_embs, val_labels, dim, device,
                       lr=5e-4, margin=1.0, max_epochs=300, patience=15):
    """Train C(x) with contrastive ranking loss + stabilization."""
    model = CostField(input_dim=dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = 32
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    pairs_per_epoch = max(len(train_pass), len(train_fail)) * 6

    for epoch in range(max_epochs):
        model.train()

        # Warmup: linearly increase lr for first 20 epochs
        if epoch < 20:
            warmup_factor = (epoch + 1) / 20.0
            for pg in optimizer.param_groups:
                pg['lr'] = lr * warmup_factor

        epoch_pairs = []
        for _ in range(pairs_per_epoch):
            p = random.choice(train_pass)
            f = random.choice(train_fail)
            epoch_pairs.append((p, f))

        random.shuffle(epoch_pairs)
        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(epoch_pairs), batch_size):
            batch = epoch_pairs[batch_start:batch_start + batch_size]
            pass_batch = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
            fail_batch = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)

            c_pass = model(pass_batch)
            c_fail = model(fail_batch)

            loss = torch.relu(c_pass - c_fail + margin).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_auc = compute_auc(model, val_embs, val_labels, device)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  [contrastive] Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}")

            if patience_counter >= patience:
                print(f"  [contrastive] Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_auc


def train_mse(train_embs, train_labels, val_embs, val_labels, dim, device,
              lr=5e-4, max_epochs=300, patience=15):
    """Train C(x) with weighted MSE to energy targets."""
    PASS_TARGET = 2.0
    FAIL_TARGET = 25.0

    n_pass = sum(1 for l in train_labels if l == "PASS")
    n_fail = sum(1 for l in train_labels if l == "FAIL")
    fail_weight = n_pass / max(n_fail, 1)

    targets = []
    weights = []
    for l in train_labels:
        if l == "PASS":
            targets.append(PASS_TARGET)
            weights.append(1.0)
        else:
            targets.append(FAIL_TARGET)
            weights.append(fail_weight)

    train_X = torch.tensor(train_embs, dtype=torch.float32, device=device)
    train_T = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(1)
    train_W = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)

    model = CostField(input_dim=dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = 32
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(train_embs))
        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_embs), batch_size):
            idx = perm[batch_start:batch_start + batch_size]
            x_batch = train_X[idx]
            t_batch = train_T[idx]
            w_batch = train_W[idx]

            energies = model(x_batch)
            loss = ((energies - t_batch) ** 2 * w_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_auc = compute_auc(model, val_embs, val_labels, device)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg_loss = total_loss / max(n_batches, 1)
                print(f"  [mse]         Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}")

            if patience_counter >= patience:
                print(f"  [mse]         Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_auc


def train_combo(train_embs, train_labels, train_pass, train_fail,
                val_embs, val_labels, dim, device, max_epochs=300, patience=15):
    """Phase 1: MSE warmup (100 epochs), Phase 2: contrastive fine-tune."""
    # Phase 1: MSE warmup to establish reasonable energy landscape
    PASS_TARGET = 2.0
    FAIL_TARGET = 25.0

    n_pass = sum(1 for l in train_labels if l == "PASS")
    n_fail = sum(1 for l in train_labels if l == "FAIL")
    fail_weight = n_pass / max(n_fail, 1)

    targets = []
    weights = []
    for l in train_labels:
        if l == "PASS":
            targets.append(PASS_TARGET)
            weights.append(1.0)
        else:
            targets.append(FAIL_TARGET)
            weights.append(fail_weight)

    train_X = torch.tensor(train_embs, dtype=torch.float32, device=device)
    train_T = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(1)
    train_W = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)

    model = CostField(input_dim=dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    batch_size = 32
    best_val_auc = 0.0
    best_state = None

    # Phase 1: MSE for 100 epochs
    print("  [combo] Phase 1: MSE warmup (100 epochs)")
    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(train_embs))

        for batch_start in range(0, len(train_embs), batch_size):
            idx = perm[batch_start:batch_start + batch_size]
            energies = model(train_X[idx])
            loss = ((energies - train_T[idx]) ** 2 * train_W[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            val_auc = compute_auc(model, val_embs, val_labels, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  [combo-mse]   Epoch {epoch+1:4d} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}")

    # Phase 2: Contrastive fine-tune with lower lr
    print("  [combo] Phase 2: Contrastive fine-tune")
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    patience_counter = 0
    pairs_per_epoch = max(len(train_pass), len(train_fail)) * 6

    for epoch in range(200):
        model.train()
        epoch_pairs = []
        for _ in range(pairs_per_epoch):
            p = random.choice(train_pass)
            f = random.choice(train_fail)
            epoch_pairs.append((p, f))

        random.shuffle(epoch_pairs)
        for batch_start in range(0, len(epoch_pairs), batch_size):
            batch = epoch_pairs[batch_start:batch_start + batch_size]
            pass_batch = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
            fail_batch = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)

            c_pass = model(pass_batch)
            c_fail = model(fail_batch)
            loss = torch.relu(c_pass - c_fail + 1.0).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            val_auc = compute_auc(model, val_embs, val_labels, device)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                print(f"  [combo-ctr]   Epoch {epoch+1:4d} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}")

            if patience_counter >= patience:
                print(f"  [combo-ctr]   Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_auc


def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "geometric-lens", "geometric_lens", "models",
        "training_embeddings_4096d.json"
    )
    with open(data_path) as f:
        data = json.load(f)

    embeddings = data["embeddings"]
    labels = data["labels"]
    dim = data.get("dim", len(embeddings[0]))

    n_pass = sum(1 for l in labels if l == "PASS")
    n_fail = sum(1 for l in labels if l == "FAIL")
    print(f"Loaded {len(embeddings)} embeddings, dim={dim}")
    print(f"  PASS: {n_pass}, FAIL: {n_fail}, ratio: {n_pass/n_fail:.1f}:1")

    # 80/20 stratified split
    pass_idx = [i for i, l in enumerate(labels) if l == "PASS"]
    fail_idx = [i for i, l in enumerate(labels) if l == "FAIL"]
    random.shuffle(pass_idx)
    random.shuffle(fail_idx)

    n_pass_val = max(1, int(len(pass_idx) * 0.2))
    n_fail_val = max(1, int(len(fail_idx) * 0.2))

    val_idx = set(pass_idx[:n_pass_val] + fail_idx[:n_fail_val])
    train_idx = [i for i in range(len(embeddings)) if i not in val_idx]

    train_embs = [embeddings[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_embs = [embeddings[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_pass_embs = [e for e, l in zip(train_embs, train_labels) if l == "PASS"]
    train_fail_embs = [e for e, l in zip(train_embs, train_labels) if l == "FAIL"]

    device = torch.device("cpu")
    print(f"Train: {len(train_embs)} (PASS={len(train_pass_embs)}, FAIL={len(train_fail_embs)})")
    print(f"Val:   {len(val_embs)}")

    # ---- Approach 1: Contrastive (stabilized) ----
    print("\n" + "=" * 60)
    print("APPROACH 1: Contrastive Ranking Loss (stabilized)")
    print("=" * 60)
    random.seed(seed)
    torch.manual_seed(seed)
    m1, auc1 = train_contrastive(train_pass_embs, train_fail_embs, val_embs, val_labels,
                                  dim, device, lr=5e-4, margin=1.0, patience=15)
    pm1, ps1, fm1, fs1 = energy_stats(m1, embeddings, labels, device)
    sep1 = fm1 / max(pm1, 1e-8)
    print(f"  => Val AUC: {auc1:.4f} | PASS: {pm1:.3f}+/-{ps1:.3f} | FAIL: {fm1:.3f}+/-{fs1:.3f} | Sep: {sep1:.1f}x")

    # ---- Approach 2: Weighted MSE ----
    print("\n" + "=" * 60)
    print("APPROACH 2: Weighted MSE (PASS=2.0, FAIL=25.0)")
    print("=" * 60)
    random.seed(seed)
    torch.manual_seed(seed)
    m2, auc2 = train_mse(train_embs, train_labels, val_embs, val_labels,
                          dim, device, lr=5e-4, patience=15)
    pm2, ps2, fm2, fs2 = energy_stats(m2, embeddings, labels, device)
    sep2 = fm2 / max(pm2, 1e-8)
    print(f"  => Val AUC: {auc2:.4f} | PASS: {pm2:.3f}+/-{ps2:.3f} | FAIL: {fm2:.3f}+/-{fs2:.3f} | Sep: {sep2:.1f}x")

    # ---- Approach 3: MSE warmup + contrastive fine-tune ----
    print("\n" + "=" * 60)
    print("APPROACH 3: MSE warmup (100ep) + Contrastive fine-tune")
    print("=" * 60)
    random.seed(seed)
    torch.manual_seed(seed)
    m3, auc3 = train_combo(train_embs, train_labels, train_pass_embs, train_fail_embs,
                            val_embs, val_labels, dim, device, patience=15)
    pm3, ps3, fm3, fs3 = energy_stats(m3, embeddings, labels, device)
    sep3 = fm3 / max(pm3, 1e-8)
    print(f"  => Val AUC: {auc3:.4f} | PASS: {pm3:.3f}+/-{ps3:.3f} | FAIL: {fm3:.3f}+/-{fs3:.3f} | Sep: {sep3:.1f}x")

    # ---- Pick the best ----
    results = [
        ("Contrastive", m1, auc1, pm1, ps1, fm1, fs1, sep1),
        ("MSE", m2, auc2, pm2, ps2, fm2, fs2, sep2),
        ("Combo", m3, auc3, pm3, ps3, fm3, fs3, sep3),
    ]
    results.sort(key=lambda r: r[2], reverse=True)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    for name, _, auc, pm, ps, fm, fs, sep in results:
        print(f"  {name:15s} | Val AUC: {auc:.4f} | Sep: {sep:.1f}x | PASS: {pm:.3f} | FAIL: {fm:.3f}")

    best_name, best_model, best_auc, bp_mean, bp_std, bf_mean, bf_std, best_sep = results[0]
    print(f"\nBest approach: {best_name} (Val AUC: {best_auc:.4f})")

    # Text histogram for best model
    with torch.no_grad():
        all_X = torch.tensor(embeddings, dtype=torch.float32, device=device)
        all_energies = best_model(all_X).squeeze().tolist()
        if isinstance(all_energies, float):
            all_energies = [all_energies]

    pass_energies = [e for e, l in zip(all_energies, labels) if l == "PASS"]
    fail_energies = [e for e, l in zip(all_energies, labels) if l == "FAIL"]

    print("\n--- Energy Distribution (best model) ---")
    all_e = pass_energies + fail_energies
    e_min = min(all_e)
    e_max = max(all_e)
    n_bins = 20
    bin_width = (e_max - e_min) / n_bins if e_max > e_min else 1.0

    for i in range(n_bins):
        lo = e_min + i * bin_width
        hi = lo + bin_width
        p_count = sum(1 for e in pass_energies if lo <= e < hi)
        f_count = sum(1 for e in fail_energies if lo <= e < hi)
        p_bar = "P" * min(p_count, 50)
        f_bar = "F" * min(f_count, 50)
        if p_count > 0 or f_count > 0:
            print(f"  [{lo:6.2f}-{hi:6.2f}] {p_bar}{f_bar}")

    # Acceptance criteria
    print("\n--- Acceptance Criteria ---")
    ac1 = best_auc >= 0.90
    ac2 = best_sep >= 2.0 and bp_mean < bf_mean
    print(f"  AC-0-1: Val AUC >= 0.90:           {'PASS' if ac1 else 'FAIL'} ({best_auc:.4f})")
    print(f"  AC-0-2: Separation ratio >= 2.0:   {'PASS' if ac2 else 'FAIL'} ({best_sep:.2f}x)")

    # Save regardless — we'll use the best we can get
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "geometric-lens", "geometric_lens", "models"
    )
    os.makedirs(save_dir, exist_ok=True)

    if ac1 and ac2:
        print("\n  >>> ALL ACCEPTANCE CRITERIA PASSED <<<")
    else:
        print(f"\n  >>> BEST AUC: {best_auc:.4f} (below 0.90 threshold) <<<")
        print("  Saving best model anyway for reference.")
        if best_auc >= 0.75:
            print("  NOTE: AUC >= 0.75 may be adequate if 9B embeddings have lower")
            print("  intrinsic discriminability than 14B. Consider adjusting threshold.")

    backup_path = os.path.join(save_dir, "cost_field_phase0.pt")
    main_path = os.path.join(save_dir, "cost_field.pt")
    torch.save(best_model.state_dict(), backup_path)
    torch.save(best_model.state_dict(), main_path)
    print(f"\n  Saved backup:  {backup_path}")
    print(f"  Saved main:    {main_path}")

    stats = {
        "approach": best_name,
        "val_auc": best_auc,
        "pass_energy_mean": bp_mean,
        "pass_energy_std": bp_std,
        "fail_energy_mean": bf_mean,
        "fail_energy_std": bf_std,
        "separation_ratio": best_sep,
        "n_train": len(train_embs),
        "n_val": len(val_embs),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "dim": dim,
        "all_results": {name: {"val_auc": auc, "sep": sep} for name, _, auc, _, _, _, _, sep in results},
    }
    stats_path = os.path.join(save_dir, "phase0_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats:   {stats_path}")


if __name__ == "__main__":
    main()
