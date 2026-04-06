#!/usr/bin/env python3
import os
ATLAS_DIR = os.environ.get("ATLAS_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Retrain Geometric Lens C(x) from V3 benchmark results.

Harvests embeddings from the running llama-server for all tasks with code,
then trains C(x) using retrain_cost_field_bce.

Usage:
    python3 scripts/retrain_lens_from_results.py [--results-dir DIR] [--llama-url URL]

Needs:
    - llama-server running with /embedding endpoint
    - V3 benchmark results with per-task JSON files (code + pass/fail)
    - torch (run inside geometric-lens container or a torch-enabled environment)
"""

import json
import os
import struct
import sys
import time
import urllib.request
import urllib.error

RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "" + ATLAS_DIR + "/benchmark/results/v3_full_14b_final/v3_lcb/per_task",
)
LLAMA_URL = os.environ.get("LLAMA_URL", "http://10.42.0.134:8000")
SAVE_PATH = os.environ.get(
    "SAVE_PATH",
    "" + ATLAS_DIR + "/geometric-lens/geometric_lens/models/cost_field.pt",
)
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))  # 0 = all


def get_embedding(text: str, url: str) -> list:
    """Get embedding vector from llama-server /embedding endpoint.

    Response format: [{"index": 0, "embedding": [[d0, d1, ...]]}]
    May also be: {"embedding": [d0, d1, ...]} depending on server version.
    """
    body = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/embedding",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # Handle list response: [{"index": 0, "embedding": [[...]]}]
    if isinstance(data, list) and data:
        emb = data[0].get("embedding", [])
    elif isinstance(data, dict):
        emb = data.get("embedding", [])
    else:
        return []

    # Unwrap nested list: [[d0, d1, ...]] -> [d0, d1, ...]
    if emb and isinstance(emb[0], list):
        emb = emb[0]
    return emb


def load_results(results_dir: str, max_tasks: int = 0) -> tuple:
    """Load pass/fail codes from per-task result files.

    Returns: (codes, labels) where codes=list[str], labels=list[str].
    """
    codes = []
    labels = []
    files = sorted(os.listdir(results_dir))
    for f in files:
        if not f.endswith(".json"):
            continue
        with open(os.path.join(results_dir, f)) as fh:
            d = json.load(fh)
        code = d.get("code")
        if not code:
            continue
        passed = d.get("passed", False)
        codes.append(code)
        labels.append("PASS" if passed else "FAIL")
        if max_tasks and len(codes) >= max_tasks:
            break
    return codes, labels


def harvest_embeddings(codes: list, url: str) -> tuple:
    """Get embeddings for all codes. Returns (embeddings, indices_kept)."""
    embeddings = []
    kept_indices = []
    total = len(codes)
    for i, code in enumerate(codes):
        try:
            emb = get_embedding(code, url)
            if emb and len(emb) > 0:
                embeddings.append(emb)
                kept_indices.append(i)
        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR: {e}")
            continue
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] dim={len(emb) if emb else '?'}")
    return embeddings, kept_indices


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Retrain Lens C(x) from V3 results")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--llama-url", default=LLAMA_URL)
    parser.add_argument("--save-path", default=SAVE_PATH)
    parser.add_argument("--max-tasks", type=int, default=MAX_TASKS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true",
                        help="Load results but skip embedding harvest and training")
    args = parser.parse_args()

    print("=" * 60)
    print("LENS RETRAINING FROM V3 RESULTS")
    print("=" * 60)

    # Load results
    print(f"\n1. Loading results from {args.results_dir}")
    codes, labels = load_results(args.results_dir, args.max_tasks)
    n_pass = sum(1 for l in labels if l == "PASS")
    n_fail = sum(1 for l in labels if l == "FAIL")
    print(f"   Found {len(codes)} tasks with code: {n_pass} PASS, {n_fail} FAIL")

    if n_fail < 5:
        print(f"\n   WARNING: Only {n_fail} FAIL samples. Need >=5 for training.")
        print("   The V3 results only store code for passing tasks.")
        print("   Generating synthetic FAIL samples from passing code...")

        # Create synthetic failures by truncating/corrupting passing code
        import random
        random.seed(42)
        pass_codes = [c for c, l in zip(codes, labels) if l == "PASS"]
        for _ in range(min(50, len(pass_codes))):
            code = random.choice(pass_codes)
            # Truncate to create a "failed" version
            truncated = code[:len(code) // 3]
            codes.append(truncated)
            labels.append("FAIL")
        n_fail = sum(1 for l in labels if l == "FAIL")
        print(f"   After augmentation: {len(codes)} samples ({n_pass} PASS, {n_fail} FAIL)")

    if args.dry_run:
        print("\n   DRY RUN — skipping embedding harvest and training")
        return

    # Harvest embeddings
    print(f"\n2. Harvesting embeddings from {args.llama_url}")
    embeddings, kept = harvest_embeddings(codes, args.llama_url)
    labels_kept = [labels[i] for i in kept]
    dim = len(embeddings[0]) if embeddings else 0
    print(f"   Got {len(embeddings)} embeddings, dim={dim}")
    n_pass = sum(1 for l in labels_kept if l == "PASS")
    n_fail = sum(1 for l in labels_kept if l == "FAIL")
    print(f"   PASS: {n_pass}, FAIL: {n_fail}")

    if n_fail < 5 or n_pass < 5:
        print("\n   ERROR: Not enough samples for training. Aborting.")
        sys.exit(1)

    # Save embeddings for future reuse
    emb_cache = args.save_path.replace("cost_field.pt", f"training_embeddings_{dim}d.json")
    print(f"\n3. Caching embeddings to {emb_cache}")
    with open(emb_cache, "w") as f:
        json.dump({"embeddings": embeddings, "labels": labels_kept, "dim": dim}, f)
    print(f"   Saved {len(embeddings)} embeddings ({os.path.getsize(emb_cache) / 1e6:.1f} MB)")

    # Train
    print(f"\n4. Training C(x) on {dim}-dim embeddings")
    # Import torch here so the script can validate results without torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "geometric-lens"))
    from geometric_lens.training import retrain_cost_field_bce

    result = retrain_cost_field_bce(
        embeddings=embeddings,
        labels=labels_kept,
        epochs=args.epochs,
        save_path=args.save_path,
    )

    print(f"\n5. Results:")
    print(f"   Val AUC:      {result.get('val_auc', 0):.4f}")
    print(f"   Train AUC:    {result.get('train_auc', 0):.4f}")
    print(f"   Val accuracy: {result.get('val_accuracy', 0):.2%}")
    print(f"   Pass energy:  {result.get('pass_energy_mean', 0):.4f}")
    print(f"   Fail energy:  {result.get('fail_energy_mean', 0):.4f}")
    print(f"   Model saved:  {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
