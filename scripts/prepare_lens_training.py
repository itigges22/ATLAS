#!/usr/bin/env python3
"""Prepare Geometric Lens training data from benchmark embeddings.

Reads the binary .emb file from a benchmark run and converts to the
JSON format expected by geometric_lens/training.py.

Usage:
    python3 scripts/prepare_lens_training.py benchmark/results/<run_id>/telemetry/embeddings.emb

Output: writes gate_embeddings.json to geometric-lens/geometric_lens/
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.v3.embedding_store import EmbeddingReader


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/prepare_lens_training.py <embeddings.emb> [output.json]")
        sys.exit(1)

    emb_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "geometric-lens/geometric_lens/gate_embeddings.json"

    reader = EmbeddingReader(emb_path)
    records = reader.read_all()

    # Filter to PASS/FAIL only (skip UNKNOWN)
    labeled = [r for r in records if r["label"] in ("PASS", "FAIL")]

    embeddings = [r["embedding"] for r in labeled]
    labels = [1 if r["label"] == "PASS" else 0 for r in labeled]

    n_pass = sum(labels)
    n_fail = len(labels) - n_pass
    dim = len(embeddings[0]) if embeddings else 0

    print(f"Total records: {len(records)}")
    print(f"Labeled (PASS/FAIL): {len(labeled)}")
    print(f"  PASS: {n_pass}")
    print(f"  FAIL: {n_fail}")
    print(f"  Embedding dim: {dim}")

    if len(labeled) < 10:
        print("WARNING: Very few labeled samples. Lens training needs 100+ of each class.")

    data = {
        "embeddings": embeddings,
        "labels": labels,
        "metadata": {
            "source": emb_path,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "dim": dim,
        }
    }

    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"Written to: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
