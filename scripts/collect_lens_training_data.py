#!/usr/bin/env python3
"""Collect Geometric Lens training data from completed benchmark results.

Post-hoc script: reads per-task JSON results, extracts code, calls the
embedding endpoint, and creates the training JSON for C(x).

This is more reliable than inline embedding (which competes with generation
for the single GPU slot) because it runs AFTER the benchmark completes.

Usage:
    python3 scripts/collect_lens_training_data.py benchmark/results/<run_id>/

Output: geometric-lens/geometric_lens/gate_embeddings.json
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
import glob

LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:32735")


def extract_embedding(text: str, max_retries: int = 5) -> list:
    """Extract embedding from llama-server with retries."""
    body = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{LLAMA_URL}/embedding",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            emb = data[0]["embedding"][0] if isinstance(data, list) else data.get("embedding", [])
            if isinstance(emb[0], list):
                emb = emb[0]
            return emb
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/{max_retries} (wait {wait}s): {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {max_retries} retries: {e}")
                return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/collect_lens_training_data.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1].rstrip("/")
    output_path = sys.argv[2] if len(sys.argv) > 2 else "geometric-lens/geometric_lens/gate_embeddings.json"

    # Find per-task JSON files
    task_files = sorted(glob.glob(f"{results_dir}/v3_lcb/per_task/*.json"))
    if not task_files:
        # Try without v3_lcb subdirectory
        task_files = sorted(glob.glob(f"{results_dir}/per_task/*.json"))

    if not task_files:
        print(f"No task files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(task_files)} task results")

    embeddings = []
    labels = []
    skipped = 0

    for i, f in enumerate(task_files):
        d = json.load(open(f))
        task_id = d.get("task_id", os.path.basename(f))
        code = d.get("code", "")
        passed = d.get("passed", False)

        if not code or len(code) < 10:
            print(f"  [{i+1}/{len(task_files)}] {task_id}: SKIP (no code)")
            skipped += 1
            continue

        label = 1 if passed else 0
        status = "PASS" if passed else "FAIL"

        # Build the same text format as score_candidate
        task_prompt = d.get("task_prompt", "")  # might not be stored
        text = f"SOLUTION: {code}"

        print(f"  [{i+1}/{len(task_files)}] {task_id}: {status} ({len(code)} chars)...", end=" ", flush=True)

        emb = extract_embedding(text)
        if emb is None:
            print("FAILED")
            skipped += 1
            continue

        embeddings.append(emb)
        labels.append(label)
        print(f"OK (dim={len(emb)})")

        # Small delay to avoid overwhelming the server
        time.sleep(0.5)

    n_pass = sum(labels)
    n_fail = len(labels) - n_pass
    dim = len(embeddings[0]) if embeddings else 0

    print(f"\n=== Summary ===")
    print(f"Total: {len(embeddings)} (PASS={n_pass}, FAIL={n_fail})")
    print(f"Skipped: {skipped}")
    print(f"Embedding dim: {dim}")

    if n_pass < 5 or n_fail < 5:
        print("WARNING: Too few samples for reliable training!")

    data = {
        "embeddings": embeddings,
        "labels": labels,
        "metadata": {
            "source": results_dir,
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
