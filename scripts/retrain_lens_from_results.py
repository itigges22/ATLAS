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
import sys
import urllib.request

# Internal service auth: cover this script's llama/lens calls.
try:
    import pathlib as _pl
    _tok_path = __import__("os").environ.get(
        "ATLAS_SERVICE_TOKEN_FILE",
        str(_pl.Path(__file__).resolve().parent.parent / "secrets" /
            "service-token"))
    with open(_tok_path) as _fh:
        _svc_tok = _fh.read().strip()
    if _svc_tok:
        _op = urllib.request.build_opener()
        _op.addheaders = [("Authorization", f"Bearer {_svc_tok}")]
        urllib.request.install_opener(_op)
except OSError:
    pass  # no token => auth disabled
import urllib.error

RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "" + ATLAS_DIR + "/benchmark/results/v3_lcb/per_task",
)
def _read_env_var(name: str) -> str:
    """Read one variable from the repo Docker .env ("" when unavailable)."""
    try:
        envp = os.path.join(ATLAS_DIR, ".env")
        if os.path.exists(envp):
            with open(envp, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("export "):
                        line = line[len("export "):].lstrip()
                    if line.startswith(name + "="):
                        value = line.split("=", 1)[1]
                        # Drop a whitespace-preceded inline comment.
                        value = value.lstrip()
                        head, hash_sep, _ = value.partition("#")
                        if hash_sep and head and head[-1] in " \t":
                            value = head
                        value = value.strip().strip('"').strip("'")
                        if value:
                            return value
    except Exception:
        # The compose .env is optional for this standalone helper; callers
        # fall back to their historical defaults when it cannot be read.
        pass
    return ""


def _default_llama_url() -> str:
    """Docker .env host port (ATLAS_LLAMA_PORT) if present, else K3s NodePort."""
    port = _read_env_var("ATLAS_LLAMA_PORT")
    if port:
        return f"http://localhost:{port}"
    return "http://localhost:32735"


def _lens_base_urls() -> list:
    """Candidate lens-service base URLs: Docker .env port (default 8099)
    first, then the K3s NodePort fallback."""
    port = _read_env_var("ATLAS_LENS_PORT") or "8099"
    urls = [f"http://localhost:{port}"]
    if port != "31144":
        urls.append("http://localhost:31144")
    return urls


LLAMA_URL = os.environ.get("LLAMA_URL", _default_llama_url())
SAVE_PATH = os.environ.get(
    "SAVE_PATH",
    "" + ATLAS_DIR + "/geometric-lens/geometric_lens/models/cost_field.pt",
)
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))  # 0 = all


def get_embedding(text: str, url: str) -> list:
    """Get embedding vector from llama-server /embedding endpoint.

    Response format: [{"index": 0, "embedding": [[t0], [t1], ...]}]
    (per-token) or {"embedding": [d0, d1, ...]} (pooled) depending on
    server pooling mode/version.
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

    # Per-token response: mean-pool across token rows, matching what
    # geometric_lens.embedding_extractor.extract_embedding feeds the lens
    # at serve time. (Taking only the first token's row would train C(x)
    # on a different representation than it scores.)
    if emb and isinstance(emb[0], list):
        per_token = emb
        dim = len(per_token[0])
        pooled = [0.0] * dim
        for tok_emb in per_token:
            for i, v in enumerate(tok_emb):
                pooled[i] += v
        n_tokens = len(per_token)
        for i in range(dim):
            pooled[i] /= n_tokens
        emb = pooled
    return emb


def _now_iso() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


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


def request_lens_reload() -> bool:
    """POST /internal/lens/reload so the running service picks up new weights."""
    for base in _lens_base_urls():
        try:
            req = urllib.request.Request(
                f"{base}/internal/lens/reload",
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"   Lens reload via {base}: "
                      f"{resp.read().decode('utf-8')[:200]}")
            return True
        except Exception as e:
            print(f"   Lens reload failed at {base}: {e}")
    print("   Lens reload not applied — POST /internal/lens/reload manually "
          "or restart the geometric-lens service")
    return False


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

    print("\n5. Results:")
    print(f"   Val AUC:      {result.get('val_auc', 0):.4f}")
    print(f"   Train AUC:    {result.get('train_auc', 0):.4f}")
    print(f"   Val accuracy: {result.get('val_accuracy', 0):.2%}")
    print(f"   Pass energy:  {result.get('pass_energy_mean', 0):.4f}")
    print(f"   Fail energy:  {result.get('fail_energy_mean', 0):.4f}")
    print(f"   Model saved:  {args.save_path}")

    if result.get("skipped", False):
        print("   Retrain was skipped (not enough per-class samples) — "
              "calibration and reload skipped too")
        print("=" * 60)
        return

    # Refresh the per-model C(x) calibration beside the new weights so
    # normalized scores track the retrained energy distribution.
    print("\n6. Refreshing C(x) calibration + hot-reloading the lens service")
    from geometric_lens.calibration import (
        derive_cx_normalization, save_cx_normalization,
    )
    try:
        calibration = derive_cx_normalization(
            result["pass_energy_mean"], result["fail_energy_mean"])
        calib_path = save_cx_normalization(
            os.path.dirname(args.save_path) or ".", calibration)
        print(f"   Calibration saved: {calib_path}")
    except ValueError as e:
        print(f"   WARNING: calibration not refreshed: {e}")

    # Stamp the bundle's model identity — the lens load path hard-requires
    # model_identity.json, so a retrained bundle without it fails the
    # identity check on the next service restart and the lens stays off.
    model_name = os.environ.get("ATLAS_MODEL_NAME", "").strip() \
        or _read_env_var("ATLAS_MODEL_NAME")
    if model_name:
        from geometric_lens.identity import save_model_identity
        identity_path = save_model_identity(
            os.path.dirname(args.save_path) or ".", model_name, dim)
        print(f"   Identity saved:    {identity_path} (model={model_name})")
    else:
        print("   WARNING: ATLAS_MODEL_NAME unresolved — "
              "model_identity.json not written; the reloaded bundle will "
              "fail the identity check on restart")

    # Write the per-bundle provenance manifest (SUPPORT_MATRIX §9.5) so
    # the bundle is reproducible and its status is auditable.
    try:
        from geometric_lens.provenance import build_manifest, save_provenance
        save_dir = os.path.dirname(args.save_path) or "."
        manifest = build_manifest(
            model=model_name or "(unknown)", embedding_dim=dim,
            created_at=_now_iso(),
            dataset=os.path.basename(os.path.normpath(args.results_dir)),
            n_samples=len(labels), n_pass=n_pass, n_fail=n_fail,
            metrics={
                "val_auc": result.get("val_auc"),
                "train_auc": result.get("train_auc"),
                "pass_energy_mean": result.get("pass_energy_mean"),
                "fail_energy_mean": result.get("fail_energy_mean"),
            },
            normalization=locals().get("calibration") or {},
            hyperparameters={"epochs": args.epochs},
            seed=42, save_dir=save_dir)
        prov_path = save_provenance(save_dir, manifest)
        print(f"   Provenance saved:  {prov_path}")
    except Exception as e:
        print(f"   WARNING: provenance manifest not written: {e}")

    request_lens_reload()
    print("=" * 60)


if __name__ == "__main__":
    main()
