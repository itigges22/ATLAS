"""Geometric Lens continuous learning utilities for the benchmark runner."""

import json
import math
import random
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EMBEDDING_DIM = 5120


# --- Embedding extraction -----------------------------------------------------

def extract_embedding_urllib(text: str, llama_url: str) -> Optional[List[float]]:
    """
    Extract a mean-pooled embedding from llama-server's /embedding endpoint.

    The endpoint returns a list of per-token embedding vectors. This function
    averages across all tokens to produce a single 5120-dim vector.

    Args:
        text: Input text to embed.
        llama_url: Base URL for llama-server (e.g. "http://localhost:8080").

    Returns:
        List of floats (length EMBEDDING_DIM), or None on failure.
    """
    body = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{llama_url}/embedding",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None

    # Response format: [{"index": 0, "embedding": [[d0, d1, ...], [d0, d1, ...], ...]}]
    try:
        token_vectors = data[0]["embedding"]
    except (KeyError, IndexError, TypeError):
        return None

    if not token_vectors:
        return None

    # If the embedding is already a flat vector (single-token or pre-pooled),
    # return it directly.
    if not isinstance(token_vectors[0], list):
        return token_vectors

    n_tokens = len(token_vectors)
    n_dims = len(token_vectors[0])
    pooled = [0.0] * n_dims
    for vec in token_vectors:
        for i, v in enumerate(vec):
            pooled[i] += v
    for i in range(n_dims):
        pooled[i] /= n_tokens

    return pooled


# --- Spearman rank correlation ------------------------------------------------

def _normal_cdf(x: float) -> float:
    """
    Approximate the standard normal CDF using the Abramowitz & Stegun formula.

    Handbook of Mathematical Functions, formula 26.2.17 (max error ~7.5e-8).
    """
    if x < 0:
        return 1.0 - _normal_cdf(-x)

    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    t = 1.0 / (1.0 + p * x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    cdf = 1.0 - pdf * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5)
    return cdf


def _assign_ranks(values: List[float]) -> List[float]:
    """
    Assign ranks using the average-rank method for tied values.

    Args:
        values: The data values to rank.

    Returns:
        List of ranks (1-based), with ties receiving the average of their positions.
    """
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        # Find the end of the group of tied values
        j = i + 1
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1

        # Average rank for positions i..j-1 (1-based)
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j

    return ranks


def compute_spearman_rho(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Compute the Spearman rank correlation coefficient and two-tailed p-value.

    Uses the average-rank method for handling ties. The p-value is computed via
    the t-distribution approximation converted through the normal CDF (valid
    for n >= 10, reasonable approximation for smaller n).

    Args:
        x: First variable (list of numeric values).
        y: Second variable (list of numeric values, same length as x).

    Returns:
        Tuple of (rho, p_value).

    Raises:
        ValueError: If x and y have different lengths or length < 2.
    """
    n = len(x)
    if n != len(y):
        raise ValueError(f"x and y must have the same length, got {n} and {len(y)}")
    if n < 2:
        raise ValueError(f"Need at least 2 data points, got {n}")

    rx = _assign_ranks(x)
    ry = _assign_ranks(y)

    # Compute Pearson correlation on the ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for i in range(n):
        dx = rx[i] - mean_rx
        dy = ry[i] - mean_ry
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy

    denom = math.sqrt(den_x * den_y)
    if denom == 0.0:
        return (0.0, 1.0)

    rho = num / denom

    # Clamp to [-1, 1] to avoid floating-point issues in sqrt
    rho = max(-1.0, min(1.0, rho))

    # p-value via t-distribution approximation: t = rho * sqrt((n-2)/(1-rho^2))
    if abs(rho) >= 1.0:
        p_value = 0.0
    elif n <= 2:
        p_value = 1.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
        # Two-tailed p-value using normal CDF approximation
        p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))

    return (rho, p_value)


# --- Learning curve tracker ---------------------------------------------------

class LearningCurveTracker:
    """
    Tracks training embeddings and epoch-level statistics for continuous learning.

    Manages JSONL telemetry files and produces a learning curve summary that
    records how model accuracy evolves across epochs of incremental retraining.

    Args:
        run_dir: Root directory for this benchmark run.
    """

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.telemetry_dir = self.run_dir / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_file = self.telemetry_dir / "training_embeddings.jsonl"
        self.epochs: List[Dict] = []

    def record_embedding(
        self,
        task_id: str,
        embedding: List[float],
        label: str,
        epoch: int,
    ) -> None:
        """
        Append a training embedding record to the JSONL telemetry file.

        Args:
            task_id: Identifier for the benchmark task.
            embedding: The embedding vector (list of floats).
            label: "PASS" or "FAIL".
            epoch: The epoch number this embedding belongs to.
        """
        record = {
            "task_id": task_id,
            "embedding": embedding,
            "label": label,
            "epoch": epoch,
        }
        with open(self.embeddings_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_training_data(self) -> List[Dict]:
        """
        Read all embedding records from the JSONL telemetry file.

        Returns:
            List of dicts, each containing task_id, embedding, label, epoch.
        """
        records = []
        if not self.embeddings_file.exists():
            return records
        with open(self.embeddings_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def count_labels(self, max_epoch: Optional[int] = None) -> Tuple[int, int]:
        """
        Count pass/fail labels in the training data, optionally up to a max epoch.

        Args:
            max_epoch: If provided, only count records with epoch <= max_epoch.

        Returns:
            Tuple of (n_pass, n_fail).
        """
        n_pass = 0
        n_fail = 0
        for record in self.load_training_data():
            if max_epoch is not None and record.get("epoch", 0) > max_epoch:
                continue
            if record.get("label") == "PASS":
                n_pass += 1
            else:
                n_fail += 1
        return (n_pass, n_fail)

    def record_epoch(
        self,
        epoch: int,
        total: int,
        passed: int,
        retrain_metrics: Optional[Dict] = None,
    ) -> None:
        """
        Store epoch-level statistics.

        Args:
            epoch: Epoch number.
            total: Total tasks evaluated in this epoch.
            passed: Number of tasks that passed.
            retrain_metrics: Optional dict of retraining metrics (loss, AUC, etc.).
        """
        entry = {
            "epoch": epoch,
            "total_tasks": total,
            "passed_tasks": passed,
            "pass_rate": passed / max(total, 1),
        }
        if retrain_metrics is not None:
            entry["retrain_metrics"] = retrain_metrics
        self.epochs.append(entry)

    def prepare_retrain_payload(
        self, max_epoch: Optional[int] = None
    ) -> List[Dict]:
        """
        Build a list of {embedding, label} dicts suitable for retraining.

        Args:
            max_epoch: If provided, only include records with epoch <= max_epoch.

        Returns:
            List of dicts with "embedding" and "label" keys.
        """
        payload = []
        for record in self.load_training_data():
            if max_epoch is not None and record.get("epoch", 0) > max_epoch:
                continue
            payload.append({
                "embedding": record["embedding"],
                "label": record["label"],
            })
        return payload

    def save_summary(self) -> None:
        """
        Write the learning curve summary to run_dir/telemetry/learning_curve.json.

        The summary includes all epoch-level statistics and cumulative label counts.
        """
        n_pass, n_fail = self.count_labels()
        summary = {
            "total_pass": n_pass,
            "total_fail": n_fail,
            "total_samples": n_pass + n_fail,
            "epochs": self.epochs,
        }
        summary_path = self.telemetry_dir / "learning_curve.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


# --- Epoch splitting ----------------------------------------------------------

def shuffle_and_split_epochs(
    tasks: list, seed: int = 42
) -> List[list]:
    """
    Shuffle tasks deterministically and split into 5 epochs.

    Epoch sizes: [100, 200, 200, 200, remainder]. If there are fewer tasks
    than needed for all 5 epochs, later epochs are empty lists.

    Args:
        tasks: List of tasks to split.
        seed: Random seed for reproducible shuffling.

    Returns:
        List of 5 task lists, one per epoch.
    """
    shuffled = list(tasks)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    boundaries = [100, 300, 500, 700]
    epochs: List[list] = []
    for i in range(5):
        if i == 0:
            start = 0
            end = min(boundaries[0], len(shuffled))
        elif i < 4:
            start = boundaries[i - 1]
            end = min(boundaries[i], len(shuffled))
        else:
            start = boundaries[3]
            end = len(shuffled)

        if start >= len(shuffled):
            epochs.append([])
        else:
            epochs.append(shuffled[start:end])

    return epochs
