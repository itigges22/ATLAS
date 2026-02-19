"""GO/NO-GO Gate (GL-3.5): Embedding separability analysis.

Tests whether llama-server embeddings separate PASS/FAIL tasks well enough
to justify building a learned cost field C(x).

Gate criterion: AUC > 0.65 on held-out test set.
"""

import json
import logging
import math
import os
import random
import sys
from typing import List, Tuple

# Allow running from rag-api/ or project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric_lens.embedding_extractor import extract_embedding

logger = logging.getLogger(__name__)

# ----- Task definitions -----
# PASS tasks: simple, well-defined functions a 14B model handles reliably
PASS_TASKS = [
    "Write a function that returns the sum of two numbers.",
    "Write a function that reverses a string.",
    "Write a function to check if a number is even.",
    "Write a function that returns the maximum of a list.",
    "Write a function that counts vowels in a string.",
    "Write a function to calculate the factorial of a number.",
    "Write a function that checks if a string is a palindrome.",
    "Write a function to find the length of a list.",
    "Write a function that converts Celsius to Fahrenheit.",
    "Write a function that returns the absolute value of a number.",
    "Write a function that concatenates two strings.",
    "Write a function to check if a number is positive.",
    "Write a function that returns the first element of a list.",
    "Write a function that returns the last element of a list.",
    "Write a function to square a number.",
    "Write a function that checks if a list is empty.",
    "Write a function to find the minimum of two numbers.",
    "Write a function to convert a string to uppercase.",
    "Write a function to convert a string to lowercase.",
    "Write a function that returns True if a number is odd.",
    "Write a function that doubles every element in a list.",
    "Write a function that returns the sum of a list of numbers.",
    "Write a function that checks if a character is a digit.",
    "Write a function that returns 'Hello, World!'.",
    "Write a function that swaps two variables.",
    "Write a function to compute the area of a rectangle given width and height.",
    "Write a function that checks if a string contains a substring.",
    "Write a function that removes whitespace from both ends of a string.",
    "Write a function that returns the index of an element in a list.",
    "Write a function that joins a list of strings with a separator.",
]

# FAIL tasks: complex, multi-step, tricky edge cases — tasks that 14B models
# commonly get wrong on first attempt
FAIL_TASKS = [
    "Write a function that solves the N-Queens problem and returns all valid board configurations as lists of column positions.",
    "Write a function that implements a red-black tree with insert, delete, and rebalance operations maintaining all invariants.",
    "Write a function that parses a mathematical expression string with nested parentheses, operator precedence, and unary minus into an AST and evaluates it.",
    "Write a function that finds the longest common subsequence of three strings using dynamic programming with O(n*m*k) complexity.",
    "Write a function that implements the A* pathfinding algorithm on a weighted graph with obstacles, handling diagonal movement and tie-breaking.",
    "Write a function that converts a regular expression string to an NFA using Thompson's construction, then converts the NFA to a DFA.",
    "Write a function that implements a skip list with probabilistic balancing supporting insert, delete, and range queries.",
    "Write a function that solves a Sudoku puzzle using constraint propagation and backtracking, handling puzzles with multiple solutions.",
    "Write a function that implements the Aho-Corasick string matching algorithm for simultaneous multi-pattern search.",
    "Write a function that computes the convex hull of a set of 2D points handling collinear points and degenerate cases correctly.",
    "Write a function that implements a B-tree of order m with split, merge, and rebalancing on insert and delete.",
    "Write a function that implements the Knuth-Morris-Pratt string matching algorithm with proper failure function computation.",
    "Write a function that finds all strongly connected components in a directed graph using Tarjan's algorithm with proper low-link values.",
    "Write a function that implements a trie with wildcard search supporting '.' (any single char) and '*' (any sequence) patterns.",
    "Write a function that solves the traveling salesman problem using dynamic programming with bitmask for up to 20 cities.",
    "Write a function that implements a concurrent-safe LRU cache with TTL expiration, handling race conditions without deadlocks.",
    "Write a function that parses and evaluates a SQL-like query string supporting SELECT, WHERE, JOIN, GROUP BY, and HAVING clauses on in-memory data.",
    "Write a function that implements the Bellman-Ford algorithm detecting negative weight cycles and reconstructing the shortest path.",
    "Write a function that implements a segment tree with lazy propagation supporting range updates and range queries.",
    "Write a function that computes the edit distance between two strings with support for transposition (Damerau-Levenshtein distance), tracking the actual edit operations.",
    "Write a function that implements a balanced AVL tree with insert, delete, single and double rotations, and in-order traversal.",
    "Write a function that implements Dijkstra's algorithm with a Fibonacci heap for optimal time complexity on sparse graphs.",
    "Write a function that performs topological sorting on a DAG with cycle detection, handling disconnected components and returning all valid orderings.",
    "Write a function that implements the Floyd-Warshall algorithm for all-pairs shortest paths with path reconstruction and negative cycle detection.",
    "Write a function that solves the 0/1 knapsack problem with backtracking to find the actual items selected, not just the optimal value.",
    "Write a function that implements a suffix array with the LCP array for efficient string matching and longest repeated substring.",
    "Write a function that implements the Hungarian algorithm to solve the assignment problem in O(n^3) time.",
    "Write a function that computes the maximum flow in a network using the Edmonds-Karp algorithm with BFS augmentation.",
    "Write a function that implements a persistent data structure (persistent array) supporting versioned read/write operations.",
    "Write a function that implements Hopcroft-Karp algorithm for maximum bipartite matching.",
]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_similarity_stats(
    embeddings: List[List[float]], labels: List[int]
) -> dict:
    """Compute intra-class and inter-class cosine similarity."""
    pass_embs = [e for e, l in zip(embeddings, labels) if l == 1]
    fail_embs = [e for e, l in zip(embeddings, labels) if l == 0]

    # Intra-class similarity
    pass_sims = []
    for i in range(len(pass_embs)):
        for j in range(i + 1, len(pass_embs)):
            pass_sims.append(cosine_similarity(pass_embs[i], pass_embs[j]))

    fail_sims = []
    for i in range(len(fail_embs)):
        for j in range(i + 1, len(fail_embs)):
            fail_sims.append(cosine_similarity(fail_embs[i], fail_embs[j]))

    # Inter-class similarity
    inter_sims = []
    for p in pass_embs:
        for f in fail_embs:
            inter_sims.append(cosine_similarity(p, f))

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def std(lst):
        if len(lst) < 2:
            return 0.0
        m = mean(lst)
        return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))

    return {
        "pass_intra_mean": mean(pass_sims),
        "pass_intra_std": std(pass_sims),
        "fail_intra_mean": mean(fail_sims),
        "fail_intra_std": std(fail_sims),
        "inter_mean": mean(inter_sims),
        "inter_std": std(inter_sims),
        "n_pass": len(pass_embs),
        "n_fail": len(fail_embs),
    }


def train_logistic_regression(
    X_train: List[List[float]],
    y_train: List[int],
    X_test: List[List[float]],
    y_test: List[int],
    lr: float = 0.01,
    epochs: int = 200,
    reg: float = 0.001,
) -> Tuple[float, List[float]]:
    """Train logistic regression from scratch (no sklearn needed).

    Returns (AUC, test_probabilities).
    """
    dim = len(X_train[0])
    n = len(X_train)

    # Initialize weights
    w = [0.0] * dim
    b = 0.0

    for epoch in range(epochs):
        # Forward pass + gradient accumulation
        grad_w = [0.0] * dim
        grad_b = 0.0

        for i in range(n):
            # Linear combination
            z = sum(w[j] * X_train[i][j] for j in range(dim)) + b
            # Sigmoid
            z = max(-500, min(500, z))  # clip for stability
            p = 1.0 / (1.0 + math.exp(-z))
            # Gradient
            err = p - y_train[i]
            for j in range(dim):
                grad_w[j] += err * X_train[i][j] / n + reg * w[j]
            grad_b += err / n

        # Update
        for j in range(dim):
            w[j] -= lr * grad_w[j]
        b -= lr * grad_b

    # Predict on test set
    probs = []
    for x in X_test:
        z = sum(w[j] * x[j] for j in range(dim)) + b
        z = max(-500, min(500, z))
        probs.append(1.0 / (1.0 + math.exp(-z)))

    # Compute AUC using the trapezoidal rule
    auc = compute_auc(y_test, probs)
    return auc, probs


def compute_auc(y_true: List[int], y_scores: List[float]) -> float:
    """Compute AUC-ROC using the trapezoidal rule."""
    # Sort by score descending
    pairs = sorted(zip(y_scores, y_true), reverse=True)

    tp, fp = 0, 0
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for i, (score, label) in enumerate(pairs):
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr

    return auc


def reduce_dimensions_pca(
    embeddings: List[List[float]], n_components: int = 2
) -> List[List[float]]:
    """Simple PCA via power iteration (no numpy needed).

    Returns first n_components principal components for visualization.
    """
    n = len(embeddings)
    dim = len(embeddings[0])

    # Center data
    means = [0.0] * dim
    for emb in embeddings:
        for j in range(dim):
            means[j] += emb[j]
    for j in range(dim):
        means[j] /= n

    centered = [[emb[j] - means[j] for j in range(dim)] for emb in embeddings]

    components = []
    for comp_idx in range(n_components):
        # Power iteration to find top eigenvector
        v = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in v))
        v = [x / norm for x in v]

        for _ in range(100):
            # X^T X v
            new_v = [0.0] * dim
            for i in range(n):
                dot = sum(centered[i][j] * v[j] for j in range(dim))
                for j in range(dim):
                    new_v[j] += centered[i][j] * dot

            norm = math.sqrt(sum(x * x for x in new_v))
            if norm < 1e-10:
                break
            v = [x / norm for x in new_v]

        components.append(v)

        # Deflate: remove this component from data
        for i in range(n):
            dot = sum(centered[i][j] * v[j] for j in range(dim))
            for j in range(dim):
                centered[i][j] -= dot * v[j]

    # Project original centered data
    # Recenter
    centered_orig = [[emb[j] - means[j] for j in range(dim)] for emb in embeddings]
    projected = []
    for emb in centered_orig:
        coords = []
        for v in components:
            coords.append(sum(emb[j] * v[j] for j in range(dim)))
        projected.append(coords)

    return projected


def run_gate_analysis(llama_url: str = None) -> dict:
    """Run the full GO/NO-GO gate analysis.

    Returns dict with AUC, similarity stats, and pass/fail determination.
    """
    if llama_url:
        os.environ["LLAMA_URL"] = llama_url

    print("=" * 60)
    print("GEOMETRIC LENS GO/NO-GO GATE (GL-3.5)")
    print("=" * 60)

    # Collect embeddings
    all_tasks = []
    all_labels = []
    all_embeddings = []

    print(f"\nCollecting {len(PASS_TASKS)} PASS task embeddings...")
    for i, task in enumerate(PASS_TASKS):
        try:
            emb = extract_embedding(task)
            all_tasks.append(task)
            all_labels.append(1)
            all_embeddings.append(emb)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(PASS_TASKS)} done")
        except Exception as e:
            print(f"  FAILED on task {i}: {e}")

    print(f"\nCollecting {len(FAIL_TASKS)} FAIL task embeddings...")
    for i, task in enumerate(FAIL_TASKS):
        try:
            emb = extract_embedding(task)
            all_tasks.append(task)
            all_labels.append(0)
            all_embeddings.append(emb)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(FAIL_TASKS)} done")
        except Exception as e:
            print(f"  FAILED on task {i}: {e}")

    total = len(all_embeddings)
    n_pass = sum(all_labels)
    n_fail = total - n_pass
    print(f"\nTotal embeddings: {total} (PASS={n_pass}, FAIL={n_fail})")

    if total < 20:
        print("ERROR: Too few embeddings collected. Aborting gate.")
        return {"gate_result": "ERROR", "reason": "insufficient_data"}

    # Similarity analysis
    print("\n--- Similarity Analysis ---")
    sim_stats = compute_similarity_stats(all_embeddings, all_labels)
    print(f"PASS intra-class similarity: {sim_stats['pass_intra_mean']:.4f} ± {sim_stats['pass_intra_std']:.4f}")
    print(f"FAIL intra-class similarity: {sim_stats['fail_intra_mean']:.4f} ± {sim_stats['fail_intra_std']:.4f}")
    print(f"Inter-class similarity:      {sim_stats['inter_mean']:.4f} ± {sim_stats['inter_std']:.4f}")

    separation = (sim_stats['pass_intra_mean'] + sim_stats['fail_intra_mean']) / 2 - sim_stats['inter_mean']
    print(f"Separation gap (intra - inter): {separation:.4f}")

    # Train/test split (stratified)
    pass_indices = [i for i, l in enumerate(all_labels) if l == 1]
    fail_indices = [i for i, l in enumerate(all_labels) if l == 0]
    random.seed(42)
    random.shuffle(pass_indices)
    random.shuffle(fail_indices)

    # 70/30 split
    n_pass_train = int(0.7 * len(pass_indices))
    n_fail_train = int(0.7 * len(fail_indices))

    train_idx = pass_indices[:n_pass_train] + fail_indices[:n_fail_train]
    test_idx = pass_indices[n_pass_train:] + fail_indices[n_fail_train:]

    X_train = [all_embeddings[i] for i in train_idx]
    y_train = [all_labels[i] for i in train_idx]
    X_test = [all_embeddings[i] for i in test_idx]
    y_test = [all_labels[i] for i in test_idx]

    print(f"\nTrain: {len(X_train)} (PASS={sum(y_train)}, FAIL={len(y_train)-sum(y_train)})")
    print(f"Test:  {len(X_test)} (PASS={sum(y_test)}, FAIL={len(y_test)-sum(y_test)})")

    # Logistic regression on full embeddings (5120-dim)
    print("\n--- Logistic Regression (full 5120-dim) ---")
    auc_full, probs_full = train_logistic_regression(
        X_train, y_train, X_test, y_test, lr=0.001, epochs=300
    )
    print(f"AUC (full dim): {auc_full:.4f}")

    # Also try on PCA-reduced embeddings (50-dim)
    print("\n--- PCA Reduction ---")
    pca_50 = reduce_dimensions_pca(all_embeddings, n_components=50)
    X_train_pca = [pca_50[i] for i in train_idx]
    X_test_pca = [pca_50[i] for i in test_idx]

    auc_pca, probs_pca = train_logistic_regression(
        X_train_pca, y_train, X_test_pca, y_test, lr=0.01, epochs=500
    )
    print(f"AUC (PCA-50): {auc_pca:.4f}")

    # Use best AUC
    best_auc = max(auc_full, auc_pca)
    best_method = "full_5120" if auc_full >= auc_pca else "pca_50"

    # PCA-2D for visualization report
    pca_2d = reduce_dimensions_pca(all_embeddings, n_components=2)

    # Gate decision
    print("\n" + "=" * 60)
    print(f"BEST AUC: {best_auc:.4f} (method: {best_method})")

    if best_auc > 0.65:
        gate_result = "PASS"
        print("GATE RESULT: ✓ PASS — Proceed to Phase 2")
    elif best_auc >= 0.55:
        gate_result = "MARGINAL"
        print("GATE RESULT: ~ MARGINAL — Proceed cautiously")
    else:
        gate_result = "FAIL"
        print("GATE RESULT: ✗ FAIL — Embeddings do not separate")
    print("=" * 60)

    result = {
        "gate_result": gate_result,
        "best_auc": best_auc,
        "best_method": best_method,
        "auc_full": auc_full,
        "auc_pca50": auc_pca,
        "similarity": sim_stats,
        "separation_gap": separation,
        "n_total": total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "pca_2d": pca_2d,
        "labels": all_labels,
        "embeddings": all_embeddings,
        "tasks": all_tasks,
    }

    # Save results to JSON (without large embeddings for the report)
    report = {k: v for k, v in result.items() if k not in ("embeddings", "pca_2d")}
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gate_report.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # Save embeddings + labels for training if gate passes
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gate_embeddings.json"
    )
    with open(data_path, "w") as f:
        json.dump(
            {"embeddings": all_embeddings, "labels": all_labels, "tasks": all_tasks},
            f,
        )
    print(f"Embeddings saved to {data_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GL-3.5 gate analysis")
    parser.add_argument("--llama-url", default=None, help="llama-server URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_gate_analysis(llama_url=args.llama_url)
