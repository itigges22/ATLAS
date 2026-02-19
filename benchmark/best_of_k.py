"""Best-of-K Lens Selection utilities for the benchmark runner."""

import json
import math
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def score_candidate(text: str, rag_api_url: str) -> Tuple[float, float]:
    """Score candidate text through the Geometric Lens.

    Args:
        text: Full text to score (typically "TASK: {prompt}\\n\\nSOLUTION: {response}").
        rag_api_url: Base URL for rag-api (e.g. "http://localhost:31144").

    Returns:
        Tuple of (raw_energy, normalized_energy). Returns (999.0, 1.0) on failure.
    """
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{rag_api_url}/internal/lens/score-text",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("energy", 999.0), data.get("normalized", 1.0))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return (999.0, 1.0)


def get_temperature(k: int, eval_mode: str = "function") -> float:
    """Return sampling temperature for best-of-k generation.

    Code tasks (function/stdio) need higher temperature for diverse solutions.
    Non-code tasks (mcq/ifbench) need lower temperature to keep reasoning
    coherent — too high and the model degenerates into terse answers.
    """
    if k <= 1:
        return 0.0
    if eval_mode in ("mcq", "ifbench"):
        return 0.3
    elif k <= 5:
        return 0.6
    else:
        return 0.8


class BestOfKTracker:
    """Tracks best-of-k selection statistics for benchmark telemetry.

    Records per-task selection events and computes aggregate metrics
    for the benchmark report.

    Args:
        run_dir: Root directory for this benchmark run.
    """

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.telemetry_dir = self.run_dir / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.telemetry_dir / "best_of_k_events.jsonl"
        self.events: List[Dict] = []

    def record_event(
        self,
        task_id: str,
        k: int,
        candidates: List[Dict],
        selected_index: int,
        sandbox_calls: int,
        selected_passed: bool,
        oracle_has_pass: bool,
    ) -> None:
        """Record a single best-of-k selection event.

        Args:
            task_id: Benchmark task identifier.
            k: Number of candidates generated.
            candidates: List of dicts with keys: energy, passed (bool), index, code_hash.
            selected_index: Index of the lens-selected candidate (0-based in sorted order).
            sandbox_calls: Number of sandbox executions before finding PASS (or k if none).
            selected_passed: Whether the selected candidate passed.
            oracle_has_pass: Whether any candidate passed (oracle pass@k).
        """
        energies = [c["energy"] for c in candidates]
        pass_energies = [c["energy"] for c in candidates if c.get("passed")]
        fail_energies = [c["energy"] for c in candidates if not c.get("passed")]
        n_unique = len(set(c.get("code_hash", "") for c in candidates))

        event = {
            "task_id": task_id,
            "k": k,
            "selected_index": selected_index,
            "sandbox_calls": sandbox_calls,
            "selected_passed": selected_passed,
            "oracle_has_pass": oracle_has_pass,
            "n_unique_solutions": n_unique,
            "energies": energies,
            "energy_mean": sum(energies) / max(len(energies), 1),
            "energy_std": _std(energies),
            "pass_energy_mean": sum(pass_energies) / max(len(pass_energies), 1) if pass_energies else None,
            "fail_energy_mean": sum(fail_energies) / max(len(fail_energies), 1) if fail_energies else None,
        }
        self.events.append(event)

        with open(self.events_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def get_summary(self) -> Dict:
        """Compute aggregate statistics from all recorded events.

        Returns:
            Dict with selection_accuracy, oracle_pass_rate, effective_pass_rate,
            avg_sandbox_calls, avg_energy_std, avg_unique_solutions, and
            energy_distribution sub-dict.
        """
        if not self.events:
            return {}

        n = len(self.events)
        has_pass_events = [e for e in self.events if e["oracle_has_pass"]]
        n_has_pass = len(has_pass_events)

        lens_picked_passer = sum(1 for e in has_pass_events if e["selected_passed"])
        # NOTE: selection_accuracy is always 1.0 because the sandbox tries
        # candidates in energy order and early-exits on first pass. The more
        # meaningful metric is first_pick_accuracy (did rank-0 candidate pass).
        selection_accuracy = lens_picked_passer / max(n_has_pass, 1)
        first_pick_passed = sum(1 for e in has_pass_events if e["sandbox_calls"] == 1)
        first_pick_accuracy = first_pick_passed / max(n_has_pass, 1)
        oracle_pass_rate = n_has_pass / max(n, 1)
        effective_pass_rate = sum(1 for e in self.events if e["selected_passed"]) / max(n, 1)

        sandbox_events = [e for e in has_pass_events if e["selected_passed"]]
        avg_sandbox_calls = (
            sum(e["sandbox_calls"] for e in sandbox_events) / max(len(sandbox_events), 1)
            if sandbox_events else 0
        )

        all_pass_energies = []
        all_fail_energies = []
        all_selected_energies = []
        all_rejected_energies = []
        for e in self.events:
            if e.get("pass_energy_mean") is not None:
                all_pass_energies.append(e["pass_energy_mean"])
            if e.get("fail_energy_mean") is not None:
                all_fail_energies.append(e["fail_energy_mean"])
            if e["energies"]:
                all_selected_energies.append(e["energies"][0])
                all_rejected_energies.extend(e["energies"][1:])

        return {
            "total_tasks": n,
            "tasks_with_pass_candidate": n_has_pass,
            "lens_picked_passer": lens_picked_passer,
            "selection_accuracy": selection_accuracy,
            "first_pick_accuracy": first_pick_accuracy,
            "first_pick_passed": first_pick_passed,
            "oracle_pass_rate": oracle_pass_rate,
            "effective_pass_rate": effective_pass_rate,
            "avg_sandbox_calls": avg_sandbox_calls,
            "avg_energy_std": sum(e["energy_std"] for e in self.events) / max(n, 1),
            "avg_unique_solutions": sum(e["n_unique_solutions"] for e in self.events) / max(n, 1),
            "energy_distribution": {
                "selected_mean": sum(all_selected_energies) / max(len(all_selected_energies), 1) if all_selected_energies else 0,
                "selected_std": _std(all_selected_energies),
                "rejected_mean": sum(all_rejected_energies) / max(len(all_rejected_energies), 1) if all_rejected_energies else 0,
                "rejected_std": _std(all_rejected_energies),
                "pass_mean": sum(all_pass_energies) / max(len(all_pass_energies), 1) if all_pass_energies else 0,
                "pass_std": _std(all_pass_energies),
                "fail_mean": sum(all_fail_energies) / max(len(all_fail_energies), 1) if all_fail_energies else 0,
                "fail_std": _std(all_fail_energies),
            },
        }

    def save_summary(self) -> None:
        """Write aggregated summary to telemetry/best_of_k_summary.json."""
        summary = self.get_summary()
        summary_path = self.telemetry_dir / "best_of_k_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


def _std(values: list) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
