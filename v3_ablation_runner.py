#!/usr/bin/env python3
"""ATLAS V3 Ablation Study Runner
==================================
5-Condition ablation isolating each V3 phase's contribution to LCB pass@1.

Conditions:
  A: Baseline     — All V3 OFF (V2-style single gen, should match ~37%)
  B: +Phase1      — PlanSearch + DivSampling + BudgetForcing, k=3
  C: +Phase1+2    — + Lens-driven adaptive K, ReASC, S*
  D: +Phase1+3    — + PR-CoT repair/refinement (IMPORT from v3_full_14b_final)
  E: Full V3      — Everything active (fresh run with Lens)

Usage:
    # Full ablation (all conditions)
    nohup python3 v3_ablation_runner.py 2>&1 | tee v3_ablation.log &

    # Specific conditions only
    python3 v3_ablation_runner.py --conditions A B C

    # Import existing + analysis only
    python3 v3_ablation_runner.py --analysis-only

    # Smoke test (10 tasks)
    python3 v3_ablation_runner.py --smoke --conditions A B
"""

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Force line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.config import config
from benchmark.datasets import LiveCodeBenchDataset
from benchmark.models import BenchmarkTask
from benchmark.v3_runner import (
    V3BenchmarkRunner,
    atomic_write_json,
    find_completed_tasks,
    load_lcb_tasks,
    RAG_API_URL,
    LLAMA_URL,
)

# Import cache manager (hyphen in filename requires importlib)
_cache_mgr_path = PROJECT_ROOT / "scripts" / "llama-cache-manager.py"
_spec = importlib.util.spec_from_file_location("llama_cache_manager", str(_cache_mgr_path))
_cache_mgr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cache_mgr)
check_and_manage = _cache_mgr.check_and_manage


# ===========================================================================
# Constants
# ===========================================================================

ALL_CONDITIONS = ["A", "B", "C", "D", "E"]

CONDITION_DEFS = {
    "A": {
        "name": "Baseline",
        "description": "All V3 features OFF — V2-style single generation",
        "enable_phase1": False,
        "enable_phase2": False,
        "enable_phase3": False,
        "enable_feedback": False,
    },
    "B": {
        "name": "+Phase1",
        "description": "PlanSearch + DivSampling + BudgetForcing (k=3, no repair)",
        "enable_phase1": True,
        "enable_phase2": False,
        "enable_phase3": False,
        "enable_feedback": False,
    },
    "C": {
        "name": "+Phase1+2",
        "description": "Phase 1 + Lens-driven adaptive K, ReASC, S* tiebreaking + feedback",
        "enable_phase1": True,
        "enable_phase2": True,
        "enable_phase3": False,
        "enable_feedback": True,
    },
    "D": {
        "name": "+Phase1+3",
        "description": "Phase 1 + PR-CoT repair/refinement (no adaptive K)",
        "enable_phase1": True,
        "enable_phase2": False,
        "enable_phase3": True,
        "enable_feedback": False,
    },
    "E": {
        "name": "Full V3",
        "description": "All phases active with Geometric Lens + feedback",
        "enable_phase1": True,
        "enable_phase2": True,
        "enable_phase3": True,
        "enable_feedback": True,
    },
}

CONDITION_DIRS = {
    "A": "condition_a_baseline",
    "B": "condition_b_phase1",
    "C": "condition_c_phase1_2",
    "D": "condition_d_phase1_3",
    "E": "condition_e_full",
}

# Path to import for Condition D
V3_FULL_RESULTS = config.results_dir / "v3_full_14b_final" / "v3_lcb" / "per_task"

SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://localhost:30820")

# Max retries when llama-server dies mid-condition
MAX_SERVER_RETRIES = 10
SERVER_POLL_INTERVAL = 15  # seconds between health checks
SERVER_WAIT_TIMEOUT = 300  # 5 minutes max wait for server recovery


# ===========================================================================
# Server Health (thin wrappers — heavy lifting in cache manager)
# ===========================================================================

def wait_for_server(url: str = LLAMA_URL, timeout: int = SERVER_WAIT_TIMEOUT,
                    poll_interval: int = SERVER_POLL_INTERVAL) -> bool:
    """Poll llama-server /health until it responds OK or timeout."""
    start = time.time()
    attempts = 0
    while time.time() - start < timeout:
        attempts += 1
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if data.get('status') == 'ok':
                    log(f"  llama-server back online after {attempts} polls "
                        f"({int(time.time() - start)}s)")
                    return True
        except Exception:
            pass
        time.sleep(poll_interval)
    return False


def manage_server_memory() -> Dict[str, Any]:
    """Check memory and take action (erase slots or restart) via cache manager.

    Returns dict with: action, memory_before_mb, memory_after_mb, success.
    """
    result = check_and_manage()
    action = result.get("action", "none")
    if action != "none":
        mem_before = result.get("memory_before_mb")
        mem_after = result.get("memory_after_mb")
        log(f"  Cache manager: {action} "
            f"({mem_before}MB -> {mem_after}MB, "
            f"success={result.get('success')})")
    return result


# ===========================================================================
# Logger
# ===========================================================================

class Logger:
    """Timestamped logging with elapsed time and ETA."""

    def __init__(self):
        self._start = time.time()

    def log(self, msg: str, level: str = "INFO"):
        elapsed = timedelta(seconds=int(time.time() - self._start))
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{elapsed}] [{level}] {msg}", flush=True)

    def log_eta(self, done: int, total: int, label: str = ""):
        if done == 0:
            return
        elapsed = time.time() - self._start
        per_item = elapsed / done
        remaining = (total - done) * per_item
        eta = timedelta(seconds=int(remaining))
        prefix = f"{label}: " if label else ""
        self.log(f"{prefix}{done}/{total} ({done/total*100:.1f}%) — ETA {eta}")

    def elapsed_str(self) -> str:
        return str(timedelta(seconds=int(time.time() - self._start)))


logger = Logger()
log = logger.log


# ===========================================================================
# PreflightChecker
# ===========================================================================

class PreflightChecker:
    """Validate infrastructure before ablation."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results = {}

    def run(self, need_lens: bool = True) -> bool:
        """Run all preflight checks. Returns True if all critical checks pass."""
        log("=" * 60)
        log("PRE-FLIGHT CHECKS")
        log("=" * 60)

        ok = True

        # 1. llama-server
        try:
            req = urllib.request.Request(f"{LLAMA_URL}/health")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            status = data.get('status', 'unknown')
            self.results['llama_server'] = {'ok': True, 'status': status}
            log(f"  llama-server: OK ({status})")
        except Exception as e:
            self.results['llama_server'] = {'ok': False, 'error': str(e)}
            log(f"  llama-server: FAILED ({e})", "ERROR")
            ok = False

        # 2. RAG API
        try:
            req = urllib.request.Request(f"{RAG_API_URL}/health")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            self.results['rag_api'] = {'ok': True, 'status': data.get('status', '?')}
            log(f"  RAG API: OK ({data.get('status', '?')})")
        except Exception as e:
            self.results['rag_api'] = {'ok': False, 'error': str(e)}
            log(f"  RAG API: WARNING ({e})", "WARN")
            if need_lens:
                ok = False

        # 3. Lens model (for conditions C and E)
        if need_lens:
            try:
                body = json.dumps({"text": "def hello(): return 42"}).encode("utf-8")
                req = urllib.request.Request(
                    f"{RAG_API_URL}/internal/lens/score-text",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    lens_data = json.loads(resp.read().decode('utf-8'))
                if lens_data.get("error"):
                    self.results['lens'] = {'ok': False, 'error': lens_data['error']}
                    log(f"  Lens model: NOT LOADED ({lens_data['error']})", "WARN")
                else:
                    energy = lens_data.get('energy', '?')
                    self.results['lens'] = {'ok': True, 'energy': energy}
                    log(f"  Lens model: OK (energy={energy})")
            except Exception as e:
                self.results['lens'] = {'ok': False, 'error': str(e)}
                log(f"  Lens model: UNAVAILABLE ({e})", "WARN")

        # 4. Sandbox
        try:
            req = urllib.request.Request(f"{SANDBOX_URL}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
            self.results['sandbox'] = {'ok': True}
            log(f"  Sandbox: OK")
        except Exception as e:
            # Sandbox health may not exist; try execute endpoint
            self.results['sandbox'] = {'ok': True, 'note': 'health endpoint unavailable'}
            log(f"  Sandbox: ASSUMED OK (no health endpoint)")

        # 5. Disk space
        try:
            stat = os.statvfs(str(self.output_dir))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            self.results['disk'] = {'ok': free_gb > 1.0, 'free_gb': round(free_gb, 1)}
            if free_gb < 1.0:
                log(f"  Disk: LOW ({free_gb:.1f} GB free)", "WARN")
            else:
                log(f"  Disk: OK ({free_gb:.1f} GB free)")
        except Exception:
            self.results['disk'] = {'ok': True, 'note': 'check failed'}

        # 6. V3 full results for import (Condition D)
        if V3_FULL_RESULTS.exists():
            count = len(list(V3_FULL_RESULTS.glob("*.json")))
            self.results['v3_full_import'] = {'ok': count > 0, 'task_count': count}
            log(f"  V3 full results (import): {count} task files")
        else:
            self.results['v3_full_import'] = {'ok': False, 'error': 'not found'}
            log(f"  V3 full results: NOT FOUND at {V3_FULL_RESULTS}", "WARN")

        # Save preflight results
        atomic_write_json(self.output_dir / "preflight.json", {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'all_ok': ok,
            'checks': self.results,
        })

        log(f"  Overall: {'PASS' if ok else 'FAIL'}")
        return ok


# ===========================================================================
# ConditionRunner
# ===========================================================================

class ConditionRunner:
    """Execute a single ablation condition using V3BenchmarkRunner."""

    def __init__(self, condition: str, output_dir: Path, tasks: List[BenchmarkTask]):
        self.condition = condition
        self.cond_def = CONDITION_DEFS[condition]
        self.cond_dir = output_dir / CONDITION_DIRS[condition]
        self.tasks = tasks

    def find_completed(self) -> Set[str]:
        """Find already-completed task IDs for crash recovery."""
        per_task_dir = self.cond_dir / "v3_lcb" / "per_task"
        if not per_task_dir.exists():
            return set()
        completed = set()
        for f in per_task_dir.glob("*.json"):
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                    if 'task_id' in data:
                        completed.add(data['task_id'])
            except (json.JSONDecodeError, IOError):
                pass
        return completed

    def load_results(self) -> Dict[str, Dict]:
        """Load all existing per-task results."""
        per_task_dir = self.cond_dir / "v3_lcb" / "per_task"
        results = {}
        if not per_task_dir.exists():
            return results
        for f in per_task_dir.glob("*.json"):
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                    if 'task_id' in data:
                        results[data['task_id']] = data
            except (json.JSONDecodeError, IOError):
                pass
        return results

    def import_results(self, source_dir: Path) -> Dict[str, Dict]:
        """Import results from an existing run (for Condition D)."""
        log(f"  Importing from {source_dir}")
        per_task_dir = self.cond_dir / "v3_lcb" / "per_task"
        per_task_dir.mkdir(parents=True, exist_ok=True)

        imported = {}
        source_files = sorted(source_dir.glob("*.json"))
        for f in source_files:
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                if 'task_id' not in data:
                    continue
                # Tag as imported
                data['imported_from'] = str(source_dir)
                # Write to condition directory
                safe_name = data['task_id'].replace('/', '_')
                atomic_write_json(per_task_dir / f"{safe_name}.json", data)
                imported[data['task_id']] = data
            except (json.JSONDecodeError, IOError) as e:
                log(f"    Failed to import {f.name}: {e}", "WARN")

        # Save summary
        passed = sum(1 for r in imported.values() if r.get("passed"))
        summary = {
            "condition": self.condition,
            "name": self.cond_def["name"],
            "total_tasks": len(imported),
            "passed_tasks": passed,
            "pass_rate": passed / max(len(imported), 1),
            "imported": True,
            "source": str(source_dir),
            "phase_breakdown": _phase_breakdown(imported),
        }
        atomic_write_json(self.cond_dir / "summary.json", summary)
        log(f"  Imported {len(imported)} tasks ({passed} passed, "
            f"{passed/max(len(imported),1)*100:.1f}%)")
        return imported

    def run(self) -> Dict[str, Dict]:
        """Run the condition through V3BenchmarkRunner.

        Resilient to llama-server OOM crashes: if the runner fails mid-run,
        waits for the server to come back and resumes from saved checkpoints.
        Uses the cache manager for memory management between retries.
        """
        self.cond_dir.mkdir(parents=True, exist_ok=True)
        completed = self.find_completed()

        if len(completed) >= len(self.tasks):
            log(f"  Condition {self.condition} already complete ({len(completed)} tasks)")
            return self.load_results()

        if completed:
            log(f"  Resuming Condition {self.condition}: {len(completed)}/{len(self.tasks)} done")

        start = time.time()
        results = None

        for retry in range(MAX_SERVER_RETRIES):
            try:
                with V3BenchmarkRunner(
                    run_dir=self.cond_dir,
                    enable_phase1=self.cond_def["enable_phase1"],
                    enable_phase2=self.cond_def["enable_phase2"],
                    enable_phase3=self.cond_def["enable_phase3"],
                    enable_feedback=self.cond_def.get("enable_feedback", False),
                ) as runner:
                    results = runner.run_lcb(self.tasks)

                # Check if too many tasks errored (sign of server death mid-run)
                error_count = sum(
                    1 for r in results.values()
                    if r.get("phase_solved") == "error"
                )
                if error_count > 20:
                    log(f"  {error_count} error tasks detected — likely server crash",
                        "WARN")
                    self._purge_error_results()
                    raise RuntimeError(
                        f"Too many errors ({error_count}) — server likely died"
                    )

                break  # Success

            except Exception as e:
                if retry >= MAX_SERVER_RETRIES - 1:
                    log(f"  Condition {self.condition} FAILED after "
                        f"{MAX_SERVER_RETRIES} retries: {e}", "ERROR")
                    results = self.load_results()
                    break

                log(f"  Run interrupted (attempt {retry + 1}/"
                    f"{MAX_SERVER_RETRIES}): {e}", "WARN")
                log(f"  Waiting for llama-server to recover...")

                if wait_for_server():
                    log(f"  Server recovered — running cache manager check...")
                    manage_server_memory()
                    log(f"  Retrying condition {self.condition}...")
                else:
                    log(f"  Server did not recover within "
                        f"{SERVER_WAIT_TIMEOUT}s — forcing restart via cache manager",
                        "WARN")
                    result = check_and_manage(hard_mb=0)  # Force restart
                    if result.get("success"):
                        wait_for_server()
                        log(f"  Retrying condition {self.condition}...")
                    else:
                        log(f"  Server restart failed", "ERROR")
                        results = self.load_results()
                        break

        if results is None:
            results = self.load_results()

        elapsed = time.time() - start

        # Save condition summary
        passed = sum(1 for r in results.values() if r.get("passed"))
        total_tokens = sum(r.get("total_tokens", 0) for r in results.values())
        summary = {
            "condition": self.condition,
            "name": self.cond_def["name"],
            "description": self.cond_def["description"],
            "enable_phase1": self.cond_def["enable_phase1"],
            "enable_phase2": self.cond_def["enable_phase2"],
            "enable_phase3": self.cond_def["enable_phase3"],
            "enable_feedback": self.cond_def.get("enable_feedback", False),
            "total_tasks": len(results),
            "passed_tasks": passed,
            "pass_rate": passed / max(len(results), 1),
            "total_tokens": total_tokens,
            "wall_time_s": elapsed,
            "phase_breakdown": _phase_breakdown(results),
        }
        atomic_write_json(self.cond_dir / "summary.json", summary)

        return results

    def _purge_error_results(self):
        """Remove per-task results that have phase_solved='error'.

        These are artifacts of llama-server dying mid-run. Removing them
        lets crash recovery re-process those tasks on retry.
        """
        per_task_dir = self.cond_dir / "v3_lcb" / "per_task"
        if not per_task_dir.exists():
            return
        purged = 0
        for f in per_task_dir.glob("*.json"):
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                if data.get("phase_solved") == "error":
                    f.unlink()
                    purged += 1
            except (json.JSONDecodeError, IOError):
                pass
        if purged:
            log(f"  Purged {purged} error result files for retry")


def _phase_breakdown(results: Dict[str, Dict]) -> Dict[str, int]:
    """Compute breakdown of which phase solved each task."""
    breakdown = {
        "phase1": 0, "pr_cot": 0, "refinement": 0,
        "derivation": 0, "none": 0, "error": 0,
    }
    for r in results.values():
        phase = r.get("phase_solved", "none")
        if phase in breakdown:
            breakdown[phase] += 1
        else:
            breakdown["none"] += 1
    return breakdown


# ===========================================================================
# AblationAnalyzer
# ===========================================================================

class AblationAnalyzer:
    """Cross-condition analysis (pure computation, no LLM calls)."""

    def __init__(self, output_dir: Path, condition_results: Dict[str, Dict[str, Dict]]):
        self.output_dir = output_dir
        self.ablation_dir = output_dir / "ablation"
        self.ablation_dir.mkdir(parents=True, exist_ok=True)
        # condition_results: {"A": {task_id: result, ...}, "B": {...}, ...}
        self.results = condition_results

    def run_all(self):
        """Run all analysis and save to ablation/ directory."""
        log("=" * 60)
        log("CROSS-CONDITION ANALYSIS")
        log("=" * 60)

        self._condition_comparison()
        self._phase_deltas()
        self._difficulty_quartiles()
        self._phase3_rescue()
        self._phase2_efficiency()
        self._v2_comparison()
        self._token_efficiency()
        self._task_concordance()

        log("Analysis complete")

    def _condition_comparison(self):
        """Per-condition pass rate, phase breakdown, and token stats."""
        log("  Computing condition comparison...")
        comparison = {}
        for cond in sorted(self.results.keys()):
            res = self.results[cond]
            total = len(res)
            passed = sum(1 for r in res.values() if r.get("passed"))
            total_tok = sum(r.get("total_tokens", 0) for r in res.values())
            total_time = sum(r.get("total_time_ms", 0) for r in res.values())

            comparison[cond] = {
                "name": CONDITION_DEFS[cond]["name"],
                "total_tasks": total,
                "passed": passed,
                "pass_rate": passed / max(total, 1),
                "total_tokens": total_tok,
                "avg_tokens_per_task": total_tok / max(total, 1),
                "total_time_ms": total_time,
                "phase_breakdown": _phase_breakdown(res),
            }

        atomic_write_json(self.ablation_dir / "condition_comparison.json", comparison)
        return comparison

    def _phase_deltas(self):
        """Pairwise comparisons: A->B, B->C, B->D, A->E with task-level gained/lost."""
        log("  Computing phase deltas...")
        pairs = [
            ("A", "B", "Phase 1 contribution"),
            ("B", "C", "Phase 2 contribution (on top of P1)"),
            ("B", "D", "Phase 3 contribution (on top of P1)"),
            ("A", "E", "Full V3 vs baseline"),
            ("C", "E", "Phase 3 contribution (on top of P1+P2)"),
            ("D", "E", "Phase 2 contribution (on top of P1+P3)"),
        ]

        deltas = []
        for from_c, to_c, desc in pairs:
            if from_c not in self.results or to_c not in self.results:
                continue
            from_res = self.results[from_c]
            to_res = self.results[to_c]

            # Task-level comparison
            all_tasks = set(from_res.keys()) | set(to_res.keys())
            gained = []  # Failed in from, passed in to
            lost = []    # Passed in from, failed in to
            both_pass = 0
            both_fail = 0

            for tid in all_tasks:
                fr = from_res.get(tid, {}).get("passed", False)
                tr = to_res.get(tid, {}).get("passed", False)
                if not fr and tr:
                    gained.append(tid)
                elif fr and not tr:
                    lost.append(tid)
                elif fr and tr:
                    both_pass += 1
                else:
                    both_fail += 1

            from_rate = sum(1 for r in from_res.values() if r.get("passed")) / max(len(from_res), 1)
            to_rate = sum(1 for r in to_res.values() if r.get("passed")) / max(len(to_res), 1)

            deltas.append({
                "from": from_c,
                "to": to_c,
                "description": desc,
                "from_pass_rate": from_rate,
                "to_pass_rate": to_rate,
                "delta_pp": (to_rate - from_rate) * 100,
                "tasks_gained": len(gained),
                "tasks_lost": len(lost),
                "both_pass": both_pass,
                "both_fail": both_fail,
                "gained_task_ids": gained[:50],  # Cap to avoid huge files
                "lost_task_ids": lost[:50],
            })

        atomic_write_json(self.ablation_dir / "phase_deltas.json", deltas)
        return deltas

    def _difficulty_quartiles(self):
        """Analyze pass rates by Lens energy quartiles (conditions C/E that have Lens)."""
        log("  Computing difficulty quartiles...")
        quartile_data = {}

        for cond in ["C", "E"]:
            if cond not in self.results:
                continue
            res = self.results[cond]

            # Collect tasks with energy data from telemetry
            tasks_with_energy = []
            for tid, r in res.items():
                energy = r.get("telemetry", {}).get("lens_energy")
                if energy is None:
                    energy = r.get("telemetry", {}).get("blend_asc_energy")
                if energy is not None:
                    tasks_with_energy.append({
                        "task_id": tid,
                        "energy": energy,
                        "passed": r.get("passed", False),
                    })

            if len(tasks_with_energy) < 4:
                quartile_data[cond] = {"note": "insufficient energy data"}
                continue

            # Sort by energy and split into quartiles
            tasks_with_energy.sort(key=lambda x: x["energy"])
            n = len(tasks_with_energy)
            q_size = n // 4

            quartiles = {}
            for qi, label in enumerate(["Q1 (easy)", "Q2", "Q3", "Q4 (hard)"]):
                start = qi * q_size
                end = start + q_size if qi < 3 else n
                q_tasks = tasks_with_energy[start:end]
                q_pass = sum(1 for t in q_tasks if t["passed"])
                energy_vals = [t["energy"] for t in q_tasks]
                quartiles[label] = {
                    "count": len(q_tasks),
                    "passed": q_pass,
                    "pass_rate": q_pass / max(len(q_tasks), 1),
                    "energy_range": [min(energy_vals), max(energy_vals)] if energy_vals else [],
                    "energy_mean": sum(energy_vals) / max(len(energy_vals), 1),
                }

            quartile_data[cond] = {
                "total_with_energy": len(tasks_with_energy),
                "quartiles": quartiles,
            }

        atomic_write_json(self.ablation_dir / "difficulty_quartiles.json", quartile_data)
        return quartile_data

    def _phase3_rescue(self):
        """Analyze Phase 3 rescue rates for conditions D and E."""
        log("  Computing Phase 3 rescue analysis...")
        rescue_data = {}

        for cond in ["D", "E"]:
            if cond not in self.results:
                continue
            res = self.results[cond]

            # Tasks rescued by Phase 3 sub-phases
            pr_cot_rescued = []
            refinement_rescued = []
            derivation_rescued = []
            total_phase1 = 0
            total_failed = 0

            for tid, r in res.items():
                phase = r.get("phase_solved", "none")
                if phase == "phase1":
                    total_phase1 += 1
                elif phase == "pr_cot":
                    pr_cot_rescued.append(tid)
                elif phase == "refinement":
                    refinement_rescued.append(tid)
                elif phase == "derivation":
                    derivation_rescued.append(tid)
                elif phase in ("none", "error"):
                    total_failed += 1

            total_rescued = len(pr_cot_rescued) + len(refinement_rescued) + len(derivation_rescued)
            # Rescue rate = rescued / (rescued + failed) — tasks that needed rescue
            rescue_pool = total_rescued + total_failed
            rescue_rate = total_rescued / max(rescue_pool, 1)

            rescue_data[cond] = {
                "total_tasks": len(res),
                "phase1_solved": total_phase1,
                "pr_cot_rescued": len(pr_cot_rescued),
                "refinement_rescued": len(refinement_rescued),
                "derivation_rescued": len(derivation_rescued),
                "total_rescued": total_rescued,
                "total_failed": total_failed,
                "rescue_pool": rescue_pool,
                "rescue_rate": rescue_rate,
                "pr_cot_task_ids": pr_cot_rescued[:30],
                "refinement_task_ids": refinement_rescued[:30],
            }

        atomic_write_json(self.ablation_dir / "phase3_rescue_analysis.json", rescue_data)
        return rescue_data

    def _phase2_efficiency(self):
        """Analyze Phase 2 metrics: ReASC stops, adaptive K, S* triggers."""
        log("  Computing Phase 2 efficiency...")
        efficiency_data = {}

        for cond in ["C", "E"]:
            if cond not in self.results:
                continue
            res = self.results[cond]

            adaptive_k_values = []
            s_star_count = 0
            reasc_stops = 0
            total_tokens = []

            for tid, r in res.items():
                tel = r.get("telemetry", {})
                ak = tel.get("adaptive_k")
                if ak is not None:
                    adaptive_k_values.append(ak)
                if tel.get("s_star_triggered"):
                    s_star_count += 1
                if tel.get("reasc_stopped"):
                    reasc_stops += 1
                total_tokens.append(r.get("total_tokens", 0))

            # K distribution
            k_dist = {}
            for k in adaptive_k_values:
                k_dist[str(k)] = k_dist.get(str(k), 0) + 1

            efficiency_data[cond] = {
                "total_tasks": len(res),
                "adaptive_k_distribution": k_dist,
                "adaptive_k_mean": sum(adaptive_k_values) / max(len(adaptive_k_values), 1) if adaptive_k_values else None,
                "s_star_triggers": s_star_count,
                "reasc_early_stops": reasc_stops,
                "total_tokens": sum(total_tokens),
                "avg_tokens_per_task": sum(total_tokens) / max(len(total_tokens), 1),
            }

        atomic_write_json(self.ablation_dir / "phase2_efficiency.json", efficiency_data)
        return efficiency_data

    def _v2_comparison(self):
        """Compare V2 baseline (~37%) to V3 full."""
        log("  Computing V2 comparison...")
        v2_baseline = 0.37  # Known V2 baseline

        comparison = {
            "v2_baseline_pass_rate": v2_baseline,
            "conditions": {},
        }

        for cond in sorted(self.results.keys()):
            res = self.results[cond]
            passed = sum(1 for r in res.values() if r.get("passed"))
            total = len(res)
            rate = passed / max(total, 1)
            comparison["conditions"][cond] = {
                "name": CONDITION_DEFS[cond]["name"],
                "pass_rate": rate,
                "delta_vs_v2_pp": (rate - v2_baseline) * 100,
                "relative_improvement": (rate - v2_baseline) / max(v2_baseline, 0.01) * 100,
            }

        atomic_write_json(self.ablation_dir / "v2_comparison.json", comparison)
        return comparison

    def _token_efficiency(self):
        """Tokens/task, tokens/pass, marginal cost per extra pass."""
        log("  Computing token efficiency...")
        efficiency = {}

        for cond in sorted(self.results.keys()):
            res = self.results[cond]
            total = len(res)
            passed = sum(1 for r in res.values() if r.get("passed"))
            tokens = [r.get("total_tokens", 0) for r in res.values()]
            total_tok = sum(tokens)

            # Tokens for passed vs failed tasks
            pass_tokens = [r.get("total_tokens", 0) for r in res.values() if r.get("passed")]
            fail_tokens = [r.get("total_tokens", 0) for r in res.values() if not r.get("passed")]

            efficiency[cond] = {
                "name": CONDITION_DEFS[cond]["name"],
                "total_tokens": total_tok,
                "tokens_per_task": total_tok / max(total, 1),
                "tokens_per_pass": total_tok / max(passed, 1),
                "avg_pass_tokens": sum(pass_tokens) / max(len(pass_tokens), 1) if pass_tokens else 0,
                "avg_fail_tokens": sum(fail_tokens) / max(len(fail_tokens), 1) if fail_tokens else 0,
                "passed": passed,
                "total": total,
            }

        # Marginal cost: extra tokens per extra pass vs baseline (A)
        if "A" in efficiency:
            base = efficiency["A"]
            for cond in efficiency:
                if cond == "A":
                    efficiency[cond]["marginal_tokens_per_extra_pass"] = None
                    continue
                extra_passes = efficiency[cond]["passed"] - base["passed"]
                extra_tokens = efficiency[cond]["total_tokens"] - base["total_tokens"]
                if extra_passes > 0:
                    efficiency[cond]["marginal_tokens_per_extra_pass"] = extra_tokens / extra_passes
                else:
                    efficiency[cond]["marginal_tokens_per_extra_pass"] = None

        atomic_write_json(self.ablation_dir / "token_efficiency.json", efficiency)
        return efficiency

    def _task_concordance(self):
        """Agreement matrix between conditions — which tasks agree/disagree."""
        log("  Computing task concordance...")
        conds = sorted(self.results.keys())

        # Build pass/fail vectors
        all_tasks = set()
        for res in self.results.values():
            all_tasks.update(res.keys())
        all_tasks = sorted(all_tasks)

        # Pairwise concordance
        matrix = {}
        for c1 in conds:
            matrix[c1] = {}
            for c2 in conds:
                if c1 == c2:
                    matrix[c1][c2] = 1.0
                    continue
                agree = 0
                total = 0
                for tid in all_tasks:
                    r1 = self.results.get(c1, {}).get(tid)
                    r2 = self.results.get(c2, {}).get(tid)
                    if r1 is None or r2 is None:
                        continue
                    total += 1
                    if r1.get("passed") == r2.get("passed"):
                        agree += 1
                matrix[c1][c2] = agree / max(total, 1)

        # Difficulty tiers: easy (all pass), hard (all fail), mixed
        easy = []
        hard = []
        mixed = []
        for tid in all_tasks:
            passes = sum(
                1 for c in conds
                if self.results.get(c, {}).get(tid, {}).get("passed", False)
            )
            if passes == len(conds):
                easy.append(tid)
            elif passes == 0:
                hard.append(tid)
            else:
                mixed.append(tid)

        concordance = {
            "conditions": conds,
            "agreement_matrix": matrix,
            "difficulty_tiers": {
                "easy_all_pass": len(easy),
                "hard_all_fail": len(hard),
                "mixed": len(mixed),
            },
            "total_tasks": len(all_tasks),
        }

        atomic_write_json(self.ablation_dir / "task_concordance.json", concordance)
        return concordance


# ===========================================================================
# ReportGenerator
# ===========================================================================

class ReportGenerator:
    """Generate Markdown and JSON reports from ablation analysis."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.ablation_dir = output_dir / "ablation"

    def generate(self):
        """Generate all reports."""
        log("=" * 60)
        log("GENERATING REPORTS")
        log("=" * 60)

        self._generate_summary_json()
        self._generate_ablation_report()
        self._generate_docs_page()

        log("Reports complete")

    def _load_json(self, name: str) -> dict:
        """Load a JSON file from ablation directory."""
        path = self.ablation_dir / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _generate_summary_json(self):
        """Generate top-level summary.json."""
        comparison = self._load_json("condition_comparison.json")
        deltas = self._load_json("phase_deltas.json")
        v2 = self._load_json("v2_comparison.json")
        token_eff = self._load_json("token_efficiency.json")

        summary = {
            "study": "ATLAS V3 Ablation Study",
            "version": "3.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "Qwen3-14B-Q4_K_M (frozen)",
            "benchmark": "LiveCodeBench v5 (599 tasks)",
            "conditions": {},
        }

        for cond, data in sorted(comparison.items()):
            summary["conditions"][cond] = {
                "name": data["name"],
                "pass_rate": data["pass_rate"],
                "passed": data["passed"],
                "total": data["total_tasks"],
                "total_tokens": data.get("total_tokens", 0),
            }

        # Phase contribution deltas
        if isinstance(deltas, list):
            summary["phase_contributions"] = []
            for d in deltas:
                summary["phase_contributions"].append({
                    "comparison": f"{d['from']} -> {d['to']}",
                    "description": d["description"],
                    "delta_pp": round(d["delta_pp"], 2),
                    "tasks_gained": d["tasks_gained"],
                    "tasks_lost": d["tasks_lost"],
                })

        atomic_write_json(self.output_dir / "summary.json", summary)
        log(f"  Written: {self.output_dir / 'summary.json'}")

    def _generate_ablation_report(self):
        """Generate full ABLATION_REPORT.md."""
        comparison = self._load_json("condition_comparison.json")
        deltas = self._load_json("phase_deltas.json")
        rescue = self._load_json("phase3_rescue_analysis.json")
        p2_eff = self._load_json("phase2_efficiency.json")
        quartiles = self._load_json("difficulty_quartiles.json")
        token_eff = self._load_json("token_efficiency.json")
        concordance = self._load_json("task_concordance.json")
        v2 = self._load_json("v2_comparison.json")

        md = []
        md.append("# ATLAS V3 Ablation Study Report")
        md.append("")
        md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"Model: Qwen3-14B-Q4_K_M (frozen, Q4_K_M + 0.6B draft)")
        md.append(f"Benchmark: LiveCodeBench v5 (599 tasks)")
        md.append("")

        # Executive Summary
        md.append("## Executive Summary")
        md.append("")
        if comparison:
            for cond in ["A", "B", "C", "D", "E"]:
                if cond in comparison:
                    c = comparison[cond]
                    md.append(f"- **Condition {cond}** ({c['name']}): "
                              f"**{c['pass_rate']*100:.1f}%** "
                              f"({c['passed']}/{c['total_tasks']})")
            md.append("")

            if "A" in comparison and "E" in comparison:
                a_rate = comparison["A"]["pass_rate"] * 100
                e_rate = comparison["E"]["pass_rate"] * 100
                md.append(f"**Total V3 lift: {e_rate - a_rate:+.1f}pp** "
                          f"({a_rate:.1f}% -> {e_rate:.1f}%)")
                md.append("")

        # Condition Comparison Table
        md.append("## Condition Comparison")
        md.append("")
        md.append("| Condition | Name | P1 | P2 | P3 | Pass Rate | Passed | Tokens/Task |")
        md.append("|-----------|------|----|----|----|-----------|--------|-------------|")
        for cond in ["A", "B", "C", "D", "E"]:
            if cond not in comparison:
                continue
            c = comparison[cond]
            d = CONDITION_DEFS[cond]
            p1 = "ON" if d["enable_phase1"] else "OFF"
            p2 = "ON" if d["enable_phase2"] else "OFF"
            p3 = "ON" if d["enable_phase3"] else "OFF"
            tok_per = c.get("avg_tokens_per_task", 0)
            md.append(f"| {cond} | {c['name']} | {p1} | {p2} | {p3} | "
                      f"{c['pass_rate']*100:.1f}% | {c['passed']}/{c['total_tasks']} | "
                      f"{tok_per:,.0f} |")
        md.append("")

        # Phase Delta Table
        if isinstance(deltas, list) and deltas:
            md.append("## Phase Contributions (Pairwise Deltas)")
            md.append("")
            md.append("| Comparison | Description | Delta (pp) | Gained | Lost | Net |")
            md.append("|------------|-------------|-----------|--------|------|-----|")
            for d in deltas:
                net = d["tasks_gained"] - d["tasks_lost"]
                md.append(f"| {d['from']} -> {d['to']} | {d['description']} | "
                          f"{d['delta_pp']:+.1f} | +{d['tasks_gained']} | "
                          f"-{d['tasks_lost']} | {net:+d} |")
            md.append("")

        # Phase Breakdown per Condition
        md.append("## Phase Breakdown by Condition")
        md.append("")
        md.append("| Condition | Phase 1 | PR-CoT | Refinement | Derivation | Failed | Error |")
        md.append("|-----------|---------|--------|------------|------------|--------|-------|")
        for cond in ["A", "B", "C", "D", "E"]:
            if cond not in comparison:
                continue
            pb = comparison[cond].get("phase_breakdown", {})
            md.append(f"| {cond} ({comparison[cond]['name']}) | "
                      f"{pb.get('phase1', 0)} | {pb.get('pr_cot', 0)} | "
                      f"{pb.get('refinement', 0)} | {pb.get('derivation', 0)} | "
                      f"{pb.get('none', 0)} | {pb.get('error', 0)} |")
        md.append("")

        # Phase 3 Rescue Analysis
        if rescue:
            md.append("## Phase 3 Rescue Analysis")
            md.append("")
            for cond in ["D", "E"]:
                if cond not in rescue:
                    continue
                r = rescue[cond]
                md.append(f"### Condition {cond} ({CONDITION_DEFS[cond]['name']})")
                md.append(f"- Phase 1 solved: {r['phase1_solved']}")
                md.append(f"- Rescue pool (needed Phase 3): {r['rescue_pool']}")
                md.append(f"- PR-CoT rescued: {r['pr_cot_rescued']}")
                md.append(f"- Refinement rescued: {r['refinement_rescued']}")
                md.append(f"- Derivation rescued: {r['derivation_rescued']}")
                md.append(f"- **Rescue rate: {r['rescue_rate']*100:.1f}%** "
                          f"({r['total_rescued']}/{r['rescue_pool']})")
                md.append("")

        # Phase 2 Efficiency
        if p2_eff:
            md.append("## Phase 2 Efficiency Metrics")
            md.append("")
            for cond in ["C", "E"]:
                if cond not in p2_eff:
                    continue
                p = p2_eff[cond]
                md.append(f"### Condition {cond} ({CONDITION_DEFS[cond]['name']})")
                md.append(f"- Adaptive K mean: {p.get('adaptive_k_mean', 'N/A')}")
                md.append(f"- K distribution: {json.dumps(p.get('adaptive_k_distribution', {}))}")
                md.append(f"- S* triggers: {p['s_star_triggers']}")
                md.append(f"- ReASC early stops: {p['reasc_early_stops']}")
                md.append(f"- Avg tokens/task: {p['avg_tokens_per_task']:,.0f}")
                md.append("")

        # Difficulty Quartiles
        if quartiles:
            md.append("## Difficulty Quartiles (by Lens Energy)")
            md.append("")
            for cond in ["C", "E"]:
                if cond not in quartiles or "quartiles" not in quartiles.get(cond, {}):
                    continue
                md.append(f"### Condition {cond}")
                md.append("")
                md.append("| Quartile | Count | Passed | Pass Rate | Energy Range |")
                md.append("|----------|-------|--------|-----------|-------------|")
                for qlabel, qdata in quartiles[cond]["quartiles"].items():
                    erange = qdata.get("energy_range", [])
                    estr = f"{erange[0]:.2f}-{erange[1]:.2f}" if len(erange) == 2 else "N/A"
                    md.append(f"| {qlabel} | {qdata['count']} | {qdata['passed']} | "
                              f"{qdata['pass_rate']*100:.1f}% | {estr} |")
                md.append("")

        # Token Efficiency
        if token_eff:
            md.append("## Token Efficiency")
            md.append("")
            md.append("| Condition | Tokens/Task | Tokens/Pass | Marginal Cost |")
            md.append("|-----------|-------------|-------------|---------------|")
            for cond in ["A", "B", "C", "D", "E"]:
                if cond not in token_eff:
                    continue
                t = token_eff[cond]
                marginal = t.get("marginal_tokens_per_extra_pass")
                mstr = f"{marginal:,.0f}" if marginal is not None else "—"
                md.append(f"| {cond} ({t['name']}) | {t['tokens_per_task']:,.0f} | "
                          f"{t['tokens_per_pass']:,.0f} | {mstr} |")
            md.append("")

        # Task Concordance
        if concordance and "agreement_matrix" in concordance:
            md.append("## Task Concordance")
            md.append("")
            tiers = concordance.get("difficulty_tiers", {})
            md.append(f"- Easy (all conditions pass): {tiers.get('easy_all_pass', 0)}")
            md.append(f"- Hard (all conditions fail): {tiers.get('hard_all_fail', 0)}")
            md.append(f"- Mixed (conditions disagree): {tiers.get('mixed', 0)}")
            md.append("")

            conds = concordance.get("conditions", [])
            if conds:
                header = "| | " + " | ".join(conds) + " |"
                sep = "|---|" + "|".join(["---"] * len(conds)) + "|"
                md.append("### Agreement Matrix")
                md.append("")
                md.append(header)
                md.append(sep)
                mtx = concordance["agreement_matrix"]
                for c1 in conds:
                    row = f"| **{c1}** |"
                    for c2 in conds:
                        val = mtx.get(c1, {}).get(c2, 0)
                        row += f" {val*100:.0f}% |"
                    md.append(row)
                md.append("")

        # V2 Comparison
        if v2 and "conditions" in v2:
            md.append("## V2 Baseline Comparison")
            md.append("")
            md.append(f"V2 baseline: {v2['v2_baseline_pass_rate']*100:.0f}%")
            md.append("")
            md.append("| Condition | Pass Rate | Delta vs V2 | Relative Improvement |")
            md.append("|-----------|-----------|-------------|---------------------|")
            for cond in ["A", "B", "C", "D", "E"]:
                if cond not in v2["conditions"]:
                    continue
                vc = v2["conditions"][cond]
                md.append(f"| {cond} ({vc['name']}) | {vc['pass_rate']*100:.1f}% | "
                          f"{vc['delta_vs_v2_pp']:+.1f}pp | "
                          f"{vc['relative_improvement']:+.0f}% |")
            md.append("")

        md.append("---")
        md.append("*Generated by ATLAS V3 Ablation Runner*")
        md.append("")

        report_path = self.output_dir / "ABLATION_REPORT.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(md))
        log(f"  Written: {report_path}")

    def _generate_docs_page(self):
        """Generate shorter docs/V3_ABLATION_STUDY.md for project docs."""
        comparison = self._load_json("condition_comparison.json")
        deltas = self._load_json("phase_deltas.json")
        rescue = self._load_json("phase3_rescue_analysis.json")
        v2 = self._load_json("v2_comparison.json")

        md = []
        md.append("# V3 Ablation Study")
        md.append("")
        md.append("## Overview")
        md.append("")
        md.append("5-condition ablation isolating each V3 phase's contribution to LCB pass@1.")
        md.append("Model: Qwen3-14B-Q4_K_M (frozen). Benchmark: LiveCodeBench v5 (599 tasks).")
        md.append("")

        md.append("## Conditions")
        md.append("")
        md.append("| Condition | Phases | Description |")
        md.append("|-----------|--------|-------------|")
        for cond in ["A", "B", "C", "D", "E"]:
            d = CONDITION_DEFS[cond]
            phases = []
            if d["enable_phase1"]:
                phases.append("P1")
            if d["enable_phase2"]:
                phases.append("P2")
            if d["enable_phase3"]:
                phases.append("P3")
            pstr = "+".join(phases) if phases else "None"
            md.append(f"| {cond} | {pstr} | {d['description']} |")
        md.append("")

        # Results table
        if comparison:
            md.append("## Results")
            md.append("")
            md.append("| Condition | Name | Pass Rate | Passed/Total |")
            md.append("|-----------|------|-----------|-------------|")
            for cond in ["A", "B", "C", "D", "E"]:
                if cond not in comparison:
                    continue
                c = comparison[cond]
                md.append(f"| {cond} | {c['name']} | "
                          f"**{c['pass_rate']*100:.1f}%** | "
                          f"{c['passed']}/{c['total_tasks']} |")
            md.append("")

        # Phase deltas
        if isinstance(deltas, list) and deltas:
            md.append("## Phase Contributions")
            md.append("")
            md.append("| Comparison | Description | Delta (pp) |")
            md.append("|------------|-------------|-----------|")
            for d in deltas:
                md.append(f"| {d['from']}->{d['to']} | {d['description']} | "
                          f"**{d['delta_pp']:+.1f}** |")
            md.append("")

        # Phase 3 rescue
        if rescue:
            md.append("## Phase 3 Rescue")
            md.append("")
            for cond in ["D", "E"]:
                if cond not in rescue:
                    continue
                r = rescue[cond]
                md.append(f"- **Condition {cond}**: rescued {r['total_rescued']}/{r['rescue_pool']} "
                          f"({r['rescue_rate']*100:.1f}% rescue rate)")
            md.append("")

        md.append("## Methodology")
        md.append("")
        md.append("- Each condition runs all 599 LCB tasks through the V3 pipeline")
        md.append("  with specific phases enabled/disabled.")
        md.append("- Condition D imports results from the initial v3_full_14b_final run")
        md.append("  (P2 was active but Lens was not loaded, so P2 fell back to k=3).")
        md.append("- All conditions share the same frozen model, prompts, and seed.")
        md.append("- Crash recovery: each condition resumes from per-task checkpoint files.")
        md.append("")
        md.append("Full results: `v3_ablation_results/ABLATION_REPORT.md`")
        md.append("")

        docs_dir = PROJECT_ROOT / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        docs_path = docs_dir / "V3_ABLATION_STUDY.md"
        with open(docs_path, 'w') as f:
            f.write("\n".join(md))
        log(f"  Written: {docs_path}")


# ===========================================================================
# V3AblationRunner — Top-level orchestrator
# ===========================================================================

class V3AblationRunner:
    """Top-level orchestrator: Preflight -> Load LCB -> Run conditions -> Analyze -> Report."""

    def __init__(self, output_dir: Path, conditions: List[str],
                 skip_preflight: bool = False, skip_analysis: bool = False,
                 analysis_only: bool = False, smoke: bool = False,
                 max_tasks: Optional[int] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conditions = [c.upper() for c in conditions]
        self.skip_preflight = skip_preflight
        self.skip_analysis = skip_analysis
        self.analysis_only = analysis_only
        self.smoke = smoke
        self.max_tasks = max_tasks
        self.condition_results: Dict[str, Dict[str, Dict]] = {}

    def run(self):
        """Execute the full ablation study."""
        log("=" * 60)
        log("ATLAS V3 ABLATION STUDY")
        log("=" * 60)
        log(f"Output dir: {self.output_dir}")
        log(f"Conditions: {', '.join(self.conditions)}")
        if self.smoke:
            log(f"SMOKE MODE: 10 tasks only")
        if self.analysis_only:
            log(f"ANALYSIS-ONLY MODE: loading existing results")
        log("")

        # Save study config
        self._save_config()

        # Preflight
        need_lens = any(c in self.conditions for c in ["C", "E"])
        if not self.skip_preflight and not self.analysis_only:
            checker = PreflightChecker(self.output_dir)
            if not checker.run(need_lens=need_lens):
                log("PRE-FLIGHT FAILED. Aborting.", "ERROR")
                log("Use --skip-preflight to bypass.")
                return False

        # Load dataset
        log("\nLoading LiveCodeBench...", "INFO")
        tasks = load_lcb_tasks()
        log(f"Loaded {len(tasks)} tasks")

        if self.smoke:
            tasks = tasks[:10]
            log(f"SMOKE MODE: using {len(tasks)} tasks")
        elif self.max_tasks:
            tasks = tasks[:self.max_tasks]
            log(f"LIMITED MODE: using {len(tasks)} tasks")

        # Run or load conditions
        log("")
        for cond in self.conditions:
            self._run_condition(cond, tasks)

        # Analysis
        if not self.skip_analysis and self.condition_results:
            analyzer = AblationAnalyzer(self.output_dir, self.condition_results)
            analyzer.run_all()

            # Generate reports
            reporter = ReportGenerator(self.output_dir)
            reporter.generate()

        # Final summary
        self._print_summary()

        total_time = logger.elapsed_str()
        log("")
        log("=" * 60)
        log(f"V3 ABLATION STUDY COMPLETE — Total time: {total_time}")
        log("=" * 60)
        return True

    def _save_config(self):
        """Save study configuration."""
        cfg = {
            "study": "ATLAS V3 Ablation Study",
            "version": "3.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": {
                "main": "Qwen3-14B-Q4_K_M",
                "draft": "Qwen3-0.6B-Q8_0",
                "spec_decode": True,
                "frozen": True,
            },
            "conditions": {
                c: {
                    "name": CONDITION_DEFS[c]["name"],
                    "description": CONDITION_DEFS[c]["description"],
                    "enable_phase1": CONDITION_DEFS[c]["enable_phase1"],
                    "enable_phase2": CONDITION_DEFS[c]["enable_phase2"],
                    "enable_phase3": CONDITION_DEFS[c]["enable_phase3"],
                    "directory": CONDITION_DIRS[c],
                }
                for c in self.conditions
            },
            "benchmark": {
                "dataset": "LiveCodeBench v5",
                "tasks": 599,
                "eval_mode": "stdio",
                "temperature": 0.6,
            },
            "smoke": self.smoke,
            "max_tasks": self.max_tasks,
        }
        atomic_write_json(self.output_dir / "config.json", cfg)

    def _run_condition(self, cond: str, tasks: List[BenchmarkTask]):
        """Run a single condition (or import/load existing results)."""
        log("=" * 60)
        log(f"CONDITION {cond}: {CONDITION_DEFS[cond]['name']}")
        log(f"  {CONDITION_DEFS[cond]['description']}")
        d = CONDITION_DEFS[cond]
        log(f"  Phase 1: {'ON' if d['enable_phase1'] else 'OFF'} | "
            f"Phase 2: {'ON' if d['enable_phase2'] else 'OFF'} | "
            f"Phase 3: {'ON' if d['enable_phase3'] else 'OFF'}")
        log("=" * 60)

        runner = ConditionRunner(cond, self.output_dir, tasks)

        if self.analysis_only:
            # Load existing results only
            results = runner.load_results()
            if not results:
                log(f"  No existing results for Condition {cond} — skipping", "WARN")
                return
            passed = sum(1 for r in results.values() if r.get("passed"))
            log(f"  Loaded {len(results)} results ({passed} passed, "
                f"{passed/max(len(results),1)*100:.1f}%)")
        else:
            # Check memory before each live condition via cache manager
            log(f"  Pre-condition memory check...")
            manage_server_memory()
            results = runner.run()

        if results:
            self.condition_results[cond] = results
            passed = sum(1 for r in results.values() if r.get("passed"))
            log(f"\n  Condition {cond} result: {passed}/{len(results)} "
                f"({passed/max(len(results),1)*100:.1f}%)\n")

    def _print_summary(self):
        """Print final summary table."""
        if not self.condition_results:
            return

        log("")
        log("=" * 60)
        log("  ABLATION SUMMARY")
        log("=" * 60)
        log(f"  {'Cond':>4} | {'Name':<15} | {'P1':>3} {'P2':>3} {'P3':>3} | "
            f"{'Passed':>6} | {'Rate':>7} | {'Delta':>7}")
        log(f"  {'-'*4:>4} | {'-'*15:<15} | {'-'*3:>3} {'-'*3:>3} {'-'*3:>3} | "
            f"{'-'*6:>6} | {'-'*7:>7} | {'-'*7:>7}")

        baseline_rate = None
        for cond in ["A", "B", "C", "D", "E"]:
            if cond not in self.condition_results:
                continue
            res = self.condition_results[cond]
            passed = sum(1 for r in res.values() if r.get("passed"))
            total = len(res)
            rate = passed / max(total, 1)

            if cond == "A":
                baseline_rate = rate

            d = CONDITION_DEFS[cond]
            p1 = "ON" if d["enable_phase1"] else "OFF"
            p2 = "ON" if d["enable_phase2"] else "OFF"
            p3 = "ON" if d["enable_phase3"] else "OFF"

            delta = ""
            if baseline_rate is not None:
                delta = f"{(rate - baseline_rate)*100:+.1f}pp"

            log(f"  {cond:>4} | {CONDITION_DEFS[cond]['name']:<15} | "
                f"{p1:>3} {p2:>3} {p3:>3} | "
                f"{passed:>3}/{total:<3} | {rate*100:>6.1f}% | {delta:>7}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS V3 Ablation Study Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Conditions:
  A: Baseline      (all V3 OFF)
  B: +Phase1       (PlanSearch + DivSampling + BudgetForcing)
  C: +Phase1+2     (+ Lens adaptive K, ReASC, S*)
  D: +Phase1+3     (+ PR-CoT repair, imported from v3_full_14b_final)
  E: Full V3       (all phases active)

Examples:
  python3 v3_ablation_runner.py                       # Full ablation
  python3 v3_ablation_runner.py --conditions D        # Import only
  python3 v3_ablation_runner.py --analysis-only       # Re-analyze existing
  python3 v3_ablation_runner.py --smoke --conditions A B  # Smoke test
        """,
    )
    parser.add_argument("--output-dir", type=str, default="./v3_ablation_results",
                        help="Output directory (default: ./v3_ablation_results)")
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS,
                        help="Conditions to run (default: all)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preflight checks")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip cross-condition analysis")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Load existing results and re-run analysis only")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test (10 tasks only)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of tasks")

    args = parser.parse_args()

    runner = V3AblationRunner(
        output_dir=Path(args.output_dir),
        conditions=args.conditions,
        skip_preflight=args.skip_preflight,
        skip_analysis=args.skip_analysis,
        analysis_only=args.analysis_only,
        smoke=args.smoke,
        max_tasks=args.max_tasks,
    )

    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
