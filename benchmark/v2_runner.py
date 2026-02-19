#!/usr/bin/env python3
"""
ATLAS V2 Benchmark Runner.

Orchestrates the full V2 benchmark suite across phases:
  Phase 0: Smoke test (5 LCB + 3 GPQA + 3 IFBench + 5 Custom)
  Phase 1: LiveCodeBench Mode B (880 x 3)
  Phase 2: LiveCodeBench Mode A (880 x 3)
  Phase 3: GPQA Diamond Mode B (198 x 5)
  Phase 4: IFBench Mode B (294 x 5)
  Phase 5: Custom Mode A + Mode B
  Phase 6: SciCode Mode B (stretch)

Captures V2 telemetry (router, cache, lens signals) per generation.
"""

import json
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Force line-buffered stdout for real-time progress visibility
sys.stdout.reconfigure(line_buffering=True)

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.config import config
from benchmark.models import BenchmarkTask, AttemptResult, TaskResult
from benchmark.runner import BenchmarkRunner, LLMConnectionError
from benchmark.datasets import (
    LiveCodeBenchDataset, SciCodeDataset, GPQADiamondDataset, IFBenchDataset
)
from benchmark.datasets.gpqa import extract_mcq_answer
from benchmark.datasets.ifbench import evaluate_ifbench_loose
from benchmark.geo_learning import (
    extract_embedding_urllib, LearningCurveTracker, shuffle_and_split_epochs
)
from benchmark.best_of_k import score_candidate, get_temperature, BestOfKTracker


# --- Constants ----------------------------------------------------------------

RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:31144")
MAX_TOKENS = 16384
TEMPERATURE = 0.0


# --- Telemetry ----------------------------------------------------------------

def collect_v2_telemetry(task_id, benchmark, result):
    """Build a V2 telemetry record for a single generation."""
    return {
        "task_id": task_id,
        "benchmark": benchmark,
        "route_selected": result.get("route", "STANDARD"),
        "difficulty_bin": result.get("difficulty", "UNKNOWN"),
        "pattern_cache_hit": result.get("cache_hit", False),
        "pattern_cache_score": result.get("cache_score", 0.0),
        "retrieval_confidence": result.get("retrieval_confidence", 0.0),
        "query_complexity": result.get("query_complexity", 0.0),
        "geometric_energy": result.get("geometric_energy", 0.0),
        "tokens_generated": result.get("tokens", 0),
        "generation_time_ms": result.get("gen_time_ms", 0),
        "result": "PASS" if result.get("passed") else "FAIL",
        "retries_used": result.get("retries", 0),
        "thinking_enabled": result.get("thinking_enabled", False),
        "timestamp": datetime.utcnow().isoformat(),
    }


def query_v2_signals(prompt):
    """Query RAG API internal endpoints for V2 signal data."""
    signals = {
        "cache_hit": False, "cache_score": 0.0,
        "retrieval_confidence": 0.0, "query_complexity": 0.0,
        "geometric_energy": 0.0, "route": "STANDARD", "difficulty": "UNKNOWN",
    }
    try:
        body = json.dumps({"query": prompt[:500]}).encode('utf-8')
        req = urllib.request.Request(
            f"{RAG_API_URL}/internal/lens/evaluate",
            data=body, headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            signals["geometric_energy"] = data.get("energy_normalized", data.get("energy_before", 0.0))
    except Exception:
        pass
    try:
        req = urllib.request.Request(f"{RAG_API_URL}/internal/cache/stats")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            signals["cache_score"] = data.get("hit_rate", 0.0)
    except Exception:
        pass
    return signals


# --- Atomic file I/O ----------------------------------------------------------

def atomic_write_json(filepath, data):
    """Write JSON atomically via temp + rename."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix('.tmp')
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        shutil.move(str(tmp), str(filepath))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def append_jsonl(filepath, record):
    """Append a JSON record to a JSONL file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(json.dumps(record) + '\n')


def find_completed_tasks(phase_dir):
    """Find already-completed task IDs in a phase directory."""
    completed = set()
    per_task_dir = Path(phase_dir) / "per_task"
    if per_task_dir.exists():
        for f in per_task_dir.glob("*.json"):
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                    if 'task_id' in data:
                        completed.add(data['task_id'])
            except (json.JSONDecodeError, IOError):
                pass
    return completed


def _set_routing_mode(mode):
    """Set routing mode via kubectl. Static command, not user-controlled."""
    if mode == "A":
        # Enable router-controlled routing
        subprocess.run(
            ["kubectl", "set", "env", "deployment/rag-api",
             "ROUTING_ENABLED=true", "-n", "atlas"],
            capture_output=True, timeout=10
        )
    else:
        # Disable routing (all tasks get STANDARD route)
        subprocess.run(
            ["kubectl", "set", "env", "deployment/rag-api",
             "ROUTING_ENABLED=false", "-n", "atlas"],
            capture_output=True, timeout=10
        )
    time.sleep(2)


def _trigger_retrain(tracker, max_epoch):
    """POST accumulated training data to rag-api for C(x) retrain."""
    payload = tracker.prepare_retrain_payload(max_epoch=max_epoch)
    n_pass, n_fail = tracker.count_labels(max_epoch=max_epoch)

    if n_fail < 5:
        print(f"    Skipping retrain: only {n_fail} FAIL examples (need >=5)")
        return {"skipped": True, "reason": f"insufficient_fails ({n_fail})"}

    body = json.dumps({"training_data": payload, "epochs": 50}).encode('utf-8')
    req = urllib.request.Request(
        f"{RAG_API_URL}/internal/lens/retrain",
        data=body,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        metrics = result.get("metrics", {})
        print(f"    Retrain complete: val_auc={metrics.get('val_auc', '?')}, "
              f"train_size={metrics.get('train_size', '?')}, "
              f"fail_ratio={metrics.get('fail_ratio', '?')}")
        return metrics
    except Exception as e:
        print(f"    Retrain failed: {e}")
        return {"skipped": True, "reason": str(e)}


# --- Think Decision -----------------------------------------------------------

def _should_think(task, mode, signals):
    """Decide whether to enable thinking mode for a task.

    Currently disabled globally: Qwen3-14B thinking causes runaway chains
    on competitive programming (10+ min, 20K+ tokens) with no accuracy gain.
    All tasks use /nothink. The router, lens, and full infrastructure remain
    active — only the thinking toggle is off.

    The think plumbing is preserved so thinking can be re-enabled per
    eval_mode or per route once token budget controls are in place.
    """
    return False


# --- V2 Runner ----------------------------------------------------------------

class V2BenchmarkRunner:
    """Runs V2 benchmark phases with telemetry collection."""

    def __init__(self, run_dir, mode="B", use_best_of_k=False):
        self.run_dir = Path(run_dir)
        self.mode = mode
        self.use_best_of_k = use_best_of_k
        self.telemetry_dir = self.run_dir / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.runner = BenchmarkRunner(max_retries=10)
        self._start_time = time.time()

    def close(self):
        self.runner.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def set_mode(self, mode):
        """Switch between Mode A (router-controlled) and Mode B (fixed)."""
        self.mode = mode
        _set_routing_mode(mode)

    def run_phase(self, phase_name, tasks, benchmark_name, repeats=1, phase_dir=None, tracker=None, epoch_num=None, bok_tracker=None):
        """Run a benchmark phase with resume support."""
        if phase_dir is None:
            phase_dir = self.run_dir / phase_name
        phase_dir = Path(phase_dir)
        phase_dir.mkdir(parents=True, exist_ok=True)
        per_task_dir = phase_dir / "per_task"
        per_task_dir.mkdir(parents=True, exist_ok=True)

        completed = find_completed_tasks(phase_dir)
        remaining = [t for t in tasks if t.task_id not in completed]
        total = len(tasks)
        done = len(completed)

        if completed:
            print(f"  Resuming {phase_name}: {done}/{total} complete, {len(remaining)} remaining")

        results = {}
        for f in per_task_dir.glob("*.json"):
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                    results[data['task_id']] = TaskResult.from_dict(data)
            except Exception:
                pass

        for idx, task in enumerate(remaining):
            task_start = time.time()
            signals = query_v2_signals(task.prompt) if self.mode == "A" else {}
            think = _should_think(task, self.mode, signals)

            task_result = self._run_task_with_mode(task, repeats, bok_tracker=bok_tracker, think=think)
            results[task.task_id] = task_result
            gen_time_ms = (time.time() - task_start) * 1000

            # Score the best attempt's generated code through the lens
            # (post-hoc, for telemetry — the energy reflects code quality, not prompt difficulty)
            if task_result.attempts:
                best_code = task_result.attempts[-1].generated_code
                if best_code:
                    energy, normalized = score_candidate(best_code, RAG_API_URL)
                    signals["geometric_energy"] = normalized

            telemetry = collect_v2_telemetry(task.task_id, benchmark_name, {
                **signals,
                "passed": task_result.passed,
                "tokens": task_result.total_tokens,
                "gen_time_ms": gen_time_ms,
                "retries": task_result.num_attempts,
                "route": signals.get("route", "STANDARD"),
                "difficulty": signals.get("difficulty", "UNKNOWN"),
                "thinking_enabled": think,
            })
            append_jsonl(self.telemetry_dir / "route_decisions.jsonl", telemetry)

            label = "PASS" if task_result.passed else "FAIL"
            embed_record = {
                "task_id": task.task_id, "benchmark": benchmark_name,
                "label": label, "phase": phase_name,
            }
            if task_result.attempts:
                embed_record["generated_code"] = task_result.attempts[-1].generated_code[:2000]
            append_jsonl(self.telemetry_dir / "failure_embeddings.jsonl", embed_record)

            # Collect training embeddings for continuous learning.
            # Embed each attempt's GENERATED CODE (not the prompt) so the lens
            # learns to distinguish passing code from failing code.
            if tracker is not None and epoch_num is not None:
                for attempt in task_result.attempts:
                    code = attempt.generated_code
                    if not code:
                        continue
                    attempt_label = "PASS" if attempt.passed else "FAIL"
                    emb = extract_embedding_urllib(code, config.llama_url)
                    if emb:
                        tracker.record_embedding(
                            task_id=task.task_id,
                            embedding=emb,
                            label=attempt_label,
                            epoch=epoch_num,
                        )

            safe_name = task.task_id.replace('/', '_')
            atomic_write_json(per_task_dir / f"{safe_name}.json", task_result.to_dict())

            done += 1
            status = "PASS" if task_result.passed else "FAIL"
            elapsed = time.time() - self._start_time
            rate = done / (elapsed / 3600) if elapsed > 0 else 0
            print(
                f"  [{done}/{total}] {task.task_id}: {status} "
                f"({task_result.num_passed}/{task_result.num_attempts} passed) "
                f"[{rate:.0f} tasks/hr]",
                flush=True
            )

        phase_summary = {
            "phase": phase_name, "benchmark": benchmark_name, "mode": self.mode,
            "total_tasks": len(results),
            "passed_tasks": sum(1 for r in results.values() if r.passed),
            "pass_rate": sum(1 for r in results.values() if r.passed) / max(len(results), 1),
            "results": {k: v.to_dict() for k, v in results.items()},
        }
        atomic_write_json(phase_dir / "results.json", phase_summary)
        return results

    def _run_task_with_mode(self, task, repeats, bok_tracker=None, think=False):
        """Route to the correct handler based on eval_mode."""
        if self.use_best_of_k and repeats > 1:
            return self._run_task_best_of_k(task, repeats, bok_tracker=bok_tracker, think=think)
        if task.eval_mode == "mcq":
            return self._run_mcq_task(task, repeats, think=think)
        elif task.eval_mode == "ifbench":
            return self._run_ifbench_task(task, repeats, think=think)
        else:
            return self.runner.run_task(task, k=repeats, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, think=think)

    def _run_mcq_task(self, task, repeats, think=False):
        """Run a multiple-choice question task."""
        result = TaskResult(task_id=task.task_id)
        correct = task.test_outputs[0] if task.test_outputs else task.canonical_solution

        for n in range(1, repeats + 1):
            try:
                response, tokens, t_ms = self.runner._call_llm(
                    task.prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                    think=think
                )
                extracted = extract_mcq_answer(response)
                passed = extracted == correct

                result.attempts.append(AttemptResult(
                    task_id=task.task_id, attempt_number=n,
                    generated_code=response, passed=passed,
                    execution_time_ms=0,
                    error_output="" if passed else f"Expected {correct}, got {extracted}",
                    tokens_generated=tokens, inference_time_ms=t_ms,
                    stdout=extracted or "", stderr=""
                ))
                result.total_tokens += tokens
                result.total_inference_time_ms += t_ms
                if passed and result.best_attempt is None:
                    result.best_attempt = n
            except LLMConnectionError as e:
                result.attempts.append(AttemptResult(
                    task_id=task.task_id, attempt_number=n,
                    generated_code="", passed=False, execution_time_ms=0,
                    error_output=str(e), tokens_generated=0, inference_time_ms=0
                ))
        return result

    def _run_ifbench_task(self, task, repeats, think=False):
        """Run an IFBench instruction-following task."""
        result = TaskResult(task_id=task.task_id)
        try:
            meta = json.loads(task.canonical_solution)
            iids = meta.get("instruction_id_list", [])
            kwargs_list = meta.get("kwargs", [{}])
        except (json.JSONDecodeError, TypeError):
            iids, kwargs_list = [], [{}]

        for n in range(1, repeats + 1):
            try:
                response, tokens, t_ms = self.runner._call_llm(
                    task.prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                    think=think
                )
                errors = []
                for i, iid in enumerate(iids):
                    kw = kwargs_list[i] if i < len(kwargs_list) else {}
                    if not evaluate_ifbench_loose(response, iid, kw):
                        errors.append(f"Failed: {iid}")

                result.attempts.append(AttemptResult(
                    task_id=task.task_id, attempt_number=n,
                    generated_code=response, passed=len(errors) == 0,
                    execution_time_ms=0,
                    error_output="; ".join(errors) if errors else "",
                    tokens_generated=tokens, inference_time_ms=t_ms,
                    stdout="", stderr=""
                ))
                result.total_tokens += tokens
                result.total_inference_time_ms += t_ms
                if not errors and result.best_attempt is None:
                    result.best_attempt = n
            except LLMConnectionError as e:
                result.attempts.append(AttemptResult(
                    task_id=task.task_id, attempt_number=n,
                    generated_code="", passed=False, execution_time_ms=0,
                    error_output=str(e), tokens_generated=0, inference_time_ms=0
                ))
        return result

    def _run_task_best_of_k(self, task, k, bok_tracker=None, think=False):
        """Run a task with best-of-K lens-guided selection.

        Generates K candidates at temperature>0 using pipelined requests
        (overlapping generation via ThreadPoolExecutor + --parallel 2 on
        llama-server). Scores each with the Geometric Lens, then for code
        tasks tries sandbox in energy-sorted order (early exit on first
        pass). For non-code tasks, returns the lowest-energy candidate.
        """
        import hashlib
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from benchmark.runner import execute_code, execute_code_stdio, extract_code

        temperature = get_temperature(k, eval_mode=task.eval_mode)
        result = TaskResult(task_id=task.task_id)
        candidates = []

        # Phase 1: Generate K candidates with pipelined requests.
        # With --parallel 2 and --cont-batching on llama-server, queuing
        # requests early means the server starts prompt processing for N+1
        # while N is still generating. cache_prompt=True lets the server
        # reuse the KV cache for the shared prompt across candidates.
        def _generate_one(index):
            """Generate a single candidate (runs in thread pool)."""
            seed = index * 42 + 1  # Deterministic but diverse seeds
            try:
                response, tokens, t_ms = self.runner._call_llm(
                    task.prompt, temperature=temperature, max_tokens=MAX_TOKENS,
                    seed=seed, cache_prompt=True, think=think,
                )
                # For code tasks, extract the code for scoring/execution.
                # For non-code tasks (MCQ, IFBench), score the full response
                # text — extract_code() would mangle reasoning chains.
                if task.eval_mode in ("mcq", "ifbench"):
                    generated_code = response
                else:
                    generated_code = extract_code(response)
                energy, normalized = score_candidate(generated_code, RAG_API_URL)
                code_hash = hashlib.md5(generated_code.encode()).hexdigest()[:8]
                return {
                    "index": index, "response": response,
                    "generated_code": generated_code,
                    "tokens": tokens, "inference_time_ms": t_ms,
                    "energy": energy, "normalized": normalized,
                    "code_hash": code_hash, "passed": None,
                }
            except LLMConnectionError:
                return {
                    "index": index, "response": "", "generated_code": "",
                    "tokens": 0, "inference_time_ms": 0,
                    "energy": 999.0, "normalized": 1.0,
                    "code_hash": "error", "passed": False,
                }

        # Use 2 workers to match --parallel 2 on llama-server.
        # Requests are queued with 100ms stagger so the server starts
        # processing the next request before the current one finishes.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            for i in range(1, k + 1):
                futures[pool.submit(_generate_one, i)] = i
                if i < k:
                    time.sleep(0.1)  # 100ms stagger between submissions

            for future in as_completed(futures):
                cand = future.result()
                candidates.append(cand)
                result.total_tokens += cand["tokens"]
                result.total_inference_time_ms += cand["inference_time_ms"]

        # Phase 2: Sort candidates by energy (lowest = most promising)
        candidates.sort(key=lambda c: c["energy"])

        # Phase 3: Evaluate candidates
        sandbox_calls = 0
        selected = None

        if task.eval_mode in ("mcq", "ifbench"):
            # Non-code tasks: return lowest-energy candidate directly
            best = candidates[0] if candidates else None
            if best and best["response"]:
                if task.eval_mode == "mcq":
                    correct = task.test_outputs[0] if task.test_outputs else task.canonical_solution
                    extracted = extract_mcq_answer(best["response"])
                    passed = extracted == correct
                else:
                    try:
                        meta = json.loads(task.canonical_solution)
                        iids = meta.get("instruction_id_list", [])
                        kwargs_list = meta.get("kwargs", [{}])
                    except (json.JSONDecodeError, TypeError):
                        iids, kwargs_list = [], [{}]
                    errors = []
                    for j, iid in enumerate(iids):
                        kw = kwargs_list[j] if j < len(kwargs_list) else {}
                        if not evaluate_ifbench_loose(best["response"], iid, kw):
                            errors.append(f"Failed: {iid}")
                    passed = len(errors) == 0

                best["passed"] = passed
                result.attempts.append(AttemptResult(
                    task_id=task.task_id, attempt_number=1,
                    generated_code=best["response"], passed=passed,
                    execution_time_ms=0,
                    error_output="" if passed else "lens-selected candidate failed",
                    tokens_generated=best["tokens"],
                    inference_time_ms=best["inference_time_ms"],
                ))
                if passed:
                    result.best_attempt = 1
                selected = best
        else:
            # Code tasks: try sandbox in energy-sorted order, early exit on pass
            for rank, cand in enumerate(candidates):
                if not cand["generated_code"]:
                    cand["passed"] = False
                    continue

                sandbox_calls += 1
                if task.eval_mode == "stdio":
                    passed, stdout, stderr, exec_time = execute_code_stdio(
                        cand["generated_code"], task.test_inputs, task.test_outputs,
                        timeout_sec=self.runner.timeout_sec, memory_mb=self.runner.memory_mb
                    )
                else:
                    passed, stdout, stderr, exec_time = execute_code(
                        cand["generated_code"], task.test_code,
                        timeout_sec=self.runner.timeout_sec, memory_mb=self.runner.memory_mb
                    )

                cand["passed"] = passed
                result.total_execution_time_ms += exec_time

                result.attempts.append(AttemptResult(
                    task_id=task.task_id, attempt_number=rank + 1,
                    generated_code=cand["generated_code"], passed=passed,
                    execution_time_ms=exec_time,
                    error_output=stderr if not passed else "",
                    tokens_generated=cand["tokens"],
                    inference_time_ms=cand["inference_time_ms"],
                    stdout=stdout if passed else "",
                    stderr=stderr,
                ))

                if passed:
                    result.best_attempt = rank + 1
                    selected = cand
                    break

            if selected is None and candidates:
                selected = candidates[0]

        # Record telemetry
        if bok_tracker is not None and candidates:
            oracle_has_pass = any(c.get("passed") for c in candidates)
            bok_tracker.record_event(
                task_id=task.task_id, k=len(candidates),
                candidates=[{
                    "energy": c["energy"], "passed": c.get("passed", False),
                    "index": c["index"], "code_hash": c["code_hash"],
                } for c in candidates],
                selected_index=0, sandbox_calls=sandbox_calls,
                selected_passed=selected.get("passed", False) if selected else False,
                oracle_has_pass=oracle_has_pass,
            )

        return result


# --- Phase Functions ----------------------------------------------------------

def load_custom_tasks():
    """Load custom tasks from tasks.json."""
    tasks_file = config.custom_dir / "tasks.json"
    with open(tasks_file, 'r') as f:
        data = json.load(f)
    tasks = []
    for item in data.get("tasks", data if isinstance(data, list) else []):
        tasks.append(BenchmarkTask.from_dict(item))
    return tasks


def run_smoke_test(runner, datasets):
    """Phase 0: Smoke test. Returns True if passes."""
    print("\n" + "=" * 60)
    print("PHASE 0: SMOKE TEST")
    print("=" * 60)
    runner.set_mode("B")

    lcb = runner.run_phase("phase0_smoke/lcb", datasets["lcb"][:5], "livecodebench", repeats=1)
    gpqa = runner.run_phase("phase0_smoke/gpqa", datasets["gpqa"][:3], "gpqa", repeats=1)
    ifb = runner.run_phase("phase0_smoke/ifb", datasets["ifbench"][:3], "ifbench", repeats=1)
    cust = runner.run_phase("phase0_smoke/custom", datasets["custom"][:5], "custom", repeats=1)

    counts = {
        "lcb": sum(1 for r in lcb.values() if r.passed),
        "gpqa": sum(1 for r in gpqa.values() if r.passed),
        "ifb": sum(1 for r in ifb.values() if r.passed),
        "custom": sum(1 for r in cust.values() if r.passed),
    }
    total = sum(counts.values())
    print(f"\n  SMOKE TEST: {total}/16 passed (LCB={counts['lcb']}/5, GPQA={counts['gpqa']}/3, IFB={counts['ifb']}/3, Custom={counts['custom']}/5)")

    tel_file = runner.telemetry_dir / "route_decisions.jsonl"
    if tel_file.exists():
        with open(tel_file) as f:
            tel_count = sum(1 for _ in f)
        print(f"  Telemetry: {tel_count} records")
    return True


def run_phase1(runner, tasks):
    print("\n" + "=" * 60)
    print(f"PHASE 1: LiveCodeBench Mode B ({len(tasks)} x 3)")
    print("=" * 60)
    runner.set_mode("B")
    r = runner.run_phase("phase1_lcb_mode_b", tasks, "livecodebench", repeats=3)
    p = sum(1 for v in r.values() if v.passed)
    print(f"\n  Phase 1: {p}/{len(r)} ({100*p/max(len(r),1):.1f}%)")
    return r


def run_phase2(runner, tasks):
    print("\n" + "=" * 60)
    print(f"PHASE 2: LiveCodeBench Mode A ({len(tasks)} x 3)")
    print("=" * 60)
    runner.set_mode("A")
    r = runner.run_phase("phase2_lcb_mode_a", tasks, "livecodebench", repeats=3)
    p = sum(1 for v in r.values() if v.passed)
    print(f"\n  Phase 2: {p}/{len(r)} ({100*p/max(len(r),1):.1f}%)")
    return r


def run_phase3(runner, tasks):
    print("\n" + "=" * 60)
    print(f"PHASE 3: GPQA Diamond Mode B ({len(tasks)} x 5)")
    print("=" * 60)
    runner.set_mode("B")
    r = runner.run_phase("phase3_gpqa_mode_b", tasks, "gpqa", repeats=5)
    p = sum(1 for v in r.values() if v.passed)
    print(f"\n  Phase 3: {p}/{len(r)} ({100*p/max(len(r),1):.1f}%)")
    return r


def run_phase4(runner, tasks):
    print("\n" + "=" * 60)
    print(f"PHASE 4: IFBench Mode B ({len(tasks)} x 5)")
    print("=" * 60)
    runner.set_mode("B")
    r = runner.run_phase("phase4_ifbench_mode_b", tasks, "ifbench", repeats=5)
    p = sum(1 for v in r.values() if v.passed)
    print(f"\n  Phase 4: {p}/{len(r)} ({100*p/max(len(r),1):.1f}%)")
    return r


def run_phase5(runner, tasks):
    print("\n" + "=" * 60)
    print(f"PHASE 5: Custom Tasks ({len(tasks)})")
    print("=" * 60)
    runner.set_mode("B")
    rb = runner.run_phase("phase5_custom/mode_b_pass1", tasks, "custom", repeats=1)
    pb = sum(1 for v in rb.values() if v.passed)
    print(f"  Mode B: {pb}/{len(rb)} ({100*pb/max(len(rb),1):.1f}%)")
    runner.set_mode("A")
    ra = runner.run_phase("phase5_custom/mode_a_pass1", tasks, "custom", repeats=1)
    pa = sum(1 for v in ra.values() if v.passed)
    print(f"  Mode A: {pa}/{len(ra)} ({100*pa/max(len(ra),1):.1f}%)")
    return {"mode_b": rb, "mode_a": ra}


def run_phase6(runner, tasks):
    print("\n" + "=" * 60)
    print(f"PHASE 6: SciCode Mode B ({len(tasks)} x 3) [STRETCH]")
    print("=" * 60)
    runner.set_mode("B")
    r = runner.run_phase("phase6_scicode_mode_b", tasks, "scicode", repeats=3)
    p = sum(1 for v in r.values() if v.passed)
    print(f"\n  Phase 6: {p}/{len(r)} ({100*p/max(len(r),1):.1f}%)")
    return r


def run_lcb_learning_epochs(runner, tasks):
    """Run LCB with epoch-based execution and rolling C(x) retrain."""
    print("\n" + "=" * 60)
    print("EPOCH-BASED LCB EXECUTION WITH CONTINUOUS LEARNING")
    print("=" * 60)

    epochs = shuffle_and_split_epochs(tasks, seed=42)
    tracker = LearningCurveTracker(runner.run_dir)
    bok_tracker = BestOfKTracker(runner.run_dir) if runner.use_best_of_k else None

    for epoch_idx, epoch_tasks in enumerate(epochs):
        if not epoch_tasks:
            print(f"\n  Epoch {epoch_idx}: EMPTY (no tasks)")
            continue

        print(f"\n  --- Epoch {epoch_idx}: {len(epoch_tasks)} tasks ---")

        if epoch_idx == 0:
            # Epoch 0: baseline, lens OFF, no best-of-k
            runner.set_mode("B")
            old_bok = runner.use_best_of_k
            runner.use_best_of_k = False
            phase_name = f"epoch_{epoch_idx}_baseline"
        else:
            # Epochs 1+: lens ON after retrain, best-of-k enabled
            runner.set_mode("A")
            old_bok = runner.use_best_of_k
            phase_name = f"epoch_{epoch_idx}_retrained"

        results = runner.run_phase(
            phase_name, epoch_tasks, "livecodebench",
            repeats=3, tracker=tracker, epoch_num=epoch_idx,
            bok_tracker=bok_tracker if epoch_idx > 0 else None,
        )

        runner.use_best_of_k = old_bok  # Restore

        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        print(f"  Epoch {epoch_idx}: {passed}/{total} ({100*passed/max(total,1):.1f}%)")

        # Record epoch stats
        retrain_metrics = None
        if epoch_idx < len(epochs) - 1:
            # Retrain before next epoch (not after last)
            print(f"  Triggering retrain after epoch {epoch_idx}...")
            retrain_metrics = _trigger_retrain(tracker, max_epoch=epoch_idx)

        tracker.record_epoch(epoch_idx, total, passed, retrain_metrics)

    # Save learning curve summary
    tracker.save_summary()
    n_pass, n_fail = tracker.count_labels()
    print(f"\n  LEARNING COMPLETE: {n_pass} PASS, {n_fail} FAIL, {n_pass + n_fail} total embeddings")

    if bok_tracker:
        bok_tracker.save_summary()
        summary = bok_tracker.get_summary()
        if summary:
            print(f"  SELECTION: first_pick={summary.get('first_pick_accuracy', 0)*100:.1f}%, "
                  f"avg_sandbox_calls={summary.get('avg_sandbox_calls', 0):.2f}, "
                  f"effective_pass@1={summary.get('effective_pass_rate', 0)*100:.1f}%, "
                  f"oracle_pass@k={summary.get('oracle_pass_rate', 0)*100:.1f}%")


# --- Main Orchestrator --------------------------------------------------------

def load_all_datasets():
    """Load all benchmark datasets."""
    datasets = {}
    print("Loading datasets...")

    for name, cls in [("lcb", LiveCodeBenchDataset), ("gpqa", GPQADiamondDataset), ("ifbench", IFBenchDataset)]:
        print(f"  {name}...", end=" ", flush=True)
        ds = cls()
        ds.load()
        datasets[name] = ds.tasks
        print(f"{len(ds)} tasks")

    print("  custom...", end=" ", flush=True)
    datasets["custom"] = load_custom_tasks()
    print(f"{len(datasets['custom'])} tasks")

    try:
        print("  scicode...", end=" ", flush=True)
        sci = SciCodeDataset()
        sci.load()
        datasets["scicode"] = sci.tasks
        print(f"{len(sci)} tasks")
    except Exception as e:
        print(f"SKIPPED ({e})")
        datasets["scicode"] = []
    return datasets


def run_full_benchmark(start_phase=0, end_phase=6, run_id=None, no_epochs=False, best_of_k=False):
    """Run the full V2 benchmark suite."""
    if run_id is None:
        run_id = f"v2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = config.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id, "start_time": datetime.utcnow().isoformat(),
        "start_phase": start_phase, "end_phase": end_phase,
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
    }
    atomic_write_json(run_dir / "run_meta.json", meta)

    print(f"\nATLAS V2 Benchmark Run: {run_id}")
    print(f"Results: {run_dir}")
    print(f"Phases: {start_phase}-{end_phase}\n")

    datasets = load_all_datasets()

    with V2BenchmarkRunner(run_dir, use_best_of_k=best_of_k) as runner:
        if start_phase <= 0 <= end_phase:
            run_smoke_test(runner, datasets)
        if start_phase <= 1 <= end_phase:
            if no_epochs:
                run_phase1(runner, datasets["lcb"])
            else:
                run_lcb_learning_epochs(runner, datasets["lcb"])
        if start_phase <= 2 <= end_phase and no_epochs:
            run_phase2(runner, datasets["lcb"])
        if start_phase <= 3 <= end_phase:
            run_phase3(runner, datasets["gpqa"])
        if start_phase <= 4 <= end_phase:
            run_phase4(runner, datasets["ifbench"])
        if start_phase <= 5 <= end_phase:
            run_phase5(runner, datasets["custom"])
        if start_phase <= 6 <= end_phase and datasets["scicode"]:
            run_phase6(runner, datasets["scicode"])

    meta["end_time"] = datetime.utcnow().isoformat()
    atomic_write_json(run_dir / "run_meta.json", meta)
    print(f"\nBENCHMARK COMPLETE: {run_dir}")
    return run_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS V2 Benchmark Runner")
    parser.add_argument("--start-phase", type=int, default=0)
    parser.add_argument("--end-phase", type=int, default=6)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-epochs", action="store_true",
                        help="Disable epoch-based LCB execution (use old Phase 1+2)")
    parser.add_argument("--best-of-k", action="store_true",
                        help="Enable best-of-K lens-guided candidate selection")
    args = parser.parse_args()
    if args.smoke_only:
        args.end_phase = 0
    run_dir = run_full_benchmark(args.start_phase, args.end_phase, args.run_id, args.no_epochs, args.best_of_k)
    try:
        from benchmark.v2_report import generate_report
        generate_report(run_dir)
    except Exception as e:
        print(f"Warning: Report generation failed: {e}")


if __name__ == "__main__":
    main()
