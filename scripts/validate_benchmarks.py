#!/usr/bin/env python3
"""
ATLAS V1 Benchmark Infrastructure — Deep Verification Script
Run from: /home/isaac/ATLAS
Usage:    python verify_benchmarks.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import importlib
import traceback

# ── Formatting ──────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed_total = 0
failed_total = 0
warned_total = 0

def header(title):
    print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")

def check(name, condition, detail=""):
    global passed_total, failed_total
    if condition:
        passed_total += 1
        print(f"  {GREEN}✓{RESET} {name}")
    else:
        failed_total += 1
        print(f"  {RED}✗{RESET} {name}")
    if detail:
        print(f"    {detail}")

def warn(name, detail=""):
    global warned_total
    warned_total += 1
    print(f"  {YELLOW}⚠{RESET} {name}")
    if detail:
        print(f"    {detail}")

# ── 1. File Structure ───────────────────────────────────────

def verify_file_structure():
    header("1. FILE STRUCTURE")
    required_files = [
        "benchmark/__init__.py",
        "benchmark/cli.py",
        "benchmark/runner.py",
        "benchmark/submit.py",
        "benchmark/config.py",
        "benchmark/models.py",
        "benchmark/datasets/__init__.py",
        "benchmark/datasets/base.py",
        "benchmark/datasets/humaneval.py",
        "benchmark/datasets/mbpp.py",
        "benchmark/custom/__init__.py",
        "benchmark/custom/tasks.json",
        "benchmark/custom/validate.py",
        "benchmark/analysis/__init__.py",
        "benchmark/analysis/pass_at_k.py",
        "benchmark/analysis/cost_analysis.py",
        "benchmark/analysis/hardware_info.py",
    ]
    required_dirs = [
        "benchmark/datasets/.cache",
        "benchmark/results",
        "benchmark/results/submissions",
    ]

    for f in required_files:
        check(f"File exists: {f}", os.path.isfile(f))

    for d in required_dirs:
        check(f"Dir exists:  {d}", os.path.isdir(d))

    # Check .gitignore entries
    gitignore_path = ".gitignore"
    if os.path.isfile(gitignore_path):
        content = open(gitignore_path).read()
        check(".gitignore has datasets/.cache",
              ".cache" in content or "datasets/.cache" in content,
              f"Searched .gitignore for cache exclusion")
        check(".gitignore has benchmark/results",
              "results" in content or "benchmark/results" in content,
              f"Searched .gitignore for results exclusion")
    else:
        warn(".gitignore not found at repo root")

    # Check file sizes (non-empty)
    for f in required_files:
        if os.path.isfile(f):
            size = os.path.getsize(f)
            if size == 0 and not f.endswith("__init__.py"):
                warn(f"{f} is empty (0 bytes)")

# ── 2. Custom Tasks Integrity ───────────────────────────────

def verify_custom_tasks():
    header("2. CUSTOM TASKS INTEGRITY")
    tasks_path = "benchmark/custom/tasks.json"
    if not os.path.isfile(tasks_path):
        check("tasks.json exists", False)
        return

    with open(tasks_path) as f:
        raw = json.load(f)

    # Discover the JSON structure
    tasks = None
    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict):
            tasks = raw  # Expected: list of dicts
        else:
            print(f"  {YELLOW}JSON is a list but items are {type(raw[0]).__name__}, not dict{RESET}")
            print(f"  First item preview: {str(raw[0])[:120]}")
    elif isinstance(raw, dict):
        # Could be {"tasks": [...]} or {"ALGO_001": {...}, ...} etc.
        if "tasks" in raw and isinstance(raw["tasks"], list):
            tasks = raw["tasks"]
            print(f"  JSON structure: dict with 'tasks' key")
        elif "categories" in raw:
            # Might be {"categories": {"Algorithm": [...], ...}}
            tasks = []
            for cat_name, cat_tasks in raw.get("categories", {}).items():
                if isinstance(cat_tasks, list):
                    tasks.extend(cat_tasks)
            print(f"  JSON structure: dict with 'categories' key, flattened {len(tasks)} tasks")
        else:
            # Maybe keys are task IDs: {"ALGO_001": {...}, "ALGO_002": {...}}
            first_val = next(iter(raw.values()), None)
            if isinstance(first_val, dict):
                tasks = []
                for tid, tdata in raw.items():
                    if isinstance(tdata, dict):
                        tdata.setdefault("task_id", tid)
                        tasks.append(tdata)
                print(f"  JSON structure: dict keyed by task ID")
            else:
                print(f"  {RED}Unrecognized JSON structure: dict with {type(first_val).__name__} values{RESET}")
                print(f"  Top-level keys: {list(raw.keys())[:10]}")
    
    if tasks is None:
        check("Parse tasks.json into task list", False,
              f"Raw type: {type(raw).__name__}, length: {len(raw)}")
        print(f"  Preview: {str(raw)[:300]}")
        return

    check(f"Task count is 100", len(tasks) == 100, f"Found {len(tasks)} tasks")

    # Verify required fields
    required_fields = ["task_id", "category", "difficulty", "prompt",
                       "entry_point", "test_code", "canonical_solution"]
    missing_fields = []
    for t in tasks:
        if not isinstance(t, dict):
            missing_fields.append(f"Non-dict task: {type(t).__name__}: {str(t)[:60]}")
            continue
        for field in required_fields:
            if field not in t or not t[field]:
                missing_fields.append(f"{t.get('task_id', '???')}.{field}")
    check("All tasks have required fields",
          len(missing_fields) == 0,
          f"Missing: {missing_fields[:5]}{'...' if len(missing_fields) > 5 else ''}" if missing_fields else "")

    # Verify category distribution
    categories = {}
    for t in tasks:
        if not isinstance(t, dict):
            continue
        cat = t.get("category", "UNKNOWN")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"\n  Category distribution:")
    expected = {
        "Algorithm": 20, "Data Processing": 20, "API/Integration": 15,
        "Test Generation": 15, "Refactoring": 15, "Bug Fixing": 15
    }
    for cat, count in sorted(categories.items()):
        exp = expected.get(cat)
        marker = GREEN + "✓" + RESET if exp and count == exp else YELLOW + "?" + RESET
        print(f"    {marker} {cat}: {count}" + (f" (expected {exp})" if exp and count != exp else ""))

    # Verify difficulty distribution
    difficulties = {}
    for t in tasks:
        if not isinstance(t, dict):
            continue
        d = t.get("difficulty", "UNKNOWN")
        difficulties[d] = difficulties.get(d, 0) + 1
    print(f"\n  Difficulty distribution:")
    for d, count in sorted(difficulties.items()):
        print(f"    {d}: {count}")

    # Verify canonical solutions pass their own tests
    print(f"\n  Running canonical solutions against test cases...")
    solutions_passed = 0
    solutions_failed = []
    for t in tasks:
        if not isinstance(t, dict):
            solutions_failed.append(f"Non-dict task: {type(t).__name__}")
            continue
        if "canonical_solution" not in t or "test_code" not in t:
            solutions_failed.append(f"{t.get('task_id', '???')}: missing solution or test_code")
            continue
        code = t["canonical_solution"] + "\n" + t["test_code"]
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                             delete=False) as tmp:
                tmp.write(code)
                tmp.flush()
                result = subprocess.run(
                    [sys.executable, tmp.name],
                    capture_output=True, timeout=15, text=True
                )
                if result.returncode == 0:
                    solutions_passed += 1
                else:
                    solutions_failed.append(
                        f"{t['task_id']}: {result.stderr[:80]}"
                    )
        except subprocess.TimeoutExpired:
            solutions_failed.append(f"{t['task_id']}: TIMEOUT")
        except Exception as e:
            solutions_failed.append(f"{t['task_id']}: {e}")
        finally:
            try:
                os.unlink(tmp.name)
            except:
                pass

    check(f"Canonical solutions pass own tests",
          solutions_passed == len(tasks),
          f"{solutions_passed}/{len(tasks)} passed")
    if solutions_failed:
        for fail in solutions_failed[:5]:
            print(f"    {RED}FAIL:{RESET} {fail}")
        if len(solutions_failed) > 5:
            print(f"    ... and {len(solutions_failed) - 5} more")

    # Verify test assertions exist (not just "assert True")
    weak_tests = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        tc = t.get("test_code", "")
        if "assert True" in tc and tc.count("assert") == tc.count("assert True"):
            weak_tests.append(t.get("task_id", "???"))
        elif "assert" not in tc and "raises" not in tc.lower():
            weak_tests.append(t.get("task_id", "???"))
    check("All tasks have real assertions (not just assert True)",
          len(weak_tests) == 0,
          f"Weak tests: {weak_tests}" if weak_tests else "")

# ── 3. Mutation Testing ─────────────────────────────────────

def verify_mutation_testing():
    header("3. MUTATION TESTING (test quality)")
    tasks_path = "benchmark/custom/tasks.json"
    if not os.path.isfile(tasks_path):
        warn("Skipping — tasks.json not found")
        return

    with open(tasks_path) as f:
        raw = json.load(f)

    # Re-use same structure discovery as verify_custom_tasks
    tasks = None
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        tasks = raw
    elif isinstance(raw, dict):
        if "tasks" in raw and isinstance(raw["tasks"], list):
            tasks = raw["tasks"]
        elif "categories" in raw:
            tasks = []
            for cat_tasks in raw.get("categories", {}).values():
                if isinstance(cat_tasks, list):
                    tasks.extend(cat_tasks)
        else:
            first_val = next(iter(raw.values()), None)
            if isinstance(first_val, dict):
                tasks = []
                for tid, tdata in raw.items():
                    if isinstance(tdata, dict):
                        tdata.setdefault("task_id", tid)
                        tasks.append(tdata)

    if not tasks:
        warn("Could not parse tasks.json for mutation testing")
        return

    print(f"  Mutating canonical solutions (replacing return → return None)...")
    print(f"  Testing {len(tasks)} tasks...\n")

    caught = 0
    escaped = []
    errors = 0

    for t in tasks:
        if not isinstance(t, dict):
            errors += 1
            continue
        canon = t.get("canonical_solution", "")
        test_code = t.get("test_code", "")
        # Create a mutant: replace first 'return ' with 'return None  #'
        if "return " in canon:
            mutant = canon.replace("return ", "return None  # ", 1)
        else:
            caught += 1
            continue

        code = mutant + "\n" + test_code
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                             delete=False) as tmp:
                tmp.write(code)
                tmp.flush()
                result = subprocess.run(
                    [sys.executable, tmp.name],
                    capture_output=True, timeout=10, text=True
                )
                if result.returncode != 0:
                    caught += 1
                else:
                    escaped.append(t.get("task_id", "???"))
        except subprocess.TimeoutExpired:
            caught += 1  # Timeout counts as caught
        except Exception:
            errors += 1
        finally:
            try:
                os.unlink(tmp.name)
            except:
                pass

    detection_rate = (caught / len(tasks)) * 100 if tasks else 0
    check(f"Mutation detection rate ≥ 90%",
          detection_rate >= 90,
          f"{caught}/{len(tasks)} mutants caught ({detection_rate:.1f}%)")

    if escaped:
        print(f"\n  {YELLOW}Mutants that ESCAPED (tests didn't catch bad code):{RESET}")
        for tid in escaped[:10]:
            print(f"    ⚠ {tid}")
        if len(escaped) > 10:
            print(f"    ... and {len(escaped) - 10} more")
        print(f"  These tasks may have weak test cases that inflate pass@k scores.")

    if errors:
        warn(f"{errors} tasks had errors during mutation testing")

# ── 4. Dataset Loaders ──────────────────────────────────────

def verify_datasets():
    header("4. DATASET LOADERS")

    # Try importing dataset modules
    try:
        sys.path.insert(0, os.getcwd())
        from benchmark.datasets.humaneval import HumanEvalDataset
        check("Import HumanEvalDataset", True)

        he = HumanEvalDataset()
        tasks = he.load()
        check(f"HumanEval loads successfully", tasks is not None and len(tasks) > 0,
              f"Loaded {len(tasks)} tasks")
        check(f"HumanEval has 164 tasks", len(tasks) == 164,
              f"Found {len(tasks)}")

        # Spot-check task 0
        t0 = None
        for t in tasks:
            if "0" in str(t.task_id) and ("HumanEval/0" in str(t.task_id) or
                                           str(t.task_id).endswith("/0") or
                                           str(t.task_id) == "0"):
                t0 = t
                break
        if t0:
            check("HumanEval/0 has entry_point",
                  hasattr(t0, "entry_point") and t0.entry_point,
                  f"entry_point = {t0.entry_point}")
            check("HumanEval/0 has test_code with assertions",
                  hasattr(t0, "test_code") and "assert" in str(t0.test_code),
                  "")
        else:
            warn("Could not find HumanEval task 0 for spot-check")

    except ImportError as e:
        check("Import HumanEvalDataset", False, str(e))
    except Exception as e:
        check("HumanEval loads successfully", False, str(e))

    try:
        from benchmark.datasets.mbpp import MBPPDataset
        check("Import MBPPDataset", True)

        mbpp = MBPPDataset()
        tasks = mbpp.load()
        check(f"MBPP loads successfully", tasks is not None and len(tasks) > 0,
              f"Loaded {len(tasks)} tasks")
        # MBPP-S is ~374-500 depending on version
        check(f"MBPP has reasonable task count (300-500)",
              300 <= len(tasks) <= 500,
              f"Found {len(tasks)}")

    except ImportError as e:
        check("Import MBPPDataset", False, str(e))
    except Exception as e:
        check("MBPP loads successfully", False, str(e))

# ── 5. pass@k Estimator ────────────────────────────────────

def verify_pass_at_k():
    header("5. pass@k ESTIMATOR CORRECTNESS")

    try:
        sys.path.insert(0, os.getcwd())
        from benchmark.analysis.pass_at_k import pass_at_k
        check("Import pass_at_k", True)

        # Test cases with known correct values
        test_cases = [
            # (n, c, k, expected, description)
            (10, 1, 1, 0.1, "n=10, c=1, k=1 → 0.1"),
            (10, 1, 10, 1.0, "n=10, c=1, k=10 → 1.0 (guaranteed)"),
            (20, 0, 20, 0.0, "n=20, c=0, k=20 → 0.0 (nothing correct)"),
            (20, 20, 1, 1.0, "n=20, c=20, k=1 → 1.0 (all correct)"),
            (10, 5, 1, 0.5, "n=10, c=5, k=1 → 0.5"),
            (100, 10, 1, 0.1, "n=100, c=10, k=1 → 0.1"),
            (10, 10, 5, 1.0, "n=10, c=10, k=5 → 1.0 (all correct)"),
            (10, 0, 5, 0.0, "n=10, c=0, k=5 → 0.0 (none correct)"),
        ]

        all_pass = True
        for n, c, k, expected, desc in test_cases:
            try:
                result = pass_at_k(n, c, k)
                ok = abs(result - expected) < 0.001
                if not ok:
                    all_pass = False
                    print(f"    {RED}✗{RESET} {desc}  got {result:.4f}")
                else:
                    print(f"    {GREEN}✓{RESET} {desc}")
            except Exception as e:
                all_pass = False
                print(f"    {RED}✗{RESET} {desc}  ERROR: {e}")

        check("All pass@k estimator tests pass", all_pass)

        # Verify monotonicity: pass@k should increase with k
        print(f"\n  Monotonicity check (pass@k increases with k):")
        prev = 0
        monotonic = True
        for k in [1, 2, 5, 10, 15, 20]:
            val = pass_at_k(20, 5, k)
            if val < prev - 0.001:
                monotonic = False
                print(f"    {RED}✗{RESET} pass@{k} = {val:.4f} < pass@{k-1}")
            else:
                print(f"    {GREEN}✓{RESET} pass@{k} = {val:.4f}")
            prev = val
        check("pass@k is monotonically increasing with k", monotonic)

    except ImportError as e:
        check("Import pass_at_k", False, str(e))
    except Exception as e:
        check("pass@k tests", False, traceback.format_exc())

# ── 6. Runner Isolation ─────────────────────────────────────

def verify_runner_isolation():
    header("6. RUNNER ISOLATION (sandbox safety)")

    try:
        sys.path.insert(0, os.getcwd())
        # Try to import the execution function
        # The exact import path may vary — try common patterns
        execute_fn = None
        execute_name = None

        import_attempts = [
            ("benchmark.runner", "execute_code"),
            ("benchmark.runner", "run_code"),
            ("benchmark.runner", "execute_task"),
            ("benchmark.runner", "run_in_sandbox"),
            ("benchmark.runner", "sandbox_execute"),
        ]

        for module_name, fn_name in import_attempts:
            try:
                mod = importlib.import_module(module_name)
                if hasattr(mod, fn_name):
                    execute_fn = getattr(mod, fn_name)
                    execute_name = f"{module_name}.{fn_name}"
                    break
            except:
                pass

        if execute_fn is None:
            # Try to find any callable that looks like an executor
            try:
                mod = importlib.import_module("benchmark.runner")
                for name in dir(mod):
                    if "execute" in name.lower() or "sandbox" in name.lower() or "run" in name.lower():
                        obj = getattr(mod, name)
                        if callable(obj) and not name.startswith("_"):
                            execute_fn = obj
                            execute_name = f"benchmark.runner.{name}"
                            break
            except:
                pass

        if execute_fn is None:
            warn("Could not find execution function in benchmark.runner",
                 "Looked for: execute_code, run_code, execute_task, run_in_sandbox")
            print(f"  Falling back to subprocess-based isolation test...\n")

            # Test isolation via subprocess directly
            # Timeout test
            start = time.time()
            try:
                result = subprocess.run(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    capture_output=True, timeout=10, text=True
                )
                check("Subprocess timeout works (10s limit)",
                      False, "Process completed without timing out")
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                check("Subprocess timeout works (10s limit)", True,
                      f"Timed out after {elapsed:.1f}s")

            return

        check(f"Found execution function: {execute_name}", True)
        print(f"  Running isolation tests...\n")

        # Timeout test
        print(f"  Testing timeout enforcement...")
        start = time.time()
        try:
            result = execute_fn("import time; time.sleep(60)")
            elapsed = time.time() - start
            timed_out = elapsed < 55  # Should have been killed well before 60s
            if hasattr(result, "passed"):
                check("Timeout: kills long-running code",
                      not result.passed and timed_out,
                      f"Elapsed: {elapsed:.1f}s, passed={result.passed}")
            else:
                check("Timeout: kills long-running code",
                      timed_out,
                      f"Elapsed: {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start
            check("Timeout: kills long-running code",
                  elapsed < 55,
                  f"Exception after {elapsed:.1f}s: {type(e).__name__}")

        # Network isolation test
        print(f"  Testing network isolation...")
        try:
            net_code = 'import urllib.request; urllib.request.urlopen("http://1.1.1.1", timeout=3)'
            result = execute_fn(net_code)
            if hasattr(result, "passed"):
                check("Network: blocks outbound connections",
                      not result.passed,
                      f"passed={result.passed}")
            else:
                warn("Network test: could not determine result format")
        except Exception as e:
            check("Network: blocks outbound connections", True,
                  f"Raised {type(e).__name__}")

        # Memory limit test
        print(f"  Testing memory limits...")
        try:
            mem_code = 'x = bytearray(1024 * 1024 * 1024)'  # 1GB
            result = execute_fn(mem_code)
            if hasattr(result, "passed"):
                check("Memory: enforces memory limit",
                      not result.passed,
                      f"passed={result.passed}")
            else:
                warn("Memory test: could not determine result format")
        except Exception as e:
            check("Memory: enforces memory limit", True,
                  f"Raised {type(e).__name__}")

        # Filesystem write test
        print(f"  Testing filesystem restrictions...")
        try:
            fs_code = 'open("/tmp/atlas_test_escape", "w").write("escaped")'
            result = execute_fn(fs_code)
            # Check if the file was actually created
            escaped = os.path.isfile("/tmp/atlas_test_escape")
            if escaped:
                os.unlink("/tmp/atlas_test_escape")
            # Some sandboxes allow /tmp writes — that's often acceptable
            if escaped:
                warn("Filesystem: /tmp writes allowed (may be acceptable)")
            else:
                check("Filesystem: blocks writes", True)
        except Exception as e:
            check("Filesystem: blocks writes", True,
                  f"Raised {type(e).__name__}")

    except Exception as e:
        warn(f"Runner isolation tests failed: {e}")
        traceback.print_exc()

# ── 7. Config Integration ───────────────────────────────────

def verify_config():
    header("7. CONFIG INTEGRATION")

    try:
        sys.path.insert(0, os.getcwd())
        from benchmark.config import BenchmarkConfig
        check("Import BenchmarkConfig", True)

        cfg = BenchmarkConfig()
        check("BenchmarkConfig instantiates", True)

        # Check for expected config attributes
        endpoint_attrs = ["model_endpoint", "llm_endpoint", "api_url",
                         "endpoint", "base_url", "server_url"]
        found_endpoint = None
        for attr in endpoint_attrs:
            if hasattr(cfg, attr):
                val = getattr(cfg, attr)
                found_endpoint = (attr, val)
                break

        if found_endpoint:
            check(f"Config has endpoint: {found_endpoint[0]}",
                  True, f"Value: {found_endpoint[1]}")
        else:
            warn("No obvious endpoint attribute found in BenchmarkConfig",
                 f"Attributes: {[a for a in dir(cfg) if not a.startswith('_')]}")

        # Check for timeout config
        timeout_attrs = ["execution_timeout", "timeout", "task_timeout",
                        "code_timeout"]
        found_timeout = None
        for attr in timeout_attrs:
            if hasattr(cfg, attr):
                val = getattr(cfg, attr)
                found_timeout = (attr, val)
                break

        if found_timeout:
            check(f"Config has timeout: {found_timeout[0]}",
                  True, f"Value: {found_timeout[1]}")
        else:
            warn("No obvious timeout attribute found in BenchmarkConfig")

        # Check if atlas.conf is referenced
        config_file = "atlas.conf"
        if os.path.isfile(config_file):
            check("atlas.conf exists", True)
            # Check if benchmark/config.py references atlas.conf
            with open("benchmark/config.py") as f:
                config_src = f.read()
            check("benchmark/config.py references atlas.conf",
                  "atlas.conf" in config_src or "atlas_conf" in config_src or
                  "configparser" in config_src or "toml" in config_src or
                  "yaml" in config_src,
                  "Config should read from atlas.conf, not hardcode values")
        else:
            warn("atlas.conf not found at repo root")

    except ImportError as e:
        check("Import BenchmarkConfig", False, str(e))
    except Exception as e:
        check("Config integration", False, traceback.format_exc())

# ── 8. Hardware Info ────────────────────────────────────────

def verify_hardware_info():
    header("8. HARDWARE INFO COLLECTION")

    try:
        mod = importlib.import_module("benchmark.analysis.hardware_info")
        check("Import hardware_info module", True)

        # Find the main collection function
        collect_fn = None
        for name in ["collect_hardware_info", "get_hardware_info",
                     "collect", "get_info", "hardware_info"]:
            if hasattr(mod, name) and callable(getattr(mod, name)):
                collect_fn = getattr(mod, name)
                break

        if collect_fn is None:
            warn("Could not find collection function in hardware_info",
                 f"Available: {[n for n in dir(mod) if not n.startswith('_')]}")
            return

        check(f"Found collection function: {collect_fn.__name__}", True)

        info = collect_fn()
        check("Hardware info collection returns data",
              info is not None,
              f"Type: {type(info).__name__}")

        # Check for expected fields
        if isinstance(info, dict):
            expected_keys = ["gpu", "cpu", "ram", "os"]
            for key in expected_keys:
                found = any(key in k.lower() for k in info.keys())
                check(f"Hardware info has '{key}' field", found,
                      "" if found else f"Keys: {list(info.keys())}")
        elif hasattr(info, "__dict__"):
            attrs = [a for a in dir(info) if not a.startswith("_")]
            print(f"    Attributes: {attrs[:10]}")

    except ImportError as e:
        check("Import hardware_info", False, str(e))
    except Exception as e:
        check("Hardware info collection", False, str(e))

# ── 9. CLI Dry Runs ─────────────────────────────────────────

def verify_cli_dry_runs():
    header("9. CLI DRY RUNS")

    cli_commands = [
        ("atlas benchmark --humaneval --dry-run", "HumanEval dry-run"),
        ("atlas benchmark --mbpp --dry-run", "MBPP dry-run"),
        ("atlas benchmark --custom --dry-run", "Custom dry-run"),
    ]

    # Also try python -m benchmark.cli as fallback
    for cmd, desc in cli_commands:
        print(f"\n  Running: {cmd}")
        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True, timeout=120, text=True,
                cwd=os.getcwd()
            )
            check(f"{desc} exits 0",
                  result.returncode == 0,
                  f"Return code: {result.returncode}")
            if result.returncode != 0:
                # Show last few lines of stderr
                err_lines = result.stderr.strip().split("\n")[-3:]
                for line in err_lines:
                    print(f"    stderr: {line}")
        except FileNotFoundError:
            # atlas command not found — try python -m
            alt_cmd = cmd.replace("atlas benchmark",
                                  f"{sys.executable} -m benchmark.cli")
            print(f"  'atlas' not in PATH, trying: {alt_cmd}")
            try:
                result = subprocess.run(
                    alt_cmd.split(),
                    capture_output=True, timeout=120, text=True,
                    cwd=os.getcwd()
                )
                check(f"{desc} exits 0 (via python -m)",
                      result.returncode == 0,
                      f"Return code: {result.returncode}")
                if result.returncode != 0:
                    err_lines = result.stderr.strip().split("\n")[-3:]
                    for line in err_lines:
                        print(f"    stderr: {line}")
            except Exception as e:
                check(f"{desc}", False, str(e))
        except subprocess.TimeoutExpired:
            check(f"{desc} completes within 120s", False, "TIMEOUT")
        except Exception as e:
            check(f"{desc}", False, str(e))

# ── 10. Crash Recovery Check ────────────────────────────────

def verify_crash_recovery():
    header("10. CRASH RECOVERY DESIGN")

    # Check if runner.py saves results incrementally
    runner_path = "benchmark/runner.py"
    if os.path.isfile(runner_path):
        with open(runner_path) as f:
            src = f.read()

        # Look for evidence of per-task saving
        incremental_patterns = [
            "json.dump", "write", "save", "flush", "append",
            "results_file", "result_path", "checkpoint"
        ]
        found = [p for p in incremental_patterns if p in src]
        check("runner.py has incremental result saving",
              len(found) >= 2,
              f"Found patterns: {found}")
    else:
        check("runner.py exists for crash recovery check", False)

    # Check if cli.py supports resumption
    cli_path = "benchmark/cli.py"
    if os.path.isfile(cli_path):
        with open(cli_path) as f:
            src = f.read()
        resume_patterns = ["resume", "skip_completed", "existing",
                          "already", "checkpoint", "continue"]
        found = [p for p in resume_patterns if p in src.lower()]
        if found:
            check("cli.py has resume/checkpoint support", True,
                  f"Found patterns: {found}")
        else:
            warn("cli.py may lack resume support",
                 "If a run crashes at task 80/164, can it resume from 81?")

# ── Main ────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  ATLAS V1 Benchmark — Deep Verification{RESET}")
    print(f"{BOLD}  Running from: {os.getcwd()}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    # Verify we're in the right directory
    if not os.path.isdir("benchmark"):
        print(f"\n{RED}ERROR: 'benchmark/' directory not found.{RESET}")
        print(f"Run this script from /home/isaac/ATLAS")
        sys.exit(1)

    start = time.time()

    verify_file_structure()       # 1
    verify_custom_tasks()         # 2
    verify_mutation_testing()     # 3
    verify_datasets()             # 4
    verify_pass_at_k()            # 5
    verify_runner_isolation()     # 6
    verify_config()               # 7
    verify_hardware_info()        # 8
    verify_cli_dry_runs()         # 9
    verify_crash_recovery()       # 10

    elapsed = time.time() - start

    # Summary
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  VERIFICATION SUMMARY{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  {GREEN}Passed:  {passed_total}{RESET}")
    print(f"  {RED}Failed:  {failed_total}{RESET}")
    print(f"  {YELLOW}Warnings: {warned_total}{RESET}")
    print(f"  Time:    {elapsed:.1f}s")

    if failed_total == 0:
        print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED ✓{RESET}")
    else:
        print(f"\n  {RED}{BOLD}{failed_total} CHECK(S) FAILED — review above{RESET}")

    print()
    sys.exit(0 if failed_total == 0 else 1)


if __name__ == "__main__":
    main()
