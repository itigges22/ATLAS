#!/usr/bin/env python3
"""ATLAS llama-server Cache Manager — proactive memory management.

Monitors llama-server RSS via kubectl and manages prompt cache
accumulation through two tiers:
  Tier 1: Slot erase (fast, no downtime) — clears KV prompt cache
  Tier 2: Pod restart (slow, ~60-90s) — when erase doesn't free enough

Modes:
  --daemon   Run as a continuous monitoring loop (default)
  --once     Run a single check-and-act cycle, then exit
  --status   Print current memory and slot status, then exit

Config: ATLAS_CACHE_MANAGER_* keys in atlas.conf
Telemetry: telemetry/cache_manager_events.jsonl

Requires: --slot-save-path on llama-server entrypoint for slot erase.

Usage:
    # Run alongside a benchmark
    python3 scripts/llama-cache-manager.py --daemon &

    # Single check
    python3 scripts/llama-cache-manager.py --once

    # Just check status
    python3 scripts/llama-cache-manager.py --status
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class CacheManagerConfig:
    enabled: bool = True
    soft_threshold_mb: int = 8192      # Tier 1: slot erase
    hard_threshold_mb: int = 10240     # Tier 2: pod restart
    check_interval_sec: int = 30
    erase_cooldown_sec: int = 60
    restart_cooldown_sec: int = 300
    llama_url: str = "http://localhost:32735"
    namespace: str = "atlas"
    deployment: str = "llama-server"
    pod_label: str = "app=llama-server"
    warmup_enabled: bool = True
    warmup_timeout_sec: int = 300
    telemetry_dir: str = ""
    dry_run: bool = False


def load_config(cli_args=None) -> CacheManagerConfig:
    """Load config from atlas.conf, env vars, and CLI overrides."""
    cfg = CacheManagerConfig()

    # Parse atlas.conf
    conf_path = PROJECT_ROOT / "atlas.conf"
    conf = {}
    if conf_path.exists():
        with open(conf_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    conf[k.strip()] = v.strip().strip('"').strip("'")

    # Map atlas.conf keys to config fields
    mapping = {
        'ATLAS_CACHE_MANAGER_ENABLED': ('enabled', lambda v: v.lower() in ('true', '1', 'yes')),
        'ATLAS_CACHE_MANAGER_SOFT_THRESHOLD_MB': ('soft_threshold_mb', int),
        'ATLAS_CACHE_MANAGER_HARD_THRESHOLD_MB': ('hard_threshold_mb', int),
        'ATLAS_CACHE_MANAGER_CHECK_INTERVAL_SEC': ('check_interval_sec', int),
        'ATLAS_CACHE_MANAGER_ERASE_COOLDOWN_SEC': ('erase_cooldown_sec', int),
        'ATLAS_CACHE_MANAGER_RESTART_COOLDOWN_SEC': ('restart_cooldown_sec', int),
        'ATLAS_CACHE_MANAGER_WARMUP_ENABLED': ('warmup_enabled', lambda v: v.lower() in ('true', '1', 'yes')),
        'ATLAS_LLAMA_NODEPORT': ('llama_url', lambda v: f"http://localhost:{v}"),
        'ATLAS_NAMESPACE': ('namespace', str),
    }

    for env_key, (attr, convert) in mapping.items():
        # Check atlas.conf first, then env vars
        val = conf.get(env_key) or os.environ.get(env_key)
        if val is not None:
            setattr(cfg, attr, convert(val))

    # CLI overrides
    if cli_args:
        if cli_args.llama_url:
            cfg.llama_url = cli_args.llama_url
        if cli_args.soft_threshold is not None:
            cfg.soft_threshold_mb = cli_args.soft_threshold
        if cli_args.hard_threshold is not None:
            cfg.hard_threshold_mb = cli_args.hard_threshold
        if cli_args.interval is not None:
            cfg.check_interval_sec = cli_args.interval
        if cli_args.telemetry_dir:
            cfg.telemetry_dir = cli_args.telemetry_dir
        if cli_args.dry_run:
            cfg.dry_run = True

    if not cfg.telemetry_dir:
        cfg.telemetry_dir = str(PROJECT_ROOT / "telemetry")

    return cfg


# ===========================================================================
# Telemetry
# ===========================================================================

def log_event(telemetry_dir: str, event: Dict[str, Any]):
    """Append event to JSONL telemetry file."""
    path = Path(telemetry_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / "cache_manager_events.jsonl"
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(filepath, 'a') as f:
        f.write(json.dumps(event) + '\n')


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [cache-mgr] [{level}] {msg}", flush=True)


# ===========================================================================
# Server Monitoring
# ===========================================================================

def get_server_memory_mb(namespace: str, pod_label: str) -> Optional[int]:
    """Get llama-server pod memory usage in MB via kubectl top."""
    try:
        result = subprocess.run(
            ["kubectl", "top", "pod", "-n", namespace,
             "-l", pod_label, "--no-headers"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                mem_str = parts[2]
                if mem_str.endswith('Mi'):
                    return int(mem_str[:-2])
                elif mem_str.endswith('Gi'):
                    return int(float(mem_str[:-2]) * 1024)
    except Exception:
        pass
    return None


def get_slot_info(llama_url: str) -> Optional[List[Dict]]:
    """Get slot state from llama-server."""
    try:
        req = urllib.request.Request(f"{llama_url}/slots")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception:
        return None


def server_is_healthy(llama_url: str) -> bool:
    """Quick health check."""
    try:
        req = urllib.request.Request(f"{llama_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('status') == 'ok'
    except Exception:
        return False


# ===========================================================================
# Actions
# ===========================================================================

def erase_all_slots(llama_url: str, dry_run: bool = False) -> Tuple[bool, int, List[Dict]]:
    """Erase all idle slots. Returns (success, num_erased, slot_details)."""
    slots = get_slot_info(llama_url)
    if slots is None:
        return False, 0, []

    erased = 0
    details = []
    for slot in slots:
        slot_id = slot.get('id', 0)
        is_processing = slot.get('is_processing', False)
        n_past = slot.get('n_past', 0)

        detail = {"id": slot_id, "n_past": n_past, "is_processing": is_processing}

        if is_processing:
            detail["action"] = "skipped_busy"
            details.append(detail)
            continue

        if dry_run:
            detail["action"] = "would_erase"
            details.append(detail)
            erased += 1
            continue

        try:
            req = urllib.request.Request(
                f"{llama_url}/slots/{slot_id}?action=erase",
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            detail["action"] = "erased"
            erased += 1
        except urllib.error.HTTPError as e:
            if e.code == 501:
                detail["action"] = "not_supported"
                detail["error"] = "slot-save-path not enabled"
            else:
                detail["action"] = "error"
                detail["error"] = f"HTTP {e.code}"
        except Exception as e:
            detail["action"] = "error"
            detail["error"] = str(e)

        details.append(detail)

    return erased > 0, erased, details


def restart_pod(cfg: CacheManagerConfig) -> bool:
    """Restart llama-server pod and wait for it to come back."""
    if cfg.dry_run:
        log("  [DRY RUN] Would restart pod")
        return True

    log("  Restarting llama-server pod...")
    try:
        subprocess.run(
            ["kubectl", "rollout", "restart",
             f"deployment/{cfg.deployment}", "-n", cfg.namespace],
            check=True, capture_output=True, timeout=30,
        )
    except Exception as e:
        log(f"  kubectl rollout restart failed: {e}", "ERROR")
        return False

    # Wait for server to come back
    time.sleep(15)
    start = time.time()
    while time.time() - start < cfg.warmup_timeout_sec:
        if server_is_healthy(cfg.llama_url):
            elapsed = int(time.time() - start)
            log(f"  Server back online after {elapsed}s")

            # Warmup request
            if cfg.warmup_enabled:
                try:
                    body = json.dumps({
                        "prompt": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
                        "n_predict": 5, "temperature": 0, "cache_prompt": False,
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        f"{cfg.llama_url}/completion", data=body,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        resp.read()
                    log(f"  Warmup complete")
                except Exception:
                    log(f"  Warmup failed (non-fatal)", "WARN")
            return True
        time.sleep(10)

    log(f"  Server did not recover within {cfg.warmup_timeout_sec}s", "ERROR")
    return False


# ===========================================================================
# Check & Manage (importable API)
# ===========================================================================

def check_and_manage(
    llama_url: str = "http://localhost:32735",
    soft_mb: int = 8192,
    hard_mb: int = 10240,
    namespace: str = "atlas",
    deployment: str = "llama-server",
    pod_label: str = "app=llama-server",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Single-shot check-and-act. Importable by benchmark runners.

    Returns dict with: action, memory_before_mb, memory_after_mb, success, details.
    """
    result = {
        "action": "none",
        "memory_before_mb": None,
        "memory_after_mb": None,
        "success": True,
        "details": {},
    }

    mem = get_server_memory_mb(namespace, pod_label)
    result["memory_before_mb"] = mem

    if mem is None:
        result["action"] = "error"
        result["success"] = False
        result["details"]["error"] = "could not read memory"
        return result

    if mem < soft_mb:
        return result  # All good

    # Tier 1: Slot erase
    if mem < hard_mb:
        log(f"  Memory {mem}MB > soft {soft_mb}MB — erasing slots")
        success, num_erased, slot_details = erase_all_slots(llama_url, dry_run)
        result["action"] = "erase"
        result["details"]["slots_erased"] = num_erased
        result["details"]["slot_details"] = slot_details
        result["success"] = success

        # Check if it helped
        time.sleep(2)
        result["memory_after_mb"] = get_server_memory_mb(namespace, pod_label)
        return result

    # Tier 2: Pod restart
    log(f"  Memory {mem}MB > hard {hard_mb}MB — restarting pod")
    cfg = CacheManagerConfig(
        llama_url=llama_url, namespace=namespace,
        deployment=deployment, dry_run=dry_run,
    )
    success = restart_pod(cfg)
    result["action"] = "restart"
    result["success"] = success
    time.sleep(5)
    result["memory_after_mb"] = get_server_memory_mb(namespace, pod_label)
    return result


# ===========================================================================
# Daemon Loop
# ===========================================================================

@dataclass
class DaemonState:
    last_erase_time: float = 0.0
    last_restart_time: float = 0.0
    consecutive_erase_failures: int = 0
    running: bool = True


def run_check_cycle(cfg: CacheManagerConfig, state: DaemonState):
    """Single monitoring cycle."""
    mem = get_server_memory_mb(cfg.namespace, cfg.pod_label)

    if mem is None:
        if not server_is_healthy(cfg.llama_url):
            log("Server unreachable — skipping cycle", "WARN")
        return

    now = time.time()

    if mem < cfg.soft_threshold_mb:
        return  # All good

    # Tier 1: Slot erase (if not in cooldown)
    if mem < cfg.hard_threshold_mb:
        if now - state.last_erase_time < cfg.erase_cooldown_sec:
            return  # In cooldown

        log(f"Memory {mem}MB > soft {cfg.soft_threshold_mb}MB — erasing slots")
        start = time.time()
        success, num_erased, details = erase_all_slots(cfg.llama_url, cfg.dry_run)
        duration_ms = (time.time() - start) * 1000

        mem_after = get_server_memory_mb(cfg.namespace, cfg.pod_label)
        freed = (mem - mem_after) if mem_after else 0

        log_event(cfg.telemetry_dir, {
            "action": "erase",
            "memory_before_mb": mem,
            "memory_after_mb": mem_after,
            "memory_freed_mb": freed,
            "soft_threshold_mb": cfg.soft_threshold_mb,
            "slots_erased": num_erased,
            "slot_details": details,
            "success": success,
            "duration_ms": round(duration_ms, 1),
            "dry_run": cfg.dry_run,
        })

        state.last_erase_time = now
        if success:
            state.consecutive_erase_failures = 0
            log(f"  Erased {num_erased} slots, freed ~{freed}MB "
                f"({mem}MB -> {mem_after}MB)")
        else:
            state.consecutive_erase_failures += 1
            log(f"  Slot erase failed (attempt {state.consecutive_erase_failures})", "WARN")
        return

    # Tier 2: Pod restart (if not in cooldown)
    if now - state.last_restart_time < cfg.restart_cooldown_sec:
        log(f"Memory {mem}MB > hard {cfg.hard_threshold_mb}MB but in restart cooldown", "WARN")
        return

    log(f"Memory {mem}MB > hard {cfg.hard_threshold_mb}MB — restarting pod")
    start = time.time()
    success = restart_pod(cfg)
    duration_ms = (time.time() - start) * 1000

    mem_after = get_server_memory_mb(cfg.namespace, cfg.pod_label)

    log_event(cfg.telemetry_dir, {
        "action": "restart",
        "memory_before_mb": mem,
        "memory_after_mb": mem_after,
        "hard_threshold_mb": cfg.hard_threshold_mb,
        "success": success,
        "duration_ms": round(duration_ms, 1),
        "dry_run": cfg.dry_run,
    })

    state.last_restart_time = now
    state.consecutive_erase_failures = 0

    if success:
        log(f"  Restart complete ({mem}MB -> {mem_after}MB)")
    else:
        log(f"  Restart failed", "ERROR")


def run_daemon(cfg: CacheManagerConfig):
    """Main monitoring loop."""
    state = DaemonState()

    def handle_signal(signum, frame):
        log(f"Received signal {signum} — shutting down")
        state.running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log("=" * 50)
    log("ATLAS llama-server Cache Manager")
    log("=" * 50)
    log(f"  Soft threshold: {cfg.soft_threshold_mb}MB (slot erase)")
    log(f"  Hard threshold: {cfg.hard_threshold_mb}MB (pod restart)")
    log(f"  Check interval: {cfg.check_interval_sec}s")
    log(f"  Erase cooldown: {cfg.erase_cooldown_sec}s")
    log(f"  Restart cooldown: {cfg.restart_cooldown_sec}s")
    log(f"  Server: {cfg.llama_url}")
    log(f"  Dry run: {cfg.dry_run}")
    log(f"  Telemetry: {cfg.telemetry_dir}")
    log("")

    while state.running:
        try:
            run_check_cycle(cfg, state)
        except Exception as e:
            log(f"Check cycle error: {e}", "ERROR")
        time.sleep(cfg.check_interval_sec)

    log("Cache manager stopped")


# ===========================================================================
# Status & One-shot
# ===========================================================================

def run_status(cfg: CacheManagerConfig):
    """Print current server memory and slot status."""
    print("=== ATLAS llama-server Cache Manager Status ===\n")

    # Memory
    mem = get_server_memory_mb(cfg.namespace, cfg.pod_label)
    if mem is not None:
        status = "OK"
        if mem >= cfg.hard_threshold_mb:
            status = "CRITICAL (above hard threshold)"
        elif mem >= cfg.soft_threshold_mb:
            status = "WARNING (above soft threshold)"
        print(f"Memory:  {mem}MB — {status}")
        print(f"  Soft threshold: {cfg.soft_threshold_mb}MB")
        print(f"  Hard threshold: {cfg.hard_threshold_mb}MB")
    else:
        print("Memory:  UNAVAILABLE (kubectl top failed)")

    # Health
    healthy = server_is_healthy(cfg.llama_url)
    print(f"\nHealth:  {'OK' if healthy else 'UNREACHABLE'} ({cfg.llama_url})")

    # Slots
    slots = get_slot_info(cfg.llama_url)
    if slots is not None:
        print(f"\nSlots:   {len(slots)}")
        for s in slots:
            processing = "BUSY" if s.get('is_processing') else "idle"
            n_past = s.get('n_past', 0)
            print(f"  Slot {s.get('id', '?')}: {processing}, "
                  f"n_past={n_past}")
    else:
        print("\nSlots:   UNAVAILABLE")

    # Slot erase support
    print()
    if slots is not None:
        _, _, details = erase_all_slots(cfg.llama_url, dry_run=True)
        supported = all(d.get("action") != "not_supported" for d in details)
        print(f"Slot erase API: {'ENABLED' if supported else 'DISABLED (needs --slot-save-path)'}")

    print()


def run_once(cfg: CacheManagerConfig):
    """Single check-and-act cycle."""
    log("Running single check cycle...")
    state = DaemonState()
    run_check_cycle(cfg, state)
    log("Done")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS llama-server Cache Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --daemon    Continuous monitoring loop (default)
  --once      Single check-and-act cycle
  --status    Print current status and exit

Examples:
  python3 scripts/llama-cache-manager.py --daemon &
  python3 scripts/llama-cache-manager.py --status
  python3 scripts/llama-cache-manager.py --once --dry-run
        """,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--daemon", action="store_true", default=True,
                      help="Run as continuous daemon (default)")
    mode.add_argument("--once", action="store_true",
                      help="Single check-and-act cycle")
    mode.add_argument("--status", action="store_true",
                      help="Print current status and exit")

    parser.add_argument("--llama-url", default=None,
                        help="llama-server URL")
    parser.add_argument("--soft-threshold", type=int, default=None,
                        help="Soft threshold MB (slot erase)")
    parser.add_argument("--hard-threshold", type=int, default=None,
                        help="Hard threshold MB (pod restart)")
    parser.add_argument("--interval", type=int, default=None,
                        help="Check interval seconds")
    parser.add_argument("--telemetry-dir", default=None,
                        help="Telemetry output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log actions without executing")

    args = parser.parse_args()
    cfg = load_config(args)

    if args.status:
        run_status(cfg)
    elif args.once:
        run_once(cfg)
    else:
        run_daemon(cfg)


if __name__ == "__main__":
    main()
