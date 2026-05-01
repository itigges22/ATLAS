"""atlas doctor — comprehensive install diagnostic (PC-053).

Verifies an ATLAS install is healthy end-to-end. Runs 11 checks across
the host environment, the docker stack, and a live request through the
proxy. Designed to be the answer to "is it really working?" — both for
humans (pretty terminal output) and for scripts (--json).

Invoke:
    atlas doctor                 # full check
    atlas doctor --quick         # skip e2e smoke test
    atlas doctor --json          # machine output (for bootstrap, CI)
    atlas doctor -v              # show detail for each check
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

# ANSI color codes
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RED   = "\033[31m"
GREEN = "\033[32m"
YELL  = "\033[33m"
CYAN  = "\033[36m"

# Defaults — overridable by env (matches docker-compose.yml interpolations)
PROXY_URL    = os.environ.get("ATLAS_PROXY_URL",     "http://localhost:8090")
LLAMA_URL    = os.environ.get("ATLAS_INFERENCE_URL", "http://localhost:8080")
LENS_URL     = os.environ.get("ATLAS_LENS_URL",      "http://localhost:8099")
SANDBOX_URL  = os.environ.get("ATLAS_SANDBOX_URL",   "http://localhost:30820")
V3_URL       = os.environ.get("ATLAS_V3_URL",        "http://localhost:8070")
MODEL_DIR    = os.environ.get("ATLAS_MODELS_DIR",    "./models")
MODEL_FILE   = os.environ.get("ATLAS_MODEL_FILE",    "Qwen3.5-9B-Q6_K.gguf")
MODEL_NAME   = os.environ.get("ATLAS_MODEL_NAME",    "Qwen3.5-9B-Q6_K")

EXPECTED_SERVICES = [
    "redis", "llama-server", "geometric-lens",
    "v3-service", "sandbox", "atlas-proxy",
]


@dataclass
class CheckResult:
    name: str
    status: str  # pass | warn | fail | skip
    message: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Subprocess + HTTP helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], timeout: int = 30,
         cwd: Optional[str] = None) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd)
        return p.returncode, p.stdout, p.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return 1, "", str(e)


def _http_get(url: str, timeout: int = 5) -> Tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return True, resp.read().decode()
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_docker() -> CheckResult:
    rc, out, err = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if rc != 0:
        return CheckResult("docker",  "fail",
            "daemon not reachable",
            (err or out).strip()[:200])
    return CheckResult("docker", "pass", f"daemon reachable (v{out.strip()})")


def check_compose() -> CheckResult:
    rc, out, err = _run(["docker", "compose", "version", "--short"])
    if rc != 0:
        return CheckResult("compose", "fail",
            "docker compose v2 not installed",
            (err or out).strip()[:200])
    return CheckResult("compose", "pass", f"v{out.strip()}")


def check_nvidia() -> CheckResult:
    """Verify nvidia-container-toolkit by running nvidia-smi inside Docker."""
    # Use the smallest CUDA base image available to keep the check fast.
    rc, out, err = _run([
        "docker", "run", "--rm", "--gpus", "all",
        "nvidia/cuda:12.0.0-base-ubuntu22.04",
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
    ], timeout=120)
    if rc != 0:
        # Distinguish "no GPU" from "toolkit broken"
        joined = (err + out).lower()
        if "could not select device driver" in joined or "nvidia-container" in joined:
            return CheckResult("nvidia", "fail",
                "nvidia-container-toolkit not configured",
                (err or out).strip()[:300])
        if "no nvidia gpu" in joined or "no devices" in joined:
            return CheckResult("nvidia", "warn",
                "no NVIDIA GPU visible to Docker (CPU-only mode)",
                (err or out).strip()[:300])
        return CheckResult("nvidia", "fail",
            "nvidia-smi failed inside Docker",
            (err or out).strip()[:300])
    gpus = [g.strip() for g in out.strip().split("\n") if g.strip()]
    return CheckResult("nvidia", "pass",
        f"{len(gpus)} GPU(s): {', '.join(gpus)}")


def _compose_ps(project_dir: str) -> List[Dict]:
    """Run `docker compose ps --format json` and parse (handles both NDJSON and array forms).

    Must run from `project_dir` — that's where docker-compose.yml lives.
    Without this, `atlas doctor` invoked from outside the repo sees
    "no containers" even when the stack is fully healthy.
    """
    rc, out, err = _run(
        ["docker", "compose", "ps", "--all", "--format", "json"],
        cwd=project_dir,
    )
    if rc != 0 or not out.strip():
        return []
    services: List[Dict] = []
    # Newer compose: NDJSON (one object per line)
    for line in out.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, list):
                services.extend(obj)
            else:
                services.append(obj)
        except json.JSONDecodeError:
            continue
    return services


def check_containers(services: List[Dict]) -> List[CheckResult]:
    if not services:
        return [CheckResult("containers", "fail",
            "no containers found — run `docker compose up -d` first",
            "compose ps returned empty")]

    found = {s.get("Service", s.get("Name", "")): s for s in services}
    results: List[CheckResult] = []
    for name in EXPECTED_SERVICES:
        svc = found.get(name)
        if svc is None:
            results.append(CheckResult(f"container/{name}", "fail",
                "not running",
                "service not in `docker compose ps` output"))
            continue
        state = svc.get("State", "?")
        health = svc.get("Health", "")
        status_str = svc.get("Status", "")
        if state == "running" and health in ("healthy", ""):
            results.append(CheckResult(f"container/{name}", "pass", state))
        elif state == "running" and health == "starting":
            results.append(CheckResult(f"container/{name}", "warn",
                f"{state}/starting", "still warming up — re-run doctor in 30s"))
        else:
            results.append(CheckResult(f"container/{name}", "fail",
                f"{state}/{health or '-'}", status_str))
    return results


def check_health_endpoints() -> List[CheckResult]:
    endpoints = [
        ("llama",   f"{LLAMA_URL}/health"),
        ("lens",    f"{LENS_URL}/health"),
        ("v3",      f"{V3_URL}/health"),
        ("sandbox", f"{SANDBOX_URL}/health"),
        ("proxy",   f"{PROXY_URL}/health"),
    ]
    results = []
    for name, url in endpoints:
        ok, body = _http_get(url)
        if not ok:
            results.append(CheckResult(f"health/{name}", "fail",
                "endpoint unreachable", body[:200]))
            continue
        try:
            data = json.loads(body)
            status = data.get("status", "ok")
        except json.JSONDecodeError:
            status = "ok (non-json)"
        results.append(CheckResult(f"health/{name}", "pass", status, body[:200]))
    return results


def check_model_file(atlas_root: str) -> CheckResult:
    # MODEL_DIR is typically `./models` (relative to the compose cwd).
    # Resolve relative paths against atlas_root, not the doctor's cwd.
    base = MODEL_DIR if os.path.isabs(MODEL_DIR) else os.path.join(atlas_root, MODEL_DIR)
    path = os.path.normpath(os.path.join(base, MODEL_FILE))
    if not os.path.exists(path):
        return CheckResult("model_file", "fail",
            f"missing: {path}",
            "run scripts/download-models.sh")
    size = os.path.getsize(path)
    if size < 100 * 1024 * 1024:  # < 100 MB
        return CheckResult("model_file", "warn",
            f"{path} exists but only {size} bytes — likely truncated",
            "expected > 1 GB for a typical GGUF; re-run download-models.sh")
    gb = size / (1024 * 1024 * 1024)
    return CheckResult("model_file", "pass", f"{MODEL_FILE} ({gb:.1f} GB)")


def check_lens_weights(atlas_root: str) -> CheckResult:
    weights_dir = os.path.join(atlas_root, "geometric-lens",
                               "geometric_lens", "models")
    required = ["cost_field.pt", "metric_tensor.pt"]
    missing = [f for f in required if not os.path.exists(
        os.path.join(weights_dir, f))]
    if missing:
        return CheckResult("lens_weights", "fail",
            f"missing: {', '.join(missing)}",
            f"expected in {weights_dir} — fetch from HuggingFace per README")
    return CheckResult("lens_weights", "pass",
        "cost_field.pt + metric_tensor.pt present")


def check_overcommit() -> CheckResult:
    """PC-011: Redis warns and AOF rewrite can fail without overcommit_memory=1."""
    try:
        with open("/proc/sys/vm/overcommit_memory") as f:
            val = f.read().strip()
        if val == "1":
            return CheckResult("vm.overcommit_memory", "pass", "= 1")
        return CheckResult("vm.overcommit_memory", "warn",
            f"= {val} (Redis prefers 1 — see PC-011)",
            "Fix: sudo sysctl vm.overcommit_memory=1 && "
            "echo 'vm.overcommit_memory=1' | sudo tee /etc/sysctl.d/99-atlas.conf")
    except Exception as e:
        return CheckResult("vm.overcommit_memory", "skip",
            "could not read /proc/sys (non-Linux?)", str(e))


def check_image_skew(services: List[Dict]) -> CheckResult:
    """PC-052 follow-up: warn if the 5 atlas-* images aren't on the same tag."""
    atlas_imgs = [s.get("Image", "") for s in services
                  if "atlas-" in s.get("Image", "")]
    if not atlas_imgs:
        return CheckResult("image_skew", "skip",
            "no atlas-* images found in compose ps")
    tags = set()
    for img in atlas_imgs:
        if ":" in img:
            tags.add(img.rsplit(":", 1)[1])
        else:
            tags.add("<no-tag>")
    if len(tags) > 1:
        return CheckResult("image_skew", "warn",
            f"mixed tags across atlas-* services: {', '.join(sorted(tags))}",
            "Pin ATLAS_IMAGE_TAG in .env to align all 5 services. "
            "Mixing major versions can break inter-service contracts.")
    return CheckResult("image_skew", "pass",
        f"all atlas-* images on tag :{next(iter(tags))}")


def check_e2e_smoke() -> CheckResult:
    """End-to-end POST to llama-server — verifies the model loads and generates.

    Targets llama-server directly (not the proxy) because the proxy's
    agent loop with ATLAS_AGENT_LOOP=1 intercepts /v1/chat/completions and
    runs the tier classifier + V3 pipeline. For a one-word smoke test the
    agent loop adds ~30s and frequently consumes all max_tokens in
    /nothink routing/planning before producing visible content. Hitting
    llama directly answers the question we care about: "is the GGUF
    actually loaded and inferring?" The proxy's reachability is already
    covered by `health/proxy`.
    """
    body = {
        "messages": [{"role": "user", "content": "Reply with the single word: ATLAS"}],
        # Qwen3.5 with thinking enabled emits 100-200 tokens of
        # reasoning_content (not surfaced as content) before the visible
        # answer. Anything < ~250 risks finish=length with empty content.
        "max_tokens": 300,
        "temperature": 0,
        "stream": False,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as e:
        return CheckResult("e2e_smoke", "fail",
            f"llama-server POST failed: {type(e).__name__}", str(e)[:300])
    choices = payload.get("choices", [])
    if not choices:
        return CheckResult("e2e_smoke", "fail",
            "llama-server returned no choices",
            json.dumps(payload)[:300])
    msg = choices[0].get("message", {})
    content = (msg.get("content", "") or "").strip()
    finish = choices[0].get("finish_reason", "")
    if not content:
        return CheckResult("e2e_smoke", "fail",
            f"llama-server returned an empty completion (finish={finish})",
            json.dumps(payload)[:400])
    return CheckResult("e2e_smoke", "pass",
        f"model produced {len(content)} chars (finish={finish})",
        content[:300])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _icon(status: str, color: bool) -> str:
    if not color:
        return {"pass": "[OK]  ", "warn": "[WARN]",
                "fail": "[FAIL]", "skip": "[SKIP]"}[status]
    return {"pass": f"{GREEN}✓{RESET}", "warn": f"{YELL}⚠{RESET}",
            "fail": f"{RED}✗{RESET}", "skip": f"{DIM}-{RESET}"}[status]


def _print_result(r: CheckResult, verbose: bool, color: bool) -> None:
    name = f"{BOLD}{r.name}{RESET}" if color else r.name
    pad = " " * max(0, 32 - len(r.name))
    print(f"  {_icon(r.status, color)} {name}{pad}  {r.message}")
    if verbose and r.detail:
        for line in r.detail.splitlines():
            print(f"      {DIM if color else ''}{line}{RESET if color else ''}")


def _emit(results: List[CheckResult], args: argparse.Namespace, color: bool) -> int:
    n_pass = sum(1 for r in results if r.status == "pass")
    n_warn = sum(1 for r in results if r.status == "warn")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_skip = sum(1 for r in results if r.status == "skip")

    if args.json:
        out = {
            "summary": {"pass": n_pass, "warn": n_warn,
                        "fail": n_fail, "skip": n_skip},
            "checks": [asdict(r) for r in results],
        }
        print(json.dumps(out, indent=2))
        return 1 if n_fail else 0

    for r in results:
        _print_result(r, args.verbose, color)
    print()
    parts = [f"{n_pass} passed"]
    if n_warn:
        parts.append(f"{YELL if color else ''}{n_warn} warnings{RESET if color else ''}")
    if n_fail:
        parts.append(f"{RED if color else ''}{n_fail} failed{RESET if color else ''}")
    if n_skip:
        parts.append(f"{n_skip} skipped")
    print("  " + ", ".join(parts))
    if n_fail == 0 and n_warn == 0:
        print(f"  {GREEN if color else ''}ATLAS install is healthy.{RESET if color else ''}")
    elif n_fail == 0:
        print(f"  {YELL if color else ''}ATLAS install is functional with warnings.{RESET if color else ''}")
    else:
        print(f"  {RED if color else ''}ATLAS install has failures — re-run with -v for detail.{RESET if color else ''}")
    return 1 if n_fail else 0


def _find_atlas_root() -> str:
    """Locate the ATLAS repo root (where docker-compose.yml lives)."""
    here = os.path.dirname(os.path.abspath(__file__))
    # atlas/cli/commands -> atlas/cli -> atlas -> ATLAS
    for _ in range(5):
        if os.path.exists(os.path.join(here, "docker-compose.yml")):
            return here
        here = os.path.dirname(here)
    # Fallback: cwd
    return os.getcwd()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas doctor",
        description="Diagnose ATLAS install health (PC-053)")
    parser.add_argument("--quick", action="store_true",
        help="skip the e2e smoke test (saves ~10s)")
    parser.add_argument("--json", action="store_true",
        help="emit JSON output (for bootstrap, CI, scripts)")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="show detail for each check")
    parser.add_argument("--no-color", action="store_true",
        help="disable ANSI color in human output")
    args = parser.parse_args(argv)

    color = sys.stdout.isatty() and not args.no_color and not args.json
    atlas_root = _find_atlas_root()

    if not args.json:
        hdr = f"{BOLD}ATLAS doctor{RESET}" if color else "ATLAS doctor"
        print(f"{hdr} — checking install health (root: {atlas_root})")
        print()

    results: List[CheckResult] = []

    # 1. Docker
    docker = check_docker()
    results.append(docker)
    if docker.status == "fail":
        # Without docker, every subsequent check is meaningless.
        results.append(CheckResult("compose", "skip",
            "skipped (docker unreachable)"))
        return _emit(results, args, color)

    # 2. Docker compose v2
    results.append(check_compose())

    # 3. NVIDIA toolkit (also slow — 60s timeout — pulls a small CUDA image first time)
    results.append(check_nvidia())

    # 4. Compose stack — pass atlas_root as cwd so compose finds
    # docker-compose.yml even when doctor is invoked from elsewhere
    # on the filesystem.
    services = _compose_ps(atlas_root)

    # 5. Per-container state
    container_results = check_containers(services)
    results.extend(container_results)

    # 6. Endpoint health (only if at least one container is running)
    if any(r.status == "pass" for r in container_results):
        results.extend(check_health_endpoints())

    # 7. Model file (host-side)
    results.append(check_model_file(atlas_root))

    # 8. Lens weights (host-side)
    results.append(check_lens_weights(atlas_root))

    # 9. vm.overcommit_memory (PC-011)
    results.append(check_overcommit())

    # 10. Image-tag skew (PC-052)
    results.append(check_image_skew(services))

    # 11. End-to-end smoke
    if args.quick:
        results.append(CheckResult("e2e_smoke", "skip",
            "skipped (--quick)"))
    else:
        results.append(check_e2e_smoke())

    return _emit(results, args, color)


if __name__ == "__main__":
    sys.exit(main())
