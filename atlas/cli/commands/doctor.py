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


def _supports_unicode() -> bool:
    """Detect whether stdout can safely encode the unicode chars we emit.

    Catches the LANG=C / ASCII-only stdout case (common via SSH from
    terminals with degraded locale, or when stdout is piped through a
    logger that defaulted to ASCII). Without this guard, doctor crashes
    with UnicodeEncodeError on the first em-dash.
    """
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if not enc:
        return False
    try:
        # Round-trip the chars we actually emit: em-dash + checkmark
        "—✓".encode(enc, errors="strict")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# Resolved at import; doctor.main() can re-evaluate if needed.
UNICODE_OK = _supports_unicode()
DASH       = "—" if UNICODE_OK else "--"

# Defaults — overridable by env (matches docker-compose.yml interpolations)
PROXY_URL    = os.environ.get("ATLAS_PROXY_URL",     "http://localhost:8090")
LLAMA_URL    = os.environ.get("ATLAS_INFERENCE_URL", "http://localhost:8080")
LENS_URL     = os.environ.get("ATLAS_LENS_URL",      "http://localhost:8099")
SANDBOX_URL  = os.environ.get("ATLAS_SANDBOX_URL",   "http://localhost:30820")
V3_URL       = os.environ.get("ATLAS_V3_URL",        "http://localhost:8070")
MODEL_DIR    = os.environ.get("ATLAS_MODELS_DIR",    "./models")
MODEL_FILE   = os.environ.get("ATLAS_MODEL_FILE",    "Qwen3.5-9B-Q6_K.gguf")
MODEL_NAME   = os.environ.get("ATLAS_MODEL_NAME",    "Qwen3.5-9B-Q6_K")
# Match docker-compose.yml's `${ATLAS_LENS_MODELS:-./geometric-lens/geometric_lens/models}`
# host-side bind-mount source so doctor checks the same directory the
# container will actually receive.
LENS_MODELS_DIR = os.environ.get("ATLAS_LENS_MODELS",
                                  "./geometric-lens/geometric_lens/models")

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
    # LENS_MODELS_DIR is typically the relative default; absolute paths
    # come from users overriding ATLAS_LENS_MODELS to mount weights from
    # outside the repo (e.g., a shared NFS mount). Resolve relative paths
    # against atlas_root, not the doctor's cwd.
    weights_dir = (LENS_MODELS_DIR if os.path.isabs(LENS_MODELS_DIR)
                   else os.path.normpath(os.path.join(atlas_root, LENS_MODELS_DIR)))
    required = ["cost_field.pt", "metric_tensor.pt"]
    missing = [f for f in required if not os.path.exists(
        os.path.join(weights_dir, f))]
    if missing:
        return CheckResult("lens_weights", "fail",
            f"missing: {', '.join(missing)}",
            f"expected in {weights_dir} — fetch from HuggingFace per README "
            f"(or set ATLAS_LENS_MODELS to point at your weights dir)")
    return CheckResult("lens_weights", "pass",
        f"cost_field.pt + metric_tensor.pt in {weights_dir}")


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


def check_tier_constraints() -> CheckResult:
    """PC-055.1 cross-check: does the host meet the recommended tier's
    per-axis minimums (RAM, CPU, disk)?

    Distinct from `tier_match`:
      - `tier_match` asks "is the configured model right for this hardware?"
      - `tier_constraints` asks "can this hardware actually run anything at
        the tier we'd recommend, given ATLAS's CPU/RAM/disk needs?"

    Catches the "16 GB GPU but 8 GB RAM" case where llama-server fits on
    the GPU but the host OOMs during V3 pipeline + sandbox compiles.
    """
    try:
        from atlas.cli.commands import tier
    except ImportError as e:
        return CheckResult("tier_constraints", "skip",
            "tier module unavailable", str(e))
    p = tier.probe()
    if not p.has_gpu:
        return CheckResult("tier_constraints", "skip",
            "no GPU detected (cpu tier)")
    recommended = tier.classify(p)
    checks = tier.evaluate_constraints(p, recommended)
    overall = tier.overall_status(checks)
    failed = [c for c in checks if c.status == "fail"]
    warned = [c for c in checks if c.status == "warn"]
    if overall == "fail":
        return CheckResult("tier_constraints", "warn",
            f"{len(failed)} hard constraint(s) below {recommended.tier}-tier minimum: "
            f"{', '.join(c.name for c in failed)}",
            "\n".join(c.message for c in failed) +
            "\n\nATLAS may OOM or fail to install at the recommended tier. "
            "Either upgrade host resources or downgrade tier "
            "(`atlas tier list` for alternatives).")
    if overall == "warn":
        return CheckResult("tier_constraints", "warn",
            f"{len(warned)} borderline constraint(s) for {recommended.tier} tier: "
            f"{', '.join(c.name for c in warned)}",
            "\n".join(c.message for c in warned) +
            "\n\nATLAS will run but may struggle under load.")
    return CheckResult("tier_constraints", "pass",
        f"{recommended.tier} tier fits comfortably "
        f"({p.cpu_cores} cores, {p.system_ram_gb:.0f} GB RAM, "
        f"{p.disk_free_gb:.0f} GB disk)")


def check_tier_match() -> CheckResult:
    """PC-055 cross-check: warn if .env settings overshoot the host's tier.

    Example: user on tier-small (8 GB GPU) running with the medium-tier
    default `Qwen3.5-9B-Q6_K.gguf` will OOM. Doctor flags this as a
    warning so the user knows to either downgrade the model or upgrade
    the GPU. We never hard-fail on tier mismatch — sometimes the user
    knows better than the heuristic (e.g., they pre-allocated VRAM
    elsewhere and want a smaller-than-recommended model).
    """
    try:
        from atlas.cli.commands import tier
    except ImportError as e:
        return CheckResult("tier_match", "skip",
            "tier module unavailable", str(e))
    p = tier.probe()
    if not p.has_gpu:
        return CheckResult("tier_match", "skip",
            "no GPU detected (cpu tier)")
    recommended = tier.classify(p)
    actual_model = MODEL_FILE
    if actual_model == recommended.model_file:
        return CheckResult("tier_match", "pass",
            f"{recommended.tier} tier matches configured model "
            f"({recommended.model_display})")
    # Mismatch — figure out direction. Find which tier the actual model
    # belongs to, then compare.
    actual_tier = None
    for t in tier.TIERS:
        if t.model_file == actual_model:
            actual_tier = t
            break
    if actual_tier is None:
        return CheckResult("tier_match", "warn",
            f"configured model `{actual_model}` is not in any tier preset",
            f"host classified as {recommended.tier}; consider one of the "
            f"presets: `atlas tier list`")
    # Warn only when actual > recommended (overshoot risks OOM).
    # Undershoot (smaller model than tier supports) is fine — just
    # leaves performance on the table.
    tiers_order = ["cpu", "small", "medium", "large", "xlarge"]
    rec_idx = tiers_order.index(recommended.tier)
    act_idx = tiers_order.index(actual_tier.tier)
    if act_idx > rec_idx:
        return CheckResult("tier_match", "warn",
            f"running {actual_tier.tier}-tier model on {recommended.tier}-tier "
            f"hardware ({p.vram_gb:.1f} GB VRAM)",
            f"OOM risk. Recommended for your VRAM: "
            f"{recommended.model_display}. Run `atlas tier` for detail.")
    return CheckResult("tier_match", "pass",
        f"running {actual_tier.tier}-tier model on {recommended.tier}-tier "
        f"hardware (under-utilized but safe)")


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

def _safe_print(s: str = "") -> None:
    """print() that survives an ASCII-only stdout.

    Without this, any em-dash, arrow, or unicode in a check message
    (most of them have one) crashes the entire run with
    UnicodeEncodeError — even though we only emit a small fixed set
    of unicode characters and could safely degrade them.
    """
    if UNICODE_OK:
        print(s)
        return
    # Replace the specific unicode chars we know we use, then encode/decode
    # as ASCII with replacement to catch anything else.
    s = (s.replace("—", "--")
          .replace("✓", "OK")
          .replace("✗", "X")
          .replace("⚠", "!")
          .replace("→", "->")
          .replace("│", "|")
          .replace("╭", "+").replace("╮", "+")
          .replace("╰", "+").replace("╯", "+")
          .replace("─", "-"))
    print(s.encode("ascii", errors="replace").decode("ascii"))


def _icon(status: str, color: bool) -> str:
    # Without color OR without unicode support, fall back to ASCII brackets.
    # This covers --no-color, non-TTY stdout, AND TTYs with ASCII-only encoding.
    if not color or not UNICODE_OK:
        return {"pass": "[OK]  ", "warn": "[WARN]",
                "fail": "[FAIL]", "skip": "[SKIP]"}[status]
    return {"pass": f"{GREEN}✓{RESET}", "warn": f"{YELL}⚠{RESET}",
            "fail": f"{RED}✗{RESET}", "skip": f"{DIM}-{RESET}"}[status]


def _print_result(r: CheckResult, verbose: bool, color: bool) -> None:
    name = f"{BOLD}{r.name}{RESET}" if color else r.name
    pad = " " * max(0, 32 - len(r.name))
    _safe_print(f"  {_icon(r.status, color)} {name}{pad}  {r.message}")
    if verbose and r.detail:
        for line in r.detail.splitlines():
            _safe_print(f"      {DIM if color else ''}{line}{RESET if color else ''}")


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
        # ensure_ascii=False keeps unicode in detail fields readable; if
        # stdout truly can't encode it, write bytes directly with
        # backslash-escape so we don't crash on the way out.
        body = json.dumps(out, indent=2, ensure_ascii=not UNICODE_OK)
        try:
            print(body)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(body.encode("ascii", errors="backslashreplace"))
            sys.stdout.buffer.write(b"\n")
        return 1 if n_fail else 0

    for r in results:
        _print_result(r, args.verbose, color)
    _safe_print()
    parts = [f"{n_pass} passed"]
    if n_warn:
        parts.append(f"{YELL if color else ''}{n_warn} warnings{RESET if color else ''}")
    if n_fail:
        parts.append(f"{RED if color else ''}{n_fail} failed{RESET if color else ''}")
    if n_skip:
        parts.append(f"{n_skip} skipped")
    _safe_print("  " + ", ".join(parts))
    if n_fail == 0 and n_warn == 0:
        _safe_print(f"  {GREEN if color else ''}ATLAS install is healthy.{RESET if color else ''}")
    elif n_fail == 0:
        _safe_print(f"  {YELL if color else ''}ATLAS install is functional with warnings.{RESET if color else ''}")
    else:
        _safe_print(f"  {RED if color else ''}ATLAS install has failures {DASH} re-run with -v for detail.{RESET if color else ''}")
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
        _safe_print(f"{hdr} {DASH} checking install health (root: {atlas_root})")
        _safe_print()

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

    # 10.5. Tier match (PC-055) — soft cross-check that .env model
    # matches host hardware. Warn on overshoot (OOM risk), pass on
    # match or undershoot.
    results.append(check_tier_match())

    # 10.6. Tier constraints (PC-055.1) — does the host meet the
    # recommended tier's CPU/RAM/disk minimums? Catches "16 GB GPU
    # but 8 GB RAM" cases where llama fits but host OOMs under V3.
    results.append(check_tier_constraints())

    # 11. End-to-end smoke
    if args.quick:
        results.append(CheckResult("e2e_smoke", "skip",
            "skipped (--quick)"))
    else:
        results.append(check_e2e_smoke())

    return _emit(results, args, color)


if __name__ == "__main__":
    sys.exit(main())
