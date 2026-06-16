"""atlas doctor — comprehensive install diagnostic (PC-053).

Verifies an ATLAS install is healthy end-to-end. Runs ~22 checks across
the host environment, the docker stack, and a live request through the
proxy: 11 individual checks (docker, compose, nvidia, model_file,
lens_weights, overcommit, image_skew, tier_match (PC-055),
tier_constraints (PC-055.1), asa_steering (BiasBusters #4),
e2e_smoke), six per-container state checks (one per service in
`EXPECTED_SERVICES`), and five per-endpoint health checks. Designed to
be the answer to "is it really working?" — both for humans (pretty
terminal output) and for scripts (--json).

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

from atlas.cli.commands import tier

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

def _read_dotenv() -> Dict[str, str]:
    """Parse the compose .env (the Docker deployment's source of truth) by
    walking up from this file. Lets the model/dir checks reflect what's actually
    configured instead of falling back to the bundled Qwen defaults when the
    shell env doesn't export ATLAS_MODEL_FILE."""
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        envp = os.path.join(cur, ".env")
        if os.path.exists(envp):
            out: Dict[str, str] = {}
            try:
                with open(envp, encoding="utf-8-sig") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("export "):
                            line = line[len("export "):].lstrip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            # Drop a whitespace-preceded inline comment.
                            v = v.lstrip()
                            head, hash_sep, _ = v.partition("#")
                            if hash_sep and head and head[-1] in " \t":
                                v = head
                            out[k.strip()] = v.strip().strip('"').strip("'")
            except OSError:
                pass
            return out
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return {}


_ENV = _read_dotenv()

# Defaults — shell env first, then the compose .env, then a bundled fallback.
PROXY_URL    = os.environ.get("ATLAS_PROXY_URL",     "http://localhost:8090")
LLAMA_URL    = os.environ.get("ATLAS_INFERENCE_URL", "http://localhost:8080")
LENS_URL     = os.environ.get("ATLAS_LENS_URL",      "http://localhost:8099")
SANDBOX_URL  = os.environ.get("ATLAS_SANDBOX_URL",   "http://localhost:30820")
V3_URL       = os.environ.get("ATLAS_V3_URL",        "http://localhost:8070")
MODEL_DIR    = os.environ.get("ATLAS_MODELS_DIR")  or _ENV.get("ATLAS_MODELS_DIR", "./models")
MODEL_FILE   = os.environ.get("ATLAS_MODEL_FILE")  or _ENV.get("ATLAS_MODEL_FILE", "Qwen3.5-9B-Q6_K.gguf")
MODEL_NAME   = os.environ.get("ATLAS_MODEL_NAME")  or _ENV.get("ATLAS_MODEL_NAME", "Qwen3.5-9B-Q6_K")
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


def _port_listening(host: str, port: int, timeout: float = 2.0) -> bool:
    """True if a TCP server is accepting connections at host:port.
    Portable replacement for `nc -z` which has different flag conventions
    across GNU netcat / BSD nc / nmap-ncat / busybox nc — and may not be
    on PATH at all (notably alpine/socat doesn't ship it).
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _resolve_backend(atlas_root: Optional[str] = None) -> Optional[str]:
    """Resolve which ATLAS_BACKEND the user has configured. Reads the
    shell env first (canonical), then falls back to .env in atlas_root.

    The shell-env-only check in main() missed the macOS hybrid case
    (#32): atlas init writes ATLAS_BACKEND=metal into .env but the user
    rarely sources .env before running atlas doctor, so the env-var
    check returns None and check_metal_native is skipped — leaving Mac
    users wondering why the metal-native diagnostic never appears.

    Returns the backend id string ('cuda' | 'rocm' | 'vulkan' | 'metal')
    or None if no backend is configured anywhere.
    """
    val = os.environ.get("ATLAS_BACKEND")
    if val:
        return val.strip().lower()
    if not atlas_root:
        return None
    env_path = os.path.join(atlas_root, ".env")
    if not os.path.isfile(env_path):
        return None
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("ATLAS_BACKEND="):
                    return line.split("=", 1)[1].strip().strip("'\"").lower()
    except OSError:
        return None
    return None


def check_arch() -> CheckResult:
    """Surface the host CPU architecture so users see why a given backend
    is or isn't available. Pass-status on x86_64 (default ATLAS target);
    warn on aarch64 Linux with a hint about the arm64 backend matrix;
    pass on Apple Silicon (the macOS hybrid Metal path is shipped, #32).

    See #115 for the multi-arch Docker build status.
    """
    arch = tier.arch_detect()
    if arch == "x86_64":
        return CheckResult("arch", "pass", "x86_64")
    if arch == "aarch64":
        # Apple Silicon gets a different message from arm64 Linux. On
        # macOS the path is native Metal (#32 hybrid), not vulkan-or-
        # cuda-sbsa-l4t. The Linux arm64 matrix doesn't apply here.
        if sys.platform == "darwin":
            return CheckResult("arch", "pass",
                "aarch64 (Apple Silicon) — Metal hybrid path supported (#32)")
        return CheckResult("arch", "warn",
            "aarch64 — vulkan + cuda (sbsa/l4t) only, no rocm",
            "AMD ROCm has no arm64 release. Use ATLAS_BACKEND=vulkan for "
            "AMD GPUs on arm64. NVIDIA CUDA needs the sbsa (DGX Spark) or "
            "l4t (Jetson) base image swap, see docs/SETUP.md#arm64.")
    return CheckResult("arch", "warn",
        f"unsupported arch '{arch}'",
        "ATLAS officially supports x86_64 and aarch64. Other arches may "
        "work via vulkan + lavapipe but are untested. See #115.")


def check_gpu() -> CheckResult:
    """Dispatcher: pick the right vendor-specific GPU check or warn if no
    GPU is detected. V3.1.1 — replaces the old check_nvidia() entry
    point. NVIDIA + AMD supported; Metal/SYCL not yet packaged.

    #115 addition: AMD GPU on aarch64 -> warn rather than dispatch to the
    rocm container check, since rocm has no arm64 release. The check would
    just fail with a confusing image-pull error otherwise.
    """
    gpus = tier.detect_gpu()
    if not gpus:
        return CheckResult("gpu", "warn",
            "no GPU detected (CPU-only mode — inference will be very slow)",
            "")
    primary = tier.primary_gpu(gpus)
    if primary is None:
        return CheckResult("gpu", "warn",
            "GPUs detected but none selectable",
            "tier.primary_gpu returned None")
    arch = tier.arch_detect()
    if primary.vendor == "nvidia":
        return _check_nvidia_via_docker()
    if primary.vendor == "amd":
        if arch != "x86_64":
            return CheckResult("gpu", "warn",
                f"AMD GPU on {arch}: ROCm has no {arch} release, use "
                f"ATLAS_BACKEND=vulkan instead (#115)",
                f"primary GPU: {primary.name}")
        return _check_amd_via_docker()
    # #32: Apple Silicon ships the Metal hybrid path. Defer the deeper
    # validation to check_metal_native (which fires when ATLAS_BACKEND
    # is metal) — at the gpu-dispatcher level just acknowledge that
    # Apple GPUs ARE supported now, the install just happens to live
    # outside Docker. Don't emit the old "not yet supported" warning.
    if primary.vendor == "apple":
        return CheckResult("gpu", "pass",
            f"[apple] {primary.name} ({primary.vram_gb:.1f} GB unified) "
            f"— Metal hybrid path supported (#32). "
            f"See metal-native check for native llama-server status.")
    return CheckResult("gpu", "warn",
        f"vendor '{primary.vendor}' detected but Docker integration not yet supported "
        f"(SYCL -> roadmap)",
        f"primary GPU: {primary.name}")


# Backwards-compat alias for any external callers that imported the old name.
def check_nvidia() -> CheckResult:
    return check_gpu()


def _check_nvidia_via_docker() -> CheckResult:
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
            return CheckResult("gpu", "fail",
                "nvidia-container-toolkit not configured",
                (err or out).strip()[:300])
        if "no nvidia gpu" in joined or "no devices" in joined:
            return CheckResult("gpu", "warn",
                "no NVIDIA GPU visible to Docker (CPU-only mode)",
                (err or out).strip()[:300])
        return CheckResult("gpu", "fail",
            "nvidia-smi failed inside Docker",
            (err or out).strip()[:300])
    gpus = [g.strip() for g in out.strip().split("\n") if g.strip()]
    return CheckResult("gpu", "pass",
        f"[nvidia] {len(gpus)} GPU(s): {', '.join(gpus)}")


def _check_amd_via_docker() -> CheckResult:
    """Verify ROCm Docker passthrough by running rocm-smi inside a ROCm
    container. Unlike NVIDIA, ROCm doesn't need a separate container
    runtime — just /dev/kfd + /dev/dri device passthrough with the
    video + render groups. This check validates that whole chain.
    """
    rc, out, err = _run([
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video", "--group-add", "render",
        "rocm/rocm-terminal:latest",
        "rocm-smi", "--showproductname",
    ], timeout=180)  # +60s headroom for the first-time image pull (~2 GB)
    if rc != 0:
        joined = (err + out).lower()
        if "permission denied" in joined or "no such device" in joined:
            return CheckResult("gpu", "fail",
                "AMD GPU detected but Docker can't reach /dev/kfd — "
                "check amdgpu kernel driver + render/video group membership",
                (err or out).strip()[:300])
        if "no gpus found" in joined or "no rocm devices" in joined:
            return CheckResult("gpu", "warn",
                "no AMD GPU visible to Docker (CPU-only mode)",
                (err or out).strip()[:300])
        return CheckResult("gpu", "fail",
            "rocm-smi failed inside Docker",
            (err or out).strip()[:300])
    # rocm-smi product output is wide; count lines starting with "GPU[" or
    # "card" as a GPU entry (output format varies across ROCm versions).
    gpus = [ln.strip() for ln in out.strip().splitlines()
            if ln.strip() and not ln.startswith(("=", "-"))]
    summary = "; ".join(g[:80] for g in gpus[:3]) if gpus else "rocm-smi succeeded"
    return CheckResult("gpu", "pass", f"[amd] {summary}")


def _check_vulkan_via_docker() -> CheckResult:
    """Verify Vulkan device passthrough by running vulkaninfo inside a
    minimal Mesa-Vulkan container (PC-114, #114).

    Unlike CUDA / ROCm this doesn't require a vendor-specific runtime
    or kernel driver — just /dev/dri passthrough so the Mesa ICDs
    (RADV/ANV/lavapipe) inside the image can find a device. NVIDIA
    users on Vulkan still need the toolkit (same as CUDA path) but
    that's caught by `check_gpu` separately.

    We use the same ubuntu+mesa stack the production Dockerfile.vulkan
    builds on so this validates the exact compat surface the runtime
    image will see. The throwaway container is ~150 MB after first
    pull — bigger than the ROCm check's terminal image but still
    bounded.
    """
    # /dev/dri may not exist on hosts with no GPU at all (or macOS Docker
    # Desktop). Short-circuit before touching docker — produces a clean
    # "no GPU passthrough" message instead of a confusing docker error.
    if not os.path.exists("/dev/dri"):
        return CheckResult("vulkan", "warn",
            "no /dev/dri on host — Vulkan container would only see "
            "the CPU lavapipe ICD (very slow)",
            "On Linux: install kernel modules for your GPU + ensure "
            "the render-node devices exist. On macOS: Vulkan-in-Docker "
            "uses MoltenVK via qemu; native install (#32) is the fast path.")
    rc, out, err = _run([
        "docker", "run", "--rm",
        "--device=/dev/dri",
        "--group-add", "video", "--group-add", "render",
        "ubuntu:22.04",
        "bash", "-c",
        # apt-install Mesa Vulkan stack + run vulkaninfo summary. Cap
        # output so a verbose ICD enum doesn't blow our 300-char detail
        # budget.
        ("apt-get update -qq >/dev/null && "
         + "apt-get install -y -qq libvulkan1 mesa-vulkan-drivers vulkan-tools "
         + ">/dev/null 2>&1 && "
         + "vulkaninfo --summary 2>&1 | head -40"),
    ], timeout=300)  # apt + image pull on cold cache
    if rc != 0:
        joined = (err + out).lower()
        if "permission denied" in joined or "no such device" in joined:
            return CheckResult("vulkan", "fail",
                "Vulkan device passthrough failed — check render/video "
                "group membership on the host",
                (err or out).strip()[:300])
        if "could not find any vulkan" in joined:
            return CheckResult("vulkan", "warn",
                "Vulkan loader found no ICDs (no GPU drivers visible to "
                "the container; lavapipe CPU fallback would still work)",
                (err or out).strip()[:300])
        return CheckResult("vulkan", "fail",
            "vulkaninfo failed inside the test container",
            (err or out).strip()[:300])
    # Pull a one-line summary out of vulkaninfo's `deviceName = ...` rows.
    devices = [ln.split("=")[-1].strip() for ln in out.splitlines()
               if "deviceName" in ln]
    if not devices:
        return CheckResult("vulkan", "warn",
            "vulkaninfo ran but no deviceName lines — Vulkan stack is "
            "responsive but couldn't enumerate physical devices",
            out.strip()[:200])
    summary = "; ".join(d[:60] for d in devices[:3])
    return CheckResult("vulkan", "pass", f"[vulkan] {len(devices)} ICD(s): {summary}")


def _check_metal_native() -> CheckResult:
    """Verify the macOS hybrid path (#32) is wired correctly: the native
    llama-server binary exists where the setup script puts it, and the
    docker stack is configured to forward to it via host.docker.internal.

    Three failure modes we surface here:
      1. Setup script never ran   -> binary missing at $HOME/.atlas/macos/bin/
      2. Setup ran but binary won't execute (corrupt download, wrong arch)
      3. The macos compose overlay isn't applied (so docker is trying to
         pull/build a normal llama-server image that won't work on Mac)

    Only fires when ATLAS_BACKEND=metal. On Linux + Windows this would
    just be noise. NOT a Docker check (the binary lives on the host),
    so it's cheap (~1ms) and runs unconditionally for Mac users.
    """
    if sys.platform != "darwin":
        return CheckResult("metal-native", "skip",
            "not on macOS — metal hybrid path doesn't apply", "")

    # Expected setup-script output location. Keep aligned with
    # scripts/atlas-setup-macos.sh DEFAULT_PREFIX.
    prefix = os.path.expanduser("~/.atlas/macos")
    binary = os.path.join(prefix, "bin", "llama-server-metal")

    if not os.path.isfile(binary):
        return CheckResult("metal-native", "fail",
            "native llama-server not found — run scripts/atlas-setup-macos.sh",
            f"expected at {binary}. See docs/SETUP_MACOS.md.")

    if not os.access(binary, os.X_OK):
        return CheckResult("metal-native", "fail",
            "native llama-server is not executable — re-run "
            "scripts/atlas-setup-macos.sh --rebuild",
            f"{binary} exists but lacks +x. Likely a botched copy or "
            f"transferred over a filesystem that strips perms (smb/nfs).")

    # Sanity-check the binary at least loads. Exit code alone isn't
    # reliable: llama-server treats `--help` as a parse failure (prints
    # usage, exits 1) by convention. Instead look for usage markers in
    # the combined output — anything matching means the binary's main
    # ran and printed the help text. No usage markers + nonzero exit =
    # the binary never reached main (dyld failure, missing dylib,
    # corrupt build from an interrupted cmake).
    rc, out, err = _run([binary, "--help"], timeout=5)
    combined = (out + err).lower()
    usage_markers = ("usage", "options", "--ctx-size", "llama-server")
    looks_like_usage = any(m in combined for m in usage_markers)
    if rc != 0 and not looks_like_usage:
        return CheckResult("metal-native", "fail",
            "native llama-server exists but won't run "
            f"(exit {rc}, no usage output) — try --rebuild",
            (err or out).strip()[:300] or "binary produced no output")

    # Confirm the host port is listening. If the user ran setup but
    # hasn't started atlas-llama-macos.sh yet, surface that as a warn
    # (not fail) — they may just not be ready yet. Use a small Python
    # socket probe instead of `nc` since macOS / BSD nc has different
    # flags than GNU nc and may not be on PATH at all.
    if not _port_listening("127.0.0.1", 8080, timeout=2):
        return CheckResult("metal-native", "warn",
            f"native llama-server installed at {binary} but nothing "
            f"listening on :8080 — start it with scripts/atlas-llama-macos.sh",
            "Open a separate terminal and run the launcher; this check "
            "will turn green once the server is up and serving.")

    return CheckResult("metal-native", "pass",
        f"native llama-server up at {binary}, listening on :8080")


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
            detail = body[:200]
            if name == "llama":
                # The most common cause: the model + KV + compute budget
                # doesn't fit in VRAM, so llama-server (--fit off) refuses
                # to start and the container crash-loops.
                detail += ("\nIf `docker compose logs llama-server` shows a "
                           "CUDA out-of-memory allocation error, the runtime "
                           "budget doesn't fit this GPU — run `atlas tier "
                           "fit --write`, then recreate the container.")
            results.append(CheckResult(f"health/{name}", "fail",
                "endpoint unreachable", detail))
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
            "Default model? run scripts/download-models.sh. "
            "Your own model? place the .gguf in ATLAS_MODELS_DIR "
            f"({MODEL_DIR}) and set ATLAS_MODEL_FILE + ATLAS_MODEL_NAME in "
            ".env — see docs/CONFIGURATION.md \"Adding your own model\", "
            "or run `atlas model install --url <hf-url>` / `atlas onboard`.")
    size = os.path.getsize(path)
    if size < 100 * 1024 * 1024:  # < 100 MB
        return CheckResult("model_file", "warn",
            f"{path} exists but only {size} bytes — likely truncated",
            "expected > 1 GB for a typical GGUF; re-download "
            "(scripts/download-models.sh for the default model, or re-fetch "
            f"your own .gguf into {MODEL_DIR}).")
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


def check_asa_steering(atlas_root: str) -> CheckResult:
    """ASA steering vector (BiasBusters #4) presence.

    Warn-not-fail: ATLAS works without it. When present, llama-server
    auto-applies it on startup via `--control-vector-scaled` (see
    `inference/entrypoint-v3.1-9b.sh`). When absent, the
    `ast_edit`-vs-`edit_file` proposal bias is unsteered and we lean
    entirely on the grammar gate downstream.

    Recovery is documented in `geometric-lens/asa_calibration/README.md`
    — or just re-run `./scripts/atlas-bootstrap.sh` which builds it as
    part of install (with HuggingFace prebuilt fallback).
    """
    # Match the entrypoint's default path so doctor and the inference
    # entrypoint check the same location.
    base = MODEL_DIR if os.path.isabs(MODEL_DIR) else os.path.join(atlas_root, MODEL_DIR)
    path = os.path.normpath(os.path.join(base, "ast_edit_steering.gguf"))
    override = os.environ.get("ATLAS_CONTROL_VECTOR", "")
    if override:
        path = override
    if not os.path.exists(path):
        return CheckResult("asa_steering", "warn",
            "ast_edit_steering.gguf not present",
            f"expected at {path} — build it via "
            f"`atlas asa build` (PC-061, runs in the lens container, "
            f"~25 min for the full 1000-pair training) or use the "
            f"manual workflow in `geometric-lens/asa_calibration/README.md`. "
            f"ATLAS continues to work without it; the ast_edit-vs-edit_file "
            f"proposal bias is just unsteered.")
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
    except OSError:
        size_mb = 0.0

    # PC-061 round-2 fix: when llama-server is reachable, run the deeper
    # dim-compat probe so doctor surfaces the same dim-mismatch verdict
    # `atlas asa check` would. Without this hook, doctor reports "pass"
    # even on a stale vector trained for a different model — the exact
    # failure mode PC-061 was supposed to surface. Best-effort: if the
    # asa module or its deps aren't importable, fall back to the
    # file-presence pass below.
    try:
        from atlas.cli.commands import asa as _asa
        verdict = _asa._check_asa(atlas_root)
        if verdict.verdict == "needs-build" and "Dim mismatch" in (verdict.reason or ""):
            return CheckResult("asa_steering", "fail",
                f"control vector dim mismatch",
                f"vector at {path} was trained for a different model "
                f"(dim {verdict.vector_dim}) than llama-server has loaded "
                f"(dim {verdict.probe.embedding_dim}). Run `atlas asa build` "
                f"to retrain for the current model.")
    except Exception:
        # Doctor keeps running even if the deeper probe fails — asa
        # isn't a hard dep of doctor, and a check shouldn't crash the
        # whole diagnostic.
        pass

    return CheckResult("asa_steering", "pass",
        f"ast_edit_steering.gguf ({size_mb:.1f} MB) at {path}")


def check_overcommit() -> CheckResult:
    """PC-011: Redis warns and AOF rewrite can fail without overcommit_memory=1.

    Linux-only — /proc/sys/vm/overcommit_memory doesn't exist on macOS
    or Windows. Short-circuit on non-Linux platforms with a clean skip
    instead of trying to read /proc and emitting a noisy 'could not
    read' message.
    """
    if sys.platform != "linux":
        return CheckResult("vm.overcommit_memory", "skip",
            f"not applicable on {sys.platform}", "")
    try:
        with open("/proc/sys/vm/overcommit_memory") as f:
            val = f.read().strip()
        if val == "1":
            return CheckResult("vm.overcommit_memory", "pass", "= 1")
        return CheckResult("vm.overcommit_memory", "warn",
            f"= {val} (Redis prefers 1 — see PC-011)",
            "Fix: sudo sysctl vm.overcommit_memory=1 && "
            "echo 'vm.overcommit_memory=1' | sudo tee /etc/sysctl.d/99-atlas.conf")
    except OSError as e:
        return CheckResult("vm.overcommit_memory", "skip",
            "could not read /proc/sys", str(e))


def check_tier_constraints(atlas_root: Optional[str] = None) -> CheckResult:
    """PC-055.1 cross-check: does the host meet the recommended tier's
    per-axis minimums (RAM, CPU, disk)?

    Distinct from `tier_match`:
      - `tier_match` asks "is the configured model right for this hardware?"
      - `tier_constraints` asks "can this hardware actually run anything at
        the tier we'd recommend, given ATLAS's CPU/RAM/disk needs?"

    Catches the "16 GB GPU but 8 GB RAM" case where llama-server fits on
    the GPU but the host OOMs during V3 pipeline + sandbox compiles.

    Passes `atlas_root` to tier.probe() so the disk-free check measures
    the partition where models will actually live (typically ATLAS_INSTALL_DIR
    or the repo root), not `/`. Without this, a user with `/opt/atlas` on
    a separate `/data` mount would get a misleading disk check.
    """
    try:
        from atlas.cli.commands import tier
    except ImportError as e:
        return CheckResult("tier_constraints", "skip",
            "tier module unavailable", str(e))
    p = tier.probe(install_dir=atlas_root)
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
        from atlas.cli.commands import tier, model_recommendations
    except ImportError as e:
        return CheckResult("tier_match", "skip",
            "tier module unavailable", str(e))
    p = tier.probe()
    if not p.has_gpu:
        return CheckResult("tier_match", "skip",
            "no GPU detected (cpu tier)")
    recommended = tier.classify(p)
    rec_model = model_recommendations.for_tier(recommended.tier)
    actual_model = MODEL_FILE
    if rec_model is not None and actual_model == rec_model.model_file:
        # PC-056.1: even on exact tier match, cross-check that the
        # claimed Lens artifacts actually exist on disk. Registry can
        # say "supported" while the .pt files are missing — config
        # drift that would otherwise hide G(x) silently no-opping.
        try:
            from atlas.cli.commands import model_registry
            atlas_root = _find_atlas_root()
            artifact_state = model_registry.lens_artifacts_present(
                rec_model, atlas_root)
            if not artifact_state["ok"]:
                return CheckResult("tier_match", "warn",
                    f"`{actual_model}` registered as Lens-supported but "
                    f"{len(artifact_state['missing_files'])} artifact "
                    f"file(s) missing: "
                    f"{', '.join(artifact_state['missing_files'])}",
                    f"Expected in {artifact_state['expected_dir']}. "
                    f"Without these files G(x) will silently no-op even "
                    f"though the registry says it should work. Either "
                    f"download the artifacts (see "
                    f"geometric-lens/geometric_lens/models/README.md) "
                    f"or set ATLAS_LENS_MODELS to point at a dir that "
                    f"has them.")
        except (ImportError, AttributeError):
            # best-effort: swallow on failure (caller continues)
            pass
        return CheckResult("tier_match", "pass",
            f"{recommended.tier} tier matches configured model "
            f"({rec_model.model_display})")
    # Mismatch — figure out direction. Reverse-lookup which tier owns
    # the configured model, then compare.
    actual_tier_name = model_recommendations.tier_for_model(actual_model)
    if actual_tier_name is None:
        return CheckResult("tier_match", "warn",
            f"configured model `{actual_model}` is not in any tier preset",
            f"host classified as {recommended.tier}; consider one of the "
            f"presets: `atlas tier list`")
    # Warn only when actual > recommended (overshoot risks OOM).
    # Undershoot (smaller model than tier supports) is fine — just
    # leaves performance on the table.
    tiers_order = ["cpu", "small", "medium", "large", "xlarge"]
    rec_idx = tiers_order.index(recommended.tier)
    act_idx = tiers_order.index(actual_tier_name)
    if act_idx > rec_idx:
        rec_display = (rec_model.model_display if rec_model is not None
                       else f"the {recommended.tier}-tier preset")
        return CheckResult("tier_match", "warn",
            f"running {actual_tier_name}-tier model on {recommended.tier}-tier "
            f"hardware ({p.vram_gb:.1f} GB VRAM)",
            f"OOM risk. Recommended for your VRAM: "
            f"{rec_display}. Run `atlas tier` for detail.")
    # Undershoot path: smaller model than the tier supports. Normally
    # safe (just leaves perf on the table). PC-056: also warn if the
    # actual model has no Lens artifacts — that means G(x) silently
    # no-ops at runtime, regardless of tier-fit. PC-056.1: also warn
    # if the model claims `supported` but the artifact files are
    # actually missing on disk — config drift between registry claim
    # and reality.
    try:
        from atlas.cli.commands import model_registry
        actual_model_record = model_registry.by_name(
            actual_model.rsplit(".", 1)[0])
        if actual_model_record is not None and \
                actual_model_record.lens_status != "supported":
            return CheckResult("tier_match", "warn",
                f"configured model `{actual_model}` has Lens status "
                f"`{actual_model_record.lens_status}` — G(x) will silently "
                f"no-op",
                "ATLAS will run llama-server but C(x)/G(x) verification is "
                "missing. See PC-058 roadmap. To switch: "
                "`atlas model recommend` for a Lens-supported alternative.")
        # PC-056.1: model claims supported — verify artifact files actually
        # exist where the registry says they should.
        if actual_model_record is not None and \
                actual_model_record.lens_status == "supported":
            atlas_root = _find_atlas_root()
            artifact_state = model_registry.lens_artifacts_present(
                actual_model_record, atlas_root)
            if not artifact_state["ok"]:
                return CheckResult("tier_match", "warn",
                    f"`{actual_model}` registered as Lens-supported but "
                    f"{len(artifact_state['missing_files'])} artifact "
                    f"file(s) missing: "
                    f"{', '.join(artifact_state['missing_files'])}",
                    f"Expected in {artifact_state['expected_dir']}. "
                    f"Without these files G(x) will silently no-op even "
                    f"though the registry says it should work. Either "
                    f"download the artifacts (see "
                    f"geometric-lens/geometric_lens/models/README.md) "
                    f"or set ATLAS_LENS_MODELS to point at a dir that "
                    f"has them.")
    except (ImportError, AttributeError):
        # best-effort: swallow on failure (caller continues)
        pass
    return CheckResult("tier_match", "pass",
        f"running {actual_tier_name}-tier model on {recommended.tier}-tier "
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

    Targets llama-server directly (not the proxy). The proxy's `/v1/agent`
    endpoint runs the full agent loop (tier classifier + V3 pipeline),
    which would add ~30s and frequently consume all `max_tokens` in
    routing/planning before producing visible content. Hitting llama
    directly answers the question we care about: "is the GGUF actually
    loaded and inferring?" The proxy's reachability is already covered
    by `health/proxy`. (`/v1/chat/completions` on the proxy itself is a
    raw passthrough to llama-server — no agent loop — but the smoke test
    skips the extra hop anyway.)
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
    # Show the remediation/detail by default for actionable problems (fail/warn)
    # — a failure with no visible fix-it hint is a dead end. `pass` details stay
    # behind --verbose to keep a healthy run terse.
    if (verbose or r.status in ("fail", "warn")) and r.detail:
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

    # 2.5. CPU architecture (#115) — surface aarch64 + the backend
    # availability matrix for arm64 hosts before the GPU check, so
    # users see why rocm gets steered to vulkan on DGX Spark / Snapdragon
    # X Elite / Apple Silicon / Jetson / Pi 5.
    results.append(check_arch())

    # 3. GPU runtime — vendor-aware (NVIDIA: nvidia-container-toolkit;
    # AMD: /dev/kfd passthrough). Slow on first run since each vendor
    # branch pulls a small base image (~500 MB CUDA, ~2 GB ROCm).
    results.append(check_gpu())

    # Resolve which backend the user has configured. Reads shell env
    # first, then .env in atlas_root. Without the .env fallback, the
    # macOS hybrid case (#32) misses the metal-native check because
    # atlas init writes ATLAS_BACKEND into .env and users rarely
    # source it before running doctor.
    backend = _resolve_backend(atlas_root)

    # 3.5. Vulkan ICD passthrough (PC-114) — only fires when the user
    # has explicitly opted into the Vulkan backend. Skipping by default
    # keeps doctor cheap on CUDA/ROCm hosts where the apt-install-
    # vulkan-tools step inside the check container would add ~30s for
    # no signal.
    if backend == "vulkan":
        results.append(_check_vulkan_via_docker())

    # 3.6. macOS hybrid path (#32) — verify native llama-server binary
    # exists at the setup-script's install prefix, is executable, and
    # is listening on :8080 (so the socat compose forward will succeed).
    # Only fires when backend == metal so it's noise-free on
    # cuda/rocm/vulkan hosts.
    if backend == "metal":
        results.append(_check_metal_native())

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

    # 8.5. ASA steering vector (BiasBusters #4 — warn-not-fail). Optional
    # but on by default when present; sits next to lens_weights since both
    # are host-side artifact checks.
    results.append(check_asa_steering(atlas_root))

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
    # Pass atlas_root so disk check measures the right partition.
    results.append(check_tier_constraints(atlas_root))

    # 11. End-to-end smoke
    if args.quick:
        results.append(CheckResult("e2e_smoke", "skip",
            "skipped (--quick)"))
    else:
        results.append(check_e2e_smoke())

    return _emit(results, args, color)


if __name__ == "__main__":
    sys.exit(main())
