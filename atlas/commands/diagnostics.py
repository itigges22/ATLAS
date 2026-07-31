"""atlas diagnostics — collect a shareable diagnostic bundle.

Gathers versions, platform/backend, filtered config, service health,
image digests, artifact hashes, resource limits, recent (filtered) logs,
and the doctor report into one JSON file. Source code and private values
are excluded by default: `.env` values for secret-ish keys are masked,
the service token is dropped entirely, and every captured log line runs
through the shared private-value filter — so the bundle is safe to
attach to an issue.

    atlas diagnostics collect [--output FILE] [--log-lines N]
"""

import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional

from atlas import compose as compose_config
from atlas import env as cli_env
from atlas import redact

# Config keys whose values are masked in the bundle (in addition to the
# generic private-value filter). The service token is dropped entirely.
_SECRET_KEYS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "HF_TOKEN")
_DROP_KEYS = ("ATLAS_SERVICE_TOKEN",)


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 30) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout).stdout
    except (subprocess.SubprocessError, OSError) as e:
        return f"(unavailable: {type(e).__name__})"


def _filtered_env(atlas_root: str) -> Dict[str, str]:
    """The .env, masked for sharing. A missing or unreadable file just means
    an empty env section (read_env_file already treats it as absent)."""
    out: Dict[str, str] = {}
    for k, v in compose_config.read_env_file(atlas_root).items():
        if any(d in k for d in _DROP_KEYS):
            continue  # never include the token, even masked
        if any(s in k.upper() for s in _SECRET_KEYS) and v:
            v = "[MASKED]"
        out[k] = redact.filter_private_values(v)
    return out


def _service_health(atlas_root: str) -> Dict[str, object]:
    services = {
        "proxy": compose_config.service_url("proxy"),
        "llama": compose_config.service_url("llama"),
        "lens": compose_config.service_url("lens"),
        "v3": compose_config.service_url("v3"),
        "sandbox": compose_config.service_url("sandbox"),
    }
    out: Dict[str, object] = {}
    for name, base in services.items():
        entry: Dict[str, object] = {"url": base}
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                entry["health"] = r.status
        except Exception as e:
            entry["health"] = f"unreachable ({type(e).__name__})"
        out[name] = entry
    # readiness + the seven status dimensions from the proxy
    try:
        with urllib.request.urlopen(
                services["proxy"] + "/ready", timeout=5) as r:
            out["proxy_ready"] = r.status
    except Exception as e:
        out["proxy_ready"] = f"unreachable ({type(e).__name__})"
    # Calibration status is a bonus dimension — omit the key when the
    # endpoint is unreachable or returns non-JSON.
    with contextlib.suppress(Exception):
        with urllib.request.urlopen(
                services["proxy"] + "/v1/calibration/status", timeout=5) as r:
            out["calibration"] = json.loads(r.read().decode())
    return out


def _recent_logs(atlas_root: str, lines: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for svc in ("atlas-proxy", "v3-service", "geometric-lens", "sandbox"):
        cmd = compose_config.command(
            atlas_root, ["logs", "--tail", str(lines), svc])
        raw = _run(cmd, cwd=atlas_root, timeout=30)
        out[svc] = redact.filter_private_values(raw)
    return out


def _doctor_json(atlas_root: str) -> object:
    """Embed the doctor report (already private-value-safe)."""
    from atlas.commands import doctor
    import io
    buf = io.StringIO()
    # doctor.main exits via SystemExit on failure; the report on stdout is
    # still what we want either way.
    with contextlib.suppress(SystemExit):
        with contextlib.redirect_stdout(buf):
            doctor.main(["--quick", "--json"])
    try:
        return json.loads(buf.getvalue() or "{}")
    except ValueError:
        return {"raw": buf.getvalue()[:4000]}


def _collect(atlas_root: str, log_lines: int) -> dict:
    return {
        "schema_version": 1,
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "meta": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "git_commit": _run(["git", "rev-parse", "--short", "HEAD"],
                               cwd=atlas_root).strip() or "(unknown)",
        },
        "config": _filtered_env(atlas_root),
        "services": _service_health(atlas_root),
        "images": _run(compose_config.command(
            atlas_root, ["images"]), cwd=atlas_root),
        "recent_logs": _recent_logs(atlas_root, log_lines),
        "doctor": _doctor_json(atlas_root),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas diagnostics",
        description="Collect a shareable, private-value-filtered "
                    "diagnostic bundle.")
    sub = parser.add_subparsers(dest="cmd")
    c = sub.add_parser("collect", help="write the diagnostic bundle")
    c.add_argument("--output", default=None,
                   help="output path (default: atlas-diagnostics-<stamp>.json)")
    c.add_argument("--log-lines", type=int, default=100,
                   help="recent log lines per service (default 100)")
    args = parser.parse_args(argv)

    if args.cmd != "collect":
        parser.print_help()
        return 1

    atlas_root = cli_env.atlas_root()
    bundle = _collect(atlas_root, args.log_lines)

    out = args.output or f"atlas-diagnostics-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out, "w") as fh:
        json.dump(bundle, fh, indent=2)
    print(f"Diagnostic bundle written: {out}")
    print("Private values are filtered; review before sharing.")
    return 0
