#!/usr/bin/env python3
"""Capture a reproducibility manifest of the running benchmark environment.

Writes /workspace/results/manifest.json with everything a reviewer needs to
verify a result was produced from a known-good build:

  - git SHA + dirty flag (snapshot of the source the container was built from)
  - vLLM version, transformers version, key Python deps via `pip freeze`
  - model SHA256 of every weight shard + tokenizer file
  - hardware fingerprint (nvidia-smi, /proc/cpuinfo, /proc/meminfo, lspci)
  - container env (env vars matching ATLAS_/LLAMA_/MODEL_/GEN_/EMBED_/HF_/MODE)
  - container start UTC, snapshot UTC
  - run id (UTC timestamp, used as the tarball filename)

Idempotent: re-running overwrites the manifest with current state. Cheap
(~1-2 s) so safe to call at start, on SIGTERM, and at end.

Designed to run inside the cloud-pod container with everything mounted.
"""

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

def _default_output() -> Path:
    """Honor explicit ATLAS_MANIFEST_PATH first, then RESULT_DIR (set by
    snapshot.sh), then fall back to the cloud-pod default. This lets the
    rehearsal harness on a dev box redirect both the snapshot and the
    manifest to a temp dir without container-only paths leaking through."""
    explicit = os.environ.get("ATLAS_MANIFEST_PATH")
    if explicit:
        return Path(explicit)
    result_dir = os.environ.get("RESULT_DIR")
    if result_dir:
        return Path(result_dir) / "manifest.json"
    return Path("/workspace/results/manifest.json")


OUTPUT = _default_output()
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/workspace/models/Qwen3.5-9B-AWQ"))
ATLAS_REPO = Path(os.environ.get("ATLAS_REPO", "/workspace/ATLAS"))


def _safe_run(cmd: List[str], timeout: int = 30) -> str:
    """Run cmd, return stdout or short error string. Never raises."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (out.stdout or out.stderr or "").strip()
    except FileNotFoundError:
        return f"<{cmd[0]} not on PATH>"
    except subprocess.TimeoutExpired:
        return f"<{cmd[0]} timeout>"
    except Exception as e:
        return f"<error: {e}>"


def _read_text(path: Path, max_bytes: int = 65536) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        return data.decode(errors="replace")
    except Exception:
        return None


def git_state() -> Dict[str, Any]:
    """Source provenance: SHA, dirty flag, and the commit subject."""
    if not (ATLAS_REPO / ".git").exists():
        # Container build typically COPYs source without .git, so this is
        # the common case. Fall back to a marker file written at build time.
        marker = ATLAS_REPO / ".git-sha"
        if marker.exists():
            return {"sha": marker.read_text().strip(), "dirty": None, "source": "build-time marker"}
        return {"sha": None, "dirty": None, "source": "no .git, no marker"}
    return {
        "sha": _safe_run(["git", "-C", str(ATLAS_REPO), "rev-parse", "HEAD"]),
        "dirty": bool(_safe_run(["git", "-C", str(ATLAS_REPO), "status", "--porcelain"])),
        "subject": _safe_run(["git", "-C", str(ATLAS_REPO), "log", "-1", "--pretty=%s"]),
        "source": "live .git",
    }


def python_packages() -> Dict[str, str]:
    """Pin a curated subset (vLLM/transformers/torch/numpy/etc) — full
    `pip freeze` is captured as a separate file for completeness."""
    pkgs = {}
    for name in [
        "vllm", "torch", "transformers", "tokenizers", "numpy", "xgboost",
        "scikit-learn", "fastapi", "uvicorn", "huggingface-hub", "accelerate",
    ]:
        out = _safe_run(["pip", "show", name])
        m = re.search(r"^Version:\s*(\S+)", out, re.MULTILINE)
        pkgs[name] = m.group(1) if m else "<missing>"
    return pkgs


def model_fingerprint(model_dir: Path) -> Dict[str, Any]:
    """SHA256 every safetensors + tokenizer + config file. Lets a reviewer
    verify the exact weights used."""
    if not model_dir.exists():
        return {"path": str(model_dir), "present": False}
    files = {}
    total_bytes = 0
    for p in sorted(model_dir.rglob("*")):
        if not p.is_file():
            continue
        # Skip .cache/ and obvious noise
        if any(part.startswith(".") for part in p.relative_to(model_dir).parts):
            continue
        if p.suffix in {".safetensors", ".json", ".jinja", ".txt"} or p.name in {"merges.txt"}:
            try:
                h = hashlib.sha256()
                size = 0
                with open(p, "rb") as fh:
                    while True:
                        chunk = fh.read(1 << 20)
                        if not chunk:
                            break
                        h.update(chunk)
                        size += len(chunk)
                files[str(p.relative_to(model_dir))] = {
                    "sha256": h.hexdigest(),
                    "bytes": size,
                }
                total_bytes += size
            except Exception as e:
                files[str(p.relative_to(model_dir))] = {"error": str(e)}
    return {
        "path": str(model_dir),
        "present": True,
        "total_bytes": total_bytes,
        "files": files,
    }


def hardware() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "uname": platform.uname()._asdict(),
        "cpu_count": os.cpu_count(),
        "nvidia_smi": _safe_run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap,pci.bus_id",
            "--format=csv",
        ]),
        "nvidia_smi_full": _safe_run(["nvidia-smi"]),
        "cuda_version_runtime": _safe_run(["nvcc", "--version"]),
        "meminfo": _read_text(Path("/proc/meminfo"), max_bytes=2048),
        "cpuinfo_first": "\n".join((_read_text(Path("/proc/cpuinfo"), max_bytes=8192) or "").splitlines()[:30]),
    }


def env_snapshot() -> Dict[str, str]:
    """Just the env vars that drive the run. We do NOT include HF_TOKEN /
    secrets — those are masked."""
    pattern = re.compile(r"^(ATLAS_|LLAMA_|MODEL_|GEN_|EMBED_|HF_|MODE$|BENCHMARK_|GEOMETRIC_|RAG_|LENS_|RESULT_|SHUTDOWN_|SKIP_|DOWNLOAD_|PYTHON|CUDA|NVIDIA|VLLM_)")
    out = {}
    for k, v in sorted(os.environ.items()):
        if not pattern.search(k):
            continue
        if any(s in k.upper() for s in ("TOKEN", "SECRET", "KEY", "PASSWORD")):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def vllm_runtime() -> Dict[str, Any]:
    """Snapshot of what the live vLLM instances think they're serving.
    Useful to confirm served-model-name / config matches manifest.

    Returns empty dict if vLLM isn't reachable yet (manifest captured at
    container start, before vLLM is up)."""
    out = {"gen": None, "embed": None}
    import urllib.request
    for key, port in (("gen", os.environ.get("GEN_PORT", "8000")),
                      ("embed", os.environ.get("EMBED_PORT", "8001"))):
        url = f"http://localhost:{port}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                out[key] = json.loads(r.read().decode())
        except Exception as e:
            out[key] = {"unreachable": str(e)}
    return out


def main() -> int:
    started = os.environ.get("ATLAS_RUN_STARTED")
    if not started:
        # First call writes the start time; subsequent calls preserve it.
        started = datetime.now(timezone.utc).isoformat()
        # Persist for sibling calls
        os.environ["ATLAS_RUN_STARTED"] = started

    manifest = {
        "schema_version": 1,
        "run_id": os.environ.get("ATLAS_RUN_ID") or started.replace(":", "-"),
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "container_started_utc": started,
        "git": git_state(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "packages": python_packages(),
        },
        "model": model_fingerprint(MODEL_PATH),
        "env": env_snapshot(),
        "hardware": hardware(),
        "vllm_runtime": vllm_runtime(),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    print(f"manifest written to {OUTPUT} ({OUTPUT.stat().st_size} bytes)")

    # Also write a separate full pip freeze (more complete than the curated
    # subset above). Useful for full reproducibility audits.
    freeze = _safe_run(["pip", "freeze"])
    (OUTPUT.parent / "pip_freeze.txt").write_text(freeze)
    return 0


if __name__ == "__main__":
    sys.exit(main())
