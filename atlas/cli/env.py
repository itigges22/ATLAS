"""Shared CLI environment resolution.

Parses the compose .env (the Docker deployment's source of truth) and
resolves the service URLs and model settings the CLI commands share.
Shell environment wins over .env values. Lives outside the command
modules so doctor, fit, lens, and publish read one source without
importing each other.
"""

import os
from typing import Dict

from atlas.cli import compose as compose_config


def _read_dotenv() -> Dict[str, str]:
    """Parse the compose .env (the Docker deployment's source of truth) by
    walking up from this file. Lets the model/dir checks reflect what's actually
    configured when the shell env doesn't export ATLAS_MODEL_FILE."""
    cur = os.path.dirname(os.path.abspath(__file__))
    # 7 hops from atlas/cli reaches the same highest ancestor as the
    # previous 8 hops from atlas/cli/commands — the walk must not gain an
    # extra ancestor (in venv layouts one more hop can reach $HOME and
    # pick up a foreign ~/.env).
    for _ in range(7):
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
                            stripped = v.lstrip()
                            if stripped.startswith("#") and stripped != v:
                                # Empty value followed by an inline comment
                                # ("KEY= # note") parses as empty.
                                v = ""
                            else:
                                v = stripped
                                head, hash_sep, _ = v.partition("#")
                                if hash_sep and head and head[-1] in " \t":
                                    v = head
                            out[k.strip()] = v.strip().strip('"').strip("'")
            except OSError:
                # An unreadable optional .env is treated as absent; callers
                # continue with process-environment values and diagnostics.
                pass
            return out
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return {}


_ENV = _read_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name) or _ENV.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Defaults — shell env first, then the compose .env's port keys. Model
# selection has no vendor fallback: the installer must choose a concrete
# model explicitly.
PROXY_URL    = compose_config.service_url("proxy",   values=_ENV)
LLAMA_URL    = compose_config.service_url("llama",   values=_ENV)
LENS_URL     = compose_config.service_url("lens",    values=_ENV)
SANDBOX_URL  = compose_config.service_url("sandbox", values=_ENV)
V3_URL       = compose_config.service_url("v3",      values=_ENV)
MODEL_DIR    = os.environ.get("ATLAS_MODELS_DIR")  or _ENV.get("ATLAS_MODELS_DIR", "./models")
MODEL_FILE   = os.environ.get("ATLAS_MODEL_FILE")  or _ENV.get("ATLAS_MODEL_FILE", "")
MODEL_NAME   = os.environ.get("ATLAS_MODEL_NAME")  or _ENV.get("ATLAS_MODEL_NAME", "local-model")
LLAMA_PORT   = _env_int("ATLAS_LLAMA_PORT", 8080)
# Match docker-compose.yml's `${ATLAS_LENS_MODELS:-./geometric-lens/geometric_lens/models}`
# host-side bind-mount source so checks see the same directory the
# container will actually receive.
LENS_MODELS_DIR = (os.environ.get("ATLAS_LENS_MODELS")
                   or _ENV.get("ATLAS_LENS_MODELS")
                   or "./geometric-lens/geometric_lens/models")
