"""Typed configuration schema + validator for the ATLAS `.env`.

A single typed spec for the ATLAS_* knobs: type (int/float/port/bool/
enum/str), range, allowed values, and deprecation. `validate()` returns
structured problems (type errors, out-of-range, bad enum, unknown keys,
deprecated keys) so misconfiguration is caught before startup rather
than surfacing as an opaque container failure.

CONFIG_SCHEMA_VERSION tracks the config contract; a `.env` carrying an
older/absent version is migrated forward additively (unknown-but-removed
keys are ignored, new keys take their defaults). See migrate().
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

CONFIG_SCHEMA_VERSION = 1

_BOOLS = {"0", "1", "true", "false", "yes", "no"}


@dataclass
class Field:
    kind: str  # int | float | port | bool | enum | str
    enum: Optional[Tuple[str, ...]] = None
    # int for int/port fields, int or float for float fields.
    min: Optional[float] = None
    max: Optional[float] = None
    deprecated: Optional[str] = None  # reason/replacement if deprecated


SCHEMA: Dict[str, Field] = {
    "ATLAS_MODEL_FILE": Field("str"),
    "ATLAS_MODEL_NAME": Field("str"),
    "ATLAS_MODELS_DIR": Field("str"),
    "ATLAS_CTX_SIZE": Field("int", min=512, max=2_000_000),
    "ATLAS_PARALLEL_SLOTS": Field("int", min=1, max=64),
    "ATLAS_UBATCH": Field("int", min=1, max=100_000),
    "ATLAS_BATCH": Field("int", min=1, max=100_000),
    "ATLAS_KV_TYPE_K": Field("enum", enum=("f16", "q8_0", "q4_0")),
    "ATLAS_KV_TYPE_V": Field("enum", enum=("f16", "q8_0", "q4_0")),
    "ATLAS_BACKEND": Field("enum", enum=("cuda", "rocm", "vulkan", "metal")),
    "ATLAS_GPU_INDEX": Field("int", min=0, max=64),
    "ATLAS_GRAMMAR_MODE": Field("enum", enum=("strict", "loose")),
    "ATLAS_LOG_FORMAT": Field("enum", enum=("line", "json")),
    "ATLAS_TRUST_MODE": Field("enum",
                              enum=("untrusted", "trusted", "fully-trusted")),
    "ATLAS_VERIFY_IN": Field("enum", enum=("sandbox", "host")),
    "ATLAS_CALL_GRAPH": Field("bool"),
    "ATLAS_RPG_PLANNING": Field("bool"),
    "ATLAS_KEEP_LLAMA_WARM": Field("bool"),
    "ATLAS_FRESH_SLOT_PER_SESSION": Field("bool"),
    "ATLAS_DEDUP_READS": Field("bool"),
    "ATLAS_SANDBOX_NET_INTERNAL": Field("bool"),
    "ATLAS_MAX_TOKENS": Field("int", min=0, max=1_000_000),
    # Repetition-control sampling forwarded to llama-server. DRY scores
    # repeated sequences; repeat_penalty scores individual tokens and is off
    # by default because it degrades code, where indentation and keywords
    # repeat legitimately. 1.0 means "no penalty" for repeat_penalty.
    "ATLAS_DRY_MULTIPLIER": Field("float", min=0, max=10),
    "ATLAS_DRY_BASE": Field("float", min=1, max=10),
    "ATLAS_DRY_ALLOWED_LENGTH": Field("int", min=1, max=1000),
    "ATLAS_DRY_PENALTY_LAST_N": Field("int", min=-1, max=2_000_000),
    "ATLAS_REPEAT_PENALTY": Field("float", min=0.5, max=2),
    "ATLAS_REPEAT_LAST_N": Field("int", min=0, max=100_000),
    "ATLAS_MAX_TURNS": Field("int", min=0, max=1000),
    "ATLAS_LENS_RETRAIN_MIN": Field("int", min=0, max=10_000_000),
    "ATLAS_SANDBOX_PIDS": Field("int", min=1, max=1_000_000),
    "ATLAS_SANDBOX_UID": Field("int", min=0, max=2_000_000),
    "ATLAS_SANDBOX_GID": Field("int", min=0, max=2_000_000),
    "ATLAS_SANDBOX_MAX_EXECUTION_TIME": Field("int", min=1, max=86_400),
    "ATLAS_LLAMA_PORT": Field("port"),
    "ATLAS_LENS_PORT": Field("port"),
    "ATLAS_PROXY_PORT": Field("port"),
    "ATLAS_SANDBOX_PORT": Field("port"),
    "ATLAS_V3_PORT": Field("port"),
    "ATLAS_V3_TIMEOUT": Field("int", min=0, max=86_400),
    "ATLAS_V3_URL": Field("str"),
    "ATLAS_PLAN_THINKING": Field("str"),
    "ATLAS_SHELL_SNAPSHOT_MAX_FILES": Field("int", min=0, max=100_000_000),
    "ATLAS_SHELL_SNAPSHOT_MAX_BYTES": Field("int", min=0),
    "ATLAS_SHELL_SNAPSHOT_MAX_FILE_BYTES": Field("int", min=0),
    "ATLAS_CONTROL_VECTOR_ALLOW_UNVERIFIED": Field("bool"),
    "ATLAS_CONTROL_VECTOR_LAYER_RANGE": Field("str"),
    "ATLAS_SERVICE_TOKEN_FILE": Field("str"),
    "ATLAS_ALLOW_CREDENTIAL_READS": Field("bool"),
    "ATLAS_CONFIG_SCHEMA_VERSION": Field("int", min=1),
    # str/path knobs (presence-only)
    "ATLAS_IMAGE_TAG": Field("str"),
    "ATLAS_GHCR_OWNER": Field("str"),
    "ATLAS_PROJECT_DIR": Field("str"),
    "ATLAS_LENS_MODELS": Field("str"),
    "ATLAS_LENS_HOST_DIR": Field("str"),
    "ATLAS_SECRETS_DIR": Field("str"),
    "ATLAS_MACOS_PREFIX": Field("str"),
    "ATLAS_LLAMA_HOST": Field("str"),
    "ATLAS_GPU_INDEX_LIST": Field("str"),
    "ATLAS_AGENT_HISTORY_BUDGET": Field("int", min=0, max=10_000_000),
    "ATLAS_SANDBOX_MEM": Field("str"),   # "4g"/"0"/bytes — freeform
    "ATLAS_SANDBOX_CPUS": Field("str"),  # "2"/"0.5"/"0" — freeform
    # Sandbox tmpfs size ceilings (docker tmpfs size strings: "2G"/"512M")
    "ATLAS_SANDBOX_TMP_SIZE": Field("str"),
    "ATLAS_SANDBOX_PIP_SIZE": Field("str"),
    "ATLAS_SANDBOX_CACHE_SIZE": Field("str"),
    # Lens SQLite state store path (host/dev runs; compose pins the
    # container path). Not ATLAS_-prefixed: read directly by the lens
    # service, listed here so migrate() carries it forward.
    "SQLITE_DB_PATH": Field("str"),
    # Removed keys: ignored on read, flagged as deprecated on validate.
    "ATLAS_ENABLE_TRAINING": Field("bool",
        deprecated="removed; training is always available"),
    "ATLAS_REGISTRY": Field("str",
        deprecated="removed; the registry is in-package"),
    "ATLAS_REDIS_MAXMEMORY": Field("str",
        deprecated="removed; lens state is SQLite (SQLITE_DB_PATH)"),
    "ATLAS_REDIS_MEM": Field("str",
        deprecated="removed; lens state is SQLite (SQLITE_DB_PATH)"),
}


def _check_value(key: str, val: str, spec: Field) -> Optional[str]:
    if spec.kind == "bool":
        if val.lower() not in _BOOLS:
            return f"{key}={val!r}: expected a boolean ({'/'.join(sorted(_BOOLS))})"
        return None
    if spec.kind == "enum":
        if spec.enum and val not in spec.enum:
            return f"{key}={val!r}: expected one of {spec.enum}"
        return None
    if spec.kind in ("int", "port"):
        try:
            n = int(val)
        except ValueError:
            return f"{key}={val!r}: expected an integer"
        lo = 1 if spec.kind == "port" else spec.min
        hi = 65535 if spec.kind == "port" else spec.max
        if lo is not None and n < lo:
            return f"{key}={n}: below minimum {lo}"
        if hi is not None and n > hi:
            return f"{key}={n}: above maximum {hi}"
        return None
    if spec.kind == "float":
        try:
            f = float(val)
        except ValueError:
            return f"{key}={val!r}: expected a number"
        if spec.min is not None and f < spec.min:
            return f"{key}={f}: below minimum {spec.min}"
        if spec.max is not None and f > spec.max:
            return f"{key}={f}: above maximum {spec.max}"
        return None
    return None  # str: presence only


def validate(env: Dict[str, str]) -> Dict[str, List[str]]:
    """Return {errors: [...], warnings: [...]}.

    errors: type/range/enum violations (block startup).
    warnings: unknown keys (typos) + deprecated keys (ignored but noisy).
    """
    errors: List[str] = []
    warnings: List[str] = []
    for key, val in env.items():
        if not key.startswith("ATLAS_"):
            continue  # non-ATLAS keys (compose/docker) are out of scope
        spec = SCHEMA.get(key)
        if spec is None:
            warnings.append(f"{key}: unknown ATLAS_ config key (typo?)")
            continue
        if spec.deprecated:
            warnings.append(f"{key}: deprecated — {spec.deprecated}")
            continue
        if val == "":
            continue  # empty = use in-code default
        problem = _check_value(key, val, spec)
        if problem:
            errors.append(problem)
    return {"errors": errors, "warnings": warnings}


def migrate(env: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """Forward-migrate a config dict to the current schema version.

    Additive + tolerant: deprecated/removed keys are dropped (recorded),
    everything else is preserved. Returns (migrated_env, notes).
    """
    notes: List[str] = []
    out: Dict[str, str] = {}
    for key, val in env.items():
        spec = SCHEMA.get(key)
        if spec and spec.deprecated:
            notes.append(f"dropped {key} (deprecated: {spec.deprecated})")
            continue
        out[key] = val
    out["ATLAS_CONFIG_SCHEMA_VERSION"] = str(CONFIG_SCHEMA_VERSION)
    return out, notes


# ---------------------------------------------------------------------------
# Precedence-aware resolution
# ---------------------------------------------------------------------------
# Documented precedence (highest first): process environment, then the
# compose .env file, then the caller-supplied default. This is the single
# place the layered lookup lives, so every reader agrees (previously each
# call site hand-rolled `os.environ.get(k) or env.get(k)`).

import os as _os


def resolve(key: str, env_file: Optional[Dict[str, str]] = None,
            default: Optional[str] = None,
            environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Resolve a config key's raw string value by precedence:
    process env > .env file > default. An empty string in a higher layer
    does NOT shadow a lower one (empty = 'unset' by ATLAS convention)."""
    env = environ if environ is not None else _os.environ
    val = env.get(key)
    if val not in (None, ""):
        return val
    if env_file:
        val = env_file.get(key)
        if val not in (None, ""):
            return val
    return default


def _coerce(key: str, raw: str) -> object:
    spec = SCHEMA.get(key)
    if spec is None:
        return raw
    if spec.kind == "bool":
        return raw.strip().lower() in ("1", "true", "yes")
    if spec.kind in ("int", "port"):
        try:
            return int(raw)
        except ValueError:
            return raw
    return raw


def resolve_typed(key: str, env_file: Optional[Dict[str, str]] = None,
                  default: Optional[str] = None,
                  environ: Optional[Dict[str, str]] = None) -> object:
    """Resolve + coerce to the schema's type (int/port -> int, bool ->
    bool, else str). Unknown keys return the raw string."""
    raw = resolve(key, env_file, default, environ)
    if raw is None:
        return None
    return _coerce(key, raw)
