"""atlas lens — Geometric Lens probe + build pipeline (PC-057, PC-058).

Two subcommands wrap the existing geometric-lens training code into a
model-path-driven workflow so users can bring their own GGUF and either
verify it's Lens-compatible (`check`) or actually train fresh artifacts
for it (`build`).

Layering:
    PC-057 `atlas lens check`  -> this file, cheap pre-flight
    PC-058 `atlas lens build`  -> this file, wraps training.train_cost_field
    PC-059 `atlas lens push`   -> roadmap, publishes to registry
    PC-060 HF middleman        -> roadmap, automated distribution

Probe contract: both subcommands talk to a *running* llama-server via
its `/embedding` and `/props` endpoints. ATLAS users typically already
have one up (`docker compose up -d`); if not, the commands print a
clear "start the stack first" hint rather than spinning their own
process. Keeping this stateless against an existing server matches the
rest of the atlas CLI surface (model.py, doctor.py, tier.py all assume
some level of running infrastructure for their richer probes).

Invoke:
    atlas lens check                    # probe currently-loaded model
    atlas lens check <model-name>       # probe a registry entry
    atlas lens check /path/to/model.gguf  # probe an arbitrary file
    atlas lens build <name|path>        # train fresh C(x) artifacts
    atlas lens --json                   # machine-readable output for scripts

Exit codes (check):
    0  compat        — artifacts exist + dim matches + server reachable
    1  needs-build   — model loadable, no artifacts at right dim
    2  incompatible  — can't probe (server down, model won't load, PC-202 missing)
"""

import argparse
import json as jsonlib
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from atlas.cli import compose as compose_config
from atlas.cli.commands import model_registry


# Reuse the color + safe-print primitives from tier/doctor for consistency.
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELL = "\033[33m"
CYAN = "\033[36m"


def _supports_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


UNICODE_OK = _supports_unicode()


def _safe_print(s: str = "") -> None:
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# llama-server probe helpers
# ---------------------------------------------------------------------------

def _llama_url() -> str:
    """Resolve where llama-server is listening.

    Mirrors geometric-lens/embedding_extractor.py's resolution order so
    `atlas lens check` agrees with what the lens service itself sees,
    then falls back to the Docker .env's port keys via compose config.
    """
    for key in ("ATLAS_LLAMA_URL", "LLAMA_EMBED_URL", "LLAMA_URL"):
        value = os.environ.get(key)
        if value:
            return value
    return compose_config.service_url("llama")


@dataclass
class LlamaProbe:
    """Snapshot of what the running llama-server can tell us about the model."""
    reachable: bool
    url: str
    embedding_dim: int = 0          # 0 when /embedding failed or didn't return
    n_layers: int = 0               # 0 when /props didn't carry n_layer
    model_name: str = ""            # whatever /props reports (often a path)
    has_hidden_states_patch: bool = False  # PC-202: layers extension present
    error: str = ""                 # short human description when reachable=False


def _http_get(url: str, timeout: float = 5.0) -> Optional[dict]:
    """GET a JSON endpoint. Returns parsed dict or None on any failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return jsonlib.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError,
            jsonlib.JSONDecodeError, OSError, ValueError):
        return None


def _http_post_json(url: str, body: dict, timeout: float = 30.0) -> Optional[dict]:
    """POST JSON, parse response. Returns parsed obj or None on failure."""
    payload = jsonlib.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return jsonlib.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError,
            jsonlib.JSONDecodeError, OSError, ValueError):
        return None


def probe_llama(url: Optional[str] = None,
                sample_text: str = "def hello():\n    return 42") -> LlamaProbe:
    """Discover what the running llama-server knows about its loaded model.

    Three probes:
      1. /health  -> is the server reachable at all?
      2. /props   -> model metadata (n_layer, model_name)
      3. /embedding (POST) -> the authoritative embedding dim, plus a
         PC-202 hidden-states ping (layers=[0]) to detect the patch.

    Failures degrade gracefully: a probe step that times out or returns
    a non-JSON body sets that field to its zero value rather than
    raising. Caller inspects reachable + embedding_dim to decide verdict.
    """
    url = url or _llama_url()
    probe = LlamaProbe(reachable=False, url=url)

    # 1. /health — fast existence check
    health = _http_get(f"{url}/health", timeout=3.0)
    if health is None:
        probe.error = (f"llama-server not reachable at {url}. "
                       f"Bring the stack up with `docker compose up -d` "
                       f"(or `docker compose -f docker-compose.yml "
                       f"-f docker-compose.rocm.yml up -d` on AMD), "
                       f"then re-run.")
        return probe
    probe.reachable = True

    # 2. /props — n_layer + model name. Field names changed across
    # llama-server versions; tolerate both `n_layer` and `default_generation_settings`.
    props = _http_get(f"{url}/props", timeout=5.0) or {}
    probe.n_layers = (
        int(props.get("n_layer", 0))
        or int((props.get("default_generation_settings") or {}).get("n_layer", 0))
    )
    probe.model_name = (props.get("model_path")
                        or props.get("model_name")
                        or props.get("model")
                        or "")

    # 3. /embedding — authoritative dim. Send a small sample; pooled or
    # per-token both yield the dim. The PC-202 patch is signalled by the
    # presence of `hidden_states_dim` in the response when `layers` is
    # requested; on an unpatched server the field is silently absent.
    emb = _http_post_json(f"{url}/embedding",
                          {"content": sample_text, "layers": [0]},
                          timeout=30.0)
    if isinstance(emb, list) and emb:
        first = emb[0]
        if isinstance(first, dict):
            raw = first.get("embedding")
            if isinstance(raw, list) and raw:
                if isinstance(raw[0], list):
                    probe.embedding_dim = len(raw[0])  # per-token
                else:
                    probe.embedding_dim = len(raw)     # pooled
            if "hidden_states_dim" in first:
                probe.has_hidden_states_patch = True
    if probe.embedding_dim == 0:
        probe.error = (f"llama-server at {url} is up but /embedding "
                       f"didn't return an embedding. Likely cause: model "
                       f"was started without `--embeddings`. Check "
                       f"inference/entrypoint-v3.1.sh.")
    return probe


# ---------------------------------------------------------------------------
# Artifact resolution + dim inspection
# ---------------------------------------------------------------------------

def _resolve_model_arg(arg: Optional[str]) -> Optional[model_registry.Model]:
    """Best-effort lookup: registry name → Model, or path/None → None.

    `atlas lens check` accepts:
      - a registry name        (e.g. "your-model-Q4_K_M")
      - a .gguf path           (any model on disk)
      - nothing                (probe whatever llama-server has loaded)
    """
    if not arg:
        return None
    for m in model_registry.REGISTRY:
        if m.name == arg or m.model_file == os.path.basename(arg):
            return m
    return None


def _canonical_model_identity(value: Optional[str]) -> str:
    """Normalize registry names, GGUF filenames, and model paths."""
    text = str(value or "").strip().replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return name.casefold()


@dataclass
class ArtifactInspection:
    """Result of looking at the on-disk Lens artifact."""
    present: bool                      # cost_field.pt exists on disk
    dim: Optional[int] = None          # input dim if introspectable
    torch_available: bool = True       # False -> dim couldn't be checked
    error: str = ""


def _inspect_cost_field(artifact_dir: str) -> ArtifactInspection:
    """Look at cost_field.pt and report what we can.

    Three outcomes:
      1. File missing                -> present=False
      2. File present, torch missing -> present=True, dim=None,
                                         torch_available=False
      3. File present, torch present -> present=True, dim=<int>
                                         (or None on a load error,
                                          with `error` set)
    Distinguishing (2) from (3) lets the verdict avoid misleading users
    into a needs-build state when the artifact really exists but the
    host Python just can't peek at it.
    """
    cost_path = os.path.join(artifact_dir, "cost_field.pt")
    if not os.path.isfile(cost_path):
        return ArtifactInspection(present=False)
    try:
        import torch
    except ImportError:
        return ArtifactInspection(present=True, dim=None,
                                  torch_available=False,
                                  error="torch not installed on host")
    try:
        state = torch.load(cost_path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            state = torch.load(cost_path, map_location="cpu")
        except Exception as e:
            return ArtifactInspection(present=True, dim=None,
                                      error=f"torch.load failed: {e}")
    if not isinstance(state, dict):
        return ArtifactInspection(present=True, dim=None,
                                  error="state dict has unexpected shape")
    # CostField.net.0 is the first Linear layer; its weight is (out, in).
    for key in ("net.0.weight", "0.weight"):
        if key in state:
            try:
                return ArtifactInspection(present=True,
                                          dim=int(state[key].shape[1]))
            except Exception:
                continue
    return ArtifactInspection(present=True, dim=None,
                              error="no recognized first-layer weight key")


# Back-compat shim — older callers (and the test suite) read just the dim.
def _read_saved_cost_field_dim(artifact_dir: str) -> Optional[int]:
    return _inspect_cost_field(artifact_dir).dim


def _missing_runtime_artifacts(artifact_dir: str) -> List[str]:
    """Return files required for calibrated C(x)+G(x) operation."""
    required = (
        "model_identity.json",
        "cx_normalization.json",
        "gx_xgboost.json",
        "gx_weights.json",
        "gx_thresholds.json",
    )
    return [name for name in required
            if not os.path.isfile(os.path.join(artifact_dir, name))]


def _invalid_runtime_artifacts(artifact_dir: str,
                               selected_model: str = "",
                               embedding_dim: int = 0) -> List[str]:
    """Return concise validation errors for present model-coupled metadata.

    Existence alone is not enough: accepting ``{}``, booleans, NaN, or an
    inverted threshold order would let publishing claim calibrated support
    while the runtime quietly disables interventions.
    """
    gl_dir = os.path.join(_atlas_root(), "geometric-lens")
    if gl_dir not in sys.path:
        sys.path.insert(0, gl_dir)
    try:
        from geometric_lens.calibration import validate_cx_normalization
        from geometric_lens.identity import (
            identity_matches,
            validate_model_identity,
        )
        from geometric_lens.thresholds import validate_gx_thresholds
    except ImportError as exc:
        return [f"calibration validators unavailable: {exc}"]

    errors = []
    validators = {
        "model_identity.json": validate_model_identity,
        "cx_normalization.json": validate_cx_normalization,
        "gx_thresholds.json": validate_gx_thresholds,
    }
    for filename, validator in validators.items():
        path = os.path.join(artifact_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                value = jsonlib.load(fh)
            validated = validator(value)
            if (filename == "model_identity.json" and selected_model
                    and not identity_matches(validated, selected_model,
                                             embedding_dim)):
                errors.append(
                    f"{filename}: artifact is for {validated['model']!r}, "
                    f"not {selected_model!r} at {embedding_dim or 'unknown'}-dim"
                )
        except (OSError, ValueError, TypeError, jsonlib.JSONDecodeError) as exc:
            errors.append(f"{filename}: {exc}")
    return errors


# ---------------------------------------------------------------------------
# atlas lens check  (PC-057)
# ---------------------------------------------------------------------------

@dataclass
class CheckVerdict:
    verdict: str          # 'compat' | 'needs-build' | 'incompatible'
    reason: str
    probe: LlamaProbe
    artifact_dir: Optional[str] = None
    artifact_dim: Optional[int] = None
    matched_model: Optional[str] = None
    # True when the artifact is present on disk but its input dim couldn't
    # be introspected (typically because torch isn't installed on the host
    # Python). Verdict stays "compat" (don't push users to needs-build for
    # a host-tooling gap) but JSON consumers can branch on this.
    unverified: bool = False

    @property
    def exit_code(self) -> int:
        return {"compat": 0, "needs-build": 1, "incompatible": 2}.get(self.verdict, 2)


def _atlas_root() -> str:
    """The repo root (the directory holding docker-compose.yml). Resolved from
    this file first so commands work from any cwd, then by walking up from the
    cwd; falls back to the cwd."""
    starts = (os.path.dirname(os.path.abspath(__file__)),
              os.path.abspath(os.getcwd()))
    for start in starts:
        cur = start
        while True:
            if os.path.isfile(os.path.join(cur, "docker-compose.yml")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return os.path.abspath(os.getcwd())


def _configured_lens_models_dir(atlas_root: str) -> Optional[str]:
    """Resolve the host Lens artifact override with shell-env precedence."""
    return (os.environ.get("ATLAS_LENS_MODELS")
            or compose_config.read_env_file(atlas_root).get(
                "ATLAS_LENS_MODELS"))


def _check_model(arg: Optional[str], atlas_root: str) -> CheckVerdict:
    """The actual probe + verdict logic. Pure function for testability."""
    probe = probe_llama()
    if not probe.reachable:
        return CheckVerdict(verdict="incompatible", reason=probe.error,
                            probe=probe)
    if probe.embedding_dim == 0:
        return CheckVerdict(verdict="incompatible", reason=probe.error,
                            probe=probe)

    matched = _resolve_model_arg(arg)
    matched_name = matched.name if matched else None
    requested_model = (matched.name if matched else arg) or ""
    if (requested_model and probe.model_name
            and _canonical_model_identity(requested_model)
            != _canonical_model_identity(probe.model_name)):
        return CheckVerdict(
            verdict="incompatible",
            reason=(f"Requested model {requested_model!r}, but llama-server "
                    f"has {probe.model_name!r} loaded. Start the requested "
                    "model before checking or building its Lens artifacts."),
            probe=probe,
            matched_model=matched_name,
        )

    # Resolve artifact dir. For known-supported registry entries this is
    # already wired; for arbitrary models we fall back to ATLAS_LENS_MODELS
    # or the global default. Either way, "is there a cost_field.pt whose
    # input dim matches the model's embedding dim?" is the decisive question.
    if matched and matched.lens_status == "supported":
        artifact_dir = model_registry.lens_artifact_dir_for(matched, atlas_root)
    else:
        env = _configured_lens_models_dir(atlas_root)
        if env:
            artifact_dir = env if os.path.isabs(env) else \
                os.path.normpath(os.path.join(atlas_root, env))
        else:
            artifact_dir = os.path.normpath(os.path.join(
                atlas_root, "geometric-lens", "geometric_lens", "models"))

    inspection = (_inspect_cost_field(artifact_dir) if artifact_dir
                  else ArtifactInspection(present=False))

    if not inspection.present:
        return CheckVerdict(
            verdict="needs-build",
            reason=(f"Model produces {probe.embedding_dim}-dim embeddings, but "
                    f"no cost_field.pt found in {artifact_dir}. Run "
                    f"`atlas lens build` to train fresh artifacts."),
            probe=probe, artifact_dir=artifact_dir,
            artifact_dim=None, matched_model=matched_name,
        )

    if inspection.dim is None and not inspection.torch_available:
        missing = _missing_runtime_artifacts(artifact_dir)
        if missing:
            return CheckVerdict(
                verdict="needs-build",
                reason=("Lens weights are incomplete or uncalibrated; missing "
                        f"{', '.join(missing)}. Run `atlas lens build`."),
                probe=probe, artifact_dir=artifact_dir,
                artifact_dim=None, matched_model=matched_name,
            )
        invalid = _invalid_runtime_artifacts(
            artifact_dir, probe.model_name, probe.embedding_dim)
        if invalid:
            return CheckVerdict(
                verdict="needs-build",
                reason=("Lens calibration is invalid: " + "; ".join(invalid)
                        + ". Run `atlas lens build`."),
                probe=probe, artifact_dir=artifact_dir,
                artifact_dim=None, matched_model=matched_name,
            )
        # Artifact exists but the host Python can't peek at its dim. Don't
        # send the user to needs-build over a tooling gap on the host —
        # the lens service in the container has its own torch and will
        # score fine. Surface the unverified state via the dedicated flag.
        return CheckVerdict(
            verdict="compat", unverified=True,
            reason=(f"cost_field.pt exists at {artifact_dir} but the host "
                    f"Python can't introspect its dim (torch not installed). "
                    f"Assuming compat — the lens service in the container "
                    f"has torch and will score normally. `pip install torch` "
                    f"on the host if you want this to verify properly."),
            probe=probe, artifact_dir=artifact_dir,
            artifact_dim=None, matched_model=matched_name,
        )

    if inspection.dim is None:
        # Torch is available but the load failed for some other reason —
        # corrupted file, unrecognized layout, etc. Treat as needs-build
        # since we can't confirm the artifact is usable.
        return CheckVerdict(
            verdict="needs-build",
            reason=(f"cost_field.pt at {artifact_dir} could not be inspected: "
                    f"{inspection.error}. Rebuild with `atlas lens build`."),
            probe=probe, artifact_dir=artifact_dir,
            artifact_dim=None, matched_model=matched_name,
        )
    artifact_dim = inspection.dim

    if artifact_dim != probe.embedding_dim:
        return CheckVerdict(
            verdict="needs-build",
            reason=(f"Dim mismatch: model emits {probe.embedding_dim}-dim "
                    f"embeddings but the saved cost_field.pt expects "
                    f"{artifact_dim}-dim input. Run `atlas lens build` to "
                    f"train fresh artifacts at the model's native dim."),
            probe=probe, artifact_dir=artifact_dir,
            artifact_dim=artifact_dim, matched_model=matched_name,
        )

    missing = _missing_runtime_artifacts(artifact_dir)
    if missing:
        return CheckVerdict(
            verdict="needs-build",
            reason=("C(x) dimension matches, but calibrated Lens operation "
                    f"requires {', '.join(missing)}. Run `atlas lens build` "
                    "for the selected model."),
            probe=probe, artifact_dir=artifact_dir,
            artifact_dim=artifact_dim, matched_model=matched_name,
        )
    invalid = _invalid_runtime_artifacts(
        artifact_dir, probe.model_name, probe.embedding_dim)
    if invalid:
        return CheckVerdict(
            verdict="needs-build",
            reason=("C(x)/G(x) calibration is invalid: "
                    + "; ".join(invalid)
                    + ". Run `atlas lens build` for the selected model."),
            probe=probe, artifact_dir=artifact_dir,
            artifact_dim=artifact_dim, matched_model=matched_name,
        )

    # Dim matches. PC-202 hidden-states patch is nice-to-have for lens
    # training embeddings but not required for C(x) scoring; report it
    # as a warning surface rather than a hard failure.
    note = ""
    if not probe.has_hidden_states_patch:
        note = (" Note: PC-202 hidden-states patch not detected on llama-server. "
                "C(x) works fine; lens training embeddings (hidden-states "
                "extraction for the C(x)+G(x) retrain) would need a patched "
                "build (inference/Dockerfile.v31).")
    return CheckVerdict(
        verdict="compat",
        reason=(f"Model emits {probe.embedding_dim}-dim embeddings; "
                f"cost_field.pt at {artifact_dir} accepts {artifact_dim}-dim. "
                f"Ready to score.{note}"),
        probe=probe, artifact_dir=artifact_dir,
        artifact_dim=artifact_dim, matched_model=matched_name,
    )


def _emit_check(args: argparse.Namespace, color: bool) -> int:
    atlas_root = _atlas_root()
    v = _check_model(args.model, atlas_root)

    if args.json:
        out = asdict(v)
        out["probe"] = asdict(v.probe)
        out["exit_code"] = v.exit_code
        print(jsonlib.dumps(out, indent=2))
        return v.exit_code

    badge = {
        "compat":       f"{GREEN}compat{RESET}"       if color else "compat",
        "needs-build":  f"{YELL}needs-build{RESET}"   if color else "needs-build",
        "incompatible": f"{RED}incompatible{RESET}"   if color else "incompatible",
    }[v.verdict]
    hdr = f"{BOLD}atlas lens check{RESET}" if color else "atlas lens check"
    _safe_print(f"{hdr}  verdict: {badge}")
    _safe_print(f"  llama-server: {v.probe.url} "
                f"({'reachable' if v.probe.reachable else 'unreachable'})")
    if v.probe.reachable:
        _safe_print(f"  model:        {v.probe.model_name or '(unknown)'}")
        _safe_print(f"  embedding:    {v.probe.embedding_dim}-dim")
        _safe_print(f"  layers:       {v.probe.n_layers or '(unknown)'}")
        _safe_print(f"  PC-202 patch: "
                    f"{'yes' if v.probe.has_hidden_states_patch else 'no'}")
    if v.artifact_dir:
        _safe_print(f"  artifact dir: {v.artifact_dir}")
    if v.artifact_dim is not None:
        _safe_print(f"  artifact dim: {v.artifact_dim}-dim")
    if v.matched_model:
        _safe_print(f"  registry hit: {v.matched_model}")
    _safe_print("")
    _safe_print(f"  {v.reason}")
    if v.verdict != "compat":
        _safe_print("")
        _safe_print("  This is the lens-only view. `atlas doctor` runs the "
                    "full stack diagnosis (service health, auth, disk, "
                    "lens/ASA state) if the fix above isn't enough.")
    return v.exit_code


# ---------------------------------------------------------------------------
# atlas lens build  (PC-058)
# ---------------------------------------------------------------------------

def _embed_text(llama_url: str, text: str,
                _depth: int = 0) -> "tuple[Optional[List[float]], int]":
    """Embed one text via /embedding. Returns (vector, pieces_used).

    llama-server rejects pooled-embedding inputs longer than its micro-batch
    (`-ub`) with an "input is too large" error — common for runaway FAIL
    candidates that hit the generation cap. On failure, split at a line
    boundary and length-weighted-average the halves' embeddings (the lens
    mean-pools per-token states anyway, so chunk-averaging is the same
    operation at coarser granularity). Depth-capped at 4 (16 chunks).
    """
    resp = _http_post_json(f"{llama_url}/embedding",
                           {"content": text}, timeout=60.0)
    vec = None
    if isinstance(resp, list) and resp and isinstance(resp[0], dict):
        raw = resp[0].get("embedding")
        if isinstance(raw, list) and raw:
            if isinstance(raw[0], list):
                # per-token: mean-pool
                n_tok = len(raw)
                dim = len(raw[0])
                vec = [sum(tok[j] for tok in raw) / n_tok
                       for j in range(dim)]
            else:
                vec = raw
    if vec is not None:
        return vec, 1
    if _depth >= 4 or len(text) < 2:
        return None, 0
    mid = text.rfind("\n", 0, len(text) // 2 + 1)
    if mid <= 0:
        mid = len(text) // 2
    left, right = text[:mid], text[mid:]
    if not left.strip() or not right.strip():
        left, right = text[:len(text) // 2], text[len(text) // 2:]
    va, pa = _embed_text(llama_url, left, _depth + 1)
    vb, pb = _embed_text(llama_url, right, _depth + 1)
    if va is None and vb is None:
        return None, 0
    if va is None or vb is None:
        # Salvage the half that embedded — for the runaway/repetitive texts
        # that hit this path, any chunk is representative.
        return (va or vb), (pa or pb)
    wa, wb = len(left), len(right)
    return ([(a * wa + b * wb) / (wa + wb) for a, b in zip(va, vb)],
            pa + pb)


def _extract_training_embeddings(samples: List[Dict],
                                  llama_url: str,
                                  color: bool,
                                  cache_path: Optional[str] = None,
                                  expect_dim: int = 0) -> Dict:
    """For each sample {text, label}, POST /embedding and collect.

    Embeddings are cached on disk keyed by text hash (extraction dominates
    build wall-clock — ~15 min per few hundred samples — and retrains on a
    grown results dir re-embed mostly the same texts). Entries whose dim
    doesn't match the probed model are ignored, so a model switch
    invalidates the cache naturally.

    Returns a dict compatible with training.train_cost_field's `data` arg:
        {"embeddings": [List[float], ...], "labels": [0|1, ...]}
    """
    import hashlib

    cache: Dict[str, List[float]] = {}
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        entry = jsonlib.loads(line)
                        if not expect_dim or entry.get("dim") == expect_dim:
                            cache[entry["h"]] = entry["v"]
                    except (jsonlib.JSONDecodeError, KeyError, TypeError):
                        continue
        except OSError:
            # A corrupt or unreadable cache only forfeits reuse; embeddings
            # are recomputed and a fresh cache may be appended below.
            pass

    cache_fh = None
    if cache_path:
        try:
            cache_fh = open(cache_path, "a", encoding="utf-8")
        except OSError:
            cache_fh = None

    embeddings: List[List[float]] = []
    labels: List[int] = []
    weights: List[float] = []  # carried through aligned with kept samples
    saw_weight = False
    n = len(samples)
    hits = 0
    try:
        for i, s in enumerate(samples):
            text = s.get("text") or s.get("content") or ""
            label = int(s.get("label", 0))
            if not text:
                continue
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            vec = cache.get(h)
            if vec is not None:
                hits += 1
            else:
                vec, pieces = _embed_text(llama_url, text)
                if vec is None:
                    _safe_print(f"  WARN: sample {i+1}/{n}: embedding failed "
                                f"even after chunking — skipped")
                    continue
                if pieces > 1:
                    _safe_print(f"  sample {i+1}/{n}: longer than the "
                                f"server's micro-batch — embedded as "
                                f"{pieces} chunks, mean-pooled")
                cache[h] = vec   # duplicate texts later in this run hit
                if cache_fh:
                    cache_fh.write(jsonlib.dumps(
                        {"h": h, "dim": len(vec), "v": vec}) + "\n")
                    cache_fh.flush()
            embeddings.append(vec)
            labels.append(label)
            if "weight" in s:
                saw_weight = True
            weights.append(float(s.get("weight", 1.0)))
            if (i + 1) % 25 == 0 or (i + 1) == n:
                _safe_print(f"  extracted {i+1}/{n} embeddings")
    finally:
        if cache_fh:
            cache_fh.close()
    if hits:
        _safe_print(f"  ({hits} from cache, {len(embeddings) - hits} embedded "
                    f"fresh)")
    out = {"embeddings": embeddings, "labels": labels}
    # Only attach weights when the samples actually carried them (collected
    # corpus); bench/`--samples` builds stay uniformly weighted as before.
    if saw_weight:
        out["weights"] = weights
    return out


def _load_telemetry_embeddings(emb_path: str,
                                expect_dim: int) -> "tuple[List[Dict], int]":
    """Load labeled embeddings banked by v3_runner during a bench run.

    Every sandbox-tested candidate (probe + PlanSearch fan-out + repair
    iterations) gets its embedding + PASS/FAIL label appended to
    telemetry/embeddings.emb as the bench runs — in V3 mode that is several
    labeled samples per task, not one. Returns ({"embedding", "label"}
    dicts, n_skipped) with UNKNOWN labels and dim mismatches dropped.
    """
    root = _atlas_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from benchmark.v3.embedding_store import EmbeddingReader

    out: List[Dict] = []
    skipped = 0
    for rec in EmbeddingReader(emb_path).read_all():
        label = {"PASS": 1, "FAIL": 0}.get(rec.get("label"))
        emb = rec.get("embedding")
        if label is None or not emb or (expect_dim
                                        and len(emb) != expect_dim):
            skipped += 1
            continue
        out.append({"embedding": emb, "label": label})
    return out, skipped


def _load_training_samples(path: Optional[str]) -> List[Dict]:
    """Load training samples from a JSON or JSONL file.

    Format: list of {"text": str, "label": 0|1} or {"content": str, "label": 0|1}.
    JSONL is detected by .jsonl extension or by leading whitespace check.
    """
    if not path or not os.path.isfile(path):
        return []
    with open(path) as fh:
        content = fh.read()
    if path.endswith(".jsonl") or content.lstrip().startswith("{\""):
        # JSONL
        samples = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(jsonlib.loads(line))
            except jsonlib.JSONDecodeError:
                continue
        return samples
    # JSON array
    try:
        parsed = jsonlib.loads(content)
        if isinstance(parsed, list):
            return parsed
    except jsonlib.JSONDecodeError:
        return []
    return []


_MANAGED_LENS_ARTIFACTS = (
    "cost_field.pt",
    "cost_field.safetensors",
    "cx_normalization.json",
    "gx_xgboost.json",
    "gx_weights.json",
    "gx_thresholds.json",
    "model_identity.json",
    # Superseded formats must not shadow a freshly trained XGBoost bundle.
    "metric_tensor.pt",
    "gx_xgboost.pkl",
)


def _activate_lens_bundle(staging_dir: str, artifact_dir: str) -> None:
    """Activate one complete Lens bundle, restoring the prior one on error."""
    missing = _missing_runtime_artifacts(staging_dir)
    if not os.path.isfile(os.path.join(staging_dir, "cost_field.pt")):
        missing.insert(0, "cost_field.pt")
    if missing:
        raise ValueError("staged Lens bundle is incomplete: "
                         + ", ".join(missing))
    invalid = _invalid_runtime_artifacts(staging_dir)
    if invalid:
        raise ValueError("staged Lens bundle is invalid: " + "; ".join(invalid))

    artifact_dir = os.path.abspath(artifact_dir)
    os.makedirs(artifact_dir, exist_ok=True)
    parent = os.path.dirname(artifact_dir)
    backup_dir = tempfile.mkdtemp(prefix=".atlas-lens-backup-", dir=parent)
    moved_new = []
    try:
        for filename in _MANAGED_LENS_ARTIFACTS:
            current = os.path.join(artifact_dir, filename)
            if os.path.exists(current):
                os.replace(current, os.path.join(backup_dir, filename))

        # Identity is the commit marker and moves last.
        staged_files = [name for name in _MANAGED_LENS_ARTIFACTS
                        if name != "model_identity.json"
                        and os.path.isfile(os.path.join(staging_dir, name))]
        staged_files.append("model_identity.json")
        for filename in staged_files:
            source = os.path.join(staging_dir, filename)
            target = os.path.join(artifact_dir, filename)
            os.replace(source, target)
            moved_new.append(target)
    except Exception:
        for path in moved_new:
            try:
                os.remove(path)
            except OSError:
                # Rollback cleanup is best-effort; the original artifacts are
                # restored from backup immediately afterward.
                pass
        for filename in os.listdir(backup_dir):
            os.replace(os.path.join(backup_dir, filename),
                       os.path.join(artifact_dir, filename))
        raise
    finally:
        import shutil
        shutil.rmtree(backup_dir, ignore_errors=True)


def _load_results_samples(results_dir: str) -> List[Dict]:
    """Load training samples from a benchmark results directory — the per-task
    JSONs written by `atlas bench` (each has `code` + `passed`). Maps to the
    same {text, label} shape `_extract_training_embeddings` consumes, so the
    model trains C(x) on its own pass/fail candidates."""
    import glob
    samples: List[Dict] = []
    for fp in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(fp) as fh:
                d = jsonlib.load(fh)
        except (OSError, jsonlib.JSONDecodeError):
            continue
        if not isinstance(d, dict):
            continue  # stray non-result JSON (array/scalar) in the dir
        code = d.get("code")
        if not code:
            continue
        samples.append({"text": code, "label": 1 if d.get("passed") else 0})
    return samples


def _collected_corpus_dir() -> str:
    """Host directory holding the lens-training corpus collected during agent
    use (per-file accept/deny + pass thumbs → labeled, weighted samples). This
    is the host side of the proxy's lens-training bind mount. ATLAS_LENS_HOST_DIR
    overrides; default <atlas_root>/lens_training."""
    env = os.environ.get("ATLAS_LENS_HOST_DIR")
    if env:
        return env
    return os.path.join(_atlas_root(), "lens_training")


def _sanitize_model_dir(name: str) -> str:
    """Mirror proxy/lens_samples.go:sanitizeModelName so the CLI finds the
    subdir the proxy wrote to."""
    if not name:
        return "default"
    out = []
    for ch in name:
        out.append("_" if ch in "/\\: " else ch)
    return "".join(out)


def _load_collected_samples(model_name: Optional[str]) -> List[Dict]:
    """Load the collected corpus for a model as [{text, label, weight}].

    Resolves <corpus>/<sanitized-model>/samples.jsonl. When that subdir is
    absent but exactly one model subdir exists, uses it (so the user doesn't
    have to name the model when there's only one). Returns [] if nothing found.
    """
    root = _collected_corpus_dir()
    if not os.path.isdir(root):
        return []
    sub = _sanitize_model_dir(model_name or os.environ.get("ATLAS_MODEL_NAME", ""))
    path = os.path.join(root, sub, "samples.jsonl")
    if not os.path.isfile(path):
        subdirs = [d for d in os.listdir(root)
                   if os.path.isfile(os.path.join(root, d, "samples.jsonl"))]
        if len(subdirs) == 1:
            path = os.path.join(root, subdirs[0], "samples.jsonl")
        else:
            return []
    samples: List[Dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = jsonlib.loads(line)
            except jsonlib.JSONDecodeError:
                continue
            text = d.get("content") or d.get("text")
            if not text:
                continue
            samples.append({"text": text, "label": int(d.get("label", 0)),
                            "weight": float(d.get("weight", 1.0))})
    return samples


def _emit_build(args: argparse.Namespace, color: bool) -> int:
    """Train fresh Lens artifacts for the model llama-server has loaded.

    Doesn't ship its own dataset — users point --samples at a labeled
    JSON/JSONL file (typically pulled from huggingface.co/datasets/itigges22/ATLAS,
    which has the V3 ablation traces with pass/fail labels). Tiny built-in
    sanity datasets are intentionally NOT bundled: a 20-sample C(x) is
    worse than no C(x) (it'll badly mis-rank and the user won't know).
    """
    atlas_root = _atlas_root()

    # 1. Pre-flight: confirm we can probe the model. Reuses the check path
    # so build's UX agrees with check's "this is/isn't compat" verdict.
    _safe_print("[1/5] Probing llama-server…")
    verdict = _check_model(args.model, atlas_root)
    if verdict.verdict == "incompatible":
        _safe_print(f"  {RED if color else ''}Cannot proceed: "
                    f"{verdict.reason}{RESET if color else ''}")
        return 2
    _safe_print(f"  Model emits {verdict.probe.embedding_dim}-dim embeddings "
                f"(model: {verdict.probe.model_name or 'unknown'})")
    if verdict.verdict == "compat" and not args.force:
        _safe_print(f"  {YELL if color else ''}Artifacts already exist at "
                    f"{verdict.artifact_dir} for the current dim. Pass "
                    f"--force to retrain anyway.{RESET if color else ''}")
        return 0

    # 2. Load training data
    _safe_print("[2/5] Loading training samples…")
    if getattr(args, "from_results", None) and args.samples:
        _safe_print(f"  {RED if color else ''}--from-results and --samples are "
                    f"mutually exclusive — pass one.{RESET if color else ''}")
        return 1
    if getattr(args, "from_results", None):
        results_dir = args.from_results
        # A relative path that doesn't exist under the cwd is resolved against
        # the repo root, so the command works from any directory.
        if not os.path.isdir(results_dir):
            rooted = os.path.join(atlas_root, results_dir)
            if os.path.isdir(rooted):
                results_dir = rooted
        samples = _load_results_samples(results_dir)
        if not samples:
            _safe_print(f"  {RED if color else ''}No usable samples in "
                        f"{results_dir} — expected per-task JSONs with "
                        f"`code` + `passed` (from `atlas bench`)."
                        f"{RESET if color else ''}")
            # Common slip: pointing at the run root instead of the per-task dir.
            for sub in ("v3_lcb/per_task", "per_task"):
                cand = os.path.join(results_dir, sub)
                if os.path.isdir(cand):
                    _safe_print(f"  Did you mean: --from-results {cand}")
                    break
            return 1
    elif getattr(args, "from_collected", False):
        samples = _load_collected_samples(args.model)
        if not samples:
            _safe_print(f"  {RED if color else ''}No collected samples found in "
                        f"{_collected_corpus_dir()} for this model. Rate some "
                        f"passes (👍/👎 + per-file accept/deny) in the TUI to "
                        f"build a corpus first.{RESET if color else ''}")
            return 1
    elif args.samples:
        samples = _load_training_samples(args.samples)
    else:
        _safe_print(f"  {RED if color else ''}Provide --from-results <dir> or "
                    f"--samples <file>.{RESET if color else ''}")
        _safe_print("  --from-results: a benchmark/results/<run>/v3_lcb/per_task "
                    "dir (this model's own candidates, via `atlas bench`)")
        _safe_print("  --samples: a labeled [{\"text\": str, \"label\": 0|1}, ...] "
                    "file (e.g. huggingface.co/datasets/itigges22/ATLAS)")
        return 1
    # Coerce labels to int once, here; rows with malformed labels are dropped
    # so the counting and embedding-extraction below can rely on clean ints.
    cleaned = []
    n_bad = 0
    for s in samples:
        try:
            s["label"] = int(s.get("label", 0))
            cleaned.append(s)
        except (TypeError, ValueError):
            n_bad += 1
    if n_bad:
        _safe_print(f"  WARN: skipped {n_bad} sample(s) with non-integer labels")
    samples = cleaned
    if len(samples) < 50:
        _safe_print(f"  {RED if color else ''}Only {len(samples)} samples "
                    f"loaded. Need >=50 for meaningful training (>=200 "
                    f"recommended).{RESET if color else ''}")
        return 1
    n_pass = sum(1 for s in samples if int(s.get("label", 0)) == 1)
    n_fail = len(samples) - n_pass
    _safe_print(f"  Loaded {len(samples)} samples (PASS={n_pass}, FAIL={n_fail})")
    if n_pass == 0 or n_fail == 0:
        _safe_print(f"  {RED if color else ''}Need both pass and fail "
                    f"samples for contrastive training.{RESET if color else ''}")
        return 1

    # 3. Extract embeddings via /embedding
    _safe_print(f"[3/5] Extracting embeddings via {verdict.probe.url}…")
    if getattr(args, "from_results", None):
        cache_path = os.path.normpath(
            os.path.join(results_dir, os.pardir, "embeddings_cache.jsonl"))
    elif getattr(args, "from_collected", False):
        cache_path = os.path.join(_collected_corpus_dir(), "embeddings_cache.jsonl")
    elif args.samples:
        cache_path = args.samples + ".embcache.jsonl"
    else:
        cache_path = None
    start = time.time()
    data = _extract_training_embeddings(
        samples, verdict.probe.url, color, cache_path=cache_path,
        expect_dim=verdict.probe.embedding_dim or 0)
    elapsed = time.time() - start
    if not data["embeddings"]:
        _safe_print(f"  {RED if color else ''}No embeddings extracted. "
                    f"Check llama-server logs.{RESET if color else ''}")
        return 1
    _safe_print(f"  Extracted {len(data['embeddings'])} embeddings "
                f"in {elapsed:.1f}s")

    # Merge the bench run's banked per-candidate embeddings (PASS/FAIL
    # labeled by the sandbox). A V3-mode bench tests several candidates per
    # task and banks each one, so this multiplies the training set without
    # any extra labeling or embedding work. Near-identical vectors (the
    # selected candidate appears in both sources) are deduped.
    if getattr(args, "from_results", None) and not getattr(
            args, "no_telemetry", False):
        emb_path = os.path.normpath(os.path.join(
            results_dir, os.pardir, os.pardir, "telemetry", "embeddings.emb"))
        if os.path.isfile(emb_path):
            try:
                import numpy as np
                dim = len(data["embeddings"][0])
                tele, skipped = _load_telemetry_embeddings(emb_path, dim)
                # The selected candidate appears in both sources, but the
                # banked copy was embedded mid-bench under concurrent
                # batching — same text, FP noise up to ~5e-5 per element.
                # Dedup numerically: a telemetry vector within 1e-3 (L-inf)
                # of any kept vector is the same sample; genuinely
                # different candidates differ by orders of magnitude more.
                kept = np.array(data["embeddings"], dtype=np.float32)
                merged = dups = 0
                for rec in tele:
                    v = np.array(rec["embedding"], dtype=np.float32)
                    if np.abs(kept - v).max(axis=1).min() < 1e-3:
                        dups += 1
                        continue
                    kept = np.vstack([kept, v])
                    data["embeddings"].append(rec["embedding"])
                    data["labels"].append(rec["label"])
                    merged += 1
                _safe_print(f"  Merged {merged} banked candidate embeddings "
                            f"from {os.path.basename(emb_path)} "
                            f"({dups} duplicates of extracted samples"
                            + (f", {skipped} skipped" if skipped else "")
                            + ") — pass --no-telemetry to train on the "
                            "results dir alone")
                n_p = sum(data["labels"])
                _safe_print(f"  Training set: {len(data['labels'])} samples "
                            f"(PASS={n_p}, FAIL={len(data['labels']) - n_p})")
            except Exception as e:
                _safe_print(f"  (telemetry merge unavailable: {e})")

    if args.dry_run:
        _safe_print("  (dry-run) skipping training + save")
        return 0

    # 4. Train + save
    _safe_print(f"[4/5] Training CostField "
                f"({args.epochs} epochs, lr={args.lr})…")
    try:
        # The training module lives in the repo, not on the host's default
        # path — resolve it from the install root so this works from any cwd.
        gl_dir = os.path.join(atlas_root, "geometric-lens")
        if gl_dir not in sys.path:
            sys.path.insert(0, gl_dir)
        from geometric_lens.training import train_cost_field, save_cost_field
        from geometric_lens.calibration import derive_cx_normalization
    except ImportError as e:
        _safe_print(f"  {RED if color else ''}Could not import training "
                    f"module: {e}.{RESET if color else ''}")
        if "torch" in str(e):
            _safe_print("  Training runs on the host and needs the train "
                        "extra (or a CPU torch, which is enough here):")
            _safe_print("    pip install 'atlas[train]'   # or:")
            _safe_print("    pip install torch --index-url "
                        "https://download.pytorch.org/whl/cpu")
        else:
            _safe_print("  The atlas CLI must point at an ATLAS checkout "
                        "containing geometric-lens/ (check `pip show atlas` "
                        "→ Editable project location).")
        return 1
    result = train_cost_field(data, epochs=args.epochs, lr=args.lr,
                              margin=args.margin)
    # train_cost_field returns final_* and best_* keys, not bare test_auc.
    # Surface "best test AUC seen during training" since the final-epoch
    # value can be lower from overfitting.
    test_auc = (result.get("best_test_auc")
                or result.get("final_test_auc") or 0.0)
    train_auc = result.get("final_train_auc") or 0.0
    _safe_print(f"  Train AUC: {train_auc:.4f}  |  Test AUC: {test_auc:.4f} "
                f"(best across epochs)")
    if test_auc < 0.7:
        if train_auc >= 0.95:
            # Memorized the train fold but doesn't generalize: the sample
            # set is too small for the capacity. Epochs won't help.
            _safe_print(f"  {YELL if color else ''}Test AUC < 0.70 with "
                        f"train AUC {train_auc:.2f} — overfit on too few "
                        f"samples. More training data is the fix (e.g. a "
                        f"larger `atlas bench --tasks N` run); more epochs "
                        f"won't help.{RESET if color else ''}")
        else:
            _safe_print(f"  {YELL if color else ''}Test AUC < 0.70 with "
                        f"train AUC {train_auc:.2f} — undertrained. Try "
                        f"more epochs (--epochs 400) or a higher learning "
                        f"rate.{RESET if color else ''}")

    artifact_dir = args.artifact_dir or verdict.artifact_dir
    artifact_parent = os.path.dirname(os.path.abspath(artifact_dir))
    os.makedirs(artifact_parent, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=".atlas-lens-build-", dir=artifact_parent) as staging_dir:
        normalization = derive_cx_normalization(
            result["pass_energy_mean"], result["fail_energy_mean"])
        save_cost_field(result["model"], save_dir=staging_dir,
                        normalization=normalization)

        # 5. Train + save G(x) on the same embeddings. Nothing reaches the
        # live directory unless both halves and their metadata are complete.
        _safe_print("[5/5] Training G(x) XGBoost…")
        try:
            from geometric_lens.training import train_gx, save_gx
            gx_result = train_gx(data)
        except ImportError as e:
            _safe_print(f"  {RED if color else ''}Could not import the G(x) "
                        f"trainer: {e}.{RESET if color else ''}")
            _safe_print("  G(x) training needs XGBoost + scikit-learn on the "
                        "host: pip install 'atlas[train]' (or pip install "
                        "xgboost scikit-learn)")
            _safe_print("  The previous live bundle is unchanged; embeddings "
                        "are cached for a quick retry.")
            return 1
        except ValueError as e:
            _safe_print(f"  {RED if color else ''}G(x) training skipped: "
                        f"{e}{RESET if color else ''}")
            _safe_print("  The previous live bundle is unchanged. Add more "
                        "bench samples and re-run.")
            return 1
        save_gx(gx_result, save_dir=staging_dir)
        from geometric_lens.identity import save_model_identity
        model_identity = (args.model or verdict.probe.model_name or "").strip()
        if not model_identity:
            _safe_print(f"  {RED if color else ''}Could not identify the loaded "
                        f"model; refusing to activate these artifacts."
                        f"{RESET if color else ''}")
            return 1
        save_model_identity(
            staging_dir, model_identity, verdict.probe.embedding_dim)
        try:
            _activate_lens_bundle(staging_dir, artifact_dir)
        except (OSError, ValueError) as e:
            _safe_print(f"  {RED if color else ''}Could not activate the "
                        f"complete Lens bundle: {e}. The previous bundle was "
                        f"restored.{RESET if color else ''}")
            return 1

    _safe_print(f"  Activated: {artifact_dir}")
    _safe_print(f"  G(x) CV AUC: {gx_result['cv_auc_mean']:.4f}")

    _safe_print("")
    _safe_print(f"  {GREEN if color else ''}Build complete "
                f"(C(x) + G(x)).{RESET if color else ''}")
    _safe_print("  The lens service loads artifacts at startup — restart it "
                "to pick these up:")
    _safe_print(f"    {BOLD if color else ''}docker compose restart "
                f"geometric-lens{RESET if color else ''}")
    _safe_print("  then `atlas lens check` should report compat.")
    _safe_print(f"  Next: `atlas lens publish` to share these artifacts and "
                f"generate a registry-PR (PC-059), or manually update the "
                f"registry entry for "
                f"{verdict.matched_model or '<your model>'} to "
                f"lens_status=\"supported\".")
    return 0


# ---------------------------------------------------------------------------
# atlas lens publish  (PC-059)
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    """Stream-hash a file (large .pt artifacts shouldn't blow memory)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_token() -> Optional[str]:
    """Resolve the HF token from the standard places huggingface_hub looks."""
    return (os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


_UPSTREAM_REPO = "itigges22/ATLAS"
_REGISTRY_PATH = "atlas/cli/commands/model_registry.py"


def _gh_api(api_args: List[str], payload: Optional[dict] = None):
    """Run `gh api …`, return (ok, parsed-json-or-text, stderr)."""
    import subprocess
    cmd = ["gh", "api"] + api_args
    stdin = None
    if payload is not None:
        cmd += ["--input", "-"]
        stdin = jsonlib.dumps(payload)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           input=stdin, timeout=60)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, None, str(e)
    if r.returncode != 0:
        return False, None, r.stderr.strip()[:300]
    try:
        return True, jsonlib.loads(r.stdout), ""
    except jsonlib.JSONDecodeError:
        return True, r.stdout.strip(), ""


def _render_registry_entry(model_label: str, model_file: str,
                           size_gb: float, tier: str, dim: int,
                           hf_repo: str, license_id: str,
                           artifact_files: List[str]) -> str:
    """A complete, committable Model(...) entry for an unregistered model."""
    files = ", ".join(f'"{f}"' for f in artifact_files)
    dim_label = str(dim) if dim else "unknown"
    return f'''    Model(
        name="{model_label}",
        tier="{tier}",
        model_file="{model_file}",
        model_display="{model_label}",
        model_size_gb={size_gb},
        lens_status="supported",
        lens_calibrated=True,
        download_url=None,
        sha256=None,
        license="{license_id}",
        lens_artifact_files=[{files}],
        lens_hf_repo="{hf_repo}",
        notes="Added via `atlas lens publish` — lens artifacts "
              "({dim_label}-dim) at https://huggingface.co/{hf_repo}. "
              "download_url not captured at publish time; maintainers "
              "can fill it in for `atlas model install` support.",
    ),
'''


def _registry_insert_entry(content: str, model_label: str,
                           entry: str) -> Optional[str]:
    """Insert a Model entry before the REGISTRY list's closing bracket.
    Returns the new content, or None when insertion isn't safe (model
    already present, or the anchor isn't found).

    The upstream registry can be older than the publisher's install —
    kwargs the upstream Model dataclass doesn't declare yet are stripped
    from the entry so the committed file stays importable.
    """
    if f'name="{model_label}"' in content:
        return None
    anchor = "REGISTRY: List[Model] = ["
    start = content.find(anchor)
    if start < 0:
        return None
    close = content.find("\n]", start)
    if close < 0:
        return None
    schema = content[:start]   # the Model dataclass definition lives above
    for field_name in ("lens_calibrated", "lens_hf_repo", "asa_hf_repo"):
        if f"{field_name}:" not in schema:
            entry = "\n".join(l for l in entry.splitlines()
                              if not l.strip().startswith(f"{field_name}="))
            if not entry.endswith("\n"):
                entry += "\n"
    return content[:close + 1] + entry + content[close + 1:]


def _registry_set_lens(content: str, model_label: str, hf_repo: str,
                       artifact_files: List[str]) -> Optional[str]:
    """Mark an existing registry entry's Lens bundle as current.

    Publishing is also the upgrade path for legacy or unverified entries, so
    it must update those entries rather than only knowing how to insert a new
    model. The upstream schema may lag behind the publisher; in that case the
    download location is retained as a comment until the field lands.
    """
    import re
    match = re.search(
        rf'(    Model\(\s*\n\s*name="{re.escape(model_label)}".*?\n    \),)',
        content,
        re.DOTALL,
    )
    if not match:
        return None

    block = match.group(1)
    files = ", ".join(f'"{name}"' for name in artifact_files)
    lens_lines = (
        '        lens_status="supported",\n'
        f'        lens_artifact_files=[{files}],\n'
    )
    schema = content[:match.start()]
    if "lens_calibrated:" in schema:
        lens_lines += '        lens_calibrated=True,\n'
    else:
        lens_lines += '        # Bundle includes current C(x)/G(x) calibration.\n'
    if "lens_hf_repo:" in schema:
        lens_lines += f'        lens_hf_repo="{hf_repo}",\n'
    else:
        lens_lines = (
            f'        # Lens artifacts: https://huggingface.co/{hf_repo}\n'
            '        # (promote to lens_hf_repo= once the registry schema carries it)\n'
        ) + lens_lines

    new_block = re.sub(r'\n\s*lens_status="[^"]*",', "", block)
    new_block = re.sub(r'\n\s*lens_calibrated=(?:True|False),', "", new_block)
    new_block = re.sub(r'\n\s*lens_artifact_files=\[[^\]]*\],', "", new_block)
    new_block = re.sub(r'\n\s*lens_hf_repo="[^"]*",', "", new_block)
    new_block = new_block.replace("\n    ),", "\n" + lens_lines + "    ),")
    return content.replace(block, new_block)


def _registry_set_asa(content: str, model_label: str, hf_repo: str,
                      artifact_files: List[str]) -> Optional[str]:
    """Within the named entry's block, set asa_status to supported and
    record the vector's HF repo + files. Returns new content or None when
    the entry (or a safe edit point) can't be found."""
    import re
    m = re.search(rf'(    Model\(\s*\n\s*name="{re.escape(model_label)}".*?'
                  rf'\n    \),)', content, re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    files = ", ".join(f'"{f}"' for f in artifact_files)
    asa_lines = (f'        asa_status="supported",\n'
                 f'        asa_artifact_files=[{files}],\n')
    # Only set fields the upstream dataclass declares (it can be older
    # than the publisher's install) — but never drop the download
    # location: without `asa_hf_repo` in the schema, record the vector's
    # HF repo as a comment so the entry still points at it (promotable
    # to the field once the newer schema lands).
    if "asa_hf_repo:" in content[:m.start()]:
        asa_lines += f'        asa_hf_repo="{hf_repo}",\n'
    else:
        asa_lines = (f'        # ASA vector: https://huggingface.co/'
                     f'{hf_repo}\n'
                     f'        # (promote to asa_hf_repo= once the '
                     f'registry schema carries it)\n') + asa_lines
    new_block = re.sub(r'\n\s*asa_status="[^"]*",', "", block)
    new_block = re.sub(r'\n\s*asa_artifact_files=\[[^\]]*\],', "", new_block)
    new_block = re.sub(r'\n\s*asa_hf_repo="[^"]*",', "", new_block)
    new_block = new_block.replace("\n    ),", "\n" + asa_lines + "    ),")
    return content.replace(block, new_block)


def open_registry_pr_via_api(model_label: str, title: str, body: str,
                             edit_fn, color: bool) -> Optional[str]:
    """Open a registry PR through the GitHub API — no local git checkout.

    edit_fn(content) -> new content or None. Flow: resolve user + base
    branch (prefers `dev`), fork if the user can't push upstream, branch,
    commit the edited registry file, open the PR against the upstream.
    Returns the PR URL, or None on any failure (caller falls back to
    printing the body).
    """
    import base64

    upstream = os.environ.get("ATLAS_UPSTREAM_REPO", _UPSTREAM_REPO)
    owner = upstream.split("/")[0]

    ok, login, err = _gh_api(["user", "-q", ".login"])
    if not ok:
        _safe_print(f"  gh api user failed: {err}")
        return None
    login = str(login).strip()

    ok, _, _ = _gh_api([f"repos/{upstream}/branches/dev"])
    base = "dev" if ok else None
    if base is None:
        ok, repo_info, err = _gh_api([f"repos/{upstream}"])
        if not ok:
            _safe_print(f"  gh api repos/{upstream} failed: {err}")
            return None
        base = repo_info.get("default_branch", "main")

    if login == owner:
        target = upstream
    else:
        ok, fork, err = _gh_api([f"repos/{upstream}/forks", "-X", "POST"])
        if not ok:
            _safe_print(f"  fork failed: {err}")
            return None
        target = fork.get("full_name", f"{login}/{upstream.split('/')[1]}")
        # Best-effort: bring the fork's base branch up to date.
        _gh_api([f"repos/{target}/merge-upstream", "-X", "POST"],
                payload={"branch": base})

    ok, br, err = _gh_api([f"repos/{target}/branches/{base}"])
    if not ok:
        _safe_print(f"  branch {base} not found on {target}: {err}")
        return None
    head_sha = br["commit"]["sha"]

    ok, blob, err = _gh_api(
        [f"repos/{target}/contents/{_REGISTRY_PATH}?ref={base}"])
    if not ok:
        _safe_print(f"  fetch registry file failed: {err}")
        return None
    content = base64.b64decode(blob["content"]).decode("utf-8")

    new_content = edit_fn(content)
    if new_content is None:
        # The edit doesn't apply to the base branch. Common cause: a prior
        # `atlas * publish` PR holding this model's entry is still open —
        # stack this edit onto that PR's branch instead of failing.
        slug = "".join(c if c.isalnum() else "-" for c in model_label.lower())
        ok, prs, _ = _gh_api(
            [f"repos/{upstream}/pulls?state=open&base={base}"])
        for pr in (prs if ok and isinstance(prs, list) else []):
            head_ref = pr.get("head", {}).get("ref", "")
            if not (head_ref.startswith("atlas-publish/")
                    and slug in head_ref):
                continue
            head_repo = pr.get("head", {}).get("repo", {}).get("full_name")
            if not head_repo:
                continue
            ok, pr_blob, err = _gh_api(
                [f"repos/{head_repo}/contents/{_REGISTRY_PATH}"
                 f"?ref={head_ref}"])
            if not ok:
                continue
            pr_content = base64.b64decode(pr_blob["content"]).decode("utf-8")
            stacked = edit_fn(pr_content)
            if stacked is None:
                continue
            pr_num = pr.get("number")
            if head_repo == upstream:
                # The pending branch lives upstream — open a SEPARATE PR
                # based on it. Its diff shows only this edit; when the
                # earlier PR merges (delete its branch), GitHub retargets
                # this one to the base branch automatically.
                ok, head_br, err = _gh_api(
                    [f"repos/{upstream}/branches/{head_ref}"])
                if not ok:
                    continue
                new_branch = f"{head_ref}-next-{int(time.time())}"
                ok, _, err = _gh_api(
                    [f"repos/{upstream}/git/refs", "-X", "POST"],
                    payload={"ref": f"refs/heads/{new_branch}",
                             "sha": head_br["commit"]["sha"]})
                if not ok:
                    _safe_print(f"  branch off PR #{pr_num} failed: {err}")
                    continue
                ok, _, err = _gh_api(
                    [f"repos/{upstream}/contents/{_REGISTRY_PATH}",
                     "-X", "PUT"],
                    payload={"message": title,
                             "content": base64.b64encode(
                                 stacked.encode("utf-8")).decode("ascii"),
                             "sha": pr_blob["sha"],
                             "branch": new_branch})
                if not ok:
                    _safe_print(f"  commit failed: {err}")
                    continue
                note = (f"\n\n---\n*Stacked on #{pr_num} (this model's "
                        f"entry lands there). Merge #{pr_num} with branch "
                        f"deletion and GitHub retargets this PR to `{base}` "
                        f"automatically — its diff shows only this change.*")
                ok, pr2, err = _gh_api(
                    [f"repos/{upstream}/pulls", "-X", "POST"],
                    payload={"title": title, "body": body + note,
                             "head": new_branch, "base": head_ref})
                if not ok:
                    _safe_print(f"  open stacked PR failed: {err}")
                    continue
                _safe_print(f"  Entry is pending in open PR #{pr_num} — "
                            f"opened a separate PR stacked on it (diff "
                            f"shows only this change).")
                return pr2.get("html_url")
            # Fork case: a PR's base must be an upstream branch, so a
            # separate stacked PR isn't expressible — commit onto the
            # pending PR's branch instead (one PR, both changes).
            ok, _, err = _gh_api(
                [f"repos/{head_repo}/contents/{_REGISTRY_PATH}", "-X", "PUT"],
                payload={"message": title,
                         "content": base64.b64encode(
                             stacked.encode("utf-8")).decode("ascii"),
                         "sha": pr_blob["sha"],
                         "branch": head_ref})
            if not ok:
                _safe_print(f"  stacking onto PR #{pr_num} failed: {err}")
                continue
            _safe_print(f"  Entry is pending in open PR #{pr_num} — this "
                        f"edit was committed onto its branch (forks can't "
                        f"host a separate stacked PR; one PR, both "
                        f"changes).")
            return pr.get("html_url")
        _safe_print("  registry edit not applicable (entry state differs "
                    "upstream, and no open publish PR holds it) — falling "
                    "back to the printed body.")
        return None

    slug = "".join(c if c.isalnum() else "-" for c in model_label.lower())
    branch = f"atlas-publish/{slug}-{int(time.time())}"
    ok, _, err = _gh_api([f"repos/{target}/git/refs", "-X", "POST"],
                         payload={"ref": f"refs/heads/{branch}",
                                  "sha": head_sha})
    if not ok:
        _safe_print(f"  create branch failed: {err}")
        return None

    ok, _, err = _gh_api(
        [f"repos/{target}/contents/{_REGISTRY_PATH}", "-X", "PUT"],
        payload={"message": title,
                 "content": base64.b64encode(
                     new_content.encode("utf-8")).decode("ascii"),
                 "sha": blob["sha"],
                 "branch": branch})
    if not ok:
        _safe_print(f"  commit failed: {err}")
        return None

    head = branch if target == upstream else f"{login}:{branch}"
    ok, pr, err = _gh_api([f"repos/{upstream}/pulls", "-X", "POST"],
                          payload={"title": title, "body": body,
                                   "head": head, "base": base})
    if not ok:
        _safe_print(f"  open PR failed: {err}")
        return None
    return pr.get("html_url")


def _gh_available() -> bool:
    """Best-effort probe for the gh CLI on PATH. Used by the publish
    pre-flight so we can tell the user up front whether the registry PR
    will auto-open or whether they'll need to paste the body manually."""
    import shutil
    return shutil.which("gh") is not None


def _huggingface_hub_available() -> bool:
    """Probe-only check that huggingface_hub imports. Lazy and quiet —
    we don't want to fail the whole CLI just because the user hasn't
    installed it yet; publish itself catches the ImportError too."""
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


def publish_preflight(kind: str, dry_run: bool, color: bool) -> bool:
    """Print the requirements block at the start of a publish run.

    Shared by both `atlas lens publish` and `atlas asa publish` so the
    contributor sees the same explicit "here's what publishing involves
    and what you need" panel regardless of which artifact they're shipping.

    Returns False if a hard requirement is missing AND we're not in
    --dry-run mode; the caller should bail with exit 1 in that case.
    Dry-run skips the auth gates entirely since nothing leaves the host.

    `kind` is "lens" or "asa" — only affects the printed wording.
    """
    GREEN_ = GREEN if color else ""
    RED_ = RED if color else ""
    YELL_ = YELL if color else ""
    RESET_ = RESET if color else ""

    _safe_print("")
    _safe_print(f"  atlas {kind} publish — submission pre-flight")
    _safe_print("  ──────────────────────────────────────────")
    _safe_print("  Publish does TWO things in one command:")
    _safe_print("    1. Uploads the artifact to a HuggingFace repo you own")
    _safe_print("    2. Opens a registry PR against github.com/itigges22/ATLAS")
    _safe_print("  Full walkthrough: docs/PUBLISHING.md")
    _safe_print("")

    token_ok = bool(_hf_token())
    hf_pkg_ok = _huggingface_hub_available()
    gh_ok = _gh_available()

    # In dry-run mode the auth gates aren't enforced, so rendering a red
    # ✗ next to "HF_TOKEN required" is alarming and confusing — users
    # think they did something wrong when --dry-run is exactly the path
    # for previewing without setting any of this up. Render missing
    # items as dim ○ in dry-run, ✗/⚠ in real-run.
    def _row(label: str, ok: bool, required: bool, hint: str) -> None:
        if ok:
            mark = f"{GREEN_}✓{RESET_}"
        elif dry_run:
            mark = "○"  # neutral — not enforced
        elif required:
            mark = f"{RED_}✗{RESET_}"
        else:
            mark = f"{YELL_}⚠{RESET_}"
        if ok or dry_run:
            # Don't print the "required" hint when we're not enforcing it —
            # it adds visual noise that contradicts the dry-run header.
            suffix = "" if ok else "  (would be needed for a real upload)"
        else:
            suffix = f"  {hint}"
        _safe_print(f"  {mark} {label}{suffix}")

    _row("HF_TOKEN env var",
         token_ok, required=True,
         hint=(f"{RED_}required{RESET_} — get a write token at "
               "https://huggingface.co/settings/tokens, then "
               "`export HF_TOKEN=hf_...`"))
    _row("huggingface_hub Python pkg",
         hf_pkg_ok, required=True,
         hint=(f"{RED_}required{RESET_} — `pip install huggingface_hub`"))
    _row("gh CLI",
         gh_ok, required=False,
         hint=(f"{YELL_}optional{RESET_} — without it we'll print the PR "
               "body for you to paste at "
               "https://github.com/itigges22/ATLAS/compare"))
    _safe_print("")

    if dry_run:
        _safe_print(f"  {YELL_}--dry-run{RESET_}: nothing will leave the host "
                    f"(no upload, no PR opened, no auth enforced)")
        _safe_print("")
        return True

    missing_required = (not token_ok) or (not hf_pkg_ok)
    if missing_required:
        _safe_print(f"  {RED_}Cannot continue: missing required credentials.{RESET_}")
        _safe_print("  Fix the items marked ✗ above, then re-run. "
                    "Or use --dry-run to preview the PR body without uploading.")
        _safe_print("")
        return False

    return True


def _render_model_card_md(model_name: str, base_model: str, dim: int,
                           sha256: str, size_bytes: int,
                           license_id: str, files_uploaded: List[str]) -> str:
    """Generate the README.md / model card body for the HF upload.

    Front-matter is the YAML block HuggingFace renders into the sidebar
    badge (license, tags, base_model). Body documents what these artifacts
    are and how to point ATLAS at them.
    """
    files_list = "\n".join(f"- `{f}`" for f in files_uploaded)
    return f"""---
license: {license_id}
tags:
- atlas
- geometric-lens
- code-evaluation
base_model: {base_model}
---

# ATLAS Geometric Lens artifacts for {model_name}

Cost-field C(x), G(x) classifier, and their per-model calibration trained
against the {base_model} embedding space. Loaded by the ATLAS geometric-lens
service to score code candidates without execution.

## Files

{files_list}

## Use

```bash
# Drop these into your ATLAS checkout
mkdir -p geometric-lens/geometric_lens/models/
huggingface-cli download <this-repo> \\
  --local-dir geometric-lens/geometric_lens/models/

# Verify ATLAS picks them up
atlas lens check
# expected: verdict: compat
```

## Artifact metadata

| Field | Value |
|---|---|
| Base model | {base_model} |
| Input embedding dim | {dim} |
| cost_field.pt SHA256 | `{sha256}` |
| cost_field.pt size | {size_bytes / (1024*1024):.2f} MB |

## Provenance

Trained locally via `atlas lens build` against {base_model}'s
self-embeddings. Architecture: {dim} -> 512 -> 128 -> 1 (SiLU, SiLU,
Softplus). Contrastive ranking loss on labeled pass/fail code samples.

## License

{license_id}. The artifact derives from a {base_model} forward-pass —
verify the base model's license is compatible with redistribution
before publishing.

## Registry submission

To get ATLAS users this support automatically via `atlas model list`,
open a PR against https://github.com/itigges22/ATLAS using the body
`atlas lens publish` produced. PC-059 (#101) tracks the manual-review
flow; PC-060 (#102) tracks the eventual auto-merge pipeline.
"""


def _render_registry_pr_body(model_name: str, hf_repo: str,
                              base_model: str, dim: int, sha256: str,
                              license_id: str,
                              artifact_files: Optional[List[str]] = None) -> str:
    """Markdown body for the registry-add PR.

    Includes the suggested Python diff so the maintainer can paste it
    directly into atlas/cli/commands/model_registry.py.
    """
    dim_label = (f"{dim}" if dim
                 else "(unverified — install torch on the publisher's host "
                      "for atlas lens publish to capture this)")
    return f"""## Add Lens artifacts for `{model_name}` (auto-generated by `atlas lens publish`)

### Summary

User-trained Geometric Lens cost-field for `{model_name}`, uploaded to
HuggingFace at https://huggingface.co/{hf_repo}.

### Verification checklist (maintainer review per PC-059)

- [ ] HF link reachable: https://huggingface.co/{hf_repo}
- [ ] License is permissive for redistribution ({license_id})
- [ ] `cost_field.pt` SHA256 matches: `{sha256}`
- [ ] Artifact input dim ({dim_label}) matches the base model's embedding dim
- [ ] Spot-check: download + run `atlas lens check` against the base model

### Suggested registry diff

Add the following to `atlas/cli/commands/model_registry.py` (or update
the existing entry's `lens_status` from `no-artifacts`/`unverified`
to `supported`):

```python
Model(
    name="{model_name}",
    # ... existing tier / model_file / model_size_gb / download_url ...
    lens_status="supported",
    lens_calibrated=True,
    lens_artifact_dir=None,  # uses ATLAS_LENS_MODELS dir; per-model layout TBD by PC-058 follow-on
    lens_artifact_files={(artifact_files or ["cost_field.pt"])!r},
    license="{license_id}",
),
```

### Provenance

Trained locally via `atlas lens build` against `{base_model}`. Contrast
the merged behavior with the prior `lens_status: {{no-artifacts | unverified}}`
state — `atlas doctor` should stop warning about lens drift on this model
once the PR merges.
"""


def _emit_publish(args: argparse.Namespace, color: bool) -> int:
    """Upload local artifacts to HF + generate a registry-add PR body.

    Pipeline (matches PC-059 issue spec):
      0. publish_preflight: print requirements panel + bail if creds missing
      1. Validate: artifact dir exists + cost_field.pt is in it
      2. Compute SHA256 for the registry entry
      3. (Unless --dry-run) upload artifacts + auto-generated model card to HF
      4. Render the registry-PR markdown
      5. (Unless --skip-pr) open the registry PR via `gh api`; otherwise print body
    """
    if not publish_preflight("lens", dry_run=args.dry_run, color=color):
        return 1

    atlas_root = _atlas_root()

    # 1. Resolve artifacts
    matched = _resolve_model_arg(args.model)
    model_label = matched.name if matched else (args.model or "<unknown-model>")

    if args.artifact_dir:
        artifact_dir = args.artifact_dir
    elif matched and matched.lens_status == "supported":
        artifact_dir = model_registry.lens_artifact_dir_for(matched, atlas_root)
    else:
        env = _configured_lens_models_dir(atlas_root)
        artifact_dir = env or os.path.normpath(os.path.join(
            atlas_root, "geometric-lens", "geometric_lens", "models"))

    cost_path = os.path.join(artifact_dir, "cost_field.pt")
    if not os.path.isfile(cost_path):
        _safe_print(f"  {RED if color else ''}No cost_field.pt at "
                    f"{cost_path}. Run `atlas lens build` first."
                    f"{RESET if color else ''}")
        return 1

    missing = _missing_runtime_artifacts(artifact_dir)
    if missing:
        _safe_print(f"  {RED if color else ''}Lens artifact is incomplete or "
                    f"uncalibrated; missing {', '.join(missing)}. Run "
                    f"`atlas lens build` before publishing."
                    f"{RESET if color else ''}")
        return 1
    invalid = _invalid_runtime_artifacts(artifact_dir, model_label)
    if invalid:
        _safe_print(f"  {RED if color else ''}Lens calibration is invalid: "
                    f"{'; '.join(invalid)}. Run `atlas lens build` before "
                    f"publishing.{RESET if color else ''}")
        return 1

    files_to_upload = ["cost_field.pt", "model_identity.json"]
    files_to_upload.append("cx_normalization.json")
    # Pickle-free twin: include only when at least as fresh as the .pt —
    # an older safetensors is a previous model's weights.
    st_path = os.path.join(artifact_dir, "cost_field.safetensors")
    if (os.path.isfile(st_path)
            and os.path.getmtime(st_path) >= os.path.getmtime(cost_path)):
        files_to_upload.append("cost_field.safetensors")
    # G(x) artifacts ship with C(x) — the build trains both halves, and a
    # registry consumer who only gets cost_field.pt would run with a
    # dormant (or wrong-dimension) G(x). (G(x) itself is XGBoost trees in
    # native JSON — already pickle-free; safetensors doesn't apply.)
    for opt in ("gx_xgboost.json", "gx_weights.json", "gx_thresholds.json"):
        if os.path.isfile(os.path.join(artifact_dir, opt)):
            files_to_upload.append(opt)
    if "gx_xgboost.json" not in files_to_upload:
        _safe_print(f"  {YELL if color else ''}No gx_xgboost.json in "
                    f"{artifact_dir} — publishing C(x) only. Re-run "
                    f"`atlas lens build` to train both halves."
                    f"{RESET if color else ''}")
        # Legacy G(x) carrier — only relevant when the JSON pair is absent.
        # `lens build` doesn't refresh this file, so on a retrained dir it
        # is a previous model's artifact and must not ship.

    # 2. Compute SHA + inspect
    _safe_print(f"[1/5] Hashing {cost_path}…")
    sha = _sha256_file(cost_path)
    size = os.path.getsize(cost_path)
    inspection = _inspect_cost_field(artifact_dir)
    dim = inspection.dim or 0
    if dim == 0:
        _safe_print(f"  {YELL if color else ''}Could not introspect "
                    f"cost_field.pt dim ({inspection.error or 'torch missing'}). "
                    f"Publish will continue but the model-card metadata "
                    f"will omit the dim field.{RESET if color else ''}")
    _safe_print(f"  SHA256: {sha}")
    _safe_print(f"  Size:   {size / (1024 * 1024):.2f} MB")
    _safe_print(f"  Dim:    {dim if dim else '(unknown)'}")
    _safe_print(f"  Files:  {', '.join(files_to_upload)}")

    base_model = (matched.model_display if matched else
                  args.model or
                  os.path.basename(cost_path).replace(".pt", ""))
    license_id = args.license or "apache-2.0"

    # 3. HF upload (or dry-run)
    hf_repo = args.repo or f"<your-hf-username>/atlas-lens-{model_label.lower()}"
    if args.dry_run:
        _safe_print(f"[2/5] (dry-run) would upload to "
                    f"https://huggingface.co/{hf_repo}")
    else:
        if not args.repo:
            _safe_print(f"  {RED if color else ''}--repo HF_USERNAME/REPO_NAME "
                        f"is required (or pass --dry-run to skip upload)."
                        f"{RESET if color else ''}")
            return 1
        token = _hf_token()
        if not token:
            _safe_print(f"  {RED if color else ''}HF_TOKEN env var not set. "
                        f"Get a write token from https://huggingface.co/settings/tokens "
                        f"and: export HF_TOKEN=hf_…{RESET if color else ''}")
            return 1
        try:
            from huggingface_hub import HfApi  # lazy import — heavy dep
        except ImportError:
            _safe_print(f"  {RED if color else ''}huggingface_hub not installed. "
                        f"Install with: pip install huggingface_hub{RESET if color else ''}")
            return 1

        _safe_print(f"[2/5] Uploading to https://huggingface.co/{hf_repo}…")
        api = HfApi(token=token)
        try:
            api.create_repo(repo_id=hf_repo, exist_ok=True)
        except Exception as e:
            _safe_print(f"  {RED if color else ''}HF create_repo failed: "
                        f"{e}{RESET if color else ''}")
            return 1

        # Upload artifact files
        for fname in files_to_upload:
            local = os.path.join(artifact_dir, fname)
            try:
                api.upload_file(path_or_fileobj=local,
                                 path_in_repo=fname,
                                 repo_id=hf_repo)
                _safe_print(f"  uploaded {fname}")
            except Exception as e:
                _safe_print(f"  {RED if color else ''}upload of {fname} "
                            f"failed: {e}{RESET if color else ''}")
                return 1

        # Upload model card
        card_md = _render_model_card_md(model_label, base_model, dim, sha,
                                          size, license_id, files_to_upload)
        try:
            api.upload_file(path_or_fileobj=card_md.encode(),
                             path_in_repo="README.md",
                             repo_id=hf_repo)
            _safe_print("  uploaded README.md (model card)")
        except Exception as e:
            _safe_print(f"  {YELL if color else ''}model card upload "
                        f"failed (artifacts uploaded fine): {e}"
                        f"{RESET if color else ''}")

    # 4. Render registry-PR body
    _safe_print("[3/5] Rendering registry-PR body…")
    pr_body = _render_registry_pr_body(model_label, hf_repo, base_model,
                                         dim, sha, license_id,
                                         artifact_files=files_to_upload)

    # 5. Open PR (or print body for paste)
    if args.skip_pr or args.dry_run:
        _safe_print("[4/5] (skipping PR open — printing body for paste)")
        _safe_print("")
        _safe_print(pr_body)
        _safe_print("")
        _safe_print(f"  {GREEN if color else ''}Publish complete (dry-run mode).{RESET if color else ''}"
                    if args.dry_run else
                    f"  {GREEN if color else ''}Upload complete.{RESET if color else ''} "
                    "Paste the body above into a PR at "
                    "https://github.com/itigges22/ATLAS/compare")
        return 0

    import shutil as _shutil
    gh_path = _shutil.which("gh")
    if not gh_path:
        _safe_print("[4/5] `gh` not found — printing PR body for manual paste")
        _safe_print("")
        _safe_print(pr_body)
        _safe_print(f"  {GREEN if color else ''}Upload complete.{RESET if color else ''} "
                    "Install `gh` (https://cli.github.com) and re-run "
                    "without --skip-pr to auto-open, or paste the body above "
                    "into https://github.com/itigges22/ATLAS/compare manually.")
        return 0

    # The PR is built through the GitHub API (branch + commit + PR) so it
    # works from any install — no local git checkout required. The commit
    # is a real registry edit, not a paste-me suggestion.
    _safe_print("[4/5] Opening registry-PR via the GitHub API…")
    title = (f"Registry: add Lens artifacts for {model_label} "
             f"(via atlas lens publish)")

    # Tier for the new entry: classify the publisher's host (the hardware
    # the artifacts were validated on); fall back to medium.
    entry_tier = "medium"
    try:
        from atlas.cli.commands.tier import classify, probe
        entry_tier = classify(probe()).tier
    except Exception:
        # Host tier detection is optional publishing metadata; medium is the
        # conservative registry fallback when hardware probing is unavailable.
        pass
    model_file = ""
    size_gb = 0.0
    try:
        from atlas.cli.commands import doctor
        model_file = doctor.MODEL_FILE
        base = (doctor.MODEL_DIR if os.path.isabs(doctor.MODEL_DIR)
                else os.path.join(atlas_root, doctor.MODEL_DIR))
        size_gb = round(os.path.getsize(
            os.path.join(base, model_file)) / (1024 ** 3), 1)
    except Exception:
        # Model size enriches the registry entry but is not required to
        # publish a verified artifact bundle.
        pass

    entry = _render_registry_entry(model_label, model_file or model_label,
                                   size_gb, entry_tier, dim, hf_repo,
                                   license_id, files_to_upload)
    def _edit_registry(content: str) -> Optional[str]:
        updated = _registry_set_lens(content, model_label, hf_repo,
                                     files_to_upload)
        if updated is not None:
            return updated
        return _registry_insert_entry(content, model_label, entry)

    pr_url = open_registry_pr_via_api(
        model_label, title, pr_body,
        _edit_registry,
        color)
    if pr_url:
        _safe_print(f"  {GREEN if color else ''}PR opened: "
                    f"{pr_url}{RESET if color else ''}")
    else:
        _safe_print(f"  {YELL if color else ''}Could not open the PR "
                    f"automatically — body below for manual paste at "
                    f"https://github.com/{_UPSTREAM_REPO}/compare"
                    f"{RESET if color else ''}")
        _safe_print("")
        _safe_print(pr_body)

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas lens",
        description="Geometric Lens compat probe + build (PC-057, PC-058)")
    sub = parser.add_subparsers(dest="subcommand")

    p_check = sub.add_parser("check",
        help="probe llama-server for Lens compatibility (PC-057)")
    p_check.add_argument("model", nargs="?", default=None,
        help="registry name or path (default: whatever llama-server has loaded)")
    p_check.add_argument("--json", action="store_true",
        help="machine-readable output")
    p_check.add_argument("--no-color", action="store_true")

    p_build = sub.add_parser("build",
        help="train fresh lens artifacts — C(x) + G(x) — for the loaded "
             "model (PC-058/PC-211)")
    p_build.add_argument("model", nargs="?", default=None,
        help="registry name or path (default: whatever llama-server has loaded)")
    p_build.add_argument("--samples", default=None,
        help="path to a labeled JSON/JSONL training file "
             "(format: [{text, label}, ...]; label is 0 or 1)")
    p_build.add_argument("--from-results", default=None,
        help="train from a benchmark results dir (per_task/) produced by "
             "`atlas bench` — uses this model's own code + pass/fail labels")
    p_build.add_argument("--epochs", type=int, default=200,
        help="training epochs (default: 200)")
    p_build.add_argument("--lr", type=float, default=1e-3,
        help="Adam learning rate (default: 1e-3)")
    p_build.add_argument("--margin", type=float, default=1.0,
        help="contrastive ranking margin (default: 1.0)")
    p_build.add_argument("--artifact-dir", default=None,
        help="where to save the artifacts (cost_field.pt, gx_xgboost.json, "
             "gx_weights.json; default: registry-resolved path)")
    p_build.add_argument("--force", action="store_true",
        help="retrain even if compatible artifacts already exist")
    p_build.add_argument("--dry-run", action="store_true",
        help="extract embeddings but skip training + save")
    p_build.add_argument("--no-telemetry", action="store_true",
        help="with --from-results: don't merge the run's banked "
             "per-candidate embeddings (telemetry/embeddings.emb)")
    p_build.add_argument("--no-color", action="store_true")

    p_retrain = sub.add_parser("retrain",
        help="retrain the lens on samples collected from your own agent use "
             "(per-file accept/deny + pass 👍/👎) — boosts quality on your "
             "workloads")
    p_retrain.add_argument("model", nargs="?", default=None,
        help="registry name or path (default: whatever llama-server has loaded)")
    p_retrain.add_argument("--epochs", type=int, default=200)
    p_retrain.add_argument("--lr", type=float, default=1e-3)
    p_retrain.add_argument("--margin", type=float, default=1.0)
    p_retrain.add_argument("--artifact-dir", default=None,
        help="where to save the artifacts (default: registry-resolved path)")
    p_retrain.add_argument("--dry-run", action="store_true",
        help="extract embeddings but skip training + save")
    p_retrain.add_argument("--no-color", action="store_true")

    p_pub = sub.add_parser("publish",
        help="upload local artifacts to HF + open registry-PR (PC-059)")
    p_pub.add_argument("model", nargs="?", default=None,
        help="registry name or path of the model these artifacts are for")
    p_pub.add_argument("--repo", default=None,
        help="HF repo to upload to (USERNAME/REPO_NAME). Required unless --dry-run.")
    p_pub.add_argument("--license", default="apache-2.0",
        help="SPDX license id (apache-2.0, mit, bsd-3-clause, ...). "
             "Used in HF model card + registry PR. Must be permissive "
             "for redistribution.")
    p_pub.add_argument("--artifact-dir", default=None,
        help="where local cost_field.pt lives "
             "(default: registry-resolved or ATLAS_LENS_MODELS)")
    p_pub.add_argument("--dry-run", action="store_true",
        help="don't upload, don't open PR — just print the body")
    p_pub.add_argument("--skip-pr", action="store_true",
        help="upload to HF but skip the registry PR (print the body)")
    p_pub.add_argument("--no-color", action="store_true")

    args = parser.parse_args(argv)
    if args.subcommand is None:
        parser.print_help()
        return 1

    color = (sys.stdout.isatty()
             and not getattr(args, "no_color", False)
             and not getattr(args, "json", False))

    if args.subcommand == "check":
        return _emit_check(args, color)
    if args.subcommand == "build":
        return _emit_build(args, color)
    if args.subcommand == "retrain":
        # Retrain is `build` sourced from the collected corpus. Set the source
        # flag + the build-only knobs build expects, then reuse its pipeline
        # (embed → C(x)+G(x) → calibrated thresholds → save). --force is
        # implied: a retrain always replaces the current artifacts.
        args.from_collected = True
        args.from_results = None
        args.samples = None
        args.force = True
        args.no_telemetry = True
        return _emit_build(args, color)
    if args.subcommand == "publish":
        return _emit_publish(args, color)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
