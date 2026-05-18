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
    atlas lens check Qwen3.5-9B-Q6_K    # probe a registry entry
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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

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

# C(x) constructor permits arbitrary input_dim; what's "compat" is whether
# the saved artifact's first-layer weight matches the model's embedding dim.
# Keeping a canonical value documents the V3.1.0 baseline.
LENS_CANONICAL_DIM = 4096  # Qwen3.5-9B hidden size


def _llama_url() -> str:
    """Resolve where llama-server is listening.

    Mirrors geometric-lens/embedding_extractor.py's resolution order so
    `atlas lens check` agrees with what the lens service itself sees.
    """
    return os.environ.get(
        "ATLAS_LLAMA_URL",
        os.environ.get(
            "LLAMA_EMBED_URL",
            os.environ.get("LLAMA_URL", "http://localhost:8080"),
        ),
    )


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
                       f"inference/entrypoint-v3.1-9b.sh.")
    return probe


# ---------------------------------------------------------------------------
# Artifact resolution + dim inspection
# ---------------------------------------------------------------------------

def _resolve_model_arg(arg: Optional[str]) -> Optional[model_registry.Model]:
    """Best-effort lookup: registry name → Model, or path/None → None.

    `atlas lens check` accepts:
      - a registry name        (e.g. "Qwen3.5-9B-Q6_K")
      - a .gguf path           (any model on disk)
      - nothing                (probe whatever llama-server has loaded)
    """
    if not arg:
        return None
    for m in model_registry.REGISTRY:
        if m.name == arg or m.model_file == os.path.basename(arg):
            return m
    return None


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
    """Best-effort: walk up from CWD looking for docker-compose.yml, fall back to CWD."""
    cur = os.path.abspath(os.getcwd())
    while True:
        if os.path.isfile(os.path.join(cur, "docker-compose.yml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.getcwd())
        cur = parent


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

    # Resolve artifact dir. For known-supported registry entries this is
    # already wired; for arbitrary models we fall back to ATLAS_LENS_MODELS
    # or the global default. Either way, "is there a cost_field.pt whose
    # input dim matches the model's embedding dim?" is the decisive question.
    if matched and matched.lens_status == "supported":
        artifact_dir = model_registry.lens_artifact_dir_for(matched, atlas_root)
    else:
        env = os.environ.get("ATLAS_LENS_MODELS")
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

    # Dim matches. PC-202 hidden-states patch is nice-to-have for G(x)
    # metric tensor work but not required for C(x) scoring; report it
    # as a warning surface rather than a hard failure.
    note = ""
    if not probe.has_hidden_states_patch:
        note = (" Note: PC-202 hidden-states patch not detected on llama-server. "
                "C(x) works fine; G(x) metric-tensor training would need a "
                "patched build (inference/Dockerfile.v31).")
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
    return v.exit_code


# ---------------------------------------------------------------------------
# atlas lens build  (PC-058)
# ---------------------------------------------------------------------------

def _extract_training_embeddings(samples: List[Dict],
                                  llama_url: str,
                                  color: bool) -> Dict:
    """For each sample {text, label}, POST /embedding and collect.

    Returns a dict compatible with training.train_cost_field's `data` arg:
        {"embeddings": [List[float], ...], "labels": [0|1, ...]}
    """
    embeddings: List[List[float]] = []
    labels: List[int] = []
    n = len(samples)
    for i, s in enumerate(samples):
        text = s.get("text") or s.get("content") or ""
        label = int(s.get("label", 0))
        if not text:
            continue
        resp = _http_post_json(f"{llama_url}/embedding",
                               {"content": text}, timeout=60.0)
        if not isinstance(resp, list) or not resp:
            _safe_print(f"  WARN: sample {i+1}/{n} returned empty response")
            continue
        raw = resp[0].get("embedding")
        if isinstance(raw, list) and raw:
            if isinstance(raw[0], list):
                # per-token: mean-pool
                n_tok = len(raw)
                dim = len(raw[0])
                pooled = [0.0] * dim
                for tok in raw:
                    for j, v in enumerate(tok):
                        pooled[j] += v
                pooled = [v / n_tok for v in pooled]
                embeddings.append(pooled)
            else:
                embeddings.append(raw)
            labels.append(label)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            _safe_print(f"  extracted {i+1}/{n} embeddings")
    return {"embeddings": embeddings, "labels": labels}


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
    _safe_print("[1/4] Probing llama-server…")
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
    _safe_print("[2/4] Loading training samples…")
    if not args.samples:
        _safe_print(f"  {RED if color else ''}--samples PATH is required. "
                    f"Point it at a labeled JSON/JSONL file.{RESET if color else ''}")
        _safe_print("  Format: [{\"text\": str, \"label\": 0|1}, ...]")
        _safe_print("  Pull the canonical training set from:")
        _safe_print("    huggingface.co/datasets/itigges22/ATLAS")
        return 1
    samples = _load_training_samples(args.samples)
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
    _safe_print(f"[3/4] Extracting embeddings via {verdict.probe.url}…")
    start = time.time()
    data = _extract_training_embeddings(samples, verdict.probe.url, color)
    elapsed = time.time() - start
    if not data["embeddings"]:
        _safe_print(f"  {RED if color else ''}No embeddings extracted. "
                    f"Check llama-server logs.{RESET if color else ''}")
        return 1
    _safe_print(f"  Extracted {len(data['embeddings'])} embeddings "
                f"in {elapsed:.1f}s")

    if args.dry_run:
        _safe_print("  (dry-run) skipping training + save")
        return 0

    # 4. Train + save
    _safe_print(f"[4/4] Training CostField "
                f"({args.epochs} epochs, lr={args.lr})…")
    try:
        from geometric_lens.training import train_cost_field, save_cost_field
    except ImportError as e:
        _safe_print(f"  {RED if color else ''}Could not import training "
                    f"module: {e}.{RESET if color else ''}")
        _safe_print("  Make sure you're running from an ATLAS checkout "
                    "(geometric-lens/ must be on PYTHONPATH).")
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
        _safe_print(f"  {YELL if color else ''}Test AUC < 0.70 — model is "
                    f"undertrained. Consider more samples or more "
                    f"epochs.{RESET if color else ''}")

    artifact_dir = args.artifact_dir or verdict.artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    cost_path = save_cost_field(result["model"], save_dir=artifact_dir)
    _safe_print(f"  Saved: {cost_path}")
    _safe_print("")
    _safe_print(f"  {GREEN if color else ''}Build complete.{RESET if color else ''}")
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
    _safe_print(f"  ──────────────────────────────────────────")
    _safe_print(f"  Publish does TWO things in one command:")
    _safe_print(f"    1. Uploads the artifact to a HuggingFace repo you own")
    _safe_print(f"    2. Opens a registry PR against github.com/itigges22/ATLAS")
    _safe_print(f"  Full walkthrough: docs/PUBLISHING.md")
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
            mark = f"○"  # neutral — not enforced
        elif required:
            mark = f"{RED_}✗{RESET_}"
        else:
            mark = f"{YELL_}⚠{RESET_}"
        if ok or dry_run:
            # Don't print the "required" hint when we're not enforcing it —
            # it adds visual noise that contradicts the dry-run header.
            suffix = "" if ok else f"  (would be needed for a real upload)"
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
        _safe_print(f"  Fix the items marked ✗ above, then re-run. "
                    f"Or use --dry-run to preview the PR body without uploading.")
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

Cost-field C(x) (and optionally metric tensor G(x)) trained against the
{base_model} embedding space. Loaded by the ATLAS geometric-lens service
to score code candidates without execution.

## Files

{files_list}

## Use

```bash
# Drop these into your ATLAS checkout
mkdir -p geometric-lens/geometric_lens/models/
huggingface-cli download <this-repo> cost_field.pt \\
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
                              license_id: str) -> str:
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
    lens_artifact_dir=None,  # uses ATLAS_LENS_MODELS dir; per-model layout TBD by PC-058 follow-on
    lens_artifact_files=["cost_field.pt"],
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
      5. (Unless --skip-pr) try `gh pr create`; otherwise print body
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
        env = os.environ.get("ATLAS_LENS_MODELS")
        artifact_dir = env or os.path.normpath(os.path.join(
            atlas_root, "geometric-lens", "geometric_lens", "models"))

    cost_path = os.path.join(artifact_dir, "cost_field.pt")
    if not os.path.isfile(cost_path):
        _safe_print(f"  {RED if color else ''}No cost_field.pt at "
                    f"{cost_path}. Run `atlas lens build` first."
                    f"{RESET if color else ''}")
        return 1

    metric_path = os.path.join(artifact_dir, "metric_tensor.pt")
    files_to_upload = ["cost_field.pt"]
    if os.path.isfile(metric_path):
        files_to_upload.append("metric_tensor.pt")

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
            _safe_print(f"  uploaded README.md (model card)")
        except Exception as e:
            _safe_print(f"  {YELL if color else ''}model card upload "
                        f"failed (artifacts uploaded fine): {e}"
                        f"{RESET if color else ''}")

    # 4. Render registry-PR body
    _safe_print(f"[3/5] Rendering registry-PR body…")
    pr_body = _render_registry_pr_body(model_label, hf_repo, base_model,
                                         dim, sha, license_id)

    # 5. Open PR (or print body for paste)
    if args.skip_pr or args.dry_run:
        _safe_print(f"[4/5] (skipping PR open — printing body for paste)")
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
        _safe_print(f"[4/5] `gh` not found — printing PR body for manual paste")
        _safe_print("")
        _safe_print(pr_body)
        _safe_print(f"  {GREEN if color else ''}Upload complete.{RESET if color else ''} "
                    "Install `gh` (https://cli.github.com) and re-run "
                    "without --skip-pr to auto-open, or paste the body above "
                    "into https://github.com/itigges22/ATLAS/compare manually.")
        return 0

    _safe_print(f"[4/5] Opening registry-PR via `gh pr create`…")
    import subprocess
    # Only pass --head when ATLAS_PUBLISH_BRANCH is explicitly set; gh
    # rejects an empty --head value outright (it won't fall back to
    # auto-detect). Without the env var, omit the flag and let gh infer
    # from the current checkout — that's the path most contributors hit.
    gh_args = ["gh", "pr", "create",
               "--repo", "itigges22/ATLAS",
               "--title", f"Registry: add Lens artifacts for {model_label} "
                          f"(via atlas lens publish)",
               "--body", pr_body]
    if branch := os.environ.get("ATLAS_PUBLISH_BRANCH", "").strip():
        gh_args += ["--head", branch]
    try:
        result = subprocess.run(
            gh_args, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            _safe_print(f"  {GREEN if color else ''}PR opened: "
                        f"{result.stdout.strip()}{RESET if color else ''}")
        else:
            _safe_print(f"  {YELL if color else ''}`gh pr create` exited "
                        f"{result.returncode}. Falling back to printing the "
                        f"body for manual paste.{RESET if color else ''}")
            if result.stderr.strip():
                _safe_print(f"    gh stderr: {result.stderr.strip()[:200]}")
            _safe_print("")
            _safe_print(pr_body)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        _safe_print(f"  {YELL if color else ''}`gh` invocation failed: {e}. "
                    f"PR body printed below for manual paste.{RESET if color else ''}")
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
        help="train fresh C(x) artifacts for the loaded model (PC-058)")
    p_build.add_argument("model", nargs="?", default=None,
        help="registry name or path (default: whatever llama-server has loaded)")
    p_build.add_argument("--samples", default=None,
        help="path to a labeled JSON/JSONL training file "
             "(format: [{text, label}, ...]; label is 0 or 1)")
    p_build.add_argument("--epochs", type=int, default=200,
        help="training epochs (default: 200)")
    p_build.add_argument("--lr", type=float, default=1e-3,
        help="Adam learning rate (default: 1e-3)")
    p_build.add_argument("--margin", type=float, default=1.0,
        help="contrastive ranking margin (default: 1.0)")
    p_build.add_argument("--artifact-dir", default=None,
        help="where to save cost_field.pt (default: registry-resolved path)")
    p_build.add_argument("--force", action="store_true",
        help="retrain even if compatible artifacts already exist")
    p_build.add_argument("--dry-run", action="store_true",
        help="extract embeddings but skip training + save")
    p_build.add_argument("--no-color", action="store_true")

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
        help="upload to HF but don't try gh pr create (print body)")
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
    if args.subcommand == "publish":
        return _emit_publish(args, color)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
