"""atlas asa — ASA control-vector probe + build + publish (PC-061, GH #113).

Parallels `atlas lens` for the BiasBusters #4 steering vectors. Same
model-coupling problem the Lens had before PC-057/058: a steering vector
trained against Qwen3.5-9B's residual-stream geometry (4096-dim, 36
layers) doesn't transfer to a different model. This module wraps the
existing `geometric-lens/asa_calibration/build_steering_vector.py`
workflow into one CLI so swap-in models can be calibrated end-to-end.

Subcommands:
    atlas asa check   - is the configured ATLAS_CONTROL_VECTOR compatible
                        with the loaded model? (PC-061 Phase 1)
    atlas asa build   - wrap build_steering_vector.py end-to-end:
                        docker cp script + pairs into the lens container,
                        run inside (it has the PC-202 hidden-states client
                        + numpy + gguf-writer), copy the .gguf back out.
    atlas asa publish - upload the trained .gguf to HF + open registry-PR
                        (shares the pipeline added by `atlas lens publish`).

Why the build step shells out to docker: the script needs llama-server
reachable via the in-cluster network AND needs the geometric_lens.embedding_extractor
client (which is /app/ in the lens container, not on host PYTHONPATH).
Bundling the script onto the host would duplicate the dep surface; the
docker-cp + docker-exec dance keeps the source of truth at one path.

Invoke:
    atlas asa check                       # probe the running stack
    atlas asa build                       # train fresh vector w/ bundled pairs
    atlas asa build --pairs custom.jsonl  # train with custom contrast pairs
    atlas asa build --limit 50            # smoke test (50 pairs, ~1 min)
    atlas asa publish --repo USER/REPO    # upload + open registry-PR

Exit codes (check):
    0  compat        - vector exists at the configured path
    1  needs-build   - vector missing
    2  incompatible  - llama-server unreachable
"""

import argparse
import json as jsonlib
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from atlas.cli.commands import lens as lens_module  # for shared helpers
from atlas.cli.commands import model_registry


# Output primitives — mirror lens.py for cross-module UX consistency.
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELL = "\033[33m"
CYAN = "\033[36m"


def _safe_print(s: str = "") -> None:
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", errors="replace").decode("ascii"))


# Default paths shared with the entrypoint (inference/entrypoint-v3.1-9b.sh).
# Keep aligned — operators expect `atlas asa` to point at the same file
# llama-server will actually --control-vector-scaled at boot.
DEFAULT_VECTOR_NAME = "ast_edit_steering.gguf"
DEFAULT_HOST_VECTOR_PATH = "/models/" + DEFAULT_VECTOR_NAME  # llama-server view
DEFAULT_LENS_CONTAINER = "atlas-geometric-lens-1"


def _configured_vector_path() -> str:
    """Path the lens / entrypoint use. Matches the env-var conventions
    in proxy/calibration_status.go's probeASAStatus()."""
    return os.environ.get("ATLAS_CONTROL_VECTOR", DEFAULT_HOST_VECTOR_PATH)


def _host_resolve_vector_path(configured: str, atlas_root: str) -> str:
    """Translate a container-path (/models/foo.gguf) into the host-visible
    equivalent if possible. ATLAS's default deploy bind-mounts
    `${ATLAS_MODELS_DIR:-./models}` into the llama-server container at
    /models, so `/models/x` on the container == `<atlas_root>/models/x`
    on the host (or wherever ATLAS_MODELS_DIR points to).

    Returns the original `configured` path if no translation applies
    (already host-absolute, or no recognizable container prefix).
    """
    if os.path.isfile(configured):
        return configured  # the path resolved directly — use it
    if configured.startswith("/models/"):
        # Try ATLAS_MODELS_DIR first, then the default <atlas_root>/models
        env_dir = os.environ.get("ATLAS_MODELS_DIR")
        candidates = []
        if env_dir:
            host_dir = (env_dir if os.path.isabs(env_dir)
                        else os.path.normpath(os.path.join(atlas_root, env_dir)))
            candidates.append(os.path.join(host_dir, configured[len("/models/"):]))
        candidates.append(os.path.join(atlas_root, "models",
                                         configured[len("/models/"):]))
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
    return configured


def _atlas_root() -> str:
    """Reuse lens.py's resolution so both subcommand families walk up the
    same way."""
    return lens_module._atlas_root()


# ---------------------------------------------------------------------------
# GGUF inspection — read the control-vector dim without loading the file
# ---------------------------------------------------------------------------

def _read_cvector_meta(path: str) -> dict:
    """Read minimal metadata from a control-vector GGUF.

    Returns:
        {"present": bool, "size_bytes": int, "dim": Optional[int],
         "layer_count": Optional[int], "model_hint": Optional[str],
         "error": str}

    We use the `gguf` Python package (already a dep of build_steering_vector.py
    via the lens container) when available on the host; otherwise we fall
    back to a magic-byte heuristic. The full dim probe requires gguf —
    without it we report "present, dim unverified" and let the user verify
    via the build step.
    """
    out = {
        "present": False, "size_bytes": 0, "dim": None,
        "layer_count": None, "model_hint": None, "error": "",
    }
    if not os.path.isfile(path):
        out["error"] = "file not found"
        return out
    out["present"] = True
    out["size_bytes"] = os.path.getsize(path)

    # Magic-byte sanity: GGUF files start with "GGUF" (0x47475546).
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
    except OSError as e:
        out["error"] = f"read failed: {e}"
        return out
    if magic != b"GGUF":
        out["error"] = f"not a GGUF (magic={magic!r})"
        return out

    # Deep parse via the gguf package, if installed.
    try:
        import gguf
    except ImportError:
        out["error"] = "gguf python pkg not installed; dim unverified"
        return out
    try:
        reader = gguf.GGUFReader(path)
        # Pull out metadata keys we know build_steering_vector.py writes
        for field_obj in reader.fields.values():
            name = field_obj.name
            if name == "controlvector.layer_count":
                try:
                    out["layer_count"] = int(field_obj.parts[-1][0])
                except (IndexError, ValueError, TypeError):
                    pass
            elif name == "controlvector.model_hint":
                try:
                    raw = field_obj.parts[-1]
                    out["model_hint"] = bytes(raw).decode("utf-8")
                except (UnicodeDecodeError, TypeError):
                    pass
        # Each "direction.<layer>" tensor has shape (hidden_dim,) — its
        # length IS the residual stream dim we need to match against the
        # model's embedding dim.
        for tensor in reader.tensors:
            if tensor.name.startswith("direction."):
                shape = list(tensor.shape)
                if shape:
                    # First (and typically only) dim
                    out["dim"] = int(shape[0])
                    break
    except Exception as e:  # tolerate broken GGUFs
        out["error"] = f"gguf parse failed: {e}"
    return out


# ---------------------------------------------------------------------------
# atlas asa check  (PC-061 Phase 1)
# ---------------------------------------------------------------------------

@dataclass
class ASACheckVerdict:
    verdict: str          # 'compat' | 'needs-build' | 'incompatible'
    reason: str
    probe: lens_module.LlamaProbe = field(default_factory=lambda: lens_module.LlamaProbe(reachable=False, url=""))
    vector_path: str = ""
    vector_present: bool = False
    vector_size_bytes: int = 0
    vector_dim: Optional[int] = None
    vector_layer_count: Optional[int] = None
    vector_model_hint: Optional[str] = None
    unverified: bool = False  # True when dim couldn't be parsed

    @property
    def exit_code(self) -> int:
        return {"compat": 0, "needs-build": 1, "incompatible": 2}.get(self.verdict, 2)


def _check_asa(atlas_root: str) -> ASACheckVerdict:
    probe = lens_module.probe_llama()
    if not probe.reachable:
        return ASACheckVerdict(
            verdict="incompatible", reason=probe.error, probe=probe,
            vector_path=_configured_vector_path(),
        )

    configured = _configured_vector_path()
    vpath = _host_resolve_vector_path(configured, atlas_root)
    meta = _read_cvector_meta(vpath)
    v = ASACheckVerdict(
        verdict="needs-build",
        reason=meta["error"] or "no control vector at " + vpath,
        probe=probe,
        vector_path=vpath,
        vector_present=meta["present"],
        vector_size_bytes=meta["size_bytes"],
        vector_dim=meta["dim"],
        vector_layer_count=meta["layer_count"],
        vector_model_hint=meta["model_hint"],
    )

    if not meta["present"]:
        v.reason = (f"no control vector at {vpath}. Run `atlas asa build` "
                    f"to train one, or drop a pre-built .gguf at the path.")
        return v

    if meta["dim"] is None:
        # gguf pkg missing on host OR parse failed. File exists, magic
        # bytes check passed. Don't fail the verdict over a host-tooling
        # gap — same fallback the lens check uses for missing torch.
        v.verdict = "compat"
        v.unverified = True
        v.reason = (f"control vector present at {vpath} "
                    f"({meta['size_bytes']} bytes). Dim verification needs "
                    f"the `gguf` Python pkg on the host (pip install gguf). "
                    f"llama-server will refuse to load an incompat vector at "
                    f"boot, so it's safe to proceed — just watch for "
                    f"`control_vector_load failed` in container logs.")
        return v

    if probe.embedding_dim > 0 and meta["dim"] != probe.embedding_dim:
        v.verdict = "needs-build"
        v.reason = (f"Dim mismatch: control vector at {vpath} is "
                    f"{meta['dim']}-dim, but model emits {probe.embedding_dim}-dim "
                    f"residuals. Vector was trained for a different model; "
                    f"run `atlas asa build` to retrain.")
        return v

    v.verdict = "compat"
    v.reason = (f"Control vector at {vpath} matches model "
                f"({meta['dim']}-dim residuals). Ready for "
                f"--control-vector-scaled.")
    return v


def _emit_check(args: argparse.Namespace, color: bool) -> int:
    atlas_root = _atlas_root()
    v = _check_asa(atlas_root)

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
    hdr = f"{BOLD}atlas asa check{RESET}" if color else "atlas asa check"
    _safe_print(f"{hdr}  verdict: {badge}")
    _safe_print(f"  llama-server: {v.probe.url} "
                f"({'reachable' if v.probe.reachable else 'unreachable'})")
    if v.probe.reachable:
        _safe_print(f"  model:        {v.probe.model_name or '(unknown)'}")
        _safe_print(f"  embed dim:    {v.probe.embedding_dim}")
        _safe_print(f"  PC-202 patch: "
                    f"{'yes' if v.probe.has_hidden_states_patch else 'no'}")
    _safe_print(f"  vector path:  {v.vector_path}")
    if v.vector_present:
        _safe_print(f"  vector size:  {v.vector_size_bytes} bytes")
        _safe_print(f"  vector dim:   {v.vector_dim if v.vector_dim else '(unverified)'}")
        if v.vector_layer_count is not None:
            _safe_print(f"  layer count:  {v.vector_layer_count}")
        if v.vector_model_hint:
            _safe_print(f"  model hint:   {v.vector_model_hint}")
    _safe_print("")
    _safe_print(f"  {v.reason}")
    return v.exit_code


# ---------------------------------------------------------------------------
# atlas asa build  (PC-061 Phase 1)
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    import shutil as _shutil
    return _shutil.which("docker") is not None


def _docker_exec(container: str, cmd: List[str], capture: bool = False):
    """Wrap docker exec with a 1h timeout — long enough for a full ASA
    training run (~25 min on a 5060 Ti) without inviting a runaway."""
    full = ["docker", "exec", "-i", container] + cmd
    if capture:
        return subprocess.run(full, capture_output=True, text=True, timeout=3600)
    return subprocess.run(full, timeout=3600)


def _emit_build(args: argparse.Namespace, color: bool) -> int:
    """Train fresh ASA vector by running build_steering_vector.py inside
    the lens container.

    Container needs to be running (operator hasn't `docker compose down`).
    Script + pairs are docker-cp'd in for the run; output .gguf is
    docker-cp'd back. This keeps the source of truth at one path
    (`geometric-lens/asa_calibration/`) instead of forking the workflow.
    """
    atlas_root = _atlas_root()

    if not _docker_available():
        _safe_print(f"  {RED if color else ''}docker not on PATH. "
                    f"`atlas asa build` shells into the lens container; "
                    f"install Docker or use the legacy direct invocation "
                    f"(geometric-lens/asa_calibration/README.md)."
                    f"{RESET if color else ''}")
        return 1

    container = args.container or DEFAULT_LENS_CONTAINER

    # 1. Pre-flight: check the container is up + lens reachable
    _safe_print(f"[1/5] Pre-flight: container {container}, "
                f"llama-server reachable…")
    probe = lens_module.probe_llama()
    if not probe.reachable:
        _safe_print(f"  {RED if color else ''}llama-server unreachable: "
                    f"{probe.error}{RESET if color else ''}")
        return 2
    if not probe.has_hidden_states_patch:
        _safe_print(f"  {RED if color else ''}llama-server does not have "
                    f"the PC-202 hidden-states patch. ASA training extracts "
                    f"per-layer residuals via this patch — rebuild llama-server "
                    f"with inference/Dockerfile.v31 (which bakes the patch) "
                    f"first.{RESET if color else ''}")
        return 2
    ping = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}",
                            container], capture_output=True, text=True)
    if ping.returncode != 0 or ping.stdout.strip() != "true":
        _safe_print(f"  {RED if color else ''}container '{container}' is not "
                    f"running. `docker compose up -d geometric-lens` first."
                    f"{RESET if color else ''}")
        return 2
    _safe_print(f"  model emits {probe.embedding_dim}-dim residuals; "
                f"container running")

    # 2. Resolve script + pairs paths on the host
    asa_dir = os.path.join(atlas_root, "geometric-lens", "asa_calibration")
    script_host = os.path.join(asa_dir, "build_steering_vector.py")
    pairs_host = args.pairs or os.path.join(asa_dir, "contrast_pairs.jsonl")
    if not os.path.isfile(script_host):
        _safe_print(f"  {RED if color else ''}can't find "
                    f"build_steering_vector.py at {script_host}. Are you in "
                    f"an ATLAS checkout?{RESET if color else ''}")
        return 1
    if not os.path.isfile(pairs_host):
        _safe_print(f"  {RED if color else ''}contrast pairs not found at "
                    f"{pairs_host}.{RESET if color else ''}")
        return 1
    # Quick line-count sanity — script needs each pair as 2 lines.
    with open(pairs_host) as fh:
        n_pairs_lines = sum(1 for line in fh if line.strip())
    _safe_print(f"  pairs file: {pairs_host} ({n_pairs_lines} non-empty lines)")

    # 3. Stage script + pairs into the container
    _safe_print(f"[2/5] Staging script + pairs into {container}…")
    for src, dst in [(script_host, "/tmp/build_steering_vector.py"),
                      (pairs_host, "/tmp/contrast_pairs.jsonl")]:
        cp = subprocess.run(["docker", "cp", src, f"{container}:{dst}"],
                             capture_output=True, text=True)
        if cp.returncode != 0:
            _safe_print(f"  {RED if color else ''}docker cp {src} failed: "
                        f"{cp.stderr.strip()}{RESET if color else ''}")
            return 1
    _safe_print(f"  staged")

    if args.dry_run:
        _safe_print(f"[3/5] (dry-run) would run build_steering_vector.py "
                    f"with --layer {args.layer} "
                    f"--limit {args.limit if args.limit else 'all'}")
        return 0

    # 4. Run the build inside the container
    _safe_print(f"[3/5] Training (layer {args.layer}, "
                f"{args.limit or 'all'} pairs). Takes ~25 min for 1000 pairs…")
    cmd = ["python3", "/tmp/build_steering_vector.py",
            "--pairs", "/tmp/contrast_pairs.jsonl",
            "--out", "/tmp/ast_edit_steering.gguf",
            "--layer", str(args.layer)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    start = time.time()
    result = _docker_exec(container, cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        _safe_print(f"  {RED if color else ''}build script exited "
                    f"{result.returncode}. Check container logs: "
                    f"docker logs {container}{RESET if color else ''}")
        return 1
    _safe_print(f"  build completed in {elapsed:.1f}s")

    # 5. Copy result back + save to artifact dir
    artifact_dir = args.artifact_dir or os.path.dirname(
        _configured_vector_path())
    os.makedirs(artifact_dir, exist_ok=True)
    out_path = args.out or os.path.join(artifact_dir, DEFAULT_VECTOR_NAME)
    _safe_print(f"[4/5] Copying built vector to {out_path}…")
    cp = subprocess.run(["docker", "cp",
                          f"{container}:/tmp/ast_edit_steering.gguf",
                          out_path],
                         capture_output=True, text=True)
    if cp.returncode != 0:
        _safe_print(f"  {RED if color else ''}docker cp out failed: "
                    f"{cp.stderr.strip()}{RESET if color else ''}")
        return 1
    size = os.path.getsize(out_path)
    _safe_print(f"  saved: {out_path} ({size} bytes)")

    _safe_print("")
    _safe_print(f"  {GREEN if color else ''}Build complete.{RESET if color else ''}")
    _safe_print(f"  Next: restart llama-server so it picks up the new vector:")
    _safe_print(f"    docker compose up -d --build llama-server --no-deps")
    _safe_print(f"  Then verify: atlas asa check")
    _safe_print(f"  Or share it: atlas asa publish --repo USER/REPO")
    return 0


# ---------------------------------------------------------------------------
# atlas asa publish  (PC-061 — shares pipeline w/ lens publish)
# ---------------------------------------------------------------------------

def _render_asa_pr_body(model_name: str, hf_repo: str, base_model: str,
                         dim: Optional[int], layer: Optional[int],
                         sha256: str, license_id: str) -> str:
    dim_label = f"{dim}-dim" if dim else "(unverified — install gguf pkg)"
    layer_label = f"layer {layer}" if layer else "(unrecorded)"
    return f"""## Add ASA control vector for `{model_name}` (auto-generated by `atlas asa publish`)

### Summary

User-trained BiasBusters #4 ASA steering vector for `{model_name}`,
uploaded to HuggingFace at https://huggingface.co/{hf_repo}.

### Verification checklist (maintainer review per PC-061)

- [ ] HF link reachable: https://huggingface.co/{hf_repo}
- [ ] License is permissive for redistribution ({license_id})
- [ ] `{DEFAULT_VECTOR_NAME}` SHA256 matches: `{sha256}`
- [ ] Residual dim ({dim_label}) matches the base model
- [ ] Trained at {layer_label} (paper recommendation: ~75% of model depth)
- [ ] Spot-check: drop the .gguf at `/models/{DEFAULT_VECTOR_NAME}` and
      confirm llama-server boots without `control_vector_load failed`
- [ ] Behavior smoke-test: pre-vs-post task that benefits from the bias
      (whole-function rewrite via `ast_edit` over `edit_file`)

### Suggested registry diff

Most ASA vectors are distributed alongside the Lens artifacts in the
same model entry. Add an `asa_artifact_files` field if the registry
doesn't already track ASA per-model (V3.1.2 work):

```python
Model(
    name="{model_name}",
    # ... existing fields ...
    lens_status="supported",
    # New (V3.1.2 forward-compat): ASA vector tracking
    asa_artifact_files=["{DEFAULT_VECTOR_NAME}"],
    asa_status="supported",
    license="{license_id}",
),
```

### Provenance

Trained locally via `atlas asa build` against `{base_model}`. Algorithm:
mean-difference (positives−negatives) over per-token residuals at
{layer_label}, projected to a single direction. Same approach as the
Feb 2026 ASA paper (arxiv 2602.04935).
"""


def _emit_publish(args: argparse.Namespace, color: bool) -> int:
    if not lens_module.publish_preflight("asa", dry_run=args.dry_run, color=color):
        return 1

    if not args.dry_run and not args.repo:
        _safe_print(f"  {RED if color else ''}--repo HF_USERNAME/REPO_NAME "
                    f"is required (or pass --dry-run to skip upload)."
                    f"{RESET if color else ''}")
        return 1

    atlas_root = _atlas_root()
    raw = args.vector or _configured_vector_path()
    vpath = _host_resolve_vector_path(raw, atlas_root)
    if not os.path.isfile(vpath):
        _safe_print(f"  {RED if color else ''}No control vector at {vpath}. "
                    f"Run `atlas asa build` first.{RESET if color else ''}")
        return 1

    _safe_print(f"[1/4] Hashing {vpath}…")
    sha = lens_module._sha256_file(vpath)
    size = os.path.getsize(vpath)
    meta = _read_cvector_meta(vpath)
    dim = meta["dim"]
    _safe_print(f"  SHA256: {sha}")
    _safe_print(f"  Size:   {size} bytes")
    _safe_print(f"  Dim:    {dim if dim else '(unverified)'}")
    if meta["model_hint"]:
        _safe_print(f"  Hint:   {meta['model_hint']}")

    matched = lens_module._resolve_model_arg(args.model)
    model_label = matched.name if matched else (args.model or "<unknown-model>")
    base_model = (matched.model_display if matched
                  else "<unknown base model>")
    license_id = args.license or "apache-2.0"

    hf_repo = args.repo or f"<your-hf-username>/atlas-asa-{model_label.lower()}"
    if args.dry_run:
        _safe_print(f"[2/4] (dry-run) would upload to "
                    f"https://huggingface.co/{hf_repo}")
    else:
        token = lens_module._hf_token()
        if not token:
            _safe_print(f"  {RED if color else ''}HF_TOKEN env var not set. "
                        f"Get a write token from https://huggingface.co/settings/tokens "
                        f"and: export HF_TOKEN=hf_…{RESET if color else ''}")
            return 1
        try:
            from huggingface_hub import HfApi
        except ImportError:
            _safe_print(f"  {RED if color else ''}huggingface_hub not installed. "
                        f"Install with: pip install huggingface_hub{RESET if color else ''}")
            return 1
        _safe_print(f"[2/4] Uploading {DEFAULT_VECTOR_NAME} to "
                    f"https://huggingface.co/{hf_repo}…")
        api = HfApi(token=token)
        try:
            api.create_repo(repo_id=hf_repo, exist_ok=True)
            api.upload_file(path_or_fileobj=vpath,
                             path_in_repo=DEFAULT_VECTOR_NAME,
                             repo_id=hf_repo)
            _safe_print(f"  uploaded")
        except Exception as e:
            _safe_print(f"  {RED if color else ''}upload failed: "
                        f"{e}{RESET if color else ''}")
            return 1

    _safe_print(f"[3/4] Rendering registry-PR body…")
    pr_body = _render_asa_pr_body(model_label, hf_repo, base_model,
                                    dim, args.layer, sha, license_id)
    if args.skip_pr or args.dry_run:
        _safe_print(f"[4/4] (skipping PR open — printing body for paste)")
        _safe_print("")
        _safe_print(pr_body)
        _safe_print("")
        _safe_print(f"  {GREEN if color else ''}Publish complete (dry-run mode).{RESET if color else ''}"
                    if args.dry_run else
                    f"  {GREEN if color else ''}Upload complete.{RESET if color else ''} "
                    "Paste the body above into a PR.")
        return 0

    import shutil as _shutil
    gh_path = _shutil.which("gh")
    if not gh_path:
        _safe_print(f"[4/4] `gh` not found — printing PR body for manual paste")
        _safe_print("")
        _safe_print(pr_body)
        return 0
    _safe_print(f"[4/4] Opening registry-PR via `gh pr create`…")
    # Only pass --head when ATLAS_PUBLISH_BRANCH is explicitly set; an
    # empty --head crashes gh outright instead of inferring the current
    # branch. Same fix as lens.py's publish flow.
    gh_args = ["gh", "pr", "create",
               "--repo", "itigges22/ATLAS",
               "--title", f"Registry: add ASA vector for {model_label} "
                          f"(via atlas asa publish)",
               "--body", pr_body]
    if branch := os.environ.get("ATLAS_PUBLISH_BRANCH", "").strip():
        gh_args += ["--head", branch]
    result = subprocess.run(gh_args, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        _safe_print(f"  {GREEN if color else ''}PR opened: "
                    f"{result.stdout.strip()}{RESET if color else ''}")
    else:
        _safe_print(f"  {YELL if color else ''}gh pr create returned "
                    f"{result.returncode} — printing body for paste"
                    f"{RESET if color else ''}")
        _safe_print("")
        _safe_print(pr_body)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="atlas asa",
        description="ASA control-vector compat + build + publish (PC-061)")
    sub = parser.add_subparsers(dest="subcommand")

    p_check = sub.add_parser("check",
        help="probe llama-server + configured control vector (PC-061)")
    # NOTE: no positional `model` arg here (unlike lens check). ASA today
    # has one global vector path (ATLAS_CONTROL_VECTOR); there's no
    # per-model artifact-dir lookup to drive. If per-model vectors land
    # later, re-introduce the arg and wire it through _check_asa.
    p_check.add_argument("--json", action="store_true",
        help="machine-readable output")
    p_check.add_argument("--no-color", action="store_true")

    p_build = sub.add_parser("build",
        help="train fresh ASA vector via lens container (PC-061)")
    p_build.add_argument("model", nargs="?", default=None,
        help="registry name (default: whatever llama-server has loaded)")
    p_build.add_argument("--pairs", default=None,
        help="contrast pairs JSONL (default: "
             "geometric-lens/asa_calibration/contrast_pairs.jsonl)")
    p_build.add_argument("--layer", type=int, default=27,
        help="layer to extract residuals from (default 27 = ~75%% of Qwen3.5-9B's 36)")
    p_build.add_argument("--limit", type=int, default=0,
        help="cap pairs processed for smoke tests (0 = all)")
    p_build.add_argument("--container", default=None,
        help="lens container name (default: " + DEFAULT_LENS_CONTAINER + ")")
    p_build.add_argument("--out", default=None,
        help="where to write the .gguf (default: "
             "<artifact-dir>/" + DEFAULT_VECTOR_NAME + ")")
    p_build.add_argument("--artifact-dir", default=None,
        help="dir to save into (default: dirname of ATLAS_CONTROL_VECTOR)")
    p_build.add_argument("--dry-run", action="store_true",
        help="stage script + pairs, skip the training run")
    p_build.add_argument("--no-color", action="store_true")

    p_pub = sub.add_parser("publish",
        help="upload ASA vector to HF + open registry-PR (PC-061)")
    p_pub.add_argument("model", nargs="?", default=None)
    p_pub.add_argument("--repo", default=None,
        help="HF repo USERNAME/REPO_NAME. Required unless --dry-run.")
    p_pub.add_argument("--vector", default=None,
        help="path to the .gguf (default: ATLAS_CONTROL_VECTOR)")
    p_pub.add_argument("--license", default="apache-2.0",
        help="SPDX license id")
    p_pub.add_argument("--layer", type=int, default=27,
        help="layer the vector was trained at (recorded in PR body)")
    p_pub.add_argument("--dry-run", action="store_true",
        help="don't upload, don't open PR")
    p_pub.add_argument("--skip-pr", action="store_true",
        help="upload to HF but don't try gh pr create")
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
