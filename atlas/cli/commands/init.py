"""`atlas init` — first-run install wizard (PC-054).

Composes existing primitives (no new install/probe/registry logic):

  * tier.classify(probe())                  → TierProfile
  * model_registry.for_tier()               → suggested Model
  * model_registry.supported_models()[0]    → fallback when tier default is no-artifacts
  * model.main(["install", ...])            → download + SHA verify (inherits PC-056.1/.2 gates)

Then writes:
  * <atlas_root>/.env                       — Compose configuration
  * <atlas_root>/secrets/api-keys.json      — bearer-token auth (mode 0600, parent 0700)

Flags:
  --yes              non-interactive, accept all defaults
  --skip-download    write .env + api-keys but skip model.install
  --reconfigure      back up existing .env → .env.bak before writing
  --dry-run          print proposed writes; touch nothing
  --json             machine-readable summary
  --models-dir PATH  override default <atlas_root>/models
  --image-tag TAG    non-interactive image-source choice (default: latest)
  --no-color
"""

from __future__ import annotations

import argparse
import json as jsonlib
import os
import secrets as secrets_mod
import shutil
import sys
from typing import List, Optional, Tuple

from atlas.cli.commands import model, model_registry, tier


# ---------------------------------------------------------------------------
# Output helpers (mirror tier.py / model.py — same conventions)
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _supports_unicode() -> bool:
    enc = (sys.stdout.encoding or "").lower()
    return "utf" in enc


UNICODE_OK = _supports_unicode()


def _safe_print(s: str = "") -> None:
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))


def _ok(color: bool) -> str:
    return f"{GREEN}OK{RESET}" if color else "OK"


def _warn(color: bool) -> str:
    return f"{YELLOW}WARN{RESET}" if color else "WARN"


def _err(color: bool) -> str:
    return f"{RED}FAIL{RESET}" if color else "FAIL"


# ---------------------------------------------------------------------------
# Path resolution — share atlas_root with model.py / doctor.py
# ---------------------------------------------------------------------------

def _find_atlas_root() -> str:
    """Walk up from CWD looking for docker-compose.yml. Falls back to CWD."""
    cur = os.path.abspath(os.getcwd())
    while True:
        if os.path.isfile(os.path.join(cur, "docker-compose.yml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.getcwd())
        cur = parent


def _resolve_models_dir(arg_models_dir: Optional[str], atlas_root: str) -> str:
    if arg_models_dir:
        return os.path.abspath(arg_models_dir)
    env = os.environ.get("ATLAS_MODELS_DIR")
    if env:
        return os.path.abspath(env)
    return os.path.join(atlas_root, "models")


# ---------------------------------------------------------------------------
# Step 1 — hardware probe
# ---------------------------------------------------------------------------

def _step_probe(args: argparse.Namespace, color: bool) -> Tuple[tier.Probe, tier.TierProfile]:
    probe = tier.probe()
    profile = tier.classify(probe)
    _safe_print(f"  Detected tier: {BOLD if color else ''}{profile.tier}{RESET if color else ''} "
                f"({profile.label})")
    if probe.has_gpu:
        _safe_print(f"    GPU: {probe.gpu_name or 'unknown'} "
                    f"({probe.vram_gb:.1f} GB VRAM)")
    else:
        _safe_print("    GPU: none detected")
    _safe_print(f"    System: {probe.system_ram_gb:.1f} GB RAM, "
                f"{probe.cpu_cores} cores, "
                f"{probe.disk_free_gb:.1f} GB free")
    return probe, profile


# ---------------------------------------------------------------------------
# Step 2 — model selection
# ---------------------------------------------------------------------------

def _step_select_model(profile: tier.TierProfile, args: argparse.Namespace,
                        color: bool) -> Optional[model_registry.Model]:
    """Pick a model for the user. Tier default if `supported`, otherwise
    surface the supported-fallback so wizard never recommends a model
    where G(x) silently no-ops."""
    tier_default = model_registry.for_tier(profile.tier)
    supported = model_registry.supported_models()
    fallback = supported[0] if supported else None

    if tier_default and tier_default.lens_status == "supported":
        _safe_print(f"  Recommended: {BOLD if color else ''}{tier_default.name}{RESET if color else ''} "
                    f"({tier_default.model_size_gb:.1f} GB, Lens supported)")
        return tier_default

    # Tier default is missing (cpu) or no-artifacts — fall back.
    if tier_default and tier_default.lens_status != "supported":
        _safe_print(f"  Tier default ({tier_default.name}) has lens_status="
                    f"{tier_default.lens_status} — G(x) verification would no-op.")
    if fallback is None:
        _safe_print(f"  {RED if color else ''}No Lens-supported model in registry; "
                    f"cannot recommend.{RESET if color else ''}")
        return None
    _safe_print(f"  Falling back to: {BOLD if color else ''}{fallback.name}{RESET if color else ''} "
                f"({fallback.model_size_gb:.1f} GB, Lens supported)")
    return fallback


# ---------------------------------------------------------------------------
# Step 3 — download (delegate to atlas model install)
# ---------------------------------------------------------------------------

def _step_download(m: model_registry.Model, models_dir: str,
                    args: argparse.Namespace, color: bool) -> int:
    """Returns 0 on success (or already-installed), non-zero on failure.
    Skips when --skip-download or --dry-run."""
    if args.skip_download:
        _safe_print(f"  Skipping download (--skip-download). "
                    f"Place {m.model_file} in {models_dir} before bringing the stack up.")
        return 0
    if model_registry.is_installed(m, models_dir):
        _safe_print(f"  Already installed: {os.path.join(models_dir, m.model_file)}")
        return 0
    if args.dry_run:
        _safe_print(f"  (dry-run) would install {m.name} into {models_dir}")
        return 0

    # Compose `atlas model install` — inherits all of PC-056.1/.2's gates,
    # SHA verification, lock, oversized-part guard, HF_TOKEN handling.
    install_argv = ["install", m.name, "--models-dir", models_dir]
    if args.no_color:
        install_argv.append("--no-color")
    if args.yes:
        install_argv.append("--yes")
    rc = model.main(install_argv)
    if rc != 0:
        _safe_print(f"  {RED if color else ''}Model install failed (rc={rc}). "
                    f"Re-run `atlas model install {m.name}` after resolving the issue, "
                    f"then re-run `atlas init --reconfigure` to finish wiring .env."
                    f"{RESET if color else ''}")
    return rc


# ---------------------------------------------------------------------------
# Step 4 — write .env
# ---------------------------------------------------------------------------

def _render_env(m: model_registry.Model, profile: tier.TierProfile,
                 models_dir: str, atlas_root: str, image_tag: str,
                 ghcr_owner: str) -> str:
    """Compose the .env body. Order is stable for diff-friendliness."""
    # models_dir written as a relative path when it's the default
    # <atlas_root>/models, absolute otherwise — keeps `.env` portable
    # across cloned checkouts that follow the same layout.
    default_models = os.path.join(atlas_root, "models")
    models_value = "./models" if os.path.abspath(models_dir) == os.path.abspath(default_models) \
        else models_dir

    keys = {
        "ATLAS_MODELS_DIR": models_value,
        "ATLAS_MODEL_FILE": m.model_file,
        "ATLAS_MODEL_NAME": m.model_file.rsplit(".", 1)[0],
        "ATLAS_CTX_SIZE": str(profile.context_length),
        "PARALLEL_SLOTS": str(profile.parallel_slots),
        "KV_CACHE_TYPE_K": profile.kv_cache_k,
        "KV_CACHE_TYPE_V": profile.kv_cache_v,
        "ATLAS_GHCR_OWNER": ghcr_owner,
        "ATLAS_IMAGE_TAG": image_tag,
        "ATLAS_LLAMA_PORT": "8080",
        "ATLAS_LENS_PORT": "8099",
        "ATLAS_V3_PORT": "8070",
        "ATLAS_SANDBOX_PORT": "30820",
        "ATLAS_PROXY_PORT": "8090",
    }

    lines = [
        "# ATLAS Compose configuration — generated by `atlas init` (PC-054).",
        f"# Tier: {profile.tier} ({profile.label})",
        f"# Model: {m.name} (lens_status={m.lens_status})",
        "# Re-run `atlas init --reconfigure` to regenerate from new defaults.",
        "",
    ]
    for k, v in keys.items():
        lines.append(f"{k}={v}")
    lines.append("")
    return "\n".join(lines)


def _backup_if_exists(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    bak = path + ".bak"
    shutil.copy2(path, bak)
    return bak


def _step_write_env(m: model_registry.Model, profile: tier.TierProfile,
                     models_dir: str, atlas_root: str, args: argparse.Namespace,
                     color: bool) -> Tuple[str, Optional[str]]:
    """Returns (env_path, backup_path_or_None). On --dry-run, no writes."""
    env_path = os.path.join(atlas_root, ".env")
    body = _render_env(m, profile, models_dir, atlas_root,
                       image_tag=args.image_tag,
                       ghcr_owner=args.ghcr_owner)
    if args.dry_run:
        _safe_print(f"  (dry-run) would write {env_path} ({len(body)} bytes)")
        return env_path, None

    backup = _backup_if_exists(env_path) if args.reconfigure else None
    if backup:
        _safe_print(f"  Backed up existing .env → {backup}")

    with open(env_path, "w") as fh:
        fh.write(body)
    _safe_print(f"  Wrote {env_path}")
    return env_path, backup


# ---------------------------------------------------------------------------
# Step 5 — generate api-keys.json
# ---------------------------------------------------------------------------

def _step_write_api_keys(atlas_root: str, args: argparse.Namespace,
                          color: bool) -> Tuple[str, Optional[str], str]:
    """Returns (path, backup_path_or_None, generated_key_or_existing).

    Permissions: parent dir 0700, file 0600. Refuses to fix loose perms
    on existing parent dir without --yes (security guardrail — the user
    might have intentionally chmod'd it for a multi-user setup)."""
    secrets_dir = os.path.join(atlas_root, "secrets")
    keys_path = os.path.join(secrets_dir, "api-keys.json")

    # Parent dir handling
    if os.path.isdir(secrets_dir):
        mode = os.stat(secrets_dir).st_mode & 0o777
        if mode & 0o077 and not args.yes:
            _safe_print(f"  {YELLOW if color else ''}secrets/ exists with "
                        f"loose permissions ({oct(mode)}). Re-run with --yes "
                        f"to chmod to 0700, or chmod manually."
                        f"{RESET if color else ''}")
            return keys_path, None, ""

    if args.dry_run:
        _safe_print(f"  (dry-run) would write {keys_path}")
        return keys_path, None, ""

    os.makedirs(secrets_dir, mode=0o700, exist_ok=True)
    try:
        os.chmod(secrets_dir, 0o700)
    except PermissionError:
        pass  # not fatal — dir already exists, perms already strict enough

    backup = _backup_if_exists(keys_path) if args.reconfigure else None
    if backup:
        _safe_print(f"  Backed up existing api-keys.json → {backup}")

    key = "sk-atlas-" + secrets_mod.token_urlsafe(32)
    payload = {key: {"user": "local", "created_by": "atlas init"}}
    body = jsonlib.dumps(payload, indent=2) + "\n"

    # Write with explicit mode via O_CREAT to avoid a brief 0644 window.
    fd = os.open(keys_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(body)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    try:
        os.chmod(keys_path, 0o600)
    except PermissionError:
        pass

    _safe_print(f"  Wrote {keys_path} (mode 0600)")
    _safe_print(f"  API key: {key}")
    _safe_print("    Set this in your client: Authorization: Bearer <key>")
    return keys_path, backup, key


# ---------------------------------------------------------------------------
# Already-configured guard
# ---------------------------------------------------------------------------

def _refuse_if_already_configured(atlas_root: str, args: argparse.Namespace,
                                    color: bool) -> bool:
    """True = refused (caller should exit). False = proceed."""
    env_path = os.path.join(atlas_root, ".env")
    if not os.path.isfile(env_path):
        return False
    if args.reconfigure:
        return False  # explicit reconfigure — proceed (will back up)
    _safe_print(f"  {RED if color else ''}Already configured: "
                f"{env_path} exists.{RESET if color else ''}")
    _safe_print("  Pass --reconfigure to back up + regenerate, or edit .env directly.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas init",
        description="First-run install wizard: probe hardware, pick a "
                    "model, write .env + api-keys.json. (PC-054)")
    parser.add_argument("--yes", action="store_true",
        help="non-interactive: accept all defaults; required for "
             "scripted bootstrap")
    parser.add_argument("--skip-download", action="store_true",
        help="write config but don't download the model "
             "(bring-your-own gguf)")
    parser.add_argument("--reconfigure", action="store_true",
        help="back up existing .env and api-keys.json (.bak suffix) "
             "before writing new ones")
    parser.add_argument("--dry-run", action="store_true",
        help="print proposed writes, touch no files, no network")
    parser.add_argument("--json", action="store_true",
        help="machine-readable summary on stdout")
    parser.add_argument("--models-dir", default=None,
        help="override default <atlas_root>/models")
    parser.add_argument("--image-tag", default="latest",
        help="ATLAS_IMAGE_TAG to write into .env (default: latest)")
    parser.add_argument("--ghcr-owner", default="itigges22",
        help="ATLAS_GHCR_OWNER to write into .env (default: itigges22)")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    color = (not args.no_color) and sys.stdout.isatty()
    atlas_root = _find_atlas_root()
    models_dir = _resolve_models_dir(args.models_dir, atlas_root)

    # Header
    _safe_print(f"{BOLD if color else ''}atlas init{RESET if color else ''} — "
                f"first-run wizard (atlas_root: {atlas_root})")
    _safe_print("")

    if _refuse_if_already_configured(atlas_root, args, color):
        return 1

    # Step 1
    _safe_print("[1/5] Probing hardware…")
    probe, profile = _step_probe(args, color)
    _safe_print("")

    # Step 2
    _safe_print("[2/5] Selecting model…")
    chosen = _step_select_model(profile, args, color)
    if chosen is None:
        _safe_print(f"  {RED if color else ''}No installable model found "
                    f"for tier={profile.tier}.{RESET if color else ''}")
        return 1
    _safe_print("")

    # Step 3
    _safe_print("[3/5] Downloading model…")
    rc = _step_download(chosen, models_dir, args, color)
    if rc != 0:
        return rc
    _safe_print("")

    # Step 4
    _safe_print("[4/5] Writing .env…")
    env_path, env_backup = _step_write_env(chosen, profile, models_dir,
                                            atlas_root, args, color)
    _safe_print("")

    # Step 5
    _safe_print("[5/5] Generating api-keys.json…")
    keys_path, keys_backup, api_key = _step_write_api_keys(atlas_root, args, color)
    _safe_print("")

    # Next steps
    if not args.dry_run:
        _safe_print(f"{GREEN if color else ''}Setup complete.{RESET if color else ''}")
        _safe_print("Next:")
        _safe_print("  1. docker compose up -d        # bring up the stack")
        _safe_print("  2. atlas doctor               # verify install health")
        _safe_print("  3. atlas                      # start using ATLAS")

    if args.json:
        out = {
            "atlas_root": atlas_root,
            "tier": profile.tier,
            "model": chosen.name,
            "models_dir": models_dir,
            "env_path": env_path,
            "env_backup": env_backup,
            "api_keys_path": keys_path,
            "api_keys_backup": keys_backup,
            "api_key": api_key,
            "image_tag": args.image_tag,
            "ghcr_owner": args.ghcr_owner,
            "dry_run": args.dry_run,
        }
        _safe_print(jsonlib.dumps(out, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
