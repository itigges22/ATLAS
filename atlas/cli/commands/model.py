"""atlas model — registry-aware install/list/recommend/remove (PC-056).

Subcommands:
    atlas model list       — table of known models with install + lens columns
    atlas model recommend  — best model for this hardware (composes tier.classify
                              + registry, honors lens_status)
    atlas model install    — download from registry's download_url with progress
                              + size check; refuses no-artifacts without --no-lens
    atlas model remove     — delete a model file from ATLAS_MODELS_DIR

The lens_status field is the central truth this command surfaces. A
user installing Qwen3.5-14B-Q5_K_M on a large-tier box gets a working
llama.cpp model but no G(x) verification — half of what makes ATLAS
*ATLAS*. Doctor warns at runtime; this command warns at install time.

Implementation notes:
- urllib (stdlib) for downloads, no third-party deps. Streams in chunks
  with a progress bar. Resume not supported in v1 — partial files are
  deleted on interrupt (PC-056.1 follow-up).
- ATLAS_MODELS_DIR resolution: env var > ./models/ relative to atlas_root
  (containing docker-compose.yml). Mirrors doctor's atlas_root finder.
- --dry-run prints what would happen without touching the network or disk;
  used by tests + by users who want to verify URLs without committing.
"""

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional

from atlas.cli.commands import model_registry, tier
from atlas.cli.commands.model_registry import Model


# Reuse tier's color + unicode-safety primitives. Keep this self-contained
# rather than importing private symbols — tier may evolve, model is its peer.
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RED   = "\033[31m"
GREEN = "\033[32m"
YELL  = "\033[33m"
CYAN  = "\033[36m"


def _supports_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if not enc:
        return False
    try:
        "—✓".encode(enc, errors="strict")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


UNICODE_OK = _supports_unicode()
DASH = "—" if UNICODE_OK else "--"


def _safe_print(s: str = "") -> None:
    if UNICODE_OK:
        print(s)
        return
    s = (s.replace("—", "--").replace("→", "->")
          .replace("│", "|").replace("─", "-")
          .replace("✓", "[OK]").replace("⚠", "[WARN]")
          .replace("✗", "[FAIL]"))
    print(s.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Path resolution — mirror doctor's atlas_root logic
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


def _resolve_models_dir(arg_models_dir: Optional[str]) -> str:
    """Resolution order: --models-dir flag > ATLAS_MODELS_DIR env >
    ./models/ relative to atlas_root."""
    if arg_models_dir:
        return os.path.abspath(arg_models_dir)
    env = os.environ.get("ATLAS_MODELS_DIR")
    if env:
        return os.path.abspath(env)
    return os.path.join(_find_atlas_root(), "models")


# ---------------------------------------------------------------------------
# Lens-status rendering
# ---------------------------------------------------------------------------

def _lens_icon(status: str, color: bool) -> str:
    if not color or not UNICODE_OK:
        return {"supported": "[OK]  ", "no-artifacts": "[WARN]",
                "unverified": "[????]"}.get(status, "[????]")
    return {"supported":   f"{GREEN}✓{RESET}",
            "no-artifacts": f"{YELL}⚠{RESET}",
            "unverified":   f"{YELL}?{RESET}"}.get(status, "?")


def _lens_label(status: str) -> str:
    return {"supported": "Lens supported",
            "no-artifacts": "Lens no-artifacts",
            "unverified": "Lens unverified"}.get(status, status)


# ---------------------------------------------------------------------------
# `atlas model list`
# ---------------------------------------------------------------------------

def _filter_models(models: List[Model], args: argparse.Namespace,
                   models_dir: str) -> List[Model]:
    out = list(models)
    if args.tier:
        out = [m for m in out if m.tier == args.tier]
    if args.installed:
        out = [m for m in out if model_registry.is_installed(m, models_dir)]
    if args.lens_supported:
        out = [m for m in out if m.lens_status == "supported"]
    return out


def _emit_list(args: argparse.Namespace, color: bool) -> int:
    models_dir = _resolve_models_dir(args.models_dir)
    models = _filter_models(model_registry.all_models(), args, models_dir)

    if args.json:
        out = []
        for m in models:
            d = model_registry.as_dict(m)
            d["installed"] = model_registry.is_installed(m, models_dir)
            d["installed_size_gb"] = model_registry.installed_size_gb(m, models_dir)
            out.append(d)
        print(json.dumps({"models_dir": models_dir, "models": out},
                         indent=2, ensure_ascii=not UNICODE_OK))
        return 0

    hdr = f"{BOLD}ATLAS model registry{RESET}" if color else "ATLAS model registry"
    _safe_print(f"{hdr} {DASH} models dir: {models_dir}")
    _safe_print()
    if not models:
        _safe_print("  (no models match these filters)")
        return 0

    # Compact table. Columns: lens-icon, name, tier, size, installed?, status note
    for m in models:
        installed = model_registry.is_installed(m, models_dir)
        inst_marker = (f"{GREEN}installed{RESET}" if color else "installed") \
                      if installed else (f"{DIM}not installed{RESET}" if color else "not installed")
        if not m.can_install:
            inst_marker = f"{DIM}(upstream gated){RESET}" if color else "(upstream gated)"
        icon = _lens_icon(m.lens_status, color)
        name_col = f"{BOLD}{m.name}{RESET}" if color else m.name
        _safe_print(f"  {icon}  {name_col}")
        _safe_print(f"      tier: {m.tier:6s}  size: {m.model_size_gb:5.1f} GB  "
                    f"{_lens_label(m.lens_status)}  {DASH}  {inst_marker}")
        if installed:
            cur = model_registry.installed_size_gb(m, models_dir)
            if cur is not None and abs(cur - m.model_size_gb) > 0.5:
                _safe_print(f"      {YELL if color else ''}note: on-disk size "
                            f"{cur:.1f} GB differs from registered "
                            f"{m.model_size_gb:.1f} GB{RESET if color else ''}")
        _safe_print()
    _safe_print(f"  {DIM if color else ''}Run `atlas model install <name>` "
                f"to download. Models marked Lens no-artifacts will install as "
                f"raw GGUFs but G(x) verification will silently no-op — pass "
                f"--no-lens to acknowledge.{RESET if color else ''}")
    return 0


# ---------------------------------------------------------------------------
# `atlas model recommend`
# ---------------------------------------------------------------------------

def _emit_recommend(args: argparse.Namespace, color: bool) -> int:
    p = tier.probe(install_dir=args.install_dir)
    t = tier.classify(p)
    rec = model_registry.for_tier(t.tier)

    if args.json:
        out = {
            "host_tier": t.tier,
            "recommendation": (model_registry.as_dict(rec) if rec else None),
            "fallback": None,
        }
        # If the tier-recommended model isn't `supported`, surface 9B as
        # the fallback that actually works end-to-end.
        if rec is None or rec.lens_status != "supported":
            supported = model_registry.supported_models()
            if supported:
                out["fallback"] = model_registry.as_dict(supported[0])
        print(json.dumps(out, indent=2, ensure_ascii=not UNICODE_OK))
        return 0

    hdr = f"{BOLD}ATLAS model recommend{RESET}" if color else "ATLAS model recommend"
    _safe_print(f"{hdr} {DASH} matching registry to your hardware tier")
    _safe_print()
    _safe_print(f"  Detected tier: {t.tier}  ({p.gpu_name or 'no GPU'}, "
                f"{p.vram_gb:.1f} GB VRAM)")
    _safe_print()
    if rec is None:
        _safe_print(f"  {YELL if color else ''}No registered model for tier "
                    f"`{t.tier}`.{RESET if color else ''}")
        return 1
    icon = _lens_icon(rec.lens_status, color)
    _safe_print(f"  {icon}  Tier-default: {BOLD if color else ''}{rec.name}{RESET if color else ''} "
                f"({rec.model_display}, {rec.model_size_gb:.1f} GB)")
    _safe_print(f"      Lens status: {_lens_label(rec.lens_status)}")
    if rec.lens_status == "supported":
        if rec.can_install:
            _safe_print(f"      {GREEN if color else ''}Ready to install:"
                        f"{RESET if color else ''} "
                        f"`atlas model install {rec.name}`")
        else:
            _safe_print(f"      {YELL if color else ''}Upstream is gated; "
                        f"see SETUP.md for manual download.{RESET if color else ''}")
        return 0

    # Tier-default has no Lens artifacts. Surface 9B as the fallback.
    _safe_print()
    _safe_print(f"  {YELL if color else ''}This tier's recommended model has "
                f"no Lens artifacts.{RESET if color else ''} G(x) verification "
                f"will silently no-op if you install it.")
    supported = model_registry.supported_models()
    if supported:
        f = supported[0]
        _safe_print()
        _safe_print(f"  {GREEN if color else ''}Recommended fallback "
                    f"(end-to-end supported):{RESET if color else ''} "
                    f"{BOLD if color else ''}{f.name}{RESET if color else ''}")
        _safe_print(f"      tier: {f.tier} (your hardware: {t.tier} {DASH} "
                    f"{'over-provisioned, fine' if t.tier in ('large','xlarge') else 'under-provisioned, may run slow'})")
        _safe_print(f"      `atlas model install {f.name}`")
    return 0


# ---------------------------------------------------------------------------
# `atlas model install`
# ---------------------------------------------------------------------------

def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _emit_install(args: argparse.Namespace, color: bool) -> int:
    m = model_registry.by_name(args.name)
    if m is None:
        _safe_print(f"  {RED if color else ''}Unknown model: `{args.name}`"
                    f"{RESET if color else ''}")
        _safe_print("  Run `atlas model list` to see available names.")
        return 1

    models_dir = _resolve_models_dir(args.models_dir)

    # Lens-status gate: refuse no-artifacts unless --no-lens.
    if m.lens_status != "supported" and not args.no_lens:
        _safe_print(f"  {YELL if color else ''}Refusing to install `{m.name}`: "
                    f"Lens status `{m.lens_status}`.{RESET if color else ''}")
        _safe_print()
        _safe_print("  This model has no trained Lens artifacts. ATLAS "
                    "will run llama-server on it, but G(x) verification "
                    "will silently no-op (gx_score: 0.5 on every "
                    "generation). Half of what makes ATLAS *ATLAS* will "
                    "be missing.")
        _safe_print()
        _safe_print("  To proceed anyway: rerun with `--no-lens` to "
                    "acknowledge.")
        _safe_print("  See PC-058 roadmap for the Lens training pipeline "
                    "that will fix this.")
        return 1

    if not m.can_install:
        _safe_print(f"  {RED if color else ''}Cannot install `{m.name}`: "
                    f"upstream gated (no public download URL).{RESET if color else ''}")
        _safe_print(f"  Notes: {m.notes}")
        return 1

    target = os.path.join(models_dir, m.model_file)

    if args.dry_run:
        _safe_print("  [DRY-RUN] Would download:")
        _safe_print(f"    URL:    {m.download_url}")
        _safe_print(f"    Target: {target}")
        _safe_print(f"    Size:   ~{m.model_size_gb:.1f} GB")
        if m.sha256:
            _safe_print(f"    SHA256: {m.sha256}")
        return 0

    # Confirm before clobbering an existing file.
    if os.path.exists(target) and not args.yes:
        cur = model_registry.installed_size_gb(m, models_dir) or 0.0
        _safe_print(f"  Target file already exists: {target} ({cur:.1f} GB)")
        _safe_print(f"  Re-download will overwrite it. Pass `--yes` to "
                    f"proceed, or `atlas model remove {m.name}` first.")
        return 1

    # Make sure models_dir exists.
    try:
        os.makedirs(models_dir, exist_ok=True)
    except OSError as e:
        _safe_print(f"  {RED if color else ''}Cannot create models dir "
                    f"`{models_dir}`: {e}{RESET if color else ''}")
        return 1

    # Free-disk sanity check: refuse if free disk < 1.2 * model size.
    try:
        free_gb = shutil.disk_usage(models_dir).free / (1024 ** 3)
    except OSError:
        free_gb = 0.0
    needed = m.model_size_gb * 1.2
    if free_gb < needed:
        _safe_print(f"  {RED if color else ''}Insufficient disk: "
                    f"{free_gb:.1f} GB free, need ~{needed:.1f} GB "
                    f"(model + headroom).{RESET if color else ''}")
        _safe_print("  Free up space or pass `--models-dir` pointing "
                    "at a larger partition.")
        return 1

    _safe_print(f"  Downloading {m.name} ({m.model_size_gb:.1f} GB)")
    _safe_print(f"    From: {m.download_url}")
    _safe_print(f"    To:   {target}")
    _safe_print()

    return _stream_download(m, target, color)


def _stream_download(m: Model, target: str, color: bool) -> int:
    """Stream-download the model with a progress bar. Deletes partial
    file on interrupt or error (no resume support in v1)."""
    tmp = target + ".part"
    chunk = 1024 * 1024  # 1 MiB
    started = time.monotonic()
    bytes_seen = 0
    last_print = 0.0

    try:
        req = urllib.request.Request(m.download_url,
                                     headers={"User-Agent": "atlas-cli/PC-056"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            total = resp.headers.get("Content-Length")
            total_n = int(total) if total else 0
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                bytes_seen += len(buf)
                now = time.monotonic()
                if now - last_print > 0.25 or (total_n and bytes_seen >= total_n):
                    last_print = now
                    _print_progress(bytes_seen, total_n, started, color)
    except KeyboardInterrupt:
        _safe_print()
        _safe_print(f"  {YELL if color else ''}Interrupted. Removing partial "
                    f"file.{RESET if color else ''}")
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return 130
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        _safe_print()
        _safe_print(f"  {RED if color else ''}Download failed: {e}"
                    f"{RESET if color else ''}")
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return 1

    _safe_print()
    # Final size sanity check
    try:
        actual = os.stat(tmp).st_size
    except OSError:
        actual = 0
    if actual < 100 * 1024 * 1024:
        _safe_print(f"  {RED if color else ''}Downloaded file is too small "
                    f"({_human_bytes(actual)}). Aborting.{RESET if color else ''}")
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return 1

    # Atomic rename .part -> final.
    try:
        os.replace(tmp, target)
    except OSError as e:
        _safe_print(f"  {RED if color else ''}Failed to move into place: {e}"
                    f"{RESET if color else ''}")
        return 1

    elapsed = time.monotonic() - started
    rate = bytes_seen / elapsed if elapsed > 0 else 0.0
    _safe_print(f"  {GREEN if color else ''}Done.{RESET if color else ''} "
                f"{_human_bytes(actual)} in {elapsed:.0f}s "
                f"({_human_bytes(rate)}/s)")
    if m.sha256:
        _safe_print(f"  Note: SHA256 verification not yet implemented "
                    f"(PC-056.1). Expected: {m.sha256[:16]}...")
    return 0


def _print_progress(seen: int, total: int, started: float, color: bool) -> None:
    elapsed = max(time.monotonic() - started, 0.001)
    rate = seen / elapsed
    if total:
        pct = seen / total * 100
        eta = (total - seen) / rate if rate > 0 else 0
        bar_w = 30
        fill = int(bar_w * seen / total)
        bar = "=" * fill + ">" + " " * max(bar_w - fill - 1, 0)
        msg = (f"  [{bar[:bar_w]}] {pct:5.1f}%  "
               f"{_human_bytes(seen)} / {_human_bytes(total)}  "
               f"{_human_bytes(rate)}/s  ETA {eta:5.0f}s")
    else:
        msg = (f"  {_human_bytes(seen)} downloaded  "
               f"{_human_bytes(rate)}/s")
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# `atlas model remove`
# ---------------------------------------------------------------------------

def _emit_remove(args: argparse.Namespace, color: bool) -> int:
    m = model_registry.by_name(args.name)
    if m is None:
        _safe_print(f"  {RED if color else ''}Unknown model: `{args.name}`"
                    f"{RESET if color else ''}")
        return 1
    models_dir = _resolve_models_dir(args.models_dir)
    target = os.path.join(models_dir, m.model_file)
    if not os.path.exists(target):
        _safe_print(f"  Model `{m.name}` is not installed at {target}.")
        return 0
    if not args.yes:
        cur = model_registry.installed_size_gb(m, models_dir) or 0.0
        _safe_print(f"  About to delete: {target} ({cur:.1f} GB)")
        _safe_print("  Pass `--yes` to confirm.")
        return 1
    try:
        os.unlink(target)
    except OSError as e:
        _safe_print(f"  {RED if color else ''}Failed to delete: {e}"
                    f"{RESET if color else ''}")
        return 1
    _safe_print(f"  {GREEN if color else ''}Removed:{RESET if color else ''} "
                f"{target}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas model",
        description="Model registry: list, install, remove, recommend (PC-056)")
    sub = parser.add_subparsers(dest="subcommand")

    p_list = sub.add_parser("list", help="show known models")
    p_list.add_argument("--tier", choices=["cpu","small","medium","large","xlarge"],
        help="filter to a specific tier")
    p_list.add_argument("--installed", action="store_true",
        help="show only models already on disk")
    p_list.add_argument("--lens-supported", action="store_true",
        help="show only models with trained Lens artifacts")
    p_list.add_argument("--models-dir", default=None,
        help="override ATLAS_MODELS_DIR")
    p_list.add_argument("--json", action="store_true",
        help="machine output")
    p_list.add_argument("--no-color", action="store_true")

    p_rec = sub.add_parser("recommend",
        help="best model for this hardware (composes atlas tier + registry)")
    p_rec.add_argument("--install-dir", default=None,
        help="probe disk free against this path (defaults to /)")
    p_rec.add_argument("--json", action="store_true")
    p_rec.add_argument("--no-color", action="store_true")

    p_inst = sub.add_parser("install", help="download a model into ATLAS_MODELS_DIR")
    p_inst.add_argument("name", help="model name (see `atlas model list`)")
    p_inst.add_argument("--dry-run", action="store_true",
        help="print what would happen, no network or disk writes")
    p_inst.add_argument("--no-lens", action="store_true",
        help="acknowledge installing a model with no Lens artifacts "
             "(G(x) verification will silently no-op)")
    p_inst.add_argument("--yes", action="store_true",
        help="overwrite existing file without prompt")
    p_inst.add_argument("--models-dir", default=None,
        help="override ATLAS_MODELS_DIR")
    p_inst.add_argument("--no-color", action="store_true")

    p_rm = sub.add_parser("remove", help="delete a model file from ATLAS_MODELS_DIR")
    p_rm.add_argument("name", help="model name (see `atlas model list`)")
    p_rm.add_argument("--yes", action="store_true", help="skip confirmation")
    p_rm.add_argument("--models-dir", default=None,
        help="override ATLAS_MODELS_DIR")
    p_rm.add_argument("--no-color", action="store_true")

    args = parser.parse_args(argv)
    if args.subcommand is None:
        parser.print_help()
        return 1

    color = (sys.stdout.isatty() and not getattr(args, "no_color", False)
             and not getattr(args, "json", False))

    if args.subcommand == "list":
        return _emit_list(args, color)
    if args.subcommand == "recommend":
        return _emit_recommend(args, color)
    if args.subcommand == "install":
        return _emit_install(args, color)
    if args.subcommand == "remove":
        return _emit_remove(args, color)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
