"""atlas tier — hardware probe + tier classification (PC-055).

Classifies the host's GPU/RAM/disk into one of five tiers and emits the
recommended ATLAS settings for that tier (model, context length, parallel
slots, KV cache quantization). The output is the foundation that PC-056
(model registry) and PC-054 (first-run wizard) compose on top of.

Tier breakpoints are based on VRAM, the hardest constraint for LLM
inference:

  cpu      no NVIDIA GPU             — ATLAS can't run llama.cpp CUDA;
                                       documented for completeness
  small    8 GB <= VRAM < 12 GB      — RTX 3060 / 4060 / T4
  medium   12 GB <= VRAM < 20 GB     — RTX 4060 Ti 16GB / 5060 Ti 16GB /
                                       3080 Ti / 4070 Ti Super 16GB
                                       (default development target)
  large    20 GB <= VRAM < 32 GB     — RTX 3090 / 4090 / 5090 24GB
  xlarge   VRAM >= 32 GB             — RTX 5090 32GB / A6000 / A100 / H100

Settings per tier are tuned for "sensible defaults that won't OOM on the
smallest GPU in the band." Users can always override in `.env` — the
tier output is a recommendation, not a lock.

Invoke:
    atlas tier              # classify this host + show recommendations
    atlas tier list         # show all 5 tier definitions
    atlas tier --json       # machine output (for PC-054 wizard, PC-056)
    atlas tier --raw        # just the probe (no classification)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

# Reuse doctor's color + unicode-safety primitives so output looks consistent.
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
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
          .replace("│", "|").replace("─", "-"))
    print(s.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Probe — read host hardware
# ---------------------------------------------------------------------------

@dataclass
class Probe:
    has_gpu: bool
    gpu_name: Optional[str]
    vram_gb: float
    gpu_count: int
    system_ram_gb: float
    disk_free_gb: float
    platform: str  # 'linux' | 'darwin' | 'windows' | 'other'

    @property
    def description(self) -> str:
        if not self.has_gpu:
            return f"{self.platform} | no GPU | {self.system_ram_gb:.0f} GB RAM"
        return (f"{self.platform} | {self.gpu_name} ({self.vram_gb:.1f} GB VRAM) "
                f"| {self.system_ram_gb:.0f} GB RAM "
                f"| {self.disk_free_gb:.0f} GB free disk")


def _read_nvidia_smi() -> Tuple[bool, Optional[str], float, int]:
    """Return (has_gpu, gpu_name_first, vram_gb_first, gpu_count).

    Picks the FIRST GPU's VRAM as the budget for tier classification —
    multi-GPU is documented as out-of-scope for v1 (PC-055 caveat).
    """
    if not shutil.which("nvidia-smi"):
        return False, None, 0.0, 0
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, None, 0.0, 0
    if p.returncode != 0:
        return False, None, 0.0, 0
    lines = [ln.strip() for ln in p.stdout.strip().split("\n") if ln.strip()]
    if not lines:
        return False, None, 0.0, 0
    first = lines[0]
    try:
        name, mem_mib = [x.strip() for x in first.split(",", 1)]
        vram_gb = float(mem_mib) / 1024.0
    except (ValueError, IndexError):
        return False, None, 0.0, 0
    return True, name, vram_gb, len(lines)


def _read_system_ram_gb() -> float:
    """Cross-platform best-effort system RAM read."""
    # Linux/most-Unix: /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, ValueError, IndexError):
        pass
    # Fallback for macOS (sysctl) or other
    try:
        p = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True, timeout=5)
        if p.returncode == 0:
            return int(p.stdout.strip()) / (1024 ** 3)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return 0.0


def _read_disk_free_gb(path: str = "/") -> float:
    try:
        st = shutil.disk_usage(path)
        return st.free / (1024 ** 3)
    except OSError:
        return 0.0


def probe(install_dir: Optional[str] = None) -> Probe:
    """Run all hardware probes and return a Probe."""
    has_gpu, gpu_name, vram_gb, gpu_count = _read_nvidia_smi()
    sys_ram = _read_system_ram_gb()
    # Probe disk against where ATLAS will live (model files are large).
    disk_path = install_dir if install_dir and os.path.isdir(install_dir) else "/"
    disk_free = _read_disk_free_gb(disk_path)
    plat = sys.platform if sys.platform in ("linux", "darwin", "win32") else "other"
    plat = "windows" if plat == "win32" else plat
    return Probe(has_gpu=has_gpu, gpu_name=gpu_name, vram_gb=vram_gb,
                 gpu_count=gpu_count, system_ram_gb=sys_ram,
                 disk_free_gb=disk_free, platform=plat)


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

@dataclass
class TierProfile:
    """Recommended ATLAS settings for a hardware tier.

    Field meanings map directly to docker-compose.yml / .env knobs:
      model_file       -> ATLAS_MODEL_FILE
      context_length   -> ATLAS_CTX_SIZE / CONTEXT_LENGTH
      parallel_slots   -> PARALLEL_SLOTS  (llama-server --parallel)
      kv_cache_k       -> KV_CACHE_TYPE_K (llama-server -ctk)
      kv_cache_v       -> KV_CACHE_TYPE_V (llama-server -ctv)
    """
    tier: str            # cpu | small | medium | large | xlarge
    label: str           # short human name
    description: str
    min_vram_gb: float
    max_vram_gb: Optional[float]  # None = unbounded above
    example_gpus: List[str]
    # Recommended settings
    model_file: str
    model_display: str
    model_size_gb: float
    context_length: int
    parallel_slots: int
    kv_cache_k: str
    kv_cache_v: str
    notes: str

    def env_vars(self) -> dict:
        """Render the recommended settings as a dict suitable for .env writing."""
        return {
            "ATLAS_MODEL_FILE": self.model_file,
            "ATLAS_MODEL_NAME": self.model_file.rsplit(".", 1)[0],
            "ATLAS_CTX_SIZE": str(self.context_length),
            # Note: PARALLEL_SLOTS / KV_CACHE_TYPE_K|V are read by the
            # llama entrypoint, not directly by docker-compose. Surface
            # them so PC-054 wizard can render them, even though writing
            # them into .env requires the entrypoint contract to honor
            # `${PARALLEL_SLOTS:-...}`.
            "PARALLEL_SLOTS": str(self.parallel_slots),
            "KV_CACHE_TYPE_K": self.kv_cache_k,
            "KV_CACHE_TYPE_V": self.kv_cache_v,
        }


# Tier breakpoints. Order matters — classify() walks top-down picking the
# first whose VRAM range matches.
TIERS: List[TierProfile] = [
    TierProfile(
        tier="cpu",
        label="CPU-only (no GPU)",
        description="No NVIDIA GPU detected. ATLAS requires a CUDA GPU "
                    "for llama.cpp inference. CPU-only is documented for "
                    "completeness but not supported in v1.",
        min_vram_gb=0.0, max_vram_gb=0.0,
        example_gpus=[],
        model_file="N/A",
        model_display="N/A — install a CUDA GPU",
        model_size_gb=0.0,
        context_length=0,
        parallel_slots=0,
        kv_cache_k="N/A", kv_cache_v="N/A",
        notes="ROCm support for AMD GPUs is on the roadmap. "
              "Apple Silicon support requires llama.cpp Metal backend.",
    ),
    TierProfile(
        tier="small",
        label="Small (entry-level GPU)",
        description="Conservative settings sized for 8 GB cards. "
                    "7B Q4 model leaves ~3 GB for KV cache + compute.",
        min_vram_gb=8.0, max_vram_gb=12.0,
        example_gpus=["RTX 3060 8GB", "RTX 4060 8GB", "T4 16GB (datacenter)"],
        model_file="Qwen3.5-7B-Q4_K_M.gguf",
        model_display="Qwen3.5 7B (Q4_K_M)",
        model_size_gb=4.4,
        context_length=8192,
        parallel_slots=1,
        kv_cache_k="q4_0", kv_cache_v="q4_0",
        notes="Q4 KV cache trades ~5% quality for ~50% memory. "
              "Increase to q8_0 if you have 12 GB and find quality lacking.",
    ),
    TierProfile(
        tier="medium",
        label="Medium (mid-range GPU)",
        description="ATLAS development target. 9B Q6 model with 32K "
                    "context fits comfortably with q8/q4 KV cache.",
        min_vram_gb=12.0, max_vram_gb=20.0,
        example_gpus=["RTX 4060 Ti 16GB", "RTX 5060 Ti 16GB",
                      "RTX 3080 Ti 12GB", "RTX 4070 Ti Super 16GB"],
        model_file="Qwen3.5-9B-Q6_K.gguf",
        model_display="Qwen3.5 9B (Q6_K)",
        model_size_gb=6.9,
        context_length=32768,
        parallel_slots=1,
        kv_cache_k="q8_0", kv_cache_v="q4_0",
        notes="Default ATLAS configuration. Verified on RTX 5060 Ti 16GB "
              "with ~3 GB headroom remaining.",
    ),
    TierProfile(
        tier="large",
        label="Large (high-end consumer GPU)",
        description="Headroom for 14B Q5/Q6 model with 32K context and "
                    "2 parallel slots for multi-conversation.",
        min_vram_gb=20.0, max_vram_gb=32.0,
        example_gpus=["RTX 3090 24GB", "RTX 4090 24GB", "RTX 5090 24GB"],
        model_file="Qwen3.5-14B-Q5_K_M.gguf",
        model_display="Qwen3.5 14B (Q5_K_M)",
        model_size_gb=10.5,
        context_length=32768,
        parallel_slots=2,
        kv_cache_k="q8_0", kv_cache_v="q8_0",
        notes="2 parallel slots lets ATLAS handle a coding session + "
              "background V3 verification without queueing.",
    ),
    TierProfile(
        tier="xlarge",
        label="X-Large (datacenter GPU)",
        description="32B+ model with 64K context, 2-4 parallel slots, "
                    "and full F16 KV cache for maximum quality.",
        min_vram_gb=32.0, max_vram_gb=None,
        example_gpus=["RTX 5090 32GB", "RTX A6000 48GB",
                      "A100 40/80GB", "H100 80GB"],
        model_file="Qwen3.5-32B-Q5_K_M.gguf",
        model_display="Qwen3.5 32B (Q5_K_M)",
        model_size_gb=23.0,
        context_length=65536,
        parallel_slots=2,
        kv_cache_k="f16", kv_cache_v="f16",
        notes="F16 KV cache + 64K context costs ~10 GB of cache. "
              "If you have 80 GB+ VRAM, bump parallel_slots to 4 and "
              "context to 131072 manually in .env.",
    ),
]


def classify(p: Probe) -> TierProfile:
    """Pick the tier whose VRAM range contains this probe's VRAM.

    GPU present but VRAM below the smallest tier (e.g., 4 GB) returns
    `small` with a notes-level warning rather than `cpu`, so users with a
    too-small GPU at least see what to upgrade to. Pure no-GPU returns
    `cpu`.
    """
    if not p.has_gpu:
        return TIERS[0]  # cpu
    for t in TIERS[1:]:
        if t.max_vram_gb is None:
            if p.vram_gb >= t.min_vram_gb:
                return t
        elif t.min_vram_gb <= p.vram_gb < t.max_vram_gb:
            return t
    # GPU present but below smallest tier breakpoint. Return small with
    # a runtime note about insufficient VRAM (caller can render the
    # tier.notes field, which already explains the trade-offs).
    small = TIERS[1]
    return TierProfile(**{**asdict(small),
        "notes": (f"Your GPU has only {p.vram_gb:.1f} GB VRAM, below the "
                  f"{small.min_vram_gb:.0f} GB minimum for the small tier. "
                  f"ATLAS will likely OOM. " + small.notes)})


def by_name(name: str) -> Optional[TierProfile]:
    """Look up a tier by its short name."""
    for t in TIERS:
        if t.tier == name:
            return t
    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_tier_card(t: TierProfile, p: Optional[Probe], color: bool) -> None:
    bar = (CYAN + "─" * 60 + RESET) if color and UNICODE_OK else ("-" * 60)
    _safe_print(bar)
    title = f"{BOLD}Tier: {t.tier}{RESET}" if color else f"Tier: {t.tier}"
    _safe_print(f"  {title}  {DASH}  {t.label}")
    _safe_print(bar)
    _safe_print(f"  {DIM if color else ''}{t.description}{RESET if color else ''}")
    _safe_print()
    if t.tier == "cpu":
        _safe_print("  VRAM range:    n/a (no CUDA GPU)")
    elif t.max_vram_gb is None:
        _safe_print(f"  VRAM range:    {t.min_vram_gb:.0f} GB and up")
    else:
        _safe_print(f"  VRAM range:    {t.min_vram_gb:.0f} GB {DASH} "
                    f"{t.max_vram_gb:.0f} GB")
    if t.example_gpus:
        _safe_print(f"  Example GPUs:  {', '.join(t.example_gpus)}")
    _safe_print()
    _safe_print(f"  {BOLD}Recommended ATLAS settings:{RESET}" if color
                else "  Recommended ATLAS settings:")
    _safe_print(f"    Model:           {t.model_display} ({t.model_size_gb:.1f} GB on disk)")
    _safe_print(f"    File:            {t.model_file}")
    _safe_print(f"    Context length:  {t.context_length:,} tokens")
    _safe_print(f"    Parallel slots:  {t.parallel_slots}")
    _safe_print(f"    KV cache K / V:  {t.kv_cache_k} / {t.kv_cache_v}")
    _safe_print()
    _safe_print(f"  Notes: {t.notes}")
    if p is not None and p.has_gpu:
        _safe_print()
        match = p.vram_gb >= t.min_vram_gb and (
            t.max_vram_gb is None or p.vram_gb < t.max_vram_gb)
        if match:
            mark = f"{GREEN}✓ matches{RESET}" if color and UNICODE_OK else "[match]"
        else:
            mark = f"{YELL}- (does not match){RESET}" if color and UNICODE_OK else "[no match]"
        _safe_print(f"  Your hardware: {p.vram_gb:.1f} GB VRAM {mark}")


def _emit_classify(p: Probe, t: TierProfile, args: argparse.Namespace,
                   color: bool) -> int:
    if args.json:
        out = {"probe": asdict(p), "tier": asdict(t),
               "env": t.env_vars()}
        print(json.dumps(out, indent=2, ensure_ascii=not UNICODE_OK))
        return 0
    if args.raw:
        for k, v in asdict(p).items():
            _safe_print(f"  {k:18s} {v}")
        return 0
    hdr = f"{BOLD}ATLAS tier{RESET}" if color else "ATLAS tier"
    _safe_print(f"{hdr} {DASH} probing host hardware")
    _safe_print()
    _safe_print(f"  Detected: {p.description}")
    _safe_print()
    _print_tier_card(t, p, color)
    _safe_print()
    if t.tier == "cpu":
        _safe_print(f"  {YELL if color else ''}Warning: ATLAS requires "
                    f"a CUDA GPU. See SETUP.md.{RESET if color else ''}")
        return 1
    _safe_print("  Apply these settings: edit .env to set the values "
                "shown above.")
    _safe_print(f"  Or run: {CYAN if color else ''}atlas wizard{RESET if color else ''}"
                f"  (when PC-054 lands).")
    return 0


def _emit_list(args: argparse.Namespace, color: bool) -> int:
    if args.json:
        print(json.dumps([asdict(t) for t in TIERS], indent=2,
                         ensure_ascii=not UNICODE_OK))
        return 0
    hdr = f"{BOLD}ATLAS tier definitions (PC-055){RESET}" if color else (
        "ATLAS tier definitions (PC-055)")
    _safe_print(hdr)
    _safe_print()
    for t in TIERS:
        _print_tier_card(t, None, color)
        _safe_print()
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas tier",
        description="Hardware tier classification (PC-055)")
    parser.add_argument("subcommand", nargs="?", default="classify",
        choices=["classify", "list"],
        help="`classify` (default) probes this host. `list` shows all tiers.")
    parser.add_argument("--json", action="store_true",
        help="emit JSON output (for PC-054 wizard, PC-056 model registry)")
    parser.add_argument("--raw", action="store_true",
        help="print probe output only, no classification")
    parser.add_argument("--no-color", action="store_true",
        help="disable ANSI color")
    parser.add_argument("--install-dir", default=None,
        help="probe disk free against this path (defaults to /)")
    args = parser.parse_args(argv)

    color = sys.stdout.isatty() and not args.no_color and not args.json

    if args.subcommand == "list":
        return _emit_list(args, color)

    p = probe(install_dir=args.install_dir)
    t = classify(p)
    return _emit_classify(p, t, args, color)


if __name__ == "__main__":
    sys.exit(main())
