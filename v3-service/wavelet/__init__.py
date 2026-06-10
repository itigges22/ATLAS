"""Wavelet-based multi-resolution structural decomposition.

Faithful pure-Python port of the wavescope-mcp wavelet engine
(https://github.com/yogthos/wavescope-mcp) for in-process use by the V3.2
RPG-style architecture-first planner (see docs/reports/RPG_WAVELET_PLANNING_V3_2.md).

Provenance: ported behavior-for-behavior from wavescope-mcp `src/`:
  language.py  <- src/language.ts
  signal.py    <- src/signal.ts
  cwt.py       <- src/wavelet.ts
  context.py   <- src/context.ts
  project.py   <- src/project.ts

The port is deliberately dependency-free (stdlib only), mirroring the upstream
TypeScript loops; numpy vectorization is a drop-in optimization if profiling of
the planning critical path ever demands it. Behavior — scales, band radii,
normalization, reflect boundary, ridge-collapse semantics — matches upstream so
the calibrated thresholds (e.g. min_coefficient default 0.3) carry over.
"""

from .language import LanguageConfig, detect_language, configs
from .signal import compute_signal
from .cwt import (
    ricker_wavelet,
    compute_cwt,
    detect_peaks,
    Peak,
    WaveletCoefficients,
    DEFAULT_SCALES,
)
from .context import (
    FileContext,
    ImportantPosition,
    BandResult,
    WaveletContextResult,
)
from .project import decompose_project, decompose_file_map, ProjectIndex
from .diff import diff_peaks, diff_contents, FileDiffResult, PeakChange, PeakDiff
from .flags import rpg_planning_enabled, ENV_VAR as RPG_PLANNING_ENV_VAR

__all__ = [
    "LanguageConfig",
    "detect_language",
    "configs",
    "compute_signal",
    "ricker_wavelet",
    "compute_cwt",
    "detect_peaks",
    "Peak",
    "WaveletCoefficients",
    "DEFAULT_SCALES",
    "FileContext",
    "ImportantPosition",
    "BandResult",
    "WaveletContextResult",
    "decompose_project",
    "decompose_file_map",
    "ProjectIndex",
    "diff_peaks",
    "diff_contents",
    "FileDiffResult",
    "PeakChange",
    "PeakDiff",
    "rpg_planning_enabled",
    "RPG_PLANNING_ENV_VAR",
]
