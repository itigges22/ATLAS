"""Multi-resolution file context (fine / medium / coarse bands).

Faithful port of wavescope-mcp `src/context.ts`. `FileContext` holds the
wavelet index for one file and assembles three resolution bands plus a ranked
list of structurally important positions. Band radii and scale ranges match
upstream so coarse-band "which sections matter" maps to the planner and the
fine band feeds per-node implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .language import detect_language, LanguageConfig
from .signal import compute_signal
from .cwt import compute_cwt, detect_peaks, Peak, WaveletCoefficients

# Label-parsing regexes (compiled once). _LABEL_SPLIT mirrors the code-delimiter
# tokenizer in signal.py; the others read names out of structural lines.
_LABEL_SPLIT = re.compile(r"[\s()\[\]{},;:'\"`=<>+*/&|^~%@#\\]+")
_WS_SPLIT = re.compile(r"\s+")
_PAREN_WS_SPLIT = re.compile(r"[\s()]+")
_BRACE_TAIL = re.compile(r"\{.*")
_CLASS_INHERIT = re.compile(r"extends|implements")


@dataclass
class ImportantPosition:
    position: int
    coefficient: float
    scale: int
    label: str
    filename: Optional[str] = None


@dataclass
class BandResult:
    range: Tuple[int, int]
    content: str


@dataclass
class WaveletContextResult:
    center: int
    clamped: bool
    bands: Dict[str, BandResult]
    wavelet_peaks: List[ImportantPosition]
    # Original `center` requested by caller, present only when clamped.
    clamped_from: Optional[int] = None


# Band scale ranges used by _build_medium_band / _build_coarse_band.
_BAND_SCALES = {
    "fine": (1, 2),
    "medium": (4, 16),
    "coarse": (32, 128),
}


class FileContext:
    """Wavelet index for a single file with multi-resolution query methods."""

    def __init__(self, filename: str, content: str):
        self.filename = filename
        lines = content.split("\n")

        # Preserve trailing newline: if content ends with \n, drop the empty
        # last element split produced.
        if content.endswith("\n") and lines and lines[-1] == "":
            lines.pop()
        # Truly empty content.
        if len(lines) == 1 and lines[0] == "":
            lines = []

        self.lines: List[str] = lines
        self.language: LanguageConfig = detect_language(filename)
        self.signal: List[float] = compute_signal(self.lines, self.language)
        self.coefficients: WaveletCoefficients = compute_cwt(self.signal)

        self._all_peaks: Optional[List[Peak]] = None
        self._ranked_peaks: Optional[List[Peak]] = None

    @property
    def line_count(self) -> int:
        return len(self.lines)

    # ─── Cached peak access ──────────────────────────────────

    def _get_all_peaks(self) -> List[Peak]:
        """All peaks (lazy). ridge_window < 0 disables cross-scale collapse so
        peaks at every scale are preserved — band assembly filters by scale and
        must see all of them."""
        if self._all_peaks is None:
            self._all_peaks = detect_peaks(self.coefficients, 0.0, 1000, -1)
        return self._all_peaks

    def _get_ranked_peaks(self) -> List[Peak]:
        """Ranked peak list with cross-scale ridge collapse applied, so a
        feature appearing at many scales yields one entry."""
        if self._ranked_peaks is None:
            self._ranked_peaks = detect_peaks(self.coefficients, 0.0, 1000)
        return self._ranked_peaks

    # ─── Public API ──────────────────────────────────────────

    def get_important_positions(
        self, min_coefficient: float = 0.3, limit: int = 20
    ) -> List[ImportantPosition]:
        """Important structural positions sorted by coefficient magnitude."""
        all_peaks = self._get_ranked_peaks()
        best: Dict[int, Peak] = {}
        for p in all_peaks:
            if abs(p.coefficient) < min_coefficient:
                continue
            existing = best.get(p.position)
            if existing is None or abs(p.coefficient) > abs(existing.coefficient):
                best[p.position] = p
        ranked = sorted(best.values(), key=lambda p: abs(p.coefficient), reverse=True)
        return [
            ImportantPosition(
                position=p.position,
                coefficient=p.coefficient,
                scale=p.scale,
                label=self._infer_label(p.position),
            )
            for p in ranked[:limit]
        ]

    def query_wavelet_context(self, center: int, radius: int) -> WaveletContextResult:
        """Multi-resolution context centered at a position.

        - fine: raw lines in a narrow window (~radius/5)
        - medium: peak-based summary in a medium window (~radius/2)
        - coarse: section-level overview across the full radius
        """
        if self.line_count == 0:
            empty = WaveletContextResult(
                center=0,
                clamped=center != 0,
                bands={
                    "fine": BandResult((0, 0), ""),
                    "medium": BandResult((0, 0), ""),
                    "coarse": BandResult((0, 0), ""),
                },
                wavelet_peaks=[],
            )
            if empty.clamped:
                empty.clamped_from = center
            return empty

        cl = max(0, min(center, self.line_count - 1))
        clamped = center != cl
        total = self.line_count

        fine_radius = max(10, radius // 5)
        fine_start = max(0, cl - fine_radius)
        fine_end = min(total - 1, cl + fine_radius)

        med_radius = radius // 2
        med_start = max(0, cl - med_radius)
        med_end = min(total - 1, cl + med_radius)

        coarse_start = max(0, cl - radius)
        coarse_end = min(total - 1, cl + radius)

        all_peaks = self._get_all_peaks()
        nearby = [p for p in all_peaks if coarse_start <= p.position <= coarse_end]

        result = WaveletContextResult(
            center=cl,
            clamped=clamped,
            bands={
                "fine": BandResult(
                    (fine_start, fine_end),
                    "\n".join(self.lines[fine_start:fine_end + 1]),
                ),
                "medium": BandResult(
                    (med_start, med_end),
                    self._build_medium_band(med_start, med_end, nearby),
                ),
                "coarse": BandResult(
                    (coarse_start, coarse_end),
                    self._build_coarse_band(coarse_start, coarse_end, nearby),
                ),
            },
            wavelet_peaks=[
                ImportantPosition(
                    position=p.position,
                    coefficient=p.coefficient,
                    scale=p.scale,
                    label=self._infer_label(p.position),
                )
                for p in self._dedup_peaks(nearby)[:10]
            ],
        )
        if clamped:
            result.clamped_from = center
        return result

    def auto_scale(self, start: int, end: int) -> int:
        """Pick a representative scale for a region of the given size, snapped
        to the closest available scale."""
        size = max(1, end - start + 1)
        if size <= 50:
            target = 2
        elif size <= 200:
            target = 8
        elif size <= 800:
            target = 32
        else:
            target = 128
        return self._find_closest_scale(target)

    def get_summary_at_scale(
        self, start: int, end: int, scale: Optional[int] = None
    ) -> str:
        """Compressed view of a region using wavelet peaks at a given scale."""
        if not self.lines:
            return ""
        max_idx = len(self.lines) - 1
        lo = min(start, end)
        hi = max(start, end)
        if lo > max_idx or hi < 0:
            return ""
        s = max(0, lo)
        e = min(max_idx, hi)

        all_peaks = self._get_all_peaks()
        resolved = self._find_closest_scale(scale) if scale is not None else self.auto_scale(s, e)
        in_range = [p for p in all_peaks if s <= p.position <= e and p.scale == resolved]
        if not in_range:
            return self._build_range_summary(s, e)
        return self._build_peak_summary(in_range, s, e)

    def get_wavelet_coefficients(self, start: int, end: int, scale: int):
        """Raw coefficient slice at the nearest available scale."""
        if len(self.coefficients.coefficients) == 0:
            return {"scale": scale, "requestedScale": scale, "coefficients": [], "clamped": False}
        resolved = self._find_closest_scale(scale)
        scale_idx = self.coefficients.scales.index(resolved)
        coeffs = self.coefficients.coefficients[scale_idx]
        if not coeffs:
            return {"scale": resolved, "requestedScale": scale, "coefficients": [], "clamped": False}
        max_idx = len(coeffs) - 1
        lo = min(start, end)
        hi = max(start, end)
        if lo > max_idx or hi < 0:
            return {
                "scale": resolved, "requestedScale": scale, "coefficients": [],
                "clamped": True, "clampedFrom": {"start": start, "end": end},
            }
        s = max(0, lo)
        e = min(max_idx, hi)
        clamped = s != start or e != end
        out = {
            "scale": resolved, "requestedScale": scale,
            "coefficients": coeffs[s:e + 1], "clamped": clamped,
        }
        if clamped:
            out["clampedFrom"] = {"start": start, "end": end}
        return out

    # ─── private helpers ──────────────────────────────────────

    def _dedup_peaks(self, peaks: List[Peak]) -> List[Peak]:
        best: Dict[int, Peak] = {}
        for p in peaks:
            existing = best.get(p.position)
            if existing is None or abs(p.coefficient) > abs(existing.coefficient):
                best[p.position] = p
        return list(best.values())

    def _find_closest_scale(self, scale: int) -> int:
        """Snap `scale` to the nearest scale present in the index. Ties resolve
        to the lower scale (stable reduce)."""
        scales = self.coefficients.scales
        if not scales:
            return scale
        best = scales[0]
        for curr in scales[1:]:
            if abs(curr - scale) < abs(best - scale):
                best = curr
        return best

    def _infer_label(self, pos: int) -> str:
        if pos < 0 or pos >= len(self.lines):
            return "unknown"
        line = self.lines[pos].strip()
        if not line:
            return f"line {pos}"

        # Token splitter mirrors signal._TOKEN_SPLIT; ws split is for keyword
        # reads where the original code preserved whitespace tokens.
        tokens = [t for t in _LABEL_SPLIT.split(line) if t]
        ws_tokens = _WS_SPLIT.split(line)

        def _after(name: str) -> str:
            idx = tokens.index(name)
            return tokens[idx + 1] if idx + 1 < len(tokens) else ""

        def _name_after_ws(i: int) -> str:
            return ws_tokens[i] if i < len(ws_tokens) else ""

        # f-string expressions below avoid backslashes / literal braces so the
        # source stays valid on Python 3.9 (project floor); regex work happens
        # in locals first.
        if self.language.name == "python":
            if ws_tokens[0] == "class":
                name = _name_after_ws(1).replace(":", "")
                return "class " + name
            if ws_tokens[0] == "def":
                name = _name_after_ws(1).split("(")[0]
                return "def " + name
            if ws_tokens[0] == "import":
                return "import " + " ".join(ws_tokens[1:])
            if ws_tokens[0] == "from":
                return "from " + " ".join(ws_tokens[1:])
            if line.startswith("@"):
                dec = _PAREN_WS_SPLIT.split(line)[0]
                return "decorator " + dec[1:]
            if line.startswith("if __name__"):
                return "main guard"
        else:
            if line.startswith("@"):
                dec = _PAREN_WS_SPLIT.split(line)[0]
                return "decorator " + dec[1:]
            if ws_tokens[0] == "import":
                return "import " + " ".join(ws_tokens[1:])
            if ws_tokens[0] == "export":
                return "export " + " ".join(ws_tokens[1:])[:40]
            if "class" in tokens:
                nxt = _BRACE_TAIL.sub("", _after("class"))
                nxt = _CLASS_INHERIT.sub("", nxt).strip()
                return "class " + nxt
            for kw in ("interface", "enum", "trait", "struct", "object"):
                if kw in tokens:
                    return kw + " " + _BRACE_TAIL.sub("", _after(kw)).strip()
            if "function" in tokens:
                return "function " + _after("function").split("(")[0]
            if "fn" in tokens and "defn" not in tokens:
                return "fn " + _after("fn").split("(")[0]
            if "fun" in tokens:
                return "fun " + _after("fun").split("(")[0]
            if "func" in tokens:
                return "func " + _after("func").split("(")[0]
            if "def" in tokens:
                return "def " + _after("def").split("(")[0]
            for kw in ("defn", "defmacro", "defprotocol", "defrecord", "deftype"):
                if kw in tokens:
                    return kw + " " + _after(kw)
            if "impl" in tokens:
                return "impl " + _after("impl").split("(")[0]
            if "protocol" in tokens:
                return "protocol " + _after("protocol")
            if "extension" in tokens:
                return "extension " + _after("extension")

        return line[:50]

    def _build_medium_band(self, start: int, end: int, peaks: List[Peak]) -> str:
        lo, hi = _BAND_SCALES["medium"]
        med = sorted(
            (p for p in peaks if start <= p.position <= end and lo <= p.scale <= hi),
            key=lambda p: p.position,
        )
        if not med:
            return self._build_range_summary(start, end)
        return self._build_peak_summary(med, start, end)

    def _build_coarse_band(self, start: int, end: int, peaks: List[Peak]) -> str:
        lo, hi = _BAND_SCALES["coarse"]
        coarse = sorted(
            (p for p in peaks if start <= p.position <= end and lo <= p.scale <= hi),
            key=lambda p: p.position,
        )
        if not coarse:
            all_in_range = sorted(
                (p for p in peaks if start <= p.position <= end),
                key=lambda p: p.position,
            )
            if not all_in_range:
                return self._build_range_summary(start, end)
            return self._build_section_summary(all_in_range, start, end)
        return self._build_section_summary(coarse, start, end)

    def _build_range_summary(self, start: int, end: int) -> str:
        if start > end:
            return ""
        preview_lines = min(5, end - start + 1)
        parts: List[str] = []
        step = -(-(end - start + 1) // preview_lines)  # ceil division
        i = start
        while i <= end:
            line = self.lines[i].strip()
            if line:
                parts.append(f"[{i}] {line[:80]}")
            i += step
        return "\n".join(parts)

    def _build_peak_summary(self, peaks: List[Peak], range_start: int, range_end: int) -> str:
        parts: List[str] = []
        prev_end = range_start - 1
        for peak in peaks:
            if peak.position > prev_end + 1:
                parts.append(f"[{prev_end + 1}-{peak.position - 1}] ...")
            parts.append(f"[{peak.position}] {self.lines[peak.position].strip()[:80]}")
            prev_end = peak.position
        if prev_end < range_end:
            parts.append(f"[{prev_end + 1}-{range_end}] ...")
        return "\n".join(parts)

    def _build_section_summary(self, peaks: List[Peak], range_start: int, range_end: int) -> str:
        parts: List[str] = []
        prev_pos = range_start
        current_section = ""
        if peaks:
            current_section = self._infer_label(peaks[0].position)
        for peak in peaks:
            if current_section and prev_pos < peak.position:
                parts.append(f"[{prev_pos}-{peak.position - 1}] {current_section}")
            current_section = self._infer_label(peak.position)
            prev_pos = peak.position
        if current_section and prev_pos <= range_end:
            parts.append(f"[{prev_pos}-{range_end}] {current_section}")
        if not parts:
            parts.append(f"[{range_start}-{range_end}] (code region)")
        return "\n".join(parts)
