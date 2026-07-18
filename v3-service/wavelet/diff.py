"""Structural peak-profile diff — the drift detector.

Faithful port of wavescope-mcp `src/diff.ts`. Diffs two sets of wavelet peaks
(e.g. a file before/after an edit) into added / removed / shifted /
magnitudeChanged / unchanged, with greedy proximity matching. Used by the
V3.2 RPG planner to detect when a file's structure has drifted away from the
plan (docs/reports/RPG_WAVELET_PLANNING_V3_2.md, Phase 3). The git-ref
orchestration (`diffFileAtRefs`) is intentionally omitted — drift compares two
in-memory contents, not git revisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .context import FileContext, ImportantPosition

_DEFAULT_WINDOW = 2
_COEFF_EPSILON = 1e-10


@dataclass
class PeakChange:
    kind: str  # "added" | "removed" | "shifted" | "magnitudeChanged" | "unchanged"
    before: Optional[ImportantPosition]
    after: Optional[ImportantPosition]


@dataclass
class PeakDiff:
    changes: List[PeakChange] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


@dataclass
class FileDiffResult:
    before_line_count: int
    after_line_count: int
    diff: PeakDiff


def diff_peaks(
    before: List[ImportantPosition],
    after: List[ImportantPosition],
    window: int = _DEFAULT_WINDOW,
) -> PeakDiff:
    """Diff two peak sets. Each after-peak greedily pairs with the closest
    unmatched before-peak within `window` lines (tiebreak by coefficient
    closeness). Unmatched before → removed; unmatched after → added. Same
    position + same coefficient → unchanged; same position, different
    coefficient → magnitudeChanged; different position within window → shifted.
    """
    changes: List[PeakChange] = []
    used_before = set()

    after_sorted = sorted(after, key=lambda p: p.position)

    for ap in after_sorted:
        best_idx = -1
        best_dist = float("inf")
        best_coef_diff = float("inf")
        for i, bp in enumerate(before):
            if i in used_before:
                continue
            dist = abs(bp.position - ap.position)
            if dist > window:
                continue
            coef_diff = abs(bp.coefficient - ap.coefficient)
            if dist < best_dist or (dist == best_dist and coef_diff < best_coef_diff):
                best_dist = dist
                best_coef_diff = coef_diff
                best_idx = i

        if best_idx != -1:
            used_before.add(best_idx)
            bp = before[best_idx]
            if bp.position == ap.position:
                if abs(bp.coefficient - ap.coefficient) < _COEFF_EPSILON:
                    changes.append(PeakChange("unchanged", bp, ap))
                else:
                    changes.append(PeakChange("magnitudeChanged", bp, ap))
            else:
                changes.append(PeakChange("shifted", bp, ap))
        else:
            changes.append(PeakChange("added", None, ap))

    for i, bp in enumerate(before):
        if i not in used_before:
            changes.append(PeakChange("removed", bp, None))

    def _pos(c: PeakChange) -> int:
        if c.kind == "removed":
            return c.before.position
        return c.after.position if c.after is not None else c.before.position

    changes.sort(key=_pos)

    summary = {"added": 0, "removed": 0, "shifted": 0, "magnitudeChanged": 0, "unchanged": 0}
    for c in changes:
        summary[c.kind] += 1

    return PeakDiff(changes=changes, summary=summary)


def diff_file_context(
    before_peaks: List[ImportantPosition],
    after_peaks: List[ImportantPosition],
    before_line_count: int,
    after_line_count: int,
    window: int = _DEFAULT_WINDOW,
) -> FileDiffResult:
    return FileDiffResult(
        before_line_count=before_line_count,
        after_line_count=after_line_count,
        diff=diff_peaks(before_peaks, after_peaks, window),
    )


def diff_contents(
    before_text: str,
    after_text: str,
    filename: str,
    min_coefficient: float = 0.3,
    limit: int = 100,
    window: int = _DEFAULT_WINDOW,
) -> FileDiffResult:
    """Convenience: build FileContexts for two file contents and diff their
    important positions. The drift entry point for RPG node verification."""
    from .project import MAX_FILE_BYTES

    # Same cap as the project walk: the pure-Python CWT makes an uncapped
    # input a multi-minute stall. Oversized content degrades to "no drift
    # information" rather than blocking the pipeline.
    if len(before_text) > MAX_FILE_BYTES:
        before_text = before_text[:MAX_FILE_BYTES]
    if len(after_text) > MAX_FILE_BYTES:
        after_text = after_text[:MAX_FILE_BYTES]
    before_ctx = FileContext(filename, before_text)
    after_ctx = FileContext(filename, after_text)
    return diff_file_context(
        before_ctx.get_important_positions(min_coefficient, limit),
        after_ctx.get_important_positions(min_coefficient, limit),
        before_ctx.line_count,
        after_ctx.line_count,
        window,
    )
