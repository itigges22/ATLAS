"""Ricker (Mexican-hat) continuous wavelet transform + peak detection.

Faithful port of wavescope-mcp `src/wavelet.ts`. The transform is a custom
direct convolution with `1/sqrt(a)` normalization, `+/-5a` kernel truncation,
and symmetric-reflect boundary handling. Peak detection collapses cross-scale
ridges so a single structural feature yields one peak. These conventions are
NOT interchangeable with PyWavelets/SciPy `mexh` — every downstream threshold
is calibrated to this exact coefficient scale (see docs/reports/RPG_WAVELET_PLANNING_V3_2.md §6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Peak:
    position: int
    coefficient: float
    scale: int
    label: Optional[str] = None


@dataclass
class WaveletCoefficients:
    scales: List[int]
    coefficients: List[List[float]]  # [scaleIndex][position]


DEFAULT_SCALES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128]


def ricker_wavelet(t: float) -> float:
    """Ricker (Mexican hat) wavelet: psi(t) = (1 - t^2) * exp(-t^2 / 2)."""
    t2 = t * t
    return (1 - t2) * math.exp(-t2 / 2)


def _make_kernel(a: float, num_points: int) -> List[float]:
    """Wavelet kernel values for scale `a`, centered at 0.

    Truncated to +/-5a (covers ~99.99% of the Ricker energy) but bounded by half
    the signal length to stay finite on short inputs. Includes 1/sqrt(a)
    normalization to keep coefficient magnitudes comparable across scales.
    """
    if not math.isfinite(a) or a <= 0:
        raise ValueError(f"Invalid scale: {a}")
    half_width = math.ceil(5 * a)
    half = min(half_width, math.ceil(num_points / 2))
    inv_sqrt_a = 1 / math.sqrt(a)
    kernel: List[float] = []
    for t in range(-half, half + 1):
        kernel.append(inv_sqrt_a * ricker_wavelet(t / a))
    return kernel


def _reflect_index(idx: int, n: int) -> int:
    """Mirror out-of-range indices back into [0, N-1] to suppress boundary
    artifacts where the wavelet's negative lobes would otherwise be clipped."""
    if n == 1:
        return 0
    period = 2 * (n - 1)
    i = idx % period  # Python modulo already yields a non-negative result here.
    return period - i if i >= n else i


def compute_cwt(
    signal: List[float],
    scales: Optional[List[int]] = None,
    boundary: str = "reflect",
) -> WaveletCoefficients:
    """Compute the Ricker CWT over the signal.

    For each scale a, psi_a(t) = (1/sqrt(a)) * psi(t/a) is convolved with the
    signal: W(a, b) = sum_t psi_a(t - b) * signal[t]. Boundary defaults to
    symmetric reflection; pass boundary="zero" for zero-padding.
    """
    if scales is None:
        scales = DEFAULT_SCALES
    n = len(signal)

    used_scales: List[int] = []
    for a in scales:
        if a not in used_scales:
            used_scales.append(a)

    if n == 0:
        return WaveletCoefficients(scales=used_scales, coefficients=[[] for _ in used_scales])

    coefficients: List[List[float]] = []
    for a in used_scales:
        kernel = _make_kernel(a, n)
        half_kernel = len(kernel) // 2
        coeffs = [0.0] * n
        for pos in range(n):
            total = 0.0
            for k, kv in enumerate(kernel):
                signal_idx = pos + k - half_kernel
                if 0 <= signal_idx < n:
                    total += kv * signal[signal_idx]
                elif boundary == "reflect":
                    total += kv * signal[_reflect_index(signal_idx, n)]
            coeffs[pos] = total
        coefficients.append(coeffs)

    return WaveletCoefficients(scales=used_scales, coefficients=coefficients)


def detect_peaks(
    cwt: WaveletCoefficients,
    threshold: float,
    max_peaks: int = 250,
    ridge_window: int = 2,
) -> List[Peak]:
    """Detect local maxima in coefficient magnitudes across all scales.

    Returns peaks sorted by |coefficient| descending. Plateau handling
    (>= left, > right) selects the rightmost element of a flat plateau.

    Cross-scale ridge collapse: after magnitude sorting, peaks within
    `ridge_window` of an already-kept stronger peak are dropped, so a single
    spike yields one peak (the dominant scale). A negative `ridge_window`
    disables collapse — every per-scale peak is preserved.
    """
    if len(cwt.coefficients) == 0:
        return []

    peaks: List[Peak] = []
    for si, scale in enumerate(cwt.scales):
        coeffs = cwt.coefficients[si]
        n = len(coeffs)
        for pos in range(n):
            mag = abs(coeffs[pos])
            # Zero magnitude is never a peak, even at threshold 0.0 — an
            # all-zero signal (blank or comment-only file) must not emit a
            # spurious flat "peak" per scale.
            if mag <= 0.0 or mag < threshold:
                continue
            left_ok = pos == 0 or mag >= abs(coeffs[pos - 1])
            right_ok = pos == n - 1 or mag > abs(coeffs[pos + 1])
            if left_ok and right_ok:
                peaks.append(Peak(position=pos, coefficient=coeffs[pos], scale=scale))

    peaks.sort(key=lambda p: abs(p.coefficient), reverse=True)

    kept: List[Peak] = []
    for peak in peaks:
        overlap = any(abs(k.position - peak.position) <= ridge_window for k in kept)
        if not overlap:
            kept.append(peak)
        if len(kept) >= max_peaks:
            break

    return kept
