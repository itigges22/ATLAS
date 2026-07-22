"""Conformance suite for the wavelet CWT port.

Translated from wavescope-mcp `src/wavelet.test.ts`. Asserts the same behaviors
so the Python port stays faithful to upstream (see
docs/reports/RPG_WAVELET_PLANNING_V3_2.md §3).
"""

import math
import sys
from pathlib import Path

import pytest

# v3-service ships the wavelet package; add it to the path the same way the
# existing v3-service tests add main.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

from wavelet.cwt import ricker_wavelet, compute_cwt, detect_peaks  # noqa: E402


class TestRickerWavelet:
    def test_value_one_at_zero(self):
        assert ricker_wavelet(0) == pytest.approx(1.0, abs=1e-5)

    def test_symmetric(self):
        for t in (0.5, 1.0, 2.0, 3.0):
            assert ricker_wavelet(t) == pytest.approx(ricker_wavelet(-t), abs=1e-8)

    def test_decays_beyond_5(self):
        assert abs(ricker_wavelet(5)) < 0.01
        assert abs(ricker_wavelet(8)) < 0.001

    def test_negative_lobes_and_zero_crossings(self):
        assert ricker_wavelet(2) < 0
        assert ricker_wavelet(1) == pytest.approx(0, abs=1e-5)
        assert ricker_wavelet(-1) == pytest.approx(0, abs=1e-5)


class TestComputeCWT:
    def test_dimensions(self):
        signal = [0.0] * 100
        scales = [1, 2, 4, 8]
        result = compute_cwt(signal, scales)
        assert result.scales == scales
        assert len(result.coefficients) == len(scales)
        assert len(result.coefficients[0]) == len(signal)

    def test_empty_signal(self):
        result = compute_cwt([], [1, 2, 4])
        assert result.scales == [1, 2, 4]
        assert len(result.coefficients) == 3
        assert all(len(c) == 0 for c in result.coefficients)

    def test_single_spike_position(self):
        signal = [0.0] * 100
        signal[50] = 1.0
        result = compute_cwt(signal, [1, 2, 4])
        coeffs = result.coefficients[0]
        peak_idx = max(range(len(coeffs)), key=lambda i: abs(coeffs[i]))
        assert 48 <= peak_idx <= 52

    def test_stronger_response_for_larger_signal(self):
        small = [0.0] * 100
        small[50] = 0.5
        large = [0.0] * 100
        large[50] = 1.5
        rs = compute_cwt(small, [2])
        rl = compute_cwt(large, [2])
        assert abs(rl.coefficients[0][50]) > abs(rs.coefficients[0][50])

    def test_smooth_signal_low_coeffs(self):
        margin = 64
        signal = [0.5] * 200
        result = compute_cwt(signal, [1, 4, 16])
        for coeffs in result.coefficients:
            for i in range(margin, len(coeffs) - margin):
                assert abs(coeffs[i]) < 0.05

    def test_large_scale_kernel_correctness(self):
        signal = [0.5] * 4096
        result = compute_cwt(signal, [128])
        coeffs = result.coefficients[0]
        margin = 1024
        for i in range(margin, len(coeffs) - margin):
            assert abs(coeffs[i]) < 0.01

    def test_raises_on_nan_scale(self):
        with pytest.raises(ValueError):
            compute_cwt([1, 2, 3], [math.nan])

    def test_raises_on_inf_scale(self):
        with pytest.raises(ValueError):
            compute_cwt([1, 2, 3], [math.inf])

    def test_dedup_repeated_scales(self):
        signal = [0.0] * 50
        signal[25] = 1
        result = compute_cwt(signal, [1, 1, 2, 2, 4])
        assert result.scales == [1, 2, 4]
        assert len(result.coefficients) == 3

    def test_reflect_boundary_near_zero_at_edges(self):
        signal = [0.5] * 200
        result = compute_cwt(signal, [4, 8, 16])
        for coeffs in result.coefficients:
            assert abs(coeffs[0]) < 0.05
            assert abs(coeffs[-1]) < 0.05

    def test_zero_boundary_back_compat(self):
        signal = [0.5] * 200
        result = compute_cwt(signal, [16], boundary="zero")
        assert len(result.coefficients[0]) == 200


class TestDetectPeaks:
    def test_finds_peaks_above_threshold(self):
        signal = [0.0] * 200
        signal[50] = 1.0
        signal[100] = 1.0
        signal[150] = 0.3
        result = compute_cwt(signal, [1, 2, 4, 8, 16, 32])
        peaks = detect_peaks(result, 0.5)
        assert len(peaks) > 0
        positions = [p.position for p in peaks]
        assert any(48 <= p <= 52 for p in positions)
        assert any(98 <= p <= 102 for p in positions)

    def test_empty_when_none_above_threshold(self):
        signal = [0.1] * 50
        result = compute_cwt(signal, [1, 2, 4])
        assert detect_peaks(result, 10.0) == []

    def test_sorted_descending(self):
        signal = [0.0] * 100
        signal[30] = 0.5
        signal[60] = 2.0
        signal[80] = 1.0
        result = compute_cwt(signal, [1, 2, 4])
        peaks = detect_peaks(result, 0.3)
        for i in range(1, len(peaks)):
            assert abs(peaks[i - 1].coefficient) >= abs(peaks[i].coefficient)

    def test_collapses_cross_scale_ridges(self):
        signal = [0.0] * 200
        signal[100] = 1.0
        result = compute_cwt(signal, [1, 2, 4, 8, 16, 32, 64, 128])
        peaks = detect_peaks(result, 0.1)
        near = [p for p in peaks if abs(p.position - 100) <= 2]
        assert len(near) == 1

    def test_disable_collapse_preserves_scales(self):
        signal = [0.0] * 200
        signal[100] = 1.0
        result = compute_cwt(signal, [1, 2, 4, 8, 16, 32, 64, 128])
        collapsed = detect_peaks(result, 0.1)
        every = detect_peaks(result, 0.1, 1000, -1)
        near_collapsed = [p for p in collapsed if abs(p.position - 100) <= 2]
        near_all = [p for p in every if abs(p.position - 100) <= 2]
        assert len(near_collapsed) == 1
        assert len(near_all) > 1
        assert len({p.scale for p in near_all}) > 1


class TestGoldenFixture:
    """Numeric parity with upstream wavescope-mcp src/wavelet.ts.

    Values captured by running the actual upstream TypeScript via
    `npx tsx` on the signal below (spikes of 1.0 at index 20, 0.5 at 40 over a
    64-sample zero signal). The Python port reproduces them to 8 decimals;
    any divergence flags a behavioral fork from upstream.
    """

    def test_coefficients_match_upstream(self):
        signal = [0.0] * 64
        signal[20] = 1.0
        signal[40] = 0.5
        result = compute_cwt(signal, [1, 2, 4, 8])
        c1 = [round(x, 8) for x in result.coefficients[0][18:23]]
        c2 = [round(x, 8) for x in result.coefficients[1][38:43]]
        assert c1 == [-0.40600585, 0.0, 1.0, 0.0, -0.40600585]
        assert c2 == [0.0, 0.23400733, 0.35355339, 0.23400733, 0.0]

    def test_peaks_match_upstream(self):
        signal = [0.0] * 64
        signal[20] = 1.0
        signal[40] = 0.5
        peaks = detect_peaks(compute_cwt(signal, [1, 2, 4, 8]), 0.1)
        got = [(p.position, round(p.coefficient, 8), p.scale) for p in peaks[:4]]
        assert got == [
            (20, 1.0, 1),
            (40, 0.5, 1),
            (16, -0.28708949, 2),
            (24, -0.28708949, 2),
        ]


def test_zero_signal_emits_no_peaks():
    # An all-zero signal (blank or comment-only file) must not produce
    # spurious zero-magnitude "peaks" at threshold 0.0.
    from wavelet.cwt import compute_cwt, detect_peaks
    coeffs = compute_cwt([0.0] * 40)
    assert detect_peaks(coeffs, 0.0) == []
