from __future__ import annotations

import unittest

import numpy as np

from image_eval.mtf_profiles import BarROIProfile, OrientedSamples
from image_eval.square_wave_fit import (
    calibrate_pixels_per_mm,
    fit_bar_roi_profiles,
    fit_square_wave,
    fit_square_wave_at_cycles,
    harmonics_below_nyquist,
)


def _samples(width: int, cycles: float, coefficients: np.ndarray, rows: int = 1) -> OrientedSamples:
    """Offset, slope, then sine/cosine pairs of odd harmonics at evenly spaced positions."""
    x = np.repeat((np.arange(width) + 0.5) / width, rows)
    values = coefficients[0] + coefficients[1] * (x - 0.5)
    for index, harmonic in enumerate(range(1, len(coefficients) - 2, 2)):
        radians = 2 * np.pi * cycles * harmonic * x
        values = values + coefficients[2 + 2 * index] * np.sin(radians)
        values = values + coefficients[3 + 2 * index] * np.cos(radians)
    return OrientedSamples(positions=x, values=values, width_px=float(width))


def _roi(samples: OrientedSamples, *, frequency: float) -> BarROIProfile:
    return BarROIProfile(
        group=0,
        element=1,
        orientation="X",
        frequency_lp_per_mm=frequency,
        rect={"x0": 0, "y0": 0, "x1": round(samples.width_px), "y1": 8},
        samples=samples,
    )


class SquareWaveFitTests(unittest.TestCase):
    def test_fits_first_three_odd_harmonics_with_linear_baseline(self) -> None:
        coefficients = np.array([0.4, 0.12, 0.8, -0.1, 0.2, 0.05, -0.08, 0.03])
        samples = _samples(64, 3.21, coefficients, rows=3)

        fit = fit_square_wave_at_cycles(samples, cycles=3.21)

        np.testing.assert_array_equal(fit.harmonics, [1, 3, 5])
        self.assertAlmostEqual(fit.offset, coefficients[0])
        self.assertAlmostEqual(fit.baseline_slope, coefficients[1])
        np.testing.assert_allclose(fit.sine_coefficients, coefficients[[2, 4, 6]], atol=1e-10)
        np.testing.assert_allclose(fit.cosine_coefficients, coefficients[[3, 5, 7]], atol=1e-10)
        np.testing.assert_allclose(fit.fitted_values, samples.values, atol=1e-10)
        self.assertAlmostEqual(fit.fundamental_amplitude, float(np.hypot(0.8, -0.1)))

    def test_searches_cycle_count(self) -> None:
        coefficients = np.array([0.4, -0.2, 0.8, -0.1, 0.2, 0.05, -0.08, 0.03])

        fit = fit_square_wave(_samples(80, 3.215, coefficients))

        self.assertAlmostEqual(fit.cycles, 3.215)
        self.assertLess(fit.residual_rms, 1e-10)

    def test_fits_scattered_positions(self) -> None:
        # Pixels of a rotated element land at irregular positions across the bars.
        coefficients = np.array([0.5, 0.0, 0.3, 0.1, 0.05, 0.0, 0.0, 0.0])
        grid = _samples(4000, 2.5, coefficients)
        pick = np.random.default_rng(0).choice(4000, 300, replace=False)
        pick.sort()
        samples = OrientedSamples(grid.positions[pick], grid.values[pick], width_px=40.0)

        fit = fit_square_wave_at_cycles(samples, cycles=2.5)

        self.assertAlmostEqual(fit.fundamental_amplitude, float(np.hypot(0.3, 0.1)))

    def test_rejects_too_few_samples(self) -> None:
        with self.assertRaises(ValueError):
            fit_square_wave(OrientedSamples(np.linspace(0, 1, 6), np.ones(6), width_px=6.0))

    def test_taper_down_weights_element_ends(self) -> None:
        samples = _samples(64, 3.0, np.array([0.5, 0, 0.3, 0, 0, 0, 0, 0]))
        values = samples.values.copy()
        values[:3] += 1.0  # bright halo past the first bar
        values[-3:] += 1.0

        fit = fit_square_wave_at_cycles(
            OrientedSamples(samples.positions, values, samples.width_px), cycles=3.0
        )

        self.assertAlmostEqual(fit.fundamental_amplitude, 0.3, delta=0.01)


class FixedPeriodFitTests(unittest.TestCase):
    def test_refits_every_roi_at_the_calibrated_period(self) -> None:
        # Two clean elements calibrate 64 px / (16 lp/mm * 3 cycles); a noisy third element
        # must be fit at its geometric period, not at whatever matches the noise.
        clean = _samples(64, 3.0, np.array([0.5, 0, 0.3, 0, 0, 0, 0, 0]))
        rng = np.random.default_rng(0)
        noisy = OrientedSamples(clean.positions, 0.5 + 0.05 * rng.standard_normal(64), 64.0)
        profiles = [
            _roi(clean, frequency=16.0),
            _roi(clean, frequency=16.0),
            _roi(noisy, frequency=20.0),
        ]

        fits = fit_bar_roi_profiles(profiles)

        self.assertAlmostEqual(fits[0].fit.cycles, 3.0, places=6)
        self.assertAlmostEqual(fits[2].fit.cycles, 20.0 * 64 / (16.0 * 64 / 3.0), places=6)

    def test_calibration_ignores_poorly_resolved_fits(self) -> None:
        clean = _samples(64, 3.0, np.array([0.5, 0, 0.3, 0, 0, 0, 0, 0]))
        weak = _samples(64, 3.6, np.array([0.5, 0, 0.02, 0, 0, 0, 0, 0]))
        profiles = [_roi(clean, frequency=16.0), _roi(weak, frequency=16.0)]
        free = [fit_square_wave(p.samples) for p in profiles]

        self.assertAlmostEqual(calibrate_pixels_per_mm(profiles, free), 16.0 * 64 / 3.0, places=4)


class HarmonicSelectionTests(unittest.TestCase):
    def test_keeps_harmonics_below_the_sampling_nyquist(self) -> None:
        np.testing.assert_array_equal(harmonics_below_nyquist(3.0, 64), [1, 3, 5])  # 15 < 32
        np.testing.assert_array_equal(harmonics_below_nyquist(3.0, 30), [1, 3])  # 15 = 15
        np.testing.assert_array_equal(harmonics_below_nyquist(3.0, 16), [1])  # 9 > 8

    def test_keeps_the_fundamental_even_above_nyquist(self) -> None:
        np.testing.assert_array_equal(harmonics_below_nyquist(10.0, 16), [1])

    def test_rois_fit_only_harmonics_below_nyquist(self) -> None:
        clean = _samples(24, 3.0, np.array([0.5, 0, 0.3, 0, 0, 0, 0, 0]))
        fits = fit_bar_roi_profiles([_roi(clean, frequency=16.0)])

        # 3 cycles over 24 px: 3rd harmonic at 9 < 12 cycles stays, 5th at 15 aliases.
        np.testing.assert_array_equal(fits[0].fit.harmonics, [1, 3])


if __name__ == "__main__":
    unittest.main()
