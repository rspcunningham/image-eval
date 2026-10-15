from __future__ import annotations

import unittest

import numpy as np

from image_eval.mtf_profiles import BarROIProfile
from image_eval.square_wave_fit import (
    CYCLE_SEARCH_MAX,
    CYCLE_SEARCH_MIN,
    CYCLE_SEARCH_STEPS,
    SQUARE_WAVE_TERMS,
    _square_wave_design_matrix,
    calibrate_pixels_per_mm,
    fit_bar_roi_profiles,
    fit_square_wave_profile,
    fit_square_wave_profile_at_cycles,
)


class SquareWaveFitTests(unittest.TestCase):
    def test_defaults_to_three_square_wave_terms(self) -> None:
        self.assertEqual(SQUARE_WAVE_TERMS, 3)
        self.assertEqual(CYCLE_SEARCH_MIN, 2.0)
        self.assertEqual(CYCLE_SEARCH_MAX, 4.0)
        self.assertEqual(CYCLE_SEARCH_STEPS, 401)

    def test_fits_first_three_odd_harmonics_with_linear_baseline(self) -> None:
        cycles = 3.21
        coefficients = np.array([0.4, 0.12, 0.8, -0.1, 0.2, 0.05, -0.08, 0.03])
        profile = _square_wave_design_matrix(64, cycles=cycles) @ coefficients

        fit = fit_square_wave_profile(profile)

        self.assertEqual(fit.terms, 3)
        self.assertAlmostEqual(fit.cycles, cycles)
        np.testing.assert_array_equal(fit.harmonics, np.array([1, 3, 5]))
        self.assertAlmostEqual(fit.offset, coefficients[0])
        self.assertAlmostEqual(fit.baseline_slope, coefficients[1])
        np.testing.assert_allclose(fit.sine_coefficients, coefficients[[2, 4, 6]], atol=1e-10)
        np.testing.assert_allclose(fit.cosine_coefficients, coefficients[[3, 5, 7]], atol=1e-10)
        np.testing.assert_allclose(fit.fitted_profile, profile, atol=1e-10)
        self.assertLess(fit.residual_rms, 1e-10)
        self.assertAlmostEqual(fit.fundamental_amplitude, float(np.hypot(0.8, -0.1)))

    def test_searches_cycle_count(self) -> None:
        cycles = 3.21
        coefficients = np.array([0.4, -0.2, 0.8, -0.1, 0.2, 0.05, -0.08, 0.03])
        profile = _square_wave_design_matrix(80, cycles=cycles) @ coefficients

        fit = fit_square_wave_profile(profile)

        self.assertAlmostEqual(fit.cycles, cycles)
        np.testing.assert_allclose(fit.fitted_profile, profile, atol=1e-10)
        self.assertLess(fit.residual_rms, 1e-10)
        self.assertAlmostEqual(fit.fundamental_amplitude, float(np.hypot(0.8, -0.1)))

    def test_rejects_profiles_that_are_too_short_for_the_fit(self) -> None:
        with self.assertRaises(ValueError):
            fit_square_wave_profile(np.ones(6))

    def test_rejects_non_finite_profile_values(self) -> None:
        profile = np.ones(16)
        profile[4] = np.nan

        with self.assertRaises(ValueError):
            fit_square_wave_profile(profile)

    def test_fits_bar_roi_profiles(self) -> None:
        coefficients = np.array([0.5, 0.0, 0.25, 0.0, 0.08, 0.0, 0.02, 0.0])
        profile = _square_wave_design_matrix(32, cycles=3.0) @ coefficients
        roi_profile = BarROIProfile(
            group=4,
            element=1,
            orientation="X",
            frequency_lp_per_mm=16,
            rect={"x0": 0, "y0": 0, "x1": 32, "y1": 8},
            profile_axis="x",
            profile=profile,
        )

        fits = fit_bar_roi_profiles([roi_profile])

        self.assertEqual(len(fits), 1)
        self.assertIs(fits[0].roi_profile, roi_profile)
        np.testing.assert_allclose(fits[0].fit.fitted_profile, profile, atol=1e-12)


if __name__ == "__main__":
    unittest.main()


class FixedPeriodFitTests(unittest.TestCase):
    def test_refits_every_roi_at_the_calibrated_period(self) -> None:
        # Two clean elements calibrate 64 px / (16 lp/mm * 3 cycles) ... then a noisy third
        # element must be fit at its geometric period, not at whatever matches the noise.
        clean = _square_wave_design_matrix(64, cycles=3.0) @ np.array([0.5, 0, 0.3, 0, 0, 0, 0, 0])
        rng = np.random.default_rng(0)
        noisy = 0.5 + 0.05 * rng.standard_normal(64)
        profiles = [
            _roi(clean, frequency=16.0),
            _roi(clean, frequency=16.0),
            _roi(noisy, frequency=20.0),
        ]

        fits = fit_bar_roi_profiles(profiles)

        pixels_per_mm = 16.0 * 64 / 3.0
        self.assertAlmostEqual(fits[0].fit.cycles, 3.0, places=6)
        self.assertAlmostEqual(fits[2].fit.cycles, 20.0 * 64 / pixels_per_mm, places=6)

    def test_calibration_ignores_poorly_resolved_and_boundary_fits(self) -> None:
        clean = _square_wave_design_matrix(64, cycles=3.0) @ np.array([0.5, 0, 0.3, 0, 0, 0, 0, 0])
        weak = _square_wave_design_matrix(64, cycles=3.6) @ np.array([0.5, 0, 0.02, 0, 0, 0, 0, 0])
        profiles = [_roi(clean, frequency=16.0), _roi(weak, frequency=16.0)]
        free = [fit_square_wave_profile(p.profile) for p in profiles]

        self.assertAlmostEqual(calibrate_pixels_per_mm(profiles, free), 16.0 * 64 / 3.0, places=4)

    def test_taper_keeps_exact_fits_exact(self) -> None:
        coefficients = np.array([0.4, 0.1, 0.6, 0.05, 0.1, 0.0, 0.02, 0.0])
        profile = _square_wave_design_matrix(48, cycles=2.7) @ coefficients

        fit = fit_square_wave_profile_at_cycles(profile, cycles=2.7)

        np.testing.assert_allclose(fit.fitted_profile, profile, atol=1e-10)

    def test_taper_down_weights_profile_ends(self) -> None:
        coefficients = np.array([0.5, 0, 0.3, 0, 0, 0, 0, 0])
        profile = _square_wave_design_matrix(64, cycles=3.0) @ coefficients
        corrupted = profile.copy()
        corrupted[:3] += 1.0  # bright halo past the first bar
        corrupted[-3:] += 1.0

        fit = fit_square_wave_profile_at_cycles(corrupted, cycles=3.0)

        self.assertAlmostEqual(fit.fundamental_amplitude, 0.3, delta=0.01)


def _roi(profile: np.ndarray, *, frequency: float) -> BarROIProfile:
    return BarROIProfile(
        group=0,
        element=1,
        orientation="X",
        frequency_lp_per_mm=frequency,
        rect={"x0": 0, "y0": 0, "x1": len(profile), "y1": 8},
        profile_axis="x",
        profile=profile,
    )
