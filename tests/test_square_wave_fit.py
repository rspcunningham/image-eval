from __future__ import annotations

import unittest

import numpy as np

from image_eval.mtf_profiles import BarROIProfile
from image_eval.square_wave_fit import (
    BAR_PROFILE_CYCLES,
    CYCLE_SEARCH_MAX,
    CYCLE_SEARCH_MIN,
    CYCLE_SEARCH_STEPS,
    SQUARE_WAVE_TERMS,
    fit_bar_roi_profiles,
    fit_square_wave_profile,
    odd_harmonics,
    square_wave_design_matrix,
)


class SquareWaveFitTests(unittest.TestCase):
    def test_defaults_to_three_square_wave_terms(self) -> None:
        self.assertEqual(SQUARE_WAVE_TERMS, 3)
        self.assertEqual(CYCLE_SEARCH_MIN, 2.0)
        self.assertEqual(CYCLE_SEARCH_MAX, 4.0)
        self.assertEqual(CYCLE_SEARCH_STEPS, 401)
        np.testing.assert_array_equal(odd_harmonics(), np.array([1, 3, 5]))

    def test_fits_first_three_odd_harmonics_with_linear_baseline(self) -> None:
        coefficients = np.array([0.4, 0.12, 0.8, -0.1, 0.2, 0.05, -0.08, 0.03])
        profile = square_wave_design_matrix(64) @ coefficients

        fit = fit_square_wave_profile(profile, cycles=BAR_PROFILE_CYCLES)

        self.assertEqual(fit.terms, 3)
        self.assertEqual(fit.cycles, BAR_PROFILE_CYCLES)
        np.testing.assert_array_equal(fit.harmonics, np.array([1, 3, 5]))
        self.assertAlmostEqual(fit.offset, coefficients[0])
        self.assertAlmostEqual(fit.baseline_slope, coefficients[1])
        np.testing.assert_allclose(fit.sine_coefficients, coefficients[[2, 4, 6]], atol=1e-12)
        np.testing.assert_allclose(fit.cosine_coefficients, coefficients[[3, 5, 7]], atol=1e-12)
        np.testing.assert_allclose(fit.fitted_profile, profile, atol=1e-12)
        self.assertLess(fit.residual_rms, 1e-12)
        self.assertAlmostEqual(fit.fundamental_amplitude, float(np.hypot(0.8, -0.1)))

    def test_searches_cycle_count_by_default(self) -> None:
        cycles = 3.21
        coefficients = np.array([0.4, -0.2, 0.8, -0.1, 0.2, 0.05, -0.08, 0.03])
        profile = square_wave_design_matrix(80, cycles=cycles) @ coefficients

        fit = fit_square_wave_profile(profile)

        self.assertAlmostEqual(fit.cycles, cycles)
        np.testing.assert_allclose(fit.fitted_profile, profile, atol=1e-10)
        self.assertLess(fit.residual_rms, 1e-10)
        self.assertAlmostEqual(fit.fundamental_amplitude, float(np.hypot(0.8, -0.1)))

    def test_rejects_profiles_that_are_too_short_for_the_fit(self) -> None:
        with self.assertRaises(ValueError):
            fit_square_wave_profile(np.ones(6))

    def test_custom_term_count_controls_sample_count(self) -> None:
        profile = square_wave_design_matrix(4, terms=1) @ np.array([0.5, 0.1, 0.25, -0.1])

        fit = fit_square_wave_profile(profile, terms=1, cycles=BAR_PROFILE_CYCLES)

        self.assertEqual(fit.terms, 1)
        np.testing.assert_array_equal(fit.harmonics, np.array([1]))
        np.testing.assert_allclose(fit.fitted_profile, profile, atol=1e-12)

    def test_rejects_non_finite_profile_values(self) -> None:
        profile = np.ones(16)
        profile[4] = np.nan

        with self.assertRaises(ValueError):
            fit_square_wave_profile(profile)

    def test_fits_bar_roi_profiles(self) -> None:
        coefficients = np.array([0.5, 0.0, 0.25, 0.0, 0.08, 0.0, 0.02, 0.0])
        profile = square_wave_design_matrix(32) @ coefficients
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
