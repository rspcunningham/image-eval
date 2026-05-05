from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_eval.mtf_plot import (
    bar_roi_fit_plot_title,
    plot_mtf_curves,
    plot_square_wave_fit,
    save_mtf_curve_plot,
    save_square_wave_fit_plot,
)
from image_eval.mtf_results import MTFResult
from image_eval.mtf_profiles import BarROIProfile
from image_eval.square_wave_fit import (
    fit_bar_roi_profile,
    fit_square_wave_profile,
    square_wave_design_matrix,
)


class MTFPlotTests(unittest.TestCase):
    def test_saves_square_wave_fit_plot_as_png(self) -> None:
        profile = square_wave_design_matrix(32) @ np.array(
            [0.5, 0.0, 0.2, 0.0, 0.08, 0.0, 0.02, 0.0]
        )
        fit = fit_square_wave_profile(profile)

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "fit.png"

            save_square_wave_fit_plot(profile, fit, output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_plot_rejects_length_mismatch(self) -> None:
        profile = square_wave_design_matrix(32) @ np.array(
            [0.5, 0.0, 0.2, 0.0, 0.08, 0.0, 0.02, 0.0]
        )
        fit = fit_square_wave_profile(profile)

        with self.assertRaises(ValueError):
            plot_square_wave_fit(profile[:-1], fit)

    def test_saves_mtf_curve_plot_as_png(self) -> None:
        results = [
            MTFResult(frequency_lp_per_mm=1, x_mtf=0.9, y_mtf=None, average_mtf=0.9),
            MTFResult(frequency_lp_per_mm=2, x_mtf=0.7, y_mtf=0.8, average_mtf=0.75),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "mtf.png"

            save_mtf_curve_plot(results, output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_mtf_curve_plot_rejects_empty_results(self) -> None:
        with self.assertRaises(ValueError):
            plot_mtf_curves([])

    def test_bar_roi_fit_plot_title_includes_roi_identity_and_mtf(self) -> None:
        profile = square_wave_design_matrix(32) @ np.array(
            [0.5, 0.0, 0.2, 0.0, 0.08, 0.0, 0.02, 0.0]
        )
        fitted = fit_bar_roi_profile(
            BarROIProfile(
                group=4,
                element=2,
                orientation="Y",
                frequency_lp_per_mm=18.0,
                rect={"x0": 0, "y0": 0, "x1": 8, "y1": 32},
                profile_axis="y",
                profile=profile,
            )
        )

        title = bar_roi_fit_plot_title(fitted, mtf_value=0.314159)

        self.assertIn("G4 E2 Y", title)
        self.assertIn("18", title)
        self.assertIn("MTF 0.3142", title)


if __name__ == "__main__":
    unittest.main()
