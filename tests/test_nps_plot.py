from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_eval.nps_plot import plot_nps_curves, save_nps_curve_plot, save_nps_spectrum_plot
from image_eval.nps_results import NPSResult, calculate_nps_report


class NPSPlotTests(unittest.TestCase):
    def test_saves_nps_curve_plot_as_png(self) -> None:
        results = [
            NPSResult(frequency=0.1, black_nps=0.02, white_nps=0.03, average_nps=0.025),
            NPSResult(frequency=0.2, black_nps=0.01, white_nps=None, average_nps=0.01),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nps.png"

            save_nps_curve_plot(results, output_path, frequency_unit="cycles per pixel")

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_nps_curve_plot_rejects_empty_results(self) -> None:
        with self.assertRaises(ValueError):
            plot_nps_curves([], frequency_unit="cycles per pixel")

    def test_nps_curve_plot_uses_log_y_scale(self) -> None:
        figure = plot_nps_curves(
            [NPSResult(frequency=0.1, black_nps=0.02, white_nps=0.03, average_nps=0.025)],
            frequency_unit="cycles per pixel",
        )
        try:
            self.assertEqual(figure.axes[0].get_yscale(), "log")
        finally:
            figure.clear()

    def test_saves_nps_spectrum_plot_as_png(self) -> None:
        image = np.zeros((4, 8), dtype=np.float64)
        image[:, 4:8] = 1.0
        report = calculate_nps_report(image, {
            "normalization_rois": {
                "black": {"x0": 0, "y0": 0, "x1": 4, "y1": 4},
                "white": {"x0": 4, "y0": 0, "x1": 8, "y1": 4},
            }
        })

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "black_2d.png"

            save_nps_spectrum_plot(
                report.spectra[0],
                output_path,
                frequency_unit="cycles per pixel",
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
